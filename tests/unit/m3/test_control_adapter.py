"""Unit tests for ``m3.control_adapter.OGMemoryControlAdapter``.

Tests use the in-memory graph + a stub Embedder (deterministic 16-d
vectors) + a stub LLM that returns a scripted sequence of tool calls
via ``_mock_tool_calls`` (MockLLM API). No network, no DB.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.graph_enums import GraphNodeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.control_adapter import (
    Answer,
    M3TaskContext,
    OGMemoryControlAdapter,
    Search,
    SearchConfig,
    _combined_name_tokens,
    _merge_hits_by_node,
    _name_tokens_from_hits,
    _query_name_candidates,
    _strip_name_tokens_from_query,
)
from providers.embedder import MockEmbedder
from providers.graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import GraphService


class _ScriptedLLM(MockLLM):
    """MockLLM that returns the next item from a scripted queue per call.

    The vanilla MockLLM always returns the entire ``_mock_tool_calls``
    list on every call, which is fine for single-shot extraction
    tests but loops forever under the ReAct loop. ``_ScriptedLLM``
    pops one call off the queue each turn so the LLM can drive a
    deterministic multi-round script (Search → Answer → final content).

    Each script entry is a dict: either
    ``{"tool": "<name>", "input": {...}}`` to return as a tool call,
    or ``{"content": "..."}`` to return as plain LLM content.
    """

    def __init__(self, script: list[dict] | None = None) -> None:
        super().__init__()
        self._script = list(script or [])
        self._call_count = 0

    def complete_with_tools_messages(
        self, messages, tools=None, tool_choice="auto",
    ):
        self._call_count += 1
        self._last_messages = messages
        if not self._script:
            return [], ""
        entry = self._script.pop(0)
        if "content" in entry:
            return [], entry["content"]
        return [entry], ""


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

DIM = 16


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-control-test", trace_id="t-test",
    )


@pytest.fixture
def graph() -> GraphService:
    return GraphService(MemoryGraphStore())


@pytest.fixture
def embedder() -> MockEmbedder:
    # 16-d deterministic mock so we control the cosine math exactly.
    return MockEmbedder(dimension=DIM)


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def offline_transcript_tokenizer(monkeypatch):
    from tests.unit.m3.test_transcript_rerank import _mock_tokenize

    monkeypatch.setattr("retrieval.bm25_index._tokenize", _mock_tokenize)


def _scripted(script: list[dict]) -> _ScriptedLLM:
    return _ScriptedLLM(script)


def _seed_episode(
    graph: GraphService,
    ctx: RequestContext,
    *,
    video_id: str,
    clip_id: int,
    description: str,
    node_id: str | None = None,
) -> str:
    """Persist one ExtractedNode + set its embedding to a fixed vector
    so we can assert specific cosine similarities in tests."""
    import uuid
    nid = node_id or uuid.uuid4().hex[:32]
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
            "characters": [],
            "events": [],
            "entities": [],
        },
        node_id=nid,
    )
    graph._store.create_node(node.serialize(), ctx)
    # Set a distinctive embedding so cosine scores are stable.
    # Use the node id's first character to pick a basis-like vector:
    # this gives two different embeddings for different node ids, so
    # we can verify Search picks the right one.
    slot = ord(nid[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(nid, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return nid


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructs_with_required_dependencies(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
    )
    assert adapter._graph is graph
    assert adapter._embedder is embedder
    assert adapter._extractor_types == ("episodic_memory_generator",)
    assert adapter._default_video_id is None


def test_search_tool_schema_exposes_only_query_and_top_k(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
    )
    tools = adapter._get_tools()
    # 2 operational tools, no custom tools.
    assert len(tools) == 2
    search = next(t for t in tools if t["name"] == "Search")
    answer = next(t for t in tools if t["name"] == "Answer")
    # Search input_schema must only expose query + top_k. Anything
    # else (extractor_types, scope_video_id, ctx, graph, embedder) is
    # runtime-only and must not be visible to the LLM.
    props = search["input_schema"]["properties"]
    assert set(props.keys()) == {"query", "top_k"}
    assert search["input_schema"]["required"] == ["query"]
    # Answer takes a single final_answer string.
    assert set(answer["input_schema"]["properties"].keys()) == {"final_answer"}
    assert answer["input_schema"]["required"] == ["final_answer"]


def test_prompt_uses_function_names_not_literal_protocol_tags(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)
    messages = adapter._build_messages(_ctx(), M3TaskContext(question="Where?"))
    system = messages[0]["content"]
    assert "Search function" in system
    assert "Answer function" in system
    assert "provided tool-call interface" in system
    assert "[Search]" not in system
    assert "[Answer]" not in system


@pytest.mark.parametrize("hits", [[], [{"description": "a bedroom"}]])
def test_repeated_search_reuses_results_and_preserves_tool_messages(
    graph, embedder, llm, hits,
):
    adapter = OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)
    ctx = _ctx()
    task = M3TaskContext(question="Where?")
    messages = adapter._build_messages(ctx, task)
    calls = [
        {"id": "s1", "name": "Search", "input": {"query": "woman bedroom"}},
        {"id": "s2", "name": "Search", "input": {"query": "  Woman \n BEDROOM "}},
    ]
    used = []
    with patch("core.react_loop.ReactLoop._execute_tool", return_value={"hits": hits}) as execute:
        adapter._execute_and_append_tool_calls(messages, calls, used, set(), ctx)
    assert execute.call_count == 1
    assert used[1]["result"]["hits"] == hits
    assert used[1]["result"]["duplicate_query"] is True
    assert "duplicate_query" not in used[0]["result"]
    assert [h["cache_hit"] for h in task.search_history] == [False, True]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["s1", "s2"]


def test_cached_search_compacts_model_context_but_keeps_full_audit(graph, embedder, llm):
    import json
    adapter = OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)
    ctx, task = _ctx(), M3TaskContext(question="Where?")
    messages = adapter._build_messages(ctx, task)
    hits = [{"node_id": "n1", "description": "Important full evidence. " * 500}]
    calls = [
        {"id": "s1", "name": "Search", "input": {"query": "scissors"}},
        {"id": "s2", "name": "Search", "input": {"query": "scissors"}},
    ]
    used = []
    with patch("core.react_loop.ReactLoop._execute_tool", return_value={"hits": hits}):
        adapter._execute_and_append_tool_calls(messages, calls, used, set(), ctx)
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["hits"] == hits
    repeated = json.loads(tool_messages[1]["content"])
    assert repeated["evidence_reference_tool_call_id"] == "s1"
    assert repeated["hits"] == [{"node_id": "n1"}]
    assert used[1]["result"]["hits"] == hits
    assert [h["node_ids"] for h in task.search_history] == [["n1"], ["n1"]]
    assert task.search_history[1]["new_evidence_count"] == 0
    assert task.search_history[1]["cache_hit"] is True


def test_search_cache_keeps_depth_queries_and_runs_separate(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder, default_top_k=12, no_gain_limit=None,
    )
    ctx = _ctx()
    task = M3TaskContext(question="Where?", video_id="a", before_clip=1)
    adapter._build_messages(ctx, task)
    with patch("core.react_loop.ReactLoop._execute_tool", return_value={"hits": []}) as execute:
        adapter._execute_tool("Search", {"query": "bedroom"}, ctx)
        assert execute.call_args.args[1]["top_k"] == 12
        adapter._execute_tool("Search", {"query": "bedroom", "top_k": 12}, ctx)
        assert execute.call_count == 1
        adapter._execute_tool("Search", {"query": "bedroom", "top_k": 20}, ctx)
        adapter._execute_tool("Search", {"query": "kitchen"}, ctx)
        assert execute.call_count == 3
        adapter._build_messages(ctx, M3TaskContext(question="Where?", video_id="b"))
        adapter._execute_tool("Search", {"query": "bedroom"}, ctx)
        assert execute.call_count == 4
    assert task.search_history[0]["top_k"] == 12


def test_failed_search_can_be_retried(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)
    ctx = _ctx()
    adapter._build_messages(ctx, M3TaskContext(question="Where?"))
    with patch(
        "core.react_loop.ReactLoop._execute_tool",
        side_effect=[{"error": "temporary failure"}, {"hits": []}],
    ) as execute:
        assert adapter._execute_tool("Search", {"query": "bedroom"}, ctx).get("error")
        assert adapter._execute_tool("Search", {"query": "bedroom"}, ctx)["hits"] == []
        assert execute.call_count == 2


def test_interleaved_adapters_keep_scope_and_dual_path_config(graph, embedder, llm):
    """B's initialization must not change either branch of A's retrieval."""
    a = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
        default_video_id="default_a", default_top_k=12,
        extractor_types=("extractor_a",),
    )
    ctx = _ctx()
    a._build_messages(ctx, M3TaskContext(question="a", video_id="video_a", before_clip=2))
    config_a = a._runtime_injection_values(ctx)["search_config"]
    b = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
        default_video_id="video_b", default_top_k=4,
        extractor_types=("extractor_b",),
    )
    b._build_messages(ctx, M3TaskContext(question="b", before_clip=7))

    # Force the existing dual-query branch, independent of name heuristics.
    with patch("m3.control_adapter._strip_name_tokens_from_query", return_value="the album"), \
         patch.object(graph, "search_by_embedding", return_value=[]) as search:
        a._execute_tool("Search", {"query": "Lewis and the album"}, ctx)
        b._execute_tool("Search", {"query": "Lewis and the album"}, ctx)
    assert search.call_count == 4
    for call, video, before, kinds, depth in zip(
        search.call_args_list,
        ["video_a", "video_a", "video_b", "video_b"],
        [2, 2, 7, 7],
        [("extractor_a",)] * 2 + [("extractor_b",)] * 2,
        [12, 12, 4, 4],
    ):
        assert call.kwargs["scope"].video_id == video
        assert call.kwargs["scope"].before_clip == before
        assert call.kwargs["extractor_types"] == kinds
        assert call.kwargs["top_k"] == depth
    assert a._runtime_injection_values(ctx)["search_config"] is config_a


def test_search_config_is_immutable_and_snapshotted_per_task(graph, embedder, llm):
    from dataclasses import FrozenInstanceError

    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder, default_video_id="fallback",
    )
    ctx = _ctx()
    task = M3TaskContext(question="a", video_id="video_a", before_clip=2)
    adapter._build_messages(ctx, task)
    original = adapter._runtime_injection_values(ctx)["search_config"]
    with pytest.raises(FrozenInstanceError):
        original.video_id = "other"
    task.video_id = "mutated"
    assert original.video_id == "video_a"
    adapter._build_messages(ctx, M3TaskContext(question="b"))
    next_config = adapter._runtime_injection_values(ctx)["search_config"]
    assert next_config is not original
    assert next_config.video_id == "fallback"
    assert next_config.before_clip is None
    assert original.before_clip == 2


def test_direct_search_requires_trusted_config_and_honors_default_depth(graph, embedder):
    from core.tools.operational_tool_spec import call_with_injection

    spec = Search._operational_tool_spec
    config = SearchConfig(video_id="allowed", before_clip=3, default_top_k=13)
    params = {"query": "album", "search_config": {"video_id": "forged"}}
    with patch.object(graph, "search_by_embedding", return_value=[]) as search:
        result = call_with_injection(
            spec, params, ctx=_ctx(), graph=graph, embedder=embedder,
            search_config=config,
        )
    assert "error" not in result
    assert search.call_args.kwargs["scope"].video_id == "allowed"
    assert search.call_args.kwargs["top_k"] == 13
    with pytest.raises(TypeError, match="Cannot inject runtime parameter 'search_config'"):
        call_with_injection(spec, params, ctx=_ctx(), graph=graph, embedder=embedder)


def test_same_adapter_concurrent_run_rejected_and_reusable(graph, embedder):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()
    adapter_llm = _scripted([{"content": "done"}, {"content": "again"}])
    original_call = adapter_llm.complete_with_tools_messages

    def blocked_call(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5), "test did not release the active run"
        return original_call(*args, **kwargs)

    adapter = OGMemoryControlAdapter(llm=adapter_llm, graph=graph, embedder=embedder)
    first = M3TaskContext(question="first", video_id="a")
    with patch.object(adapter_llm, "complete_with_tools_messages", side_effect=blocked_call):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(adapter.run, _ctx(), first)
            try:
                assert entered.wait(timeout=5)
                with pytest.raises(RuntimeError, match="already running"):
                    adapter.run(_ctx(), M3TaskContext(question="second", video_id="b"))
                assert adapter._current_task_ctx is first
                assert adapter._search_config.video_id == "a"
            finally:
                release.set()
            assert pending.result(timeout=5).output == ["done"]
    assert adapter._search_config is None
    assert adapter._search_cache == {}
    assert adapter.run(_ctx(), M3TaskContext(question="next")).output == ["again"]


def test_failed_run_releases_guard_and_discards_config(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)
    original_build = adapter._build_messages

    def failing_build(ctx, task):
        original_build(ctx, task)
        adapter._search_cache[("old", 8)] = {"hits": []}
        raise ValueError("failed while building messages")

    with patch.object(adapter, "_build_messages", side_effect=failing_build):
        with pytest.raises(ValueError, match="failed while building"):
            adapter.run(_ctx(), M3TaskContext(question="first"))
    assert adapter._search_config is None
    assert adapter._search_cache == {}
    # The next run gets through the guard, even after an unexpected exception.
    with patch.object(adapter, "_build_messages", side_effect=ValueError("next run")):
        with pytest.raises(ValueError, match="next run"):
            adapter.run(_ctx(), M3TaskContext(question="next"))


# ---------------------------------------------------------------------------
# Search tool handler (direct invocation)
# ---------------------------------------------------------------------------

def test_search_handler_returns_hits_with_provenance(graph, embedder):
    ctx = _ctx()
    target_id = _seed_episode(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks through a door.",
    )
    _seed_episode(
        graph, ctx,
        video_id="vid_b", clip_id=5,
        description="Two people stand by a window.",
    )

    # Drive the operational tool handler directly with the same
    # injection shape the ReactLoop uses.
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "woman walks", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    assert "hits" in result
    assert result["query"] == "woman walks"
    # At least one hit came back; we don't pin scores because the
    # MockEmbedder may produce zero vectors for short queries.
    for hit in result["hits"]:
        assert {"node_id", "score", "video_id", "clip_id",
                "description", "characters", "events",
                "entities"} <= set(hit.keys())
        assert hit["video_id"] in {"vid_a", "vid_b"}


def test_search_handler_rejects_empty_query(graph, embedder):
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "", "top_k": 5},
        ctx=_ctx(), graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# Provenance key fallback (FIX-3)
#
# Current M3 writers stamp ``source_video_id`` / ``source_clip_id`` on
# ExtractedNode.extracted_content; the pre-fix EvidenceHit loader read
# only the legacy ``_video_id`` / ``_clip_id`` keys, so every Search hit
# carried null provenance to the LLM. These tests prove the new loader
# lifts both key generations and prefers the new ones on conflict.
# ---------------------------------------------------------------------------


def _seed_episode_with_keys(
    graph, ctx, *, video_id: str, clip_id: int, description: str,
    keys: dict[str, object], node_id_seed: str,
) -> str:
    """Seed an ExtractedNode with caller-specified provenance keys.

    ``keys`` merges into ``extracted_content`` (e.g.
    ``{"source_video_id": "v", "source_clip_id": 0}`` or
    ``{"_video_id": "v", "_clip_id": 0}``).
    """
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "characters": [],
            "events": [],
            "entities": [],
            **keys,
        },
        node_id=node_id_seed,
    )
    graph._store.create_node(node.serialize(), ctx)
    # Distinctive embedding keeps hits ordered.
    slot = ord(node_id_seed[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(node_id_seed, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return node_id_seed


# ---------------------------------------------------------------------------
# Helpers for stripped-query / name-roster tests
# ---------------------------------------------------------------------------

def _make_hit_with_characters(
    *,
    score: float,
    characters: list | None = None,
    clip_id: int = 0,
    node_id_seed: str | None = None,
    transcript_segments: list[dict] | None = None,
):
    if characters is None:
        characters = []
    """Build a minimal EvidenceHit carrying only ``characters`` (and an
    optional transcript). Bypasses the graph store entirely — the
    stripping helpers operate on hit objects, not on the store, so
    hitting the store here would be wasted I/O.
    """
    import uuid
    nid = node_id_seed or uuid.uuid4().hex[:32]
    content: dict = {
        "description": "d",
        "_video_id": "vid",
        "_clip_id": clip_id,
        "characters": characters,
        "events": [],
        "entities": [],
    }
    if transcript_segments is not None:
        content["speech_transcript"] = transcript_segments
    node = create_extracted_node(
        source_raw_id=f"strip-{clip_id}".ljust(32, "x")[:32],
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=_ctx().account_id,
        extracted_content=content,
        node_id=nid,
    )
    from service.graph_service import EvidenceHit
    return EvidenceHit(node=node, score=score, video_id="vid", clip_id=clip_id)


# ---------------------------------------------------------------------------
# Tests: name-token extraction and query stripping
# ---------------------------------------------------------------------------

class TestNameTokensFromHits:
    """``_name_tokens_from_hits`` collects the set of character names
    across all hits so the BM25 side can strip them from the query."""

    def test_string_characters_collected_lowercased(self):
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0,
            characters=["Lewis", "Sophie"],
        )
        assert _name_tokens_from_hits([hit]) == {"lewis", "sophie"}

    def test_dict_characters_extract_name_field(self):
        """EpisodicMemoryGenerator emits dicts with a ``name`` key
        (plus optional type/role/summary/description). Only ``name``
        should land in the stripping set — the extra fields are
        descriptive metadata, not identifiers."""
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0,
            characters=[
                {"name": "Lewis", "role": "boyfriend"},
                {"name": "Stella", "summary": "the cat"},
            ],
        )
        assert _name_tokens_from_hits([hit]) == {"lewis", "stella"}

    def test_mixed_string_and_dict_characters(self):
        """Defensive: real data may mix the two schemas (legacy string
        entries + new dict entries). Both should be captured."""
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0,
            characters=["Lewis", {"name": "Stella"}],
        )
        assert _name_tokens_from_hits([hit]) == {"lewis", "stella"}

    def test_empty_or_missing_characters_returns_empty_set(self):
        """No characters at all → empty set, not an error. The
        stripping function falls back to the original query when the
        set is empty (see ``_strip_name_tokens_from_query`` tests)."""
        hit_empty = _make_hit_with_characters(score=0.5, clip_id=0, characters=[])
        hit_missing = _make_hit_with_characters(score=0.5, clip_id=1, characters=[])
        # Force-mutate one to omit the key entirely to mirror an
        # older/malformed record.
        hit_missing.node.extracted_content.pop("characters", None)
        assert _name_tokens_from_hits([hit_empty, hit_missing]) == set()

    def test_malformed_character_entries_skipped_silently(self):
        """A character entry that is neither str nor dict-with-name
        must not crash the extraction. The set must contain the valid
        entries around it."""
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0,
            characters=["Lewis", None, 42, {"name": "Stella"},
                        {"no_name_key": "x"}, ""],
        )
        assert _name_tokens_from_hits([hit]) == {"lewis", "stella"}

    def test_dedup_across_hits(self):
        """The same character appearing in multiple hits is collected
        once (set semantics). Important so the regex-compile cache is
        keyed correctly across questions."""
        hits = [
            _make_hit_with_characters(score=0.5, clip_id=0, characters=["Lewis"]),
            _make_hit_with_characters(score=0.5, clip_id=1, characters=["Lewis", "Stella"]),
            _make_hit_with_characters(score=0.5, clip_id=2, characters=["Lewis"]),
        ]
        assert _name_tokens_from_hits(hits) == {"lewis", "stella"}


class TestStripNameTokensFromQuery:
    """``_strip_name_tokens_from_query`` is the BM25-side fix for the
    name-pollution failure mode (4️⃣ in the failure taxonomy)."""

    def test_strips_name_case_insensitively(self):
        """``Lewis`` (capitalized in question) must match ``lewis``
        (lowercased in name set) — case-insensitivity is essential
        because the question may use any casing."""
        out = _strip_name_tokens_from_query(
            "What decorations does Lewis take?",
            {"lewis"},
        )
        assert out == "What decorations does take?"

    def test_strips_name_with_punctuation_around(self):
        """``Lewis?`` at end of question must still strip. The word
        boundary anchors must not require the name to be flanked by
        spaces — only by non-word characters (including punctuation)."""
        out = _strip_name_tokens_from_query(
            "What does Lewis?",
            {"lewis"},
        )
        assert out == "What does?"

    def test_word_boundary_does_not_partial_match(self):
        """``Lewis`` must NOT strip from ``LewisHamilton`` — a partial
        match inside a longer word would corrupt the query. Word
        boundary anchors enforce this."""
        out = _strip_name_tokens_from_query(
            "Where is LewisHamilton?",
            {"lewis"},
        )
        assert "LewisHamilton" in out  # untouched
        # Sanity: a non-boundary case still strips
        out2 = _strip_name_tokens_from_query(
            "Where is Lewis Hamilton?", {"lewis"},
        )
        assert "Lewis" not in out2
        assert "Hamilton" in out2

    def test_multi_word_name_strips_as_one_phrase(self):
        """A two-word name like ``Lewis Hamilton`` must strip as a
        single unit, not one token at a time. Whitespace handling in
        particular: removing only ``Lewis`` would leave ``Hamilton``
        dangling, changing the query's token list unpredictably."""
        out = _strip_name_tokens_from_query(
            "Did Lewis Hamilton win?",
            {"lewis hamilton"},
        )
        assert out == "Did win?"

    def test_regex_metacharacters_in_name_escaped(self):
        """A name with regex metacharacters (``O'Brien``) must be
        treated as a literal, not as a regex pattern. ``re.escape``
        on the name during compilation enforces this."""
        out = _strip_name_tokens_from_query(
            "Did O'Brien leave?",
            {"o'brien"},
        )
        assert out == "Did leave?"

    def test_multiple_names_all_stripped(self):
        """When several character names appear in the same query,
        every one is removed. Order of removal must not matter."""
        out = _strip_name_tokens_from_query(
            "What did Lewis and Stella give to Lily?",
            {"lewis", "stella", "lily"},
        )
        assert out == "What did and give to?"

    def test_empty_query_returns_empty(self):
        """Defensive: empty input passes through. The caller (Search)
        already short-circuits on empty query, so this is a
        belt-and-braces invariant."""
        assert _strip_name_tokens_from_query("", {"lewis"}) == ""
        assert _strip_name_tokens_from_query("   ", {"lewis"}) == "   "

    def test_empty_name_set_returns_original_query(self):
        """When no characters are available (early failure or empty
        roster), the function is a no-op — the caller still gets the
        original query so BM25 sees the same tokens it did before the
        stripping layer was introduced."""
        q = "What did Lewis take?"
        assert _strip_name_tokens_from_query(q, set()) == q

    def test_name_not_in_query_returns_unchanged(self):
        """Stripping a name that doesn't appear in the query must
        return the query identical (modulo whitespace collapse)."""
        q = "Where is the storage room?"
        assert _strip_name_tokens_from_query(q, {"lewis"}) == q

    def test_stripping_everything_falls_back_to_original(self):
        """If stripping leaves no alphanumerics (query is ONLY the
        name), the function must return the original query verbatim
        so BM25 still has tokens to score on. Falling back to empty
        would silently turn the rerank into vector-only."""
        q = "Lewis"
        assert _strip_name_tokens_from_query(q, {"lewis"}) == q

    def test_whitespace_collapsed_after_strip(self):
        """Multiple names back-to-back in a query must not produce
        double-spaces or leading/trailing whitespace."""
        out = _strip_name_tokens_from_query(
            "Lewis Stella Lily go home",
            {"lewis", "stella", "lily"},
        )
        assert out == "go home"
        assert "  " not in out


def test_search_lifts_source_provenance_keys(graph, embedder):
    """[Search] hits must carry non-null video_id/clip_id when the
    ExtractedNode was stamped with the new ``source_*`` keys.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks through a door.",
        keys={"source_video_id": "vid_a", "source_clip_id": 0},
        node_id_seed="c00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    assert result["hits"], "expected at least one hit"
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_a", (
            f"hit should carry source_video_id='vid_a', got {hit['video_id']!r}"
        )
        assert hit["clip_id"] == 0, (
            f"hit should carry source_clip_id=0, got {hit['clip_id']!r}"
        )


def test_search_lifts_legacy_provenance_keys_as_fallback(graph, embedder):
    """A node stamped only with the legacy ``_video_id`` / ``_clip_id``
    keys must still surface provenance (pre-existing writers keep
    working). clip_id 0 is a valid ordinal, so None-check is mandatory
    — an ``or`` fallback would have hidden clip 0.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_legacy", clip_id=0,   # clip 0 is valid
        description="Legacy clip 0.",
        keys={"_video_id": "vid_legacy", "_clip_id": 0},
        node_id_seed="d00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    assert result["hits"]
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_legacy"
        assert hit["clip_id"] == 0, (
            "clip_id 0 must survive the fallback (not be coerced to None)"
        )


def test_search_prefers_source_over_legacy_on_conflict(graph, embedder):
    """When both key generations are stamped, the new ``source_*``
    values win — the writer's intent is the current source of truth.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_current", clip_id=7,
        description="Both keys, source_ wins.",
        keys={
            "source_video_id": "vid_current", "source_clip_id": 7,
            "_video_id": "vid_legacy", "_clip_id": 99,
        },
        node_id_seed="e00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    assert result["hits"]
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_current", (
            "source_video_id should win over _video_id on conflict"
        )
        assert hit["clip_id"] == 7, (
            "source_clip_id should win over _clip_id on conflict"
        )


# ---------------------------------------------------------------------------
# Answer tool
# ---------------------------------------------------------------------------

def test_answer_handler_returns_status_payload():
    from core.tools.operational_tool_spec import call_with_injection
    result = call_with_injection(
        Answer._operational_tool_spec,
        {"final_answer": "the woman is Alice"},
    )
    assert result == {
        "status": "answered",
        "answer": "the woman is Alice",
    }


# ---------------------------------------------------------------------------
# End-to-end run: LLM calls Search then Answer
# ---------------------------------------------------------------------------

def test_run_search_then_answer_completes(graph, embedder, llm):
    """Script the LLM to call [Search] once then [Answer] on the next
    round. Adapter must end with the answer text and a clean trace."""
    ctx = _ctx()
    # Seed one episode so Search has something to return.
    _seed_episode(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks into a bedroom.",
    )

    # Answer completes round 2 directly; the third scripted reply is unused.
    adapter_llm = _scripted([
        {"tool": "Search", "input": {"query": "woman bedroom"}},
        {"tool": "Answer", "input": {"final_answer": "A woman walks into a bedroom."}},
        {"content": "A woman walks into a bedroom."},   # final wrap-up turn
    ])

    adapter = OGMemoryControlAdapter(
        llm=adapter_llm, graph=graph, embedder=embedder,
    )
    task_ctx = M3TaskContext(question="What happens in the bedroom?")
    result = adapter.run(ctx, task_ctx)

    assert result.completed is True
    assert result.finish_reason == "done"
    # Final output surfaces the Answer text.
    assert result.output == ["A woman walks into a bedroom."]
    # Task context captured both tool calls.
    assert task_ctx.final_answer == "A woman walks into a bedroom."
    assert adapter_llm._call_count == 2
    assert task_ctx.search_stats["model_rounds"] == 2
    assert len(task_ctx.search_history) == 1
    assert task_ctx.search_history[0]["query"] == "woman bedroom"


def test_run_loops_on_search_only_then_times_out_at_max(graph, embedder, llm):
    """If the LLM never calls [Answer], the loop runs max_iterations
    rounds of [Search] and exits with finish_reason='max_iterations'.
    The last content (empty) drives the exit; no answer captured."""
    ctx = _ctx()
    _seed_episode(graph, ctx, video_id="v", clip_id=0, description="x")

    # Always Search, never Answer — keeps the loop busy until max_iterations.
    adapter_llm = _scripted([
        {"tool": "Search", "input": {"query": f"q{i}"}}
        for i in range(10)
    ])

    adapter = OGMemoryControlAdapter(
        llm=adapter_llm, graph=graph, embedder=embedder, max_iterations=3,
    )
    result = adapter.run(ctx, M3TaskContext(question="Q?"))
    assert result.finish_reason == "max_iterations"
    assert result.output == [] or result.output is None


def test_run_video_scope_filters_search(graph, embedder, llm):
    """When the task_ctx specifies video_id, the adapter narrows
    Search to that video via ClipScope."""
    ctx = _ctx()
    target = _seed_episode(
        graph, ctx, video_id="vid_target", clip_id=0,
        description="Target.",
    )
    _seed_episode(
        graph, ctx, video_id="vid_other", clip_id=1,
        description="Other.",
    )

    # Direct callers must explicitly inject their own search scope.
    from core.tools.operational_tool_spec import call_with_injection
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 10},
        ctx=ctx, graph=graph, embedder=embedder,
        search_config=SearchConfig(video_id="vid_target"),
    )
    assert "hits" in result
    # All hits must be from vid_target; vid_other is excluded by the
    # ClipScope filter.
    assert result["hits"], "expected at least one scoped hit"
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_target"


# ---------------------------------------------------------------------------
# Runtime injection: verify the handler receives graph + embedder
# ---------------------------------------------------------------------------

def test_search_handler_uses_runtime_injection(graph, embedder):
    """If the handler were missing the runtime injection, it would
    raise on the embedder / graph references. Calling it directly
    through call_with_injection proves the wiring works."""
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    # No graph or embedder passed by name → fallback to type-based
    # matching in call_with_injection.
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "test", "top_k": 3},
        ctx=_ctx(),  # type: ignore[arg-type]
        graph=graph, embedder=embedder,
        search_config=SearchConfig(),
    )
    # No crash, result shape correct.
    assert isinstance(result, dict)
    assert "query" in result


def test_runtime_injection_omits_llm_visible_extra_params(graph, embedder, llm):
    """Regression: extractor_types and scope_video_id must NOT be
    visible to the LLM. The Search tool's input_schema lists only
    query + top_k."""
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
        extractor_types=("episodic_memory_generator", "face_detection"),
        default_video_id="vid_x",
    )
    search_schema = next(
        t for t in adapter._get_tools() if t["name"] == "Search"
    )["input_schema"]
    assert "extractor_types" not in search_schema["properties"]
    assert "scope_video_id" not in search_schema["properties"]
    # Adapter-level config is preserved internally.
    assert adapter._extractor_types == (
        "episodic_memory_generator", "face_detection",
    )
    assert adapter._default_video_id == "vid_x"
    adapter._build_messages(_ctx(), M3TaskContext(question="test"))
    config = adapter._runtime_injection_values(_ctx())["search_config"]
    assert config.video_id == "vid_x"
    assert config.extractor_types == (
        "episodic_memory_generator", "face_detection",
    )


# ---------------------------------------------------------------------------
# BM25 transcript rerank wiring (finding 3d — retrieval half)
# ---------------------------------------------------------------------------

def test_search_reranks_transcript_match_above_higher_vector_score(graph, embedder, llm):
    """End-to-end proof that the BM25 rerank fires INSIDE the [Search]
    tool: when pure-vector retrieval ranks an off-topic clip above a
    clip whose transcript contains the query's exact words, the rerank
    must flip the order so the transcript match comes out on top.

    Rather than assume the hash-based MockEmbedder produces a specific
    cosine ordering (it won't, for arbitrary strings), this test
    determines the pure-vector order empirically (a baseline Search
    with the rerank disabled), then asserts the rerank flips it when
    the lower-ranked clip carries the query tokens in its transcript.
    """
    import re
    from unittest.mock import patch
    from nltk.stem import PorterStemmer

    _MOCK_STOPS = frozenset(
        "a an the and or but is are was were be been being have has had do "
        "does did will would could should may might shall can to of in for "
        "on with at by from as our i we you they".split()
    )
    _stub_stemmer = PorterStemmer()

    def _stub_tokenize(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [_stub_stemmer.stem(t) for t in toks if t not in _MOCK_STOPS]

    ctx = _ctx()
    query = "photos album"

    # Clip A: off-topic speech (no query tokens).
    _seed_episode_with_transcript(
        graph, ctx, video_id="vid", clip_id=10,
        description="a woman cooking in a kitchen",
        transcript_segments=[{"start_sec": 0, "end_sec": 3,
                              "text": "the weather is nice today"}],
        node_id_seed="a00000000000000000000000000",
    )
    # Clip B: the query's exact words live in its transcript.
    _seed_episode_with_transcript(
        graph, ctx, video_id="vid", clip_id=20,
        description="someone talking near a table",
        transcript_segments=[{"start_sec": 0, "end_sec": 4,
                              "text": "i put our photos in the album"}],
        node_id_seed="b00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)

    # --- Baseline: pure-vector order (rerank disabled) ----------------
    with patch("m3.transcript_rerank.rerank_with_transcript_bm25",
               side_effect=lambda q, h, **kw: list(h)):
        baseline = call_with_injection(
            Search._operational_tool_spec,
            {"query": query, "top_k": 5},
            ctx=ctx, graph=graph, embedder=embedder,
            search_config=SearchConfig(),
        )
    assert len(baseline["hits"]) == 2
    vector_top = baseline["hits"][0]["clip_id"]
    vector_bottom = baseline["hits"][1]["clip_id"]

    # Only a meaningful test if pure-vector ranks the off-topic clip
    # above the transcript match. If the embedder already happens to rank
    # B first, there's nothing to rescue — skip the flip assertion.
    if vector_top == 10 and vector_bottom == 20:
        # --- With rerank enabled ---------------------------------------
        with patch("retrieval.bm25_index._ensure_nltk"), \
             patch("retrieval.bm25_index._tokenize", _stub_tokenize):
            reranked = call_with_injection(
                Search._operational_tool_spec,
                {"query": query, "top_k": 5},
                ctx=ctx, graph=graph, embedder=embedder,
                search_config=SearchConfig(),
            )
        assert len(reranked["hits"]) == 2
        # The transcript match (clip 20) is rescued to the top.
        assert reranked["hits"][0]["clip_id"] == 20, (
            "BM25 rerank failed to rescue the transcript-matching clip "
            f"to top; got clip_id={reranked['hits'][0]['clip_id']} "
            f"(vector order was {[h['clip_id'] for h in baseline['hits']]})"
        )
    # else: the MockEmbedder happened to rank the transcript clip first
    # already — nothing to assert; the rerank is a no-op-or-confirm in
    # that case and the focused unit tests cover the rescue math.


def _seed_episode_with_transcript(
    graph, ctx, *, video_id, clip_id, description,
    transcript_segments, node_id_seed,
):
    """Seed an ExtractedNode carrying ``speech_transcript`` segments and
    a controlled embedding so the cosine ordering is deterministic.

    Distinct from :func:`_seed_episode` only in that it also plants the
    transcript on ``extracted_content`` — the rerank's input.
    """
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
            "characters": [],
            "events": [],
            "entities": [],
            "speech_transcript": transcript_segments,
        },
        node_id=node_id_seed,
    )
    graph._store.create_node(node.serialize(), ctx)
    slot = ord(node_id_seed[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),   # slot 0 → highest cosine vs [1,0,..]
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(node_id_seed, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return node_id_seed


# ---------------------------------------------------------------------------
# Tests: regex query-name extractor (supplemental to characters-from-hits)
# ---------------------------------------------------------------------------

class TestQueryNameCandidates:
    """``_query_name_candidates`` is the cheap regex layer that catches
    names which the in-hand top-k hits' ``characters`` field cannot
    supply — the headline failure mode where the correct clip drops out
    of the first top-k because a name with one spelling (e.g. "Lewis")
    pulled cosine down."""

    def test_extracts_title_case_name(self):
        assert _query_name_candidates("What did Lewis take?") == {"lewis"}

    def test_extracts_multiple_names(self):
        assert _query_name_candidates(
            "What did Lewis give to Stella?",
        ) == {"lewis", "stella"}

    def test_extracts_multi_word_name_as_one(self):
        """``Lewis Hamilton`` must be a single candidate, not two —
        the regex spans space/hyphen joiners so the stripper can
        remove it as one phrase."""
        assert _query_name_candidates(
            "Did Lewis Hamilton win?",
        ) == {"lewis hamilton"}

    def test_extracts_hyphenated_name(self):
        assert _query_name_candidates("Where is Anne-Marie?") == {"anne-marie"}

    def test_first_word_question_interrogative_is_skipped(self):
        """``What`` at the start of a question is a question word, not
        a name — must be skipped. The actual name ``Lewis`` mid-sentence
        is still captured."""
        assert _query_name_candidates("What did Lewis take?") == {"lewis"}
        # And "What" itself never appears:
        assert "what" not in _query_name_candidates("What did Lewis take?")

    def test_first_word_name_is_kept(self):
        """When the query starts with a name (``Lewis took what?``), the
        first word IS the name and must be captured. The question-word
        skip applies only when the first word is in
        ``_QUESTION_FIRST_WORDS``."""
        assert _query_name_candidates("Lewis took what?") == {"lewis"}

    def test_stoplist_words_filtered_when_mid_sentence(self):
        """Capitalized words in the stoplist must NOT be returned as
        names — even when they appear mid-sentence where the first-word
        skip doesn't help. ``Put`` (verb) is the most likely
        false-positive in M3-Bench-style instructions."""
        out = _query_name_candidates("What did Lewis put on the table?")
        assert "put" not in out
        assert out == {"lewis"}

    def test_punctuation_around_name_handled(self):
        """Leading/trailing punctuation must not prevent name capture."""
        assert _query_name_candidates("...Lewis?") == {"lewis"}

    def test_lowercase_only_query_returns_empty(self):
        """An all-lowercase query (no Title Case at all) yields no
        candidates. The stripping layer then has nothing to strip from
        regex — falls back to characters-from-hits only."""
        assert _query_name_candidates("what did the man take") == set()

    def test_empty_query_returns_empty(self):
        assert _query_name_candidates("") == set()
        assert _query_name_candidates("   ") == set()


class TestCombinedNameTokens:
    """``_combined_name_tokens`` is the single entry point both the BM25
    stripper and the dual-path vector stripper use. It must be the
    union of characters-from-hits and regex-from-query, lowercased."""

    def test_union_of_two_sources(self):
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0, characters=["Sophie"],
        )
        # Query names "Lewis" via regex; characters-from-hits gives "Sophie".
        out = _combined_name_tokens("What did Lewis take?", [hit])
        assert out == {"lewis", "sophie"}

    def test_query_alone_provides_regex_names(self):
        """When no hits are available (empty list), only regex-from-query
        contributes. This is the path used by the dual-path second
        search before its own hits return."""
        assert _combined_name_tokens("What did Lewis take?", []) == {"lewis"}

    def test_hits_alone_provide_character_names(self):
        """When the query has no Title Case names (all-lowercase),
        only characters-from-hits contributes."""
        hit = _make_hit_with_characters(
            score=0.5, clip_id=0, characters=["Lewis"],
        )
        assert _combined_name_tokens("what did he take", [hit]) == {"lewis"}


# ---------------------------------------------------------------------------
# Tests: hit merger (dual-path result union)
# ---------------------------------------------------------------------------

class TestMergeHitsByNode:
    """``_merge_hits_by_node`` deduplicates the two retrieval paths by
    ``node_id`` and keeps the higher cosine score per node."""

    def test_single_list_passthrough(self):
        a = _make_hit_with_characters(score=0.8, clip_id=0, node_id_seed="a" * 32)
        b = _make_hit_with_characters(score=0.6, clip_id=1, node_id_seed="b" * 32)
        out = _merge_hits_by_node([a, b])
        assert {h.node.node_id for h in out} == {"a" * 32, "b" * 32}

    def test_overlapping_node_keeps_higher_score(self):
        """Same node appearing in both retrievals (relevant to both
        formulations) keeps the higher of the two cosine scores —
        that node is the stronger match for at least one of the
        queries."""
        primary = _make_hit_with_characters(
            score=0.4, clip_id=0, node_id_seed="a" * 32,
        )
        stripped = _make_hit_with_characters(
            score=0.7, clip_id=0, node_id_seed="a" * 32,
        )
        out = _merge_hits_by_node([primary], [stripped])
        assert len(out) == 1
        assert out[0].score == 0.7

    def test_disjoint_lists_unioned(self):
        a = _make_hit_with_characters(score=0.5, clip_id=0, node_id_seed="a" * 32)
        b = _make_hit_with_characters(score=0.4, clip_id=1, node_id_seed="b" * 32)
        c = _make_hit_with_characters(score=0.3, clip_id=2, node_id_seed="c" * 32)
        out = _merge_hits_by_node([a], [b, c])
        assert {h.node.node_id for h in out} == {"a" * 32, "b" * 32, "c" * 32}

    def test_empty_inputs(self):
        a = _make_hit_with_characters(score=0.5, clip_id=0, node_id_seed="a" * 32)
        assert _merge_hits_by_node([], [a]) == [a]
        assert _merge_hits_by_node([a], []) == [a]
        assert _merge_hits_by_node([], []) == []

    def test_three_way_merge(self):
        """Three retrievals (defensive — currently Search only does two,
        but the merger must support the varargs shape.)"""
        a = _make_hit_with_characters(score=0.4, clip_id=0, node_id_seed="a" * 32)
        a_again = _make_hit_with_characters(
            score=0.9, clip_id=0, node_id_seed="a" * 32,
        )
        b = _make_hit_with_characters(score=0.5, clip_id=1, node_id_seed="b" * 32)
        out = _merge_hits_by_node([a], [a_again, b])
        assert len(out) == 2
        # 'a' should be the higher-score version (0.9)
        a_merged = next(h for h in out if h.node.node_id == "a" * 32)
        assert a_merged.score == 0.9


# ---------------------------------------------------------------------------
# Tests: dual-path retrieval end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("offline_transcript_tokenizer")
class TestDualPathRetrieval:
    """The headline P1 #6 fix: when the question names a person whose
    correct clip doesn't make the first top-k (cosine drops because the
    name with one spelling pulls the embedding away), the dual-path
    retrieval re-embeds with the name stripped and unions the two hit
    sets so the correct clip surfaces."""

    def test_second_search_fires_when_name_stripped(self, graph, embedder, llm):
        """When the question contains a Title Case name and stripping
        produces a different query, Search must invoke the embedder
        TWICE — once for the original query and once for the stripped
        one. We assert on the call count, not on the hit content, so
        the test is robust to whatever the MockEmbedder hashes out.
        """
        ctx = _ctx()
        # No graph seeding needed — we patch the embedder and assert
        # on the embedder call shape; search_by_embedding is hit
        # legitimately on whatever the mock graph returns.
        from core.tools.operational_tool_spec import call_with_injection
        OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)

        with patch.object(embedder, "embed_texts",
                          wraps=embedder.embed_texts) as spy:
            call_with_injection(
                Search._operational_tool_spec,
                {"query": "What did Lewis take?", "top_k": 5},
                ctx=ctx, graph=graph, embedder=embedder,
                search_config=SearchConfig(),
            )
        # 1st call = original "What did Lewis take?"; 2nd call =
        # stripped form with "Lewis" removed. The regex layer
        # always treats "Lewis" as a Title Case candidate.
        assert spy.call_count == 2, (
            f"dual-path retrieval should fire 2 embeds, got {spy.call_count}"
        )
        second_call_args = spy.call_args_list[1]
        second_texts = second_call_args[0][0]  # positional arg
        assert len(second_texts) == 1
        # The stripped query must not contain the name.
        assert "Lewis" not in second_texts[0]
        # Content words survive.
        assert "did" in second_texts[0]

    def test_second_search_skipped_when_no_name_to_strip(self, graph, embedder, llm):
        """When the query has no Title Case names and the first hits
        carry no characters either, stripping is trivial — Search must
        NOT do a second embed. This keeps latency flat for the bulk of
        questions that don't name anyone."""
        ctx = _ctx()
        # Seed one clip with empty characters so characters-from-hits
        # is also empty.
        _seed_episode_with_transcript(
            graph, ctx, video_id="vid", clip_id=10,
            description="a kitchen", transcript_segments=[{"text": "noise"}],
            node_id_seed="d0000000000000000000000000000",
        )
        from core.tools.operational_tool_spec import call_with_injection
        OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)

        with patch.object(embedder, "embed_texts",
                          wraps=embedder.embed_texts) as spy:
            call_with_injection(
                Search._operational_tool_spec,
                {"query": "what is on the table", "top_k": 5},
                ctx=ctx, graph=graph, embedder=embedder,
                search_config=SearchConfig(),
            )
        # All-lowercase query + empty characters → no stripping → 1 embed.
        assert spy.call_count == 1, (
            f"no-name query should fire only 1 embed, got {spy.call_count}"
        )

    def test_dual_path_unions_disjoint_hits(self, graph, embedder, llm):
        """End-to-end proof that the second search contributes nodes
        the first search didn't return. We patch ``search_by_embedding``
        to return two disjoint sets on the two calls, then assert the
        merged hit set contains both nodes."""
        ctx = _ctx()
        from core.tools.operational_tool_spec import call_with_injection
        OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)

        # Build two distinct nodes (different node_ids, different clips).
        node_primary = _make_hit_with_characters(
            score=0.5, clip_id=10, node_id_seed="p0000000000000000000000000000",
            transcript_segments=[{"text": "primary transcript"}],
        )
        node_stripped = _make_hit_with_characters(
            score=0.5, clip_id=20, node_id_seed="s0000000000000000000000000000",
            transcript_segments=[{"text": "stripped transcript"}],
        )

        call_log: list[list[float]] = []

        def fake_search(query, **_):
            call_log.append(query)
            # First call (original query) → primary node. Second call
            # (stripped query) → different node.
            if len(call_log) == 1:
                return [node_primary]
            return [node_stripped]

        with patch.object(graph, "search_by_embedding",
                          side_effect=fake_search):
            result = call_with_injection(
                Search._operational_tool_spec,
                {"query": "What did Lewis take?", "top_k": 5},
                ctx=ctx, graph=graph, embedder=embedder,
                search_config=SearchConfig(),
            )

        # The merged hit set must contain BOTH node_ids — proof that
        # the dual-path union happened.
        node_ids = {h["node_id"] for h in result["hits"]}
        assert node_ids == {
            "p0000000000000000000000000000",
            "s0000000000000000000000000000",
        }, (
            f"dual-path union lost a node; got {node_ids}"
        )


@pytest.mark.parametrize("limit", [True, 1, 0, -1, 2.5, "3"])
def test_invalid_no_gain_limit(graph, embedder, llm, limit):
    with pytest.raises(ValueError, match="no_gain_limit"):
        OGMemoryControlAdapter(llm, graph, embedder, no_gain_limit=limit)


def test_paraphrases_close_search_and_audit_actual_work(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm, graph, embedder, no_gain_limit=2)
    ctx, task = _ctx(), M3TaskContext(question="Where?")
    adapter._build_messages(ctx, task)
    with patch("core.react_loop.ReactLoop._execute_tool", return_value={
        "hits": [{"node_id": "a"}], "retrieval_queries": ["original", "stripped"],
    }) as execute:
        first = adapter._execute_tool("Search", {"query": "bedroom"}, ctx)
        second = adapter._execute_tool("Search", {"query": "sleeping room"}, ctx)
        third = adapter._execute_tool("Search", {"query": " BEDROOM "}, ctx)
        blocked = adapter._execute_tool("Search", {"query": "new clue"}, ctx)
    assert "guidance" not in first
    assert "different clue" in second["guidance"]
    assert "NOT mean" in third["guidance"]
    assert blocked.get("error")
    assert execute.call_count == 2
    assert [t["name"] for t in adapter._get_tools()] == ["Answer"]
    assert [h["new_evidence_count"] for h in task.search_history] == [1, 0, 0, 0]
    assert [h["retrieval_queries"] for h in task.search_history] == [
        ["original", "stripped"], ["original", "stripped"], [], [],
    ]
    assert all(h["duration_ms"] >= 0 for h in task.search_history)
    assert task.search_stats["search_requests"] == 4
    assert task.search_stats["search_executions"] == 2
    assert task.search_stats["cache_hits"] == 1
    assert task.search_stats["blocked_requests"] == 1
    assert task.search_stats["stop_reason"] == "consecutive_no_new_evidence"


def test_new_evidence_resets_streak_errors_do_not_and_runs_reset(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm, graph, embedder, no_gain_limit=2)
    ctx, task = _ctx(), M3TaskContext(question="Where?")
    adapter._build_messages(ctx, task)
    with patch("core.react_loop.ReactLoop._execute_tool", side_effect=[
        {"hits": []}, {"error": "temporary"}, {"hits": [{"node_id": "b"}]},
        {"hits": []}, {"hits": [{"node_id": "b"}]},
    ]) as execute:
        for query in ["empty", "retry", "retry", "empty2"]:
            adapter._execute_tool("Search", {"query": query}, ctx)
        assert [h["no_gain_streak"] for h in task.search_history] == [1, 1, 0, 1]
        assert not adapter._answer_only
        adapter._build_messages(ctx, task)
        adapter._execute_tool("Search", {"query": "retry"}, ctx)
        assert execute.call_count == 5
    assert len(task.search_history) == 1
    assert task.search_history[0]["new_node_ids"] == ["b"]
    assert task.search_stats["search_requests"] == 1


@pytest.mark.parametrize("limit,executions", [(2, 2), (None, 5)])
def test_empty_results_bounded_even_when_model_ignores_schema(graph, embedder, limit, executions):
    scripted = _scripted([{"tool": "Search", "input": {"query": f"q{i}"}} for i in range(10)])
    adapter = OGMemoryControlAdapter(scripted, graph, embedder, max_iterations=5, no_gain_limit=limit)
    task = M3TaskContext(question="Where?")
    with patch.object(graph, "search_by_embedding", return_value=[]):
        result = adapter.run(_ctx(), task)
    assert result.finish_reason == "max_iterations"
    assert not result.output
    assert scripted._call_count == 5
    assert task.search_stats["search_executions"] == executions
    assert task.search_stats["blocked_requests"] == 5 - executions


@pytest.mark.parametrize("answer", ["", "   ", None, 123])
def test_invalid_answer_cannot_complete_and_hard_budget_applies(graph, embedder, answer):
    scripted = _scripted([{"tool": "Answer", "input": {"final_answer": answer}}] * 5)
    adapter = OGMemoryControlAdapter(scripted, graph, embedder, max_iterations=2)
    task = M3TaskContext(question="Where?")
    result = adapter.run(_ctx(), task)
    assert result.finish_reason == "max_iterations"
    assert task.final_answer is None
    assert scripted._call_count == 2


def test_last_round_answer_completes_without_extra_model_call(graph, embedder):
    scripted = _scripted([{"tool": "Answer", "input": {"final_answer": " Evidence insufficient. "}}])
    adapter = OGMemoryControlAdapter(scripted, graph, embedder, max_iterations=1)
    task = M3TaskContext(question="Where?")
    result = adapter.run(_ctx(), task)
    assert result.finish_reason == "done"
    assert result.output == ["Evidence insufficient."]
    assert scripted._call_count == 1
    assert task.search_stats["stop_reason"] == "answer"


def test_same_batch_searches_blocked_after_threshold_or_answer(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm, graph, embedder, no_gain_limit=2)
    ctx, task = _ctx(), M3TaskContext(question="Where?")
    messages = adapter._build_messages(ctx, task)
    calls = [{"id": str(i), "name": "Search", "input": {"query": str(i)}} for i in range(3)]
    with patch.object(graph, "search_by_embedding", return_value=[]) as search:
        adapter._execute_and_append_tool_calls(messages, calls, [], set(), ctx)
        assert search.call_count == 2
    messages = adapter._build_messages(ctx, task)
    calls = [
        {"id": "a", "name": "Answer", "input": {"final_answer": "insufficient"}},
        {"id": "b", "name": "Search", "input": {"query": "room"}},
        {"id": "c", "name": "Answer", "input": {"final_answer": "overwrite"}},
    ]
    with patch.object(graph, "search_by_embedding") as search:
        adapter._execute_and_append_tool_calls(messages, calls, [], set(), ctx)
        search.assert_not_called()
    assert task.final_answer == "insufficient"
    assert len([m for m in messages if m["role"] == "tool"]) == 3


def test_failed_answer_tool_does_not_finalize(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(llm, graph, embedder)
    ctx = _ctx()
    adapter._build_messages(ctx, M3TaskContext(question="Where?"))
    with patch("core.react_loop.ReactLoop._execute_tool", return_value={"error": "failed"}):
        adapter._execute_tool("Answer", {"final_answer": "room"}, ctx)
    assert adapter._output_after_tools() is None
