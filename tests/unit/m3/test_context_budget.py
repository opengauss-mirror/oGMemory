import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.errors import LLMCallError
from core.models import RequestContext
from m3.context_budget import ContextBudget, load_context_budget, pack_context, request_units
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, Search, SearchConfig
from providers.llm import MockLLM


def context():
    return RequestContext(account_id="a", user_id="u", agent_id="g", session_id="s", trace_id="t")

def hit(node="n1", text="Scissors were taken from the toolbox. Glue was also removed.", **extra):
    return {
        "node_id": node, "video_id": "v1", "clip_id": 1, "source_start_sec": 30,
        "description": text, "speech_transcript": [
            {"start_sec": 1, "end_sec": 3, "text": "Get the scissors, please."},
        ], **extra,
    }


def exchange(call, query, hits):
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call, "type": "function",
            "function": {"name": "Search", "arguments": json.dumps({"query": query})},
        }]},
        {"role": "tool", "tool_call_id": call, "content": json.dumps({
            "query": query, "hits": hits, "context_hits": [], "boundary_hits": [],
        })},
    ]


def messages(*exchanges):
    return [{"role": "system", "content": "Use the evidence."},
            {"role": "user", "content": "What came from the toolbox?"},
            *[m for pair in exchanges for m in pair]]


def evidence(view):
    records = [json.loads(m["content"]) for m in view if m["role"] == "tool"]
    docs = {k: v for r in records for k, v in r["evidence"].items()}
    assert all(ref["id"] in docs for r in records for ref in r["evidence_refs"])
    return records, docs


def test_cross_query_dedup_keeps_sources_and_raw_audit_unchanged():
    original = messages(exchange("a", "scissors", [hit()]),
                        exchange("b", "toolbox items", [hit(score=.1)]))
    before = deepcopy(original)
    view, stats = pack_context(original, [], "toolbox scissors", ContextBudget())
    records, docs = evidence(view)
    assert original == before
    assert len(docs) == 1
    assert records[1]["evidence"] == {}
    assert records[0]["evidence_refs"][0]["id"] == records[1]["evidence_refs"][0]["id"]
    assert stats["repeated_occurrences"] == 1
    doc = next(iter(docs.values()))
    for p in doc["passages"]:
        if p["field"] == "description":
            assert p["text"] == hit()["description"][p["start_char"]:p["end_char"]]
        else:
            assert p["value"] == hit()[p["field"]][p["index"]]


def test_content_changes_and_temporal_projection_are_not_deduplicated():
    original = messages(exchange("a", "x", [hit()]), exchange("b", "y", [
        hit(text="The scissors are now on the desk."),
        hit(source_end_sec=32, speech_transcript=[{"text": "Get", "end_sec": 2}]),
    ]))
    view, stats = pack_context(original, [], "scissors", ContextBudget())
    assert stats["unique_evidence_versions"] == 3
    assert len(evidence(view)[1]) == 3


def test_clip_narrative_retains_adjacent_speech_not_just_keyword_matches():
    source = hit(text="The person prepares a paper part.", speech_transcript=[
        {"text": "Prepare this part.", "start_sec": 0, "end_sec": 1},
        {"text": "First fold, then roll, finally attach.", "start_sec": 1, "end_sec": 3},
    ])
    view, _ = pack_context(messages(exchange("a", "prepare paper part", [source])),
                           [], "prepare paper part", ContextBudget())
    doc = next(iter(evidence(view)[1].values()))
    assert [p["value"] for p in doc["passages"] if p["field"] == "speech_transcript"] == source["speech_transcript"]


def test_large_multilingual_history_fits_and_references_never_dangle():
    original = messages(*[
        exchange(str(i), f"scissors {i}", [
            hit(str(n), text=("剪刀。Scissors cut paper. " * 30)) for n in range(i, i + 8)
        ]) for i in range(8)
    ])
    budget = ContextBudget(max_input_tokens=6500)
    tools = [{"name": "Search", "description": "检索", "input_schema": {"type": "object"}}]
    view, stats = pack_context(original, tools, "scissors", budget)
    assert request_units(view, tools) <= budget.input_limit
    assert stats["selected_passages"] > 0
    assert stats["admission_checks"] <= budget.max_admission_checks
    assert stats["unpacked_input_units"] > stats["estimated_input_tokens"]
    evidence(view)
    second, _ = pack_context(original, tools, "paper", budget)
    evidence(second)  # re-selection must rebuild every reference, not reuse old IDs


def test_no_external_or_cross_question_state():
    first, _ = pack_context(messages(exchange("a", "x", [hit("secret", "private first account")])),
                            [], "private", ContextBudget())
    second, _ = pack_context(messages(exchange("b", "x", [hit("public")])),
                             [], "scissors", ContextBudget())
    assert "private first account" in json.dumps(first)
    assert "private first account" not in json.dumps(second)


def test_control_messages_or_tool_schema_over_budget_fail_closed():
    budget = ContextBudget(max_input_tokens=1000)
    original = messages()
    with pytest.raises(LLMCallError, match="request not sent"):
        pack_context(original, [{"name": "huge", "description": "x" * 2000}], "", budget)
    original.append({"role": "assistant", "content": "y" * 2000})
    with pytest.raises(LLMCallError, match="request not sent"):
        pack_context(original, [], "", budget)


def test_oversized_node_is_explicitly_omitted_without_a_dangling_reference():
    original = messages(exchange("a", "x", [hit(text="x" * 1000)]))
    view, stats = pack_context(original, [], "x", ContextBudget(max_node_bytes=100))
    records, docs = evidence(view)
    assert not docs
    assert records[0]["retrieved_nodes"] == records[0]["omitted_nodes"] == 1
    assert records[0]["evidence_refs"] == []
    assert stats["oversized_node_occurrences"] == 1


def test_budget_reserves_output_and_safety():
    budget = ContextBudget(context_window=10000, max_input_tokens=9000,
                           max_output_tokens=3000, safety_tokens=1000)
    assert budget.input_limit == 6000
    for settings in ({"max_search_calls": 0}, {"max_top_k": True},
                     {"context_window": 1000}, {"max_candidates": 1},
                     {"max_neighbor_clips": 0}, {"max_neighbor_clips": True}):
        with pytest.raises(ValueError):
            ContextBudget(**settings)
    for settings in ({"tokenizer_multiplier": .5}, {"tokenizer_multiplier": float("nan")},
                     {"tokenizer_path": False}):
        with pytest.raises(ValueError):
            ContextBudget(**settings)


def test_packing_cpu_admission_checks_are_bounded():
    original = messages(exchange("a", "scissors", [hit(str(i)) for i in range(50)]))
    budget = ContextBudget(max_admission_checks=2)
    view, stats = pack_context(original, [], "scissors", budget)
    assert stats["admission_checks"] == 2
    assert stats["admission_limit_reached"]
    assert stats["selected_nodes"] == 2
    assert request_units(view, [], budget) <= budget.input_limit
    evidence(view)


def test_optional_tokenizer_counts_full_input_without_saved_truncation(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from m3.context_budget import _local_tokenizer

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "word": 1},
                                                               unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=500)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    loaded, digest = _local_tokenizer(str(path))
    assert len(loaded.encode("word " * 200).ids) == 200
    assert len(digest) == 64
    budget = ContextBudget(tokenizer_path=str(path))
    short = request_units([{"role": "user", "content": "word"}], [], budget)
    long = request_units([{"role": "user", "content": "word " * 200}], [], budget)
    assert long > short + 200


def test_loader_opt_in_and_output_reservation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("m3:\n  qa_context:\n    enabled: false\n")
    assert load_context_budget(str(path)) is None
    path.write_text("m3:\n  qa_llm:\n    max_tokens: 2048\n  qa_context:\n    enabled: true\n")
    assert load_context_budget(str(path)).max_output_tokens == 2048
    path.write_text("m3:\n  qa_context:\n    enabled: true\n    max_neighbor_clips: 7\n")
    assert load_context_budget(str(path)).max_neighbor_clips == 7
    path.write_text("m3:\n  qa_context:\n    enabled: 'false'\n")
    with pytest.raises(ValueError):
        load_context_budget(str(path))


def test_search_caps_model_top_k_and_candidate_depth():
    graph = Mock()
    graph.search_by_embedding.return_value = []
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.]]
    result = Search("scissors", top_k=1000000, graph=graph, embedder=embedder,
                    ctx=context(),
                    search_config=SearchConfig(candidate_k=1000000, context_budget=ContextBudget()))
    assert result["final_top_k"] == 16
    assert result["candidate_k"] == 32
    assert all(c.kwargs["top_k"] <= 32 for c in graph.search_by_embedding.call_args_list)


def test_total_result_cap_includes_boundary_and_neighbors(monkeypatch):
    from core.graph_models import create_extracted_node
    from service.graph_service import EvidenceHit

    def entry(i, boundary=False):
        body = hit(str(i), text="" if boundary else "Scissors", source_start_sec=float(i),
                   source_duration_sec=1., source_end_sec=float(i + 1))
        node = create_extracted_node(
            source_raw_id="0" * 32, extractor_type="episodic_memory_generator",
            extraction_schema="m3_bench_v1", account_id="a",
            extracted_content=body, node_id=f"{i:032x}",
        )
        return EvidenceHit(node=node, score=1., video_id="v1", clip_id=i)

    class Graph:
        def search_by_embedding(self, **kwargs):
            return [entry(i) for i in range(2, 18, 2)]

        def get_evidence_by_clips(self, clips, *args, **kwargs):
            assert len(clips) == 1  # ten slots minus eight main hits and one boundary
            return [entry(i) for i in clips]

        def get_boundary_transcript(self, *args, **kwargs):
            return [entry(99, boundary=True)]

    monkeypatch.setattr("m3.transcript_rerank.rerank_with_transcript_bm25",
                        lambda query, hits, **kwargs: hits)
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.]]
    result = Search("scissors", graph=Graph(), embedder=embedder, ctx=context(),
                    search_config=SearchConfig(
                        video_id="v1", before_sec=100, context_k=8,
                        context_budget=ContextBudget(max_result_nodes=10),
                    ))
    assert sum(len(result[g]) for g in ("hits", "context_hits", "boundary_hits")) == 10
    assert len(result["boundary_hits"]) == 1
    assert result["boundary_hits"][0]["description"] == ""
    assert result["retrieved_before_result_limit"] == 10


@pytest.mark.parametrize("context_k,cap,expected", [(1, 2, 5), (2000, 3, 5), (2000, 1, 0)])
def test_neighbor_requests_are_bounded_before_graph_read(monkeypatch, context_k, cap, expected):
    from types import SimpleNamespace
    from m3.context_retrieval import neighbor_plan

    graph = Mock()
    graph.search_by_embedding.return_value = [SimpleNamespace(
        node=SimpleNamespace(node_id="parent", extracted_content={"description": "scissors"}),
        score=1., video_id="v1", clip_id=5000,
    )]
    graph.get_evidence_by_clips.return_value = []
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.]]
    planner = Mock(wraps=neighbor_plan)
    monkeypatch.setattr("m3.control_adapter.neighbor_plan", planner)
    monkeypatch.setattr("m3.transcript_rerank.rerank_with_transcript_bm25",
                        lambda query, hits, **kwargs: hits)
    Search("scissors", graph=graph, embedder=embedder, ctx=context(),
           search_config=SearchConfig(
               video_id="v1", include_neighbors=2000, context_k=context_k,
               context_budget=ContextBudget(max_result_nodes=cap, max_neighbor_clips=5)))
    if expected:
        assert planner.call_args.kwargs["radius"] == 2000
        assert planner.call_args.kwargs["max_clips"] == expected
        batches = [call.args[0] for call in graph.get_evidence_by_clips.call_args_list]
        clips = [clip for batch in batches for clip in batch]
        assert len(clips) == len(set(clips)) == expected
        assert all(len(batch) <= min(8, context_k, cap - 1) for batch in batches)
    else:
        planner.assert_not_called()
        graph.get_evidence_by_clips.assert_not_called()


@pytest.mark.parametrize("radius,available,read_cap,expected_requests,expected_hits", [
    (1, [11], 64, [9, 11], [11]),  # review repro: previous clip has no index entry
    (4, [14], 64, [9, 11, 8, 12, 7, 13, 6, 14], [14]),  # beyond result cap
    (2000, [], 5, [9, 11, 8, 12, 7], []),  # read budget, not radius, stops empty reads
    (4, [14], 5, [9, 11, 8, 12, 7], []),  # explicit read-budget exhaustion
    (4, [9, 11, 14], 64, [9], [9]),  # stop as soon as result quota is satisfied
])
def test_sparse_neighbors_continue_until_evidence_or_read_budget(
    monkeypatch, radius, available, read_cap, expected_requests, expected_hits,
):
    from types import SimpleNamespace

    def entry(clip):
        return SimpleNamespace(
            node=SimpleNamespace(node_id=f"n{clip}", extracted_content={"description": "scissors"}),
            score=1., video_id="v1", clip_id=clip,
        )

    graph = Mock()
    graph.search_by_embedding.return_value = [entry(10)]
    graph.get_evidence_by_clips.side_effect = lambda clips, *args, **kwargs: [
        entry(clip) for clip in clips if clip in available
    ]
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.]]
    monkeypatch.setattr("m3.transcript_rerank.rerank_with_transcript_bm25",
                        lambda query, hits, **kwargs: hits)
    result = Search("scissors", graph=graph, embedder=embedder, ctx=context(),
                    search_config=SearchConfig(
                        video_id="v1", include_neighbors=radius, context_k=1,
                        context_budget=ContextBudget(max_result_nodes=3, max_neighbor_clips=read_cap)))
    requests = [clip for call in graph.get_evidence_by_clips.call_args_list for clip in call.args[0]]
    assert requests == expected_requests
    assert len(requests) <= read_cap
    assert [h["clip_id"] for h in result["context_hits"]] == expected_hits
    assert all(h["context_for"] == "n10" and h["context_distance"] == abs(h["clip_id"] - 10)
               for h in result["context_hits"])
    assert result["retrieval_branches"][-1] == {"kind": "exact_neighbor", "clip_ids": requests}


def test_neighbor_batches_count_only_unique_in_scope_evidence(monkeypatch):
    from types import SimpleNamespace

    def entry(clip, node=None, video="v1"):
        return SimpleNamespace(
            node=SimpleNamespace(node_id=node or f"n{clip}", extracted_content={"description": "scissors"}),
            score=1., video_id=video, clip_id=clip,
        )

    graph = Mock()
    graph.search_by_embedding.return_value = [entry(10), entry(20)]
    graph.get_evidence_by_clips.side_effect = [
        [entry(9, video="other"), entry(9, node="n10"), entry(19), entry(11), entry(11)],
        [entry(19), entry(19)],
    ]
    embedder = Mock()
    embedder.embed_texts.return_value = [[1.]]
    monkeypatch.setattr("m3.transcript_rerank.rerank_with_transcript_bm25",
                        lambda query, hits, **kwargs: hits)
    result = Search("scissors", graph=graph, embedder=embedder, ctx=context(),
                    search_config=SearchConfig(
                        video_id="v1", include_neighbors=1, context_k=2,
                        context_budget=ContextBudget(max_result_nodes=4, max_neighbor_clips=4)))
    assert [call.args[0] for call in graph.get_evidence_by_clips.call_args_list] == [[9, 11], [19]]
    assert [(h["clip_id"], h["context_for"], h["context_distance"]) for h in result["context_hits"]] == [
        (11, "n10", 1), (19, "n20", 1)]


def test_shared_neighbors_keep_each_parent_relation_without_new_content_versions():
    original = messages(exchange("a", "scissors", [hit("parent1"), hit("parent2")]),
                        exchange("b", "toolbox", []))
    for index, parents in ((3, [("parent1", 1), ("parent2", 2)]),
                           (5, [("parent2", 3)])):
        payload = json.loads(original[index]["content"])
        payload["context_hits"] = [
            hit("neighbor", context_for=parent, context_distance=distance)
            for parent, distance in parents
        ]
        original[index]["content"] = json.dumps(payload)
    before = deepcopy(original)
    view, stats = pack_context(original, [], "scissors", ContextBudget())
    records, docs = evidence(view)
    refs = [ref for record in records for ref in record["evidence_refs"]
            if ref["role"] == "context_hits"]
    assert [(r["context_for"], r["context_distance"]) for r in refs] == [
        ("parent1", 1), ("parent2", 2), ("parent2", 3)]
    assert len({r["id"] for r in refs}) == 1
    assert stats["unique_evidence_versions"] == len(docs) == 3
    assert original == before


def test_bounded_neighbor_plan_preserves_priority_and_shared_neighbor_parent():
    from m3.context_retrieval import neighbor_plan

    seeds = [("v1", 10, "parent1"), ("v1", 12, "parent2"), ("v1", 0, "parent3")]
    full = neighbor_plan(seeds, radius=5, before_clip=13)
    for count in range(1, len(full) + 1):
        bounded = neighbor_plan(seeds, radius=5, before_clip=13, max_clips=count)
        assert list(bounded.items()) == list(full.items())[:count]
    assert full[("v1", 11)] == (1, 0, "parent1")


def test_bounded_neighbor_plan_skips_huge_invalid_ranges():
    from m3.context_retrieval import neighbor_plan

    assert neighbor_plan([("v1", "bad", "n")], radius=10**12, max_clips=5) == {}
    plan = neighbor_plan([("v1", 10**12, "n")], radius=10**12,
                         before_clip=2, max_clips=2)
    assert list(plan) == [("v1", 2), ("v1", 1)]


def test_oversized_query_is_rejected_before_retrieval(monkeypatch):
    from core.react_loop import ReactLoop
    dispatch = Mock()
    monkeypatch.setattr(ReactLoop, "_execute_tool", dispatch)
    adapter = OGMemoryControlAdapter(MockLLM(), Mock(), Mock(),
                                     context_budget=ContextBudget(max_query_bytes=10))
    adapter._build_messages(context(), M3TaskContext(question="where"))
    assert adapter._execute_tool("Search", {"query": "剪刀" * 5}, context())["error"]
    dispatch.assert_not_called()


def test_search_call_limit_includes_cached_and_invalid_attempts(monkeypatch):
    from core.react_loop import ReactLoop
    dispatch = Mock(return_value={"hits": [], "context_hits": [], "boundary_hits": []})
    monkeypatch.setattr(ReactLoop, "_execute_tool", dispatch)
    adapter = OGMemoryControlAdapter(MockLLM(), Mock(), Mock(),
                                     context_budget=ContextBudget(max_search_calls=2, max_top_k=8))
    ctx = context()
    task = M3TaskContext(question="where")
    adapter._build_messages(ctx, task)
    adapter._execute_tool("Search", {"query": "scissors", "top_k": 1000000}, ctx)
    assert dispatch.call_args.args[1]["top_k"] == 8
    adapter._execute_tool("Search", {"query": "scissors"}, ctx)  # cached
    assert adapter._execute_tool("Search", {"query": "glue"}, ctx)["error"]
    assert dispatch.call_count == 1
    assert task.search_stats["stop_reason"] == "search_call_budget"
    adapter._build_messages(ctx, task)  # new question resets cache and budget
    adapter._execute_tool("Search", {"query": "scissors"}, ctx)
    assert dispatch.call_count == 2


def test_adapter_does_not_send_an_over_budget_request():
    llm = Mock()
    adapter = OGMemoryControlAdapter(llm, Mock(), Mock(),
                                     context_budget=ContextBudget(max_input_tokens=100))
    result = adapter.run(context(), M3TaskContext(question="where"))
    assert result.finish_reason == "llm_error"
    llm.complete_with_tools_messages.assert_not_called()
