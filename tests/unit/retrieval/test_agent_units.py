"""Agent units retain their identity from indexing through public consumption."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext, RetrievalConfig, SeedHit, SeedResult
from extraction.agent_memory import render_agent_memory
from index.index_record_builder import build_index_records
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.evidence_selector import hydrate_evidence, select_evidence
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.mcp_tools import query_memory, read_memory
from service.api import ReadAPI
from session.topic_buffer import SessionTopicBuffer


class SameEmbedding:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def unit_memory():
    ctx = RequestContext(account_id="acct", user_id="user", agent_id="agent",
                         session_id="session", trace_id="trace")
    uri = "ctx://acct/agents/agent/memories/tool_units/search"
    units = [
        {"key": "batch", "title": "Search batch limit", "content": "Use batches of 20 requests.",
         "grounding": [{"uri": "trace://session/event-1"}]},
        {"key": "cursor", "title": "Search pagination", "content": "Reuse the returned cursor for the next page.",
         "grounding": [{"uri": "trace://session/event-2"}]},
    ]
    fields = {"tool_name": "search", "units": units}
    node = ContextNode(
        uri, "MEMORY", "tool", 4, "agent:agent", "Search limits and pagination", "Batch limits; cursor pagination.",
        render_agent_memory("tool", fields),
        {"agent_memory_version": 1, "agent_memory_content_version": 3,
         "routing_key": "search", "structured_fields": fields,
         "agent_memory": {"units": {item["key"]: {"revision": i + 1, "grounding": item["grounding"]}
                                    for i, item in enumerate(units)}}},
    )
    user = ContextNode("ctx://acct/users/user/memories/preferences/research", "MEMORY",
                       "preference", 2, "user:user", "Research preference", "",
                       "Prefer primary sources for research.")
    experience_fields = {"key": "rollback", "title": "Preserve the previous page",
                         "content": "Keep the previous cursor until the current page is processed.",
                         "grounding": [{"uri": "trace://session/event-3"}]}
    experience = ContextNode(
        "ctx://acct/agents/agent/memories/experiences/rollback", "MEMORY", "experience", 4,
        "agent:agent", experience_fields["title"], "", render_agent_memory("experience", experience_fields),
        {"agent_memory_version": 1, "agent_memory_content_version": 3, "routing_key": "rollback",
         "structured_fields": experience_fields,
         "agent_memory": {"units": {"rollback": {"revision": 1}}}},
    )
    nodes = {node.uri: node, user.uri: user, experience.uri: experience}
    fs = SimpleNamespace(read_node=lambda uri, ctx: nodes[uri], nodes=nodes)
    return ctx, node, user, fs


@pytest.mark.parametrize("expand_hierarchy", [False, True])
def test_mixed_query_retains_each_tool_unit_and_user_memory(unit_memory, expand_hierarchy):
    ctx, node, user, fs = unit_memory
    cfg = RetrievalConfig(include_agent_memories=True, expand_hierarchy=expand_hierarchy,
                          hotness_alpha=0, default_score_threshold=0,
                          owner_scope_filter_enabled=True)
    records = [record for item in fs.nodes.values() for record in build_index_records(item)]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index = InMemoryVectorIndex(dimension=2)
    index.upsert(records)
    reader = ContextReader(fs)
    pipeline = RetrievalPipeline(QueryPlanner(cfg), SeedRetriever(index, SameEmbedding(), cfg),
                                 HierarchicalSearcher(index, cfg), ResultRanker(cfg), cfg,
                                 context_reader=reader)
    api = ReadAPI(pipeline, reader, cfg)
    service = SimpleNamespace(_cfg=SimpleNamespace(), get_read_api=lambda: api,
                              _get_context_fs=lambda: fs)

    response = query_memory(service, query="search research memory", request_context=ctx)

    assert {(hit["type"], hit.get("unit_key")) for hit in response["hits"]} == {
        ("tool_evidence", "batch"), ("tool_evidence", "cursor"),
        ("experience", "rollback"), ("preference", None)}
    for hit in response["hits"]:
        if hit["type"] == "tool_evidence":
            selected = next(unit for unit in node.metadata["structured_fields"]["units"]
                            if unit["key"] == hit["unit_key"])
            expected = render_agent_memory("tool", {"tool_name": "search", "units": [selected]})
            assert hit["content_excerpt"] == expected
            assert hit["grounding"] == selected["grounding"]
            detail = read_memory(service, uri=hit["uri"], unit_key=hit["unit_key"],
                                 level="overview", request_context=ctx)
            assert detail["content"] == expected
            assert detail["overview"] == ""
            assert detail["level"] == "content"
            assert detail["unit_revision"] == hit["unit_revision"]
            assert detail["grounding"] == selected["grounding"]
        elif hit["type"] == "experience":
            detail = read_memory(service, uri=hit["uri"], unit_key=hit["unit_key"], request_context=ctx)
            experience = fs.nodes["ctx://acct/agents/agent/memories/experiences/rollback"]
            assert detail["content"] == experience.content
            assert detail["grounding"] == experience.metadata["structured_fields"]["grounding"]


def test_fusion_and_seed_merge_preserve_distinct_units(unit_memory):
    _, node, _, _ = unit_memory
    uri = node.uri + "/content.md"
    hits = [SeedHit(uri, score, metadata={"unit_key": key, "_bm25_raw_score": 2})
            for key, score in (("batch", 0.9), ("cursor", 0.8))]
    paths = [SeedRetriever._fuse_results(hits, list(reversed(hits))),
             SeedRetriever._weighted_rrf(hits, list(reversed(hits)),
                                         semantic_weight=1, lexical_weight=1, rrf_k=60),
             RetrievalPipeline._merge_seeds([SeedResult(initial_candidates=hits),
                                            SeedResult(initial_candidates=hits)]).initial_candidates]
    for path in paths:
        assert [hit.metadata["unit_key"] for hit in path] == ["batch", "cursor"]


def test_mixed_level_fusion_preserves_user_l2(unit_memory):
    _, _, user, _ = unit_memory
    hits = [SeedHit(user.uri, 0.8, level=0), SeedHit(user.uri + "/content.md", 0.9, level=2)]
    fused = SeedRetriever._fuse_results(hits, list(reversed(hits)))
    assert [(hit.uri, hit.level) for hit in fused] == [(user.uri + "/content.md", 2), (user.uri, 0)]


def test_unit_hydration_and_delivery_confirm_only_rendered_unit(unit_memory):
    ctx, node, _, fs = unit_memory
    uri = node.uri
    hits = [SeedHit(uri + "/content.md", 0.9, category="tool", metadata={"unit_key": key})
            for key in ("batch", "cursor")]
    items = [{"uri": hit.uri, "category": "tool", "metadata": hit.metadata} for hit in hits]
    reader = ContextReader(fs)
    hydrated, stats = hydrate_evidence(items + items, reader.read, ctx)
    assert stats["read_count"] == 2
    assert [item["metadata"]["unit_key"] for item in hydrated] == ["batch", "cursor"]
    buffer = SessionTopicBuffer("session")
    buffer.set_pending_injection(hits)
    snapshot = buffer.injection_snapshot()
    text, delivery = select_evidence(hydrated[:1], "batch limit", 2000)
    assert "unit_key: batch" in text
    assert "Use batches of 20 requests." in text
    assert delivery["delivered_units"] == [{"uri": uri, "unit_key": "batch"}]
    buffer.consume_injection(snapshot, delivery["delivered_uris"], delivered_units=delivery["delivered_units"])
    assert buffer.was_injected(uri, {"unit_key": "batch"}) is True
    assert [hit.metadata["unit_key"] for hit in buffer.get_pending_injection()] == ["cursor"]
    text, delivery = select_evidence(hydrated[1:], "pagination", 2000)
    buffer.consume_injection(buffer.injection_snapshot(), delivery["delivered_uris"],
                             delivered_units=delivery["delivered_units"])
    assert buffer.was_injected(uri, {"unit_key": "cursor"}) is True
    assert buffer.get_pending_injection() is None


def test_short_unit_indexes_l0_and_canonical_l2(unit_memory):
    _, node, _, _ = unit_memory
    fields = {"tool_name": "search", "units": node.metadata["structured_fields"]["units"][:1]}
    node = replace(node, overview="", content=render_agent_memory("tool", fields),
                   metadata={**node.metadata, "structured_fields": fields})
    records = build_index_records(node)
    assert [record.level for record in records] == [0, 2]
    assert [(record.metadata["unit_key"], record.metadata["unit_revision"])
            for record in records if record.level == 2] == [("batch", 1)]
    assert {(record.filters["account_id"], record.filters["owner_space"]) for record in records} == {
        ("acct", "agent:agent")}


def test_legacy_tool_reads_canonical_units_with_original_source(unit_memory):
    ctx, node, _, fs = unit_memory
    legacy_fields = {"title": "Search batching", "scope": "Tool: search",
                     "items": [{"key": "batch", "statement": "Use batches of 20 requests."}]}
    fs.nodes[node.uri] = replace(node, metadata={**node.metadata, "structured_fields": legacy_fields},
                                 content="Legacy rendered body.")
    block = ContextReader(fs).read(node.uri, ctx)
    assert block.metadata["structured_fields"] == {
        "tool_name": "search", "units": [{"key": "batch", "title": "Search batching",
            "content": "Use batches of 20 requests.", "grounding": [{"uri": "trace://session/event-1"}]}]}
    assert block.content_excerpt == "search\n\nSearch batching\n\nUse batches of 20 requests."
    service = SimpleNamespace(_cfg=SimpleNamespace(), _get_context_fs=lambda: fs)
    detail = read_memory(service, uri="ogmemory://memories/tool_evidence/search", request_context=ctx)
    assert detail["content"] == block.content_excerpt
    assert detail["grounding"] == [{"uri": "trace://session/event-1"}]


def test_chroma_unit_identity_and_grounding_round_trip(unit_memory, tmp_path):
    pytest.importorskip("chromadb")
    from providers.vector_index.chroma_index import ChromaVectorIndex

    ctx, node, _, _ = unit_memory
    index = ChromaVectorIndex(collection_name="agent_units", persist_directory=str(tmp_path), dimension=2)
    records = [record for record in build_index_records(node) if record.level == 2]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index.upsert(records)
    hits = index.search_by_vector([1.0, 0.0],
                                  {"account_id": ctx.account_id, "owner_space": "agent:agent"}, top_k=5)
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in hits} == {
        unit["key"]: (i + 1, unit["grounding"])
        for i, unit in enumerate(node.metadata["structured_fields"]["units"])}


def test_chroma_adapter_and_bm25_reload_preserve_unit_metadata(unit_memory, monkeypatch):
    from providers.vector_index.chroma_index import ChromaVectorIndex
    from retrieval.bm25_index import Bm25Index
    from tests.unit.retrieval.test_bm25 import _mock_tokenize

    monkeypatch.setattr("retrieval.bm25_index._tokenize", _mock_tokenize)
    ctx, node, _, _ = unit_memory
    records = [record for record in build_index_records(node) if record.level == 2]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._collection = Mock()
    index._client = SimpleNamespace(get_max_batch_size=lambda: 100)
    index._locked = nullcontext
    index.upsert(records)
    saved = index._collection.upsert.call_args.kwargs
    assert all(isinstance(value, (str, int, float, bool))
               for metadata in saved["metadatas"] for value in metadata.values())
    index._collection.count.return_value = 2
    index._collection.query.return_value = {
        "ids": [saved["ids"]], "metadatas": [saved["metadatas"]],
        "documents": [saved["documents"]], "distances": [[0, 0]],
    }
    hits = index.search_by_vector([1, 0], {"account_id": ctx.account_id}, top_k=2)
    expected = {unit["key"]: (i + 1, unit["grounding"])
                for i, unit in enumerate(node.metadata["structured_fields"]["units"])}
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in hits} == expected
    index._collection.count.return_value = 3
    index._collection.get.return_value = {
        "ids": saved["ids"] + ["preference"],
        "documents": saved["documents"] + ["Prefer primary sources."],
        "metadatas": saved["metadatas"] + [{"category": "preference"}],
    }
    bm25 = Bm25Index()
    bm25.build_from_chroma(index._collection)
    lexical_hits = bm25.search("batch cursor", top_k=2)
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in lexical_hits} == expected
