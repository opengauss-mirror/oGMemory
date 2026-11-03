"""Public type filters retrieve the archived node's actual context type."""

from types import SimpleNamespace

import pytest

from commit.archive_builder import ArchiveBuilder
from core.models import CandidateMemory, RetrievalConfig, WritePlan
from index.index_record_builder import build_index_records
from providers.llm.mock_llm import MockLLM
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.mcp_tools import query_memory
from service.api import ReadAPI
from tests.unit.retrieval.test_agent_units import SameEmbedding
from tests.unit.retrieval.test_agent_units import unit_memory as unit_memory


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(("query", "types", "expected"), [
    ("rollback policy", ["skill"], {"skill"}),
    ("rollback policy", ["preference", "skill"], {"preference", "skill"}),
    ("how to use read_file", ["tool_memory"], {"tool_memory"}),
    ("rollback policy", ["tool_evidence"], {"tool_evidence"}),
    ("rollback policy", ["tool_memory", "tool_evidence"], {"tool_memory", "tool_evidence"}),
    ("rollback policy", ["experience", "skill"], {"experience", "skill"}),
    ("rollback policy", ["pattern"], {"pattern"}),
])
def test_public_category_recall(unit_memory, enabled, expand, query, types, expected):
    ctx, _, _, fs = unit_memory
    builder = ArchiveBuilder(MockLLM())
    for category, directory in [("skill", "skills/generated"), ("tool", "memories/tools"),
                                ("pattern", "memories/patterns")]:
        candidate = CandidateMemory(
            category=category, owner_scope="agent", routing_key="rollback",
            abstract="Rollback policy", overview="Preserve rollback state.",
            content="Preserve rollback state before using read_file.", confidence=0.9,
        )
        node = builder.build(candidate, WritePlan(
            "create", f"ctx://acct/agents/agent/{directory}/rollback"), ctx)
        fs.nodes[node.uri] = node
    cfg = RetrievalConfig(include_agent_memories=enabled, expand_hierarchy=expand,
                          hotness_alpha=0, default_score_threshold=0,
                          owner_scope_filter_enabled=True)
    index = InMemoryVectorIndex(dimension=2)
    records = [record for node in fs.nodes.values() for record in build_index_records(node)]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index.upsert(records)
    reader = ContextReader(fs)
    pipeline = RetrievalPipeline(
        QueryPlanner(cfg), SeedRetriever(index, SameEmbedding(), cfg),
        HierarchicalSearcher(index, cfg), ResultRanker(cfg), cfg, context_reader=reader,
    )
    api = ReadAPI(pipeline, reader, cfg)
    service = SimpleNamespace(_cfg=SimpleNamespace(), get_read_api=lambda: api,
                              _get_context_fs=lambda: fs)

    response = query_memory(service, query=query, types=types, request_context=ctx)

    assert {hit["type"] for hit in response["hits"]} == expected
    assert all(hit["content_excerpt"] for hit in response["hits"])
