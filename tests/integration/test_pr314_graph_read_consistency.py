"""Real SQL edge-snapshot races and context-type boundaries for PR314."""
from unittest.mock import Mock

import pytest

from core.models import ContextNode, LeafHit, RelationEdge, RetrievalConfig, SeedResult, TypedQuery
from retrieval.context_reader import ContextReader
from retrieval.graph_retriever import OneHopRetriever
from retrieval.graph_supplement import GraphSupplement
from retrieval.pipeline import RetrievalPipeline
from retrieval.result_ranker import ResultRanker
from service.api import ReadAPI
from service.graph_service import GraphService
from tests.integration.test_semantic_context_graph import graph_db


def read_api(graph_db, *, before_assembly=None):
    ctx, fs, index, store, _, _, worker, nodes, profile = graph_db
    reader = ContextReader(fs, storage_backend="sql")
    graph = GraphService(None, relation_reader=store)
    cfg = RetrievalConfig(enable_graph_expansion=True, expand_hierarchy=False)
    assembly = ResultRanker(cfg)
    if before_assembly:
        original = assembly.assemble

        def assemble(*args, **kwargs):
            before_assembly()
            return original(*args, **kwargs)

        assembly.assemble = assemble
    seed = Mock()
    seed.embedder = worker._embedder
    pipe = RetrievalPipeline(Mock(), seed, None, assembly, cfg, context_reader=reader,
        graph_retriever=OneHopRetriever(graph, embedder=worker._embedder, embedding_profile=profile),
        graph_supplement=GraphSupplement(graph, reader, index, worker._embedder, profile))
    pipe._run_planner = Mock(return_value=[TypedQuery("fact", "MEMORY", [], top_k=5)])
    pipe._run_seed = Mock(return_value=SeedResult(query_vector=[1., 0.]))
    pipe._run_expand = Mock(return_value=[LeafHit(nodes[0].uri, 1., category="entity")])
    return ReadAPI(pipe, read_service=reader, config=cfg)


def update(fs, uri, ctx):
    current = fs.read_node(uri, ctx)
    current.content = "New unrelated revision-two body"
    current.metadata["_expected_version"] = current.metadata["version"]
    fs.write_node(current, ctx)


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_sql_projection_is_discarded_when_endpoint_changes_after_query(graph_db, monkeypatch, endpoint):
    ctx, fs, _, store, _, outbox, worker, nodes, _ = graph_db
    a, b, _ = nodes
    store.upsert_edges([RelationEdge(a.uri, b.uri, "supports", 1., "Old version-one reason")], ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    original = store.get_relation_neighbors
    mutated = False

    def query_then_write(*args, **kwargs):
        nonlocal mutated
        rows = original(*args, **kwargs)
        if rows and not mutated:
            mutated = True
            assert rows[0].source_revision == rows[0].target_revision == 1
            update(fs, a.uri if endpoint == "source" else b.uri, ctx)
            # The update transaction has already invalidated the old edge.
            assert original(a.uri, ctx) == []
        return rows

    monkeypatch.setattr(store, "get_relation_neighbors", query_then_write)
    result = read_api(graph_db).search_memory("fact", ctx, context_type="MEMORY")
    assert mutated
    assert [hit.uri for hit in result.hits] == [a.uri]


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_sql_projection_is_rechecked_after_assembly_read(graph_db, endpoint):
    ctx, fs, _, store, _, outbox, worker, nodes, _ = graph_db
    a, b, _ = nodes
    store.upsert_edges([RelationEdge(a.uri, b.uri, "supports", 1., "Old version-one reason")], ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    api = read_api(graph_db, before_assembly=lambda: update(fs, a.uri if endpoint == "source" else b.uri, ctx))
    result = api.search_memory("fact", ctx, context_type="MEMORY", fill_content_for_top_k=5)
    assert b.uri not in [hit.uri for hit in result.hits]


def test_sql_memory_scope_excludes_skill_target_and_bridge(graph_db):
    ctx, fs, _, store, _, outbox, worker, nodes, _ = graph_db
    a, _, c = nodes
    skill = ContextNode(uri=f"ctx://{ctx.account_id}/agents/agent/skills/tool", context_type="SKILL",
        category="tool", level=2, owner_space="agent:agent", abstract="Tool", overview="Tool overview",
        content="Tool body", metadata={"_expected_version": 0})
    fs.write_node(skill, ctx)
    store.upsert_edges([RelationEdge(a.uri, skill.uri, "uses", 1., "Uses a tool"),
                        RelationEdge(skill.uri, c.uri, "supports", 1., "Tool supports fact")], ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    assert store.get_relation_neighbors(a.uri, ctx)[0].uri == skill.uri
    api = read_api(graph_db)
    result = api.search_memory("fact", ctx, context_type="MEMORY", include_debug=True,
                               collect_graph_supplement=True)
    assert [hit.uri for hit in result.hits] == [a.uri]
    assert result.trace.graph_expansion["two_hop_supplement"]["selected"] == []
    assert skill.uri not in result.trace.graph_expansion["two_hop_supplement"]["expanded_nodes"]
    # The underlying service remains able to traverse cross-type relations.
    unfiltered = api._pipeline._graph_supplement.select("fact", [LeafHit(a.uri, 1.)],
        [LeafHit(a.uri, 1.)], ctx, query_vector=[1., 0.])
    assert c.uri in [row["uri"] for row in unfiltered["selected"]]
