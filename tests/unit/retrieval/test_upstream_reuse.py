"""Reuse changes must preserve graph evidence, ACLs and upstream contracts."""
import asyncio
import math
import random
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import (RequestContext, SeedHit, LeafHit, RetrievalConfig,
                         RetrievedBlock, SeedResult, TypedQuery)
from core.vector_math import cosine_similarity
from retrieval.context_reader import ContextReader
from retrieval.graph_retriever import OneHopRetriever
from retrieval.graph_selection import cosine
from retrieval.pipeline import RetrievalPipeline
from retrieval.trace import RetrievalTrace
from server.memory_service import MemoryService
from service.graph_service import GraphService

CTX = RequestContext("acct", "alice", "agent", "session", "trace")
ROOT = "ctx://acct/users/alice/memories/entities/"
PROFILE = dict(profile_id="test", model="test", dimension=2,
               representation_version="relation-target-v2")


def retriever():
    store, embedder = Mock(), Mock()
    embedder.embed_texts.return_value = [[1., 0.]]
    store.get_relation_neighbors.return_value = [dict(uri=ROOT+"b", weight=.8,
        relation_type="related", source_revision=1, target_revision=1,
        edge_projection=dict(vector=[1., 0.], embedding_profile="test", model="test",
            dimension=2, representation_version="relation-target-v2",
            source_revision=1, target_revision=1))]
    return OneHopRetriever(GraphService(None, relation_reader=store),
                           embedder=embedder, embedding_profile=PROFILE), embedder


def test_reused_query_matches_independent_embedding_exactly():
    graph, embedder = retriever()
    args = dict(query="facts", ctx=CTX, seed_hits=[SeedHit(ROOT+"a", .9)])
    before = asyncio.run(graph.search_with_report(**args))
    embedder.embed_texts.reset_mock()
    vector = [1., 0.]
    after = asyncio.run(graph.search_with_report(**args, query_vector=vector))
    assert after == before
    assert vector == [1., 0.]
    embedder.embed_texts.assert_not_called()


@pytest.mark.parametrize("vector", [[], [float("nan"), 0.], [float("inf"), 0.]])
def test_invalid_reused_vector_preserves_anchor(vector):
    graph, embedder = retriever()
    hits, report = asyncio.run(graph.search_with_report(query="facts", ctx=CTX,
        seed_hits=[SeedHit(ROOT+"a", .9)], query_vector=vector))
    assert [hit.uri for hit in hits] == [ROOT+"a"]
    assert report.outcome == "query_embedding_failure"
    embedder.embed_texts.assert_not_called()


@pytest.mark.parametrize("same_embedder", [False, True])
def test_pipeline_reuses_only_same_embedder(same_embedder):
    graph, embedder = retriever()
    reader = Mock()
    reader.read_authorized.side_effect = lambda uri, ctx: RetrievedBlock(
        uri=uri, context_type="MEMORY", metadata={"version": 1},
        overview="b", content_excerpt="content")
    pipeline = RetrievalPipeline(Mock(), SimpleNamespace(
        embedder=embedder if same_embedder else Mock()), None, Mock(),
        RetrievalConfig(enable_graph_expansion=True), graph_retriever=graph,
        context_reader=reader)
    result = pipeline._expand_by_graph("facts", [LeafHit(ROOT+"a", .9)], CTX,
                                     RetrievalTrace(), query_vector=[1., 0.])
    assert ROOT+"b" in [hit.uri for hit in result]
    assert embedder.embed_texts.call_count == (0 if same_embedder else 1)


@pytest.mark.parametrize("seed_text,enabled,embedding_calls", [
    ("facts", True, 0), ("rewritten facts", True, 1), ("facts", False, 0),
])
def test_full_pipeline_query_matching_and_graph_toggle(seed_text, enabled, embedding_calls):
    graph, embedder = retriever()
    reader = Mock()
    reader.read_authorized.side_effect = lambda uri, ctx: RetrievedBlock(
        uri=uri, context_type="MEMORY", metadata={"version": 1}, overview="b")
    pipeline = RetrievalPipeline(Mock(), SimpleNamespace(embedder=embedder), None, Mock(),
        RetrievalConfig(enable_graph_expansion=enabled), graph_retriever=graph,
        context_reader=reader)
    pipeline._run_planner = Mock(return_value=[TypedQuery(text=seed_text, context_type="MEMORY", categories=[])])
    pipeline._run_seed = Mock(return_value=SeedResult(query_vector=[1., 0.]))
    pipeline._run_expand = Mock(return_value=[LeafHit(ROOT+"a", .9)])
    pipeline._run_assembly = Mock(return_value=[])
    pipeline.run("facts", CTX)
    assert embedder.embed_texts.call_count == embedding_calls
    leaves = pipeline._run_assembly.call_args.args[1]
    assert [hit.uri for hit in leaves] == ([ROOT+"a", ROOT+"b"] if enabled else [ROOT+"a"])


def test_third_party_retriever_signature_unchanged():
    class ExternalRetriever:
        async def search(self, query, ctx, seed_hits, top_k):
            self.called = True
            return []
    graph = ExternalRetriever()
    pipeline = RetrievalPipeline(Mock(), Mock(), None, Mock(), graph_retriever=graph)
    pipeline._expand_by_graph("facts", [LeafHit(ROOT+"a", .9)], CTX,
                              RetrievalTrace(), query_vector=[1., 0.])
    assert graph.called


def test_cosine_exact_parity_with_prior_arithmetic():
    rng = random.Random(42)
    for _ in range(100):
        a, b = ([rng.uniform(-1, 1) for _ in range(8)] for _ in range(2))
        dot = sum(x*y for x, y in zip(a, b))
        sa, sb = sum(x*x for x in a), sum(x*x for x in b)
        expected = dot/(math.sqrt(sa)*math.sqrt(sb))
        assert cosine_similarity(a, b) == expected
        assert OneHopRetriever._cosine(a, b) == expected
        assert cosine(a, b) == dot/math.sqrt(sa*sb)
    assert cosine_similarity([0., 0.], [1., 0.]) == 0.
    assert cosine_similarity([1.], [1., 0.]) == 0.
    with pytest.raises(ValueError):
        cosine([1.], [1., 0.])
    with pytest.raises(ValueError):
        cosine([float("nan")], [1.])


def test_shared_hydration_keeps_soft_read_and_strict_metadata():
    fs = Mock()
    fs.read_node.return_value = SimpleNamespace(category="entity", owner_space="user:alice",
        abstract="abstract", overview="overview", content="content", metadata={"version": 2})
    reader = ContextReader(fs, storage_backend="sql")
    soft = reader.read(ROOT+"a/content.md", CTX)
    strict = reader.read_authorized(ROOT+"a/content.md", CTX)
    assert (soft.category, soft.overview, soft.content_excerpt) == (
        strict.category, strict.overview, strict.content_excerpt)
    assert soft.abstract == strict.abstract and soft.metadata == strict.metadata
    assert strict.abstract == "abstract" and strict.metadata == {"version": 2}
    assert fs.read_node.call_args.args[0] == ROOT+"a"
    with pytest.raises(AccessDeniedError):
        reader.read_authorized("ctx://other/users/alice/memories/a", CTX)
    fs.read_node.side_effect = NodeNotFoundError(ROOT+"a")
    with pytest.raises(NodeNotFoundError):
        reader.read(ROOT+"a", CTX)
    with pytest.raises(NodeNotFoundError):
        reader.read_authorized(ROOT+"a", CTX)


@pytest.mark.parametrize("same_dsn,dialect,backend,provider,share", [
    (True, "postgresql", "sql", "opengauss", True),
    (True, "opengauss", "sql", "opengauss", True),
    (False, "postgresql", "sql", "opengauss", False),
    (True, "mysql", "sql", "opengauss", False),
    (True, "postgresql", "agfs", "opengauss", False),
    (True, "postgresql", "sql", "memory", False),
])
def test_pool_reuse_is_conditional(same_dsn, dialect, backend, provider, share):
    service = MemoryService.__new__(MemoryService)
    service._init_lock = threading.RLock()
    service._vector_index = None
    service._cfg = SimpleNamespace(storage_backend=backend, sql_connection_string="host=storage dbname=memory",
                                  sql_dialect=dialect, retrieval_hybrid_enabled=False)
    service._provider_cfg = Mock(vector_db_type=provider,
        opengauss_connection_string="host=storage dbname=memory" if same_dsn else "host=other dbname=memory")
    pool = object()
    service._get_shared_sql_pool = Mock(return_value=pool)
    index = service._get_shared_vector_index()
    assert service._get_shared_vector_index() is index
    if share:
        service._provider_cfg.create_vector_index.assert_called_once_with(pool=pool)
        service._get_shared_sql_pool.assert_called_once()
    else:
        service._provider_cfg.create_vector_index.assert_called_once_with()
        service._get_shared_sql_pool.assert_not_called()
