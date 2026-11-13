"""Canonical archive checks protect graph retrieval during index-cleanup lag."""
from unittest.mock import Mock

from core.models import LeafHit, RelationEdge, RetrievalConfig
from retrieval.context_reader import ContextReader
from retrieval.graph_retriever import OneHopRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.trace import RetrievalTrace
from service.graph_service import GraphService
from tests.integration.test_semantic_context_graph import graph_db


def test_archived_anchor_with_unchanged_revision_cannot_expand(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    a, b, _ = nodes
    store.upsert_edges([RelationEdge(a.uri,b.uri,"supports",1.,"Archived source evidence")],ctx)
    assert worker.run_once(outbox,[ctx.account_id])["failed"] == 0
    graph = OneHopRetriever(GraphService(None,relation_reader=store),
        embedder=worker._embedder,embedding_profile=profile)
    pipe = RetrievalPipeline(Mock(),Mock(),None,Mock(),RetrievalConfig(),
        context_reader=ContextReader(fs,storage_backend="sql"),graph_retriever=graph)
    pipe.seed_retriever.embedder = worker._embedder
    stale_hits = [LeafHit(a.uri,1.,category="entity")]
    active = pipe._expand_by_graph("memory",stale_hits,ctx,RetrievalTrace(),query_vector=[1.,0.])
    assert [hit.uri for hit in active] == [a.uri,b.uri]
    fs.archive_node(a.uri,ctx)
    # Reproduce the cleanup window: relation rows still exist and version
    # checks alone still accept their projections, but canonical read rejects.
    assert store.get_relation_neighbors(a.uri,ctx)
    archived = pipe._expand_by_graph("memory",stale_hits,ctx,RetrievalTrace(),query_vector=[1.,0.])
    assert archived == stale_hits
