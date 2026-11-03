"""PR314: canonical lifecycle and shared-owner authorization gates."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import ContextNode, LeafHit, RelationNeighbor, RetrievalConfig
from retrieval.context_reader import ContextReader
from retrieval.graph_retriever import OneHopRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.trace import RetrievalTrace
from tests.unit.retrieval.test_graph_supplement import fixture, CTX, ROOT, PROFILE


@pytest.mark.parametrize("authorized", [True, False])
def test_two_hop_shared_agent_uses_canonical_sql_acl(authorized):
    lane, blocks, edge = fixture()
    edge("a", "b"); edge("b", "c")
    original_graph = lane.graph._relation_reader.get_traversal_neighbors.side_effect
    shared = "ctx://acct/agents/shared/memories/entities/b"
    def neighbors(uri, ctx, **kwargs):
        rows = original_graph(ROOT+"b" if uri == shared else uri, ctx, **kwargs)
        return [{**row, "source_uri": shared if row["source_uri"] == ROOT+"b" else row["source_uri"],
            "target_uri": shared if row["target_uri"] == ROOT+"b" else row["target_uri"]} for row in rows]
    lane.graph._relation_reader.get_traversal_neighbors.side_effect = neighbors
    nodes = {uri: ContextNode(uri=uri,context_type="MEMORY",category="entity",level=3,
        owner_space="user:user",abstract=block.abstract,overview=block.overview,
        content=block.content_excerpt,metadata=block.metadata) for uri,block in blocks.items()}
    nodes[shared] = replace(nodes.pop(ROOT+"b"),uri=shared,owner_space="agent:shared")
    fs = Mock(); fs.read_node.side_effect = lambda uri, ctx: nodes[uri]
    lane.reader = ContextReader(fs,storage_backend="sql")
    ctx = replace(CTX,visible_owner_spaces=("user:user", "agent:shared") if authorized else ("user:user",))
    if authorized:
        assert lane.read(shared, ctx).owner_space == "agent:shared"
    else:
        with pytest.raises(AccessDeniedError):
            lane.read(shared, ctx)
    initial = [LeafHit(ROOT+"a",1.)]
    result = lane.select("memory", initial, initial, ctx,query_vector=[1.,0.])
    assert [row["uri"] for row in result["selected"]] == ([shared, ROOT+"c"] if authorized else [])


@pytest.mark.parametrize("rejection", [NodeNotFoundError, AccessDeniedError])
def test_one_hop_cannot_expand_archived_or_inaccessible_anchor(rejection):
    lane, blocks, edge = fixture()
    store = Mock()
    store.get_relation_neighbors.return_value = [RelationNeighbor(uri=ROOT+"b", category="entity",
        reason="Old relation reason",weight=1.,source_revision=1,target_revision=1,
        edge_projection=dict(vector=[1.,0.],embedding_profile=PROFILE["profile_id"],
            model=PROFILE["model"],dimension=2,representation_version=PROFILE["representation_version"],
            source_revision=1,target_revision=1))]
    embedder = Mock()
    graph = OneHopRetriever(store,embedder=embedder,embedding_profile=PROFILE)
    pipe = RetrievalPipeline(Mock(),Mock(),None,Mock(),RetrievalConfig(),
        context_reader=lane.reader,graph_retriever=graph)
    pipe.seed_retriever.embedder = embedder
    original_read = lane.reader.read_authorized.side_effect
    def read(uri, ctx):
        if uri == ROOT+"a":
            if rejection is AccessDeniedError:
                raise AccessDeniedError(uri,ctx.account_id,"revoked")
            raise NodeNotFoundError(uri)
        return original_read(uri,ctx)
    lane.reader.read_authorized.side_effect = read
    initial = [LeafHit(ROOT+"a",1.)]
    result = pipe._expand_by_graph("memory",initial,CTX,RetrievalTrace(),query_vector=[1.,0.])
    assert result == initial
    store.get_relation_neighbors.assert_not_called()
    # Active anchors retain the existing one-hop behavior.
    lane.reader.read_authorized.side_effect = original_read
    result = pipe._expand_by_graph("memory",initial,CTX,RetrievalTrace(),query_vector=[1.,0.])
    assert [hit.uri for hit in result] == [ROOT+"a",ROOT+"b"]
