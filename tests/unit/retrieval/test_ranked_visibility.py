"""Ranked candidate selection keeps canonical checks within the result budget."""

from types import SimpleNamespace

from core.models import ContextNode, LeafHit, RequestContext, TypedQuery
from retrieval.context_reader import ContextReader
from retrieval.result_ranker import ResultRanker


def test_ranked_selection_checks_the_returned_prefix():
    ctx = RequestContext(account_id="acct", user_id="user", agent_id="agent", session_id="session", trace_id="trace")
    checked = []

    def is_visible(uri, read_ctx):
        assert read_ctx is ctx
        checked.append(uri)
        return True

    hits = [LeafHit(uri=f"ctx://acct/users/user/memories/entity/{i}",
                    score=i / 100, level=2) for i in range(45)]
    result = ResultRanker().assemble(
        TypedQuery(text="memory", context_type="MEMORY", categories=[], top_k=15), hits, None,
        ctx=ctx, context_reader=SimpleNamespace(is_visible=is_visible),
    )

    assert [block.uri for block in result] == [hit.uri for hit in reversed(hits[30:])]
    assert checked == [block.uri for block in result]


def test_scene_selection_retains_direct_and_expanded_quotas():
    ctx = RequestContext(account_id="acct", user_id="user", agent_id="agent", session_id="session", trace_id="trace")
    hits = [LeafHit(uri=f"ctx://acct/users/user/memories/entity/{i}",
                    score=i / 10, level=2, metadata={"scene_expanded": i >= 3})
            for i in range(6)]
    result = ResultRanker().assemble(
        TypedQuery(text="memory", context_type="MEMORY", categories=[], top_k=4), hits, None,
        ctx=ctx, context_reader=SimpleNamespace(is_visible=lambda uri, ctx: True),
        direct_evidence_reserve=0.5,
    )
    assert [block.uri for block in result] == [hits[i].uri for i in (5, 4, 2, 1)]


def test_public_search_fills_ranked_content_in_one_batch():
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    uris = [f"ctx://acct/users/user/memories/entity/{i}" for i in range(3)]
    nodes = {uri: ContextNode(uri=uri, context_type="MEMORY", category="entity",
                              owner_space="user:user", level=2, abstract=f"Memory {i}",
                              overview=f"Overview {i}", content=f"Full memory {i}",
                              metadata={"version": 2}) for i, uri in enumerate(uris)}
    batches = []

    def read_nodes(keys, read_ctx):
        assert read_ctx is ctx
        batches.append(keys)
        return {uri: nodes[uri] for uri in keys}

    reader = ContextReader(SimpleNamespace(read_nodes=read_nodes, exists=lambda uri, ctx: True))
    hits = [LeafHit(uri=uri, score=0.9 - i / 10, level=2) for i, uri in enumerate(uris)]
    result = ResultRanker().assemble(
        TypedQuery(text="memory", context_type="MEMORY", categories=[], top_k=3),
        hits, None, ctx=ctx, context_reader=reader, fill_content_for_top_k=3,
    )
    assert batches == [uris]
    assert [block.content_excerpt for block in result] == [nodes[uri].content for uri in uris]
    assert [block.metadata["version"] for block in result] == [2, 2, 2]
