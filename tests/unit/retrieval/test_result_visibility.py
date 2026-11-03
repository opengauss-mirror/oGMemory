from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import ContextNode, LeafHit, RequestContext, Role, TypedQuery
from retrieval.context_reader import ContextReader
from retrieval.identity import canonical_uri
from retrieval.result_ranker import ResultRanker


class _VisibilityFS:
    def __init__(self, nodes):
        self.nodes = {node.uri: node for node in nodes}
        self.archived = set()
        self.denied = set()
        self.checked = []
        self.reads = []

    def exists(self, uri, ctx):
        self.checked.append(uri)
        return uri in self.nodes and uri not in self.archived and uri not in self.denied

    def read_node(self, uri, ctx):
        self.reads.append(uri)
        if uri in self.denied:
            raise AccessDeniedError(uri, ctx.account_id)
        if uri in self.archived or uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return self.nodes[uri]


@pytest.fixture
def ctx():
    return RequestContext("acct", "alice", "bot", "session", "trace", Role.MEMBER,
                          ("user:alice", "agent:bot"))


def _node(index, category="preference"):
    root = {"preference": "users/alice/memories/preferences",
            "session_archive": "users/alice/memories/session_archives",
            "tool": "agents/bot/memories/tool_units"}[category]
    owner = "agent:bot" if category == "tool" else "user:alice"
    return ContextNode(f"ctx://acct/{root}/{index}", "MEMORY", category, 2, owner,
                       f"private summary {index}", f"overview {index}", f"content {index}")


def _leaves(nodes):
    return [LeafHit(uri=node.uri + "/content.md", score=1 - index / 100, level=2,
                    category=node.category, owner_space=node.owner_space, abstract=node.abstract)
            for index, node in enumerate(nodes)]


@pytest.mark.parametrize("category", ["preference", "session_archive"])
@pytest.mark.parametrize("fill", [0, 3, 5])
def test_archive_is_filtered_before_top_k_even_outside_content_fill(ctx, category, fill):
    nodes = [_node(index, category) for index in range(7)]
    fs = _VisibilityFS(nodes)
    stale_leaves = _leaves(nodes)
    fs.archived.update((nodes[1].uri, nodes[4].uri))

    result = ResultRanker().assemble(TypedQuery("private", "MEMORY", [], top_k=5),
        stale_leaves, None, ctx, fill_content_for_top_k=fill, context_reader=ContextReader(fs, storage_backend="sql"))

    expected = [node.uri for index, node in enumerate(nodes) if index not in (1, 4)]
    assert [canonical_uri(block.uri) for block in result] == expected
    assert all(uri in fs.checked for uri in (node.uri for node in nodes))
    assert len(fs.reads) == min(fill, len(expected))


def test_permissions_are_checked_for_hits_outside_content_fill(ctx):
    nodes = [_node(index) for index in range(5)]
    fs = _VisibilityFS(nodes)
    fs.denied.add(nodes[4].uri)
    leaves = _leaves(nodes)
    leaves.append(LeafHit("ctx://other/users/alice/memories/preferences/secret/content.md", .5, 2,
                          abstract="foreign secret"))

    result = ResultRanker().assemble(TypedQuery("private", "MEMORY", [], top_k=6), leaves,
        None, ctx, fill_content_for_top_k=3, context_reader=ContextReader(fs, storage_backend="sql"))

    assert [canonical_uri(block.uri) for block in result] == [node.uri for node in nodes[:4]]


def test_archive_between_visibility_check_and_content_read_is_filtered(ctx):
    node = _node(0)
    fs = _VisibilityFS([node])
    normal_read = fs.read_node

    def archive_then_read(uri, current_ctx):
        fs.archived.add(uri)
        return normal_read(uri, current_ctx)

    fs.read_node = archive_then_read
    result = ResultRanker().assemble(TypedQuery("private", "MEMORY", [], top_k=1), _leaves([node]),
        None, ctx, fill_content_for_top_k=1, context_reader=ContextReader(fs, storage_backend="sql"))

    assert result == []


def test_agent_units_check_the_canonical_node_without_collapsing_units(ctx):
    node = _node(0, "tool")
    fs = _VisibilityFS([node])
    leaves = [LeafHit(node.uri + "/content.md", .9, 2, category="tool",
                      metadata={"unit_key": key}) for key in ("batch", "cursor")]

    result = ResultRanker().assemble(TypedQuery("limits", "MEMORY", [], top_k=2), leaves, None,
        ctx, context_reader=ContextReader(fs, storage_backend="sql"))

    assert [block.metadata["unit_key"] for block in result] == ["batch", "cursor"]
    assert set(fs.checked) == {node.uri}
    fs.archived.add(node.uri)
    assert ResultRanker().assemble(TypedQuery("limits", "MEMORY", [], top_k=2), leaves, None,
        ctx, context_reader=ContextReader(fs, storage_backend="sql")) == []


@pytest.mark.parametrize("reader", [None, ContextReader(), SimpleNamespace()])
def test_ranker_supports_index_only_readers(ctx, reader):
    leaves = _leaves([_node(0)])
    result = ResultRanker().assemble(TypedQuery("private", "MEMORY", [], top_k=1), leaves,
                                    None, ctx, context_reader=reader)
    assert [block.uri for block in result] == [leaves[0].uri]


def test_visibility_storage_failure_is_not_a_visible_hit(ctx):
    node = _node(0)
    fs = _VisibilityFS([node])

    def unavailable(uri, current_ctx):
        raise RuntimeError("storage unavailable")

    fs.exists = unavailable
    with pytest.raises(RuntimeError, match="storage unavailable"):
        ResultRanker().assemble(TypedQuery("private", "MEMORY", [], top_k=1), _leaves([node]),
            None, ctx, context_reader=ContextReader(fs, storage_backend="sql"))
