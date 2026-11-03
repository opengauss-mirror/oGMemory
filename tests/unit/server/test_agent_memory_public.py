"""Public memory types and unit read-through preserve the retrieved identity."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.models import RetrievedBlock, SearchMemoryResult
from server.mcp_resources import _node_summary, internal_to_external_uri
from server.mcp_server import create_mcp_server
from server.mcp_tools import _truncate_payload, query_memory, read_memory
from tests.unit.retrieval.test_agent_units import unit_memory as _unit_memory
from tests.unit.server.test_mcp_tools_readonly import _FakeService


@pytest.fixture
def unit_memory():
    return _unit_memory.__wrapped__()


@pytest.mark.parametrize("directory,kind", [("tools", "tool_memory"), ("tool_units", "tool_evidence")])
def test_public_tool_summary_matches_its_uri(unit_memory, directory, kind):
    _, node, _, _ = unit_memory
    node = replace(node, uri=f"{node.uri.rsplit('/', 2)[0]}/{directory}/search")
    summary = _node_summary(node)
    assert summary["type"] == kind
    assert summary["uri"] == f"ogmemory://memories/{kind}/search"


@pytest.mark.parametrize("kind", ["tool_memory", "tool_evidence"])
def test_public_tool_filter_selects_requested_directory_before_limit(unit_memory, kind):
    ctx, node, _, _ = unit_memory
    other = "tool_evidence" if kind == "tool_memory" else "tool_memory"
    directories = {"tool_memory": "tools", "tool_evidence": "tool_units"}
    hits = [RetrievedBlock(uri=f"{node.uri.rsplit('/', 2)[0]}/{directories[t]}/search",
                          category="tool", score=score) for t, score in [(other, 0.95), (kind, 0.9)]]
    service = _FakeService()

    def search_memory(query, ctx, **kw):
        assert kw["memory_directories"] == [directories[kind]]
        eligible = [hit for hit in hits if f"/memories/{directories[kind]}/" in hit.uri]
        return SearchMemoryResult(request_id="query", query=query, hits=eligible[:kw["top_k"]])

    service.read_api.search_memory = search_memory
    result = query_memory(service, query="search", types=[kind], top_k=1, request_context=ctx)
    assert [hit["type"] for hit in result["hits"]] == [kind]


def test_public_query_reads_selected_tool_unit_with_small_budget(unit_memory):
    ctx, node, _, fs = unit_memory
    service = _FakeService()
    service.fs = fs
    service._get_context_fs = lambda: fs
    node.metadata["structured_fields"]["units"][0]["content"] = "Batch settings. " * 500
    units = node.metadata["structured_fields"]["units"]
    service.read_api.hits = [
        RetrievedBlock(uri=node.uri + "/content.md", category="tool", abstract=u["title"],
                       metadata={"unit_key": u["key"], "unit_revision": i + 1,
                                 "grounding": u["grounding"]})
        for i, u in enumerate(units)
    ]
    result = query_memory(service, query="pagination", top_k=2, request_context=ctx)
    assert [hit.get("unit_key") for hit in result["hits"]] == ["batch", "cursor"]
    hit = result["hits"][1]
    content = read_memory(service, uri=hit["uri"], unit_key=hit["unit_key"],
                          level="content", max_chars=200, request_context=ctx)
    assert content["unit_key"] == "cursor"
    assert content["content"] == "search\n\nSearch pagination\n\nReuse the returned cursor for the next page."
    assert content["unit_revision"] == 2
    assert content["grounding"] == units[1]["grounding"]

    clipped = _truncate_payload({"content": content["content"], "grounding": units[1]["grounding"] * 20}, 200)
    assert len(clipped["content"]) + sum(len(json.dumps(g)) for g in clipped["grounding"]) <= 200
    assert clipped["truncated"] is True


def test_public_unit_read_loads_full_node_for_abstract(unit_memory):
    ctx, node, _, _ = unit_memory
    fs = SimpleNamespace(read_node=lambda uri, ctx: node,
                         read_node_fields=lambda *args: replace(node, metadata={}, content=""))
    service = _FakeService()
    service._get_context_fs = lambda: fs
    result = read_memory(service, uri=internal_to_external_uri(node.uri), unit_key="cursor",
                         level="abstract", request_context=ctx)
    assert result["abstract"] == "Search pagination"


def test_mcp_read_tool_forwards_unit_identity(unit_memory):
    ctx, node, _, fs = unit_memory
    service = _FakeService()
    service.fs = fs
    service.build_context = lambda params, identity=None: ctx
    mcp = create_mcp_server(service=service, config=service._cfg)
    tool = mcp._tool_manager.get_tool("ogm_read_memory")
    arguments = tool.fn_metadata.validate_arguments({
        "uri": internal_to_external_uri(node.uri), "unit_key": "cursor", "level": "content"})
    result = tool.fn(**arguments)
    assert result["unit_key"] == "cursor"
    assert result["content"] == "search\n\nSearch pagination\n\nReuse the returned cursor for the next page."
