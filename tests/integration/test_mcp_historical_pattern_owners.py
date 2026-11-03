"""Resolve historical pattern names in their selected owner before legacy fallback."""
from urllib.parse import quote
from uuid import uuid4

import pytest

from core.models import ContextNode, RequestContext, Role
from providers.unified_config import OgMemConfig
from server.mcp_resources import render_resource
from server.mcp_tools import delete_memory, read_memory
from server.memory_service import MemoryService
from tests.integration.test_mcp_streamable_http_protocol import _required_sql_dsn


@pytest.fixture
def pattern_store():
    service = MemoryService(config=OgMemConfig(
        storage_backend="sql", sql_connection_string=_required_sql_dsn(),
        index_workers=0, provider="mock",
    ))
    ctx = RequestContext("pattern-owner-" + uuid4().hex[:12], "alice", "bot",
                         "session", "trace", Role.ADMIN, ("user:alice", "agent:bot"))
    try:
        yield service, service._get_context_fs(), ctx
    finally:
        service.shutdown()


def _write_pattern(fs, ctx, owner, name):
    kind, owner_id = owner.split(":")
    uri = f"ctx://{ctx.account_id}/{kind}s/{owner_id}/memories/patterns/{name}"
    content = owner + " pattern"
    fs.write_node(ContextNode(uri, "MEMORY", "pattern", 2, owner,
                              content, content, content), ctx)
    return uri


@pytest.mark.integration
@pytest.mark.parametrize("agent_name", [
    "Mixed历史", quote("Mixed历史", safe=""), "Mixed%e5%8e%86%e5%8f%b2",
])
@pytest.mark.parametrize("selected_owner", ["user:alice", "agent:bot"])
@pytest.mark.parametrize("operation", ["read", "resource", "delete"])
def test_listed_historical_pattern_uses_selected_owner(pattern_store, agent_name, selected_owner, operation):
    service, fs, ctx = pattern_store
    uris = {
        "user:alice": _write_pattern(fs, ctx, "user:alice", "Mixed%E5%8e%86%e5%8F%B2"),
        "agent:bot": _write_pattern(fs, ctx, "agent:bot", agent_name),
    }
    listed = render_resource(service, "ogmemory://memories/pattern", ctx)
    public_uri = next(row["uri"] for row in listed["memories"]
                      if row["abstract"] == selected_owner + " pattern")
    if operation == "read":
        assert read_memory(service, uri=public_uri, request_context=ctx)["content"] == selected_owner + " pattern"
    elif operation == "resource":
        assert render_resource(service, public_uri, ctx)["content"] == selected_owner + " pattern"
    else:
        assert delete_memory(service, uri=public_uri, request_context=ctx)["deleted"]
        assert fs.archived_node_version(uris[selected_owner], ctx) is not None
        other_owner = "agent:bot" if selected_owner == "user:alice" else "user:alice"
        assert fs.read_node(uris[other_owner], ctx).content == other_owner + " pattern"


@pytest.mark.integration
@pytest.mark.parametrize("operation", ["read", "resource", "delete"])
def test_historical_agent_pattern_retains_legacy_fallback(pattern_store, operation):
    service, fs, ctx = pattern_store
    uri = _write_pattern(fs, ctx, "agent:bot", "Mixed%E5%8e%86%e5%8F%B2")
    public_uri = "ogmemory://memories/pattern/" + quote("Mixed历史", safe="")
    if operation == "read":
        assert read_memory(service, uri=public_uri, request_context=ctx)["content"] == "agent:bot pattern"
    elif operation == "resource":
        assert render_resource(service, public_uri, ctx)["content"] == "agent:bot pattern"
    else:
        assert delete_memory(service, uri=public_uri, request_context=ctx)["deleted"]
        assert fs.archived_node_version(uri, ctx) is not None
