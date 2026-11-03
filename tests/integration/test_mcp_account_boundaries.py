"""Exercise authenticated tenant boundaries through each MCP transport."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from core.models import ContextNode, RequestContext, Role
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.sql_control_plane_store import SQLControlPlaneStore
from server.tenant_admin import TenantAdminService
from service.tool_output import ToolOutputRepository
from tests.integration.test_mcp_streamable_http_protocol import (
    ROOT,
    _clean_subprocess_env,
    _free_port,
    _required_sql_dsn,
    _terminate,
    _tool_json,
    _wait_for_port,
)


@pytest.fixture
def tenant_data():
    dsn = _required_sql_dsn()
    pool = SharedConnectionPool(connection_string=dsn, pool_size=2)
    fs = SQLContextFS(pool=pool)
    store = SQLControlPlaneStore(pool=pool)
    keys = APIKeyManager(store)
    admin = TenantAdminService(keys, store, AuditService(store))
    run = "mcp-account-" + uuid.uuid4().hex[:12]
    rows = []
    for suffix in ("a", "b"):
        account = f"{run}-{suffix}"
        keys.create_account(account, "admin")
        keys.register_user(account, "member", "user")
        users = {row["user_id"]: row for row in keys.get_users(account)}
        admin_ctx = RequestContext(account, "admin", "", run, run, Role.ADMIN)
        for role, user in ((Role.ADMIN, "admin"), (Role.MEMBER, "member")):
            agent = f"{user}-agent"
            admin.create_agent(admin_ctx, account, agent, owner_user_id=user)
            ctx = RequestContext(account, user, agent, run, run, role,
                                 (f"user:{user}", f"agent:{agent}"))
            secret = f"{account}-{user}-private-content"
            uris = {}
            for name in ("shared", "delete", "body-delete"):
                uri = f"ctx://{account}/users/{user}/memories/preferences/{name}"
                fs.write_node(ContextNode(uri, "MEMORY", "preference", 2, f"user:{user}",
                                          secret, secret, secret), ctx)
                uris[name] = uri
            original = ToolOutputRepository(fs).store_original(
                ctx, slug="shared", content_digest=hashlib.sha256(secret.encode()).hexdigest(),
                original=secret, filtered="summary", tool_name="Read", strategy="head_tail")
            rows.append({"ctx": ctx, "key": users[user]["key"], "secret": secret,
                         "uris": uris, "original": original})
    try:
        yield {"dsn": dsn, "fs": fs, "rows": rows, "run": run}
    finally:
        pool.close_all()


def _env(dsn):
    env = _clean_subprocess_env()
    env.update({
        "VECTOR_DB_TYPE": "memory", "CONTEXTENGINE_PROVIDER": "mock",
        "INDEX_WORKERS": "0",
        "STORAGE_BACKEND": "sql", "SQL_CONNECTION_STRING": dsn,
        "OGMEM_MCP_ENABLED": "true", "OGMEM_MCP_DEFAULT_TOOLSET": "all",
        "OGMEM_MCP_ENABLE_WRITE_TOOLS": "true", "OGMEM_MCP_ENABLE_DELETE_TOOLS": "true",
        "OG_ROLE_CONTROL_ENABLED": "true", "OG_ROOT_API_KEY": "account-test-root",
        "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false", "OGMEM_AFTER_TURN_THRESHOLD": "100000",
        "OGMEM_CONFIG": str(ROOT / "__missing_mcp_account_test_config__.yaml"),
    })
    return env


def _headers(row):
    ctx = row["ctx"]
    return {"X-API-Key": row["key"], "X-Account-ID": ctx.account_id, "X-Agent-ID": ctx.agent_id}


@asynccontextmanager
async def _connect(transport, row, env, port):
    if transport == "stdio":
        stdio_env = {**env, "OGMEM_MCP_API_KEY": row["key"],
                     "OGMEM_MCP_ACCOUNT_ID": row["ctx"].account_id,
                     "OGMEM_MCP_USER_ID": "forged-user"}
        with tempfile.TemporaryFile(mode="w+") as err:
            async with stdio_client(StdioServerParameters(
                command=sys.executable, args=["-m", "cli.main", "mcp", "--transport", "stdio"],
                env=stdio_env, cwd=str(ROOT)), errlog=err) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session
    elif transport == "sse":
        session_ids = []
        async with sse_client(f"http://127.0.0.1:{port}/sse", headers=_headers(row),
                              on_session_created=session_ids.append) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                session._account_test_session_id = session_ids[0]
                yield session
    else:
        session_ids = []

        async def capture_session(response):
            if session_id := response.headers.get("mcp-session-id"):
                session_ids.append(session_id)

        async with httpx.AsyncClient(headers=_headers(row), timeout=60, trust_env=False,
                                     event_hooks={"response": [capture_session]}) as client:
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=client) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    session._account_test_session_id = session_ids[0]
                    yield session


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["streamable-http", "sse", "stdio"])
def test_mcp_account_boundaries(tenant_data, transport):
    anyio.run(_exercise_boundaries, tenant_data, transport)


async def _exercise_boundaries(data, transport):
    env = _env(data["dsn"])
    proc = None
    port = _free_port()
    if transport != "stdio":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "cli.main", "mcp", "--transport", transport,
                 "--host", "127.0.0.1", "--port", str(port),
                 "--path", "/sse" if transport == "sse" else "/mcp"],
                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            proc._ogmemory_log_path = log.name
        _wait_for_port(port, proc)
    try:
        for index, row in enumerate(data["rows"]):
            other = data["rows"][(index + 2) % 4]
            before = data["fs"].read_node(other["uris"]["shared"], other["ctx"])
            async with _connect(transport, row, env, port) as session:
                agent = {"agent_id": row["ctx"].agent_id}
                shared_uri = "ogmemory://memories/preference/shared"
                listed = await session.call_tool("ogm_list_memories", agent)
                assert not listed.is_error
                payload = _tool_json(listed)
                assert shared_uri in {item["uri"] for item in payload["memories"]}
                assert other["secret"] not in json.dumps(payload)
                read = await session.call_tool("ogm_read_memory", {"uri": shared_uri, **agent})
                assert not read.is_error
                assert _tool_json(read)["content"] == row["secret"]
                own_original = await session.call_tool("ogm_read_memory", {
                    "uri": row["original"], "recover_original": True, **agent})
                assert not own_original.is_error
                assert row["secret"] in json.dumps(_tool_json(own_original))
                for uri, recover in ((other["uris"]["shared"], False), (other["original"], True)):
                    denied = await session.call_tool("ogm_read_memory", {
                        "uri": uri, "recover_original": recover, **agent})
                    assert denied.is_error
                    assert other["secret"] not in str(denied)
                denied = await session.call_tool("ogm_delete_memory", {
                    "uri": other["uris"]["shared"], **agent})
                assert denied.is_error
                body_calls = {
                    "query_memory": {"query": "private-content"},
                    "list_memories": {},
                    "read_memory": {"uri": shared_uri},
                    "write_memory": {"text": "Remember my preference", "session_id": data["run"] + str(index)},
                    "delete_memory": {"uri": "ogmemory://memories/preference/body-delete"},
                }
                if transport != "stdio":
                    sid = session._account_test_session_id
                    path = "/mcp" if transport == "streamable-http" else f"/messages/?session_id={sid}"
                    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                        for name, args in body_calls.items():
                            response = await client.post(f"http://127.0.0.1:{port}" + path,
                                headers={**_headers(other), "Mcp-Session-Id": sid,
                                         "MCP-Protocol-Version": "2025-11-25",
                                         "Accept": "application/json, text/event-stream"},
                                json={"jsonrpc": "2.0", "id": 80, "method": "tools/call",
                                      "params": {"name": "ogm_" + name, "arguments": args}})
                            assert response.status_code == 404
                            assert row["secret"] not in response.text
                    reread = await session.call_tool("ogm_read_memory", {"uri": shared_uri, **agent})
                    assert not reread.is_error and _tool_json(reread)["content"] == row["secret"]
                for name, args in body_calls.items():
                    for field in ("account_id", "accountId", "user_id", "userId", "role"):
                        result = await session.call_tool("ogm_" + name, {
                            **args, **agent, field: other["ctx"].account_id})
                        assert result.is_error, (transport, name, field)
                preserved = data["fs"].read_node(row["uris"]["body-delete"], row["ctx"])
                assert preserved.content == row["secret"]
                written = await session.call_tool("ogm_write_memory", {
                    "text": row["secret"], "session_id": data["run"] + str(index), **agent})
                assert not written.is_error
                assert _tool_json(written)["ok"]
                query = await session.call_tool("ogm_query_memory", {"query": "private-content", **agent})
                assert not query.is_error
                assert other["secret"] not in json.dumps(_tool_json(query))
                deleted = await session.call_tool("ogm_delete_memory", {
                    "uri": "ogmemory://memories/preference/delete", **agent})
                assert not deleted.is_error and _tool_json(deleted)["deleted"]
                after = data["fs"].read_node(other["uris"]["shared"], other["ctx"])
                assert (after.content, after.metadata["version"]) == (before.content, before.metadata["version"])
            if transport != "stdio":
                async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                    for name, args in body_calls.items():
                        path = "/mcp" if transport == "streamable-http" else "/messages/?session_id=foreign"
                        response = await client.post(f"http://127.0.0.1:{port}" + path,
                            headers={**_headers(row), "X-Account-ID": other["ctx"].account_id,
                                     "Accept": "application/json, text/event-stream"},
                            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "ogm_" + name, "arguments": args}})
                        assert response.status_code == 401
    finally:
        if proc is not None:
            _terminate(proc)
