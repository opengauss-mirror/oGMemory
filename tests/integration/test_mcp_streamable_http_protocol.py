from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import anyio
import httpx
import psycopg2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from core.models import ContextNode, RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from server.mcp_resources import external_to_internal_uri

ROOT = Path(__file__).resolve().parents[2]
_ENV_PASSTHROUGH = {
    "COMSPEC",
    "HOME",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
}


@pytest.mark.integration
def test_mcp_streamable_http_protocol_exposes_tools_and_resources():
    anyio.run(_run_streamable_http_protocol_check)


@pytest.mark.integration
def test_mcp_streamable_http_protocol_exposes_all_tools_when_enabled():
    anyio.run(_run_streamable_http_all_tools_check)


@pytest.mark.integration
def test_mcp_streamable_http_protocol_can_call_all_tools_when_enabled():
    anyio.run(_run_streamable_http_all_tools_call_check)


async def _run_streamable_http_protocol_check() -> None:
    port = _free_port()
    sql_dsn = _required_sql_dsn()
    account_id = "acct-protocol-http-read"
    _seed_reviewed_skill(sql_dsn, account_id=account_id, agent_id="main")
    proc = _start_http_server(port, sql_dsn=sql_dsn)
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": account_id,
            "X-User-ID": "root",
            "X-Agent-ID": "main",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init = await session.initialize()
                    assert init.server_info.name == "oGMemory"

                    tools = await session.list_tools()
                    tool_names = {tool.name for tool in tools.tools}
                    assert tool_names == {
                        "ogm_list_memories",
                        "ogm_query_memory",
                        "ogm_read_memory",
                        "ogm_search_code",
                        "ogm_search_skills",
                        "ogm_list_skills",
                        "ogm_read_skill",
                        "ogm_list_skill_candidates",
                    }
                    resources = await session.list_resources()
                    assert {
                        "ogmemory://status", "ogmemory://skills",
                        "ogmemory://skill-candidates",
                    } <= {str(resource.uri) for resource in resources.resources}
                    templates = await session.list_resource_templates()
                    assert "ogmemory://memories/{memory_type}" in {
                        template.uri_template for template in templates.resource_templates
                    }
                    assert "ogmemory://skills/{name}" in {
                        template.uri_template for template in templates.resource_templates
                    }

                    read_result = await session.call_tool(
                        "ogm_read_memory",
                        {"uri": "ogmemory://types"},
                    )
                    assert "types" in _tool_json(read_result)

                    listed = _tool_json(await session.call_tool("ogm_list_skills", {}))
                    searched = _tool_json(await session.call_tool(
                        "ogm_search_skills", {"query": "protocol review", "top_k": 1}
                    ))
                    skill = _tool_json(await session.call_tool(
                        "ogm_read_skill", {"name": "protocol-review"}
                    ))
                    candidates = _tool_json(await session.call_tool(
                        "ogm_list_skill_candidates", {}
                    ))
                    assert any(row["name"] == "protocol-review" for row in listed["skills"])
                    assert searched["skills"][0]["name"] == "protocol-review"
                    assert skill["content"].startswith("---\nname: protocol-review")
                    assert (
                        any(
                            row["capability_key"] == "protocol-review-candidate"
                            for row in candidates["candidates"]
                        )
                    )

                    status_payload = _resource_json(
                        await session.read_resource("ogmemory://status")
                    )
                    skills_payload = _resource_json(
                        await session.read_resource("ogmemory://skills")
                    )
                    skill_payload = _resource_json(
                        await session.read_resource("ogmemory://skills/protocol-review")
                    )
                    candidate_payload = _resource_json(
                        await session.read_resource("ogmemory://skill-candidates")
                    )
                    assert status_payload["status"] == "ok"
                    protocol_skill = next(
                        row for row in skills_payload["skills"]
                        if row["name"] == "protocol-review"
                    )
                    assert protocol_skill["uri"] == (
                        "ogmemory://skills/generated/protocol-review"
                    )
                    assert skill_payload["name"] == "protocol-review"
                    assert skill_payload["content"].startswith(
                        "---\nname: protocol-review"
                    )
                    protocol_candidate = next(
                        row for row in candidate_payload["candidates"]
                        if row["capability_key"] == "protocol-review-candidate"
                    )
                    assert protocol_candidate == {
                        "capability_key": "protocol-review-candidate",
                        "state": "accumulating",
                        "evidence_count": 1,
                        "distinct_session_count": 1,
                        "summary": "Safe candidate summary.",
                        "reason": "Awaiting more evidence.",
                    }
                    _assert_no_trace_provenance(
                        skills_payload, skill_payload, candidate_payload
                    )
    finally:
        _terminate(proc)


async def _run_streamable_http_all_tools_check() -> None:
    port = _free_port()
    proc = _start_http_server(port, default_toolset="all")
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-protocol-http-tools",
            "X-User-ID": "root",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    tool_names = {tool.name for tool in tools.tools}
                    assert tool_names == {
                        "ogm_list_memories",
                        "ogm_query_memory",
                        "ogm_read_memory",
                        "ogm_search_code",
                        "ogm_search_skills",
                        "ogm_list_skills",
                        "ogm_read_skill",
                        "ogm_list_skill_candidates",
                        "ogm_install_skill",
                        "ogm_update_skill",
                        "ogm_delete_skill",
                        "ogm_write_memory",
                        "ogm_delete_memory",
                    }
                    assert all("." not in name for name in tool_names)

                    status_payload = _resource_json(
                        await session.read_resource("ogmemory://status")
                    )
                    assert status_payload["toolsets"] == {
                        "read_only": True,
                        "write": True,
                        "delete": True,
                        "code_edit": False,
                        "runtime": False,
                        "diagnostic": False,
                    }
    finally:
        _terminate(proc)


async def _run_streamable_http_all_tools_call_check() -> None:
    port = _free_port()
    sql_dsn = _required_sql_dsn()
    skill_source = tempfile.TemporaryDirectory(prefix="ogmemory-mcp-skill-")
    skill_dir = Path(skill_source.name) / "protocol-managed-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: protocol-managed-skill\n"
        "description: Exercise MCP Skill management\n---\n\n"
        "# Protocol managed Skill\n\nUse the updated protocol workflow.\n",
        encoding="utf-8",
    )
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "check.py").write_text("print('updated')\n", encoding="utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "protocol-managed-skill/SKILL.md",
            "---\nname: protocol-managed-skill\n"
            "description: Exercise MCP Skill management\n---\n\n"
            "# Protocol managed Skill\n\nUse the protocol workflow.\n",
        )
    update_archive = io.BytesIO()
    with zipfile.ZipFile(update_archive, "w", zipfile.ZIP_DEFLATED) as package:
        for file in skill_dir.rglob("*"):
            if file.is_file():
                package.write(file, file.relative_to(skill_dir.parent))
    proc = _start_http_server(
        port,
        default_toolset="all",
        sql_dsn=sql_dsn,
    )
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": f"acct-protocol-http-call-{uuid.uuid4().hex}",
            "X-User-ID": "root",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    query_result = await session.call_tool(
                        "ogm_query_memory",
                        {"query": "nothing has been written for this account", "top_k": 1},
                    )
                    assert not query_result.is_error
                    assert _tool_json(query_result)["hits"] == []

                    list_result = await session.call_tool("ogm_list_memories", {})
                    assert not list_result.is_error
                    list_payload = _tool_json(list_result)
                    assert list_payload["count"] == len(list_payload["memories"])

                    read_result = await session.call_tool(
                        "ogm_read_memory",
                        {"uri": "ogmemory://types"},
                    )
                    assert not read_result.is_error
                    assert "types" in _tool_json(read_result)

                    write_result = await session.call_tool(
                        "ogm_write_memory",
                        {
                            "session_id": "mcp-streamable-http-all-tools",
                            "text": "Remember that OpenClaw must be able to call every oGMemory MCP tool.",
                            "wait": False,
                            "client_request_id": "mcp-e2e-write-1",
                        },
                    )
                    assert not write_result.is_error
                    write_payload = _tool_json(write_result)
                    assert write_payload["ok"] is True
                    assert write_payload["accepted_messages"] == 1
                    assert write_payload["client_request_id"] == "mcp-e2e-write-1"

                    install_result = await session.call_tool(
                        "ogm_install_skill",
                        {
                            "agent_id": "main",
                            "archive_base64": base64.b64encode(
                                archive.getvalue()
                            ).decode("ascii"),
                        },
                    )
                    assert not install_result.is_error, _log_tail(proc, 8000)
                    assert _tool_json(install_result)["action"] == "create"

                    update_result = await session.call_tool(
                        "ogm_update_skill",
                        {
                            "name": "protocol-managed-skill",
                            "agent_id": "main",
                            "archive_base64": base64.b64encode(update_archive.getvalue()).decode("ascii"),
                        },
                    )
                    assert not update_result.is_error, _log_tail(proc, 8000)
                    assert _tool_json(update_result)["action"] == "update"

                    delete_skill_result = await session.call_tool(
                        "ogm_delete_skill",
                        {
                            "name": "protocol-managed-skill",
                            "source": "imported",
                            "agent_id": "main",
                        },
                    )
                    assert not delete_skill_result.is_error
                    assert _tool_json(delete_skill_result)["action"] == "delete"

                    account_id = headers["X-Account-ID"]
                    pool = SharedConnectionPool(connection_string=sql_dsn, pool_size=1)
                    try:
                        fs = SQLContextFS(pool=pool)
                        ctx = RequestContext(account_id=account_id, user_id="root", agent_id="main",
                                             session_id="protocol-delete", trace_id="protocol-delete")
                        fs.write_node(ContextNode(
                            uri=external_to_internal_uri("ogmemory://memories/preference/protocol", ctx),
                            context_type="MEMORY", category="preference", level=0,
                            owner_space="user:root", abstract="Protocol user", overview="Protocol preference",
                            content="Protocol user preference",
                        ), ctx)
                    finally:
                        pool.close_all()
                    preference_result = await session.call_tool(
                        "ogm_read_memory", {"uri": "ogmemory://memories/preference/protocol"}
                    )
                    assert not preference_result.is_error
                    assert "Protocol user preference" in _tool_json(preference_result)["content"]

                    delete_result = await session.call_tool(
                        "ogm_delete_memory",
                        {
                            "uri": "ogmemory://memories/preference/protocol",
                            "reason": "Exercise preference deletion through Streamable HTTP",
                            "client_request_id": "mcp-e2e-delete-1",
                        },
                    )
                    assert not delete_result.is_error, _log_tail(proc, 8000)
                    deleted = _tool_json(delete_result)
                    assert deleted["ok"] is True
                    assert deleted["deleted"] is True
                    assert deleted["client_request_id"] == "mcp-e2e-delete-1"

                    legacy_calls = (
                        ("ogm.query_memory", {"query": "legacy alias must be rejected", "top_k": 1}),
                        ("ogm.read_memory", {"uri": "ogmemory://types"}),
                        (
                            "ogm.write_memory",
                            {
                                "session_id": "legacy-name-rejection",
                                "text": "legacy alias must be rejected",
                                "wait": False,
                                "client_request_id": "mcp-http-legacy-write-1",
                            },
                        ),
                        (
                            "ogm.delete_memory",
                            {
                                "uri": "ogmemory://memories/preference/legacy-name-rejection",
                                "reason": "legacy alias must be rejected",
                                "client_request_id": "mcp-http-legacy-delete-1",
                            },
                        ),
                    )
                    for tool_name, arguments in legacy_calls:
                        legacy_result = await session.call_tool(tool_name, arguments)
                        assert legacy_result.is_error is True
                        error_text = "\n".join(
                            content.text
                            for content in legacy_result.content
                            if getattr(content, "type", None) == "text"
                        )
                        assert "Unknown tool" in error_text
                        assert tool_name in error_text
    finally:
        _terminate(proc)
        skill_source.cleanup()


@pytest.mark.integration
def test_mcp_streamable_http_rejects_missing_auth_for_tools_and_resources():
    anyio.run(_run_streamable_http_auth_rejection_check)


async def _run_streamable_http_auth_rejection_check() -> None:
    port = _free_port()
    proc = _start_http_server(port)
    try:
        _wait_for_port(port, proc)
        async with httpx.AsyncClient(timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    with pytest.raises(Exception, match="Server returned an error response"):
                        await session.initialize()
    finally:
        _terminate(proc)


def _start_http_server(
    port: int,
    *,
    default_toolset: str = "read-only",
    enable_write_tools: bool | None = None,
    enable_delete_tools: bool | None = None,
    sql_dsn: str | None = None,
) -> subprocess.Popen:
    env = _clean_subprocess_env()
    if enable_write_tools is None:
        enable_write_tools = default_toolset in {"read-write", "all", "full"}
    if enable_delete_tools is None:
        enable_delete_tools = default_toolset in {"all", "full"}
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": default_toolset,
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": str(enable_write_tools).lower(),
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": str(enable_delete_tools).lower(),
            "OG_ROLE_CONTROL_ENABLED": "true",
            "OG_ROOT_API_KEY": "root-key",
            "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false",
            "OGMEM_AFTER_TURN_THRESHOLD": "100000",
            "OGMEM_CONFIG": str(ROOT / "__missing_mcp_test_config__.yaml"),
        }
    )
    if sql_dsn:
        env.update({"STORAGE_BACKEND": "sql", "SQL_CONNECTION_STRING": sql_dsn})
    log_file = tempfile.NamedTemporaryFile(
        prefix="ogmemory-mcp-http-", suffix=".log", delete=False, mode="w", encoding="utf-8",
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli.main",
            "mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
        # A file avoids blocking on unread Windows pipes while preserving
        # bounded diagnostics when startup fails.
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    proc._ogmemory_log_path = log_file.name
    return proc


def _clean_subprocess_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENV_PASSTHROUGH
    }


def _required_sql_dsn() -> str:
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN")
    if not dsn:
        pytest.fail(
            "OGMEM_TEST_SQL_DSN is required; protocol acceptance must exercise "
            "the DB-backed skill tools and may not silently downgrade"
        )
    return dsn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"MCP HTTP server exited early with {proc.returncode}: {_log_tail(proc)}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"MCP HTTP server did not listen on port {port}: {_log_tail(proc)}")


def _log_tail(proc: subprocess.Popen, limit: int = 4000) -> str:
    path = getattr(proc, "_ogmemory_log_path", None)
    if not path:
        return "<no log file>"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unable to read log: {exc}>"
    return text[-limit:] or "<empty log>"


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_path = getattr(proc, "_ogmemory_log_path", None)
    if log_path:
        Path(log_path).unlink(missing_ok=True)


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    try:
        return json.loads(content.text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool result is not JSON: {content.text!r}") from exc


def _resource_json(result: Any) -> dict[str, Any]:
    content = result.contents[0]
    return json.loads(content.text)


def _assert_no_trace_provenance(*payloads: dict[str, Any]) -> None:
    serialized = json.dumps(payloads, ensure_ascii=False).lower()
    assert "raw_trace_secret" not in serialized
    assert "secret-protocol-session" not in serialized
    assert '"trace_id"' not in serialized
    assert '"session_id"' not in serialized


def _seed_reviewed_skill(dsn: str, *, account_id: str, agent_id: str) -> None:
    SQLTraceEventStore(connection_string=dsn, pool_size=1)
    candidate_id = f"protocol-candidate-{account_id}"
    uri = f"ctx://{account_id}/agents/{agent_id}/skills/protocol-review"
    content = "---\nname: protocol-review\ndescription: Review protocol behavior.\n---\n# Protocol review\n"
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account_id,))
            cur.execute(
                "DELETE FROM trace_sessions WHERE account_id = %s AND turn_id = %s",
                (account_id, "protocol-candidate-turn"),
            )
            cur.execute(
                "DELETE FROM skill_candidates WHERE account_id = %s AND candidate_id = %s",
                (account_id, candidate_id),
            )
            cur.execute("DELETE FROM context_nodes WHERE uri = %s", (uri,))
            cur.execute(
                """
                INSERT INTO context_nodes
                    (uri, account_id, owner_space, category, context_type, level,
                     status, content, abstract, overview, relations, metadata, version)
                VALUES (%s, %s, %s, 'skill', 'SKILL', 0, 'ACTIVE', %s, %s, %s,
                        '[]'::jsonb, %s::jsonb, 1)
                """,
                (uri, account_id, f"agent:{agent_id}", content,
                 "Review protocol behavior.", "Agent skill: protocol-review",
                 json.dumps({"ogmemory_generated": True,
                             "publication_status": "published", "review_score": .95})),
            )
            cur.execute(
                """
                INSERT INTO trace_sessions
                    (turn_id, uri, account_id, owner_space, agent_id, session_id,
                     started_at, ended_at, event_count, abstract, overview,
                     content, metadata, status, stop_event_index)
                VALUES
                    ('protocol-candidate-turn', %s, %s, %s, %s,
                     'secret-protocol-session', NOW() - INTERVAL '1 minute', NOW(),
                     1, '', '', %s::jsonb, '{}'::jsonb, 'finalized', 1)
                """,
                (
                    f"ctx://{account_id}/agents/{agent_id}/traces/protocol-candidate-turn",
                    account_id,
                    f"agent:{agent_id}",
                    agent_id,
                    json.dumps([{"event": "RAW_TRACE_SECRET"}]),
                ),
            )
            cur.execute(
                """
                INSERT INTO skill_candidates
                    (candidate_id, account_id, agent_id, capability_key,
                     state, assessment)
                VALUES (%s, %s, %s, 'protocol-review-candidate',
                        'assessed', %s::jsonb)
                """,
                (
                    candidate_id,
                    account_id,
                    agent_id,
                    json.dumps(
                        {
                            "summary": "Safe candidate summary.",
                            "reason": "Awaiting more evidence.",
                        }
                    ),
                ),
            )
            cur.execute(
                """
                INSERT INTO skill_candidate_evidence
                    (account_id, candidate_id, evidence_uri, evidence_type, session_id)
                VALUES (%s, %s, %s, 'trace', 'secret-protocol-session')
                """,
                (
                    account_id,
                    candidate_id,
                    f"ctx://{account_id}/agents/{agent_id}/traces/protocol-candidate-turn",
                ),
            )
        conn.commit()
    finally:
        conn.close()
