"""Real protocol proof for the public code-memory HTTP and MCP adapters.

This is end-to-end evidence for the adapter boundary only.  It starts a
task-owned PostgreSQL container plus production Gunicorn/Flask and FastMCP
streamable-HTTP processes, then crosses authenticated HTTP, MCP, canonical
services, SQL ContextFS, durable code records, shared BM25, and owner-scoped
tool-output recovery.  Semantic retrieval is explicitly disabled because it
is outside this adapter proof; no mocked service replaces the asserted path.

Set ``OGMEM_REQUIRE_REAL_BACKENDS=1`` to run this formal check.  Without that
flag the test is skipped so ordinary unit suites do not create Docker state.
Under the flag every unavailable real prerequisite is a failure, never a
skip.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

from core.models import RequestContext

_REQUIRE_REAL_BACKENDS = os.environ.get("OGMEM_REQUIRE_REAL_BACKENDS", "").strip() == "1"

# Keep the opt-in test collectible in lean unit-test environments.  When the
# formal flag is present, missing dependencies remain a hard collection
# failure rather than being converted into acceptance-looking skip evidence.
if _REQUIRE_REAL_BACKENDS:
    try:
        import psycopg2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        from fs.sql_adapter.pool import SharedConnectionPool
        from fs.sql_adapter.sql_context_fs import SQLContextFS
        from service.tool_output import ToolOutputRepository
    except ImportError as exc:
        raise RuntimeError(
            "OGMEM_REQUIRE_REAL_BACKENDS=1 requires psycopg2 and the project MCP/SQL dependencies"
        ) from exc
else:
    pytestmark = pytest.mark.skip(
        reason="set OGMEM_REQUIRE_REAL_BACKENDS=1 for real HTTP/MCP protocol evidence"
    )

ROOT = Path(__file__).resolve().parents[2]
_POSTGRES_IMAGE = "postgres:18"
_ROOT_API_KEY = "code-memory-protocol-root-key"
_ENV_PASSTHROUGH = {
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
}


@dataclass(frozen=True)
class _PostgresRuntime:
    docker: str
    container_name: str
    dsn: str
    password: str


@dataclass(frozen=True)
class _ServerProcess:
    process: subprocess.Popen[str]
    log_path: Path


@pytest.fixture
def isolated_postgres() -> Generator[_PostgresRuntime, None, None]:
    """Create one disposable loopback SQL runtime owned solely by this test."""

    if os.environ.get("OGMEM_REQUIRE_REAL_BACKENDS", "").strip() != "1":
        pytest.skip("set OGMEM_REQUIRE_REAL_BACKENDS=1 for real HTTP/MCP protocol evidence")
    docker = shutil.which("docker")
    if not docker:
        pytest.fail("Docker is required for the formal real HTTP/MCP protocol check")
    assert docker is not None
    inspected = subprocess.run(
        [docker, "image", "inspect", _POSTGRES_IMAGE],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        pytest.fail(f"required local image {_POSTGRES_IMAGE!r} is unavailable")

    port = _free_port()
    container_name = f"ogmemory-code-api-mcp-{uuid.uuid4().hex[:16]}"
    password = secrets.token_urlsafe(24)
    dsn = f"postgresql://ogmemory:{password}@127.0.0.1:{port}/ogmemory"
    started = subprocess.run(
        [
            docker,
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--publish",
            f"127.0.0.1:{port}:5432",
            "--env",
            "POSTGRES_USER=ogmemory",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--env",
            "POSTGRES_DB=ogmemory",
            _POSTGRES_IMAGE,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if started.returncode != 0:
        pytest.fail(
            "unable to start task-owned PostgreSQL runtime: "
            f"{_redact(started.stderr[-2000:], password)}"
        )

    runtime = _PostgresRuntime(
        docker=docker,
        container_name=container_name,
        dsn=dsn,
        password=password,
    )
    try:
        _wait_for_postgres(runtime)
        yield runtime
    finally:
        # The explicit, unique name is the full destructive target.  No shared
        # container, network, volume, or account state is inspected or removed.
        subprocess.run(
            [docker, "rm", "--force", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )


@pytest.mark.integration
def test_code_memory_public_http_and_mcp_protocol(
    isolated_postgres: _PostgresRuntime,
    tmp_path: Path,
) -> None:
    """Exercise public adapters over real loopback transports and SQL state."""

    account_id = f"code-protocol-{uuid.uuid4().hex[:16]}"
    agent_id = "protocol-agent"
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def protocol_feature_marker(value: str) -> str:\n"
        "    return f'initial:{value}'\n",
        encoding="utf-8",
    )
    original_marker = "durable-original-code-memory-protocol"
    recovery_uri = _seed_durable_original(
        isolated_postgres.dsn,
        account_id=account_id,
        agent_id=agent_id,
        marker=original_marker,
    )

    flask_port = _free_port()
    mcp_port = _free_port()
    runtime_env = _runtime_env(isolated_postgres.dsn, workspace)
    flask = _start_flask_server(
        flask_port,
        runtime_env,
        tmp_path / "flask.log",
        isolated_postgres.password,
    )
    mcp: _ServerProcess | None = None
    headers = {
        "X-API-Key": _ROOT_API_KEY,
        "X-Account-ID": account_id,
        "X-User-ID": "protocol-user",
        "X-Agent-ID": agent_id,
    }
    try:
        base_url = f"http://127.0.0.1:{flask_port}"
        with httpx.Client(headers=headers, timeout=30.0) as client:
            unauthenticated = httpx.post(
                f"{base_url}/api/v1/resources/code/search",
                json={"query": "protocol_feature_marker"},
                timeout=30.0,
            )
            assert unauthenticated.status_code == 401

            registration = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/sync/registrations",
                {
                    "path": str(workspace),
                    "name": "protocol-root",
                    "trigger_mode": "manual",
                    "wait_mode": "filesystem_complete",
                    "agentId": agent_id,
                },
            )
            root_id = registration.get("root_id")
            assert isinstance(root_id, str) and root_id
            assert registration["resource_root_slug"] == "protocol-root"
            assert "workspace_path" not in registration
            assert registration["diagnostics"]["state"] == "accepted"
            assert registration["diagnostics"]["freshness"] == {
                "accepted": True,
                "filesystem_complete": False,
                "index_visible": False,
                "dirty": True,
                "syncing": False,
                "last_attempt_at": None,
                "last_success_at": None,
                "last_index_visible_at": None,
            }

            synced = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/sync",
                {
                    "root_id": root_id,
                    "wait_mode": "filesystem_complete",
                    "agentId": agent_id,
                },
            )
            assert synced["accepted"] is True
            assert synced["filesystem_complete"] is True
            assert "workspace_path" not in synced
            assert synced["diagnostics"]["freshness"]["filesystem_complete"] is True
            assert synced["diagnostics"]["freshness"]["dirty"] is False

            converged = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/sync",
                {
                    "root_id": root_id,
                    "wait_mode": "index_visible",
                    "agentId": agent_id,
                },
            )
            assert converged["filesystem_complete"] is True
            assert converged["index_visible"] is True
            assert converged["diagnostics"]["state"] == "index_visible"
            assert converged["diagnostics"]["freshness"]["index_visible"] is True

            source.write_text(
                "def protocol_feature_marker(value: str) -> str:\n"
                "    return f'updated-code-memory-marker:{value}'\n",
                encoding="utf-8",
            )
            upsert = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/upsert",
                {
                    "path": "src/service.py",
                    "root_slug": "protocol-root",
                    "agentId": agent_id,
                },
            )
            assert upsert["status"] == "PENDING"

            searched = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/search",
                {
                    "query": "protocol_feature_marker",
                    "scope": {"root_slugs": ["protocol-root"]},
                    "budget": {
                        "candidate_limit": 999,
                        "result_limit": 2,
                        "snippet_chars": 128,
                        "batch_chars": 256,
                    },
                    "agentId": agent_id,
                },
            )
            assert searched["status"] in {"ok", "partial_failure"}
            evidence = searched["evidence"]
            assert evidence
            code_uri = evidence[0]["uri"]
            assert evidence[0]["locator"].startswith("protocol-root/src/service.py:")
            assert "updated-code-memory-marker" in evidence[0]["excerpt"]
            search_diagnostics = searched["diagnostics"]
            assert search_diagnostics["duration_ms"] >= 0
            assert search_diagnostics["budget_clamps"]["candidate_limit"]["requested"] == 999
            assert search_diagnostics["budget_clamps"]["candidate_limit"]["effective"] < 999
            assert set(search_diagnostics["routes"]) == {"semantic", "bm25", "symbolic", "graph"}
            assert all(route["duration_ms"] >= 0 for route in search_diagnostics["routes"].values())

            zero_result = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/search",
                {
                    "query": "bounded_unmatched_protocol_marker",
                    "scope": {"root_slugs": ["missing-root"]},
                    "budget": {"candidate_limit": 999, "result_limit": 1, "batch_chars": 64},
                    "agentId": agent_id,
                },
            )
            assert zero_result["status"] == "zero_result"
            assert zero_result["evidence"] == []
            assert zero_result["diagnostics"]["duration_ms"] >= 0
            assert zero_result["diagnostics"]["budget_clamps"]["candidate_limit"]["requested"] == 999

            snippet = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/snippet",
                {"uri": code_uri, "max_chars": 32, "agentId": agent_id},
            )
            assert snippet["uri"] == code_uri
            assert snippet["locator"].startswith("protocol-root/src/service.py:")
            assert len(snippet["excerpt"]) <= 32

            foreign = _post_json(
                client,
                base_url,
                "/api/v1/resources/code/snippet",
                {"uri": "ctx://other-account/resources/foreign", "agentId": agent_id},
                expected_status=400,
            )
            assert "exact account-local code URI" in foreign["error"]

            recovered_http = _post_json(
                client,
                base_url,
                "/api/v1/tool_output_original",
                {"memory_uri": recovery_uri, "max_chars": 160, "agentId": agent_id},
            )
            assert recovered_http["ok"] is True
            assert original_marker in recovered_http["original"]
            assert recovered_http["returned_chars"] <= 160
            assert recovered_http["diagnostics"] == {
                "state": "recovered",
                "reason": None,
                "input_chars": recovered_http["original_chars"],
                "output_chars": recovered_http["returned_chars"],
                "truncated": recovered_http["truncated"],
                "budget": {"configured_max_chars": 50000, "effective_max_chars": 160},
            }

            filter_marker = "private-filter-protocol-marker"
            filtered_bash = _post_json(
                client,
                base_url,
                "/api/v1/filter_bash",
                {
                    "command": "python -m pytest",
                    "output": (filter_marker + "\n") * 500,
                    "agentId": agent_id,
                },
            )
            filter_diagnostics = filtered_bash["diagnostics"]
            assert filtered_bash["ok"] is True
            assert filter_diagnostics["state"] in {"filtered", "fallback"}
            assert filter_diagnostics["input_chars"] == len((filter_marker + "\n") * 500)
            assert filter_diagnostics["output_chars"] == filtered_bash["filtered_chars"]
            assert filter_marker not in json.dumps(filter_diagnostics, sort_keys=True)

        _assert_no_workspace_path(
            workspace,
            registration,
            synced,
            converged,
            upsert,
            searched,
            zero_result,
            snippet,
            recovered_http,
            filtered_bash,
        )

        mcp = _start_mcp_server(
            mcp_port,
            runtime_env,
            tmp_path / "mcp.log",
            isolated_postgres.password,
        )
        mcp_payloads = anyio.run(
            _exercise_streamable_mcp,
            mcp_port,
            headers,
            recovery_uri,
            original_marker,
            recovered_http["diagnostics"],
            mcp.log_path,
        )
        code_text = mcp_payloads["code"]
        assert isinstance(code_text, str)
        assert "# code_search status:" in code_text
        assert "# query: protocol_feature_marker" in code_text
        assert "protocol-root/src/service.py" in code_text
        assert "updated-code-memory-marker" in code_text
        assert "diagnostics" not in code_text
        assert code_uri not in code_text
        assert mcp_payloads["edit"] == {
            "ok": True,
            "relative_path": "src/service.py",
            "occurrences": 1,
            "edit_state": "modified",
            "refresh_state": "accepted",
        }
        assert "mcp-edited-code-memory-marker" in mcp_payloads["code_after_edit"]
        assert "mcp-edited-code-memory-marker" in source.read_text(encoding="utf-8")
        _assert_no_workspace_path(
            workspace,
            mcp_payloads["code"],
            mcp_payloads["edit"],
            mcp_payloads["code_after_edit"],
            mcp_payloads["recovery"],
        )
    finally:
        if mcp is not None:
            _stop_server(mcp)
        _stop_server(flask)


async def _exercise_streamable_mcp(
    port: int,
    headers: dict[str, str],
    recovery_uri: str,
    original_marker: str,
    expected_recovery_diagnostics: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as http_client:
        async with streamable_http_client(
            f"http://127.0.0.1:{port}/mcp",
            http_client=http_client,
        ) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                assert initialized.server_info.name == "oGMemory"
                tool_names = {tool.name for tool in (await session.list_tools()).tools}
                assert {
                    "ogm_query_memory",
                    "ogm_search_code",
                    "ogm_read_memory",
                    "ogm_edit_file",
                } <= tool_names
                assert not {
                    "search_code",
                    "get_original_tool_output",
                    "index_codebase",
                    "read_filtered",
                } & tool_names

                code_result = await session.call_tool(
                    "ogm_search_code",
                    {
                        "query": "protocol_feature_marker",
                        "scope": {"paths": ["src/service.py"]},
                        "budget": {"result_limit": 2},
                    },
                )
                assert code_result.is_error is False
                code_payload = _tool_text(code_result)
                assert "# code_search status:" in code_payload
                assert "protocol-root/src/service.py" in code_payload
                assert "updated-code-memory-marker" in code_payload

                edit_result = await session.call_tool(
                    "ogm_edit_file",
                    {
                        "locator": "protocol-root/src/service.py",
                        "old_string": "updated-code-memory-marker",
                        "new_string": "mcp-edited-code-memory-marker",
                    },
                )
                assert edit_result.is_error is False
                edit_payload = _tool_json(edit_result)
                assert edit_payload["ok"] is True
                assert edit_payload["relative_path"] == "src/service.py"
                assert edit_payload["refresh_state"] in {"accepted", "visible"}

                code_after_edit_result = await session.call_tool(
                    "ogm_search_code",
                    {
                        "query": "protocol_feature_marker",
                        "scope": {"paths": ["src/service.py"]},
                        "budget": {"result_limit": 2},
                    },
                )
                assert code_after_edit_result.is_error is False
                code_after_edit_payload = _tool_text(code_after_edit_result)
                assert "mcp-edited-code-memory-marker" in code_after_edit_payload

                recovery_result = await session.call_tool(
                    "ogm_read_memory",
                    {
                        "uri": recovery_uri,
                        "recover_original": True,
                        "max_chars": 1600,
                        "agent_id": "protocol-agent",
                    },
                )
                assert recovery_result.is_error is False, log_path.read_text()[-8000:]
                recovery_payload = _tool_json(recovery_result)
                assert recovery_payload["ok"] is True
                assert original_marker in recovery_payload["original"]
                assert len(json.dumps(recovery_payload, ensure_ascii=False, sort_keys=True)) <= 1600
                assert recovery_payload["diagnostics"] == {
                    **expected_recovery_diagnostics,
                    "output_chars": len(recovery_payload["original"]),
                    "truncated": recovery_payload["truncated"],
                    "budget": {"configured_max_chars": 50000, "effective_max_chars": 1600},
                }
                return {
                    "code": code_payload,
                    "edit": edit_payload,
                    "code_after_edit": code_after_edit_payload,
                    "recovery": recovery_payload,
                }

    raise AssertionError("Streamable MCP session ended before returning protocol payloads")


def _runtime_env(dsn: str, workspace_root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENV_PASSTHROUGH
    }
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "STORAGE_BACKEND": "sql",
            "SQL_CONNECTION_STRING": dsn,
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_CONFIG": str(ROOT / "__missing_code_memory_protocol_config__.yaml"),
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": "read-only",
            "OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS": "true",
            "OG_ROLE_CONTROL_ENABLED": "true",
            "OG_ROOT_API_KEY": _ROOT_API_KEY,
            "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false",
            "OGMEM_AFTER_TURN_THRESHOLD": "100000",
            "OGMEM_CODE_RESOURCES_ENABLED": "true",
            "OGMEM_CODE_SYNC_ENABLED": "true",
            "OGMEM_CODE_WORKSPACE_ALLOWED_ROOTS": str(workspace_root),
            "OGMEM_CODE_RETRIEVAL_ENABLED": "true",
            "OGMEM_CODE_RETRIEVAL_SEMANTIC_ENABLED": "false",
            "OGMEM_CODE_RETRIEVAL_BM25_ENABLED": "true",
            "OGMEM_CODE_RETRIEVAL_SYMBOLIC_ENABLED": "true",
            "OGMEM_CODE_RETRIEVAL_GRAPH_ENABLED": "false",
            "OGMEM_CODE_RETRIEVAL_ROUTE_DEADLINE_MS": "2000",
            "OGMEM_TOOL_OUTPUT_FILTER_ENABLED": "true",
        }
    )
    return env


def _start_flask_server(
    port: int,
    runtime_env: dict[str, str],
    log_path: Path,
    secret: str,
) -> _ServerProcess:
    env = {**runtime_env, "OGMEM_HTTP_PORT": str(port)}
    return _start_server(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "--workers",
            "1",
            "--bind",
            f"127.0.0.1:{port}",
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "server.app:app",
        ],
        env,
        port,
        log_path,
        secret,
        "Flask HTTP",
    )


def _start_mcp_server(
    port: int,
    runtime_env: dict[str, str],
    log_path: Path,
    secret: str,
) -> _ServerProcess:
    return _start_server(
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
        runtime_env,
        port,
        log_path,
        secret,
        "MCP streamable HTTP",
    )


def _start_server(
    command: list[str],
    env: dict[str, str],
    port: int,
    log_path: Path,
    secret: str,
    label: str,
) -> _ServerProcess:
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    server = _ServerProcess(process=process, log_path=log_path)
    try:
        _wait_for_port(port, server, secret, label)
    except Exception:
        _stop_server(server)
        raise
    return server


def _wait_for_postgres(runtime: _PostgresRuntime, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = psycopg2.connect(runtime.dsn, connect_timeout=1)
        except psycopg2.OperationalError:
            time.sleep(0.1)
            continue
        connection.close()
        return
    pytest.fail("task-owned PostgreSQL runtime did not accept loopback connections")


def _wait_for_port(
    port: int,
    server: _ServerProcess,
    secret: str,
    label: str,
    timeout: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise AssertionError(
                f"{label} exited early with {server.process.returncode}: "
                f"{_tail(server.log_path, secret)}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"{label} did not listen on loopback: {_tail(server.log_path, secret)}")


def _stop_server(server: _ServerProcess) -> None:
    if server.process.poll() is not None:
        return
    server.process.terminate()
    try:
        server.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.process.kill()
        server.process.wait(timeout=5)


def _seed_durable_original(
    dsn: str,
    *,
    account_id: str,
    agent_id: str,
    marker: str,
) -> str:
    pool = SharedConnectionPool(dsn, pool_size=2)
    context_fs = SQLContextFS(pool=pool)
    repository = ToolOutputRepository(context_fs)
    ctx = RequestContext(
        account_id=account_id,
        user_id="protocol-user",
        agent_id=agent_id,
        session_id="protocol-session",
        trace_id="protocol-trace",
    )
    try:
        return repository.store_original(
            ctx,
            slug=f"tool_output_bash_{uuid.uuid4().hex}",
            content_digest="a" * 64,
            original=marker + "\n" + ("bounded original recovery body\n" * 12),
            filtered=marker + "\n",
            tool_name="Bash",
            strategy="bash_protocol_fixture",
            metadata={
                "bash_category": "test_runner",
                "exit_code": 0,
                "original_lines": 13,
                "filtered_lines": 1,
                "was_shortened": True,
            },
        )
    finally:
        pool.close_all()


def _post_json(
    client: httpx.Client,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    expected_status: int = 200,
) -> dict[str, Any]:
    response = client.post(f"{base_url}{path}", json=payload)
    try:
        body = response.json()
    except ValueError as exc:
        raise AssertionError(f"{path} returned non-JSON HTTP {response.status_code}") from exc
    assert response.status_code == expected_status, body
    assert isinstance(body, dict)
    return body


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    try:
        decoded = json.loads(content.text)
    except (AttributeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"MCP result was not JSON: {content!r}") from exc
    assert isinstance(decoded, dict)
    return decoded


def _tool_text(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, str):
        return structured
    content = result.content[0]
    text = getattr(content, "text", None)
    if not isinstance(text, str):
        raise AssertionError(f"MCP result was not text: {content!r}")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    return decoded if isinstance(decoded, str) else text


def _assert_no_workspace_path(workspace: Path, *payloads: Any) -> None:
    serialized = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    assert str(workspace) not in serialized


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, secret: str, limit: int = 4000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unable to read log: {exc}>"
    return _redact(content[-limit:] or "<empty>", secret)


def _redact(value: str, secret: str) -> str:
    return value.replace(secret, "<redacted>")
