from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from server.mcp_server import (
    DELETE_MEMORY_TOOL_NAME,
    DELETE_SKILL_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LIST_MEMORY_TOOL_NAME,
    LIST_SKILL_CANDIDATES_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    QUERY_MEMORY_TOOL_NAME,
    READ_MEMORY_TOOL_NAME,
    READ_SKILL_TOOL_NAME,
    SEARCH_CODE_TOOL_NAME,
    SEARCH_SKILLS_TOOL_NAME,
    UPDATE_SKILL_TOOL_NAME,
    WRITE_MEMORY_TOOL_NAME,
)

ROOT = Path(__file__).resolve().parents[2]

MCP_BUSINESS_TOOLS = {
    "ogm_get_write_request_status",
    LIST_MEMORY_TOOL_NAME,
    QUERY_MEMORY_TOOL_NAME,
    READ_MEMORY_TOOL_NAME,
    WRITE_MEMORY_TOOL_NAME,
    DELETE_MEMORY_TOOL_NAME,
    SEARCH_SKILLS_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    READ_SKILL_TOOL_NAME,
    SEARCH_CODE_TOOL_NAME,
    LIST_SKILL_CANDIDATES_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    UPDATE_SKILL_TOOL_NAME,
    DELETE_SKILL_TOOL_NAME,
}
EXPECTED_OPENCLAW_BUSINESS_TOOLS = {f"ogmemory__{name}" for name in MCP_BUSINESS_TOOLS}
EXPECTED_OPENCLAW_RESOURCE_TOOLS = {
    "ogmemory__resources_list",
    "ogmemory__resources_read",
}
EXPECTED_OPENCLAW_TOOLS = EXPECTED_OPENCLAW_BUSINESS_TOOLS | EXPECTED_OPENCLAW_RESOURCE_TOOLS
OPENCLAW_INCLUDE_FILTER = "ogm_*,resources_*"
QUERY_ONLY_INCLUDE_FILTER = "ogm_query_memory,resources_*"
QUERY_ONLY_OPENCLAW_TOOLS = {
    f"ogmemory__{QUERY_MEMORY_TOOL_NAME}",
    *EXPECTED_OPENCLAW_RESOURCE_TOOLS,
}


def test_openclaw_include_filter_matches_underscore_tool_names():
    patterns = OPENCLAW_INCLUDE_FILTER.split(",")

    assert all(any(fnmatchcase(name, pattern) for pattern in patterns) for name in MCP_BUSINESS_TOOLS)
    assert not any(
        any(fnmatchcase(name.replace("_", ".", 1), pattern) for pattern in patterns)
        for name in MCP_BUSINESS_TOOLS
    )


def test_parse_openclaw_probe_json_ignores_bundle_logs():
    output = """
[bundle-mcp] tool "ogm_query_memory" from server "ogmemory" registered.
{
  "generatedAt": "2026-06-24T10:14:08.071Z",
  "servers": {
    "ogmemory": {
      "launch": "http://127.0.0.1:8091/mcp",
      "tools": 14,
      "resources": true,
      "prompts": true
    }
  },
  "tools": [
    "ogmemory__ogm_list_memories",
    "ogmemory__ogm_query_memory",
    "ogmemory__ogm_read_memory",
    "ogmemory__ogm_get_write_request_status",
    "ogmemory__ogm_write_memory",
    "ogmemory__ogm_delete_memory",
    "ogmemory__ogm_search_skills",
    "ogmemory__ogm_list_skills",
    "ogmemory__ogm_read_skill",
    "ogmemory__ogm_list_skill_candidates",
    "ogmemory__ogm_search_code",
    "ogmemory__ogm_install_skill",
    "ogmemory__ogm_update_skill",
    "ogmemory__ogm_delete_skill",
    "ogmemory__resources_list",
    "ogmemory__resources_read"
  ],
  "diagnostics": []
}
"""

    payload = _parse_openclaw_probe_json(output)

    assert payload["servers"]["ogmemory"]["tools"] == len(EXPECTED_OPENCLAW_BUSINESS_TOOLS)
    assert set(payload["tools"]) == EXPECTED_OPENCLAW_TOOLS
    assert payload["diagnostics"] == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("include_filter", "expected_tools"),
    [
        pytest.param(OPENCLAW_INCLUDE_FILTER, EXPECTED_OPENCLAW_TOOLS, id="all-tools"),
        pytest.param(
            QUERY_ONLY_INCLUDE_FILTER,
            QUERY_ONLY_OPENCLAW_TOOLS,
            id="query-and-resources-only",
        ),
    ],
)
def test_real_openclaw_native_mcp_probe_applies_include_filter(
    tmp_path,
    include_filter,
    expected_tools,
):
    """Run against a real OpenClaw CLI when available.

    This test intentionally skips instead of faking OpenClaw. It verifies the
    actual `openclaw mcp add` + `openclaw mcp probe --json` path against a real
    oGMemory Streamable HTTP MCP server with the full toolset enabled.
    """

    openclaw = _find_openclaw()
    if not openclaw:
        pytest.skip("OPENCLAW_BIN/openclaw not available; real OpenClaw MCP e2e skipped")

    port = _free_port()
    proc = _start_ogmemory_mcp_server(port)
    try:
        _wait_for_port(port, proc)
        env = _openclaw_env(tmp_path)
        url = f"http://127.0.0.1:{port}/mcp"

        _run_openclaw(
            openclaw,
            [
                "mcp",
                "add",
                "ogmemory",
                "--transport",
                "streamable-http",
                "--url",
                url,
                "--header",
                "X-API-Key=root-key",
                "--header",
                "X-Account-ID=acct-openclaw-e2e",
                "--header",
                "X-User-ID=openclaw-e2e",
                "--include",
                include_filter,
                "--parallel",
                "--timeout",
                "60",
                "--connect-timeout",
                "20",
                "--no-probe",
            ],
            env=env,
        )
        probe = _run_openclaw(openclaw, ["mcp", "probe", "ogmemory", "--json"], env=env)
        payload = _parse_openclaw_probe_json(probe.stdout)
        expected_business_tools = expected_tools & EXPECTED_OPENCLAW_BUSINESS_TOOLS

        assert set(payload.get("tools", [])) == expected_tools
        assert payload["servers"]["ogmemory"]["tools"] == len(expected_business_tools)
        assert payload["servers"]["ogmemory"]["resources"] is True
        assert payload["diagnostics"] == []
    finally:
        _terminate(proc)


def _parse_openclaw_probe_json(output: str) -> dict:
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            return json.loads(output[index:])
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"OpenClaw probe output did not contain JSON: {output!r}")


def _find_openclaw() -> str | None:
    configured = os.environ.get("OPENCLAW_BIN")
    if configured:
        return configured
    return shutil.which("openclaw")


def _openclaw_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    state = home / ".openclaw"
    home.mkdir(parents=True)
    state.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "OPENCLAW_STATE_DIR": str(state),
            "OPENCLAW_REAPPLY_CONFIG": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    return env


def _run_openclaw(
    openclaw: str,
    args: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [openclaw, *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            "OpenClaw command failed\n"
            f"command: {[openclaw, *args]!r}\n"
            f"exit: {result.returncode}\n"
            f"output:\n{result.stdout}"
        )
    return result


def _start_ogmemory_mcp_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": "all",
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": "true",
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": "true",
            "OGMEM_MCP_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "OG_ROLE_CONTROL_ENABLED": "true",
            "OG_ROOT_API_KEY": "root-key",
            "OG_ACCOUNT_ID": "acct-openclaw-e2e",
            "OG_USER_ID": "openclaw-e2e",
            "OG_AGENT_ID": "openclaw-e2e-agent",
            "OGMEM_CONFIG": str(ROOT / "__missing_openclaw_mcp_e2e_config__.yaml"),
        }
    )
    return subprocess.Popen(
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(
                f"oGMemory MCP server exited early with {proc.returncode}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"oGMemory MCP server did not listen on port {port}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
