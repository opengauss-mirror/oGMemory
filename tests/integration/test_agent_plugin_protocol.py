"""Real MCP SDK -> shipped stdio proxy -> production MCP -> service double."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]


def test_agent_plugin_protocol():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.integration.agent_plugin_fixture", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise AssertionError("Fixture server exited before listening")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError("Fixture did not start")
        anyio.run(_check, node, port)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def _check(node, port):
    env = {**os.environ, "OGMEM_MCP_URL": f"http://127.0.0.1:{port}/mcp",
           "OGMEM_MCP_API_KEY": "agent-plugin-test-key", "OGMEM_MCP_ACCOUNT_ID": "test",
           "OGMEM_MCP_USER_ID": "alice", "OGMEM_MCP_AGENT_ID": "test-agent"}
    # Avoid a developer's explicitly selected config affecting this test.
    env.pop("OGMEM_AGENT_PLUGIN_CONFIG", None)
    server = StdioServerParameters(command=node, args=[str(ROOT / "plugins/agent-plugins/servers/mcp-proxy.mjs")], env=env)
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            assert {t.name for t in (await client.list_tools()).tools} == {
                "ogm_query_memory", "ogm_read_memory", "ogm_write_memory"}
            saved = await client.call_tool("ogm_write_memory", {
                "session_id": "write-session", "text": "Use uv for this project.",
                "force_extract": True, "wait": True})
            assert not saved.is_error
            assert json.loads(saved.content[0].text)["status"] == "completed"
    # New transport and business session, same user: persisted fixture data.
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            found = await client.call_tool("ogm_query_memory", {"query": "project", "session_id": "read-session"})
            assert not found.is_error
            hit = json.loads(found.content[0].text)["hits"][0]
            detail = await client.call_tool("ogm_read_memory", {"uri": hit["uri"]})
            assert not detail.is_error
            assert "uv" in json.loads(detail.content[0].text)["content"]
