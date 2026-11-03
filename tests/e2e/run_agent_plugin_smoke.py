"""Opt-in live write/read check. Uses configured service; writes one unique fact.

Set OGMEM_AGENT_PLUGIN_CONFIG to a TEST identity's private connection JSON.
Requires Node.js and the repository MCP Python SDK dependency.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]


def payload(result):
    if result.is_error:
        raise RuntimeError("MCP tool reported a failure")
    return json.loads(next(item.text for item in result.content if item.type == "text"))


async def main():
    if not os.environ.get("OGMEM_AGENT_PLUGIN_CONFIG"):
        raise SystemExit("Set OGMEM_AGENT_PLUGIN_CONFIG to an isolated test connection before writing.")
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js is required")
    server = StdioServerParameters(command=node, args=[str(ROOT / "plugins/agent-plugins/servers/mcp-proxy.mjs")], env=dict(os.environ))
    marker = "plugin-check-" + uuid4().hex[:12]
    fact = f"请记住：{marker} 项目固定在每周四北京时间20点发布，包管理工具统一使用 uv。"
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=180) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert names == {"ogm_query_memory", "ogm_read_memory", "ogm_write_memory"}, names
            saved = payload(await session.call_tool("ogm_write_memory", {
                "session_id": marker + "-write", "client_request_id": marker,
                "text": fact, "force_extract": True, "wait": True}))
            print(json.dumps({"phase": "write", "marker": marker, "result": saved}, ensure_ascii=False), flush=True)
            assert saved.get("ok") is True
    # New connection and a different business session: recall without chat history.
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=180) as session:
            await session.initialize()
            found = None
            for _ in range(6):
                found = payload(await session.call_tool("ogm_query_memory", {
                    "query": f"{marker} 项目的发布时间和包管理工具是什么？",
                    "session_id": marker + "-read", "top_k": 5}))
                if any(marker in json.dumps(hit, ensure_ascii=False) for hit in found.get("hits", [])):
                    break
                await anyio.sleep(2)
            hit = next(hit for hit in found.get("hits", []) if marker in json.dumps(hit, ensure_ascii=False))
            detail = payload(await session.call_tool("ogm_read_memory", {"uri": hit["uri"]}))
            assert marker in json.dumps(detail, ensure_ascii=False)
            print(json.dumps({"phase": "recall", "marker": marker, "uri": hit["uri"], "detail": detail}, ensure_ascii=False), flush=True)
            print("PASS: three tools; live write; new-session query and exact-URI read.", flush=True)


if __name__ == "__main__":
    anyio.run(main)
