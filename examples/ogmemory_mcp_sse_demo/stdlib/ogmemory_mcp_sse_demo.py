#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Zero-dependency oGMemory SSE MCP client demo."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class DemoError(RuntimeError):
    pass


def config() -> tuple[str, dict[str, str]]:
    required = {
        "X-API-Key": "OGMEM_MCP_API_KEY",
        "X-Account-ID": "OGMEM_MCP_ACCOUNT_ID",
        "X-User-ID": "OGMEM_MCP_USER_ID",
        "X-Agent-ID": "OGMEM_MCP_AGENT_ID",
    }
    missing = [name for name in required.values() if not os.environ.get(name)]
    if missing:
        raise DemoError("Missing environment variables: " + ", ".join(missing))
    return (
        os.environ.get("OGMEM_MCP_URL", "http://127.0.0.1:8191/mcp"),
        {header: os.environ[env_name] for header, env_name in required.items()},
    )


class SseMcpClient:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self.endpoint: str | None = None
        self.response: Any = None
        self.next_id = 1
        self.endpoint_ready = threading.Event()
        self.pending: dict[int, queue.Queue[Any]] = {}
        self.pending_lock = threading.Lock()
        self.reader_error: Exception | None = None

    def __enter__(self) -> SseMcpClient:
        request = urllib.request.Request(
            self.url,
            headers={**self.headers, "Accept": "text/event-stream"},
            method="GET",
        )
        try:
            self.response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            raise DemoError(f"SSE connection failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise DemoError(f"Cannot reach MCP server: {exc.reason}") from exc
        content_type = self.response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            self.response.close()
            raise DemoError(f"Expected text/event-stream, got {content_type or 'unknown'}")
        threading.Thread(target=self._read_events, daemon=True).start()
        if not self.endpoint_ready.wait(30):
            self.close()
            raise DemoError("MCP server did not provide an SSE message endpoint")
        if self.reader_error:
            raise DemoError(f"SSE reader failed ({type(self.reader_error).__name__})")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.response is not None:
            self.response.close()

    def _read_events(self) -> None:
        event = "message"
        data_lines: list[str] = []
        try:
            for raw_line in self.response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    self._dispatch(event, "\n".join(data_lines))
                    event, data_lines = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except Exception as exc:
            self.reader_error = exc
            self.endpoint_ready.set()
            with self.pending_lock:
                waits = list(self.pending.values())
            for waiting in waits:
                waiting.put(exc)

    def _dispatch(self, event: str, data: str) -> None:
        if not data:
            return
        if event == "endpoint":
            endpoint = urllib.parse.urljoin(self.url, data)
            base, resolved = urllib.parse.urlparse(self.url), urllib.parse.urlparse(endpoint)
            if (base.scheme, base.netloc) != (resolved.scheme, resolved.netloc):
                self.reader_error = DemoError("SSE endpoint changed origin")
            else:
                self.endpoint = endpoint
            self.endpoint_ready.set()
            return
        if event != "message":
            return
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return
        request_id = message.get("id")
        if isinstance(request_id, int):
            with self.pending_lock:
                waiting = self.pending.get(request_id)
            if waiting:
                waiting.put(message)

    def _post(self, payload: dict[str, Any]) -> None:
        if not self.endpoint:
            raise DemoError("SSE message endpoint is unavailable")
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={**self.headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status not in (200, 202):
                    raise DemoError(f"MCP message POST returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            raise DemoError(f"MCP message POST failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise DemoError(f"MCP message POST failed: {exc.reason}") from exc

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self.next_id
        self.next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        waiting: queue.Queue[Any] = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.pending[request_id] = waiting
        try:
            self._post(payload)
            try:
                response = waiting.get(timeout=300)
            except queue.Empty as exc:
                raise DemoError(f"Timed out waiting for MCP response to {method}") from exc
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)
        if isinstance(response, Exception):
            raise DemoError(f"SSE reader failed ({type(response).__name__})")
        if response.get("error"):
            error = response["error"]
            raise DemoError(f"MCP error {error.get('code')}: {error.get('message')}")
        if "result" not in response:
            raise DemoError("MCP response has no result")
        return response["result"]

    def notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method})

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ogmemory-stdlib-sse-demo", "version": "1.0.0"},
            },
        )
        self.notify("notifications/initialized")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict) and result.get("isError"):
            raise DemoError(f"MCP tool {name} returned an error: {tool_text(result)}")
        return result


def tool_text(result: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


def resolve_tool(client: SseMcpClient, operation: str) -> str:
    tools = client.request("tools/list").get("tools", [])
    available = {tool.get("name") for tool in tools}
    for candidate in (f"ogm_{operation}_memory", f"ogm.{operation}_memory"):
        if candidate in available:
            return candidate
    raise DemoError(
        f"MCP server does not expose {operation}; available tools: "
        + ", ".join(sorted(name for name in available if name))
    )


def write_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "session_id": args.session_id,
        "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
        "force_extract": args.force_extract,
        "wait": args.wait,
    }
    if args.text is not None:
        values["text"] = args.text
    else:
        try:
            messages = json.loads(args.messages_json)
        except json.JSONDecodeError as exc:
            raise DemoError(f"--messages-json is not valid JSON: {exc.msg}") from exc
        if not isinstance(messages, list):
            raise DemoError("--messages-json must be a JSON array")
        values["messages"] = messages
    if args.client_request_id:
        values["client_request_id"] = args.client_request_id
    return values


def run_command(client: SseMcpClient, args: argparse.Namespace) -> Any:
    if args.command == "list-tools":
        return client.request("tools/list")
    if args.command == "query":
        return client.call_tool(
            resolve_tool(client, "query"),
            {
                "query": args.query,
                "top_k": args.top_k,
                "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
            },
        )
    if args.command == "read":
        return client.call_tool(
            resolve_tool(client, "read"),
            {
                "uri": args.uri,
                "level": args.level,
                "max_chars": args.max_chars,
                "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
            },
        )
    if args.command == "write":
        return client.call_tool(resolve_tool(client, "write"), write_arguments(args))
    if args.command == "delete":
        if not args.yes:
            raise DemoError("delete requires --yes because it archives the target memory")
        values = {
            "uri": args.uri,
            "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
        }
        for name in ("reason", "session_id", "client_request_id"):
            value = getattr(args, name)
            if value:
                values[name] = value
        return client.call_tool(resolve_tool(client, "delete"), values)
    raise DemoError(f"Unknown command: {args.command}")


def add_write_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--messages-json")
    parser.add_argument("--agent-id")
    parser.add_argument("--client-request-id")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--wait", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list-tools")
    query = commands.add_parser("query")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--agent-id")
    read = commands.add_parser("read")
    read.add_argument("uri")
    read.add_argument("--level", default="all", choices=("abstract", "overview", "content", "all"))
    read.add_argument("--max-chars", type=int, default=12000)
    read.add_argument("--agent-id")
    write = commands.add_parser("write")
    add_write_options(write)
    delete = commands.add_parser("delete")
    delete.add_argument("uri")
    delete.add_argument("--reason")
    delete.add_argument("--session-id")
    delete.add_argument("--agent-id")
    delete.add_argument("--client-request-id")
    delete.add_argument("--yes", action="store_true")
    return root


def main() -> int:
    try:
        args = build_parser().parse_args()
        url, headers = config()
        with SseMcpClient(url, headers) as client:
            client.initialize()
            result = run_command(client, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DemoError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
