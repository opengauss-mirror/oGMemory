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

"""oGMemory SSE MCP demo using the official Python MCP SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client


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


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


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


async def resolve_tool(session: ClientSession, operation: str) -> str:
    available = {tool.name for tool in (await session.list_tools()).tools}
    for candidate in (f"ogm_{operation}_memory", f"ogm.{operation}_memory"):
        if candidate in available:
            return candidate
    raise DemoError(
        f"MCP server does not expose {operation}; available tools: " + ", ".join(sorted(available))
    )


async def checked_call(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments)
    if result.isError:
        text = "\n".join(getattr(item, "text", "") for item in result.content)
        raise DemoError(f"MCP tool {name} returned an error: {text}")
    return jsonable(result)


async def run_command(session: ClientSession, args: argparse.Namespace) -> Any:
    if args.command == "list-tools":
        return jsonable(await session.list_tools())
    if args.command == "query":
        name = await resolve_tool(session, "query")
        return await checked_call(
            session,
            name,
            {
                "query": args.query,
                "top_k": args.top_k,
                "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
            },
        )
    if args.command == "read":
        name = await resolve_tool(session, "read")
        return await checked_call(
            session,
            name,
            {
                "uri": args.uri,
                "level": args.level,
                "max_chars": args.max_chars,
                "agent_id": args.agent_id or os.environ["OGMEM_MCP_AGENT_ID"],
            },
        )
    if args.command == "write":
        name = await resolve_tool(session, "write")
        return await checked_call(session, name, write_arguments(args))
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
        tool = await resolve_tool(session, "delete")
        return await checked_call(session, tool, values)
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


async def async_main(args: argparse.Namespace) -> Any:
    url, headers = config()
    async with sse_client(
        url,
        headers=headers,
        timeout=30,
        sse_read_timeout=300,
    ) as streams:
        read_stream, write_stream = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await run_command(session, args)


def main() -> int:
    try:
        result = asyncio.run(async_main(build_parser().parse_args()))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DemoError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: MCP operation failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
