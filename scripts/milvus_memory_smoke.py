#!/usr/bin/env python3
"""Real HTTP/model/AGFS/Milvus acceptance against an isolated deployment.

--write creates fictional memories and incurs model API calls. Use --marker after
restarting oGMemory for a read-only persistence check; --delete checks index removal.
Credentials stay in the deployment's private config/environment and are never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap_compose import ensure_identity
from scripts.compose_memory_smoke import verify_write


def wait_for_atomic_recall(post, marker: str, timeout: float = 180, interval: float = 2) -> str:
    """Require an entity memory with the expected facts, never an archive/summary echo."""
    from server.mcp_resources import parse_ogmemory_uri

    deadline = time.monotonic() + timeout
    session_id = "milvus-read-" + uuid4().hex
    while time.monotonic() < deadline:
        result = post(
            "memories/query",
            {
                "sessionId": session_id,
                "query": marker + " 项目使用什么语言？",
                "types": ["entity"],
                "top_k": 20,
            },
        )
        for hit in result.get("hits", []):
            parsed = parse_ogmemory_uri(hit["uri"])
            if parsed.kind != "item" or parsed.memory_type != "entity":
                continue
            memory = post(
                "memories/read", {"sessionId": session_id, "uri": hit["uri"], "level": "content"}
            )
            content = memory.get("content", "")
            if marker.casefold() in content.casefold() and "python" in content.casefold():
                return str(hit["uri"])
        time.sleep(interval)
    raise RuntimeError("persisted entity with expected project facts was not recalled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--marker")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    if not args.write and not args.marker:
        parser.error("read-only/deletion checks require --marker from an earlier write")
    marker = args.marker or "青柚Milvus验收" + uuid4().hex[:10]
    print(json.dumps({"marker": marker}, ensure_ascii=False), flush=True)
    logging.disable(logging.CRITICAL)
    from providers.unified_config import OgMemConfig

    config = OgMemConfig.load(os.environ.get("OGMEM_CONFIG", "/etc/ogmem/config.yaml"))
    if config.vector_db_type != "milvus":
        raise RuntimeError("this acceptance requires vector_db.type=milvus")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root_key = config.root_api_key

    def call(method, route, payload=None, *, account=None, key=None):
        account = account or config.account_id
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": key or root_key,
            "X-Account-ID": account,
            "X-User-ID": config.user_id,
            "X-Agent-ID": config.agent_id,
        }
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{config.http_port}/api/v1/{route}", body, headers, method=method
        )
        with opener.open(request, timeout=650) as response:
            return json.load(response)

    def tenant(account):
        ensure_identity(
            lambda method, route, payload=None: call(method, route, payload, account=account),
            account,
            config.user_id,
            config.agent_id,
        )
        users = call("GET", f"admin/accounts/{account}/users", account=account)["users"]
        return next(user["key"] for user in users if user["user_id"] == config.user_id)

    key = tenant(config.account_id)

    def post(route, payload):
        return call(
            "POST",
            route,
            {
                "accountId": config.account_id,
                "userId": config.user_id,
                "agentId": config.agent_id,
                **payload,
            },
            key=key,
        )

    if args.write:
        result = post(
            "after_turn",
            {
                "sessionId": "milvus-write-" + uuid4().hex,
                "messages": [
                    {
                        "role": "user",
                        "content": f"我的测试项目“{marker}”使用Python开发，每周四16:20复盘。",
                    },
                    {"role": "assistant", "content": "已记录这个测试项目的安排。"},
                ],
                "forceExtract": True,
                "wait": True,
                "timeoutSeconds": 600,
                "clientRequestId": "milvus-smoke-" + uuid4().hex,
            },
        )
        verify_write(result)
        print(
            json.dumps(
                {key: result.get(key) for key in ("status", "writes_completed", "writes_failed")}
            ),
            flush=True,
        )
    uri = wait_for_atomic_recall(post, marker)
    from core.models import RequestContext
    from server.mcp_resources import external_to_internal_uri

    ctx = RequestContext(
        config.account_id, config.user_id, config.agent_id, uuid4().hex, uuid4().hex
    )
    internal_uri = external_to_internal_uri(uri, ctx)
    index = config.to_provider_config().create_vector_index()
    try:
        if index.count({"account_id": config.account_id, "uri": internal_uri}) <= 0:
            raise RuntimeError("recalled memory has no corresponding Milvus index record")
        print(
            json.dumps(
                {
                    "recalled": True,
                    "read_from_agfs": True,
                    "uri": uri,
                    "milvus_records": index.count({"account_id": config.account_id}),
                }
            ),
            flush=True,
        )
        other = config.account_id + "-isolation"
        other_key = tenant(other)
        result = call(
            "POST",
            "memories/query",
            {
                "accountId": other,
                "userId": config.user_id,
                "agentId": config.agent_id,
                "sessionId": uuid4().hex,
                "query": marker,
                "top_k": 20,
            },
            account=other,
            key=other_key,
        )
        if result.get("hits"):
            raise RuntimeError("fictional memory leaked to another account")
        print(json.dumps({"http_tenant_isolation": True}), flush=True)
        if args.delete:
            result = post(
                "memories/delete",
                {
                    "uri": uri,
                    "reason": "isolated Milvus acceptance cleanup",
                    "sessionId": uuid4().hex,
                    "client_request_id": uuid4().hex,
                },
            )
            if result.get("ok") is False:
                raise RuntimeError("memory deletion failed")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if index.count({"account_id": config.account_id, "uri": internal_uri}) == 0:
                    print(json.dumps({"http_delete_and_milvus_removal": True}), flush=True)
                    break
                time.sleep(1)
            else:
                raise RuntimeError("deleted memory remained in Milvus")
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
