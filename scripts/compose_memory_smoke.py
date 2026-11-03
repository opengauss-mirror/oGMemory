#!/usr/bin/env python3
"""Explicit fictional-memory acceptance test; --write incurs model calls and writes."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import urllib.request
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_deployment import config_path


def verify_write(result):
    if (result.get('ok') is not True or result.get('status') != 'completed'
            or result.get('writes_completed', 0) < 1 or result.get('writes_failed', 0) != 0):
        raise RuntimeError('memory extraction did not complete successfully')


def wait_for_recall(post, marker, timeout=120, interval=2):
    # Querying a fresh session verifies indexing; reading content verifies AGFS.
    deadline = time.monotonic() + timeout
    session_id = 'compose-read-' + uuid4().hex
    while time.monotonic() < deadline:
        result = post('memories/query', {'sessionId': session_id,
                      'query': marker + ' 项目使用什么语言？', 'top_k': 20})
        for hit in result.get('hits', []):
            memory = post('memories/read', {'sessionId': session_id, 'uri': hit['uri'], 'level': 'content'})
            content = memory.get('content', '')
            if marker in content and 'Python' in content:
                return hit['uri']
        time.sleep(interval)
    raise RuntimeError('persisted atomic memory was not recalled before timeout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='write a new fictional memory, then query it')
    parser.add_argument('--marker', help='marker printed by the initial write; reuse for read-only persistence checks')
    args = parser.parse_args()
    if not args.write and not args.marker:
        parser.error('read-only checks require --marker from a previous write')
    marker = args.marker or ('青柚部署验收' + uuid4().hex[:12])
    print(json.dumps({'marker': marker}, ensure_ascii=False), flush=True)
    logging.disable(logging.CRITICAL)
    try:
        from providers.unified_config import OgMemConfig
        cfg = OgMemConfig.load(config_path())
        headers = {'Content-Type': 'application/json', 'X-API-Key': os.environ['OG_TENANT_API_KEY'],
                   'X-Account-ID': cfg.account_id, 'X-User-ID': cfg.user_id, 'X-Agent-ID': cfg.agent_id}
        identity = {'accountId': cfg.account_id, 'userId': cfg.user_id, 'agentId': cfg.agent_id}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(route, payload):
            request = urllib.request.Request(f'http://127.0.0.1:{cfg.http_port}/api/v1/{route}',
                                             json.dumps({**identity, **payload}).encode(), headers)
            with opener.open(request, timeout=650) as response:
                return json.load(response)

        if args.write:
            session_id = 'compose-write-' + uuid4().hex
            result = post('after_turn', {'sessionId': session_id,
                'messages': [{'role': 'user', 'content': f'我的测试项目“{marker}”使用Python开发，每周四16:20复盘。'},
                             {'role': 'assistant', 'content': '已记录这个测试项目的安排。'}],
                'forceExtract': True, 'wait': True, 'timeoutSeconds': 600,
                'clientRequestId': 'compose-smoke-' + uuid4().hex})
            verify_write(result)
            print(json.dumps({key: result.get(key) for key in
                              ('status', 'writes_completed', 'writes_failed', 'retryable')}), flush=True)
        uri = wait_for_recall(post, marker)
        print(json.dumps({'recalled': True, 'uri': uri, 'marker': marker}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error_type': type(exc).__name__, 'marker': marker}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
