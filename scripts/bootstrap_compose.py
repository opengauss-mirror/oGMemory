#!/usr/bin/env python3
"""Register the configured user and owned Agent once, without printing keys."""
import json
import logging
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_deployment import config_path


def ensure_identity(call, account, user, agent):
    base = 'admin/accounts/' + urllib.parse.quote(account, safe='')
    users = call('GET', base + '/users')['users']
    agents = call('GET', base + '/agents')['agents']
    existing = next((row for row in agents if row['agent_id'] == agent), None)
    if existing and existing.get('owner_user_id') != user:
        raise ValueError('configured Agent has a different owner')
    if not any(row['user_id'] == user for row in users):
        call('POST', base + '/users', {'user_id': user, 'role': 'user'})
    if not existing:
        call('POST', base + '/agents', {'agent_id': agent, 'owner_user_id': user})


def main():
    logging.disable(logging.CRITICAL)
    try:
        from providers.unified_config import OgMemConfig
        cfg = OgMemConfig.load(config_path())
        headers = {'Content-Type': 'application/json', 'X-API-Key': os.environ['OG_ROOT_API_KEY'],
                   'X-Account-ID': cfg.account_id, 'X-User-ID': cfg.user_id}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def call(method, route, payload=None):
            body = json.dumps(payload).encode() if payload is not None else None
            request = urllib.request.Request(f'http://127.0.0.1:{cfg.http_port}/api/v1/{route}',
                                             body, headers, method=method)
            with opener.open(request, timeout=15) as response:
                return json.load(response)
        ensure_identity(call, cfg.account_id, cfg.user_id, cfg.agent_id)
        print(json.dumps({'identity_ready': True}))
        return 0
    except Exception as exc:
        print(json.dumps({'identity_ready': False, 'error_type': type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
