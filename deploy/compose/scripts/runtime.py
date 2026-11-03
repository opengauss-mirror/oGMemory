#!/usr/bin/env python3
"""Service readiness, identity initialization and connection information."""
import argparse
import json
import logging
import os
from pathlib import Path
import socket
import urllib.parse
import urllib.request
import yaml


def run_checks(checks):
    result = {}
    for name, check in checks.items():
        try:
            check()
            result[name] = {'ok': True}
        except Exception as exc:
            result[name] = {'ok': False, 'error_type': type(exc).__name__}
    return result


def config_path():
    path = Path('/tmp/ogmem.resolved.yaml')
    if not path.exists(): path = Path(os.environ.get('OGMEM_CONFIG', '/etc/ogmem/config.yaml'))
    return path


def config():
    return yaml.safe_load(config_path().read_text())


def call(cfg, method, route, payload=None, root=False, timeout=20):
    identity = cfg['identity']
    headers = {'Content-Type': 'application/json',
               'X-API-Key': os.environ['OG_ROOT_API_KEY' if root else 'OG_TENANT_API_KEY'],
               'X-Account-ID': identity['account_id'], 'X-User-ID': identity['user_id'],
               'X-Agent-ID': identity['agent_id']}
    if payload is not None:
        payload = dict(accountId=identity['account_id'], userId=identity['user_id'],
                       agentId=identity['agent_id'], **payload)
    request = urllib.request.Request('http://127.0.0.1:8090/api/v1/' + route,
                                      None if payload is None else json.dumps(payload).encode(), headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def local_checks(cfg):
    def http():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:8090/api/v1/health', timeout=3) as response:
            value = json.load(response)
            if value.get('status') != 'ok' or value.get('agfs') is not True:
                raise RuntimeError('HTTP/AGFS is not ready')
    def database():
        import psycopg2
        with psycopg2.connect(cfg['vector_db']['connection_string'], connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute('SET statement_timeout = 3000')
                cur.execute("SELECT '[1,0]'::vector <=> '[1,0]'::vector")
                if cur.fetchone()[0] != 0: raise RuntimeError('vector operator unavailable')
    def index():
        if not any(b'run_index_service.py' in p.read_bytes() for p in Path('/proc').glob('[0-9]*/cmdline') if p.exists()):
            raise RuntimeError('index process is not running')
    def mcp():
        with socket.create_connection(('127.0.0.1', 8091), timeout=3):pass
    checks = {'http_agfs': http, 'database_vector': database, 'index_process': index}
    if cfg['mcp']['enabled']: checks['mcp_listener'] = mcp
    return run_checks(checks)


def bootstrap(cfg):
    identity = cfg['identity']
    prefix = 'admin/accounts/' + urllib.parse.quote(identity['account_id'], safe='')
    users = call(cfg, 'GET', prefix + '/users', root=True)['users']
    agents = call(cfg, 'GET', prefix + '/agents', root=True)['agents']
    agent = next((a for a in agents if a['agent_id'] == identity['agent_id']), None)
    if agent and agent.get('owner_user_id') != identity['user_id']:
        raise ValueError('Agent belongs to another user')
    if not any(u['user_id'] == identity['user_id'] for u in users):
        call(cfg, 'POST', prefix + '/users', {'user_id': identity['user_id'], 'role': 'user'}, root=True)
    if not agent:
        call(cfg, 'POST', prefix + '/agents', {'agent_id': identity['agent_id'], 'owner_user_id': identity['user_id']}, root=True)
    return {'identity_ready': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['health', 'bootstrap', 'connection'])
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        cfg = config()
        if args.action == 'health':
            result = local_checks(cfg)
            print(json.dumps(result))
            return 0 if all(item['ok'] for item in result.values()) else 1
        if args.action == 'bootstrap':
            result = bootstrap(cfg)
        else:
            result = {'identity': cfg['identity'], 'http_container_port':8090, 'mcp_container_port':8091,
                      'api_key_location':'DATA_ROOT/COMPOSE_PROJECT_NAME/secrets.env: OG_TENANT_API_KEY'}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({'ok':False, 'error_type':type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
