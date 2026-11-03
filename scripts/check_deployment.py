#!/usr/bin/env python3
"""Read-only local readiness; optional explicit model-provider smoke calls."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_checks(checks):
    results = {}
    for name, check in checks.items():
        try:
            check()
            results[name] = {'ok': True}
        except Exception as exc:
            # DSNs, request URLs and provider errors may contain secrets.
            results[name] = {'ok': False, 'error_type': type(exc).__name__}
    return results


def validate_embedding(vectors, dimension):
    if len(vectors) != 1 or len(vectors[0]) != dimension:
        raise ValueError('embedding dimension mismatch')
    if not all(math.isfinite(value) for value in vectors[0]) or not any(vectors[0]):
        raise ValueError('embedding must contain finite, nonzero values')


def check_http(cfg):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{cfg.http_port}/api/v1/health', timeout=3) as response:
        if json.load(response).get('status') != 'ok':
            raise RuntimeError('HTTP is not ready')


def check_database(cfg):
    import psycopg2
    # A local socket probe alone cannot verify DB authentication or vector support.
    with psycopg2.connect(cfg.opengauss_connection_string, connect_timeout=3) as conn:
        with conn.cursor() as cursor:
            cursor.execute('SET statement_timeout = 3000')
            cursor.execute("SELECT '[1,0]'::vector <=> '[1,0]'::vector")
            if cursor.fetchone()[0] != 0:
                raise RuntimeError('vector operator unavailable')


def check_mcp(cfg):
    with socket.create_connection(('127.0.0.1', cfg.mcp_port), timeout=3):
        pass


def check_models(cfg):
    from core.operation_deadline import operation_deadline
    providers = cfg.to_provider_config()
    with operation_deadline(30):
        result = providers.create_llm().complete_json(
            'Deployment connectivity check. Return exactly {"ok":true}.',
            {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']},
        )
        if result.get('ok') is not True:
            raise ValueError('LLM did not return the expected JSON')
    validate_embedding(providers.create_embedder().embed_texts(['deployment connectivity check']),
                       cfg.opengauss_dimension)


def config_path(resolved=Path("/tmp/ogmem.resolved.yaml")):
    return str(resolved) if resolved.is_file() else os.environ.get("OGMEM_CONFIG", "/etc/ogmem/config.yaml")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true', help='local checks only (the default)')
    parser.add_argument('--models', action='store_true', help='also call LLM and Embedding; may incur provider charges')
    args = parser.parse_args()
    # Do not emit dependency exception text or SDK HTTP logs containing URLs/keys.
    logging.disable(logging.CRITICAL)
    try:
        from providers.unified_config import OgMemConfig
        os.environ["OGMEM_CONFIG"] = config_path()
        cfg = OgMemConfig.load(os.environ["OGMEM_CONFIG"])
        checks = {'http_agfs': lambda: check_http(cfg), 'database_vector': lambda: check_database(cfg)}
        if cfg.mcp_enabled:
            checks['mcp_listener'] = lambda: check_mcp(cfg)
        if args.models:
            checks['models'] = lambda: check_models(cfg)
        result = run_checks(checks)
    except Exception as exc:
        result = {'config': {'ok': False, 'error_type': type(exc).__name__}}
    print(json.dumps(result))
    return 0 if all(item['ok'] for item in result.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
