#!/usr/bin/env python3
"""Mask known deployment credentials before displaying container logs."""
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, unquote

import yaml
from psycopg2.extensions import parse_dsn


def redact(text, path):
    cfg = yaml.safe_load(Path(path).read_text())
    secrets = set()
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('api_key', 'root_api_key') and isinstance(item, str):secrets.add(item)
                elif key == 'admin_api_keys':
                    for token in item:secrets.add(token.split(':', 1)[-1])
                elif key == 'connection_string' and isinstance(item, str):
                    password = parse_dsn(item).get('password')
                    if password:secrets.add(password)
                elif key == 'base_url' and isinstance(item, str):
                    password = urlsplit(item).password
                    if password:secrets.add(password)
                else:walk(item)
        elif isinstance(value, list):
            for item in value:walk(item)
    walk(cfg)
    for name in ('DEPLOY_HTTP_PROXY', 'DEPLOY_HTTPS_PROXY'):
        proxy = os.environ.get(name, '')
        if proxy:
            secrets.add(proxy)
            password = urlsplit(proxy).password
            if password:
                secrets.update((password, unquote(password)))
    for secret in sorted((x for x in secrets if x), key=len, reverse=True):
        text = text.replace(secret, '[REDACTED]')
    return text


if __name__ == '__main__':
    try:sys.stdout.write(redact(sys.stdin.read(), sys.argv[1]))
    except Exception as exc:
        print(json.dumps({'logs_redacted': False, 'error_type': type(exc).__name__}));sys.exit(1)
