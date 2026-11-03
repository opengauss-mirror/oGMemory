#!/usr/bin/env python3
"""Resolve standalone configuration and own every container child process."""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

import yaml
from psycopg2.extensions import make_dsn


def resolve_config(source: Path, target: Path) -> None:
    # Substitute after parsing: quotes/newlines in credentials must stay scalar data.
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError('configuration must be a YAML mapping')

    def expand(value):
        if isinstance(value, str):
            return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m[1], m[0]), value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    raw = expand(raw)
    keys = ('HOST', 'PORT', 'DBNAME', 'USER', 'PASSWORD')
    if any(os.environ.get('OPENGAUSS_' + key) for key in keys):
        if not isinstance(raw.get('vector_db'), dict):
            raise ValueError('database overrides require vector_db configuration')
        raw['vector_db']['connection_string'] = make_dsn(
            host=os.environ.get('OPENGAUSS_HOST', '127.0.0.1'),
            port=os.environ.get('OPENGAUSS_PORT', '5432'),
            dbname=os.environ.get('OPENGAUSS_DBNAME', 'postgres'),
            user=os.environ.get('OPENGAUSS_USER', 'gaussdb'),
            password=os.environ.get('OPENGAUSS_PASSWORD', ''),
        )
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        yaml.safe_dump(raw, stream, allow_unicode=True)


class Supervisor:
    def __init__(self, shutdown_timeout: float = 20):
        self.children: dict[str, subprocess.Popen] = {}
        self.stopping = False
        self.shutdown_timeout = shutdown_timeout

    def start(self, name: str, command: list[str]):
        print(f'[entrypoint] Starting {name}', flush=True)
        child = subprocess.Popen(command, start_new_session=True)
        self.children[name] = child
        return child

    def request_stop(self, *_):
        self.stopping = True

    def failed_child(self):
        return next((name for name, child in self.children.items() if child.poll() is not None), None)

    def wait_ready(self, url: str, timeout: float = 30):
        deadline = time.monotonic() + timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline and not self.stopping:
            if self.failed_child():
                raise RuntimeError('AGFS exited before becoming ready')
            try:
                with opener.open(url, timeout=min(1, max(.01, deadline - time.monotonic()))) as response:
                    if response.status == 200:
                        return
            except (OSError, ValueError):
                pass
            time.sleep(.1)
        raise RuntimeError('AGFS did not become ready before startup ended')

    def stop(self):
        # Keep storage alive while HTTP/MCP and the index runner drain their work.
        # Both phases share one deadline inside Compose's stop_grace_period.
        deadline = time.monotonic() + self.shutdown_timeout

        def stop_group(children):
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for child in children:
                try:
                    child.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            # Clean up remaining workers too, even if their parent already exited.
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()

        stop_group([child for name, child in self.children.items() if name != 'AGFS'])
        stop_group([child for name, child in self.children.items() if name == 'AGFS'])

    def monitor(self):
        try:
            while not self.stopping:
                failed = self.failed_child()
                if failed:
                    print(f'[entrypoint] Required process exited: {failed}', flush=True)
                    return 1
                time.sleep(.1)
            return 0
        finally:
            self.stop()


def main():
    supervisor = Supervisor()
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    try:
        resolved = Path('/tmp/ogmem.resolved.yaml')
        resolve_config(Path(os.environ.get('OGMEM_CONFIG', '/etc/ogmem/config.yaml')), resolved)
        os.environ['OGMEM_CONFIG'] = str(resolved)
        from providers.unified_config import OgMemConfig
        cfg = OgMemConfig.load(str(resolved))
        # This entrypoint includes AGFS and its standalone index runner.
        if cfg.storage_backend != 'agfs' or cfg.retrieval_hybrid_enabled:
            raise ValueError('standalone runtime requires AGFS storage without SQL hybrid retrieval')
        Path('/data/agfs').mkdir(parents=True, exist_ok=True)
        os.chdir('/opt/ogmem')
        supervisor.start('AGFS', ['/opt/agfs/agfs-server', '-c', '/opt/agfs/config.yaml'])
        supervisor.wait_ready(cfg.agfs_base_url.rstrip('/') + '/api/v1/files?path=/serverinfo/version')
        supervisor.start('IndexService', [sys.executable, '/opt/ogmem/scripts/run_index_service.py'])
        if cfg.mcp_enabled:
            supervisor.start('MCP', [sys.executable, '-m', 'cli.main', 'mcp',
                                    '--transport', 'streamable-http', '--host', cfg.mcp_host,
                                    '--port', str(cfg.mcp_port), '--path', cfg.mcp_path])
        supervisor.start('HTTP', [sys.executable, '-m', 'gunicorn', '--config', 'python:server.gunicorn_config',
                                 '--bind', f'0.0.0.0:{cfg.http_port}',
                                 '--workers', '1', '--worker-class', 'gthread', '--threads', str(cfg.threads),
                                 '--timeout', os.environ.get('OGMEM_GUNICORN_TIMEOUT', '900'),
                                 '--access-logfile', '-', 'server.app:app'])
        return supervisor.monitor()
    except Exception as exc:
        # Parser/provider exceptions may contain credentials: report the class only.
        print(f'[entrypoint] Startup failed ({type(exc).__name__}); check config, data permissions and process logs',
              file=sys.stderr, flush=True)
        supervisor.stop()
        return 0 if supervisor.stopping else 1


if __name__ == '__main__':
    raise SystemExit(main())
