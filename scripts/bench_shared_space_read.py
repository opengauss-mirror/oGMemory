"""Measure SQL key lookup and authorized shared-memory reads.

Run against an existing disposable SQL schema:
  $env:OGMEM_SPACE_TEST_DSN = '<test database DSN>'
  python scripts/bench_shared_space_read.py --output <directory>

Each run seeds an isolated tenant and writes config, request JSONL, and summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import secrets
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg2.extras import Json, execute_values  # noqa: E402

from core.models import RequestContext, Role  # noqa: E402
from fs.agfs_adapter.space_bridge import AGFSSpaceBridge  # noqa: E402
from fs.sql_adapter.pool import SharedConnectionPool  # noqa: E402
from server.api_keys import APIKeyManager  # noqa: E402
from server.space_repository import SpaceRepository  # noqa: E402
from server.space_rules import digest, validate_rule  # noqa: E402
from server.sql_control_plane_store import SQLControlPlaneStore  # noqa: E402

_timings = ContextVar('shared_read_timings', default=None)


@contextmanager
def measure(stage):
    timings = _timings.get()
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] += (time.perf_counter() - start) * 1000


class TimedSpaceRepository(SpaceRepository):
    @contextmanager
    def transaction(self, *args, **kwargs):
        with measure('sql_authorization_ms'), super().transaction(*args, **kwargs) as cur:
            yield cur


class TimedAGFSSpaceBridge(AGFSSpaceBridge):
    def load_content(self, rows):
        with measure('agfs_body_read_ms'):
            return super().load_content(rows)


class ExistingSchemaStore(SQLControlPlaneStore):
    def _ensure_table(self):
        pass


def environment(repository):
    machine = {
        'os': platform.platform(), 'python': platform.python_version(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
    }
    if os.name == 'nt':
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '$cpu=Get-CimInstance Win32_Processor; $machine=Get-CimInstance Win32_ComputerSystem; '
             '@{cpu=$cpu.Name; cores=$cpu.NumberOfCores; logical_cpus=$cpu.NumberOfLogicalProcessors; '
             'ram_bytes=$machine.TotalPhysicalMemory}|ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=15, check=True,
        )
        machine.update(json.loads(result.stdout))
    conn = repository.pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version(),current_setting('shared_buffers'),current_setting('max_connections')")
            version, shared_buffers, max_connections = cur.fetchone()
        conn.rollback()
    finally:
        repository.pool.return_connection(conn)
    return {'client_machine': machine, 'database': {'version': version,
            'shared_buffers': shared_buffers, 'max_connections': int(max_connections)}}


def seed(repository, account, count):
    """Populate one registered reader, one source owner, and paired source/items."""
    api_key = secrets.token_hex(24)
    spec = validate_rule({'memory_types': ['event'],
                          'description': 'Reusable troubleshooting steps.'})
    with repository.transaction(account) as cur:
        batch_size = 50 if repository.pool.dialect == 'opengauss' else 1000
        cur.execute("INSERT INTO cp_accounts(account_id,created_at,status) VALUES(%s,NOW(),'active')", (account,))
        execute_values(cur, "INSERT INTO cp_users(account_id,user_id,role,key,created_at,status) VALUES %s",
                       [(account, 'alice', 'user', secrets.token_hex(24), datetime.now(UTC), 'active'),
                        (account, 'bob', 'user', api_key, datetime.now(UTC), 'active')])
        cur.execute("INSERT INTO spaces(account_id,space_id,name,owner_user_id) VALUES(%s,'perf','Read benchmark','alice')", (account,))
        execute_values(cur, 'INSERT INTO space_members(account_id,space_id,principal,user_id,role) VALUES %s',
                       [(account, 'perf', 'user:alice', 'alice', 'manager'),
                        (account, 'perf', 'user:bob', 'bob', 'member')])
        cur.execute("INSERT INTO space_source_bindings(account_id,space_id,binding_id,principal,memory_types,authorized_by) VALUES(%s,'perf','alice','user:alice',%s,'alice')", (account, Json(['event'])))
        cur.execute("INSERT INTO space_rules(account_id,space_id,rule_id,name,state) VALUES(%s,'perf','fixture','Fixture','PAUSED')", (account,))
        cur.execute("INSERT INTO space_rule_revisions(account_id,space_id,rule_id,rule_rev,name,state,spec,spec_hash) VALUES(%s,'perf','fixture',1,'Fixture','PAUSED',%s,%s)", (account, Json(spec), digest(spec)))
        source_rows, shared_rows, item_rows, acl_rows = [], [], [], []
        for index in range(count):
            identity = f'item-{index:06d}'
            source_uri = f'ctx://{account}/users/alice/memories/events/{identity}'
            shared_uri = f'ctx://{account}/spaces/perf/memories/experience/{identity}'
            content = f'Database connection pool troubleshooting case {index}. ' + 'Check capacity, inspect waits, validate recovery. ' * 4
            source_rows.append((source_uri, account, 'user:alice', 'event', 'MEMORY', content, 'Source case', Json({})))
            shared_rows.append((shared_uri, account, 'space:perf', 'experience', 'MEMORY', content, 'Reusable case',
                                Json({'shared_projection_ready': True, 'indexed_content_rev': 1})))
            item_rows.append((account, 'perf', identity, 'fixture', 1, shared_uri, 'experience',
                              'restricted' if index % 2 else 'inherit', 0.9,
                              Json(spec['maintenance_policy']), identity, hashlib.sha256(identity.encode()).hexdigest()))
            if index % 2:
                acl_rows.append((account, 'perf', identity, 'user:bob'))
        insert_nodes = 'INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content,abstract,metadata) VALUES %s'
        if repository.content_store:
            def persist(row):
                metadata = dict(row[-1].adapted)
                metadata.update(repository.content_store.store_shared(row[0], row[5], row[6], '', metadata))
                return (*row[:-1], Json(metadata))
            with ThreadPoolExecutor(max_workers=8) as executor:
                source_rows = list(executor.map(persist, source_rows))
                shared_rows = list(executor.map(persist, shared_rows))
        execute_values(cur, insert_nodes, source_rows, page_size=batch_size)
        execute_values(cur, insert_nodes, shared_rows, page_size=batch_size)
        execute_values(cur, "INSERT INTO space_items(account_id,space_id,item_id,rule_id,rule_rev,uri,kind,acl_mode,quality_score,policy_snapshot,topic_key,evidence_fingerprint,expires_at) VALUES %s",
                       item_rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()+INTERVAL '365 days')", page_size=batch_size)
        cur.execute("SELECT uri,version,metadata->>'shared_source_epoch' AS epoch,content FROM context_nodes WHERE account_id=%s AND owner_space='user:alice'", (account,))
        sources = cur.fetchall()
        ledger = [(account, 'perf', row['uri'].rsplit('/', 1)[-1], 'alice', row['uri'], 1,
                   row['version'], hashlib.sha256(row['content'].encode()).hexdigest(), row['epoch'] or '') for row in sources]
        execute_values(cur, 'INSERT INTO space_item_sources(account_id,space_id,item_id,binding_id,source_uri,grant_rev,source_content_rev,source_hash,source_epoch) VALUES %s', ledger, page_size=batch_size)
        execute_values(cur, 'INSERT INTO space_read_acl(account_id,space_id,item_id,principal) VALUES %s', acl_rows, page_size=batch_size)
    conn = repository.pool.get_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in ('context_nodes', 'space_items', 'space_item_sources', 'space_members', 'space_read_acl', 'cp_users'):
                cur.execute('ANALYZE ' + table)
    finally:
        conn.autocommit = False
        repository.pool.return_connection(conn)
    return api_key, len(source_rows[0][5].encode())


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def run(args):
    connection = json.loads(Path(args.config).read_text(encoding='utf8')) if args.config else {}
    dsn = connection.get('dsn') or os.environ['OGMEM_SPACE_TEST_DSN']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    account = args.account or 'space-perf-' + run_id.lower()
    pool = SharedConnectionPool(dsn, pool_size=args.concurrency, dialect='auto')
    content_store = None
    if connection.get('agfs_url'):
        from pyagfs import AGFSClient

        content_store = TimedAGFSSpaceBridge(pool, AGFSClient(api_base_url=connection['agfs_url']),
                                            connection.get('agfs_mount_prefix', '/local'))
    repository = TimedSpaceRepository(pool, initialize=False, content_store=content_store)
    key_manager = APIKeyManager(ExistingSchemaStore(pool=pool))
    seeded_at = time.perf_counter()
    api_key, content_bytes = seed(repository, account, args.items)
    seed_seconds = time.perf_counter() - seeded_at
    config = {
        'run_id': run_id, 'account_id': account, 'started_at': datetime.now(UTC).isoformat(),
        'workload': {'shared_items': args.items, 'source_nodes': args.items, 'sources_per_item': 1,
                     'restricted_acl_fraction': 0.5, 'candidate_count': args.candidates,
                     'content_bytes_first_item': content_bytes, 'requests': args.requests,
                     'concurrency': args.concurrency, 'pool_size': args.concurrency,
                     'warmup_requests': args.warmup, 'seed_seconds': seed_seconds,
                     'sampling': 'deterministic stride, batch of consecutive candidate URIs',
                     'cache': 'warm process and database caches; no explicit cache flush',
                     'index_state': 'fixture records marked READY; indexed_content_rev=1',
                     'identity': 'registered ordinary user bob; user-level request',
                     'rule_state': 'PAUSED fixture; no benchmark background workers',
                     'scope': 'in-process KeyManager SQL key lookup then SpaceRepository.visible_items',
                     'timing': 'perf_counter at worker start; total=key_lookup+authorized_read+residual; '
                               'authorized_read=sql_authorization+agfs_body_read+read_processing; '
                               'sql_authorization includes initial and post-hydration transactions; '
                               'all stages include failed-call elapsed time',
                     'load_model': 'closed loop with concurrency-sized thread pool',
                     'included': 'pool validation/wait, SQL/driver calls, Python result materialization',
                     'data_retention': 'isolated benchmark tenant retained in disposable test database'},
        **environment(repository),
    }
    if content_store:
        config['workload']['scope'] += ' with AGFS hydration and authorization/version recheck'
        config['workload']['included'] += ', AGFS HTTP reads, content hash verification, and current-policy redaction'
        config['storage'] = {'metadata': 'openGauss', 'body': 'AGFS', 'read_concurrency': 8}
    config['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'scripts/bench_shared_space_read.py', 'server/space_repository.py', 'server/api_keys.py',
        'server/sql_control_plane_store.py', 'fs/sql_adapter/pool.py', 'fs/agfs_adapter/space_bridge.py')}
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    def request(index, measured=True):
        request_id = f'{run_id}-{"read" if measured else "warmup"}-{index:04d}'
        start = time.perf_counter()
        timestamp = datetime.now(UTC).isoformat()
        stages = dict.fromkeys(('key_lookup_ms', 'authorized_read_ms', 'sql_authorization_ms', 'agfs_body_read_ms'), 0.0)
        token = _timings.set(stages)
        count = 0
        error = None
        try:
            with measure('key_lookup_ms'):
                resolved = key_manager.resolve_user_key(api_key, account)
            if resolved != {'account_id': account, 'user_id': 'bob', 'role': 'user'}:
                raise RuntimeError('Fixture identity mismatch')
            offset = index * 7919 % args.items
            candidates = [f'ctx://{account}/spaces/perf/memories/experience/item-{(offset + number) % args.items:06d}' for number in range(args.candidates)]
            ctx = RequestContext(account, 'bob', '', 'benchmark', request_id, role=Role.MEMBER)
            with measure('authorized_read_ms'):
                rows = repository.visible_items(ctx, ['perf'], uris=candidates, limit=args.candidates)
            count = len(rows)
            if count != args.candidates or {row['uri'] for row in rows} != set(candidates):
                raise RuntimeError('Fixture result mismatch')
            if any('Database connection pool troubleshooting case' not in row['content'] for row in rows):
                raise RuntimeError('Fixture content mismatch')
        except Exception as exc:
            error = type(exc).__name__
        finally:
            _timings.reset(token)
        end = time.perf_counter()
        stages['read_processing_ms'] = stages['authorized_read_ms'] - stages['sql_authorization_ms'] - stages['agfs_body_read_ms']
        stages['residual_ms'] = (end - start) * 1000 - stages['key_lookup_ms'] - stages['authorized_read_ms']
        return {'request_id': request_id, 'timestamp': timestamp, 'operation': 'SQL_AUTHORIZED_SHARED_READ',
                'workload_index': index, 'total_ms': (end - start) * 1000,
                'stages': stages, 'returned_count': count, 'expected_count': args.candidates,
                'error': error}
    # Establish every pool connection before the measured burst.
    connections = [pool.get_connection() for _ in range(args.concurrency)]
    for connection in connections:
        pool.return_connection(connection)
    for index in range(args.warmup):
        result = request(index, measured=False)
        if result['error']:
            raise RuntimeError(f"Warmup failed: {result['error']}")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(request, range(args.requests)))
    wall_seconds = time.perf_counter() - start
    if content_store:
        content_store.close()
    pool.close_all()
    (output / 'requests.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf8')
    metrics = {}
    for name in ('total_ms', *rows[0]['stages']):
        values = [row['total_ms'] if name == 'total_ms' else row['stages'][name] for row in rows]
        metrics[name] = {'count': len(values), 'mean': statistics.mean(values), 'p50': percentile(values, 0.50),
                         'p95': percentile(values, 0.95), 'p99': percentile(values, 0.99), 'max': max(values)}
    summary = {'run_id': run_id, 'measured_requests': len(rows), 'errors': sum(bool(row['error']) for row in rows),
               'wall_seconds': wall_seconds, 'throughput_requests_per_second': len(rows) / wall_seconds,
               'successful_items': sum(row['returned_count'] for row in rows), 'metrics': metrics,
               'scope': config['workload']['scope']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary['errors'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Private JSON connection configuration for openGauss and AGFS')
    parser.add_argument('--account', help='Fresh tenant ID; also reuses AGFS versions after an interrupted fixture load')
    parser.add_argument('--items', type=int, default=10000)
    parser.add_argument('--candidates', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--requests', type=int, default=400)
    parser.add_argument('--warmup', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.candidates <= min(args.items, 2000) or not 1 <= args.concurrency <= 64 or not 1 <= args.items <= 100000 or not 1 <= args.requests <= 10000:
        parser.error('Use bounded positive workload values')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
