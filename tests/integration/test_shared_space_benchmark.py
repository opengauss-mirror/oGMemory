"""Benchmark accounting through real SQL and AGFS reads."""
import contextlib
import io
import json
import os
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from scripts import bench_shared_space_read as bench
from server import space_repository


@pytest.fixture
def workload(tmp_path):
    config_path = os.environ.get('OGMEM_SPACE_TEST_CONFIG')
    if not config_path:
        pytest.skip('OGMEM_SPACE_TEST_CONFIG selects an isolated SQL and AGFS deployment')
    config = json.loads(Path(config_path).read_text())

    def run(storage='agfs', concurrency=1, requests=2):
        selected = dict(config)
        if storage == 'sql':
            selected.pop('agfs_url', None)
        path = tmp_path / 'connection.json'
        path.write_text(json.dumps(selected))
        args = Namespace(config=str(path), output=str(tmp_path / 'output'), account=None,
                         items=4, candidates=2, concurrency=concurrency, requests=requests, warmup=0)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status = bench.run(args)
        finally:
            path.unlink()
        output = Path(args.output)
        rows = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
        summary = json.loads((output / 'summary.json').read_text())
        print(json.dumps({'request': {'storage': storage, 'concurrency': concurrency,
                                     'requests': requests, 'candidates': 2},
                          'exit_code': status, 'responses': rows, 'errors': summary['errors']},
                         ensure_ascii=False))
        for row in rows:
            stages = row['stages']
            assert row['total_ms'] == pytest.approx(
                stages['key_lookup_ms'] + stages['authorized_read_ms'] + stages['residual_ms'])
        return status, rows, summary

    return run


@pytest.mark.parametrize('storage,concurrency', [('sql', 1), ('agfs', 1), ('agfs', 3)])
def test_benchmark_measures_complete_read(workload, monkeypatch, storage, concurrency):
    queries = Counter()
    redactions = []
    visible = bench.SpaceRepository._visible_rows
    redact = space_repository.redact_item

    def observed(self, cur, ctx, *args, **kwargs):
        queries[ctx.trace_id] += 1
        if queries[ctx.trace_id] == 2:
            time.sleep(0.08)
        return visible(self, cur, ctx, *args, **kwargs)

    def observed_redaction(row, policy):
        redactions.append(row['uri'])
        return redact(row, policy)

    monkeypatch.setattr(bench.SpaceRepository, '_visible_rows', observed)
    monkeypatch.setattr(space_repository, 'redact_item', observed_redaction)
    status, rows, summary = workload(storage, concurrency, requests=6)
    print(json.dumps({'authorization_queries': dict(queries), 'redacted_items': len(redactions)}))
    assert status == 0
    assert summary['successful_items'] == 12
    assert len(redactions) == 12
    for row in rows:
        assert queries[row['request_id']] == (2 if storage == 'agfs' else 1)
        stages = row['stages']
        assert stages['authorized_read_ms'] == pytest.approx(
            stages['sql_authorization_ms'] + stages['agfs_body_read_ms'] + stages.get('read_processing_ms', 0))
        if storage == 'agfs':
            assert stages['sql_authorization_ms'] >= 80
            assert stages['agfs_body_read_ms'] > 0


@pytest.mark.parametrize('operation,metric', [
    ('key', 'key_lookup_ms'), ('sql', 'sql_authorization_ms'),
    ('recheck', 'sql_authorization_ms'), ('agfs', 'agfs_body_read_ms'),
])
def test_benchmark_accounts_stage_wait_and_recovers(workload, monkeypatch, operation, metric):
    owner, name = {
        'key': (bench.APIKeyManager, 'resolve_user_key'),
        'sql': (bench.SpaceRepository, '_visible_rows'),
        'recheck': (bench.SpaceRepository, '_visible_rows'),
        'agfs': (AGFSSpaceBridge, 'load_content'),
    }[operation]
    original = getattr(owner, name)
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (2 if operation == 'recheck' else 1):
            time.sleep(0.08)
            raise TimeoutError('Controlled stage timeout')
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, interrupted)
    status, rows, summary = workload()
    print(json.dumps({'injected_stage': operation, 'wait_ms': 80, 'attributed_ms': rows[0]['stages'][metric]}))
    assert status == 1
    assert summary['errors'] == 1
    assert rows[0]['error'] == 'TimeoutError'
    assert rows[0]['stages'][metric] >= 80
    assert rows[1]['returned_count'] == 2
    assert summary['successful_items'] == 2
