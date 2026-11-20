"""Durable job regressions shared by all migration operations."""
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from service.migration.jobs import InstanceJobs
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.package import PackageError


@pytest.fixture
def queue(tmp_path):
    migration = SimpleNamespace(workspace=tmp_path, gate=MaintenanceStore(tmp_path / 'gate.db', 'test'),
                                cfg=SimpleNamespace(root_api_key='test-secret'))
    return InstanceJobs(migration)


def wait(queue, identity):
    for _ in range(500):
        state = queue.get(identity)
        if state['status'] != 'running':
            return state
        time.sleep(.01)
    pytest.fail('job did not finish')


def test_upload_bound_removes_partial_file(queue, monkeypatch):
    monkeypatch.setattr('service.migration.jobs.MAX_UPLOAD_BYTES', 8)
    with pytest.raises(PackageError, match='upload limit'):
        queue.upload(io.BytesIO(b'x' * 9))
    assert not list(queue.root.glob('upload-*'))


def test_execution_lock_shared_by_queue_instances(queue):
    from service.migration.file_lock import exclusive_file
    other = InstanceJobs(queue.migration)
    with exclusive_file(queue.root / 'execution.lock'):
        with pytest.raises(MaintenanceError, match='another migration'):
            other.submit('export', identity=str(uuid.uuid4()))
    queue.migration.export = lambda *a, **kw: {'ok': True}
    result = queue.submit('export', identity=str(uuid.uuid4()))
    assert wait(queue, result['id'])['status'] == 'succeeded'


def test_unicode_report_survives_restart_and_polling(queue):
    queue.migration.preview = lambda source: {'model': '本地模型🌍', 'size': source.stat().st_size}
    artifact = queue.upload(io.BytesIO(b'package'))['artifact_id']
    job = queue.submit('preflight', identity=str(uuid.uuid4()), artifact=artifact)
    state = wait(queue, job['id'])
    assert state['result']['model'] == '本地模型🌍'
    assert InstanceJobs(queue.migration).get(job['id']) == state


def test_job_progress_can_be_polled_during_updates(queue):
    job = dict(id=str(uuid.uuid4()), kind='export', status='running', count=0, created_at=time.time())
    queue._save(job)
    def write():
        for i in range(100):
            job['count'] = i
            queue._save(job)
    def poll():
        for _ in range(100):
            assert 0 <= queue.get(job['id'])['count'] < 100
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(write), *(pool.submit(poll) for _ in range(3))]
        for future in futures:
            future.result(timeout=30)
    assert queue.get(job['id'])['count'] == 99


@pytest.mark.parametrize('kind', ['export', 'preflight', 'import'])
def test_failure_persists_redacted_cause_and_call_sites(queue, kind, caplog):
    cfg = queue.migration.cfg
    cfg.openai_embedding_api_key = 'embedding-credential'
    cfg.sql_connection_string = "host=localhost dbname=test password='db secret'"
    def fail(*a, **kw):
        try:
            raise ConnectionError("connection refused embedding-credential db secret test-secret password='labelled-secret'")
        except ConnectionError as cause:
            raise RuntimeError('embedding connection failed') from cause
    queue.migration.export = queue.migration.preview = queue.migration.merge = fail
    artifact = queue.upload(io.BytesIO(b'package'))['artifact_id']
    job = queue.submit(kind, identity=str(uuid.uuid4()), artifact=artifact if kind != 'export' else None)
    state = wait(queue, job['id'])
    assert state['status'] == 'failed'
    text = state['error']
    assert f"job_id={job['id']} kind={kind}" in caplog.text
    assert text in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert 'ConnectionError' in text and 'RuntimeError' in text and '.fail:' in text
    assert 'Exception chain:' in text and 'raise ' not in text
    for secret in ('embedding-credential', 'db secret', 'test-secret', 'labelled-secret', cfg.sql_connection_string):
        assert secret not in str(state) and secret not in caplog.text


def test_diagnostic_failure_does_not_leave_running_job(queue, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError('private failure')
    queue.migration.export = fail
    monkeypatch.setattr('service.migration.diagnostics._failure_diagnostic', fail)
    state = wait(queue, queue.submit('export', identity=str(uuid.uuid4()))['id'])
    assert state['status'] == 'failed'
    assert 'diagnostic unavailable' in state['error']
    assert 'private failure' not in str(state)


def test_abandoned_export_reopens_without_erasing_writer(queue):
    identity = str(uuid.uuid4())
    token = queue.gate.admit(write=True)
    queue.gate.enter(identity, mode='export', resources={})
    queue._save(dict(id=identity, kind='export', status='running', created_at=time.time()))
    assert queue.recent()[0]['phase'] == 'interrupted'
    assert queue.gate.status() is None
    with queue.gate._transaction() as db:
        assert db.execute('SELECT token FROM operations').fetchone()[0] == token
    queue.gate.finish_operation(token)
    queue.gate.finish_operation(queue.gate.admit(write=True))


def test_live_export_executor_is_not_reopened(queue):
    identity = str(uuid.uuid4())
    queue.gate.enter(identity, mode='export', resources={})
    queue._save(dict(id=identity, kind='export', status='running', created_at=time.time()))
    with queue.gate.execution():
        assert queue.recent()[0]['status'] == 'running'
        assert queue.gate.status()['id'] == identity


def test_interrupted_import_remains_closed(queue):
    identity = str(uuid.uuid4())
    queue.gate.enter(identity, mode='import', resources={})
    queue._save(dict(id=identity, kind='import', status='running', created_at=time.time()))
    assert queue.recent()[0]['phase'] == 'interrupted'
    assert queue.gate.status()['id'] == identity


def test_export_process_exit_can_resume_writes(queue):
    import subprocess
    import sys
    identity = str(uuid.uuid4())
    code = '''
import os,sys,time
from pathlib import Path
from types import SimpleNamespace
from service.migration.jobs import InstanceJobs
from service.migration.maintenance import MaintenanceStore
from service.migration.file_lock import exclusive_file
from service.migration.snapshot import SourceSnapshot
root, identity = Path(sys.argv[1]), sys.argv[2]
gate = MaintenanceStore(root / 'gate.db', 'test')
queue = InstanceJobs(SimpleNamespace(workspace=root,gate=gate))
with exclusive_file(queue.root / 'execution.lock'):
 queue._save(dict(id=identity,kind='export',status='running',created_at=time.time()))
 with SourceSnapshot(gate,identity).scope():
  os._exit(17)
'''
    result = subprocess.run([sys.executable, '-c', code, str(queue.migration.workspace), identity], timeout=15)
    assert result.returncode == 17
    assert queue.gate.status()['mode'] == 'export'
    assert queue.recent()[0]['status'] == 'failed'
    assert queue.gate.status() is None
    queue.gate.finish_operation(queue.gate.admit(write=True))
