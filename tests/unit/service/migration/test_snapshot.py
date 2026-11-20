import pytest

from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.snapshot import SourceSnapshot


def test_snapshot_readers_continue_and_writes_resume(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    other = Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))
    snapshot = SourceSnapshot(gate, 'export')
    with snapshot.scope() as check:
        check()
        with other.scope():
            pass
        with pytest.raises(MaintenanceError):
            with other.scope(write=True):
                pass
        assert snapshot.timestamp
    with other.scope(write=True):
        pass


def test_snapshot_timeout_reopens_source_without_changing_writer(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    with admission.scope(write=True):
        with pytest.raises(TimeoutError):
            with SourceSnapshot(gate, 'export', timeout=0).scope():
                pytest.fail('unfinished writer cannot be snapshotted')
        assert gate.status() is None
    with admission.scope(write=True):
        pass


def test_snapshot_checks_deadline_inside_agfs_scan(tmp_path):
    import time
    from types import SimpleNamespace

    from core.errors import OperationDeadlineExceededError
    from service.migration.storage import AGFSStorage
    class SlowClient:
        timeout = 10
        calls = 0
        def ls(self, path):
            self.calls += 1
            time.sleep(.03)
            return [{'name': 'a', 'isDir': True}]
    client = SlowClient()
    adapter = AGFSStorage(SimpleNamespace(agfs_mount_prefix='/local'), client)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    with pytest.raises(OperationDeadlineExceededError):
        with SourceSnapshot(gate, 'export', timeout=.02).scope():
            adapter.check()
    assert client.calls == 1
    assert gate.status() is None
    assert client.timeout == 10


def test_attachment_hash_checks_each_chunk(tmp_path, monkeypatch):
    from core.errors import OperationDeadlineExceededError
    from service.migration.assets import _file_digest
    path = tmp_path / 'large'
    path.write_bytes(b'x' * (3 * 1024 * 1024))
    checks = []
    def check():
        checks.append(1)
        if len(checks) == 3:
            raise OperationDeadlineExceededError('expired')
    monkeypatch.setattr('core.operation_deadline.check_deadline', check)
    with pytest.raises(OperationDeadlineExceededError):
        _file_digest(path)
    assert len(checks) == 3


def test_snapshot_bounds_real_http_read_wait(tmp_path):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    from pyagfs import AGFSClient
    from pyagfs.exceptions import AGFSClientError

    from service.migration.storage import AGFSStorage
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(.4)
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gate = MaintenanceStore(tmp_path / 'gate.db', 'http-budget')
    adapter = AGFSStorage(SimpleNamespace(agfs_mount_prefix='/local'),
                          AGFSClient(api_base_url=f'http://127.0.0.1:{server.server_port}', timeout=10))
    try:
        start = time.monotonic()
        with pytest.raises(AGFSClientError):
            with SourceSnapshot(gate, 'export', timeout=.05).scope():
                adapter.check()
        assert time.monotonic() - start < .3
        assert gate.status() is None
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()


def test_source_model_discovery_runs_after_writes_reopen(tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from service.migration.engine import MigrationService
    from service.migration.package import open_package

    gate = MaintenanceStore(tmp_path / 'model-gate.db', 'model-review')
    observed = []
    def create_embedder():
        observed.append(gate.status())
        token = gate.admit(write=True)
        gate.finish_operation(token)
        return SimpleNamespace(dimension=384)
    cfg = SimpleNamespace(embedding_provider='st', provider='st', st_model='test-model',
        vector_db_type='chroma', shared_spaces_enabled=False,
        to_provider_config=lambda: SimpleNamespace(create_embedder=create_embedder))
    @contextmanager
    def adapter_factory(cfg):
        yield SimpleNamespace(backend='sql', check=lambda: None,
            description=lambda: dict(backend='sql', database_type='opengauss', database_version='7'),
            records=lambda kind: iter(()))
    snapshot = SourceSnapshot(gate, 'export-model', timeout=1)
    destination = tmp_path / 'model.tar.gz'
    MigrationService(cfg, adapter_factory=adapter_factory, snapshot_scope=snapshot.scope).export(destination)
    assert observed == [None]
    with open_package(destination) as package:
        assert package.manifest['source_environment']['embedding']['dimension'] == 384
