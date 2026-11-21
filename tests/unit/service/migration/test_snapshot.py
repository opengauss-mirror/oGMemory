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
            with other.content_scope():
                pass
        assert snapshot.timestamp
    with other.content_scope():
        pass


def test_snapshot_timeout_reopens_source_without_changing_writer(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    with admission.content_scope():
        with pytest.raises(TimeoutError):
            with SourceSnapshot(gate, 'export', timeout=0).scope():
                pytest.fail('unfinished writer cannot be snapshotted')
        assert gate.status() is None
    with admission.content_scope():
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
def test_snapshot_does_not_wait_for_request_after_persistence(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    with admission.scope(write=True):
        with admission.backend(write=True):
            pass
        # The enclosing request can still calculate or stream its response.
        with SourceSnapshot(gate, 'snapshot').scope():
            with admission.backend(write=True, content=False):
                pass  # Index projection can continue.
            with pytest.raises(MaintenanceError), admission.backend(write=True):
                pass
        gate.enter('import', mode='import', resources={})
        with pytest.raises(TimeoutError):
            gate.wait_drained('import', 0)
    gate.reopen('import')


def test_snapshot_waits_for_whole_content_mutation_then_reopens(tmp_path):
    import threading
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    started, finish = threading.Event(), threading.Event()
    def write():
        with admission.content_scope():
            with admission.backend(write=True):
                started.set()
                assert finish.wait(2)
            with admission.backend(write=True):
                pass  # Completing an admitted mutation remains permitted.
    thread = threading.Thread(target=write)
    thread.start()
    assert started.wait(2)
    try:
        with pytest.raises(TimeoutError), SourceSnapshot(gate, 'slow', timeout=.02).scope():
            pytest.fail('content mutation has not finished')
        assert gate.status() is None
    finally:
        finish.set()
        thread.join(2)
    assert not thread.is_alive()
    with SourceSnapshot(gate, 'complete').scope():
        pass


def test_runtime_and_projection_writes_do_not_block_snapshot(tmp_path):
    from fs.agfs_adapter.client import NativeAGFSClient
    from service.migration.admission import _content_path
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    client = object.__new__(NativeAGFSClient)
    # Preserve concrete transport type, replace actual I/O in this policy test.
    from types import MethodType
    client.write = MethodType(lambda self, path, data: None, client)
    admission.guard(client)
    with SourceSnapshot(gate, 'runtime').scope():
        with admission.backend(write=True, content=False):
            client.write('/local/accounts/acme/users/u/memories/x/.outbox/e.json', b'{}')
        client.write('/local/accounts/acme/sessions/s/state/content.md', b'{}')
        with pytest.raises(MaintenanceError):
            client.write('/local/accounts/acme/users/u/memories/x/content.md', b'changed')
    assert not _content_path('/local/accounts/acme/sessions/s/state/content.md')
    assert _content_path('/local/accounts/acme/sessions/s/history/a/content.md')


@pytest.mark.parametrize('name', ['_system', '.outbox', '.foo-lock', 'traces', 'tool-results'])
def test_business_names_do_not_bypass_content_barrier(tmp_path, name):
    from service.migration.admission import _content_path
    assert _content_path(f'/local/accounts/acme/users/alice/memories/preferences/{name}/content.md')


def test_keyword_move_destination_is_checked(tmp_path):
    from types import MethodType
    from fs.agfs_adapter.client import NativeAGFSClient
    client = object.__new__(NativeAGFSClient)
    calls = []
    client.mv = MethodType(lambda self, source, destination: calls.append(destination), client)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'move')
    Admission(gate).guard(client)
    with SourceSnapshot(gate, 'snapshot').scope():
        with pytest.raises(MaintenanceError):
            client.mv('/local/accounts/acme/sessions/s/state/file',
                      destination='/local/accounts/acme/users/alice/memories/preferences/x/content.md')
    assert calls == []


def test_runtime_to_content_move_requires_content_permit():
    from service.migration.admission import _content_arguments
    assert _content_arguments('ctx://acme/sessions/s/state',
                              'ctx://acme/users/alice/memories/preferences/x')


@pytest.mark.parametrize('failure', ['statement', 'commit'])
def test_shared_transaction_preserves_unknown_commit(tmp_path, failure):
    from contextlib import nullcontext
    from unittest.mock import Mock
    from server.space_repository import SpaceRepository
    gate = MaintenanceStore(tmp_path / 'gate.db', 'space')
    repo = object.__new__(SpaceRepository)
    repo._migration_admission = Admission(gate)
    conn = Mock()
    conn.cursor.return_value = nullcontext(Mock())
    repo.pool = Mock(get_connection=Mock(return_value=conn))
    if failure == 'commit':
        conn.commit.side_effect = TimeoutError('commit confirmation lost')
    with pytest.raises((ValueError, TimeoutError)):
        with repo.transaction('acme'):
            if failure == 'statement':
                raise ValueError('rejected')
    gate.enter('snapshot', mode='export', resources={})
    if failure == 'commit':
        with pytest.raises(TimeoutError):
            gate.wait_drained('snapshot', 0)
    else:
        gate.wait_drained('snapshot', 0)
