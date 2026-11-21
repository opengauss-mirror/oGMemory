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
