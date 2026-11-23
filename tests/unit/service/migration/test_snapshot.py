import multiprocessing
import os

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


def test_runtime_to_content_move_requires_content_permit():
    from service.migration.admission import _content_arguments
    assert _content_arguments('ctx://acme/sessions/s/state',
                              'ctx://acme/users/alice/memories/preferences/x')


def test_normalized_transport_path_cannot_escape_runtime():
    from service.migration.admission import _content_path
    assert _content_path('/local/accounts/acme/_system/../users/alice/memories/preferences/x/content.md')


def _exit_during_snapshot(path, mode):
    gate = MaintenanceStore(path, 'instance')
    if mode == 'export':
        with SourceSnapshot(gate, 'interrupted').scope():
            os._exit(0)
    with gate.execution():
        gate.enter('interrupted', mode='import', resources={})
        os._exit(0)


def _exit_during_write(path, content):
    from service.migration.participants import Participant

    gate = MaintenanceStore(path, 'instance')
    participant = Participant(gate, pause=lambda: True, refresh=lambda: None, resume=lambda: None)
    with participant.admission.backend(write=True, content=content):
        os._exit(0)


def _run_abrupt_exit(target, *args):
    process = multiprocessing.get_context('spawn').Process(target=target, args=args)
    process.start()
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(5)
    assert process.exitcode == 0


@pytest.mark.parametrize('resume', ['write', 'upgrade', 'snapshot'])
def test_abandoned_export_reopens_on_next_operation(tmp_path, resume):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    reader = gate.admit() if resume == 'upgrade' else None
    _run_abrupt_exit(_exit_during_snapshot, gate.path, 'export')
    assert gate.status()['id'] == 'interrupted'
    if resume == 'snapshot':
        with SourceSnapshot(gate, 'next').scope():
            assert gate.status()['id'] == 'next'
    elif resume == 'upgrade':
        gate.upgrade(reader)
        gate.finish_operation(reader)
    else:
        gate.finish_operation(gate.admit(write=True))
    assert gate.status() is None
    assert gate.status('interrupted')['phase'] == 'unchanged'


def test_abandoned_import_remains_closed(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    _run_abrupt_exit(_exit_during_snapshot, gate.path, 'import')
    with pytest.raises(MaintenanceError):
        gate.admit(write=True)
    with pytest.raises(MaintenanceError), SourceSnapshot(gate, 'next').scope():
        pytest.fail('snapshot cannot replace an active import')
    assert gate.status()['id'] == 'interrupted'


@pytest.mark.parametrize('content', [True, False])
def test_snapshot_only_waits_for_content_after_writer_exit(tmp_path, content):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    _run_abrupt_exit(_exit_during_write, gate.path, content)
    snapshot = SourceSnapshot(gate, 'next', timeout=.1)
    if content:
        with pytest.raises(TimeoutError), snapshot.scope():
            pytest.fail('unknown content write must block snapshots')
    else:
        with snapshot.scope():
            pass
    gate.enter('restore', mode='import', resources={})
    with pytest.raises(MaintenanceError, match='unfinished backend operations'):
        gate.wait_drained('restore', 0)


def test_composite_projection_continues_without_delaying_source_snapshot(tmp_path):
    from threading import Event, Thread

    from providers.composite_index import CompositeProjectionWriter

    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    entered, release, finished = Event(), Event(), Event()

    class Lexical:
        def delete(self, ids):
            if ids == ['in-flight']:
                entered.set()
                assert release.wait(5)

    writer = admission.guard(CompositeProjectionWriter(None, Lexical()))

    def project():
        writer.delete(['in-flight'])
        finished.set()

    thread = Thread(target=project)
    thread.start()
    try:
        assert entered.wait(3)
        with SourceSnapshot(gate, 'snapshot', timeout=.5).scope():
            assert not finished.is_set()
            release.set()
            assert finished.wait(3)
            writer.delete(['during-export'])
    finally:
        release.set()
        thread.join(5)
    gate.enter('restore', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        writer.delete(['during-import'])


def test_shared_agfs_body_load_is_read_only_during_source_snapshot(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from core.models import ContextNode
    from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
    from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient

    gate = MaintenanceStore.for_workspace(tmp_path)
    admission = Admission(gate)
    bridge = object.__new__(AGFSSpaceBridge)
    transport = _AtomicMemoryAGFSClient()
    bridge._agfs = SimpleNamespace(_mount_prefix='/local', _client=transport)
    uri = 'ctx://acme/users/alice/memories/preferences/editor'
    payload = json.dumps(dict(uri=uri, content='body', abstract='title', overview='overview')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = f'{bridge._content_directory(uri)}/{digest}.json'
    transport.write(path, payload)
    node = ContextNode(uri=uri, context_type='MEMORY', category='preferences', level=2,
                       owner_space='user:alice', abstract='', overview='', content='',
                       metadata={'agfs_content': {'path': path, 'sha256': digest, 'bytes': len(payload)}})
    bridge._agfs._client = admission.guard(transport)
    bridge = admission.guard(bridge)
    with SourceSnapshot(gate, 'export').scope():
        assert bridge.hydrate_nodes([node])[0].content == 'body'
        assert bridge.load_content([{'uri': uri, 'metadata': node.metadata}])[0]['content'] == 'body'
    gate.enter('import', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        bridge.hydrate_nodes([node])
    with pytest.raises(MaintenanceError):
        bridge.load_content([{'uri': uri, 'metadata': node.metadata}])
