from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import RequestContext, Role
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from core.persistence_outcome import WriteCompletion, write_completion
from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceStore
from tests.migration_fixtures import ConfirmedMemoryAGFSClient
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


def admission(tmp_path):
    return Admission(MaintenanceStore(tmp_path / 'gate.db', 'test'))


def assert_drain(a, blocked=False):
    a.store.enter('export', mode='export', resources={})
    try:
        if blocked:
            with pytest.raises(TimeoutError):
                a.store.wait_drained('export', 0)
        else:
            a.store.wait_drained('export', 0)
    finally:
        a.store.reopen('export')


@pytest.mark.parametrize('prior_unknown', [False, True])
def test_business_error_inside_real_agfs_lock(tmp_path, prior_unknown):
    a = admission(tmp_path)
    client = _AtomicMemoryAGFSClient()
    fs = a.guard(AGFSContextFS(client=client, mount_prefix='/local'))
    ctx = RequestContext('acme', 'alice', 'agent', 's', 't', Role.MEMBER)
    with a.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), a.backend(write=True):
                raise TimeoutError('unconfirmed mutation')
        with pytest.raises(ValueError, match='business'):
            with fs.resource_mutation_lock('ctx://acme/users/alice/memories/preferences/x', ctx):
                raise ValueError('business validation')
    assert_drain(a, blocked=prior_unknown)


@pytest.mark.parametrize('stage', ['enter', 'exit'])
def test_context_backend_failure_retains_permit(tmp_path, stage):
    a = admission(tmp_path)

    @contextmanager
    def resource():
        if stage == 'enter':
            raise TimeoutError('remote acquire')
        yield
        raise TimeoutError('remote release')

    with pytest.raises(TimeoutError):
        with a.wrap(resource, write=True, backend=True)():
            pass
    assert_drain(a, blocked=True)


@pytest.mark.parametrize('case', ['missing', 'archived', 'permission'])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_agfs_archive_deterministic_rejection(tmp_path, case, prior_unknown):
    a = admission(tmp_path)
    client = _AtomicMemoryAGFSClient()
    fs = a.guard(AGFSContextFS(client=client, mount_prefix='/local'))
    ctx = RequestContext('acme', 'alice', 'agent', 's', 't', Role.MEMBER)
    if case == 'archived':
        fs._read_file = lambda path: '{"status":"ARCHIVED"}'
    uri = 'ctx://' + ('other' if case == 'permission' else 'acme') + '/users/alice/memories/preferences/x'
    with a.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), a.backend(write=True):
                raise TimeoutError('unconfirmed mutation')
        with pytest.raises((AccessDeniedError, NodeNotFoundError)):
            fs.archive_node(uri, ctx)
    assert_drain(a, blocked=prior_unknown)


@pytest.mark.parametrize('was_running', [False, True])
def test_scene_executor_resumes_only_if_previously_running(tmp_path, monkeypatch, was_running):
    from providers.unified_config import OgMemConfig
    from scene.config import SceneConfig
    from scene.manager import SceneMemoryManager
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle

    a = admission(tmp_path)
    service = MemoryService(OgMemConfig(provider='mock', skill_sedimentation_enabled=False),
                            admission=a, defer_start=True)
    jobs = Mock()
    jobs.claim.return_value = None
    def manager():
        return SceneMemoryManager(jobs, Mock(), Mock(config=SceneConfig()), config=SceneConfig())
    original = manager()
    service._scene_manager = original
    if was_running:
        original.start(poll_seconds=.01)
    lifecycle = ServiceLifecycle(service)
    a.store.enter('import', mode='import', resources={})
    assert lifecycle.pause()
    lifecycle.refresh()
    a.store.reopen('import')
    def initialize():
        service._scene_manager = manager()
    monkeypatch.setattr(service, '_ensure_scene_runtime', initialize)
    monkeypatch.setattr(service, 'start_extraction_recovery', lambda: None)
    try:
        lifecycle.resume()
        assert not original.running
        assert (service._scene_manager is not None) == was_running
        if was_running:
            assert service._scene_manager.running
    finally:
        service.shutdown()


@pytest.mark.parametrize('close_result', [True, False])
def test_pause_closes_scene_index_before_refresh(tmp_path, close_result):
    from types import SimpleNamespace
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle
    a = admission(tmp_path)
    service = MemoryService(OgMemConfig(provider='mock', skill_sedimentation_enabled=False),
                            admission=a, defer_start=True)
    index = Mock()
    index.close.return_value = close_result
    service._scene_source = SimpleNamespace(candidate_index=index)
    service._scene_manager = SimpleNamespace(running=False, stop=lambda timeout: True)
    a.store.enter('import', mode='import', resources={})
    lifecycle = ServiceLifecycle(service)
    try:
        paused = lifecycle.pause()
        index.close.assert_called_once()
        assert paused is close_result
        if paused:
            lifecycle.refresh()
            assert service._scene_source is None
    finally:
        service._scene_source = None
        service._scene_manager = None
        service.shutdown()


@pytest.mark.parametrize('failure', ['statement', 'rollback', 'commit'])
def test_scene_transaction_releases_only_confirmed_rollback(tmp_path, failure):
    from scene.sql_store import _SQLStore
    a = admission(tmp_path)
    store = object.__new__(_SQLStore)
    connection = Mock()
    connection.cursor.return_value = nullcontext(Mock())
    if failure == 'rollback':
        connection.rollback.side_effect = TimeoutError('rollback unknown')
    if failure == 'commit':
        connection.commit.side_effect = TimeoutError('commit unknown')
    store.pool = Mock(get_connection=Mock(return_value=connection))
    with pytest.raises((ValueError, TimeoutError)):
        with a.backend(write=True), store._transaction('acme'):
            if failure != 'commit':
                raise ValueError('business conflict')
    assert_drain(a, blocked=failure != 'statement')


@pytest.mark.parametrize('exists', [False, True])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_atomic_resource_conflict_settles_after_cleanup(tmp_path, exists, prior_unknown):
    from core.errors import ConcurrentModificationError
    from tests.unit.fs.test_agfs_context_fs import _atomic_code_stack, _atomic_code_context, _atomic_code_node
    _, fs, outbox = _atomic_code_stack()
    ctx = _atomic_code_context()
    if exists:
        node = _atomic_code_node()
        fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))
    a = admission(tmp_path)
    a.guard(fs)
    stale = _atomic_code_node(expected_version=2)
    with a.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), a.backend(write=True):
                raise TimeoutError('unconfirmed prior mutation')
        with pytest.raises(ConcurrentModificationError):
            fs.write_node_with_outbox(stale, ctx, outbox.build_write_event(stale))
    assert_drain(a, blocked=prior_unknown)


@pytest.mark.parametrize('failed_cleanup', ['candidate', 'lock'])
def test_atomic_resource_conflict_keeps_unknown_cleanup(tmp_path, failed_cleanup, monkeypatch):
    from core.errors import ConcurrentModificationError
    from tests.unit.fs.test_agfs_context_fs import _atomic_code_stack, _atomic_code_context, _atomic_code_node
    client, fs, outbox = _atomic_code_stack()
    a = admission(tmp_path)
    a.guard(fs)
    if failed_cleanup == 'candidate':
        monkeypatch.setattr(fs, '_remove_tree_if_present', lambda _: False)
    else:
        original_rm = client.rm
        def remove(path, **kwargs):
            if path.endswith('.resource-mutation-lock'):
                raise TimeoutError('unknown lock deletion')
            return original_rm(path, **kwargs)
        monkeypatch.setattr(client, 'rm', remove)
    node = _atomic_code_node(expected_version=2)
    with pytest.raises(ConcurrentModificationError) as error:
        fs.write_node_with_outbox(node, _atomic_code_context(), outbox.build_write_event(node))
    assert write_completion(error.value) is not WriteCompletion.SETTLED
    assert_drain(a, blocked=True)


@pytest.mark.parametrize('method', ['write_json', '_write_json', 'mutate_code_workspace_sync_state'])
@pytest.mark.parametrize('failure', ['statement', 'rollback', 'commit'])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_sql_control_failure_completion(tmp_path, method, failure, prior_unknown):
    from server.sql_control_plane_store import SQLControlPlaneStore
    store = object.__new__(SQLControlPlaneStore)
    connection, cursor = Mock(), Mock()
    cursor.fetchone.return_value = ({}, None, None)
    connection.cursor.return_value = nullcontext(cursor)
    store._get_connection = lambda: connection
    store._return_connection = lambda _: None
    store._snapshots = SimpleNamespace(values={})
    store._dialect = Mock()
    if failure == 'commit':
        connection.commit.side_effect = TimeoutError('commit unknown')
    else:
        cursor.execute.side_effect = ValueError('statement rejected')
        store._dialect.insert_audit_log.side_effect = ValueError('statement rejected')
        if failure == 'rollback':
            connection.rollback.side_effect = TimeoutError('rollback unknown')
    a = admission(tmp_path)
    arguments = {
        'write_json': (store.global_accounts_path(), {'accounts': {}}),
        '_write_json': (store.audit_logs_path('acme'), {'events': [{}]}),
        'mutate_code_workspace_sync_state': ('acme', 'root', lambda _: {}),
    }
    with a.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), a.backend(write=True):
                raise TimeoutError('earlier unknown write')
        with pytest.raises((ValueError, TimeoutError)):
            a.wrap(getattr(store, method), write=True, backend=True)(*arguments[method])
    assert_drain(a, blocked=prior_unknown or failure != 'statement')


@pytest.mark.parametrize('lock_kind', ['resource', 'update'])
def test_joined_lease_renews_during_import(tmp_path, monkeypatch, lock_kind):
    import threading
    from core.models import ContextNode
    import fs.agfs_adapter.agfs_context_fs as module

    a = admission(tmp_path)
    client = a.guard(ConfirmedMemoryAGFSClient())
    fs = a.guard(AGFSContextFS(client=client, mount_prefix='/local'))
    ctx = RequestContext('acme', 'alice', 'agent', 's', 't', Role.MEMBER)
    uri = 'ctx://acme/users/alice/memories/preferences/x'
    node = ContextNode(uri=uri, context_type='MEMORY', category='preferences',
                       owner_space='user:alice', level=2, abstract='memory',
                       overview='memory', content='before', metadata={'version': 1})
    if lock_kind == 'update':
        fs.write_node(node, ctx)
    renew = '_write_resource_mutation_lease' if lock_kind == 'resource' else '_write_update_lease'
    monkeypatch.setattr(module, '_RESOURCE_MUTATION_LOCK_RENEW_INTERVAL_SECONDS', .01)
    monkeypatch.setattr(module, '_UPDATE_LOCK_RENEW_INTERVAL_SECONDS', .01)
    entered, renewed = threading.Event(), threading.Event()
    write_lease = getattr(fs, renew)

    def track_renewal(*args):
        write_lease(*args)
        if entered.is_set() and threading.current_thread() is not threading.main_thread():
            renewed.set()

    monkeypatch.setattr(fs, renew, track_renewal)

    def during_mutation():
        a.store.enter('import', mode='import', resources={})
        entered.set()
        assert renewed.wait(2)
        with pytest.raises(TimeoutError):
            a.store.wait_drained('import', 0)

    if lock_kind == 'resource':
        with fs.resource_mutation_lock(uri, ctx) as lease:
            during_mutation()
            lease.ensure_owned()
    else:
        write_files = fs._write_node_files

        def write_during_maintenance(*args, **kwargs):
            during_mutation()
            return write_files(*args, **kwargs)

        monkeypatch.setattr(fs, '_write_node_files', write_during_maintenance)
        node.content = 'after'
        node.metadata = {'_expected_version': 1}
        fs.write_node(node, ctx)
    a.store.wait_drained('import', 0)
    a.store.reopen('import')
    if lock_kind == 'update':
        assert fs.read_node(uri, ctx).content == 'after'


@pytest.mark.parametrize('committing', [False, True])
@pytest.mark.parametrize('rollback_fails', [False, True])
def test_sql_rollback_settlement_and_connection_retirement(committing, rollback_fails):
    from fs.sql_adapter.pool import rollback_write
    conn = Mock()
    error = ValueError('write rejected')
    if rollback_fails:
        conn.rollback.side_effect = RuntimeError('connection lost')
        with pytest.raises(RuntimeError, match='connection lost'):
            rollback_write(conn, error, committing=committing)
        conn.close.assert_called_once()
        assert write_completion(error) is WriteCompletion.UNKNOWN
    else:
        rollback_write(conn, error, committing=committing)
        conn.close.assert_not_called()
        assert (write_completion(error) is WriteCompletion.SETTLED) is (not committing)


def test_maintenance_error_classification_uses_type():
    from core.errors import MaintenanceError
    from core.write_contract import error_fields
    class DerivedMaintenance(MaintenanceError):
        pass
    assert error_fields(DerivedMaintenance())['http_status'] == 503
    unrelated = type('MaintenanceError', (RuntimeError,), {})
    assert error_fields(unrelated())['http_status'] == 500
