from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import RequestContext, Role
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceStore
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


@pytest.mark.parametrize('rollback_fails', [False, True])
def test_scene_transaction_releases_only_confirmed_rollback(tmp_path, rollback_fails):
    from scene.sql_store import _SQLStore
    a = admission(tmp_path)
    store = object.__new__(_SQLStore)
    connection = Mock()
    connection.cursor.return_value = nullcontext(Mock())
    if rollback_fails:
        connection.rollback.side_effect = TimeoutError('rollback unknown')
    store.pool = Mock(get_connection=Mock(return_value=connection))
    with pytest.raises((ValueError, TimeoutError)):
        with a.backend(write=True), store._transaction('acme'):
            raise ValueError('business conflict')
    assert_drain(a, blocked=rollback_fails)


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
    assert not getattr(error.value, 'agfs_rejection_settled', False)
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
