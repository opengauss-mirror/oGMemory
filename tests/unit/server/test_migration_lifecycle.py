import json
from contextlib import nullcontext

import pytest

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from server.migration_lifecycle import ServiceLifecycle
from service.migration.admission import install
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import Participant
from session.session_manager import SessionManager
from tests.integration.conftest import InMemoryContextFS


def test_real_session_checkpoint_precedes_cache_discard(tmp_path, monkeypatch):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False))
    lifecycle = ServiceLifecycle(service)
    participant = Participant(store, pause=lifecycle.pause, refresh=lifecycle.refresh, resume=lambda: None)
    install(service, participant.admission)
    fs = participant.admission.guard(InMemoryContextFS())
    manager = SessionManager(get_context_fs=lambda: fs)
    service._session_mgr = manager
    monkeypatch.setattr(service, '_session_mutation_scope', lambda *a, **k: nullcontext())
    ctx = RequestContext('acme', 'alice', 'agent', 'session', 'test')
    manager.add_message('session', 'user', 'accepted message', ctx)
    store.enter('task', mode='import', resources={})
    try:
        participant.step()
        assert store.participants_ready('task', 'drain')
        durable = next(iter(fs._store.values()))
        assert json.loads(durable.content)['buffer_messages'][0]['content'] == 'accepted message'
        store.backed_up('task', {})
        store.begin_changes('task', {})
        store.verified('task')
        store.request_refresh('task')
        participant.step()
        assert service._session_mgr is None
        assert json.loads(durable.content)['buffer_messages'][0]['content'] == 'accepted message'
        store.reopen('task')
    finally:
        participant.close()
        service.shutdown()


def test_health_during_import_does_not_initialize_storage(tmp_path, monkeypatch):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    from service.migration.admission import Admission
    store.enter('task', mode='import', resources={})
    service = MemoryService(OgMemConfig(), admission=Admission(store))
    monkeypatch.setattr(service, '_get_shared_sql_pool', lambda: pytest.fail('must not initialize'))
    assert service.health()['status'] == 'maintenance'
    with pytest.raises(MaintenanceError):
        service.get_session_manager()
    service.shutdown()


def test_unmodified_target_keeps_original_session_cache(tmp_path, monkeypatch):
    from service.migration.admission import Admission

    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    service = MemoryService(OgMemConfig(), admission=Admission(store))
    original = SessionManager()
    ctx = RequestContext('acme', 'alice', 'agent', 'session', 'test')
    original.add_message('session', 'user', 'not yet checkpointed', ctx)
    service._session_mgr = original
    store.enter('cancelled', mode='import', resources={})
    lifecycle = ServiceLifecycle(service)
    lifecycle.refresh()
    assert service._session_mgr is original
    assert original.get_or_create('session', ctx).messages[0].content == 'not yet checkpointed'
    service.shutdown()


def test_export_blocks_real_agfs_constructor_but_keeps_existing_reads(tmp_path, monkeypatch):
    from service.migration.admission import Admission
    from tests.migration_fixtures import LocalClient

    client = LocalClient(tmp_path / 'files')
    client.path('/local').mkdir(parents=True)
    monkeypatch.setattr('server.memory_service.AGFSClient', lambda **kwargs: client)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    service = MemoryService(OgMemConfig(storage_backend='agfs', agfs_mount_prefix='/local/warm',
                                      skill_sedimentation_enabled=False), admission=Admission(gate))
    fs = service._get_context_fs()
    assert client.path('/local/warm').is_dir()
    client.path('/local/warm/accounts/acme/users/alice/memories').mkdir(parents=True)
    service._mount_prefix = '/local/cold'
    gate.enter('export', mode='export', resources={})
    try:
        assert service._get_context_fs() is fs
        service._migration_context_fs = None
        with pytest.raises(MaintenanceError):
            service._get_context_fs()
        assert not (tmp_path / 'files/local/cold').exists()
        with service._admission.scope():
            assert fs.list_children('ctx://acme/users/alice/memories/', RequestContext('acme', 'alice', 'agent', 's', 'test')) == []
    finally:
        gate.reopen('export')
        service.shutdown()


def test_refresh_discards_shared_space_worker_repository_and_content(tmp_path):
    from service.migration.admission import Admission

    gate = MaintenanceStore(tmp_path / 'gate.db', 'spaces')
    service = MemoryService(OgMemConfig(), admission=Admission(gate), defer_start=True)
    service._space_worker = object()
    service._space_repository = object()
    service._space_content_store = object()
    gate.enter('restore', mode='import', resources={})
    ServiceLifecycle(service).refresh()
    assert service._space_worker is None
    assert service._space_repository is None
    assert service._space_content_store is None
    service.shutdown()


def test_resume_restarts_previously_running_workspace_scheduler(tmp_path, monkeypatch):
    from unittest.mock import Mock
    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False), defer_start=True)
    sync = Mock(_scheduler=object())
    service._code_workspace_sync_service = sync
    lifecycle = ServiceLifecycle(service)
    monkeypatch.setattr(service, '_get_code_workspace_sync_service', lambda: sync)
    monkeypatch.setattr(service, 'start_extraction_recovery', lambda: None)
    assert lifecycle.pause()
    sync.shutdown.assert_called_once_with(wait=True)
    lifecycle.resume()
    sync.start.assert_called_once()
    lifecycle.resume()
    sync.start.assert_called_once()
    service._code_workspace_sync_service = None
    service.shutdown()


def test_fts_must_stop_before_refresh_and_is_discarded(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from service.migration.admission import Admission

    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    service = MemoryService(OgMemConfig(), admission=Admission(store))
    cache = SimpleNamespace(close=Mock(return_value=False))
    service._agfs_fts = cache
    lifecycle = ServiceLifecycle(service)
    store.enter('cancelled', mode='import', resources={})
    assert lifecycle.pause() is False
    assert store.status() is not None
    cache.close.return_value = True
    assert lifecycle.pause() is True
    lifecycle.refresh()
    assert service._agfs_fts is None
    assert service._read_api is None
    service.shutdown()


@pytest.mark.parametrize('task_id', ['agent-turn', 'archive'])
def test_import_waits_for_retry_execution_threads(tmp_path, monkeypatch, task_id):
    import threading

    store = MaintenanceStore(tmp_path / 'gate.db', 'retry-threads')
    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False), defer_start=True)
    from unittest.mock import Mock
    scene = Mock(running=True)
    service._scene_manager = scene
    lifecycle = ServiceLifecycle(service, timeout=.05)
    participant = Participant(store, pause=lifecycle.pause, refresh=lifecycle.refresh,
                              resume=lambda: None)
    entered, finish = threading.Event(), threading.Event()
    def attempt(*args):
        entered.set()
        assert finish.wait(3)
    monkeypatch.setattr(service._extraction_retry, 'run_once', attempt)
    ctx = RequestContext('acme', 'alice', 'agent', 's', 'retry')
    service._extraction_retry.wake(ctx, {
        'id': task_id, 'status': 'queued', 'agent_async': task_id == 'agent-turn',
    })
    assert entered.wait(2)
    with service._extraction_retry._condition:
        if task_id == 'agent-turn':
            assert any(thread.is_alive() for thread in service._extraction_retry._agent_threads)
            assert service._extraction_retry._legacy_thread is None
        else:
            assert service._extraction_retry._legacy_thread.is_alive()
            assert not service._extraction_retry._agent_threads
    store.enter('restore', mode='import', resources={})
    try:
        with pytest.raises(MaintenanceError, match='workers'):
            participant.step()
        assert not store.participants_ready('restore', 'drain')
        scene.stop.assert_not_called()
        finish.set()
        assert service._extraction_retry.close(1)
        store.retry_participants('restore')
        participant.step()
        assert store.participants_ready('restore', 'drain')
    finally:
        finish.set()
        service._extraction_retry.close(1)
        participant.close()
        service.shutdown()


def test_pause_wakes_idle_outbox_worker():
    import threading

    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False), defer_start=True)
    entered = threading.Event()
    def idle_outbox():
        entered.set()
        service._outbox_wakeup.wait(3)
    thread = threading.Thread(target=idle_outbox)
    service._outbox_thread = thread
    thread.start()
    assert entered.wait(1)
    try:
        assert ServiceLifecycle(service, timeout=.2).pause()
        assert not thread.is_alive()
    finally:
        service._outbox_wakeup.set()
        thread.join(1)
        service.shutdown()



@pytest.mark.parametrize('supports_release', [False, True])
def test_pause_then_shutdown_closes_composite_vector_once(tmp_path, supports_release):
    from unittest.mock import Mock

    from providers.composite_index import CompositeProjectionWriter

    closed = Mock()

    class Vector:
        def close(self):
            closed('close')

    class CachedVector(Vector):
        def release_for_migration(self):
            closed('release')

    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False), defer_start=True)
    lifecycle = ServiceLifecycle(service)
    store = MaintenanceStore(tmp_path / 'gate.db', 'projection')
    participant = Participant(store, pause=lifecycle.pause, refresh=lifecycle.refresh, resume=lifecycle.resume)
    install(service, participant.admission)
    vector = participant.admission.guard(CachedVector() if supports_release else Vector())
    lexical = Mock()
    service._vector_index = vector
    service._lexical_index = lexical
    service._projection_index = participant.admission.guard(CompositeProjectionWriter(vector, lexical))
    store.enter('restore', mode='import', resources={})
    try:
        participant.step()
        assert store.participants_ready('restore', 'drain')
        service.shutdown()
        closed.assert_called_once_with('release' if supports_release else 'close')
        lexical.close.assert_called_once()
    finally:
        participant.close()
