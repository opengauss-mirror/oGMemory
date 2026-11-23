import time

import pytest

from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from server.service_factory import close_service, create_service
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import wait_participants


def test_server_startup_registers_before_workers_and_closes(tmp_path, monkeypatch):
    cfg = OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False)
    seen = []

    def startup(service):
        store = service._admission.store
        with store._transaction() as db:
            assert db.execute('SELECT count(*) FROM participants').fetchone()[0] == 1
            assert db.execute('SELECT count(*) FROM operations WHERE writing=1').fetchone()[0] == 1
        seen.append('started')

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', startup)
    service = create_service(cfg)
    assert seen == ['started']
    gate = service._admission.store
    close_service(service)
    with gate._transaction() as db:
        assert db.execute('SELECT count(*) FROM participants').fetchone()[0] == 0
    with pytest.raises(MaintenanceError, match='stopping'):
        with service._admission.scope():
            pass


def test_server_starting_during_import_remains_dormant(tmp_path, monkeypatch):
    gate = MaintenanceStore(tmp_path / 'maintenance.sqlite', str(tmp_path.resolve()))
    gate.enter('import', mode='import', resources={})
    seen = []
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda s: seen.append('started'))
    service = create_service(OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False))
    try:
        assert seen == []
        assert service.health()['status'] == 'maintenance'
        with pytest.raises(MaintenanceError):
            service.get_session_manager()
        wait_participants(gate, 'import', 'drain', timeout=2)
        gate.request_refresh('import')
        wait_participants(gate, 'import', 'refresh', timeout=2)
        gate.reopen('import')
        deadline = time.monotonic() + 2
        while not seen and time.monotonic() < deadline:
            time.sleep(.01)
        assert seen == ['started']
    finally:
        close_service(service)


def test_rejected_shutdown_preserves_maintenance_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda s: None)
    service = create_service(OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False))
    gate = service._admission.store
    gate.enter('import', mode='import', resources={})
    try:
        wait_participants(gate, 'import', 'drain', timeout=2)
        with pytest.raises(MaintenanceError):
            close_service(service)
        assert not service._migration_participant.stop.is_set()
        gate.request_refresh('import')
        wait_participants(gate, 'import', 'refresh', timeout=2)
        gate.reopen('import')
    finally:
        close_service(service)


def test_shutdown_can_retry_after_inflight_reader_finishes(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda s: None)
    service = create_service(OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False))
    entered, finish = threading.Event(), threading.Event()
    def reader():
        with service._admission.scope():
            entered.set()
            finish.wait(3)
    worker = threading.Thread(target=reader)
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(MaintenanceError, match='operations have not stopped'):
            close_service(service, timeout=.1)
        with pytest.raises(MaintenanceError, match='stopping'):
            with service._admission.scope():
                pass
    finally:
        finish.set()
        worker.join(2)
    close_service(service, timeout=2)
    assert service._migration_participant is None


def test_shared_worker_start_and_shutdown_follow_registered_lifecycle(tmp_path, monkeypatch):
    seen = []
    cfg = OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False)
    cfg.shared_spaces_enabled = True
    monkeypatch.setattr('server.memory_service._check_shared_agfs_storage', lambda *a, **kw: None)
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda service: None)
    class Worker:
        def stop(self, timeout):
            seen.append('stopped')
            return True
    def start(service):
        assert service._admission._operation.get() is not None
        service._space_worker = Worker()
        seen.append('started')
    monkeypatch.setattr(MemoryService, '_start_space_worker', start)
    service = create_service(cfg)
    assert seen == ['started']
    close_service(service)
    assert seen == ['started', 'stopped']


@pytest.mark.parametrize('composite', [False, True])
def test_server_can_close_chroma_without_migration_release(tmp_path, monkeypatch, composite):
    from unittest.mock import Mock

    from providers.composite_index import CompositeProjectionWriter
    from providers.vector_index.chroma_index import ChromaVectorIndex

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda service: None)
    service = create_service(OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False))
    participant = service._migration_participant
    try:
        vector = object.__new__(ChromaVectorIndex)
        vector._client = object()  # SDK versions without close().
        service._vector_index = vector
        if composite:
            service._lexical_index = Mock()
            service._projection_index = CompositeProjectionWriter(vector, service._lexical_index)
        pool = service._sql_pool = Mock()

        close_service(service)

        pool.close_all.assert_called_once()
        assert service._migration_participant is None
        with participant.store._transaction() as db:
            assert db.execute('SELECT 1 FROM participants').fetchone() is None
    finally:
        if service._migration_participant is not None:
            service._projection_index = service._vector_index = service._lexical_index = None
            close_service(service)


def test_server_starting_during_export_starts_extraction_discovery(tmp_path, monkeypatch):
    from service.extraction_retry import ExtractionRetryWorker
    from service.migration.snapshot import SourceSnapshot

    cfg = OgMemConfig(migration_workspace=str(tmp_path), skill_sedimentation_enabled=False)
    gate = MaintenanceStore.for_workspace(tmp_path)
    discoveries = []
    monkeypatch.setattr(ExtractionRetryWorker, 'start_discovery',
                        lambda worker, discovery: discoveries.append(discovery))
    service = None
    try:
        with SourceSnapshot(gate, 'export').scope():
            service = create_service(cfg)
            assert discoveries and discoveries[0] is service._extraction_discovery
        assert len(discoveries) == 1
    finally:
        if service is not None:
            close_service(service)
