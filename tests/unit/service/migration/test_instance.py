"""Registered-service migration with real Chroma and local AGFS byte transport."""
import json

import pytest

from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from server.service_factory import close_service, create_service
from service.migration.instance import InstanceMigration
from service.migration.rebuild import Rebuilder
from service.migration.storage import AGFSStorage
from tests.unit.service.migration.test_merge_recovery import package
from tests.unit.service.migration.test_tree_backup import LocalClient


@pytest.fixture
def instance(tmp_path, monkeypatch):
    pytest.importorskip('chromadb')
    client = LocalClient(tmp_path / 'files')
    client.path('/local/accounts').mkdir(parents=True)
    monkeypatch.setattr('pyagfs.AGFSClient', lambda **kw: client)
    monkeypatch.setattr('server.memory_service.AGFSClient', lambda **kw: client)
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda s: None)
    cfg = OgMemConfig(storage_backend='agfs', agfs_mount_prefix='/local', provider='mock',
                     skill_sedimentation_enabled=False, vector_db_type='chroma', opengauss_dimension=3,
                     chroma_persist_directory=str(tmp_path / 'chroma'), migration_workspace=str(tmp_path / 'control'))
    service = create_service(cfg)
    # Populate the process's native client before migration opens another one.
    assert service._get_shared_vector_index().count() == 0
    yield service, InstanceMigration(service, timeout=2), client
    if service._admission.store.status() is None:
        close_service(service)
    else:
        # Test fixtures may deliberately leave an uncertain migration operation.
        service._migration_participant.close()
        service.shutdown()


def test_registered_instance_merges_reopens_and_skips_repeat(instance, tmp_path):
    service, migration, client = instance
    source = package(tmp_path)
    assert migration.preview(source)['counts']['new'] == 2
    result = migration.merge(source, task='first')
    assert result['phase'] == 'succeeded'
    assert service._admission.store.status() is None
    assert service._get_shared_vector_index().count() == 4
    assert migration.preview(source)['counts']['unchanged'] == 2
    before = list(AGFSStorage(service._cfg, client).records('nodes'))
    repeated = migration.merge(source, task='second')
    assert repeated['phase'] == 'succeeded'
    assert service._get_shared_vector_index().count() == 4
    assert list(AGFSStorage(service._cfg, client).records('nodes')) == before


def test_registered_instance_restores_after_verification_failure(instance, tmp_path, monkeypatch):
    service, migration, client = instance
    source = package(tmp_path)
    migration.merge(source, task='seed')
    before = {r.id: r for r in service._get_shared_vector_index().iter_records()}
    original = list(AGFSStorage(service._cfg, client).records('nodes'))
    def reject(*args):
        raise ValueError('injected verification failure')
    monkeypatch.setattr(Rebuilder, 'verify', reject)
    with pytest.raises(ValueError, match='injected'):
        migration.merge(source, task='rollback')
    assert service._admission.store.status('rollback')['phase'] == 'restored'
    assert {r.id: r for r in service._get_shared_vector_index().iter_records()} == before
    assert list(AGFSStorage(service._cfg, client).records('nodes')) == original
    assert json.loads(service._admission.store.status('rollback')['backup'])['vectors']['records'] == 4


def test_interrupted_import_restores_on_explicit_recovery(instance, tmp_path, monkeypatch):
    service, migration, client = instance
    source = package(tmp_path)
    original_verify = Rebuilder.verify
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(Rebuilder, 'verify', interrupt)
    with pytest.raises(KeyboardInterrupt):
        migration.merge(source, task='interrupted')
    assert service._admission.store.status()['modified']
    monkeypatch.setattr(Rebuilder, 'verify', original_verify)
    assert migration.recover(source, task='interrupted')['phase'] == 'restored'
    assert list(AGFSStorage(service._cfg, client).records('nodes')) == []
    assert service._get_shared_vector_index().count() == 0


def test_unknown_vector_write_completion_never_starts_rollback(instance, tmp_path, monkeypatch):
    from providers.vector_index.chroma_index import ChromaVectorIndex
    service, migration, client = instance
    migration.timeout = .1
    def timeout(*args):
        raise TimeoutError('write completion unknown')
    monkeypatch.setattr(ChromaVectorIndex, 'upsert', timeout)
    with pytest.raises(TimeoutError):
        migration.merge(package(tmp_path), task='unknown')
    assert service._admission.store.status()['phase'] == 'modifying'
    with service._admission.store._transaction() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0] >= 1
    # Clearing canonical data would be unsafe while the mutation is unresolved.
    assert len(list(AGFSStorage(service._cfg, client).records('nodes'))) == 2


def test_missing_target_vector_rejects_before_backup_or_business_write(instance, tmp_path, monkeypatch):
    from service.migration.package import PackageError

    service, migration, client = instance
    source = package(tmp_path)
    migration.merge(source, task='seed')
    vector = service._get_shared_vector_index()
    vector.delete([next(iter(vector.iter_records())).id])
    before = list(AGFSStorage(service._cfg, client).records('nodes'))
    with pytest.raises(PackageError, match='repair the target index'):
        migration.preview(source)
    monkeypatch.setattr('service.migration.instance.backup_components',
                        lambda *a, **kw: pytest.fail('incomplete target must reject before backup'))
    with pytest.raises(PackageError, match='repair the target index'):
        migration.merge(source, task='missing-index')
    state = service._admission.store.status('missing-index')
    assert state['modified'] == 0 and state['backup'] is None
    assert service._admission.store.status() is None
    assert service._get_shared_vector_index().count() == 3
    assert list(AGFSStorage(service._cfg, client).records('nodes')) == before


def test_initialized_content_reader_remains_available_during_export(instance):
    service, migration, client = instance
    fs = service._get_context_fs()
    with migration.gate.execution():
        migration.gate.enter('read-during-export', mode='export', resources={})
        try:
            with service._admission.scope():
                assert service._get_context_fs() is fs
        finally:
            migration.gate.reopen('read-during-export')


def test_close_failure_cannot_reopen_with_old_index(instance, tmp_path, monkeypatch):
    _, migration, _ = instance
    source = package(tmp_path)
    failed = []
    original = Rebuilder.close
    def fail_once(self):
        if not failed:
            failed.append(self)
            raise RuntimeError('injected close failure')
        return original(self)
    monkeypatch.setattr(Rebuilder, 'close', fail_once)
    try:
        with pytest.raises(RuntimeError, match='injected close failure'):
            migration.merge(source, task='close-failed')
        assert migration.gate.status() is not None
        migration.recover(source, task='close-failed')
        assert migration.gate.status() is not None or failed[0].vector is None, 'admission reopened while migration vector remained live'
    finally:
        for obj in failed:
            original(obj)
