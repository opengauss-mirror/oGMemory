"""Exercise the complete merge executor using local files and real Chroma."""
import json
from contextlib import contextmanager
from copy import deepcopy

import pytest

from providers.unified_config import OgMemConfig
from service.migration.engine import MigrationService, inventory
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.package import PackageError, write_package
from service.migration.rebuild import Rebuilder
from service.migration.recovery import RecoveryImport
from service.migration.storage import AGFSStorage
from service.migration.tree_backup import AGFSTreeBackup
from service.migration.vector_backup import VectorBackup
from tests.unit.service.migration.test_engine import node
from tests.unit.service.migration.test_tree_backup import LocalClient


@pytest.fixture
def instance(tmp_path):
    pytest.importorskip('chromadb')
    cfg = OgMemConfig(storage_backend='agfs', agfs_mount_prefix='/local',
                     provider='mock', vector_db_type='chroma', opengauss_dimension=3,
                     chroma_persist_directory=str(tmp_path / 'chroma'),
                     retrieval_hybrid_enabled=False, graph_backend='none')
    client = LocalClient(tmp_path / 'files')
    client.path('/local/accounts/acme/users').mkdir(parents=True)
    adapter = AGFSStorage(cfg, client)
    existing = node()
    existing['metadata']['memory_write_receipts'] = {'original': 'retained'}
    adapter.restore('nodes', [existing])
    indexes = Rebuilder(cfg)
    indexes.probe()
    with inventory(tmp_path / 'original.db', adapter.records) as inv:
        indexes.rebuild(inv, lambda **_: None)
    # These executor tests supply injected lifecycle callbacks and share the
    # fixture's index. Native close/reopen is exercised by test_instance instead.
    release_indexes = indexes.close
    indexes.close = lambda: None
    gate = MaintenanceStore(tmp_path / 'gate.db', 'local-test')

    @contextmanager
    def recovery(request_identity, incoming):
        yield RecoveryImport(gate, tmp_path / 'backups', {
            'content': AGFSTreeBackup(client, ['/local/accounts/acme/users'], lambda: {'root': 'users'}),
            'vectors': VectorBackup(indexes.vector, lambda: {'collection': str(indexes.vector._collection.id)}),
        }, quiesce=lambda: True, refresh=lambda: None, request_identity=request_identity)

    @contextmanager
    def storage(_):
        yield adapter
    service = MigrationService(cfg, adapter_factory=storage, index_factory=lambda _: indexes)
    yield service, recovery, gate, adapter, indexes
    release_indexes()


def package(tmp_path):
    incoming, added = node(), node()
    incoming['content'] = '新增的偏好内容'
    added['uri'] = added['uri'].replace('/editor', '/keyboard')
    path = tmp_path / 'source.tar.gz'
    write_package(path, {'nodes': [incoming, added], 'archives': [], 'relations': []},
                  source_backend='agfs', source_version='test')
    return path


def test_merge_preserves_runtime_and_repeated_package_is_noop(instance, tmp_path):
    service, recovery, gate, adapter, indexes = instance
    path = package(tmp_path)
    phases = []
    def progress(**event):
        phases.append(event['phase'])
        if event['phase'] == 'merging':
            with pytest.raises(MaintenanceError):
                gate.admit()
    state = service.merge(path, task='first', recovery_factory=recovery, llm=None, progress=progress)
    assert state['phase'] == 'succeeded'
    assert gate.status() is None
    assert 'backing_up' in phases and 'verifying' in phases
    records = list(adapter.records('nodes'))
    assert len(records) == 2
    assert indexes.vector.count() == 4
    meta_path = '/local/accounts/acme/users/alice/memories/preferences/editor/.meta.json'
    assert json.loads(adapter.client.read(meta_path))['memory_write_receipts'] == {'original': 'retained'}
    before = deepcopy(records)
    repeated = service.merge(path, task='second', recovery_factory=recovery, llm=None)
    assert repeated['phase'] == 'succeeded'
    assert json.loads(repeated['result'])['changes']['skipped'] == 2
    assert list(adapter.records('nodes')) == before


def test_failed_verification_restores_data_vectors_and_no_receipts(instance, tmp_path, monkeypatch):
    service, recovery, gate, adapter, indexes = instance
    original = list(adapter.records('nodes'))
    vectors = list(indexes.vector.iter_records())
    def failed(_):
        raise PackageError('injected verification failure')
    monkeypatch.setattr(indexes, 'verify', failed)
    with pytest.raises(PackageError, match='injected'):
        service.merge(package(tmp_path), task='failed', recovery_factory=recovery, llm=None)
    assert gate.status('failed')['phase'] == 'restored'
    assert gate.status() is None
    assert list(adapter.records('nodes')) == original
    assert {r.id: r for r in indexes.vector.iter_records()} == {r.id: r for r in vectors}
    with gate._transaction() as db:
        assert db.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0


def test_conflict_policy_runs_only_after_verified_backup(instance, tmp_path, monkeypatch):
    from service.migration.merge import PolicyConflictResolver

    service, recovery, gate, adapter, indexes = instance
    original = list(adapter.records('nodes'))
    def reject(self, existing, incoming):
        state = gate.status()
        assert state['phase'] == 'modifying'
        assert set(json.loads(state['backup'])) == {'content', 'vectors'}
        raise PackageError('unsupported conflict')
    monkeypatch.setattr(PolicyConflictResolver, '__call__', reject)
    with pytest.raises(PackageError, match='unsupported conflict'):
        service.merge(package(tmp_path), task='policy', recovery_factory=recovery, llm=None)
    assert gate.status('policy')['phase'] == 'restored'
    assert gate.status() is None
    assert list(adapter.records('nodes')) == original


def test_unknown_existing_model_rejected_before_backup_or_changes(instance, tmp_path, monkeypatch):
    service, recovery, gate, adapter, indexes = instance
    original = list(adapter.records('nodes'))
    monkeypatch.setattr(type(indexes.embedder), "model", property(lambda self: None))
    with pytest.raises(PackageError, match='existing vectors'):
        service.merge(package(tmp_path), task='unknown-model', recovery_factory=recovery, llm=None)
    assert gate.status('unknown-model')['phase'] == 'unchanged'
    assert gate.status('unknown-model')['backup'] is None
    assert list(adapter.records('nodes')) == original


def test_merge_keeps_target_attachment_with_original_filename(instance, tmp_path):
    service, recovery, gate, adapter, indexes = instance
    assets = tmp_path / 'assets'
    assets.mkdir()
    original = assets / 'manual.txt'
    original.write_text('original attachment')
    old = list(adapter.records('nodes'))[0]
    old['metadata']['source_path'] = str(original)
    adapter.merge_nodes([old])
    service.asset_root = assets
    from service.migration.assets import Assets
    normalizer = Assets(assets, tmp_path)
    with inventory(tmp_path / 'attached.db', lambda k: (
            normalizer.capture(r, copy_bytes=False) if k == 'nodes' else r
            for r in adapter.records(k))) as inv:
        indexes.rebuild(inv, lambda **_: None)
    incoming = node()
    incoming['content'] = 'New preference'
    path = tmp_path / 'update.tar.gz'
    write_package(path, {'nodes': [incoming], 'archives': [], 'relations': []},
                  source_backend='agfs', source_version='test')
    result = service.merge(path, task='attached', recovery_factory=recovery, llm=None)
    assert result['phase'] == 'succeeded'
    restored = list(adapter.records('nodes'))[0]
    assert restored['metadata']['source_path'] == str(original)
    assert original.read_text() == 'original attachment'
    assert gate.status() is None
