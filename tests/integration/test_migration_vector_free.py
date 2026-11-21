"""Real write/search migration flow on an isolated HTTP AGFS service."""
import json
import os
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from core.models import RequestContext, Role
from providers.config import ProviderConfig
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from server.service_factory import close_service, create_service
from service.migration.instance import InstanceMigration
from service.migration.package import open_package
from service.migration.rebuild import Rebuilder
from service.migration.storage import storage
from tests.unit.commit.test_memory_identity import memory


@pytest.mark.parametrize('source_mode,target_mode,fts', [
    ('agfs', 'agfs', 'literal'), ('agfs', 'agfs', 'sqlite_bm25'),
    ('agfs', 'agfs', 'disabled'), ('default', 'agfs', 'literal'),
    ('agfs', 'default', 'literal'),
])
def test_vector_free_migration_and_recovery(tmp_path, monkeypatch, source_mode, target_mode, fts):
    if 'default' in (source_mode, target_mode):
        pytest.importorskip('chromadb')
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda s: None)
    remote = os.environ.get('MIGRATION_TEST_AGFS_URL')
    if not remote:
        pytest.skip('MIGRATION_TEST_AGFS_URL required')
    services = []
    configs = []
    ctx = RequestContext('acme', 'alice', '', 'migration', 'test', Role.ROOT)
    def create(mode, name):
        cfg = OgMemConfig(storage_backend='agfs', retrieval_mode=mode,
            provider='mock', embedding_provider='mock', skill_sedimentation_enabled=False,
            account_id='acme', user_id='alice', vector_db_type='chroma', opengauss_dimension=3,
            retrieval_agfs_fts_enabled=fts != 'disabled',
            retrieval_agfs_fts_ranker='literal' if fts == 'disabled' else fts,
            agfs_base_url=remote or 'http://test', agfs_mount_prefix='/local/migration_' + uuid4().hex,
            chroma_persist_directory=str(tmp_path / (name + '-chroma')),
            migration_workspace=str(tmp_path / (name + '-control')))
        with storage(cfg) as adapter:
            adapter._mkdir(cfg.agfs_mount_prefix + '/accounts')
        service = create_service(cfg)
        services.append(service)
        configs.append(cfg)
        return service
    def write(service, key, text):
        return service.get_write_api().write_memory(replace(
            memory(key, category='preference', content=text), provenance_ids=[]), ctx)['target_uri']
    def search(service, term):
        return service.get_read_api().search_memory(term, ctx, top_k=10)
    def warm(service, term):
        for _ in range(100):
            result = search(service, term)
            cache = service._agfs_fts
            if cache is None or cache._snapshots:
                return result
            time.sleep(.02)
        pytest.fail('FTS cache did not warm')
    def forbidden(*args, **kwargs):
        pytest.fail('vector-free migration must not create embedding or vector providers')
    try:
        # For two vector-free endpoints, guard even service startup and export.
        with monkeypatch.context() as guard:
            if source_mode == target_mode == 'agfs':
                guard.setattr(ProviderConfig, 'create_embedder', forbidden)
                guard.setattr(ProviderConfig, 'create_vector_index', forbidden)
            source = create(source_mode, 'source')
            uri = write(source, 'incoming', 'salamander migration preference')
            package = tmp_path / 'source.tar.gz'
            InstanceMigration(source).export(package, task='export')
            with open_package(package) as data:
                assert data.manifest['source_environment']['embedding']['provider'] == (
                    'none' if source_mode == 'agfs' else 'mock')
            close_service(source)
            services.remove(source)
            target = create(target_mode, 'target')
            original = write(target, 'original', 'original tortoise preference')
            if target_mode == 'agfs':
                assert warm(target, 'tortoise').hits
                old_cache = target._agfs_fts
                # The source may have used vectors, but the target must not.
                guard.setattr(ProviderConfig, 'create_embedder', forbidden)
                guard.setattr(ProviderConfig, 'create_vector_index', forbidden)
            migration = InstanceMigration(target)
            # Normal outbox writes are asynchronous in vector mode.
            for attempt in range(100):
                try:
                    preview = migration.preview(package)
                    break
                except ValueError:
                    if attempt == 99:
                        raise
                    time.sleep(.05)
            assert preview['counts']['new'] >= 1
            if target_mode == 'agfs':
                assert preview['compatibility']['target']['vector_storage']['backend'] == 'none'
                assert any('不创建模型' in notice for notice in preview['compatibility']['notices'])
            with monkeypatch.context() as failure:
                failure.setattr(Rebuilder, 'verify', lambda *a: (_ for _ in ()).throw(RuntimeError('injected')))
                with pytest.raises(RuntimeError, match='injected'):
                    migration.merge(package, task='rollback')
            assert migration.gate.status() is None
            assert target.get_read_api().read_memory(original, ctx).content_excerpt == 'original tortoise preference'
            if target_mode == 'agfs':
                assert warm(target, 'tortoise').hits
                assert not search(target, 'salamander').hits
                assert target._agfs_fts is not old_cache or old_cache is None
            result = migration.merge(package, task='import')
            assert result['phase'] == 'succeeded'
            assert target.get_read_api().read_memory(uri, ctx).content_excerpt == 'salamander migration preference'
            assert warm(target, 'salamander').hits
            if target_mode == 'agfs':
                assert result['result'] and json.loads(result['result'])['index_records'] == 0
                assert 'vectors' not in json.loads(result['backup'])
            else:
                assert target._get_shared_vector_index().count() > 0
    finally:
        for service in services:
            close_service(service)
        if remote:
            for cfg in configs:
                with storage(cfg) as adapter:
                    adapter.client.rm(cfg.agfs_mount_prefix, recursive=True)
