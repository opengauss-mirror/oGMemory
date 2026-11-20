"""Real shared-catalog producer/adapter/reader contract on isolated databases."""
import os
import time
from contextlib import closing
from dataclasses import replace
from uuid import uuid4

import pytest


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
@pytest.mark.parametrize('authority', ['same', 'admin_changed', 'agent_owner_changed'])
@pytest.mark.parametrize('workflow', ['adapter', 'online', 'dependency'])
def test_shared_catalog_preserves_produced_content_and_evidence(tmp_path, backend, authority, workflow, monkeypatch):
    dsn = os.environ.get('MIGRATION_TEST_ADMIN_DSN')
    if not dsn:
        pytest.skip('MIGRATION_TEST_ADMIN_DSN selects a dedicated admin database')
    if backend == 'agfs' and not os.environ.get('AGFS_NATIVE_LIBRARY'):
        pytest.skip('AGFS_NATIVE_LIBRARY selects a real native AGFS runtime')
    import psycopg2
    import yaml
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn

    from core.models import RequestContext, Role
    from providers.unified_config import OgMemConfig
    from server.service_factory import close_service, create_service
    from service.migration.engine import MigrationService, inventory
    from service.migration.instance import InstanceMigration
    from service.migration.package import PackageError, open_package
    from service.migration.rebuild import Rebuilder
    from service.migration.shared_storage import SharedStorage
    from service.migration.space_catalog import COLUMNS, SpaceCatalog, record_key
    from service.migration.space_inventory import SpaceInventory
    from service.migration.space_selection import select_spaces
    from service.shared_memory import SharedMemoryRetrievalService
    from service.space_lifecycle import SpaceMemoryWorker
    from tests.integration.test_shared_space_lifecycle import EvidenceLLM, project_items
    from tests.unit.commit.test_memory_identity import memory

    # This suite executes the real worker explicitly with EvidenceLLM.
    # Do not let an automatic worker claim the same job with another model.
    monkeypatch.setattr(SpaceMemoryWorker, 'start', lambda self: None)
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    names, services, configs = [], [], []
    native = tmp_path / 'native.yaml'
    native_data = tmp_path / 'content'
    native_data.mkdir()
    mount = '/migration-' + uuid4().hex
    native.write_text(yaml.safe_dump({'plugins': {'localfs': {
        'enabled': True, 'path': mount, 'config': {'local_dir': str(native_data)},
    }}}))
    try:
        with admin.cursor() as cur:
            cur.execute('SELECT version()')
            dialect = 'opengauss' if 'opengauss' in cur.fetchone()[0].lower() else 'postgresql'
            for side in ('source', 'target'):
                name = 'migration_' + side + '_' + uuid4().hex[:10]
                cur.execute(sql.SQL('CREATE DATABASE {}' + (" DBCOMPATIBILITY 'PG'" if dialect == 'opengauss' else '')).format(sql.Identifier(name)))
                names.append(name)
                connection = make_dsn(dsn, dbname=name)
                cfg = OgMemConfig(storage_backend=backend, shared_spaces_enabled=True,
                    sql_connection_string=connection, sql_dialect=dialect,
                    provider='mock', embedding_provider='mock', vector_db_type=os.environ.get('MIGRATION_TEST_VECTOR_BACKEND', 'opengauss'),
                    chroma_persist_directory=str(tmp_path / ('chroma-' + side)),
                    chroma_collection_name='shared_migration',
                    milvus_uri=os.environ.get('OGMEM_TEST_MILVUS_URI', 'http://127.0.0.1:19530'),
                    milvus_collection_name='shared_' + name,
                    vector_db_dialect=dialect, opengauss_connection_string=connection,
                    opengauss_dimension=3, directory_summary_enabled=False,
                    skill_sedimentation_enabled=False, root_api_key='isolated-root',
                    role_control_enabled=True, account_id='acme', user_id='alice',
                    migration_workspace=str(tmp_path / side), agfs_transport='native',
                    agfs_native_library=os.environ.get('AGFS_NATIVE_LIBRARY', ''),
                    agfs_native_config=str(native), agfs_mount_prefix=mount + '/' + side)
                configs.append(cfg)
                service = create_service(cfg)
                services.append(service)
                service.get_key_manager().create_account('acme', 'alice')
                service.get_key_manager().register_user('acme', 'bob',
                    role='admin' if authority == 'admin_changed' and side == 'source' else 'user')
                service.get_key_manager().register_user('acme', 'outsider')
        ctx = RequestContext('acme', 'alice', '', 'migration', 'test', Role.MEMBER)
        result = services[0].get_write_api().write_memory(
            replace(memory('editor', category='preference', content='Prefer compact explanations.'), provenance_ids=[]), ctx)
        uri = result['target_uri']
        repo = services[0].get_space_repository()
        repo.create_space(ctx, {'space_id': 'team', 'name': 'Team'})
        repo.set_members(ctx, 'team', {'members': [
            {'principal': 'user:alice', 'role': 'manager'},
            {'principal': 'user:bob', 'role': 'member'},
        ]}, expected_rev=1)
        grantor = replace(ctx, user_id='bob', role=Role.ADMIN) if authority == 'admin_changed' else ctx
        repo.put_source(grantor, 'team', 'alice-source', {'principal': 'user:alice', 'memory_types': ['preference']}, expected_rev=0)
        if authority == 'agent_owner_changed':
            for service, owner in zip(services, ('bob', 'alice'), strict=True):
                with service.get_space_repository().transaction('acme') as cur:
                    cur.execute("INSERT INTO cp_agents(account_id,agent_id,owner_user_id) VALUES('acme','worker',%s)", (owner,))
            repo.set_members(ctx, 'team', {'members': [
                {'principal': 'user:alice', 'role': 'manager'}, {'principal': 'user:bob', 'role': 'member'},
                {'principal': 'agent:worker', 'role': 'member'},
            ]}, expected_rev=repo.permissions(ctx, 'team')['authz_rev'])
            repo.put_source(replace(ctx, user_id='bob'), 'team', 'agent-source',
                            {'principal': 'agent:worker', 'memory_types': ['case']}, expected_rev=0)
        repo.save_rule(ctx, 'team', 'preferences', {'name': 'Shared preferences', 'spec': {
            'memory_types': ['preference'], 'description': 'Reusable preferences',
            'trigger_policy': {'debounce_seconds': 0, 'min_interval_seconds': 0, 'reconcile_seconds': 1},
        }}, expected_rev=0)
        worker = SpaceMemoryWorker(repo, EvidenceLLM(), worker_count=1)
        repo.dispatch_source_events(ctx.account_id)
        with repo.transaction(ctx.account_id) as cur:
            cur.execute('UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s', (ctx.account_id,))
        repo.reconcile_due_rules(ctx.account_id)
        job = repo.claim_job(ctx.account_id)
        assert job is not None
        worker._execute(job)
        completed = next(row for row in repo.list_jobs(ctx, 'team')['items'] if row['job_id'] == job['job_id'])
        assert completed['state'] == 'SUCCEEDED', completed.get('last_error')
        assert len(completed['result']['items']) == 1
        project_items(repo, ctx)
        assert SharedMemoryRetrievalService(repo).search(replace(ctx, user_id='bob'), 'compact', ['team'])['total'] == 1
        dependent_uri = None
        if workflow == 'dependency':
            from core.provenance_resolver import ProvenanceResolver
            shared_uri = completed['result']['items'][0]['uri']
            dependent = replace(memory('shared-reference', category='preference', content='Follow the shared explanation style.'),
                provenance_ids=[ProvenanceResolver.build_id('memory', shared_uri, 'L2')])
            dependent_uri = services[0].get_write_api().write_memory(dependent, ctx)['target_uri']
        package_path = tmp_path / 'shared.tar.gz'
        if workflow != 'adapter':
            report = InstanceMigration(services[0]).export(package_path, task='snapshot-' + uuid4().hex)
        for service in services:
            close_service(service)
        services.clear()
        if workflow == 'adapter':
            report = MigrationService(configs[0]).export(package_path)
        assert report['shared_spaces'] == {'spaces': 1, 'action': 'exported'}
        with open_package(package_path) as package:
            assert package.manifest['format_version'] == 2
            packaged_nodes = list(package.records('space_nodes'))
            packaged_catalog = list(package.records('spaces'))
        if workflow != 'adapter':
            service = create_service(configs[1])
            services.append(service)
            migration = InstanceMigration(service)
            if workflow == 'merge':
                service.get_write_api().write_memory(
                    replace(memory('editor', category='preference', content='Prefer detailed explanations.'), provenance_ids=[]), ctx)
                target_repo = service.get_space_repository()
                target_repo.create_space(ctx, {'space_id': 'existing', 'name': 'Existing'})
                target_repo.set_members(ctx, 'existing', {'members': [
                    {'principal': 'user:alice', 'role': 'manager'},
                    {'principal': 'user:bob', 'role': 'member'},
                ]}, expected_rev=1)
                target_repo.put_source(ctx, 'existing', 'alice-source',
                    {'principal': 'user:alice', 'memory_types': ['preference']}, expected_rev=0)
                target_repo.save_rule(ctx, 'existing', 'preferences', {'name': 'Preferences', 'spec': {
                    'memory_types': ['preference'], 'description': 'Reusable preferences',
                    'trigger_policy': {'debounce_seconds': 0, 'min_interval_seconds': 0, 'reconcile_seconds': 1},
                }}, expected_rev=0)

                def recompute(*, background=False):
                    target_repo.dispatch_source_events('acme')
                    with target_repo.transaction('acme') as cur:
                        cur.execute('UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s', ('acme',))
                    target_repo.reconcile_due_rules('acme')
                    job = target_repo.claim_job('acme')
                    assert job is not None
                    SpaceMemoryWorker(target_repo, EvidenceLLM(), worker_count=1)._execute(job)
                    assert next(row for row in target_repo.list_jobs(ctx, 'existing')['items']
                                if row['job_id'] == job['job_id'])['state'] == 'SUCCEEDED'
                    if not background:
                        project_items(target_repo, ctx)

                recompute()
                with closing(SharedStorage(configs[1])) as existing:
                    before = next(row for row in existing.records('nodes') if row['uri'] == uri)
                    with inventory(tmp_path / 'seed-index.sqlite', existing.records,
                                   shared_records=existing.records('space_nodes')) as inv:
                        with closing(Rebuilder(configs[1])) as index:
                            index.probe()
                            index.rebuild(inv, lambda **kw: None)
                search = SharedMemoryRetrievalService(target_repo)
                assert search.search(replace(ctx, user_id='bob'), 'detailed', ['existing'])['total'] == 1
                choice = migration.preview(package_path)['shared_spaces']
                assert choice['reason'] == 'target_business_not_empty'
                assert migration.merge(package_path, task='ordinary-merge-' + uuid4().hex,
                                       shared_selection=choice)['phase'] == 'succeeded'
                target_repo = service.get_space_repository()
                search = SharedMemoryRetrievalService(target_repo)
                with closing(SharedStorage(configs[1])) as existing:
                    after = next(row for row in existing.records('nodes') if row['uri'] == uri)
                assert after['metadata']['shared_source_epoch'] == before['metadata']['shared_source_epoch']
                assert after['version'] > before['version']
                assert search.search(replace(ctx, user_id='bob'), 'detailed', ['existing'])['total'] == 0
                recompute(background=True)
                search = service.get_shared_memory_retrieval(mode='semantic')
                # Verify the resumed real worker and the configured vector
                # backend, waiting for normal asynchronous publication.
                deadline = time.monotonic() + 10
                while True:
                    result = search.search(replace(ctx, user_id='bob'), 'compact', ['existing'], mode='semantic')
                    if result['total'] == 1 or time.monotonic() >= deadline:
                        break
                    time.sleep(.05)
                assert result['total'] == 1, result
                return
            if dependent_uri and authority != 'same':
                with pytest.raises(PackageError, match='depend on skipped'):
                    migration.preview(package_path)
                assert service._get_shared_vector_index().count() == 0
                return
            preview = migration.preview(package_path)
            choice = preview['shared_spaces']
            assert choice['action'] == ('import' if authority == 'same' else 'skipped')
            if authority == 'same' and workflow == 'online':
                # Change real target identity state after preview, in both
                # directions; neither stale selection may write business data.
                for status in ('deleted', 'active'):
                    confirmed = migration.preview(package_path)['shared_spaces']
                    with service.get_space_repository().transaction('acme') as cur:
                        cur.execute("UPDATE cp_users SET status=%s WHERE account_id='acme' AND user_id='bob'", (status,))
                    with pytest.raises(PackageError, match='selection changed'):
                        migration.merge(package_path, task='changed-' + uuid4().hex, shared_selection=confirmed)
                    with closing(SharedStorage(configs[1])) as unchanged:
                        assert list(unchanged.records('nodes')) == []
                        assert list(SpaceCatalog(unchanged).records()) == []
                    assert service._get_shared_vector_index().count() == 0
            if authority == 'same':
                def reject_verification(self, inv):
                    if workflow == 'dependency':
                        raise SystemExit('injected executor interruption')
                    raise RuntimeError('injected failure after shared restore and index writes')
                failed_task = 'shared-failed-' + uuid4().hex
                with monkeypatch.context() as patch:
                    patch.setattr(Rebuilder, 'verify', reject_verification)
                    with pytest.raises(SystemExit if workflow == 'dependency' else RuntimeError, match='injected'):
                        migration.merge(package_path, task=failed_task, shared_selection=choice)
                if workflow == 'dependency':
                    assert migration.gate.status(failed_task)['modified']
                    assert migration.recover(package_path, task=failed_task)['phase'] == 'restored'
                assert migration.gate.status(failed_task)['phase'] == 'restored'
                with closing(SharedStorage(configs[1])) as restored:
                    assert list(restored.records('nodes')) == []
                    assert list(restored.records('space_nodes')) == []
                    assert list(SpaceCatalog(restored).records()) == []
                assert service._get_shared_vector_index().count() == 0
            task = 'shared-' + uuid4().hex
            result = migration.merge(package_path, task=task, shared_selection=choice)
            assert result['phase'] == 'succeeded'
            assert migration.merge(package_path, task=task, shared_selection=choice)['phase'] == 'succeeded'
            assert service.get_read_api().read_memory(uri, ctx).content_excerpt == 'Prefer compact explanations.'
            if dependent_uri:
                assert service.get_read_api().read_memory(dependent_uri, ctx).content_excerpt == 'Follow the shared explanation style.'
            for mode in ('keyword', 'semantic', 'hybrid'):
                search = service.get_shared_memory_retrieval(mode)
                found = search.search(replace(ctx, user_id='bob'), 'compact', ['team'], mode=mode)
                assert found['total'] == int(authority == 'same')
                assert search.search(replace(ctx, user_id='outsider'), 'compact', ['team'], mode=mode)['total'] == 0
            if authority == 'same':
                assert service.get_space_repository().list_rules(ctx, 'team')['items'][0]['state'] == 'PAUSED'
                # A second task sees a nonempty target and skips the whole space.
                if dependent_uri:
                    with pytest.raises(PackageError, match='depend on skipped'):
                        migration.preview(package_path)
                    return
                preview = migration.preview(package_path)
                assert preview['shared_spaces']['reason'] == 'target_business_not_empty'
                repeated = migration.merge(package_path, task='shared-repeat-' + uuid4().hex,
                                           shared_selection=preview['shared_spaces'])
                assert repeated['phase'] == 'succeeded'
            return
        source, target = (SharedStorage(cfg) for cfg in configs)
        try:
            records = list(source.records('nodes'))
            record = next(row for row in records if row['uri'] == uri)
            assert record['metadata']['shared_source_epoch']
            assert 'agfs_content' not in record['metadata']
            source_catalog, target_catalog = SpaceCatalog(source), SpaceCatalog(target)
            assert source_catalog.description() == target_catalog.description()
            catalog = list(source_catalog.records())
            shared_nodes = list(source.records('space_nodes'))
            assert packaged_nodes == shared_nodes
            def order(entry):
                return entry['table'], record_key(entry['table'], entry['row'])
            assert sorted(packaged_catalog, key=order) == sorted(catalog, key=order)
            assert len(shared_nodes) == 1
            with inventory(tmp_path / 'ordinary.sqlite', source.records) as ordinary:
                spaces = SpaceInventory(tmp_path / 'spaces.sqlite', catalog, shared_nodes)
                try:
                    spaces.validate_dependencies(ordinary)
                    with inventory(tmp_path / 'empty-target.sqlite', target.records) as empty_target:
                        class EmptyVector:
                            def count(self):
                                return 0
                        selection = select_spaces({'source_backend': backend,
                            'shared_spaces': source_catalog.description()}, spaces, configs[1],
                            target, empty_target, EmptyVector())
                    if authority != 'same':
                        assert selection['action'] == 'skipped'
                        assert selection['reason'] == 'target_source_authority_changed'
                        assert list(target.records('nodes')) == []
                        assert list(target.records('space_nodes')) == []
                        assert list(target_catalog.records()) == []
                        return
                    assert selection['action'] == 'import'
                finally:
                    spaces.close()
            target.restore('nodes', records)
            assert list(target.records('nodes')) == records
            target.restore('space_nodes', shared_nodes)
            def reader(table):
                return (record['row'] for record in catalog if record['table'] == table)
            target_catalog.restore(reader)
            actual = list(target_catalog.records())
            for table in COLUMNS:
                expected = list(reader(table))
                if table in {'space_rules', 'space_rule_revisions'}:
                    expected = [{**row, 'state': 'PAUSED'} if row['state'] == 'ACTIVE' else row for row in expected]
                assert [record['row'] for record in actual if record['table'] == table] == expected
            # This is the normal service content reader, not a SQL text query.
            service = create_service(configs[1])
            services.append(service)
            read = service.get_read_api().read_memory(uri, ctx)
            assert read.content_excerpt == 'Prefer compact explanations.'
            target_repo = service.get_space_repository()
            bob = replace(ctx, user_id='bob')
            assert target_repo.permissions(bob, 'team')['capabilities']['search']
            assert target_repo.list_rules(ctx, 'team')['items'][0]['state'] == 'PAUSED'
            search = SharedMemoryRetrievalService(target_repo)
            assert search.search(bob, 'compact', ['team'])['total'] == 0
            index = Rebuilder(configs[1])
            try:
                index.probe()
                with pytest.raises(PackageError, match='index record'):
                    target.publish_space_projections(shared_nodes, index.vector)
                # Reuse the ordinary embedding/upsert path with the shared records.
                with inventory(tmp_path / 'rebuild.sqlite', lambda kind: ()) as empty:
                    index.rebuild(empty, lambda **kw: None, changed=lambda: iter(shared_nodes))
                target.publish_space_projections(shared_nodes, index.vector)
            finally:
                index.close()
            provider = configs[1].to_provider_config()
            vector, embedder = provider.create_vector_index(), provider.create_embedder()
            try:
                search = SharedMemoryRetrievalService(target_repo, vector, embedder)
                for mode in ('keyword', 'semantic', 'hybrid'):
                    result = search.search(bob, 'compact', ['team'], mode=mode)
                    assert result['total'] == 1
                    assert result['items'][0]['content'] == 'Prefer compact explanations.'
                    assert result['items'][0]['source_count'] == 1
                    assert search.search(replace(ctx, user_id='outsider'), 'compact', ['team'], mode=mode)['total'] == 0
                with target_repo.transaction('acme', write=False) as cur:
                    cur.execute('SELECT count(*) AS n FROM space_jobs')
                    assert cur.fetchone()['n'] == 0
            finally:
                close_vector = getattr(vector, 'release_for_migration', None) or getattr(vector, 'close', None)
                if callable(close_vector):
                    close_vector()
                close = getattr(embedder, 'close', None)
                if callable(close):
                    close()
            if backend == 'agfs':
                with target.conn.cursor() as cur:
                    cur.execute('SELECT metadata FROM context_nodes WHERE uri=%s', (uri,))
                    pointer = cur.fetchone()[0]['agfs_content']
                assert pointer['path'].startswith(mount + '/target/')
        finally:
            source.close()
            target.close()
    finally:
        for service in services:
            close_service(service)
        with admin.cursor() as cur:
            for name in reversed(names):
                cur.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
        admin.close()


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
def test_ordinary_conflict_preserves_target_shared_source_identity(tmp_path, backend, monkeypatch):
    test_shared_catalog_preserves_produced_content_and_evidence(tmp_path, backend, 'same', 'merge', monkeypatch)
