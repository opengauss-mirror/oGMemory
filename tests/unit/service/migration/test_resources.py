from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from service.migration.package import PackageError
from service.migration.resources import agfs_business_roots, sql_business_tables, vector_backup
from service.migration.storage import AGFSStorage
from tests.unit.service.migration.test_engine import Files, config
from tests.unit.service.migration.test_vector_backup import record


def test_agfs_scope_keeps_control_plane_and_covers_new_accounts():
    files = Files()
    for path in ('/local/accounts/acme/_system', '/local/accounts/acme/users', '/local/accounts/old/_system'):
        files.mkdir(path)
    source = SimpleNamespace(records=lambda kind: [{'account_id': 'new'}] if kind == 'nodes' else [])
    roots = agfs_business_roots(AGFSStorage(config(), files), source)
    assert '/local/accounts/new/users' in roots
    assert '/local/accounts/new/agents' in roots
    assert '/local/accounts/old/sessions' in roots
    assert '/local/.ogmemory/trace-store' in roots
    assert all('/_system' not in root and not root.endswith('/accounts') for root in roots)
    assert all(not other.startswith(root + '/') for root in roots for other in roots if other != root)


def test_sql_scope_excludes_credentials_audit_and_other_app_tables():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.__iter__.return_value = iter((name,) for name in [
        'context_nodes', 'relation_edges', 'outbox_events', 'session_archives', 'edge_projection_jobs',
        'cp_accounts', 'cp_users', 'cp_agents', 'cp_audit_logs', 'another_application', 'trace_events'])
    tables = sql_business_tables(conn)
    assert 'trace_events' in tables
    assert not any(name.startswith('cp_') for name in tables)
    assert 'another_application' not in tables


def test_agfs_recovery_scope_keeps_skipped_source_accounts():
    source = SimpleNamespace(records=lambda kind: [], recovery_accounts=frozenset({'shared-only'}))
    roots = agfs_business_roots(AGFSStorage(config(), Files()), source)
    assert '/local/accounts/shared-only/.space-content' in roots
    assert '/local/accounts/shared-only/users' in roots


def test_recreated_chroma_collection_is_rejected_before_restore(tmp_path):
    pytest.importorskip('chromadb')
    from providers.vector_index.chroma_index import ChromaVectorIndex

    vector = ChromaVectorIndex(collection_name='resource-test', persist_directory=str(tmp_path / 'chroma'), dimension=3)
    vector.upsert([record()])
    backup = vector_backup(vector)
    manifest = backup.capture(tmp_path / 'backup.sqlite')
    original = vector.migration_identity()
    vector._client.delete_collection('resource-test')
    replacement = ChromaVectorIndex(collection_name='resource-test', persist_directory=str(tmp_path / 'chroma'), dimension=3)
    replacement.upsert([record()])
    assert vector.migration_identity()['collection_id'] != original['collection_id']
    with pytest.raises(PackageError, match='target differs'):
        backup.restore(tmp_path / 'backup.sqlite', manifest)
    assert replacement.count() == 1


def test_agfs_recovery_removes_new_business_but_preserves_identity_files(tmp_path):
    from service.migration.tree_backup import AGFSTreeBackup
    from tests.unit.service.migration.test_tree_backup import LocalClient

    client = LocalClient(tmp_path / 'files')
    client.path('/local/accounts/acme/users/alice').mkdir(parents=True)
    client.path('/local/accounts/acme/_system').mkdir()
    client.write('/local/accounts/acme/users/alice/memory', b'original')
    client.write('/local/accounts/acme/_system/users.json', b'credentials')
    adapter = AGFSStorage(config(), client)
    source = SimpleNamespace(records=lambda kind: [{'account_id': 'new'}] if kind == 'nodes' else [])
    roots = agfs_business_roots(adapter, source)
    backup = AGFSTreeBackup(client, roots, lambda: {'roots': roots})
    manifest = backup.capture(tmp_path / 'backup')
    client.write('/local/accounts/acme/users/alice/memory', b'changed')
    client.path('/local/accounts/new/users').mkdir(parents=True)
    client.write('/local/accounts/new/users/imported', b'imported')
    backup.restore(tmp_path / 'backup', manifest)
    assert client.read('/local/accounts/acme/users/alice/memory') == b'original'
    assert client.read('/local/accounts/acme/_system/users.json') == b'credentials'
    assert not client.path('/local/accounts/new/users').exists()


def test_native_binding_tracks_mount_config_without_exposing_secrets(tmp_path):
    from providers.unified_config import OgMemConfig
    from service.migration.bundle import configuration_binding

    path = tmp_path / 'native.yaml'
    path.write_text('secret: first')
    cfg = OgMemConfig(agfs_transport='native', agfs_native_config=str(path))
    first = configuration_binding(cfg, None)
    assert 'secret' not in str(first)
    path.write_text('secret: changed')
    assert configuration_binding(cfg, None) != first


def test_sql_backup_scope_includes_space_business_and_runtime_tables():
    from service.migration.resources import SQL_BUSINESS_TABLES

    assert {'spaces', 'space_members', 'space_source_bindings', 'space_rules',
            'space_rule_revisions', 'space_rule_runtime', 'space_items',
            'space_item_sources', 'space_read_acl', 'space_jobs', 'space_requests'} <= SQL_BUSINESS_TABLES
    assert 'cp_users' not in SQL_BUSINESS_TABLES


def test_agfs_recovery_scope_includes_shared_content():
    files = Files()
    files.mkdir('/local/accounts/acme/.space-content')
    source = SimpleNamespace(records=lambda kind: [])
    roots = agfs_business_roots(AGFSStorage(config(), files), source)
    assert '/local/accounts/acme/.space-content' in roots
