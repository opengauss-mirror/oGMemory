"""Regression checks for maintenance at actual storage and HTTP boundaries."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.errors import ConcurrentModificationError
from core.models import ContextNode, RequestContext, Role
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.migration.admission import Admission, install
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


@pytest.mark.parametrize('prior_unknown', [False, True])
@pytest.mark.parametrize('cleanup_ok', [False, True])
def test_agfs_stale_version_releases_only_settled_operation(tmp_path, monkeypatch, prior_unknown, cleanup_ok):
    client = _AtomicMemoryAGFSClient()
    fs = AGFSContextFS(client=client, mount_prefix='/local')
    ctx = RequestContext(account_id='acme', user_id='alice', agent_id='agent', session_id='s', trace_id='t', role=Role.ROOT)
    node = ContextNode(uri='ctx://acme/users/alice/memories/preferences/editor',
                       context_type='MEMORY', category='preferences', owner_space='user:alice',
                       level=2, abstract='original', overview='original', content='original', metadata={'version': 2})
    fs.write_node(node, ctx)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    admission.guard(fs)
    if not cleanup_ok:
        monkeypatch.setattr(fs, '_remove_file_if_present', lambda path: False)
    with admission.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), admission.backend(write=True):
                raise TimeoutError('unknown prior write')
        with pytest.raises(ConcurrentModificationError):
            fs.write_node(replace(node, metadata={'_expected_version': 1}), ctx)
        fs.write_node(replace(node, metadata={'_expected_version': 2}), ctx)
    gate.enter('export', mode='export', resources={})
    if prior_unknown or not cleanup_ok:
        with pytest.raises(TimeoutError):
            gate.wait_drained('export', 0)
    else:
        gate.wait_drained('export', 0)


def test_full_service_control_reads_during_import(tmp_path, monkeypatch):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.migration_fixtures import LocalClient

    (tmp_path / 'files').mkdir()
    monkeypatch.setattr('server.memory_service.AGFSClient', lambda **kw: LocalClient(tmp_path / 'files'))
    cfg = OgMemConfig(storage_backend='agfs', agfs_mount_prefix='/local', provider='mock')
    service = MemoryService(config=cfg)
    admission = Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))
    service.get_key_manager().create_account('acme', 'alice')
    service._control_store = service._key_manager = None
    install(service, admission)
    business = service._get_agfs_client()
    admission.store.enter('import', mode='import', resources={})
    try:
        assert service.get_control_plane_store().list_code_workspace_sync_registrations() == []
        assert any(row['account_id'] == 'acme' for row in service.get_key_manager().get_accounts())
        with pytest.raises(MaintenanceError):
            business.read('/local/_system/accounts.json')
        with pytest.raises(MaintenanceError):
            service.get_control_plane_store()._client.read('/local/accounts/acme/users/alice/memories/x')
        with pytest.raises(MaintenanceError):
            service.get_key_manager().create_account('other', 'bob')
    finally:
        admission.store.reopen('import')
        service.shutdown()


@pytest.mark.parametrize('mode', ['export', 'import'])
def test_write_denial_returns_maintenance_response(tmp_path, monkeypatch, mode):
    import server.app as app_module
    import server.mcp_tools as tools
    from server.maintenance_http import business_wsgi

    admission = Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))
    service = SimpleNamespace(_admission=admission, _cfg=SimpleNamespace(
        http_ip_allowlist=[], mcp_default_toolset='all', mcp_enable_delete_tools=True))
    monkeypatch.setattr(app_module, '_service', service)
    monkeypatch.setattr(app_module, '_build_authenticated_context', lambda p: SimpleNamespace(agent_id='agent'))
    def delete(*a, **kw):
        with admission.backend(write=True):
            pytest.fail('write must not start')
    monkeypatch.setattr(tools, 'delete_memory', delete)
    monkeypatch.setattr(app_module.app, 'wsgi_app', business_wsgi(app_module.app.wsgi_app, lambda: service))
    admission.store.enter('task', mode=mode, resources={})
    result = app_module.app.test_client().post('/api/v1/memories/delete', json={'uri': 'ogmemory://memories/preference/x'})
    assert result.status_code == 503
    assert result.json['error'] == 'migration_maintenance'
    assert result.json['retryable'] is True
    assert result.headers['Retry-After'] == '5'
    admission.store.wait_drained('task', 0)
    with app_module.app.app_context():
        response, status = app_module._lifecycle_error_response(MaintenanceError('paused'), {})
        assert status == 503 and response.headers['Retry-After'] == '5'
        assert response.json['error_detail']['code'] == 'MIGRATION_MAINTENANCE'




def test_memory_write_structured_maintenance_response(tmp_path, monkeypatch):
    import server.app as app_module
    from server.maintenance_http import business_wsgi

    admission = Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))
    def after_turn(params):
        with admission.backend(write=True):
            pytest.fail('write must not start')
    service = SimpleNamespace(_admission=admission, after_turn=after_turn, _cfg=SimpleNamespace(
        http_ip_allowlist=[], mcp_default_toolset='all', mcp_enable_write_tools=True,
        mcp_max_request_bytes=1048576))
    monkeypatch.setattr(app_module, '_service', service)
    monkeypatch.setattr(app_module, '_build_authenticated_context', lambda p: SimpleNamespace(agent_id='agent'))
    monkeypatch.setattr(app_module.app, 'wsgi_app', business_wsgi(app_module.app.wsgi_app, lambda: service))
    admission.store.enter('export', mode='export', resources={})
    result = app_module.app.test_client().post('/api/v1/memories/write', json={
        'text': 'remember this', 'session_id': 'session'})
    assert result.status_code == 503, result.json
    assert result.json['error_detail']['code'] == 'MIGRATION_MAINTENANCE'
    assert result.headers['Retry-After'] == '5'
    admission.store.wait_drained('export', 0)


def test_shared_search_does_not_update_last_used_during_export(tmp_path):
    from unittest.mock import MagicMock

    from core.models import RequestContext, Role
    from server.space_repository import SpaceRepository
    from service.migration.admission import Admission
    from service.shared_memory import SharedMemoryRetrievalService
    repo = object.__new__(SpaceRepository)
    repo.pool = MagicMock()
    gate = MaintenanceStore(tmp_path / 'shared-gate.db', 'review-shared')
    repo._migration_admission = Admission(gate)
    item = dict(item_id='i', space_id='s', uri='ctx://acme/spaces/s/memories/experience/i', kind='experience',
                content='example', abstract='example', rule_id='r', rule_rev=1, version=1, updated_at='2026-11-20')
    repo.visible_items = lambda *a, **kw: [item]
    ctx = RequestContext('acme', 'bob', '', '', 'review', Role.MEMBER)
    gate.enter('snapshot', mode='export', resources={})
    search = SharedMemoryRetrievalService(repo)
    assert search.search(ctx, 'example', ['s'])['total'] == 1
    repo.pool.get_connection.assert_not_called()
    gate.reopen('snapshot')
    assert search.search(ctx, 'example', ['s'])['total'] == 1
    cur = repo.pool.get_connection.return_value.cursor.return_value.__enter__.return_value
    assert any('UPDATE space_items' in str(call) for call in cur.execute.call_args_list)


def test_agfs_shared_hydration_is_read_admission(tmp_path):
    from fs.agfs_adapter.space_bridge import AGFSSpaceBridge

    bridge = object.__new__(AGFSSpaceBridge)
    gate = MaintenanceStore(tmp_path / 'hydrate.db', 'hydrate')
    Admission(gate).guard(bridge)
    gate.enter('export', mode='export', resources={})
    assert bridge.load_content([]) == []
    gate.wait_drained('export', 0)
    gate.reopen('export')
    gate.enter('import', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        bridge.load_content([])



def test_agfs_archive_id_lookup_remains_readable_during_export(tmp_path):
    from session.archive_store import SessionArchiveStore
    from service.migration.snapshot import SourceSnapshot

    gate = MaintenanceStore(tmp_path / 'gate.db', 'archive')
    admission = Admission(gate)
    client = admission.guard(_AtomicMemoryAGFSClient())
    archive = SessionArchiveStore(admission.guard(AGFSContextFS(client)))
    ctx = RequestContext('acme', 'alice', 'agent', 'session', 'trace', Role.ROOT)
    result = archive.write_archive('session', 'summary', 'summary', [], ctx, archive_id='archive')
    assert result.success
    with SourceSnapshot(gate, 'export').scope():
        found = archive.read_archive_by_id('archive', ctx)
        assert found is not None and found.session_id == 'session'
    gate.enter('restore', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        archive.read_archive_by_id('archive', ctx)
