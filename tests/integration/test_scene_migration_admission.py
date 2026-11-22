"""Maintenance boundaries against an isolated SQL scene runtime."""
import os
import uuid
from unittest.mock import Mock

import pytest

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.migration.admission import Admission, business_wsgi
from service.migration.maintenance import MaintenanceError, MaintenanceStore

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not os.environ.get('MIGRATION_TEST_ADMIN_DSN'),
                                 reason='isolated SQL administrator DSN required')]


def test_scene_runtime_configuration_worker_and_resume(tmp_path, monkeypatch):
    import psycopg2
    from psycopg2.extensions import make_dsn
    from flask import Flask, jsonify
    from fs.sql_adapter.pool import SharedConnectionPool
    from server.scene_api import create_scene_blueprint
    from server.migration_lifecycle import ServiceLifecycle

    admin = psycopg2.connect(os.environ['MIGRATION_TEST_ADMIN_DSN'])
    admin.autocommit = True
    database = 'scene_gate_' + uuid.uuid4().hex
    dialect = os.environ.get('MIGRATION_TEST_DIALECT', 'postgresql')
    with admin.cursor() as cursor:
        cursor.execute('CREATE DATABASE ' + database + (" DBCOMPATIBILITY 'PG'" if dialect == 'opengauss' else ''))
    pool = SharedConnectionPool(make_dsn(os.environ['MIGRATION_TEST_ADMIN_DSN'], dbname=database),
                                pool_size=4, dialect=dialect)
    service = None
    try:
        gate = MaintenanceStore(tmp_path / 'gate.db', 'scene')
        cfg = OgMemConfig(storage_backend='sql', provider='mock', skill_sedimentation_enabled=False,
                          scene_enabled=True, scene_generation_enabled=False,
                          sql_connection_string=make_dsn(os.environ['MIGRATION_TEST_ADMIN_DSN'], dbname=database),
                          sql_dialect=dialect)
        service = MemoryService(cfg, admission=Admission(gate), defer_start=True)
        monkeypatch.setattr(service, '_get_shared_sql_pool', lambda: pool)
        monkeypatch.setattr(service, '_get_context_fs', lambda: Mock())
        monkeypatch.setattr(service, '_get_shared_vector_index', lambda: None)
        monkeypatch.setattr(service, '_get_shared_embedder', lambda: Mock())
        monkeypatch.setattr(service, '_get_archive_store', lambda: None)
        monkeypatch.setattr(service, 'get_llm', lambda: Mock())
        monkeypatch.setattr(service, 'start_extraction_recovery', lambda: None)
        monkeypatch.setattr(service, 'get_auth_service', lambda: Mock(role_control_active=lambda: False))
        _check_sql_control_plane(service, gate, monkeypatch)
        ctx = RequestContext('acme', 'alice', 'agent', 's', 't', Role.MEMBER)
        # Import still forbids constructing business runtime.
        gate.enter('cold', mode='import', resources={})
        with pytest.raises(MaintenanceError):
            service.get_scene_service()
        gate.reopen('cold')
        from scene.models import SceneSnapshot
        lexical_factory = Mock()
        monkeypatch.setattr('providers.lexical_index.factory.create_lexical_index', lexical_factory)
        runtime = service.get_scene_service()
        app = Flask(__name__)
        def error(exc):
            return jsonify(error=str(exc)), 503 if isinstance(exc, MaintenanceError) else 400
        app.register_blueprint(create_scene_blueprint(lambda: service, lambda _: ctx, error))
        app.wsgi_app = business_wsgi(app.wsgi_app, lambda: service)
        client = app.test_client()
        gate.enter('snapshot', mode='export', resources={})
        # Canonical content writes are covered separately; runtime projections can continue.
        result = client.put('/api/v1/scenes/configuration', buffered=True, json={'revision': 0, 'values': {'max_pending_nodes': 17}})
        assert result.status_code == 200
        assert runtime.config_provider.get_effective('acme', 'user:alice').revision == 1
        assert client.get('/api/v1/scenes/configuration', buffered=True).status_code == 200
        assert service._scene_manager.process_next('worker') is None
        gate.wait_drained('snapshot', 0)
        gate.reopen('snapshot')
        assert client.put('/api/v1/scenes/configuration', buffered=True, json={'revision': 1, 'values': {}}).status_code == 200
        # A confirmed rollback from an invalid revision must not poison shutdown.
        assert client.put('/api/v1/scenes/configuration', buffered=True, json={'revision': 0, 'values': {}}).status_code == 400
        gate.enter('after-conflict', mode='export', resources={})
        gate.wait_drained('after-conflict', 0)
        gate.reopen('after-conflict')
        manager = service._scene_manager
        manager.start(poll_seconds=.01)
        lifecycle = ServiceLifecycle(service)
        gate.enter('import', mode='import', resources={})
        gate.wait_drained('import', 2)
        old_index = service._scene_source.candidate_index
        assert lifecycle.pause()
        assert old_index._closed
        lifecycle.refresh()
        gate.reopen('import')
        lifecycle.resume()
        assert service._scene_manager is not manager
        assert service._scene_manager.running
    finally:
        if service is not None:
            service.shutdown()
        pool.close_all()
        with admin.cursor() as cursor:
            cursor.execute('DROP DATABASE ' + database)
        admin.close()


def _check_sql_control_plane(service, gate, monkeypatch):
    import importlib
    module = importlib.import_module('server.app')
    root = RequestContext('acme', 'root', 'agent', 's', 't', Role.ROOT)
    service.get_key_manager().create_account('acme', 'admin')
    with monkeypatch.context() as patch:
        patch.setattr(module, '_service', service)
        patch.setattr(module, '_build_authenticated_context', lambda _: root)
        patch.setattr(service, 'get_auth_service', lambda: Mock(role_control_active=lambda: True))
        client = module.app.test_client()
        gate.enter('control-export', mode='export', resources={})
        response = client.post('/api/v1/admin/accounts/acme/users',
                               json={'user_id': 'new-user'}, buffered=True)
        assert response.status_code == 200
        assert {u['user_id'] for u in service.get_key_manager().get_users('acme')} == {'admin', 'new-user'}
        assert client.get('/api/v1/admin/accounts', buffered=True).status_code == 200
        gate.wait_drained('control-export', 0)
        gate.reopen('control-export')
        assert client.post('/api/v1/admin/accounts/acme/users',
                           json={'user_id': 'after-export'}, buffered=True).status_code == 200
        # A competing request between read and write causes a confirmed rollback.
        from server.sql_control_plane_store import SQLControlPlaneStore
        from server.api_keys import APIKeyManager
        store = service.get_control_plane_store()
        path = store.users_path('acme')
        payload = store.read_json(path, {'users': {}})
        other = SQLControlPlaneStore(pool=service._get_shared_sql_pool())
        APIKeyManager(other).register_user('acme', 'racer')
        payload['users']['racer'] = {'role': 'user', 'key': 'unused', 'status': 'active'}
        with pytest.raises(ValueError, match='already exists'):
            store.write_json(path, payload)
        def reject(_):
            raise ValueError('workspace conflict')
        with pytest.raises(ValueError, match='workspace conflict'):
            store.mutate_code_workspace_sync_state('acme', 'root', reject)
        gate.enter('control-import', mode='import', resources={})
        service._control_store = service._key_manager = None
        patch.setattr(SQLControlPlaneStore, '_ensure_table',
                      lambda self: pytest.fail('maintenance must not initialize SQL schema'))
        assert client.get('/api/v1/admin/accounts', buffered=True).status_code == 200
        assert client.post('/api/v1/admin/accounts/acme/users',
                           json={'user_id': 'another'}, buffered=True).status_code == 503
        gate.wait_drained('control-import', 0)
        gate.reopen('control-import')
