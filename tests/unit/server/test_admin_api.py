"""Tests for V1 admin HTTP routes."""

from __future__ import annotations

import server.app as app_module
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService


def _make_service(tmp_path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        admin_api_keys=[],
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service._auth = service.get_auth_service().__class__(cfg, service._key_manager)
    service.get_key_manager().create_account("acct-1", "alice")
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    return service, member_key


def test_admin_routes_enforce_member_forbidden(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.patch(
        "/api/v1/admin/accounts/acct-1/users/alice/role",
        json={"role": "admin"},
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_admin_roles_route_for_root(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get(
        "/api/v1/admin/accounts/acct-1/roles",
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"},
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert any(row["user_id"] == "alice" for row in payload["roles"])


def test_admin_create_duplicate_user_returns_400(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    client.post(
        "/api/v1/admin/accounts/acct-1/users",
        json={"user_id": "bob", "role": "user"},
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"},
    )
    resp = client.post(
        "/api/v1/admin/accounts/acct-1/users",
        json={"user_id": "bob", "role": "user"},
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"},
    )

    assert resp.status_code == 400


def test_admin_update_missing_agent_returns_404(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.patch(
        "/api/v1/admin/accounts/acct-1/agents/missing-agent",
        json={"owner_user_id": "alice"},
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"},
    )

    assert resp.status_code == 404


def test_admin_routes_unavailable_when_role_control_disabled(monkeypatch, tmp_path):
    cfg = OgMemConfig(
        role_control_enabled=False,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get("/api/v1/admin/accounts")

    assert resp.status_code == 503
    assert "Admin API is unavailable" in resp.get_json()["error"]


def test_readonly_admin_keeps_authentication_during_migration(monkeypatch, tmp_path):
    from service.migration.admission import Admission
    from service.migration.maintenance import MaintenanceStore
    service, member_key = _make_service(tmp_path)
    service._admission = Admission(MaintenanceStore(tmp_path / 'gate.db', 'admin-test'))
    monkeypatch.setattr(app_module, '_service', service)
    service._admission.store.enter('import', mode='import', resources={})
    client = app_module.app.test_client()
    headers = {'X-API-Key': 'root-key', 'X-Account-ID': 'acct-1', 'X-User-ID': 'root'}
    assert client.get('/api/v1/admin/accounts', headers=headers).status_code == 200
    assert client.get('/api/v1/admin/accounts').status_code == 401
    assert client.get('/api/v1/admin/accounts', headers={'X-API-Key': member_key}).status_code == 403
    assert client.post('/api/v1/admin/accounts/acct-1/users', headers=headers, json={}).status_code == 503
    assert client.head('/api/v1/admin/accounts/acct-1/users', headers=headers, json={'user_id': 'head-created'}).status_code == 503
    service._admission.store.reopen('import')
    service.shutdown()
