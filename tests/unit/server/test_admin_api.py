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


def test_tenant_bootstrap_role_assignment_and_key_rotation(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    base = "/api/v1/admin/accounts"

    created = client.post(
        base, headers={"X-API-Key": "root-key"},
        json={"account_id": "ops", "admin_user_id": "alice"},
    )
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "no-store"
    alice = {"X-API-Key": created.get_json()["user_key"]}
    users = f"{base}/ops/users"
    bob = client.post(users, headers=alice, json={"user_id": "bob"}).get_json()
    assert bob["role"] == "MEMBER"
    read_key = client.post(f"{users}/bob/key/reveal", headers=alice)
    assert read_key.status_code == 200
    assert read_key.headers["Cache-Control"] == "no-store"
    assert read_key.get_json()["user_key"] == bob["user_key"]
    assert any(event["action"] == "user_key_revealed" for event in service.get_audit_service().list_logs("ops"))
    agents = f"{base}/ops/agents"
    client.post(agents, headers=alice, json={"agent_id": "Ops-Bot", "owner_user_id": "bob"})
    client.post(agents, headers=alice, json={"agent_id": "Other-Bot", "owner_user_id": "alice"})
    page = client.get(f"{agents}?limit=1&q=bot&owner_user_id=bob", headers=alice).get_json()
    assert [agent["agent_id"] for agent in page["agents"]] == ["Ops-Bot"]
    assert page["next_after"] is None
    unassigned = client.post(agents, headers=alice, json={"agent_id": "unassigned"}).get_json()
    assert unassigned["owner_user_id"] == ""
    bound = client.patch(f"{agents}/unassigned", headers=alice, json={"owner_user_id": "bob"}).get_json()
    assert bound["owner_user_id"] == "bob"
    unbound = client.patch(f"{agents}/unassigned", headers=alice, json={"owner_user_id": None}).get_json()
    assert unbound["owner_user_id"] == ""
    assert client.patch(f"{users}/bob/role", headers=alice, json={"role": "ADMIN"}).get_json() == {
        "updated": True, "role": "ADMIN",
    }

    first_page = client.get(f"{users}?limit=1", headers=alice).get_json()
    assert first_page["next_after"] == "alice"
    assert set(first_page["users"][0]) == {"user_id", "role", "status", "created_at"}
    second_page = client.get(f"{users}?limit=1&after=alice&q=bob", headers=alice).get_json()
    assert second_page["users"][0]["user_id"] == "bob"
    assert second_page["next_after"] is None

    rotated = client.post(f"{users}/bob/key", headers=alice)
    assert rotated.status_code == 200
    assert rotated.headers["Cache-Control"] == "no-store"
    assert rotated.get_json()["user_key"] != bob["user_key"]
    roles = client.get(f"{base}/ops/roles", headers={"X-API-Key": rotated.get_json()["user_key"]})
    assert roles.status_code == 200
    assert {"user_id": "bob", "role": "ADMIN"} in roles.get_json()["roles"]


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
