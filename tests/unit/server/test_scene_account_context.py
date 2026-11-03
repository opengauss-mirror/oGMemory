"""Scene account selection through real HTTP authentication and tenant registries."""

from dataclasses import replace

import pytest

import server.app as app_module
from core.models import Role
from tests.unit.server.test_http_auth_v1 import _make_service
from tests.unit.server.test_scene_management import management as management


@pytest.fixture
def authenticated_scenes(tmp_path, monkeypatch, management):
    facade, member_key = _make_service(tmp_path)
    keys = facade.get_key_manager()
    keys.create_account("acct-default", "root")
    admin_key = keys.create_account("acct-2", "alice")
    facade._scene_service = management.service
    root = replace(management.ctx, account_id="acct-default", user_id="root", role=Role.ROOT)
    for account in ("acct-default", "acct-1", "acct-2"):
        for owner in ("user:alice", "agent:helper"):
            management.service.save(replace(root, account_id=account), {
                "revision": 0, "owner_space": owner, "title": account,
                "description": "Account-specific release evidence",
            })
    facade.get_tenant_admin_service().create_agent(root, "acct-2", "helper", "alice")
    keys.register_user("acct-2", "carol")
    for account, user, agent in (("acct-2", "carol", "delegate"), ("acct-1", "bob", "personal")):
        facade.get_tenant_admin_service().create_agent(root, account, agent, user)
        for owner in (f"user:{user}", f"agent:{agent}"):
            management.service.save(replace(root, account_id=account), {
                "revision": 0, "owner_space": owner, "title": f"{account}/{owner}",
                "description": "Registered owner evidence",
            })
    monkeypatch.setattr(app_module, "_service", facade)
    yield app_module.app.test_client(), management.service, root, admin_key, member_key
    facade.shutdown()


@pytest.mark.parametrize("owner", ["user:alice", "agent:helper"])
def test_root_switches_scene_account_using_authenticated_request(authenticated_scenes, owner):
    client, _, _, _, _ = authenticated_scenes
    params = {"accountId": "acct-2", "userId": "alice", "owner_space": owner,
              "agentId": "helper" if owner.startswith("agent:") else ""}
    response = client.get("/api/v1/scenes", query_string=params, headers={"X-API-Key": "root-key"})
    assert response.status_code == 200
    assert response.json["items"][0]["cluster"]["label"] == "acct-2"


def test_root_mutates_selected_account_then_reads_same_scene(authenticated_scenes):
    client, scenes, root, _, _ = authenticated_scenes
    payload = {"accountId": "acct-2", "userId": "alice", "owner_space": "user:alice",
               "revision": 1, "title": "Selected account", "description": "Verified release"}
    response = client.post("/api/v1/scenes", json=payload, headers={"X-API-Key": "root-key"})
    assert response.status_code == 200
    selected = scenes.list(replace(root, account_id="acct-2"), owner_space="user:alice")
    assert selected["total"] == 2
    assert selected["items"][-1]["cluster"]["label"] == "Selected account"
    assert scenes.list(root, owner_space="user:alice")["total"] == 1


@pytest.mark.parametrize("role", ["root", "admin"])
@pytest.mark.parametrize("owner", ["user:carol", "agent:delegate"])
@pytest.mark.parametrize("explicit_user", [True, False])
def test_administrator_selects_registered_owner(authenticated_scenes, role, owner, explicit_user):
    client, _, _, admin_key, _ = authenticated_scenes
    params = {"accountId": "acct-2", "owner_space": owner}
    if explicit_user:
        params["userId"] = "carol"
        params["agentId"] = "delegate" if owner.startswith("agent:") else ""
    key = "root-key" if role == "root" else admin_key
    response = client.get("/api/v1/scenes", query_string=params, headers={"X-API-Key": key})
    assert response.status_code == 200
    assert response.json["items"][0]["cluster"]["label"] == f"acct-2/{owner}"


@pytest.mark.parametrize("owner", ["user:bob", "agent:personal"])
def test_member_reads_registered_personal_scope(authenticated_scenes, owner):
    client, _, _, _, member_key = authenticated_scenes
    response = client.get("/api/v1/scenes", query_string={
        "userId": "bob", "owner_space": owner,
        "agentId": "personal" if owner.startswith("agent:") else "",
    }, headers={"X-API-Key": member_key})
    assert response.status_code == 200
    assert response.json["items"][0]["cluster"]["label"] == f"acct-1/{owner}"


def test_selected_account_management_lifecycle(authenticated_scenes):
    client, scenes, root, _, _ = authenticated_scenes
    target = {"accountId": "acct-2", "userId": "alice", "owner_space": "user:alice"}
    headers = {"X-API-Key": "root-key"}
    created = client.post("/api/v1/scenes", json={**target, "revision": 1,
                          "title": "Release", "description": "Verified release"}, headers=headers)
    assert created.status_code == 200
    scene_id = created.json["scene"]["scene_id"]
    updated = client.put(f"/api/v1/scenes/{scene_id}", json={**target, "revision": 2,
                         "description": "Friday release"}, headers=headers)
    assert updated.status_code == 200
    assert updated.json["scene"]["cluster"]["summary"] == "Friday release"
    configured = client.put("/api/v1/scenes/configuration", json={**target, "revision": 0,
                            "values": {"neighbor_k": 12}}, headers=headers)
    assert configured.status_code == 200
    current = client.get("/api/v1/scenes/configuration", query_string=target, headers=headers)
    assert current.json["values"]["neighbor_k"] == 12
    assert scenes.configuration(root, "user:alice")["revision"] == 0
    scenes.manager.jobs.enqueue("acct-2", "user:alice", ["release-evidence"], token_count=1, trigger_tokens=10)
    status = client.get("/api/v1/scenes/status", query_string=target, headers=headers)
    assert status.json["pending_nodes"] == 1
    flushed = client.post("/api/v1/scenes/flush", json=target, headers=headers)
    assert flushed.json["active_job"]["node_uris"] == ["release-evidence"]
    archived = client.delete(f"/api/v1/scenes/{scene_id}", json={**target, "revision": 3}, headers=headers)
    assert archived.status_code == 200
    assert archived.json["status"] == "ARCHIVED"
    assert client.get("/api/v1/scenes", query_string=target, headers=headers).json["total"] == 1


def test_root_account_headers_and_local_mode_select_same_scene(authenticated_scenes):
    client, _, _, _, _ = authenticated_scenes
    params = {"userId": "alice", "owner_space": "user:alice"}
    authenticated = client.get("/api/v1/scenes", query_string=params, headers={
        "X-API-Key": "root-key", "X-Account-Id": "acct-2", "X-User-Id": "alice",
    })
    assert authenticated.status_code == 200
    app_module._service._cfg.role_control_enabled = False
    local = client.get("/api/v1/scenes", query_string={**params, "accountId": "acct-2"})
    assert local.status_code == 200
    assert local.json == authenticated.json
