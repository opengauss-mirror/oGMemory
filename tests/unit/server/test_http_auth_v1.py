"""Tests for V1 auth wiring on HTTP entrypoints."""

from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest

from core.errors import NodeNotFoundError
from core.models import ContextNode, RetrievedBlock, SearchMemoryResult
from fs.sql_adapter.sql_context_fs import SQLContextFS
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService
from server.control_plane_store import ControlPlaneStore
from server.mcp_resources import render_collection_resource, render_item_resource
from server.mcp_tools import list_memories
from server.mcp_security import UNTRUSTED_DATA_NOTICE  # noqa: E402
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService

_HAS_FLASK = importlib.util.find_spec("flask") is not None
pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")

if _HAS_FLASK:
    import server.app as app_module
else:
    app_module = None


def _make_service(tmp_path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service._auth = AuthService(cfg, service._key_manager)
    service.get_key_manager().create_account("acct-1", "alice")
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    service.compose = lambda params: {
        "ok": True,
        "account": params["_ctx"].account_id,
        "user": params["_ctx"].user_id,
        "agent": params["_ctx"].agent_id,
    }
    return service, member_key


def _make_service_with_allowlist(tmp_path, allowlist, *, trust_proxy=False, trusted_proxies=None):
    cfg = OgMemConfig(
        role_control_enabled=False,
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
        http_ip_allowlist=list(allowlist),
        http_ip_allowlist_trust_proxy=trust_proxy,
        http_trusted_proxies=list(trusted_proxies or []),
    )
    service = MemoryService(config=cfg)
    service.compose = lambda params: {"ok": True}
    return service


def test_http_reads_shared_agent_case_returned_by_query(monkeypatch, tmp_path):
    service, bob_key = _make_service(tmp_path)
    service._cfg.agent_shared_mode = "user"
    service._cfg.agent_shared_list = ["shared-bot"]
    service.get_key_manager().create_account("acct-other", "other-admin")
    carol_key = service.get_key_manager().register_user("acct-other", "carol", "user")
    admin = service.get_tenant_admin_service()
    root = service.get_auth_service().resolve_identity({"X-API-Key": "root-key", "X-Account-ID": "acct-1"})
    root_ctx = service.build_context({}, identity=root)
    for agent, owner in (("bob-bot", "bob"), ("shared-bot", "alice"), ("private-bot", "alice")):
        admin.create_agent(root_ctx, "acct-1", agent, owner_user_id=owner)

    shared = "ctx://acct-1/agents/shared-bot/memories/cases/shared-case"
    private = "ctx://acct-1/agents/private-bot/memories/cases/private-case"
    nodes = {uri: ContextNode(uri=uri, context_type="MEMORY", category="case", level=0,
                              owner_space=f"agent:{uri.split('/')[4]}", abstract="", overview="",
                              content=f"memory at {uri.rsplit('/', 1)[-1]}") for uri in (shared, private)}

    listed_dirs = []

    class CaseFS:
        def exists(self, uri, ctx):
            return uri in nodes

        def read_node(self, uri, ctx):
            SQLContextFS._ensure_accessible(uri, ctx)
            if uri not in nodes:
                raise NodeNotFoundError(uri)
            return nodes[uri]

        def read_node_fields(self, uri, ctx, level):
            return self.read_node(uri, ctx)

        def list_children(self, uri, ctx):
            SQLContextFS._ensure_accessible(uri, ctx)
            listed_dirs.append(uri)
            return [node_uri for node_uri in nodes if node_uri.rsplit("/", 1)[0] == uri]

    query_hit = shared

    def search_memory(query, ctx, **kwargs):
        assert SQLContextFS._is_accessible(shared, ctx)
        assert not SQLContextFS._is_accessible(private, ctx)
        return SearchMemoryResult(request_id="query-1", query=query, hits=[
            RetrievedBlock(uri=query_hit, score=0.9, category="case", has_content=True),
        ])

    monkeypatch.setattr(service, "_get_context_fs", lambda: CaseFS())
    monkeypatch.setattr(service, "get_read_api", lambda: Mock(search_memory=search_memory))
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    headers = {"X-API-Key": bob_key, "X-Account-ID": "acct-1"}
    result = client.post("/api/v1/memories/query", headers=headers, json={"agentId": "bob-bot", "query": "memory"})
    assert result.status_code == 200
    uri = result.json["hits"][0]["uri"]
    assert uri == "ogmemory://memories/case/shared-case"
    assert result.json["hits"][0]["owner_agent_id"] == "shared-bot"

    def read(agent, uri, headers, **fields):
        return client.post("/api/v1/memories/read", headers=headers,
                           json={"agentId": agent, "uri": uri, "level": "content", **fields})

    for agent in ("shared-bot", "bob-bot"):
        result = read(agent, uri, headers)
        assert result.status_code == 200
        assert result.json["content"] == "memory at shared-case"
    assert shared.rsplit("/", 1)[0] not in listed_dirs
    assert read("bob-bot", "ogmemory://memories/case/private-case", headers).status_code == 403
    assert read("", uri, {"X-API-Key": carol_key, "X-Account-ID": "acct-other"}).status_code == 403
    assert private.rsplit("/", 1)[0] not in listed_dirs

    local = shared.replace("shared-bot", "bob-bot")
    nodes[local] = ContextNode(uri=local, context_type="MEMORY", category="case", level=0,
                               owner_space="agent:bob-bot", abstract="", overview="", content="local body")
    listed_dirs.clear()
    for owner, body in (("shared-bot", "memory at shared-case"), ("bob-bot", "local body")):
        result = read("bob-bot", uri, headers, owner_agent_id=owner)
        assert result.status_code == 200
        assert result.json["content"] == body
        assert result.json["owner_agent_id"] == owner
    assert listed_dirs == []

    result = read("bob-bot", uri, headers)
    assert result.status_code == 400
    assert "Ambiguous case URI" in result.json["error"]
    ctx = service.build_context({"agentId": "bob-bot"}, service.get_auth_service().resolve_identity(headers))
    with pytest.raises(ValueError, match="Ambiguous case URI"):
        render_item_resource(service, uri, ctx)
    assert read("bob-bot", uri, headers, owner_agent_id="private-bot").status_code == 403
    assert read("bob-bot", "ogmemory://memories/case/private-case", headers,
                owner_agent_id="shared-bot").status_code == 403
    assert read("", uri, {"X-API-Key": carol_key, "X-Account-ID": "acct-other"},
                owner_agent_id="shared-bot").status_code == 403

    shared_dir = shared.rsplit("/", 1)[0]
    historical = f"{shared_dir}/old%2Dcase"
    nodes[historical] = ContextNode(uri=historical, context_type="MEMORY", category="case", level=0,
                                    owner_space="agent:shared-bot", abstract="", overview="", content="old body")
    query_hit = historical
    hit = client.post("/api/v1/memories/query", headers=headers,
                      json={"agentId": "bob-bot", "query": "old"}).json["hits"][0]
    listed_dirs.clear()
    assert hit["uri"] == "ogmemory://memories/case/old-case"
    assert read("bob-bot", hit["uri"], headers, owner_agent_id=hit["owner_agent_id"]).json["content"] == "old body"
    assert listed_dirs == [shared_dir]

    user_dir = "ctx://acct-1/users/bob/memories/cases"
    encoded = f"{user_dir}/%E4%B8%AD"
    nodes[encoded] = ContextNode(uri=encoded, context_type="MEMORY", category="case", level=0,
                                 owner_space="user:bob", abstract="", overview="", content="encoded")
    listed_dirs.clear()
    assert read("bob-bot", "ogmemory://memories/case/%e4%b8%ad", headers, owner_scope="user").json["content"] == "encoded"
    assert listed_dirs == []

    # Legacy user-owned cases can also share a name with an Agent case.
    user_case = f"{user_dir}/old-case"
    nodes[user_case] = ContextNode(uri=user_case, context_type="MEMORY", category="case", level=0,
                                  owner_space="user:bob", abstract="", overview="", content="user body")
    query_hit = user_case
    user_hit = client.post("/api/v1/memories/query", headers=headers,
                           json={"agentId": "bob-bot", "query": "user body"}).json["hits"][0]
    assert "owner_agent_id" not in user_hit
    assert user_hit["owner_scope"] == "user"
    listed_dirs.clear()
    selected = read("bob-bot", user_hit["uri"], headers, owner_scope=user_hit["owner_scope"])
    assert selected.status_code == 200
    assert selected.json["content"] == "user body"
    assert selected.json["owner_scope"] == "user"
    query_hit = historical
    agent_hit = client.post("/api/v1/memories/query", headers=headers,
                            json={"agentId": "bob-bot", "query": "old body"}).json["hits"][0]
    assert agent_hit["uri"] == user_hit["uri"]
    assert read("bob-bot", agent_hit["uri"], headers, owner_agent_id=agent_hit["owner_agent_id"]).json["content"] == "old body"
    assert read("bob-bot", user_hit["uri"], headers).status_code == 400
    tool_list = list_memories(service, agent_id="bob-bot", resolved_identity=service.get_auth_service().resolve_identity(headers))
    resource_list = render_collection_resource(service, "case", ctx)
    for listing in (tool_list, resource_list):
        item = next(item for item in listing["memories"] if item["uri"] == user_hit["uri"])
        assert item["owner_scope"] == "user"
        assert read("bob-bot", item["uri"], headers, owner_scope=item["owner_scope"]).json["content"] == "user body"
    assert read("bob-bot", uri, headers, owner_scope="user").status_code == 403
    assert read("bob-bot", uri, headers, owner_scope="user", owner_agent_id="shared-bot").status_code == 400
    assert read("bob-bot", uri, headers, owner_scope="alice").status_code == 400
    assert read("", user_hit["uri"], {"X-API-Key": carol_key, "X-Account-ID": "acct-other"},
                owner_scope="user").status_code == 403


def test_http_reads_legacy_agent_case_with_owner_from_query(monkeypatch, tmp_path):
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient

    service, bob_key = _make_service(tmp_path)
    # An existing registry can contain IDs that the current create-agent API disallows.
    store = service.get_control_plane_store()
    store.write_json(store.agents_path("acct-1"), {"agents": {
        "_legacy": {"owner_user_id": "bob"}, "_private": {"owner_user_id": "alice"},
    }})
    headers = {"X-API-Key": bob_key, "X-Account-ID": "acct-1"}
    ctx = service.build_context({"agentId": "_legacy"}, service.get_auth_service().resolve_identity(headers))
    uri = "ctx://acct-1/agents/_legacy/memories/cases/example"
    fs = AGFSContextFS(_AtomicMemoryAGFSClient())
    fs.write_node(ContextNode(uri=uri, context_type="MEMORY", category="case", level=0,
                              owner_space="agent:_legacy", abstract="", overview="", content="legacy body"), ctx)
    hit = RetrievedBlock(uri=uri, score=1.0, category="case", has_content=True)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "get_read_api", lambda: Mock(search_memory=Mock(
        return_value=SearchMemoryResult(request_id="q", query="example", hits=[hit]))))
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    def post(operation, **params):
        return client.post(f"/api/v1/memories/{operation}", headers=headers, json={"agentId": "_legacy", **params})

    result = post("query", query="example")
    assert result.status_code == 200
    item = result.json["hits"][0]
    assert item["owner_agent_id"] == "_legacy"
    result = post("read", uri=item["uri"], owner_agent_id=item["owner_agent_id"], level="content")
    assert result.status_code == 200
    assert result.json["content"] == "legacy body"
    assert post("read", uri=item["uri"], owner_agent_id="_private").status_code == 403
    for invalid in ("", "../bot", "a/b", "a\\b", "%2F", "a\nb", 42):
        assert post("read", uri=item["uri"], owner_agent_id=invalid).status_code == 400


def test_data_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/compose", json={"messages": []})

    assert resp.status_code == 401


@pytest.mark.parametrize("status", ["processing", "completed", "failed"])
def test_operator_task_poll_auth_and_account_boundary(monkeypatch, tmp_path, status):
    from extraction.operators.task_store import InMemoryTaskStore
    from service.operator_service import OperatorService

    service, owner_key = _make_service(tmp_path)
    service.get_key_manager().create_account("acct-2", "eve")
    other_key = service.get_key_manager().register_user("acct-2", "mallory", "user")
    tasks = InMemoryTaskStore()
    rec = tasks.submit(
        raw_node_id="private-node", operator_service="test", account_id="acct-1",
    )
    rec.status = status
    rec.result = {"secret": "private result"}
    rec.error = "private error"
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(
        app_module, "_operator_service", OperatorService(Mock(), task_store=tasks),
    )
    client = app_module.app.test_client()
    url = f"/api/v1/operators/tasks/{rec.task_id}"

    for headers in ({}, {"X-API-Key": "invalid-key"}):
        assert client.get(url, headers=headers).status_code == 401
        assert client.get("/api/v1/operators/tasks/missing", headers=headers).status_code == 401

    # Neither knowing the task ID nor supplying the owner's account parameter
    # may override the account bound to the authenticated key.
    for suffix in ("", "?account_id=acct-1"):
        denied = client.get(url + suffix, headers={"X-API-Key": other_key})
        assert denied.status_code == 404
        assert denied.get_json() == {"error": "task not found"}

    allowed = client.get(url, headers={"X-API-Key": owner_key})
    assert allowed.status_code == 200
    assert allowed.get_json()["result"] == rec.result
    assert allowed.get_json()["account_id"] == "acct-1"
    assert client.get(
        "/api/v1/operators/tasks/missing", headers={"X-API-Key": owner_key},
    ).status_code == 404


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "invalid-key"}])
def test_code_memory_only_compose_requires_valid_api_key(monkeypatch, tmp_path, headers):
    service, _ = _make_service(tmp_path)
    service.compose = Mock(side_effect=AssertionError("unauthenticated request delegated"))
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/compose",
        json={"messages": [], "code_memory_only": True},
        headers=headers,
    )

    assert response.status_code == 401
    service.compose.assert_not_called()


@pytest.mark.parametrize(
    ("route", "method_name", "payload"),
    [
        (
            "/api/v1/resources/code/sync/registrations",
            "register_code_workspace",
            {"path": "/private/repository", "name": "repo"},
        ),
        (
            "/api/v1/resources/code/sync",
            "sync_code_workspace",
            {"root_id": "root-1"},
        ),
        (
            "/api/v1/resources/code/upsert",
            "upsert_code_resource",
            {"path": "src/service.py", "root_slug": "repo"},
        ),
        (
            "/api/v1/resources/code/search",
            "search_code_resources",
            {"query": "service", "scope": {"root_slugs": ["repo"]}},
        ),
        (
            "/api/v1/resources/code/snippet",
            "read_focused_code_snippet",
            {"uri": "ctx://acct-1/resources/code-node", "max_chars": 32},
        ),
    ],
)
def test_code_memory_routes_delegate_with_authenticated_context(
    monkeypatch,
    tmp_path,
    route,
    method_name,
    payload,
):
    """Unit-only Flask adapter coverage; canonical services are recording doubles."""

    service, member_key = _make_service(tmp_path)
    captured = []

    def delegate(params):
        captured.append(dict(params))
        return {"ok": True, "root_id": "root-1", "locator": "repo/src/service.py:1-2"}

    monkeypatch.setattr(service, method_name, delegate)
    monkeypatch.setattr(app_module, "_service", service)
    body = {**payload, "accountId": "foreign-request-value", "userId": "mallory"}

    response = app_module.app.test_client().post(
        route,
        json=body,
        headers={"X-API-Key": member_key},
    )

    assert response.status_code == 200
    assert captured[0]["_ctx"].account_id == "acct-1"
    assert captured[0]["_ctx"].user_id == "bob"
    assert "/private/repository" not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    ("route", "method_name"),
    [
        ("/api/v1/resources/code/sync/registrations", "register_code_workspace"),
        ("/api/v1/resources/code/sync", "sync_code_workspace"),
        ("/api/v1/resources/code/upsert", "upsert_code_resource"),
        ("/api/v1/resources/code/search", "search_code_resources"),
        ("/api/v1/resources/code/snippet", "read_focused_code_snippet"),
    ],
)
@pytest.mark.parametrize("body", [b"[]", b"null", b"{", b""])
def test_code_memory_routes_reject_missing_invalid_or_non_object_json_before_delegation(
    monkeypatch,
    tmp_path,
    route,
    method_name,
    body,
):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(
        service,
        method_name,
        lambda params: (_ for _ in ()).throw(AssertionError(f"unexpected delegation: {params}")),
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        route,
        data=body,
        content_type="application/json",
        headers={"X-API-Key": member_key},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "request body must be a JSON object"


@pytest.mark.parametrize(
    "method",
    [
        "register_code_workspace",
        "sync_code_workspace",
        "refresh_code_resource",
        "upsert_code_resource",
        "search_code_resources",
        "read_focused_code_snippet",
        "edit_code_file",
    ],
)
def test_generic_call_does_not_expose_code_memory_canonical_facades(monkeypatch, method):
    monkeypatch.setattr(
        app_module,
        "_get_service",
        lambda: (_ for _ in ()).throw(AssertionError("generic call initialized the service")),
    )

    response = app_module.app.test_client().post(f"/api/v1/call/{method}", json={})

    assert response.status_code == 404
    assert response.get_json() == {"error": f"unknown method: {method}"}


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("/api/v1/resources/code/sync/registrations", {"path": "/private/repository"}),
        ("/api/v1/resources/code/sync", {"root_id": "root-1"}),
        ("/api/v1/resources/code/refresh", {"root_slug": "repo", "path": "src/service.py"}),
        ("/api/v1/resources/code/upsert", {"root_slug": "repo", "path": "src/service.py"}),
    ],
)
def test_workspace_mutation_routes_reject_member_before_delegation(
    monkeypatch,
    tmp_path,
    route,
    payload,
):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        route,
        json=payload,
        headers={"X-API-Key": member_key},
    )

    assert response.status_code == 403


def test_workspace_refresh_fails_closed_when_role_control_is_disabled(monkeypatch):
    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            code_resources_enabled=True,
        )
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/resources/code/refresh",
        json={"root_slug": "repo", "path": "src/service.py"},
    )

    assert response.status_code == 503


def test_code_memory_search_route_requires_authentication(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(
        service,
        "search_code_resources",
        lambda params: (_ for _ in ()).throw(AssertionError("unauthenticated request delegated")),
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/resources/code/search",
        json={"query": "service"},
    )

    assert response.status_code == 401


def test_public_code_resource_delete_route_is_not_registered(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().delete(
        "/api/v1/resources/code-node",
        json={},
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("route", "method_name", "error", "status"),
    [
        (
            "/api/v1/resources/code/search",
            "search_code_resources",
            ValueError("invalid code search"),
            400,
        ),
        (
            "/api/v1/resources/code/snippet",
            "read_focused_code_snippet",
            FileNotFoundError("missing code snippet"),
            404,
        ),
    ],
)
def test_code_memory_routes_use_existing_error_mapping(
    monkeypatch,
    tmp_path,
    route,
    method_name,
    error,
    status,
):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(
        service,
        method_name,
        lambda params: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        route,
        json={"query": "service", "uri": "ctx://acct-1/resources/code-node"},
        headers={"X-API-Key": member_key},
    )

    assert response.status_code == status


def test_session_working_set_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get("/api/v1/session_working_set")

    assert resp.status_code == 401


def test_evict_idle_sessions_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/evict_idle_sessions", json={"maxIdleSeconds": 1})

    assert resp.status_code == 401


def test_data_route_uses_trusted_identity_over_request_fields(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={
            "messages": [],
            "accountId": "evil",
            "userId": "mallory",
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["account"] == "acct-1"
    assert payload["user"] == "bob"


@pytest.mark.parametrize("code_memory_only", [None, "true", 1, {}])
def test_compose_route_rejects_non_boolean_code_memory_only(
    monkeypatch,
    tmp_path,
    code_memory_only,
):
    service, member_key = _make_service(tmp_path)
    # _make_service uses a recording Compose stub for auth tests; restore the
    # actual boundary implementation to verify Flask maps its validation error.
    service.compose = MemoryService.compose.__get__(service, MemoryService)
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/compose",
        json={"messages": [], "code_memory_only": code_memory_only},
        headers={"X-API-Key": member_key},
    )

    assert response.status_code == 400
    assert "code_memory_only" in response.get_json()["error"]


def test_data_route_rejects_explicit_unowned_agent(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={
            "messages": [],
            "agentId": "agent-x",
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_tool_output_filter_route_reaches_memory_service_with_authenticated_context(
    monkeypatch, tmp_path
):
    service, _ = _make_service(tmp_path)
    captured = {}

    def filter_bash(params):
        captured.update(params)
        return {"ok": True, "skipped": True, "reason": "below_min_chars"}

    service.filter_bash = filter_bash
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    response = client.post(
        "/api/v1/filter_bash",
        json={"agentId": "agent-1", "command": "pytest -q", "output": "short"},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["reason"] == "below_min_chars"
    assert captured["_ctx"].account_id == "acct-default"
    assert captured["_ctx"].agent_id == "agent-1"


def test_tool_output_recovery_marks_original_as_untrusted(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    captured = {}

    def recover_original(params):
        captured.update(params)
        return {
            "ok": True,
            "memory_uri": params["memory_uri"],
            "original": "external command output",
        }

    service.read_tool_output_original = recover_original
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    uri = "ctx://acct-default/agents/agent-1/memories/tool_outputs/item"

    response = client.post(
        "/api/v1/tool_output_original",
        json={"agentId": "agent-1", "memory_uri": uri},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["untrusted_data_notice"] == UNTRUSTED_DATA_NOTICE
    assert captured["_ctx"].account_id == "acct-default"
    assert captured["_ctx"].agent_id == "agent-1"


def test_trace_route_persists_canonical_non_boundary_event_without_finalizing(
    monkeypatch, tmp_path
):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-1",
        "event_index": 4,
    }
    service.finalize_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("non-boundary events must not finalize a turn")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "timestamp": "2026-07-15T08:00:00Z",
            "event": {
                "event_type": "tool_start",
                "hook_type": "PreToolUse",
                "hook": "PreToolUse",
                "hookName": "PreToolUse",
                "source": "claude-code",
                "producer": "claude-code",
                "producer_id": "claude-code-producer",
                "producerId": "claude-code-producer-camel",
                "agent_id": "client-agent",
                "agentId": "client-agent-camel",
                "agent": "client-agent-alias",
                "subagent_id": "client-subagent",
                "subagentId": "client-subagent-camel",
                "account_id": "client-account",
                "accountId": "client-account-camel",
                "account": "client-account-alias",
                "user_id": "client-user",
                "userId": "client-user-camel",
                "user": "client-user-alias",
                "session_id": "client-session",
                "sessionId": "client-session-camel",
                "session": "client-session-alias",
            },
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert calls[0]["_ctx"].account_id == "acct-1"
    assert calls[0]["eventId"] == "evt-1"
    assert calls[0]["hookType"] == "tool_start"
    for stripped_field in (
        "hook_type",
        "hook",
        "hookName",
        "source",
        "producer",
        "producer_id",
        "producerId",
        "agent_id",
        "agentId",
        "agent",
        "subagent_id",
        "subagentId",
        "account_id",
        "accountId",
        "account",
        "user_id",
        "userId",
        "user",
        "session_id",
        "sessionId",
        "session",
    ):
        assert stripped_field not in calls[0]["event"]


def test_trace_route_defaults_missing_hook_type_to_unknown(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-unknown",
        "event_index": 5,
    }
    service.finalize_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("unknown events must not finalize a turn")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-unknown",
            "session_id": "sess",
            "event": {"metadata": "only"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert calls[0]["hookType"] == "unknown"
    assert calls[0]["event"] == {"event_type": "unknown", "metadata": "only"}


def test_trace_route_requires_api_key(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("unauthenticated trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
    )

    assert resp.status_code == 401


def test_trace_route_finalizes_stop_event(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-stop",
        "event_index": 12,
    }
    finalized = []
    service.finalize_session_trace = lambda params: finalized.append(params) or {
        "ok": True,
        "turn_id": "turn_sess_12",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-stop",
            "session_id": "sess",
            "event": {"hook_type": "Stop", "event_type": "assistant_message"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert resp.get_json()["turn"]["turn_id"] == "turn_sess_12"
    assert calls[0]["hookType"] == "turn_end"
    assert calls[0]["event"] == {
        "event_type": "turn_end",
    }
    assert finalized[0]["stopEventIndex"] == 12


def test_trace_route_finalizes_canonical_turn_end_event(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-turn-end",
        "event_index": 13,
    }
    finalized = []
    service.finalize_session_trace = lambda params: finalized.append(params) or {
        "ok": True,
        "turn_id": "turn_sess_13",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-turn-end",
            "session_id": "sess",
            "event": {"event_type": "turn_end"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert calls[0]["hookType"] == "turn_end"
    assert resp.get_json()["turn"]["turn_id"] == "turn_sess_13"
    assert finalized[0]["stopEventIndex"] == 13


@pytest.mark.parametrize(
    ("retryable", "expected_status"), [(False, 400), (True, 503)]
)
def test_trace_route_maps_finalize_failure_retryability(
    monkeypatch, tmp_path, retryable, expected_status
):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": True,
        "event_id": "evt-stop",
        "event_index": 12,
    }
    service.finalize_session_trace = lambda params: {
        "ok": False,
        "error": "finalize failed",
        "retryable": retryable,
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-stop",
            "session_id": "sess",
            "event": {"hook_type": "Stop"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == expected_status


def test_trace_route_returns_retryable_failure_when_store_is_unavailable(
    monkeypatch, tmp_path
):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": False,
        "error": "trace store unavailable",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 503


def test_trace_route_rejects_unowned_agent(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("unauthorized trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "agent_id": "unowned-agent",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_trace_route_rejects_oversized_body_before_persistence(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("oversized trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        data=b"x" * (64 * 1024 + 1),
        content_type="application/json",
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 413


def test_trace_route_rejects_overly_deep_event_before_persistence(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("deep trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    nested = "leaf"
    for _ in range(70):
        nested = [nested]

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"event_type": "tool_start", "payload": nested},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 413


def test_trace_route_rejects_json_recursion_before_persistence(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("recursive trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    body = (
        b'{"request_id":"evt-1","session_id":"sess","event":'
        + (b"[" * 20_000)
        + b"0"
        + (b"]" * 20_000)
        + b"}"
    )

    resp = client.post(
        "/api/v1/trace/events",
        data=body,
        content_type="application/json",
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 413


def test_trace_route_rejects_too_many_event_nodes_before_persistence(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("wide trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"event_type": "tool_start", "payload": [0] * 5_100},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 413


def test_trace_route_maps_permanent_validation_to_400(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": False,
        "error": "invalid event_time",
        "retryable": False,
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 400


def test_ip_allowlist_blocks_non_whitelisted_remote_addr(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "IP not allowed"


def test_ip_allowlist_allows_whitelisted_remote_addr(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1", "10.0.0.0/8"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_uses_forwarded_for_only_when_trusted(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["203.0.113.7"], trust_proxy=True, trusted_proxies=["10.10.10.10"]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_ignores_forwarded_for_from_untrusted_proxy(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["203.0.113.7"], trust_proxy=True, trusted_proxies=["10.0.0.0/8"]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7"},
        environ_base={"REMOTE_ADDR": "198.51.100.20"},
    )

    assert resp.status_code == 403


def test_ip_allowlist_ignores_forwarded_headers_when_proxy_not_configured(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["10.10.10.10"], trust_proxy=True, trusted_proxies=[]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7"},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_rejects_invalid_client_ip_with_log(monkeypatch, tmp_path, caplog):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "not-an-ip"},
    )

    assert resp.status_code == 403
    assert "Invalid client IP format" in caplog.text


def test_allowlist_rejection_does_not_initialize_service(monkeypatch, tmp_path):
    app_module._service = None
    monkeypatch.setattr(
        app_module,
        "get_config",
        lambda: OgMemConfig(http_ip_allowlist=["127.0.0.1"]),
    )

    def _boom():
        raise AssertionError("_get_service should not be called for rejected IPs")

    monkeypatch.setattr(app_module, "_get_service", _boom)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 403
