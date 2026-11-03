from __future__ import annotations

from types import SimpleNamespace

import pytest

import server.app as app_module
import server.memory_viewer as memory_viewer_module
from commit.code_resource_events import build_code_resource_write_event
from core.models import ContextNode, OutboxEvent, RequestContext, Role
from server.auth import ResolvedIdentity


class _ContextFS:
    def __init__(self, node):
        self.node = node

    def exists(self, uri, ctx):
        return uri.endswith("/memories/profile")

    def read_node(self, uri, ctx):
        return self.node

    def list_children(self, uri, ctx):
        return []


class _Service:
    def __init__(self):
        self._cfg = SimpleNamespace(
            http_ip_allowlist=[],
            http_ip_allowlist_trust_proxy=False,
            http_trusted_proxies=[],
        )
        self.node = SimpleNamespace(
            uri="ctx://acct/users/alice/memories/profile",
            category="profile",
            abstract="Alice prefers concise answers.",
            overview="Communication preferences",
            content="# Profile\n\nConcise answers.",
            metadata={"updated_at": "2026-08-25T12:00:00Z"},
        )
        self.context_fs = _ContextFS(self.node)

    def get_auth_service(self):
        return SimpleNamespace(role_control_active=lambda: False)

    @property
    def default_agent_id(self):
        return "agent"

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct",
            user_id="alice",
            agent_id=params.get("agentId") or "agent",
            session_id="http",
            trace_id="trace",
            role=Role.ROOT,
        )

    def _get_context_fs(self):
        return self.context_fs

    def _get_outbox_store(self, fs):
        return _TaskStore()

    def get_model_usage_metrics(self, days=14, *, account_id="", owner_spaces=()):
        return {
            "enabled": True,
            "models": {"llm": "qwen-plus", "embedding": "text-embedding-v3"},
            "today": {
                "date": "2026-08-27",
                "llm_input": 120,
                "llm_output": 30,
                "embedding_input": 50,
                "total": 200,
            },
            "cumulative": {
                "total_tokens": 500,
                "llm": {"input_tokens": 300, "output_tokens": 100, "calls": 4},
                "embedding": {"total_tokens": 100, "calls": 2},
            },
            "series": [],
            "days": days,
            "scope": {"account_id": account_id, "owner_spaces": list(owner_spaces)},
        }

    def get_monitoring_overview(self, ctx):
        return {
            "account_id": ctx.account_id,
            "status": "ok",
            "healthy": True,
            "healthy_count": 6,
            "total_count": 6,
            "updated_at": "2026-08-29T00:00:00+00:00",
            "components": [],
        }

    def list_published_skills(self, ctx, *, limit=100, cursor=None, source=None):
        return {
            "skills": [
                {
                    "name": "review-code",
                    "description": "Review code before release.",
                    "source": "generated",
                    "updated_at": "2026-08-25T13:00:00Z",
                }
            ],
            "truncated": False,
        }

    def read_published_skill(self, ctx, name, source=None):
        return SimpleNamespace(
            uri="ctx://acct/agents/agent/skills/generated/review-code",
            category="skill",
            abstract="Review code before release.",
            overview="Published Agent Skill",
            content="# Review code",
            metadata={"publication_status": "published"},
        )


class _TaskStore:
    def list_pending(self, account_id):
        return [
            (
                "ctx://acct/users/alice/memories/profile",
                OutboxEvent(
                    event_id="task-alice",
                    event_type="UPSERT_CONTEXT",
                    uri="ctx://acct/users/alice/memories/profile",
                    payload={"filters": {"owner_space": "user:alice"}},
                    status="PENDING",
                    created_at="2026-08-31T08:00:00+00:00",
                ),
            ),
            (
                "ctx://acct/users/bob/memories/events/item",
                OutboxEvent(
                    event_id="task-bob",
                    event_type="UPSERT_DIRECTORY",
                    uri="ctx://acct/users/bob/memories/events/item",
                    payload={"filters": {"owner_space": "user:bob"}},
                    status="PROCESSING",
                    created_at="2026-08-31T07:59:00+00:00",
                ),
            ),
        ]


class _RetryTaskStore(_TaskStore):
    def __init__(self):
        self.events = {
            "task-alice-failed": OutboxEvent(
                event_id="task-alice-failed",
                event_type="UPSERT_CONTEXT",
                uri="ctx://acct/users/alice/memories/profile",
                payload={
                    "filters": {"owner_space": "user:alice"},
                    "failure_reason": "embedding unavailable",
                },
                status="FAILED",
                retry_count=5,
            ),
            "task-bob-failed": OutboxEvent(
                event_id="task-bob-failed",
                event_type="UPSERT_CONTEXT",
                uri="ctx://acct/users/bob/memories/profile",
                payload={"filters": {"owner_space": "user:bob"}},
                status="FAILED",
            ),
        }
        self.retried = []

    def get_event(self, account_id, event_id):
        return self.events.get(event_id)

    def retry_failed(self, account_id, event):
        self.retried.append((account_id, event.event_id))
        event.status = "PENDING"
        event.retry_count = 0
        return event


class _RBACService(_Service):
    def get_auth_service(self):
        return SimpleNamespace(
            role_control_active=lambda: True,
            resolve_identity=lambda headers: ResolvedIdentity(
                role=Role.MEMBER,
                account_id="acct",
                user_id="alice",
            ),
        )

    def get_tenant_admin_service(self):
        return SimpleNamespace(
            list_visible_agent_ids=lambda account_id, user_id, cfg: ["agent-a", "agent-b"]
        )

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct",
            user_id="alice",
            agent_id=params.get("agentId", ""),
            session_id="http",
            trace_id="trace",
            role=Role.MEMBER,
            visible_owner_spaces=("user:alice", "agent:agent-a", "agent:agent-b"),
        )


class _RetryService(_RBACService):
    def __init__(self):
        super().__init__()
        self.task_store = _RetryTaskStore()

    def _get_outbox_store(self, fs):
        return self.task_store


class _AdminService(_Service):
    def get_auth_service(self):
        return SimpleNamespace(
            role_control_active=lambda: True,
            resolve_identity=lambda headers: ResolvedIdentity(
                role=Role.ADMIN,
                account_id="acct",
                user_id="admin",
            ),
        )

    def get_tenant_admin_service(self):
        return SimpleNamespace(
            list_users=lambda ctx, account_id: {
                "users": [
                    {"user_id": "alice", "role": "user"},
                    {"user_id": "bob", "role": "user"},
                ]
            },
            list_agents=lambda ctx, account_id: {
                "agents": [
                    {"agent_id": "agent-a", "owner_user_id": "alice"},
                    {"agent_id": "agent-b", "owner_user_id": "bob"},
                ]
            },
        )

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct",
            user_id="admin",
            agent_id="",
            session_id="http",
            trace_id="trace",
            role=Role.ADMIN,
        )


class _RootService(_AdminService):
    def get_auth_service(self):
        return SimpleNamespace(
            role_control_active=lambda: True,
            resolve_identity=lambda headers: ResolvedIdentity(
                role=Role.ROOT,
                account_id="root",
                user_id="root",
            ),
        )

    def get_tenant_admin_service(self):
        admin = super().get_tenant_admin_service()
        admin.list_accounts = lambda ctx: {"accounts": [{"account_id": "acct"}]}
        return admin

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="root",
            user_id="root",
            agent_id="",
            session_id="http",
            trace_id="trace",
            role=Role.ROOT,
        )


def test_memory_viewer_serves_ui_and_memory_data(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    client = app_module.app.test_client()

    page = client.get("/memory-viewer/")
    scope = client.get("/api/v1/memory-viewer/scope")
    listing = client.get("/api/v1/memories", query_string={"agentId": "agent"})
    detail = client.get(
        "/api/v1/memories/detail",
        query_string={"uri": "ogmemory://memories/profile/self"},
    )
    skill = client.get(
        "/api/v1/memories/detail",
        query_string={
            "agentId": "agent",
            "uri": "ogmemory://skills/generated/review-code",
        },
    )

    assert page.status_code == 200
    assert "oGMemory" in page.get_data(as_text=True)
    assert "任务中心" in page.get_data(as_text=True)
    assert "task-retry" in page.get_data(as_text=True)
    assert scope.get_json()["agents"] == [{"agent_id": "agent"}]
    assert scope.get_json()["role_control_active"] is False
    assert scope.get_json()["admin_mode"] is False
    assert {item["type"] for item in listing.get_json()["memories"]} == {"profile", "skill"}
    assert detail.get_json()["content"] == "# Profile\n\nConcise answers."
    assert skill.get_json()["content"] == "# Review code"


def test_memory_viewer_frontend_shares_session_api_key_and_ignores_stale_tree_requests(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())

    response = app_module.app.test_client().get("/memory-viewer/app.js")
    script = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "sessionStorage.getItem(API_KEY_STORAGE_KEY)" in script
    assert "sessionStorage.setItem(API_KEY_STORAGE_KEY" in script
    assert "new BroadcastChannel(API_KEY_CHANNEL_NAME)" in script
    assert "localStorage.getItem(API_KEY_STORAGE_KEY" not in script
    assert "localStorage.setItem(API_KEY_STORAGE_KEY" not in script
    assert "localStorage.removeItem(API_KEY_STORAGE_KEY)" in script
    assert "treeRequestId" in script
    assert "requestId !== state.treeRequestId" in script
    assert "/api/v1/memory-viewer/tasks" in script
    assert "失败原因" in script
    assert "TASK_ERROR_MESSAGES" in script
    assert "formatTaskError" in script
    assert "/retry`" in script
    assert "taskRetryEnabled" in script
    assert "state.view === 'tasks' && !document.hidden" in script
    assert "loadTasks({ silent: true })" in script
    assert "tasksSnapshot" in script
    assert "response.headers.get('content-type')" in script


def test_memory_viewer_without_role_control_lists_user_directory(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())

    response = app_module.app.test_client().get(
        "/api/v1/memories",
        query_string={"type": "profile", "limit": 100, "sort": "name"},
    )

    assert response.status_code == 200
    assert [item["type"] for item in response.get_json()["memories"]] == ["profile"]
    assert response.get_json()["memories"][0]["context_uri"] == (
        "ctx://acct/users/alice/memories/profile"
    )


def test_memory_viewer_detail_does_not_hide_storage_failure(monkeypatch):
    service = _Service()
    calls = []

    def read_node(uri, ctx):
        calls.append(uri)
        if len(calls) == 1:
            raise RuntimeError("storage unavailable")
        return service.node

    service.context_fs.read_node = read_node
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get(
        "/api/v1/memories/detail",
        query_string={"uri": "ogmemory://memories/pattern/demo"},
    )

    assert response.status_code == 500
    assert response.get_json()["error"] == "storage unavailable"
    assert len(calls) == 1


def test_memory_viewer_detail_falls_back_after_not_found(monkeypatch):
    service = _Service()
    calls = []

    def read_node(uri, ctx):
        calls.append(uri)
        if len(calls) == 1:
            raise FileNotFoundError(uri)
        return service.node

    service.context_fs.read_node = read_node
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get(
        "/api/v1/memories/detail",
        query_string={"uri": "ogmemory://memories/pattern/demo"},
    )

    assert response.status_code == 200
    assert len(calls) == 2


def test_memory_viewer_public_listing_reads_one_bounded_page(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    calls = []

    def list_page(service, context, *, limit, cursor):
        calls.append((limit, cursor))
        return (
            [
                (
                    "ogmemory://memories/event/first",
                    {
                        "uri": "ogmemory://memories/event/first",
                        "type": "event",
                        "id": "first",
                    },
                )
            ],
            False,
        )

    monkeypatch.setattr(memory_viewer_module, "_list_memory_node_entries_page", list_page)

    response = app_module.app.test_client().get(
        "/api/v1/memories",
        query_string={"agentId": "agent", "limit": 1},
    )

    assert response.status_code == 200
    assert response.get_json()["truncated"] is True
    assert calls == [(1, None)]


def test_memory_viewer_rejects_invalid_query_parameters(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _AdminService())
    client = app_module.app.test_client()
    headers = {"X-API-Key": "admin-key"}

    invalid_limit = client.get(
        "/api/v1/memories",
        query_string={"limit": "abc"},
        headers=headers,
    )
    out_of_range_page = client.get(
        "/api/v1/memory-viewer/admin/memories",
        query_string={"accountId": "acct", "userId": "alice", "page": 0},
        headers=headers,
    )
    invalid_uri = client.get(
        "/api/v1/memories/detail",
        query_string={"uri": "invalid"},
        headers=headers,
    )

    assert invalid_limit.status_code == 400
    assert out_of_range_page.status_code == 400
    assert invalid_uri.status_code == 400


def test_memory_viewer_lists_one_bounded_skill_page():
    skills = [
        {
            "name": f"skill-{index:03d}",
            "description": f"Skill {index}",
            "source": "generated",
        }
        for index in range(501)
    ]

    class PaginatedSkillService:
        def list_published_skills(self, ctx, *, limit=100, cursor=None, source=None):
            start = int(cursor or 0)
            page = skills[start : start + limit]
            next_cursor = start + len(page)
            return {
                "skills": page,
                "truncated": next_cursor < len(skills),
                "next_cursor": str(next_cursor),
            }

    summaries, truncated = memory_viewer_module._list_skill_page(
        PaginatedSkillService(),
        SimpleNamespace(agent_id="agent"),
        limit=100,
        cursor=None,
    )

    assert len(summaries) == 100
    assert truncated is True
    assert summaries[-1]["id"] == "skill-099"


def test_memory_viewer_scope_lists_member_visible_agents(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RBACService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/scope",
        headers={"X-API-Key": "member-key"},
    )

    assert response.get_json()["role_control_active"] is True
    assert response.get_json()["admin_mode"] is False
    assert response.get_json()["agents"] == [
        {"agent_id": "agent-a"},
        {"agent_id": "agent-b"},
    ]


def test_memory_viewer_scope_enables_admin_mode_for_admin(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _AdminService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/scope",
        headers={"X-API-Key": "admin-key"},
    )

    assert response.get_json()["admin_mode"] is True


def test_memory_viewer_metrics_allows_admin(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _AdminService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/metrics",
        query_string={"days": 30},
        headers={"X-API-Key": "admin-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["days"] == 30
    assert response.get_json()["scope"]["account_id"] == "acct"


def test_memory_viewer_metrics_denies_admin_cross_account(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _AdminService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/metrics",
        query_string={"accountId": "other"},
        headers={"X-API-Key": "admin-key"},
    )

    assert response.status_code == 403


def test_memory_viewer_metrics_allows_root_account_switch(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RootService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/metrics",
        query_string={"accountId": "acct"},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["scope"]["account_id"] == "acct"


def test_memory_viewer_metrics_scopes_member_to_visible_owners(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RBACService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/metrics",
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["scope"] == {
        "account_id": "acct",
        "owner_spaces": ["user:alice", "agent:agent-a", "agent:agent-b"],
    }


def test_memory_viewer_monitoring_uses_selected_root_account(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RootService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/monitoring",
        query_string={"accountId": "acct"},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["account_id"] == "acct"


def test_memory_viewer_tasks_lists_outbox_work(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())

    response = app_module.app.test_client().get("/api/v1/memory-viewer/tasks")

    assert response.status_code == 200
    assert [task["task_id"] for task in response.get_json()["tasks"]] == [
        "task-alice",
        "task-bob",
    ]
    assert response.get_json()["tasks"][0] == {
        "created_at": "2026-08-31T08:00:00+00:00",
        "event_type": "UPSERT_CONTEXT",
        "error_code": "",
        "error_params": {},
        "resource_id": "ctx://acct/users/alice/memories/profile",
        "retry_count": 0,
        "stage": "index_sync",
        "status": "pending",
        "task_id": "task-alice",
        "task_type": "memory_index",
        "updated_at": "2026-08-31T08:00:00+00:00",
    }


def test_memory_viewer_failed_task_hides_internal_error():
    event = OutboxEvent(
        event_id="failed-task",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/users/alice/memories/profile",
        payload={"failure_reason": "'MemoryService' object has no attribute 'get_vector_index'"},
        status="FAILED",
        retry_count=5,
    )

    task = memory_viewer_module._task_record(event)

    assert task["error_code"] == "TASK_COMPONENT_ERROR"
    assert task["error_params"] == {
        "component": "任务处理组件",
        "resource_id": "ctx://acct/users/alice/memories/profile",
    }


def test_memory_viewer_failed_task_does_not_invent_missing_reason():
    event = OutboxEvent(
        event_id="historical-task",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/users/alice/memories/profile",
        payload={},
        status="FAILED",
        retry_count=5,
    )

    task = memory_viewer_module._task_record(event)

    assert task["error_code"] == ""
    assert task["error_params"] == {}


def test_memory_viewer_vector_failure_includes_action():
    event = OutboxEvent(
        event_id="vector-timeout",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/users/alice/memories/profile",
        payload={"failure_reason": "向量数据库连接超时"},
        status="FAILED",
        retry_count=5,
    )

    task = memory_viewer_module._task_record(event)

    assert task["error_code"] == "TASK_VECTOR_DB_UNAVAILABLE"
    assert task["error_params"] == {
        "dependency": "openGauss",
        "resource_id": "ctx://acct/users/alice/memories/profile",
    }


def test_memory_viewer_resource_failure_returns_template_params():
    event = OutboxEvent(
        event_id="missing-resource",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/users/alice/memories/profile",
        payload={"failure_reason": "待索引资源不存在"},
        status="FAILED",
        retry_count=1,
    )

    task = memory_viewer_module._task_record(event)

    assert task["error_code"] == "TASK_RESOURCE_NOT_FOUND"
    assert task["error_params"] == {
        "failed_step": "ContextWriter 将记忆节点写入 ContextFS",
        "resource_id": "ctx://acct/users/alice/memories/profile",
        "upstream_step": "记忆写入 → ContextWriter 构建节点 → 写入 ContextFS",
    }


def test_memory_viewer_session_resource_failure_identifies_upstream_step():
    event = OutboxEvent(
        event_id="missing-session-archive",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/users/alice/memories/session_archives/session-1",
        payload={"failure_reason": "待索引资源不存在"},
        status="FAILED",
        retry_count=1,
    )

    task = memory_viewer_module._task_record(event)

    assert task["error_code"] == "TASK_RESOURCE_NOT_FOUND"
    assert task["error_params"]["failed_step"] == "将会话归档写入 ContextFS"
    assert task["error_params"]["upstream_step"] == (
        "会话提交 → 生成会话归档 → 写入 ContextFS"
    )


def test_memory_viewer_tasks_filters_member_visibility(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RBACService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/tasks",
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 200
    assert [task["task_id"] for task in response.get_json()["tasks"]] == ["task-alice"]


def test_memory_viewer_member_retries_own_failed_task(monkeypatch):
    service = _RetryService()
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/tasks/task-alice-failed/retry",
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["task"]["status"] == "pending"
    assert response.get_json()["task"]["retry_count"] == 0
    assert service.task_store.retried == [("acct", "task-alice-failed")]


def test_memory_viewer_member_cannot_retry_other_users_task(monkeypatch):
    service = _RetryService()
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/tasks/task-bob-failed/retry",
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 404
    assert service.task_store.retried == []


def test_memory_viewer_rejects_retry_after_task_state_changes(monkeypatch):
    service = _RetryService()
    service.task_store.events["task-alice-failed"].status = "PENDING"
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/tasks/task-alice-failed/retry",
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 409
    assert service.task_store.retried == []


def test_task_visibility_resolves_session_owner_from_event_payload():
    event = build_code_resource_write_event(ContextNode(
        uri="ctx://acct/sessions/session-1/history/archive-1",
        context_type="MEMORY",
        category="event",
        level=4,
        owner_space="session:session-1",
        abstract="Session archive",
        overview="Session archive",
        content="Archived conversation",
        metadata={"owner_user_id": "alice", "owner_agent_id": "agent-a"},
    ))

    assert memory_viewer_module._task_visible(event, ("user:alice",))
    assert memory_viewer_module._task_visible(event, ("agent:agent-a",))
    assert not memory_viewer_module._task_visible(event, ("user:bob",))


def test_memory_viewer_search_scopes_member_to_selected_agent(monkeypatch):
    captured = {}

    def search(service, **kwargs):
        captured.update(kwargs)
        return {"query": kwargs["query"], "hits": []}

    monkeypatch.setattr(app_module, "_service", _RBACService())
    monkeypatch.setattr(memory_viewer_module, "query_memory", search)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/search",
        json={"query": "concise answers", "agentId": "agent-a", "top_k": 10},
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 200
    assert captured["request_context"].visible_owner_spaces == (
        "user:alice",
        "agent:agent-a",
    )
    assert captured["agent_id"] == "agent-a"


def test_memory_viewer_search_scopes_member_to_user_only(monkeypatch):
    captured = {}

    def search(service, **kwargs):
        captured.update(kwargs)
        return {"query": kwargs["query"], "hits": []}

    monkeypatch.setattr(app_module, "_service", _RBACService())
    monkeypatch.setattr(memory_viewer_module, "query_memory", search)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/search",
        json={"query": "concise answers", "userId": "alice", "top_k": 10},
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 200
    assert captured["request_context"].agent_id == ""
    assert captured["request_context"].visible_owner_spaces == ("user:alice",)
    assert captured["agent_id"] == ""


def test_memory_viewer_search_rejects_other_user_for_member(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RBACService())

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/search",
        json={"query": "private memory", "userId": "bob"},
        headers={"X-API-Key": "member-key"},
    )

    assert response.status_code == 403


def test_memory_viewer_search_allows_admin_target(monkeypatch):
    captured = {}

    def search(service, **kwargs):
        captured.update(kwargs)
        return {"query": kwargs["query"], "hits": []}

    monkeypatch.setattr(app_module, "_service", _AdminService())
    monkeypatch.setattr(memory_viewer_module, "query_memory", search)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/search",
        json={
            "query": "release review",
            "accountId": "acct",
            "userId": "alice",
            "agentId": "agent-a",
            "sessionId": "selected-session",
            "types": ["pattern"],
            "top_k": 5,
        },
        headers={"X-API-Key": "admin-key"},
    )

    assert response.status_code == 200
    assert captured["types"] == ["pattern"]
    assert captured["top_k"] == 5
    assert captured["request_context"].session_id == "selected-session"
    assert captured["request_context"].visible_owner_spaces == (
        "user:alice",
        "agent:agent-a",
    )


def test_memory_viewer_search_allows_admin_account_scope(monkeypatch):
    captured = {}

    def search(service, **kwargs):
        captured.update(kwargs)
        return {"query": kwargs["query"], "hits": []}

    monkeypatch.setattr(app_module, "_service", _AdminService())
    monkeypatch.setattr(memory_viewer_module, "query_memory", search)

    response = app_module.app.test_client().post(
        "/api/v1/memory-viewer/search",
        json={"query": "release review", "accountId": "acct", "top_k": 10},
        headers={"X-API-Key": "admin-key"},
    )

    assert response.status_code == 200
    assert captured["request_context"].visible_owner_spaces == (
        "user:alice",
        "user:bob",
        "agent:agent-a",
        "agent:agent-b",
    )


def test_memory_viewer_admin_hierarchy_and_targets(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _AdminService())
    client = app_module.app.test_client()
    headers = {"X-API-Key": "admin-key"}

    tree = client.get("/api/v1/memory-viewer/admin/tree", headers=headers)
    user_memories = client.get(
        "/api/v1/memory-viewer/admin/memories",
        query_string={"accountId": "acct", "userId": "alice", "type": "profile"},
        headers=headers,
    )
    agent_memories = client.get(
        "/api/v1/memory-viewer/admin/memories",
        query_string={
            "accountId": "acct",
            "userId": "alice",
            "agentId": "agent-a",
            "type": "skill",
        },
        headers=headers,
    )
    skill = client.get(
        "/api/v1/memory-viewer/admin/detail",
        query_string={
            "accountId": "acct",
            "userId": "alice",
            "agentId": "agent-a",
            "uri": "ogmemory://skills/generated/review-code",
        },
        headers=headers,
    )

    assert tree.get_json()["users"][0] == {
        "user_id": "alice",
        "role": "user",
        "agents": [{"agent_id": "agent-a"}],
    }
    assert tree.get_json()["agents"][0] == {
        "agent_id": "agent-a",
        "owner_user_id": "alice",
    }
    assert [item["type"] for item in user_memories.get_json()["memories"]] == ["profile"]
    assert user_memories.get_json()["memories"][0]["context_uri"] == (
        "ctx://acct/users/alice/memories/profile"
    )
    assert [item["type"] for item in agent_memories.get_json()["memories"]] == ["skill"]
    assert skill.get_json()["content"] == "# Review code"
    assert skill.get_json()["context_uri"] == (
        "ctx://acct/agents/agent/skills/generated/review-code"
    )


def test_memory_viewer_inventory_matches_directories_and_account_scope(monkeypatch):
    class InventoryService(_RootService):
        def get_tenant_admin_service(self):
            return SimpleNamespace(
                list_accounts=lambda ctx: {"accounts": [
                    {"account_id": "acct"}, {"account_id": "other"},
                ]},
                list_users=lambda ctx, account_id: {"users": [
                    {"user_id": "alice", "role": "user"},
                ]},
                list_agents=lambda ctx, account_id: {"agents": [
                    {"agent_id": "helper", "owner_user_id": "alice"},
                ]},
            )

        def list_published_skills(self, ctx, *, limit=100, cursor=None, source=None):
            return {"skills": [{"name": "check", "source": "generated"}], "truncated": False}

    service = InventoryService()
    service.context_fs = SimpleNamespace(count_memory_children=lambda ctx: {
        f"ctx://{ctx.account_id}/users/alice/memories/profile": 1,
        f"ctx://{ctx.account_id}/users/alice/memories/events": 130,
        f"ctx://{ctx.account_id}/agents/helper/memories/tools": 2,
        f"ctx://{ctx.account_id}/users/hidden/memories/events": 99,
    })
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    result = client.get(
        "/api/v1/memory-viewer/inventory", query_string={"accountId": "other"},
        headers={"X-API-Key": "root-key"},
    )

    assert result.status_code == 200
    data = result.get_json()
    assert data["account_id"] == "other"
    assert data["memories"] == 133
    assert data["skills"] == 1
    assert data["directories"]["ctx://other/users/alice/memories/events"] == 130
    assert "ctx://other/users/hidden/memories/events" not in data["directories"]

    monkeypatch.setattr(app_module, "_service", _AdminService())
    denied = client.get(
        "/api/v1/memory-viewer/inventory", query_string={"accountId": "other"},
        headers={"X-API-Key": "admin-key"},
    )
    assert denied.status_code in {403, 404}


def test_memory_viewer_inventory_scopes_public_user(monkeypatch):
    service = _Service()
    service.context_fs = SimpleNamespace(count_memory_children=lambda ctx: {
        "ctx://acct/users/alice/memories/profile": 1,
        "ctx://acct/agents/agent/memories/tools": 3,
        "ctx://acct/users/bob/memories/events": 80,
    })
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get("/api/v1/memory-viewer/inventory")
    assert response.status_code == 200
    body = response.get_json()
    assert body["memories"] == 4
    assert body["skills"] == 1
    assert "ctx://acct/users/bob/memories/events" not in body["directories"]


def test_memory_viewer_filters_skill_source(monkeypatch):
    service = _AdminService()

    def list_skills(ctx, *, limit=100, cursor=None, source=None):
        assert source == "imported"
        return {
            "skills": [
                {
                    "name": "imported-skill",
                    "description": "Imported Skill",
                    "source": "imported",
                }
            ],
            "truncated": False,
        }

    service.list_published_skills = list_skills
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/admin/memories",
        query_string={
            "accountId": "acct",
            "userId": "alice",
            "agentId": "agent-a",
            "type": "skill",
            "source": "imported",
        },
        headers={"X-API-Key": "admin-key"},
    )

    assert response.status_code == 200, response.get_json()
    assert [item["id"] for item in response.get_json()["memories"]] == ["imported-skill"]
    assert response.get_json()["memories"][0]["context_uri"] == (
        "ctx://acct/agents/agent-a/skills/imported/imported-skill"
    )


def test_memory_viewer_root_falls_back_from_stale_account(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _RootService())

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/admin/tree",
        query_string={"accountId": "stale-account"},
        headers={"X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["account_id"] == "acct"


def test_memory_viewer_admin_memories_uses_bounded_cursor_page(monkeypatch):
    service = _AdminService()
    calls = []

    class PagedContextFS(_ContextFS):
        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            calls.append((uri, limit, after_uri))
            start = int(after_uri.rsplit("-", 1)[-1]) + 1 if after_uri else 0
            return [
                SimpleNamespace(
                    uri=f"{uri}/memory-{index:02d}",
                    category="event",
                    abstract=f"Memory {index}",
                    content="",
                    metadata={},
                )
                for index in range(start, min(start + limit, 25))
            ]

    service.context_fs = PagedContextFS(service.node)
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get(
        "/api/v1/memory-viewer/admin/memories",
        query_string={
            "accountId": "acct",
            "userId": "alice",
            "type": "event",
            "limit": 10,
            "cursor": "ogmemory://memories/event/memory-09",
            "sort": "name",
        },
        headers={"X-API-Key": "admin-key"},
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["count"] == 10
    assert body["truncated"] is True
    assert body["memories"][0]["id"] == "memory-10"
    assert calls == [
        (
            "ctx://acct/users/alice/memories/events",
            11,
            "ctx://acct/users/alice/memories/events/memory-09",
        )
    ]


@pytest.mark.parametrize("admin,typed", [(False, False), (False, True), (True, True)])
def test_memory_viewer_pages_legacy_and_unicode_names(monkeypatch, admin, typed):
    service = _AdminService() if admin else _Service()
    names = ["%E6%9D%8E%E9%9B%B7", "atlas", "李雷"]
    nodes = [
        SimpleNamespace(
            uri=f"ctx://acct/users/alice/memories/entities/{name}",
            category="entity", abstract=name, content="", metadata={},
        )
        for name in names
    ]

    class PagedContextFS(_ContextFS):
        def exists(self, uri, ctx):
            return any(node.uri == uri for node in nodes)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            if not uri.endswith("/memories/entities"):
                return []
            return [node for node in nodes if after_uri is None or node.uri > after_uri][:limit]

    service.context_fs = PagedContextFS(service.node)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    path = "/api/v1/memory-viewer/admin/memories" if admin else "/api/v1/memories"
    query = {"limit": 1}
    if typed:
        query["type"] = "entity"
    if admin:
        query.update(accountId="acct", userId="alice")
    headers = {"X-API-Key": "admin-key"} if admin else {}

    seen = []
    cursors = []
    for page_index in range(len(names)):
        response = client.get(path, query_string=query, headers=headers)
        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        seen.extend(item["id"] for item in body["memories"])
        if typed and page_index == 0:
            assert body["memories"][0]["context_uri"].endswith("/%E6%9D%8E%E9%9B%B7")
        if not body["truncated"]:
            break
        cursors.append(body["next_cursor"])
        query["cursor"] = body["next_cursor"]

    assert seen == ["李雷", "atlas", "李雷"]
    assert len(cursors) == len(set(cursors))
