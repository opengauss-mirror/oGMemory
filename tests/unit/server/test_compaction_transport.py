from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("flask")

import server.app as app_module
from features.tool_result_externalization.store_base import ToolResultArtifactStore
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.session_manager import SessionOperationInProgress


def _make_service() -> MemoryService:
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct-test",
            user_id="user-test",
            agent_id="agent-test",
        )
    )
    service._get_context_fs = Mock(return_value=None)
    return service


class _FailingArtifactStore(ToolResultArtifactStore):
    def write(self, metadata, output_text, ctx):
        raise OSError("artifact store unavailable")

    def read_all(self, artifact_id, session_id, ctx):
        raise LookupError

    def list(self, session_id, ctx, limit=50, cursor=None):
        return {"artifacts": [], "truncated": False, "next_cursor": None}

    def delete_session_artifacts(self, session_id, ctx):
        return 0


def _make_real_compaction_failure_service(session_id: str) -> MemoryService:
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct-private",
            user_id="user-private",
            agent_id="agent-private",
            tool_result_externalization_enabled=True,
            tool_result_single_chars=1,
            after_turn_threshold=999_999,
        ),
        tool_result_store=_FailingArtifactStore(),
    )
    service._get_context_fs = Mock(return_value=None)
    ctx = service.build_context({"sessionId": session_id})
    service.get_session_manager().add_message(
        session_id,
        "toolResult",
        "private tool output",
        ctx,
    )
    return service


def _make_cleaning_service(session_id: str) -> MemoryService:
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct-cleaning",
            user_id="user-cleaning",
            agent_id="agent-cleaning",
            tool_result_externalization_enabled=True,
            tool_result_single_chars=1,
            after_turn_threshold=999_999,
        ),
        tool_result_store=_FailingArtifactStore(),
    )
    service._get_context_fs = Mock(return_value=None)
    ctx = service.build_context({"sessionId": session_id})
    service.get_session_manager().get_or_create(session_id, ctx=ctx).cleaning = True
    return service


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _nested_keys(child)
        }
    if isinstance(value, list):
        return {
            key
            for child in value
            for key in _nested_keys(child)
        }
    return set()


def test_prepare_compaction_http_route_returns_prepare_token(monkeypatch):
    service = _make_service()
    service.prepare_compaction = Mock(return_value={
        "prepareToken": "prepare-123",
        "messages": [{"role": "user", "content": "hello"}],
    })
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/prepare_compaction", json={"sessionId": "sess-prepare"})

    assert resp.status_code == 200
    assert resp.get_json() == {
        "ok": True,
        "prepared": True,
        "prepareToken": "prepare-123",
        "messagesPrepared": 1,
    }
@pytest.mark.parametrize("path", ["/api/v1/prepare_compaction", "/api/v1/compact"])
def test_real_compaction_governance_failure_excludes_internal_state(
    monkeypatch,
    path,
):
    session_id = "sess-private"
    service = _make_real_compaction_failure_service(session_id)
    monkeypatch.setattr(app_module, "_service", service)

    resp = app_module.app.test_client().post(path, json={"sessionId": session_id})

    assert resp.status_code == 503
    body = resp.get_json()
    assert body["ok"] is False
    assert body["retryable"] is True
    assert body["status"] == "tool_result_governance_pending"
    assert body["session_id"] == session_id
    assert body["governance"] == {
        "governed": False,
        "changed": 1,
        "failed_count": 1,
        "retryable": True,
    }
    assert _nested_keys(body).isdisjoint({
        "ctx",
        "mgr",
        "buffer",
        "account_id",
        "user_id",
        "agent_id",
        "trace_id",
        "visible_owner_spaces",
        "role",
    })


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/compose", {"messages": [{"role": "user", "content": "next"}]}),
        ("/api/v1/after_turn", {"messages": [{"role": "user", "content": "next"}]}),
        ("/api/v1/ingest", {"messages": [{"role": "user", "content": "next"}]}),
        ("/api/v1/compact", {}),
        ("/api/v1/prepare_compaction", {}),
        ("/api/v1/dispose", {}),
        ("/api/v1/sessions/{session_id}/messages", {
            "role": "user",
            "content": "next",
        }),
    ],
)
def test_session_deletion_conflict_is_structured_nonretryable_409(
    monkeypatch,
    path,
    body,
):
    session_id = "sess-cleaning"
    service = _make_cleaning_service(session_id)
    monkeypatch.setattr(app_module, "_service", service)
    resolved_path = path.format(session_id=session_id)

    resp = app_module.app.test_client().post(
        resolved_path,
        json={"sessionId": session_id, **body},
    )

    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["status"] == 409
    assert payload["error_code"] == "session_operation_in_progress"
    assert payload["conflict_type"] == "session_deleting"
    assert payload["retryable"] is False


def test_session_replacement_conflict_is_structured_retryable_409(monkeypatch):
    service = _make_service()
    service.compose = Mock(side_effect=SessionOperationInProgress(
        conflict_type="session_replaced",
        retryable=True,
    ))
    monkeypatch.setattr(app_module, "_service", service)

    resp = app_module.app.test_client().post(
        "/api/v1/compose",
        json={"sessionId": "sess-replaced", "messages": []},
    )

    assert resp.status_code == 409
    assert resp.get_json() == {
        "ok": False,
        "status": 409,
        "error_code": "session_operation_in_progress",
        "conflict_type": "session_replaced",
        "retryable": True,
    }
def test_compact_http_route_rejects_prepared_without_token(monkeypatch):
    service = _make_service()
    service.get_session_manager().add_message(
        "sess-compact",
        "user",
        "pending content",
        service.build_context({"sessionId": "sess-compact"}),
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compact",
        json={"sessionId": "sess-compact", "prepared": True},
    )

    assert resp.status_code == 400
    assert "prepare token required" in resp.get_json()["error"]


def test_session_working_set_http_route_lists_access_stats(monkeypatch):
    service = _make_service()
    mgr = service.get_session_manager()
    ctx = service.build_context({"sessionId": "sess-ws"})
    buf = mgr.get_or_create("sess-ws", ctx)
    buf.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get("/api/v1/session_working_set")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["sessions"][0]["session_id"] == "sess-ws"
    assert body["sessions"][0]["last_accessed_at"] == "2026-05-20T02:00:00+00:00"


def test_evict_idle_sessions_http_route_uses_last_accessed_at(monkeypatch):
    service = _make_service()
    mgr = service.get_session_manager()
    ctx = service.build_context({"sessionId": "active"})
    active = mgr.get_or_create("active", ctx)
    idle = mgr.get_or_create("idle", ctx)
    active.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/evict_idle_sessions",
        json={"maxIdleSeconds": 1800, "nowIso": "2026-05-20T02:00:01+00:00"},
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "evicted": ["idle"]}
    assert mgr.has_session("active") is True
    assert mgr.has_session("idle") is False
