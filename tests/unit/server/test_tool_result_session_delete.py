from __future__ import annotations

import pytest

from core.models import RequestContext, Role
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


@pytest.fixture(autouse=True)
def local_session_backend(monkeypatch):
    monkeypatch.setattr(MemoryService, "_get_context_fs", lambda self: None)


def _ctx(role=Role.ADMIN, **overrides):
    values = {
        "account_id": "acct",
        "user_id": "user",
        "agent_id": "agent",
        "session_id": "session",
        "trace_id": "trace",
        "role": role,
    }
    values.update(overrides)
    return RequestContext(**values)


def _service(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=1,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_archive_store = lambda: None
    manager = service.get_session_manager()
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _delete_params(ctx):
    return {
        "sessionId": "session",
        "_ctx": ctx,
    }


def test_session_delete_returns_409_while_operation_is_active(tmp_path, monkeypatch):
    service = _service(tmp_path)
    mgr = service.get_session_manager()
    buf = mgr.get_or_create("session", ctx=_ctx())
    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("user", "agent"))
    buf.begin_operation()
    try:
        result = service.delete_session(_delete_params(_ctx()))
    finally:
        buf.end_operation()
    assert result == {
        "ok": False,
        "status": 409,
        "error_code": "session_operation_in_progress",
    }
def test_session_delete_cleans_artifacts_archives_state_and_buffer(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service.add_session_message({
        "sessionId": "session",
        "message": {"role": "toolResult", "content": "secret"},
        "_ctx": _ctx(Role.MEMBER),
    })
    mgr = service.get_session_manager()
    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("user", "agent"))
    monkeypatch.setattr(mgr, "delete_session_archives", lambda *args: 2)
    monkeypatch.setattr(mgr, "delete_persisted_session_state", lambda *args: 1)

    result = service.delete_session(_delete_params(_ctx()))
    assert result["ok"] is True
    assert result["steps"]["artifacts"]["deleted"] == 1
    assert result["steps"]["archives"]["deleted"] == 2
    assert result["steps"]["session_state"]["deleted"] == 1
    assert not mgr.has_session("session", ctx=_ctx())
    assert service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"] == []


def test_session_delete_cleans_prior_artifacts_while_externalization_is_disabled(
    tmp_path,
    monkeypatch,
):
    store = LocalToolResultArtifactStore(tmp_path)
    enabled = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=1,
            after_turn_threshold=999_999,
        ),
        tool_result_store=store,
    )
    enabled.govern_first_view_tool_result({
        "sessionId": "session",
        "adapterId": "test",
        "turnId": "turn-old-artifact",
        "toolCallId": "call-old-artifact",
        "toolName": "fixture",
        "content": "old artifact",
        "_ctx": _ctx(Role.MEMBER),
    })
    assert store.list("session", _ctx())["artifacts"]

    disabled = MemoryService(
        OgMemConfig(tool_result_externalization_enabled=False),
        tool_result_store=store,
    )
    assert disabled._tool_result_externalizer is None
    mgr = disabled.get_session_manager()
    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("user", "agent"))
    monkeypatch.setattr(mgr, "delete_session_archives", lambda *args: 0)
    monkeypatch.setattr(mgr, "delete_persisted_session_state", lambda *args: 0)

    result = disabled.delete_session(_delete_params(_ctx()))

    assert result["ok"] is True
    assert result["steps"]["artifacts"]["deleted"] == 1
    assert store.list("session", _ctx())["artifacts"] == []
    assert disabled._tool_result_externalizer is None


def test_session_delete_rejects_untrusted_member_and_durable_owner_mismatch(
    tmp_path,
    monkeypatch,
):
    service = _service(tmp_path)
    mgr = service.get_session_manager()
    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("user", "agent"))
    with pytest.raises(PermissionError, match="trusted_agent_lifecycle_adapter_required"):
        service.delete_session(_delete_params(_ctx(Role.MEMBER)))

    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("other", "agent"))
    with pytest.raises(PermissionError, match="not_found_or_not_visible"):
        service.delete_session(_delete_params(_ctx()))
def test_partial_artifact_delete_keeps_tombstone_until_retry_converges(
    tmp_path,
    monkeypatch,
):
    service = _service(tmp_path)
    service.add_session_message({
        "sessionId": "session",
        "message": {"role": "toolResult", "content": "secret"},
        "_ctx": _ctx(Role.MEMBER),
    })
    mgr = service.get_session_manager()
    buf = mgr.get_or_create("session", ctx=_ctx())
    monkeypatch.setattr(mgr, "durable_session_owner", lambda *args: ("user", "agent"))
    monkeypatch.setattr(mgr, "delete_session_archives", lambda *args: 0)
    monkeypatch.setattr(mgr, "delete_persisted_session_state", lambda *args: 0)
    original_delete = service._tool_result_store.delete_session_artifacts
    attempts = 0

    def partial_then_recover(*args):
        nonlocal attempts
        attempts += 1
        deleted = original_delete(*args)
        if attempts == 1:
            raise OSError("partial artifact failure")
        return deleted

    monkeypatch.setattr(
        service._tool_result_store,
        "delete_session_artifacts",
        partial_then_recover,
    )

    first = service.delete_session(_delete_params(_ctx()))
    assert first["ok"] is False
    assert buf.cleaning is True
    assert buf.cleanup_retry_pending is True
    with pytest.raises(RuntimeError, match="session_operation_in_progress"):
        service.compose({"sessionId": "session", "messages": [], "_ctx": _ctx()})

    second = service.delete_session(_delete_params(_ctx()))
    assert second["ok"] is True
    assert not mgr.has_session("session", ctx=_ctx())
