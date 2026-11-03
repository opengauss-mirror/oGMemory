"""Tests for the MemoryService trace persistence boundary."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _context():
    return RequestContext(
        account_id="acct",
        user_id="alice",
        agent_id="agent",
        session_id="sess",
        trace_id="request",
    )


def _service(**overrides):
    values = {
        "storage_backend": "sql",
        "sql_connection_string": "postgresql://unused/test",
        "index_workers": 0,
    }
    values.update(overrides)
    return MemoryService(
        config=OgMemConfig(**values)
    )


def test_add_session_trace_passes_tenant_identity_and_event_time():
    service = _service()
    store = MagicMock()
    store.insert.return_value = {"ok": True, "event_index": 7}
    service._get_trace_event_store = lambda: store

    result = service.add_session_trace(
        {
            "_ctx": _context(),
            "eventId": "evt-1",
            "sessionId": "sess",
            "hookType": "PreToolUse",
            "eventTime": "2026-07-15T08:00:00Z",
            "event": {"event_type": "tool_start"},
        }
    )

    assert result == {"ok": True, "event_index": 7}
    call = store.insert.call_args.kwargs
    assert call["event_id"] == "evt-1"
    assert call["account_id"] == "acct"
    assert call["owner_space"] == "user:alice"
    assert call["agent_id"] == "agent"
    assert call["event_time"] == datetime(2026, 7, 15, 8, 0, tzinfo=UTC)


def test_add_session_trace_requires_configured_store():
    service = _service()
    service._get_trace_event_store = lambda: None

    result = service.add_session_trace(
        {
            "_ctx": _context(),
            "eventId": "evt-1",
            "sessionId": "sess",
            "event": {"event_type": "tool_start"},
        }
    )

    assert result == {
        "ok": False,
        "error": "trace store unavailable",
        "retryable": True,
    }


def test_finalize_session_trace_delegates_stop_boundary():
    service = _service()
    store = MagicMock()
    store.finalize_session.return_value = {"ok": True, "turn_id": "turn_sess_12"}
    service._get_trace_event_store = lambda: store

    result = service.finalize_session_trace(
        {"_ctx": _context(), "sessionId": "sess", "stopEventIndex": 12}
    )

    assert result["turn_id"] == "turn_sess_12"
    store.finalize_session.assert_called_once_with(
        account_id="acct",
        owner_space="user:alice",
        agent_id="agent",
        session_id="sess",
        stop_event_index=12,
    )
    store.enqueue_skill_candidate_from_trace.assert_not_called()


def test_finalize_session_trace_creates_skill_candidate_only_when_enabled(monkeypatch):
    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    service = _service(skill_sedimentation_enabled=True)
    service._get_context_fs = MagicMock()
    service._get_outbox_store = MagicMock()
    store = MagicMock()
    store.finalize_session.return_value = {
        "ok": True,
        "turn_id": "turn_sess_12",
        "uri": "ctx://acct/traces/sess/turns/turn_sess_12",
    }
    store.enqueue_skill_candidate_from_trace.return_value = {
        "ok": True,
        "candidate_id": "ctx://acct/skills/candidates/turn_sess_12",
    }
    service._get_trace_event_store = lambda: store

    result = service.finalize_session_trace(
        {"_ctx": _context(), "sessionId": "sess", "stopEventIndex": 12}
    )

    store.enqueue_skill_candidate_from_trace.assert_called_once_with(
        account_id="acct",
        agent_id="agent",
        session_id="sess",
        turn_id="turn_sess_12",
        trace_uri="ctx://acct/traces/sess/turns/turn_sess_12",
    )
    assert result["skill_candidate_id"] == "ctx://acct/skills/candidates/turn_sess_12"


def test_add_session_trace_hides_database_error_details():
    service = _service()
    store = MagicMock()
    store.insert.side_effect = RuntimeError("password authentication failed for dbhost")
    service._get_trace_event_store = lambda: store

    result = service.add_session_trace(
        {
            "_ctx": _context(),
            "eventId": "evt-1",
            "sessionId": "sess",
            "event": {"event_type": "tool_start"},
        }
    )

    assert result["retryable"] is True
    assert result["error_code"] == "trace_persistence_unavailable"
    assert "password" not in result["error"]


def test_add_session_trace_delegates_to_agfs_store():
    service = MemoryService(config=OgMemConfig(storage_backend="agfs"))
    store = MagicMock()
    store.insert.return_value = {"ok": True, "event_index": 1}
    service._get_trace_event_store = lambda: store

    result = service.add_session_trace(
        {
            "_ctx": _context(),
            "eventId": "evt-1",
            "sessionId": "sess",
            "event": {"event_type": "tool_start"},
        }
    )

    assert result == {"ok": True, "event_index": 1}
    store.insert.assert_called_once()
