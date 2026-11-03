from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.models import RequestContext
from providers import unified_config
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.topic_buffer import SlotContent


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="sess-compact",
        trace_id="trace-1",
    )


@pytest.fixture
def service() -> MemoryService:
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    # These lifecycle tests intentionally exercise legacy in-memory writers.
    # Durable first-turn reads are covered with real storage in recovery tests.
    service._get_context_fs = Mock(return_value=None)
    return service


def _seed_buffer(service: MemoryService, ctx: RequestContext, session_id: str = "sess-compact"):
    mgr = service.get_session_manager()
    mgr.add_message(session_id, "user", "first user fact", ctx)
    mgr.add_message(session_id, "assistant", "assistant filler", ctx)
    mgr.add_message(session_id, "user", "second user fact", ctx)
    return mgr.get_or_create(session_id)


def test_prepare_compaction_uses_incremental_non_assistant_messages(service: MemoryService, ctx: RequestContext):
    buf = _seed_buffer(service, ctx)
    buf.extraction_watermark = 1
    buf.extraction_summary = "existing-summary"
    buf.tool_usage_stats = {
        "search_web": {
            "call_count": 2,
            "success_count": 2,
            "fail_count": 0,
            "total_duration_ms": 14.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }
    }
    buf.messages[2].created_at = "2026-04-19T01:02:03+00:00"

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    }
    service._write_api = write_api

    result = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["buffer"] is buf
    write_api.commit_session.assert_called_once()
    call_messages = write_api.commit_session.call_args.kwargs["messages"]
    assert len(call_messages) == 1
    assert call_messages[0]["role"] == "user"
    assert call_messages[0]["content"] == "second user fact"
    assert "id" in call_messages[0]
    assert write_api.commit_session.call_args.kwargs["session_summary"] == "existing-summary"
    assert write_api.commit_session.call_args.kwargs["session_time"].isoformat() == "2026-04-19T01:02:03+00:00"
    assert "search_web" in write_api.commit_session.call_args.kwargs["tool_stats_text"]
    assert buf.extraction_watermark == 3
    assert "task-1" in buf.extraction_summary


def test_prepare_compaction_invalidates_cached_identity_when_profile_is_written(
    service: MemoryService,
    ctx: RequestContext,
):
    buf = _seed_buffer(service, ctx)
    topic = service.get_session_manager().get_topic_buffer("sess-compact")
    topic.set_cached_slot(
        "identity",
        SlotContent(
            content="stale identity",
            uris=["ctx://acct-test/users/user-test/memories/profile"],
            tokens=4,
        ),
    )

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [
            {
                "action": "merge",
                "target_uri": "ctx://acct-test/users/user-test/memories/profile",
            }
        ],
    }
    service._write_api = write_api

    service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert buf.extraction_watermark == 3
    assert topic.get_cached_slot("identity") is None


def test_prepare_compaction_raises_when_write_api_unavailable(service: MemoryService, ctx: RequestContext):
    _seed_buffer(service, ctx)
    service.get_write_api = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="write API unavailable"):
        service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})


def test_prepare_compaction_advances_watermark_when_only_assistant_messages(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "assistant", "only assistant", ctx)
    buf = mgr.get_or_create("sess-compact")
    existing_summary = buf.extraction_summary

    write_api = Mock()
    service._write_api = write_api

    result = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["messages"] == []
    write_api.commit_session.assert_not_called()
    assert buf.extraction_watermark == 1
    assert buf.extraction_summary == existing_summary


def test_after_turn_deduplicates_completed_client_request(service: MemoryService, ctx: RequestContext):
    service._cfg.session_state_bridge_enabled = True
    service._cfg.after_turn_threshold = 10_000
    manager = service.get_session_manager()
    manager.save_session_state = Mock(return_value=True)
    params = {
        "sessionId": "sess-idempotent",
        "clientRequestId": "pending-1",
        "messages": [{"role": "user", "content": "remember once"}],
        "_ctx": ctx,
    }

    first = service.after_turn(params)
    replay = service.after_turn(params)

    assert first["status"] == "accumulating"
    assert first["client_request_id"] == "pending-1"
    assert replay == {
        "ok": True,
        "status": "duplicate",
        "client_request_id": "pending-1",
    }
    assert len(manager.get_or_create("sess-idempotent").messages) == 1
    assert manager.save_session_state.call_count == 2


def test_after_turn_retries_until_idempotency_state_is_durable(
    service: MemoryService,
    ctx: RequestContext,
):
    service._cfg.session_state_bridge_enabled = True
    service._cfg.after_turn_threshold = 10_000
    manager = service.get_session_manager()
    manager.save_session_state = Mock(side_effect=[False, True])
    params = {
        "sessionId": "sess-idempotent-save",
        "clientRequestId": "pending-save-1",
        "messages": [{"role": "user", "content": "remember once"}],
        "_ctx": ctx,
    }

    first = service.after_turn(params)
    replay = service.after_turn(params)

    assert first == {
        "ok": False,
        "status": "failed",
        "error": "idempotency state unavailable",
        "retryable": True,
        "client_request_id": "pending-save-1",
    }
    assert replay == {
        "ok": True,
        "status": "duplicate",
        "client_request_id": "pending-save-1",
    }
    assert len(manager.get_or_create("sess-idempotent-save").messages) == 1


def test_after_turn_does_not_complete_replay_while_original_request_is_processing(
    service: MemoryService,
    ctx: RequestContext,
):
    service._cfg.session_state_bridge_enabled = True
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    manager = service.get_session_manager()
    manager.save_session_state = Mock(return_value=True)
    manager.add_message("sess-idempotent-overlap", "user", "remember once", ctx)
    buffer = manager.get_or_create("sess-idempotent-overlap")
    buffer.claim_incremental_extraction()

    replay = service.after_turn({
        "sessionId": "sess-idempotent-overlap",
        "clientRequestId": "pending-overlap-1",
        "messages": [{"role": "user", "content": "remember once"}],
        "forceExtract": True,
        "wait": True,
        "_ctx": ctx,
    })

    assert replay == {
        "ok": False,
        "status": "processing",
        "retryable": True,
        "pending_tokens": buffer.pending_tokens,
        "client_request_id": "pending-overlap-1",
    }
    assert buffer.request_completed("pending-overlap-1") is False
    manager.save_session_state.assert_not_called()


def test_after_turn_claims_client_request_before_processing(
    service: MemoryService,
    ctx: RequestContext,
):
    service._cfg.session_state_bridge_enabled = True
    service._cfg.after_turn_threshold = 10_000
    manager = service.get_session_manager()
    save_started = threading.Event()
    release_save = threading.Event()

    def save_session_state(_session_id, _ctx):
        save_started.set()
        assert release_save.wait(timeout=2)
        return True

    manager.save_session_state = Mock(side_effect=save_session_state)
    params = {
        "sessionId": "sess-idempotent-claim",
        "clientRequestId": "pending-claim-1",
        "messages": [{"role": "user", "content": "remember once"}],
        "_ctx": ctx,
    }
    first_result = []
    first = threading.Thread(target=lambda: first_result.append(service.after_turn(params)))
    first.start()
    assert save_started.wait(timeout=2)

    replay = service.after_turn(params)
    release_save.set()
    first.join(timeout=2)

    assert replay == {
        "ok": False,
        "status": "processing",
        "retryable": True,
        "client_request_id": "pending-claim-1",
    }
    assert first_result[0]["status"] == "accumulating"
    assert len(manager.get_or_create("sess-idempotent-claim").messages) == 1


def test_after_turn_uses_in_memory_idempotency_when_state_bridge_is_disabled(
    service: MemoryService,
    ctx: RequestContext,
):
    service._cfg.after_turn_threshold = 10_000
    manager = service.get_session_manager()
    manager.save_session_state = Mock(return_value=False)
    params = {
        "sessionId": "sess-idempotent-memory-only",
        "clientRequestId": "pending-memory-only-1",
        "messages": [{"role": "user", "content": "remember once"}],
        "_ctx": ctx,
    }

    first = service.after_turn(params)
    replay = service.after_turn(params)

    assert first["status"] == "accumulating"
    assert replay == {
        "ok": True,
        "status": "duplicate",
        "client_request_id": "pending-memory-only-1",
    }
    assert len(manager.get_or_create("sess-idempotent-memory-only").messages) == 1
    manager.save_session_state.assert_not_called()


def test_prepare_compaction_does_not_advance_watermark_or_issue_token_when_write_fails(service: MemoryService, ctx: RequestContext):
    buf = _seed_buffer(service, ctx)
    buf.extraction_watermark = 1
    buf.extraction_summary = "existing-summary"

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 1,
        "plans": [],
    }
    service._write_api = write_api

    with pytest.raises(RuntimeError, match="compaction extraction write failed"):
        service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert buf.extraction_watermark == 1
    assert buf.extraction_summary == "existing-summary"
    assert buf.compaction_prepare_token == ""
    assert buf.compaction_prepare_token_created_at == ""


def test_prepare_compaction_reuses_existing_prepare_token_and_archive_id(
    service: MemoryService,
    ctx: RequestContext,
):
    _seed_buffer(service, ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 0,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [],
    }
    service._write_api = write_api

    first = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    second = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert second["prepareToken"] == first["prepareToken"]
    assert second["archive_id"] == first["archive_id"]
    write_api.commit_session.assert_called_once()


def test_after_turn_uses_configured_threshold_before_env(
    ctx: RequestContext,
    monkeypatch,
):
    monkeypatch.setenv("OGMEM_AFTER_TURN_THRESHOLD", "1")
    service = MemoryService(config=OgMemConfig(after_turn_threshold=999999))
    service._get_context_fs = Mock(return_value=None)
    service.get_write_api = Mock(
        side_effect=AssertionError("after_turn should still be accumulating")
    )

    result = service.after_turn({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "messages": [
            {"role": "user", "content": "x" * 80},
            {"role": "assistant", "content": "ack"},
        ],
    })

    assert result["status"] == "accumulating"
    assert result["pending_tokens"] < 999999


def test_compact_returns_required_not_compacted_result_shape_when_commit_has_no_archive(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": False, "reason": "empty_buffer"})

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx})

    assert result == {
        "ok": True,
        "compacted": False,
        "reason": "empty_buffer",
        "result": {
            "summary": "",
            "tokensBefore": 5,
        },
    }


def test_compact_surfaces_failed_archive_commit_and_keeps_buffer(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["ok"] is False
    assert result["compacted"] is False
    assert result["status"] == "failed"
    assert "archive store unavailable" in result["error"]
    session = mgr.get_session("sess-compact", ctx)
    assert session["message_count"] == 1


def test_after_turn_no_extractable_archive_failure_rewinds_watermark(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service.drain_outbox_sync = Mock(return_value={"processed": 0, "succeeded": 0, "failed": 0})
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    result = service.after_turn({
        "sessionId": "sess-compact",
        "messages": [{"role": "assistant", "content": "assistant only content"}],
        "_ctx": ctx,
    })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive store unavailable"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_after_turn_background_archive_failure_rewinds_watermark(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "remember this failure path"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is True
    assert result["status"] == "processing"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_after_turn_background_invalidates_cached_identity_when_profile_is_written(
    service: MemoryService,
    ctx: RequestContext,
    monkeypatch,
):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [
            {
                "action": "merge",
                "target_uri": "ctx://acct-test/users/user-test/memories/profile",
            }
        ],
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": False})
    topic = mgr.get_topic_buffer("sess-compact")
    topic.set_cached_slot(
        "identity",
        SlotContent(
            content="stale identity",
            uris=["ctx://acct-test/users/user-test/memories/profile"],
            tokens=4,
        ),
    )

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "my profile changed"}],
            "_ctx": ctx,
        })

    assert result["ok"] is True
    assert result["status"] == "processing"
    assert topic.get_cached_slot("identity") is None


def test_after_turn_wait_true_runs_extraction_and_outbox_drain_synchronously(
    service: MemoryService,
    ctx: RequestContext,
):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    })
    service._async_drain = Mock()
    service.drain_outbox_sync = Mock(return_value={"processed": 1, "succeeded": 1, "failed": 0})
    mgr = service.get_session_manager()
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": True, "status": "completed"})

    result = service.after_turn({
        "sessionId": "sess-compact",
        "messages": [{"role": "user", "content": "remember synchronously"}],
        "_ctx": ctx,
        "wait": True,
    })

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["drain"] == {"processed": 1, "succeeded": 1, "failed": 0}
    service.drain_outbox_sync.assert_called_once_with(account_id="acct-test")
    service._async_drain.assert_not_called()


@pytest.mark.parametrize("extractor_error", [False, True])
def test_after_turn_background_write_failure_rewinds_watermark_and_skips_archive_commit(service: MemoryService, ctx: RequestContext, monkeypatch, extractor_error):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 1,
        "plans": [],
    })
    if extractor_error:
        # Exercise the real candidate pipeline boundary, not just a write-result
        # failure: model/parser exceptions must reach the watermark rewind path.
        from commit.candidate_pipeline import CandidatePipeline

        extractor = Mock()
        extractor.extract.side_effect = ValueError("No valid JSON found in LLM output")
        pipeline = CandidatePipeline([extractor])
        service._chunk_and_extract.side_effect = lambda **kwargs: pipeline.extract(
            kwargs["messages"], kwargs["ctx"],
        )
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": True,
        "uri": "ctx://acct-test/sessions/sess-compact/history/arc-ok",
        "archive_id": "arc-ok",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": True})

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "trigger write failure"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is True
    assert result["status"] == "processing"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1
    mgr.commit_snapshot.assert_not_called()


def test_after_turn_sync_fallback_archive_failure_rewinds_watermark_and_fails(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    class FailingThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("thread unavailable")

    with patch("server.memory_service.threading.Thread", FailingThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "remember sync fallback failure"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive store unavailable"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_compact_rejects_prepared_flag_without_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(side_effect=AssertionError("prepare_compaction should not run"))

    with pytest.raises(ValueError, match="prepare token required"):
        service.compact({"sessionId": "sess-compact", "_ctx": ctx, "prepared": True})


def test_compact_rejects_invalid_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(side_effect=AssertionError("prepare_compaction should not run"))

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": "bad-token",
        })


def test_compact_invalid_prepare_token_does_not_invalidate_valid_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": "bad-token",
        })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


def test_compact_accepts_one_time_verified_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"
    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": prepared["prepareToken"],
        })


def test_compact_reuses_prepare_archive_id_for_archive_commit(
    service: MemoryService,
    ctx: RequestContext,
):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)

    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    archive_id = prepared["archive_id"]

    mgr.commit = Mock(return_value={"ok": True, "archived": True, "archive_id": archive_id})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert write_api.commit_session.call_args.kwargs["archive_id"] == archive_id
    mgr.commit.assert_called_once_with("sess-compact", ctx, wait=True, archive_id=archive_id)


def test_compact_invalid_prepare_token_ttl_falls_back_to_default(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    service._cfg.compact_prepare_token_ttl = "bad"
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


def test_compact_rejects_expired_prepare_token(service: MemoryService, ctx: RequestContext):
    service._cfg.compact_prepare_token_ttl = 300
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    buf = mgr.get_or_create("sess-compact")
    buf.compaction_prepare_token_created_at = "2000-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": prepared["prepareToken"],
        })

    assert buf.compaction_prepare_token == ""
    assert buf.compaction_prepare_token_created_at == ""


def test_compact_trims_summary_and_does_not_fabricate_first_kept_entry_id(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True, "archive_id": "archive-123"})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx, "tokenBudget": 128_000})

    assert result["ok"] is True
    assert result["compacted"] is True
    assert result["result"]["summary"] == "90ABCDEFGHIJ"
    assert result["result"]["tokensAfter"] == 42
    assert "firstKeptEntryId" not in result["result"]


def test_compact_request_summary_max_chars_overrides_config(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "summaryMaxChars": 8,
    })

    assert result["result"]["summary"] == "CDEFGHIJ"


def test_compact_invalid_request_summary_max_chars_falls_back_to_config(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "summaryMaxChars": 0,
    })

    assert result["result"]["summary"] == "90ABCDEFGHIJ"


def test_compact_sync_short_term_index_mode_drains_outbox_best_effort(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 42,
    })
    service.drain_outbox_sync = Mock(side_effect=RuntimeError("index unavailable"))

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "shortTermIndexMode": "sync",
    })

    service.drain_outbox_sync.assert_called_once()
    assert result["ok"] is True
    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


@pytest.mark.parametrize("mode", ["async", "off"])
def test_compact_async_and_off_short_term_index_modes_skip_drain(service: MemoryService, ctx: RequestContext, mode: str):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 42,
    })
    service.drain_outbox_sync = Mock()

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "shortTermIndexMode": mode,
    })

    service.drain_outbox_sync.assert_not_called()
    assert result["ok"] is True
    assert result["compacted"] is True


def test_dispose_no_pending_messages_removes_session_when_context_build_fails(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    service.build_context = Mock(side_effect=RuntimeError("bad ctx"))

    result = service.dispose({"sessionId": "sess-compact"})

    assert result["disposed"] is True
    assert result["flushed"] is False
    assert result["reason"] == "no_pending_messages"
    assert not mgr.has_session("sess-compact")


def test_dispose_treats_cross_identity_session_as_not_found(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    other_ctx = RequestContext(
        account_id=ctx.account_id,
        user_id="other-user",
        agent_id=ctx.agent_id,
        session_id=ctx.session_id,
        trace_id="trace-other",
    )

    result = service.dispose({"sessionId": "sess-compact", "_ctx": other_ctx})

    assert result == {"disposed": True, "flushed": False, "reason": "session_not_found"}
    assert mgr.has_session("sess-compact", ctx=ctx)


def test_wait_until_idle_keeps_completed_session(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    service.drain_outbox_sync = Mock(return_value={"processed": 0, "succeeded": 0})

    result = service.wait_until_idle({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["idle"] is True
    assert result["reason"] == "idle"
    assert mgr.has_session("sess-compact")
    service.drain_outbox_sync.assert_called_once_with(account_id="acct-test")


def test_wait_until_idle_treats_cross_identity_session_as_not_found(
    service: MemoryService,
    ctx: RequestContext,
):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    other_ctx = RequestContext(
        account_id=ctx.account_id,
        user_id=ctx.user_id,
        agent_id="other-agent",
        session_id=ctx.session_id,
        trace_id="trace-other",
    )
    service.drain_outbox_sync = Mock(return_value={"processed": 0, "succeeded": 0})

    result = service.wait_until_idle({"sessionId": "sess-compact", "_ctx": other_ctx})

    assert result["idle"] is True
    assert result["reason"] == "session_not_found"
    assert mgr.has_session("sess-compact", ctx=ctx)
    service.drain_outbox_sync.assert_called_once_with(account_id="acct-test")


def test_wait_until_idle_can_poll_outbox_without_draining(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    service.drain_outbox_sync = Mock()
    outbox_store = Mock()
    outbox_store.count_unfinished.return_value = {
        "pending": 0,
        "processing": 0,
        "total": 0,
    }
    service._get_context_fs = Mock(return_value=Mock())
    service._get_outbox_store = Mock(return_value=outbox_store)

    result = service.wait_until_idle({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "drainOutbox": False,
        "waitOutbox": True,
    })

    assert result["idle"] is True
    assert result["reason"] == "idle"
    assert result["outbox"]["total"] == 0
    service.drain_outbox_sync.assert_not_called()
    outbox_store.count_unfinished.assert_called_once_with(account_id="acct-test")


def test_wait_until_idle_reports_outbox_in_progress_without_draining(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    service.drain_outbox_sync = Mock()
    outbox_store = Mock()
    outbox_store.count_unfinished.return_value = {
        "pending": 1,
        "processing": 1,
        "total": 2,
    }
    service._get_context_fs = Mock(return_value=Mock())
    service._get_outbox_store = Mock(return_value=outbox_store)

    result = service.wait_until_idle({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "timeoutSeconds": 0,
        "drainOutbox": False,
        "waitOutbox": True,
    })

    assert result["idle"] is False
    assert result["reason"] == "outbox_in_progress_timeout"
    assert result["outbox"]["total"] == 2
    service.drain_outbox_sync.assert_not_called()


def test_wait_until_idle_reports_in_progress_without_removing_session(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    buf = mgr.get_or_create("sess-compact", ctx=ctx)
    buf.begin_extraction()
    service.drain_outbox_sync = Mock()

    result = service.wait_until_idle({"sessionId": "sess-compact", "_ctx": ctx, "timeoutSeconds": 0.01})

    assert result["idle"] is False
    assert result["reason"] == "extraction_in_progress_timeout"
    assert mgr.has_session("sess-compact")
    service.drain_outbox_sync.assert_not_called()
    buf.end_extraction()


def test_wait_until_idle_reports_dispose_in_progress(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    buf = mgr.get_or_create("sess-compact", ctx=ctx)
    with buf.extraction_lock:
        buf.dispose_in_progress = True
    service.drain_outbox_sync = Mock()

    result = service.wait_until_idle({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "timeoutSeconds": 0.01,
    })

    assert result["idle"] is False
    assert result["reason"] == "dispose_in_progress_timeout"
    assert result["dispose_in_progress"] is True
    assert result["extraction_in_progress"] is False
    assert mgr.has_session("sess-compact")
    service.drain_outbox_sync.assert_not_called()


def test_wait_until_idle_drains_when_live_session_is_gone(service: MemoryService, ctx: RequestContext):
    service.drain_outbox_sync = Mock(return_value={"processed": 0, "succeeded": 0})

    result = service.wait_until_idle({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["idle"] is True
    assert result["reason"] == "session_not_found"
    service.drain_outbox_sync.assert_called_once_with(account_id="acct-test")


def test_ogmem_config_uses_default_summary_max_chars():
    cfg = OgMemConfig()

    assert cfg.summary_max_chars == 4000


def test_ogmem_config_load_reads_summary_max_chars_from_yaml_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OGMEM_SUMMARY_MAX_CHARS", "222")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"summary_max_chars": 123}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.summary_max_chars == 123

    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.summary_max_chars == 222


# ---------------------------------------------------------------------------
# ingest() buffer writing + after_turn() dedup
# ---------------------------------------------------------------------------

class TestIngestWritesToBuffer:
    """ingest() should write messages to the session buffer."""

    def test_ingest_writes_user_message_to_buffer(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "hello from ingest"}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 1

        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1
        assert buf.messages[0].role == "user"
        assert buf.messages[0].content == "hello from ingest"

    def test_ingest_empty_messages_returns_early(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({"sessionId": "sess-compact", "messages": [], "_ctx": ctx})

        assert result["ingested"] is True

    def test_ingest_skips_empty_content(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": ""}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 0

    def test_ingest_multiple_messages(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
            ],
            "_ctx": ctx,
        })

        assert result["count"] == 2
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2

    def test_ingest_batch_uses_same_logic(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest_batch({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "batch msg"}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 1
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1


class TestAfterTurnDedup:
    """after_turn() should skip messages already written by ingest()."""

    def test_after_turn_skips_ingest_written_user_message(self, service: MemoryService, ctx: RequestContext):
        # Simulate OperatorAgent flow: ingest → after_turn
        service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "hello from user"}],
            "_ctx": ctx,
        })

        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1  # ingest wrote 1 message

        # after_turn sends user + assistant (same user message as ingest)
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "hello from user"},
                {"role": "assistant", "content": "hi from assistant"},
            ],
            "_ctx": ctx,
        })

        assert result["ok"] is True
        # Should NOT duplicate user message — only assistant is new
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2
        assert buf.messages[0].role == "user"
        assert buf.messages[0].content == "hello from user"
        assert buf.messages[1].role == "assistant"
        assert buf.messages[1].content == "hi from assistant"

    def test_after_turn_no_dedup_when_ingest_not_called(self, service: MemoryService, ctx: RequestContext):
        # Direct after_turn without prior ingest — no dedup needed
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "_ctx": ctx,
        })

        assert result["ok"] is True
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2

    def test_ingest_then_compact_finds_buffer(self, service: MemoryService, ctx: RequestContext):
        """compact() should find messages written by ingest() — the fix for empty_buffer."""
        service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "important data"}],
            "_ctx": ctx,
        })

        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")

        # compact's commit should find buffer non-empty
        result = mgr.commit("sess-compact", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is True
        assert result.get("reason") != "empty_buffer"


def test_openclaw_plugin_manifest_exposes_compaction_config_schema():
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "plugins/openclaw" / "openclaw.plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    props = manifest["configSchema"]["properties"]

    assert props["compactTakeoverEnabled"]["type"] == "boolean"
    assert props["compactTakeoverEnabled"]["default"] is True
    assert props["summaryMaxChars"]["type"] == "integer"
    assert props["summaryMaxChars"]["default"] == 4000
    assert props["shortTermIndexMode"]["enum"] == ["sync", "async", "off"]
    assert props["shortTermIndexMode"]["default"] == "sync"


@pytest.mark.parametrize("counts", [
    {"pending": 0, "processing": 0, "total": 0, "failed": 1},
    {"total": 0, "supported": False},
    {"total": 0, "error": "database unavailable"},
])
def test_indexing_barrier_fails_closed(service, ctx, counts):
    service._count_outbox_unfinished = Mock(return_value=counts)
    result = service._wait_for_outbox_idle(ctx, timeout_seconds=0)
    assert result["idle"] is False
    assert result["outbox"] == counts
