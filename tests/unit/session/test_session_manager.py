"""Unit tests for SessionManager."""

import json
import threading
from unittest.mock import Mock, patch

import pytest

from core.models import ContextNode, RequestContext, Role
from session.models import ArchiveEntry, SessionMessage, SessionMeta
from session.session_manager import SessionBuffer, SessionManager, SessionOperationInProgress
from session.session_state import TaskState
from session.topic_buffer import SlotContent


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct-test",
        user_id="u-test",
        agent_id="agent-test",
        session_id="sess-1",
        trace_id="trace-1",
    )


@pytest.fixture
def mgr():
    return SessionManager()


@pytest.mark.parametrize("entry", ["commit", "snapshot_and_clear"])
@pytest.mark.parametrize("batch", [{}, {"unknown_format": True}, {"lifecycle": {}}])
def test_source_clear_preserves_unrecognized_pending_batch(ctx, entry, batch):
    manager = SessionManager()
    buffer = manager.get_or_create(ctx.session_id, ctx)
    buffer.add("user", "Source awaiting recovery")
    buffer.runtime_extensions["memory_extraction_recovery"] = {"version": 1, "batch": batch}
    before = buffer.capture_rollback_state()

    with pytest.raises(SessionOperationInProgress) as conflict:
        if entry == "commit":
            manager.commit(ctx.session_id, ctx, wait=True)
        else:
            buffer.snapshot_and_clear()

    assert conflict.value.conflict_type == (
        "lifecycle_source_reserved" if "lifecycle" in batch else "extraction_source_reserved")
    assert buffer.capture_rollback_state() == before
    assert buffer.runtime_extensions["memory_extraction_recovery"]["batch"] == batch
    assert not manager._tasks and not buffer.commit_in_progress


@pytest.fixture
def mgr_with_deps():
    mock_llm = Mock()
    mock_write_api = Mock()
    mock_write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "writes_completed": 1,
        "task_id": "t-1",
        "status": "completed",
    }
    return SessionManager(
        get_llm=lambda: mock_llm,
        get_write_api=lambda: mock_write_api,
        get_agfs=lambda: None,  # No AGFS in unit tests
    )


def test_list_session_working_set_orders_by_last_accessed_at(ctx):
    mgr = SessionManager()
    first = mgr.get_or_create("older", ctx)
    second = mgr.get_or_create("newer", ctx)
    first.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    second.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"

    rows = mgr.list_session_working_set()

    assert [row["session_id"] for row in rows] == ["newer", "older"]
    assert rows[0]["last_accessed_at"] == "2026-05-20T02:00:00+00:00"
    assert rows[0]["message_count"] == 0
    assert rows[0]["pending_tokens"] == 0


def test_evict_idle_sessions_uses_last_accessed_at(ctx):
    mgr = SessionManager()
    active = mgr.get_or_create("active", ctx)
    idle = mgr.get_or_create("idle", ctx)
    active.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"

    evicted = mgr.evict_idle_sessions(
        max_idle_seconds=1800,
        now_iso="2026-05-20T02:00:01+00:00",
        ctx=ctx,
    )

    assert evicted == ["idle"]
    assert mgr.has_session("active") is True
    assert mgr.has_session("idle") is False


def test_evict_idle_sessions_skips_sessions_with_pending_messages(ctx):
    mgr = SessionManager()
    idle = mgr.get_or_create("idle", ctx)
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    mgr.add_message("idle", "user", "pending message", ctx)

    evicted = mgr.evict_idle_sessions(
        max_idle_seconds=1800,
        now_iso="2026-05-20T02:00:01+00:00",
        ctx=ctx,
    )

    assert evicted == []
    assert mgr.has_session("idle") is True


def test_evict_idle_sessions_saves_state_before_removing(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    idle = mgr.get_or_create("idle", ctx)
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    mgr.get_session_state().update_task_state("idle", TaskState(objective="Persist before eviction"))

    evicted = mgr.evict_idle_sessions(
        max_idle_seconds=1800,
        now_iso="2026-05-20T02:00:01+00:00",
        ctx=ctx,
    )

    assert evicted == ["idle"]
    fs.write_node.assert_called_once()
    assert mgr.has_session("idle") is False


def test_evict_idle_session_reloads_durable_runtime_extensions(ctx):
    initial = ContextNode(
        uri="ctx://acct-test/sessions/idle/state.json",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:idle",
        abstract="Session state",
        overview="Session state",
        content=json.dumps({
            "runtime_extensions": {"turn_ledger": {"sequence": 7}},
        }),
        metadata={},
    )
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = initial
    mgr = SessionManager(get_context_fs=lambda: fs)
    idle = mgr.get_or_create("idle", ctx)
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    assert idle.runtime_extensions["turn_ledger"]["sequence"] == 7

    evicted = mgr.evict_idle_sessions(
        max_idle_seconds=1800,
        now_iso="2026-05-20T02:00:01+00:00",
        ctx=ctx,
    )
    saved = fs.write_node.call_args.args[0]
    fs.read_node.return_value = saved
    restored = mgr.get_or_create("idle", ctx)

    assert evicted == ["idle"]
    assert restored.runtime_extensions["turn_ledger"]["sequence"] == 7
    assert fs.read_node.call_count == 2


def test_evict_idle_sessions_saves_each_session_state_under_own_tenant():
    fs = Mock()
    fs.exists.return_value = False
    mgr = SessionManager(get_context_fs=lambda: fs)
    tenant_a_ctx = RequestContext(
        account_id="acct-a",
        user_id="user-a",
        agent_id="agent-a",
        session_id="tenant-a-session",
        trace_id="trace-a",
    )
    tenant_b_ctx = RequestContext(
        account_id="acct-b",
        user_id="user-b",
        agent_id="agent-b",
        session_id="tenant-b-session",
        trace_id="trace-b",
    )
    admin_ctx = RequestContext(
        account_id="admin-acct",
        user_id="admin-user",
        agent_id="admin-agent",
        session_id="admin-session",
        trace_id="admin-trace",
        role=Role.ADMIN,
    )
    tenant_a = mgr.get_or_create("tenant-a-session", tenant_a_ctx)
    tenant_b = mgr.get_or_create("tenant-b-session", tenant_b_ctx)
    tenant_a.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    tenant_b.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"

    evicted = mgr.evict_idle_sessions(
        max_idle_seconds=1800,
        now_iso="2026-05-20T02:00:01+00:00",
        ctx=admin_ctx,
    )

    assert evicted == ["tenant-a-session", "tenant-b-session"]
    write_calls = fs.write_node.call_args_list
    assert [call.args[0].uri for call in write_calls] == [
        "ctx://acct-a/sessions/tenant-a-session/state.json",
        "ctx://acct-b/sessions/tenant-b-session/state.json",
    ]
    assert [call.args[1].account_id for call in write_calls] == ["acct-a", "acct-b"]
    assert [call.args[1].user_id for call in write_calls] == ["user-a", "user-b"]
    assert [call.args[1].agent_id for call in write_calls] == ["agent-a", "agent-b"]
    assert [call.args[1].session_id for call in write_calls] == [
        "tenant-a-session",
        "tenant-b-session",
    ]
    assert [call.args[1].trace_id for call in write_calls] == [
        "admin-trace",
        "admin-trace",
    ]
    assert [call.args[1].visible_owner_spaces for call in write_calls] == [(), ()]


# ---------------------------------------------------------------------------
# SessionBuffer tests
# ---------------------------------------------------------------------------


class TestSessionBuffer:
    def test_add_message(self):
        buf = SessionBuffer(session_id="s1")
        msg = buf.add("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert len(buf.messages) == 1
        assert buf.meta.message_count == 1

    def test_pending_tokens(self):
        buf = SessionBuffer(session_id="s1")
        buf.add("user", "a" * 100)  # ~25 tokens
        buf.add("assistant", "b" * 200)  # ~50 tokens
        assert buf.pending_tokens == 75

    def test_tool_usage_stats_setter_merges_existing_usage(self):
        buf = SessionBuffer(session_id="s1")
        buf.usage_stats.record_tool_call(
            tool_name="new_tool",
            status="success",
            prompt_tokens=7,
            completion_tokens=3,
        )

        buf.tool_usage_stats = {
            "legacy_tool": {
                "call_count": 1,
                "success_count": 1,
                "fail_count": 0,
                "total_prompt_tokens": 2,
                "total_completion_tokens": 1,
            }
        }

        stats = buf.tool_usage_stats
        assert stats["new_tool"]["call_count"] == 1
        assert stats["new_tool"]["total_tokens"] == 10
        assert stats["legacy_tool"]["call_count"] == 1
        assert stats["legacy_tool"]["total_tokens"] == 3

    def test_snapshot_and_clear(self):
        buf = SessionBuffer(session_id="s1")
        buf.add("user", "hello")
        buf.add("assistant", "world")
        snap = buf.snapshot_and_clear()
        assert len(snap) == 2
        assert len(buf.messages) == 0
        assert buf.pending_tokens == 0
        assert buf.meta.message_count == 0

    def test_remove_messages_by_id_adjusts_watermark(self):
        buf = SessionBuffer(session_id="s1")
        first = buf.add("user", "first")
        second = buf.add("user", "second")
        third = buf.add("user", "third")
        buf.extraction_watermark = 3

        removed = buf.remove_messages_by_id({second.id})

        assert removed == 1
        assert [m.id for m in buf.messages] == [first.id, third.id]
        assert buf.extraction_watermark == 2

    def test_remove_messages_by_id_recomputes_turn_count(self):
        buf = SessionBuffer(session_id="s1")
        archived = [buf.add("user", f"archived {idx}") for idx in range(10)]

        assert buf.should_compress()

        removed = buf.remove_messages_by_id({msg.id for msg in archived})

        assert removed == 10
        assert len(buf.messages) == 0
        assert buf.turn_count == 0
        assert not buf.should_compress()

        buf.add("user", "new message")

        assert buf.turn_count == 1
        assert not buf.should_compress()

    def test_remove_messages_by_id_clamps_session_state_sync_turn_count(self):
        buf = SessionBuffer(session_id="s1")
        archived = [buf.add("user", f"archived {idx}") for idx in range(10)]
        buf.window_state.session_state_sync_turn_count = 10

        removed = buf.remove_messages_by_id({msg.id for msg in archived})

        assert removed == 10
        assert buf.turn_count == 0
        assert buf.window_state.session_state_sync_turn_count == 0


# ---------------------------------------------------------------------------
# SessionManager tests
# ---------------------------------------------------------------------------


class TestSessionManager:
    def test_get_or_create(self, mgr, ctx):
        buf = mgr.get_or_create("s1")
        assert isinstance(buf, SessionBuffer)
        assert buf.session_id == "s1"

    def test_get_or_create_idempotent(self, mgr, ctx):
        buf1 = mgr.get_or_create("s1")
        buf1.add("user", "hello")
        buf2 = mgr.get_or_create("s1")
        assert len(buf2.messages) == 1  # Same buffer

    def test_get_or_create_rejects_cross_identity_session_reuse(self, mgr, ctx):
        mgr.get_or_create("s1", ctx)
        other_ctx = RequestContext(
            account_id=ctx.account_id,
            user_id="other-user",
            agent_id=ctx.agent_id,
            session_id="s1",
            trace_id="trace-other",
        )

        with pytest.raises(PermissionError, match="not_found_or_not_visible"):
            mgr.get_or_create("s1", other_ctx)

    def test_has_and_remove_session_can_scope_by_context(self, mgr, ctx):
        mgr.get_or_create("s1", ctx)
        other_ctx = RequestContext(
            account_id=ctx.account_id,
            user_id=ctx.user_id,
            agent_id="other-agent",
            session_id="s1",
            trace_id="trace-other",
        )

        assert mgr.has_session("s1", ctx=other_ctx) is False
        mgr.remove_session("s1", ctx=other_ctx)
        assert mgr.has_session("s1", ctx=ctx) is True

    def test_add_message(self, mgr, ctx):
        result = mgr.add_message("s1", "user", "hello", ctx)
        assert result["ok"] is True
        assert "message_id" in result
        assert result["pending_tokens"] > 0

    def test_add_message_increments_tokens(self, mgr, ctx):
        mgr.add_message("s1", "user", "a" * 100, ctx)
        r1 = mgr.get_session("s1", ctx)
        mgr.add_message("s1", "assistant", "b" * 200, ctx)
        r2 = mgr.get_session("s1", ctx)
        assert r2["pending_tokens"] > r1["pending_tokens"]

    def test_get_session(self, mgr, ctx):
        mgr.add_message("s1", "user", "hello", ctx)
        result = mgr.get_session("s1", ctx)
        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["message_count"] == 1
        assert result["pending_tokens"] > 0

    def test_commit_empty_buffer(self, mgr, ctx):
        result = mgr.commit("s1", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is False
        assert result["reason"] == "empty_buffer"

    def test_commit_sync(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello world", ctx)
        mgr.add_message("s1", "assistant", "hi there", ctx)
        result = mgr.commit("s1", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is True
        assert "archive_id" in result
        assert result["status"] == "completed"

        # Buffer should be cleared
        session = mgr.get_session("s1", ctx)
        assert session["pending_tokens"] == 0

    def test_commit_async(self, mgr_with_deps, ctx):
        import time
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello world", ctx)
        result = mgr.commit("s1", ctx, wait=False)
        assert result["ok"] is True
        assert result["status"] == "processing"
        assert "task_id" in result

        # Wait for background thread
        time.sleep(0.5)

        # Check task completed
        task = mgr.get_task(result["task_id"])
        assert task is not None
        assert task["status"] == "completed"

    def test_commit_in_progress_rejects_second(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello", ctx)

        # Manually set commit_in_progress
        buf = mgr.get_or_create("s1")
        buf.commit_in_progress = True

        result = mgr.commit("s1", ctx, wait=True)
        assert result["reason"] == "commit_in_progress"

    def test_commit_updates_meta(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello", ctx)
        mgr.commit("s1", ctx, wait=True)

        session = mgr.get_session("s1", ctx)
        assert session["commit_count"] == 1
        assert session["last_commit_at"] != ""

    def test_commit_snapshot_does_not_clear_live_buffer(self, mgr_with_deps, ctx):
        from session.session_manager import generate_archive_id
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "first", ctx)
        mgr.add_message("s1", "user", "second", ctx)
        buf = mgr.get_or_create("s1")
        snapshot = [buf.messages[0]]

        result = mgr.commit_snapshot("s1", snapshot, ctx, wait=True, archive_id=generate_archive_id())

        assert result["ok"] is True
        assert result["archived"] is True
        assert [m.content for m in buf.messages] == ["first", "second"]

    def test_get_context_empty_session(self, mgr, ctx):
        result = mgr.get_context("s1", 128_000, ctx)
        assert result["ok"] is True
        assert result["pending_tokens"] == 0
        assert result["archive_count"] == 0
        assert result["active_message_count"] == 0

    def test_get_context_with_active_messages(self, mgr, ctx):
        mgr.add_message("s1", "user", "a" * 100, ctx)
        result = mgr.get_context("s1", 128_000, ctx)
        assert result["active_message_count"] == 1
        assert result["pending_tokens"] > 0

    def test_remove_session(self, mgr, ctx):
        mgr.add_message("s1", "user", "hello", ctx)
        mgr.remove_session("s1")
        # get_or_create should create fresh
        buf = mgr.get_or_create("s1")
        assert len(buf.messages) == 0

    def test_get_task_unknown(self, mgr):
        assert mgr.get_task("nonexistent") is None

    def test_prepare_token_sets_created_at(self, mgr):
        token = mgr.issue_compaction_prepare_token("s1")
        buf = mgr.get_or_create("s1")

        assert token
        assert buf.compaction_prepare_token == token
        assert buf.compaction_prepare_token_created_at

    def test_expired_prepare_token_is_rejected_and_cleared(self, mgr):
        token = mgr.issue_compaction_prepare_token("s1")
        buf = mgr.get_or_create("s1")
        buf.compaction_prepare_token_created_at = "2000-01-01T00:00:00+00:00"

        assert mgr.consume_compaction_prepare_token("s1", token, ttl_seconds=300) is False
        assert buf.compaction_prepare_token == ""
        assert buf.compaction_prepare_token_created_at == ""

    def test_valid_prepare_token_consumes_normally(self, mgr):
        token = mgr.issue_compaction_prepare_token("s1")

        assert mgr.consume_compaction_prepare_token("s1", token, ttl_seconds=300) is True
        buf = mgr.get_or_create("s1")
        assert buf.compaction_prepare_token == ""
        assert buf.compaction_prepare_token_created_at == ""

    def test_commit_fails_when_required_archive_store_unavailable(self, ctx):
        mock_llm = Mock()
        mock_write_api = Mock()
        mgr = SessionManager(
            get_llm=lambda: mock_llm,
            get_write_api=lambda: mock_write_api,
            get_archive_store=lambda: None,
            archive_store_required=True,
        )

        mgr.add_message("s1", "user", "hello world", ctx)
        result = mgr.commit("s1", ctx, wait=True)

        assert result["ok"] is True
        assert result["archived"] is False
        assert result["status"] == "failed"
        assert "archive store unavailable" in result["error"]
        mock_write_api.commit_session.assert_not_called()

    def test_get_latest_archive_context_prefers_archive_store(self, ctx):
        sql_store = Mock()
        sql_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-old",
                session_id="s1",
                overview="old overview",
                abstract="old abstract",
                messages=[],
                created_at="2026-01-01T00:00:00+00:00",
            ),
            ArchiveEntry(
                archive_id="a-new",
                session_id="s1",
                overview="new overview",
                abstract="new abstract",
                messages=[],
                created_at="2026-01-02T00:00:00+00:00",
            ),
        ]
        mgr = SessionManager(
            get_archive_store=lambda: sql_store,
            get_agfs=lambda: Mock(),
        )

        with patch("session.archive_store.SessionArchiveStore") as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "new overview"
        assert abstract == "new abstract"
        sql_store.list_archives.assert_called_once_with("s1", ctx)
        agfs_store_cls.assert_not_called()

    def test_get_latest_archive_context_falls_back_to_agfs_when_store_missing(self, ctx):
        agfs_store = Mock()
        agfs_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-only",
                session_id="s1",
                overview="agfs overview",
                abstract="agfs abstract",
                messages=[],
                created_at="2026-01-03T00:00:00+00:00",
            ),
        ]
        agfs_fs = Mock()
        mgr = SessionManager(
            get_archive_store=lambda: None,
            get_agfs=lambda: agfs_fs,
        )

        with patch("session.archive_store.SessionArchiveStore", return_value=agfs_store) as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "agfs overview"
        assert abstract == "agfs abstract"
        agfs_store_cls.assert_called_once_with(fs=agfs_fs)
        agfs_store.list_archives.assert_called_once_with("s1", ctx)

    def test_get_latest_archive_context_falls_back_to_agfs_when_store_errors(self, ctx):
        broken_store = Mock()
        broken_store.list_archives.side_effect = RuntimeError("db unavailable")
        agfs_store = Mock()
        agfs_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-fallback",
                session_id="s1",
                overview="fallback overview",
                abstract="fallback abstract",
                messages=[],
                created_at="2026-01-04T00:00:00+00:00",
            ),
        ]
        agfs_fs = Mock()
        mgr = SessionManager(
            get_archive_store=lambda: broken_store,
            get_agfs=lambda: agfs_fs,
        )

        with patch("session.archive_store.SessionArchiveStore", return_value=agfs_store) as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "fallback overview"
        assert abstract == "fallback abstract"
        broken_store.list_archives.assert_called_once_with("s1", ctx)
        agfs_store_cls.assert_called_once_with(fs=agfs_fs)


def test_commit_saves_session_state(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "hello", ctx)
    mgr._write_archive = Mock(return_value={"success": True, "uri": "memory://s1/a1", "archive_id": "a1"})
    mgr._compress = Mock(return_value=("overview", "abstract"))
    mgr.get_session_state().update_task_state("s1", TaskState(objective="Persist on commit"))

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is True
    fs.write_node.assert_called_once()


def test_commit_does_not_save_session_state_when_archive_fails(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "hello", ctx)
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "a1",
        "error": "archive write failed",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))
    mgr.get_session_state().update_task_state("s1", TaskState(objective="Do not persist on failure"))

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is False
    assert result["status"] == "failed"
    fs.write_node.assert_not_called()


def test_commit_failure_restores_full_buffer_state(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "please preserve this task", ctx)
    mgr.add_message("s1", "assistant", "assistant response", ctx)
    buf = mgr.get_or_create("s1")
    buf.extraction_watermark = 1
    buf.extraction_summary = "existing extraction summary"
    buf.window_state.active_task = "please preserve this task"
    buf.window_state.confirmed_constraints.append("keep constraint")
    prepare_token = mgr.issue_compaction_prepare_token("s1")
    prepare_created_at = buf.compaction_prepare_token_created_at
    original_ids = [msg.id for msg in buf.messages]
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "a1",
        "error": "archive write failed",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive write failed"
    assert [msg.id for msg in buf.messages] == original_ids
    assert buf.turn_count == 1
    assert buf.meta.message_count == 2
    assert buf.window_state.active_task == "please preserve this task"
    assert buf.window_state.confirmed_constraints == ["keep constraint"]
    assert buf.extraction_watermark == 1
    assert buf.extraction_summary == "existing extraction summary"
    assert buf.compaction_prepare_token == prepare_token
    assert buf.compaction_prepare_message_count == 2
    assert buf.compaction_prepare_watermark == 1
    assert buf.compaction_prepare_token_created_at == prepare_created_at
    fs.write_node.assert_not_called()


def test_commit_snapshot_does_not_save_session_state_when_archive_fails(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    snapshot = [SessionMessage(id="m1", role="user", content="hello")]
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "a1",
        "error": "archive write failed",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))
    mgr.get_session_state().update_task_state("s1", TaskState(objective="Do not persist on failure"))

    result = mgr.commit_snapshot("s1", snapshot, ctx, wait=True)

    assert result["archived"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive write failed"
    fs.write_node.assert_not_called()


def test_archive_merge_runs_when_threshold_exceeded(ctx):
    store = Mock()
    entries = [
        ArchiveEntry(
            archive_id=f"arc{i}",
            session_id="s1",
            overview=f"overview {i}",
            abstract=f"abstract {i}",
            messages=[{"role": "user", "content": f"msg {i}"}],
            created_at=f"2026-05-{i + 1:02d}T00:00:00+00:00",
            metadata={"archive_id": f"arc{i}"},
        )
        for i in range(4)
    ]
    store.list_archives.return_value = entries
    store.write_archive.return_value = Mock(success=True, archive_id="merged_001", uri="uri")
    store.mark_archive_merged.return_value = True
    mgr = SessionManager(
        get_archive_store=lambda: store,
        archive_merge_threshold=3,
        archive_max_count=3,
    )

    mgr._maybe_merge_archives("s1", ctx, wait=True)

    store.write_archive.assert_called_once()
    merged_messages = store.write_archive.call_args.kwargs["messages"]
    assert any(msg["content"] == "msg 1" for msg in merged_messages)
    assert store.write_archive.call_args.kwargs["metadata"]["source_archive_ids"] == [
        "arc0",
        "arc1",
    ]
    assert store.mark_archive_merged.call_count == 2


def test_archive_merge_invalidates_cached_archive_history(ctx):
    store = Mock()
    entries = [
        ArchiveEntry(
            archive_id=f"arc{i}",
            session_id="s1",
            overview=f"overview {i}",
            abstract=f"abstract {i}",
            messages=[{"role": "user", "content": f"msg {i}"}],
            created_at=f"2026-05-{i + 1:02d}T00:00:00+00:00",
            metadata={"archive_id": f"arc{i}"},
        )
        for i in range(3)
    ]
    store.list_archives.return_value = entries
    store.read_archive.side_effect = entries
    store.write_archive.return_value = Mock(success=True, archive_id="merged_001", uri="uri")
    store.mark_archive_merged.return_value = True
    mgr = SessionManager(
        get_archive_store=lambda: store,
        archive_merge_threshold=3,
        archive_max_count=2,
    )
    topic = mgr.get_topic_buffer("s1")
    topic.set_cached_slot("archive_history", SlotContent(content="stale archive"))

    assert mgr._merge_archives_once("s1", ctx) is True

    assert topic.get_cached_slot("archive_history") is None


def test_commit_invalidates_cached_archive_history(ctx):
    mgr = SessionManager()
    topic = mgr.get_topic_buffer("s1")
    topic.set_cached_slot("archive_history", SlotContent(content="stale archive"))
    mgr.add_message("s1", "user", "archive me", ctx)
    mgr._write_archive = Mock(
        return_value={"success": True, "uri": "memory://s1/a1", "archive_id": "a1"}
    )
    mgr._compress = Mock(return_value=("overview", "abstract"))

    mgr.commit("s1", ctx, wait=True)

    assert topic.get_cached_slot("archive_history") is None


def test_archive_merge_not_triggered_below_threshold(ctx):
    store = Mock()
    store.list_archives.return_value = [
        ArchiveEntry(
            archive_id="arc0",
            session_id="s1",
            overview="overview",
            abstract="abstract",
            messages=[{"role": "user", "content": "msg 0"}],
            created_at="2026-05-01T00:00:00+00:00",
        )
    ]
    mgr = SessionManager(
        get_archive_store=lambda: store,
        archive_merge_threshold=2,
    )

    mgr._maybe_merge_archives("s1", ctx, wait=True)

    store.write_archive.assert_not_called()


def test_archive_merge_requires_at_least_two_archives(ctx):
    store = Mock()
    store.list_archives.return_value = [
        ArchiveEntry(
            archive_id="arc0",
            session_id="s1",
            overview="overview",
            abstract="abstract",
            messages=[{"role": "user", "content": "msg 0"}],
            created_at="2026-05-01T00:00:00+00:00",
        )
    ]
    mgr = SessionManager(
        get_archive_store=lambda: store,
        archive_merge_threshold=1,
    )

    mgr._maybe_merge_archives("s1", ctx, wait=True)

    store.write_archive.assert_not_called()
    store.mark_archive_merged.assert_not_called()


def test_archive_merge_rolls_back_new_archive_when_marking_fails(ctx):
    store = Mock()
    entries = [
        ArchiveEntry(
            archive_id=f"arc{i}",
            session_id="s1",
            overview=f"overview {i}",
            abstract=f"abstract {i}",
            messages=[{"role": "user", "content": f"msg {i}"}],
            created_at=f"2026-05-{i + 1:02d}T00:00:00+00:00",
            metadata={"archive_id": f"arc{i}"},
        )
        for i in range(3)
    ]
    store.list_archives.return_value = entries
    store.read_archive.side_effect = entries
    store.write_archive.return_value = Mock(success=True, archive_id="merged_001", uri="uri")
    store.mark_archive_merged.side_effect = [True, False]
    store.unmark_archive_merged.return_value = True
    store.delete_archive.return_value = True
    mgr = SessionManager(
        get_archive_store=lambda: store,
        archive_merge_threshold=3,
        archive_max_count=2,
    )

    assert mgr._merge_archives_once("s1", ctx) is False
    store.unmark_archive_merged.assert_called_once_with("s1", "arc0", ctx, "merged_001")
    store.delete_archive.assert_called_once_with("s1", "merged_001", ctx)


def test_consume_prepare_token_clears_any_expired_token(ctx):
    mgr = SessionManager()
    mgr.issue_compaction_prepare_token("s1")
    buf = mgr.get_or_create("s1")
    buf.compaction_prepare_token_created_at = "2000-01-01T00:00:00+00:00"

    assert mgr.consume_compaction_prepare_token("s1", "wrong-token", ttl_seconds=300) is False
    assert buf.compaction_prepare_token == ""
    assert buf.compaction_prepare_token_created_at == ""


def test_commit_records_compression_quality_metrics(ctx):
    mgr = SessionManager(compression_quality_enabled=True)
    mgr.add_message("s1", "user", "Use PostgreSQL and AGFS for Project Falcon.", ctx)
    mgr._write_archive = Mock(
        return_value={"success": True, "uri": "memory://s1/a1", "archive_id": "a1"}
    )
    mgr._compress = Mock(
        return_value=("Project Falcon uses PostgreSQL and AGFS.", "Project Falcon storage")
    )

    result = mgr.commit("s1", ctx, wait=True)
    task = mgr.get_task(result["task_id"])

    assert task["compression_quality"]["entity_retention_ratio"] == 1.0
    assert task["compression_quality"]["information_retention_ratio"] == 1.0
    assert "missing_entities" not in task["compression_quality"]


def test_commit_persists_compression_quality_metadata_when_enabled(ctx):
    store = Mock()
    store.write_archive.return_value = Mock(
        success=True,
        uri="ctx://acct-test/sessions/s1/history/arc1",
        archive_id="arc1",
        error="",
    )
    mgr = SessionManager(
        get_llm=lambda: None,
        get_archive_store=lambda: store,
        compression_quality_enabled=True,
        compression_quality_persist_metadata=True,
    )
    mgr.add_message("s1", "user", "Remember ProjectAtlas quality metrics", ctx)

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is True
    kwargs = store.write_archive.call_args.kwargs
    assert "compression_quality" in kwargs["metadata"]
    assert kwargs["metadata"]["compression_quality"]["entity_retention_ratio"] >= 0.0


def test_commit_skips_compression_quality_when_disabled(ctx):
    store = Mock()
    store.write_archive.return_value = Mock(
        success=True,
        uri="ctx://acct-test/sessions/s1/history/arc1",
        archive_id="arc1",
        error="",
    )
    mgr = SessionManager(
        get_llm=lambda: None,
        get_archive_store=lambda: store,
        compression_quality_enabled=False,
        compression_quality_persist_metadata=True,
    )
    mgr.add_message("s1", "user", "Remember ProjectAtlas quality metrics", ctx)

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is True
    kwargs = store.write_archive.call_args.kwargs
    assert kwargs["metadata"] == {}
    task = mgr.get_task(result["task_id"])
    assert "compression_quality" not in task


def test_load_session_state_caches_successful_load(ctx):
    payload = {"task_state": {"objective": "Cached state"}}
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    mgr = SessionManager(get_context_fs=lambda: fs)

    assert mgr.load_session_state("s1", ctx) is True
    assert mgr.load_session_state("s1", ctx) is True
    assert fs.exists.call_count == 1
    assert fs.read_node.call_count == 1


def test_load_session_state_negative_caches_missing_state(ctx, monkeypatch):
    payload = {
        "task_state": {
            "objective": "Recovered later",
        }
    }
    fs = Mock()
    fs.exists.side_effect = [False, False, True]
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state.json",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    mgr = SessionManager(get_context_fs=lambda: fs)
    now = [100.0]
    monkeypatch.setattr("session.session_manager.monotonic", lambda: now[0])

    assert mgr.load_session_state("s1", ctx) is False
    assert mgr.load_session_state("s1", ctx) is False
    assert fs.exists.call_count == 2

    now[0] += 5.0
    assert mgr.load_session_state("s1", ctx) is True
    assert fs.exists.call_count == 3
    assert mgr.get_session_state().get_task_state("s1").objective == "Recovered later"


def test_get_or_create_does_not_repeat_missing_state_reads(ctx):
    fs = Mock()
    fs.exists.return_value = False
    mgr = SessionManager(get_context_fs=lambda: fs)

    mgr.get_or_create("s1", ctx)
    mgr.get_or_create("s1", ctx)
    mgr.add_message("s1", "user", "hello", ctx)
    mgr.get_session("s1", ctx)

    assert fs.exists.call_count == 2
    assert fs.read_node.call_count == 0


def test_save_session_state_clears_missing_state_cache(ctx):
    fs = Mock()
    fs.exists.side_effect = [False, False, True]
    saved_node = None

    def write_node(node, _write_ctx):
        nonlocal saved_node
        saved_node = node

    fs.write_node.side_effect = write_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "persist before reconnect", ctx)

    assert mgr.save_session_state("s1", ctx) is True
    assert saved_node is not None

    mgr.remove_session("s1", ctx)
    fs.read_node.return_value = saved_node
    restored = mgr.get_or_create("s1", ctx)

    assert fs.exists.call_count == 3
    assert [message.content for message in restored.messages] == [
        "persist before reconnect"
    ]


def test_save_session_state_does_not_reload_over_live_buffer(ctx):
    fs = Mock()
    fs.exists.return_value = False
    saved_node = None

    def write_node(node, _write_ctx):
        nonlocal saved_node
        saved_node = node

    fs.write_node.side_effect = write_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    buf = mgr.get_or_create("s1", ctx)
    buf.add("user", "persisted message")

    assert mgr.save_session_state("s1", ctx) is True
    assert saved_node is not None

    buf.add("assistant", "new live message")
    fs.exists.return_value = True
    fs.read_node.return_value = saved_node

    same = mgr.get_or_create("s1", ctx)

    assert same is buf
    assert [message.content for message in same.messages] == [
        "persisted message",
        "new live message",
    ]
    fs.read_node.assert_not_called()


def test_load_session_state_retries_after_storage_error(ctx):
    fs = Mock()
    fs.exists.side_effect = RuntimeError("storage unavailable")
    mgr = SessionManager(get_context_fs=lambda: fs)

    assert mgr.load_session_state("s1", ctx) is False
    assert mgr.load_session_state("s1", ctx) is False
    assert fs.exists.call_count == 4


def test_remove_session_invalidates_loaded_state_cache(ctx):
    payload = {"runtime_extensions": {"example": {"value": 1}}}
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state.json",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    mgr = SessionManager(get_context_fs=lambda: fs)

    first = mgr.get_or_create("s1", ctx=ctx)
    assert first.runtime_extensions == payload["runtime_extensions"]
    mgr.remove_session("s1", ctx)
    second = mgr.get_or_create("s1", ctx=ctx)

    assert second is not first
    assert second.runtime_extensions == payload["runtime_extensions"]
    assert fs.read_node.call_count == 2


def test_commit_persists_runtime_window_state_as_session_state(ctx):
    fs = Mock()
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "please preserve this runtime task", ctx)
    mgr._write_archive = Mock(return_value={
        "success": True,
        "uri": "memory://s1/a1",
        "archive_id": "a1",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    result = mgr.commit("s1", ctx, wait=True)

    assert result["archived"] is True
    node = fs.write_node.call_args.args[0]
    payload = json.loads(node.content)
    assert payload["task_state"]["objective"] == "please preserve this runtime task"
    assert payload["session_meta"]["commit_count"] == 1
    assert payload["session_meta"]["last_commit_at"]


def test_remove_session_removes_topic_buffer_under_manager_lock(ctx):
    mgr = SessionManager()
    first = mgr.get_topic_buffer("s1")
    first.set_cached_slot("archive_history", SlotContent(content="cached", tokens=1))

    mgr.remove_session("s1")
    second = mgr.get_topic_buffer("s1")

    assert second is not first
    assert second.get_cached_slot("archive_history") is None


def test_commit_succeeded_requires_archive_flag():
    assert SessionManager._commit_succeeded({"status": "completed", "archived": True}) is True
    assert SessionManager._commit_succeeded({"status": "completed", "archived": False}) is False
    assert SessionManager._commit_succeeded({"status": "failed", "archived": True}) is False


def test_get_or_create_concurrent_same_session_returns_same_buffer(monkeypatch):
    created_buffers: list[SessionBuffer] = []
    first_constructor_entered = threading.Event()
    release_first_constructor = threading.Event()
    original_init = SessionBuffer.__init__

    def blocking_init(self, *args, **kwargs):
        created_buffers.append(self)
        if len(created_buffers) == 1:
            first_constructor_entered.set()
            release_first_constructor.wait(timeout=2)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SessionBuffer, "__init__", blocking_init)
    mgr = SessionManager()
    results: list[SessionBuffer] = []

    first = threading.Thread(target=lambda: results.append(mgr.get_or_create("s1")))
    first.start()
    assert first_constructor_entered.wait(timeout=2)

    second = threading.Thread(target=lambda: results.append(mgr.get_or_create("s1")))
    second.start()
    release_first_constructor.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert results[0] is results[1]
    assert len(created_buffers) == 1


def test_get_or_create_waits_for_initial_load_before_returning(ctx, monkeypatch):
    real_event = threading.Event
    release_load = threading.Event()
    load_started = threading.Event()
    first_returned = threading.Event()
    second_returned = threading.Event()
    load_gate_events = []

    class InstrumentedEvent:
        def __init__(self):
            self._inner = real_event()
            self.wait_entered = real_event()
            load_gate_events.append(self)

        def wait(self, timeout=None):
            self.wait_entered.set()
            return self._inner.wait(timeout)

        def set(self):
            self._inner.set()

        def is_set(self):
            return self._inner.is_set()

    monkeypatch.setattr("session.session_manager.threading.Event", InstrumentedEvent)
    payload = {
        "task_state": {"objective": "Recovered durable task"},
        "session_meta": {
            "session_id": "s1",
            "created_at": "2026-05-20T01:00:00+00:00",
            "updated_at": "2026-05-20T01:01:00+00:00",
            "message_count": 0,
            "commit_count": 3,
            "last_commit_at": "2026-05-20T01:02:00+00:00",
            "account_id": "acct-test",
            "user_id": "u-test",
            "agent_id": "agent-test",
        },
        "window_state": {"active_task": ""},
    }
    fs = Mock()
    fs.exists.return_value = True

    def read_node(uri, read_ctx):
        load_started.set()
        release_load.wait(timeout=2)
        return ContextNode(
            uri=uri,
            context_type="RESOURCE",
            category="state",
            level=0,
            owner_space="session:s1",
            abstract="Session state",
            overview="Session state",
            content=json.dumps(payload),
            metadata={},
        )

    fs.read_node.side_effect = read_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    results: list[SessionBuffer] = []

    first = threading.Thread(
        target=lambda: (results.append(mgr.get_or_create("s1", ctx)), first_returned.set())
    )
    first.start()
    assert load_started.wait(timeout=2)

    second = threading.Thread(
        target=lambda: (results.append(mgr.get_or_create("s1", ctx)), second_returned.set())
    )
    second.start()
    assert load_gate_events[0].wait_entered.wait(timeout=2)
    assert not second_returned.is_set()

    release_load.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert first_returned.is_set()
    assert second_returned.is_set()
    assert len(results) == 2
    assert results[0] is results[1]
    assert [buf.meta.commit_count for buf in results] == [3, 3]
    assert [buf.window_state.active_task for buf in results] == [
        "Recovered durable task",
        "Recovered durable task",
    ]


def test_get_or_create_ctx_after_ctxless_creation_loads_persisted_state(ctx):
    payload = {
        "task_state": {"objective": "Recovered after ctxless create"},
        "session_meta": {
            "session_id": "s1",
            "created_at": "2026-05-20T01:00:00+00:00",
            "updated_at": "2026-05-20T01:01:00+00:00",
            "message_count": 0,
            "commit_count": 4,
            "last_commit_at": "2026-05-20T01:02:00+00:00",
            "account_id": "acct-test",
            "user_id": "u-test",
            "agent_id": "agent-test",
        },
        "window_state": {"active_task": ""},
    }
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state.json",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    mgr = SessionManager(get_context_fs=lambda: fs)

    first = mgr.get_or_create("s1")
    second = mgr.get_or_create("s1", ctx)

    assert second is first
    assert second.meta.commit_count == 4
    assert second.window_state.active_task == "Recovered after ctxless create"


def test_save_session_state_serializes_sync_and_write(ctx):
    fs = Mock()
    fs.exists.return_value = False
    events: list[tuple[str, str]] = []
    release_write = threading.Event()
    write_started = threading.Event()

    def write_node(node, write_ctx):
        payload = json.loads(node.content)
        events.append((
            payload["task_state"]["objective"],
            payload["window_state"]["active_task"],
        ))
        write_started.set()
        release_write.wait(timeout=2)

    fs.write_node.side_effect = write_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "please preserve first runtime task", ctx)

    first = threading.Thread(target=mgr.save_session_state, args=("s1", ctx))
    first.start()
    assert write_started.wait(timeout=2)

    second = threading.Thread(target=mgr.save_session_state, args=("s1", ctx))
    second.start()
    assert len(events) == 1

    release_write.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert events == [
        ("please preserve first runtime task", "please preserve first runtime task"),
        ("please preserve first runtime task", "please preserve first runtime task"),
    ]


def test_save_session_state_snapshots_messages_and_watermark_atomically(ctx):
    fs = Mock()
    fs.exists.return_value = False
    saved_payload = None
    iteration_finished = threading.Event()
    release_iteration = threading.Event()

    class BlockingMessages(list):
        def __iter__(self):
            yield from super().__iter__()
            iteration_finished.set()
            assert release_iteration.wait(timeout=2)

    def write_node(node, _write_ctx):
        nonlocal saved_payload
        saved_payload = json.loads(node.content)

    fs.write_node.side_effect = write_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    buf = mgr.get_or_create("s1", ctx)
    first = buf.add("user", "first durable message")
    buf.add("assistant", "second durable message")
    buf.extraction_watermark = 2
    buf.messages = BlockingMessages(buf.messages)

    save_thread = threading.Thread(target=mgr.save_session_state, args=("s1", ctx))
    save_thread.start()
    assert iteration_finished.wait(timeout=2)

    snapshot_locked = not buf.extraction_lock.acquire(blocking=False)
    if not snapshot_locked:
        try:
            buf.remove_messages_by_id({first.id})
        finally:
            buf.extraction_lock.release()
    release_iteration.set()
    save_thread.join(timeout=2)

    assert not save_thread.is_alive()
    assert snapshot_locked is True
    assert saved_payload is not None
    assert len(saved_payload["buffer_messages"]) == 2
    assert saved_payload["extraction_watermark"] == 2


def test_completed_request_ids_survive_session_state_reload(ctx):
    fs = Mock()
    fs.exists.return_value = False
    saved_node = None

    def write_node(node, _write_ctx):
        nonlocal saved_node
        saved_node = node

    fs.write_node.side_effect = write_node
    manager = SessionManager(get_context_fs=lambda: fs)
    buffer = manager.get_or_create("s1", ctx)
    buffer.add("user", "pending durable message", "2026-08-18T10:00:00+00:00")
    buffer.mark_request_completed("pending-1")

    assert manager.save_session_state("s1", ctx) is True
    assert saved_node is not None

    fs.exists.return_value = True
    fs.read_node.return_value = saved_node
    restored = SessionManager(get_context_fs=lambda: fs).get_or_create("s1", ctx)

    assert restored.request_completed("pending-1") is True
    assert [(message.role, message.content) for message in restored.messages] == [
        ("user", "pending durable message")
    ]
    assert restored.turn_count == 1


def test_remove_session_reloads_persisted_buffer(ctx):
    payload = {
        "buffer_messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "durable pending message",
                "created_at": "2026-08-19T00:00:00+00:00",
            }
        ]
    }
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    manager = SessionManager(get_context_fs=lambda: fs)

    first = manager.get_or_create("s1", ctx)
    manager.remove_session("s1", ctx)
    restored = manager.get_or_create("s1", ctx)

    assert restored is not first
    assert [message.content for message in restored.messages] == [
        "durable pending message"
    ]
    assert fs.read_node.call_count == 2


def test_request_claim_is_atomic_and_released_on_completion():
    buffer = SessionBuffer(session_id="s1")

    assert buffer.claim_request("pending-1") == "claimed"
    assert buffer.claim_request("pending-1") == "processing"
    buffer.mark_request_completed("pending-1")
    assert buffer.claim_request("pending-1") == "processing"
    buffer.release_request("pending-1")
    assert buffer.claim_request("pending-1") == "completed"


def test_commit_wait_process_exception_returns_failed_and_restores_buffer(ctx):
    mgr = SessionManager()
    mgr.add_message("s1", "user", "please preserve failed commit", ctx)
    original = [msg.content for msg in mgr.get_or_create("s1").messages]
    mgr._process_snapshot = Mock(side_effect=RuntimeError("boom"))

    result = mgr.commit("s1", ctx, wait=True)
    buf = mgr.get_or_create("s1")
    task = mgr.get_task(result["task_id"])

    assert result["archived"] is False
    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert task["status"] == "failed"
    assert task["error"] == "boom"
    assert buf.commit_in_progress is False
    assert [msg.content for msg in buf.messages] == original


def test_commit_wait_save_io_does_not_hold_manager_lock(ctx):
    fs = Mock()
    write_started = threading.Event()
    release_write = threading.Event()
    add_completed = threading.Event()

    def write_node(node, write_ctx):
        write_started.set()
        release_write.wait(timeout=2)

    fs.write_node.side_effect = write_node
    mgr = SessionManager(get_context_fs=lambda: fs)
    mgr.add_message("s1", "user", "please persist without blocking manager", ctx)
    mgr._write_archive = Mock(
        return_value={"success": True, "uri": "memory://s1/a1", "archive_id": "a1"}
    )
    mgr._compress = Mock(return_value=("overview", "abstract"))

    commit_thread = threading.Thread(target=mgr.commit, args=("s1", ctx), kwargs={"wait": True})
    commit_thread.start()
    assert write_started.wait(timeout=2)

    add_thread = threading.Thread(
        target=lambda: (
            mgr.add_message("s2", "user", "should not wait for s1 write", ctx),
            add_completed.set(),
        )
    )
    add_thread.start()
    add_thread.join(timeout=2)

    assert not add_thread.is_alive()
    assert add_completed.is_set()

    release_write.set()
    commit_thread.join(timeout=2)

    assert not commit_thread.is_alive()


def test_commit_snapshot_wait_process_exception_returns_failed_task(ctx):
    mgr = SessionManager()
    snapshot = [SessionMessage(id="m1", role="user", content="snapshot message")]
    mgr._process_snapshot = Mock(side_effect=RuntimeError("snapshot boom"))

    result = mgr.commit_snapshot("s1", snapshot, ctx, wait=True)
    task = mgr.get_task(result["task_id"])

    assert result["archived"] is False
    assert result["status"] == "failed"
    assert result["error"] == "snapshot boom"
    assert task["status"] == "failed"
    assert task["error"] == "snapshot boom"


def test_get_or_create_lazy_loads_persisted_session_state_and_meta(ctx):
    payload = {
        "task_state": {
            "objective": "Recovered durable task",
            "current_stage": None,
            "next_step": None,
            "blockers": ["Recovered blocker"],
        },
        "commitments": [],
        "session_meta": {
            "session_id": "s1",
            "created_at": "2026-05-20T01:00:00+00:00",
            "updated_at": "2026-05-20T01:01:00+00:00",
            "message_count": 0,
            "commit_count": 2,
            "last_commit_at": "2026-05-20T01:02:00+00:00",
            "account_id": "acct-test",
            "user_id": "u-test",
            "agent_id": "agent-test",
        },
        "window_state": {
            "active_task": "",
            "last_accessed_at": "2026-05-20T01:03:00+00:00",
        },
    }
    fs = Mock()
    fs.exists.return_value = True
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-test/sessions/s1/state.json",
        context_type="RESOURCE",
        category="state",
        level=0,
        owner_space="session:s1",
        abstract="Session state",
        overview="Session state",
        content=json.dumps(payload),
        metadata={},
    )
    mgr = SessionManager(get_context_fs=lambda: fs)

    buf = mgr.get_or_create("s1", ctx)

    assert buf.meta.commit_count == 2
    assert buf.meta.last_commit_at == "2026-05-20T01:02:00+00:00"
    assert buf.window_state.active_task == "Recovered durable task"
    assert buf.window_state.uncertainties == ["Recovered blocker"]


# ---------------------------------------------------------------------------
# SessionMessage model tests
# ---------------------------------------------------------------------------


class TestSessionMessage:
    def test_estimated_tokens(self):
        msg = SessionMessage(id="m1", role="user", content="a" * 100)
        assert msg.estimated_tokens == 25

    def test_estimated_tokens_minimum(self):
        msg = SessionMessage(id="m1", role="user", content="")
        assert msg.estimated_tokens == 1  # max(1, 0)


# ---------------------------------------------------------------------------
# SessionMeta model tests
# ---------------------------------------------------------------------------


class TestSessionMeta:
    def test_defaults(self):
        meta = SessionMeta()
        assert meta.session_id == ""
        assert meta.message_count == 0
        assert meta.commit_count == 0

    def test_with_session_id(self):
        meta = SessionMeta(session_id="s1")
        assert meta.session_id == "s1"
