"""Integration tests for session lifecycle.

Tests the full flow: add messages → threshold → commit → get context.
Uses mocked AGFS/LLM to avoid external dependencies.
"""

from unittest.mock import Mock, patch

import pytest

from core.models import RequestContext
from server.memory_service import MemoryService, extract_content_text
from session.session_manager import SessionManager


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct-test",
        user_id="u-test",
        agent_id="agent-test",
        session_id="sess-lifecycle",
        trace_id="trace-1",
    )


@pytest.fixture
def mock_write_api():
    api = Mock()
    api.commit_session.return_value = {
        "candidates_extracted": 1,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "task_id": "t-1",
        "status": "completed",
    }
    return api


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.complete_json.return_value = {
        "overview": "Test overview of the conversation",
        "abstract": "Test abstract",
    }
    return llm


@pytest.fixture
def mgr(mock_llm, mock_write_api):
    return SessionManager(
        get_llm=lambda: mock_llm,
        get_write_api=lambda: mock_write_api,
        get_agfs=lambda: None,
    )


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_add_messages_commit_get_context(self, mgr, ctx):
        """Full lifecycle: add → threshold → commit → context."""
        sid = "sess-full"

        # 1. Add messages
        for i in range(5):
            mgr.add_message(sid, "user", f"Message {i} " + "x" * 200, ctx)
            mgr.add_message(sid, "assistant", f"Reply {i} " + "y" * 200, ctx)

        session = mgr.get_session(sid, ctx)
        assert session["message_count"] == 10
        assert session["pending_tokens"] > 0

        # 2. Commit
        result = mgr.commit(sid, ctx, wait=True)
        assert result["archived"] is True
        assert "archive_id" in result

        # Buffer should be cleared
        session_after = mgr.get_session(sid, ctx)
        assert session_after["pending_tokens"] == 0
        assert session_after["commit_count"] == 1

        # 3. Get context
        context = mgr.get_context(sid, 128_000, ctx)
        assert context["active_message_count"] == 0
        assert context["archive_count"] == 0  # No AGFS, so no persisted archives

    def test_threshold_triggers_commit(self, mgr, ctx):
        """Add messages until threshold triggers auto-commit logic."""
        sid = "sess-threshold"
        threshold = 500  # Low threshold for testing

        total_tokens = 0
        msgs_added = 0
        while total_tokens < threshold:
            mgr.add_message(sid, "user", "x" * 200, ctx)  # ~50 tokens each
            session = mgr.get_session(sid, ctx)
            total_tokens = session["pending_tokens"]
            msgs_added += 1

        assert total_tokens >= threshold

        # Commit
        result = mgr.commit(sid, ctx, wait=True)
        assert result["archived"] is True

    def test_multiple_commits(self, mgr, ctx):
        """Multiple commits produce separate archives."""
        sid = "sess-multi"

        # First commit
        mgr.add_message(sid, "user", "First batch " + "a" * 200, ctx)
        r1 = mgr.commit(sid, ctx, wait=True)
        assert r1["archived"] is True

        # Second commit
        mgr.add_message(sid, "user", "Second batch " + "b" * 200, ctx)
        r2 = mgr.commit(sid, ctx, wait=True)
        assert r2["archived"] is True
        assert r1["archive_id"] != r2["archive_id"]

        session = mgr.get_session(sid, ctx)
        assert session["commit_count"] == 2


# ---------------------------------------------------------------------------
# MemoryService integration
# ---------------------------------------------------------------------------


class TestMemoryServiceSessionIntegration:
    @pytest.fixture
    def service(self):
        import uuid

        from providers.unified_config import OgMemConfig
        cfg = OgMemConfig(
            account_id=f"session-test-{uuid.uuid4().hex}",
            agfs_base_url="http://localhost:1833",
            agfs_mount_prefix="/local/plugin",
        )
        return MemoryService(config=cfg)

    def test_after_turn_accumulates(self, service):
        """after_turn adds messages to buffer without committing under threshold."""
        params = {
            "sessionId": "sess-at",
            "messages": [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ],
            "prePromptMessageCount": 0,
            "commitTokenThreshold": 100_000,  # High threshold, no commit
        }
        result = service.after_turn(params)
        assert result["ok"] is True
        assert result["status"] == "accumulating"
        assert result["pending_tokens"] > 0

    def test_after_turn_force_extract_bypasses_threshold(self, service):
        mgr = service.get_session_manager()
        with (
            patch.object(service, "get_write_api", return_value=None),
            patch.object(mgr, "commit", return_value={"status": "processing"}) as commit,
        ):
            result = service.after_turn({
                "sessionId": "sess-force",
                "messages": [{"role": "user", "content": "Short memory"}],
                "forceExtract": True,
            })

        assert result["status"] == "processing"
        commit.assert_called_once()

    def test_after_turn_commits_at_threshold(self, service):
        """after_turn triggers commit when pending_tokens >= threshold."""
        # First turn: accumulate
        big_msg = "x" * 800  # ~200 tokens
        params = {
            "sessionId": "sess-thresh",
            "messages": [
                {"role": "user", "content": big_msg},
                {"role": "assistant", "content": big_msg},
            ],
            "prePromptMessageCount": 0,
            "commitTokenThreshold": 100,  # Low threshold
        }
        result = service.after_turn(params)
        # Should have triggered commit (400 tokens > 100 threshold)
        assert result["ok"] is True
        assert result.get("status") in ("processing", "completed")

    def test_compact_returns_summary(self, service):
        """compact() commits and returns summary with tokensBefore/After."""
        service._cfg.after_turn_threshold = 100_000
        # Add messages first via after_turn with high threshold
        for i in range(3):
            accumulated = service.after_turn({
                "sessionId": "sess-compact",
                "messages": [
                    {"role": "user", "content": f"Question {i}: " + "a" * 200},
                    {"role": "assistant", "content": f"Answer {i}: " + "b" * 200},
                ],
                "prePromptMessageCount": 0,
            })
            assert accumulated["status"] == "accumulating"

        # Now compact
        result = service.compact({
            "sessionId": "sess-compact",
            "tokenBudget": 128_000,
        })
        assert result["ok"] is True
        assert result["compacted"] is True
        assert "result" in result
        assert "summary" in result["result"]
        assert "tokensBefore" in result["result"]
        assert "tokensAfter" in result["result"]
        assert isinstance(result["result"]["summary"], str)
        assert "firstKeptEntryId" not in result["result"]

    def test_compact_empty_session(self, service):
        """compact() on empty session returns not compacted."""
        result = service.compact({
            "sessionId": "sess-empty",
            "tokenBudget": 128_000,
        })
        assert result["ok"] is True
        assert result["compacted"] is False

    def test_extract_content_text_variants(self):
        """extract_content_text handles various content formats."""
        assert extract_content_text("plain text") == "plain text"
        assert extract_content_text([{"text": "hello"}]) == "hello"
        assert extract_content_text([{"text": "a"}, {"text": "b"}]) == "a b"
        assert extract_content_text(["a", "b"]) == "a b"
        assert extract_content_text(123) == "123"

    def test_after_turn_skips_pre_prompt(self, service):
        """after_turn skips messages before prePromptMessageCount."""
        params = {
            "sessionId": "sess-skip",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "system", "content": "More instructions"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "prePromptMessageCount": 2,
            "commitTokenThreshold": 100_000,
        }
        result = service.after_turn(params)
        assert result["ok"] is True
        assert result["status"] == "accumulating"

        # Check that only 2 messages (not 4) were added
        ctx = service.build_context(params)
        session = service.get_session_manager().get_session("sess-skip", ctx)
        # "Hello" (~1 token) + "Hi!" (~1 token) = 2 tokens
        assert session["message_count"] == 2

    def test_after_turn_returns_failure_when_archive_commit_fails(self, service):
        """after_turn surfaces archive commit failures instead of reporting success."""
        mgr = service.get_session_manager()

        with patch.object(
            mgr,
            "commit_snapshot",
            return_value={
                "ok": True,
                "archived": False,
                "archive_id": "arc-failed",
                "status": "failed",
                "error": "archive store unavailable",
            },
        ):
            result = service.after_turn({
                "sessionId": "sess-archive-fail",
                "messages": [
                    {"role": "assistant", "content": "y" * 800},
                ],
                "prePromptMessageCount": 0,
                "commitTokenThreshold": 100,
            })

        assert result["ok"] is False
        assert result["status"] == "failed"
        assert "archive store unavailable" in result["error"]
