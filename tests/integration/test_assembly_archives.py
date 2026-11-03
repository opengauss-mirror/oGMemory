"""Integration tests for archive-aware assembly pipeline.

Tests the full assemble() flow:
- All context injected into messages as user messages (stable → dynamic order)
- systemPromptAddition/systemPromptSuffix/memoryUserMessage: empty
- Budget constrains archive content correctly
- Graceful degradation when store unavailable
"""

from unittest.mock import MagicMock, patch

from core.models import ArchiveRef, RequestContext, RetrievedBlock, TokenBudget
from server.memory_service import MemoryService, build_archive_refs
from session.models import ArchiveEntry


def make_entry(archive_id: str, session_id: str = "s1",
               overview: str = "overview", abstract: str = "abstract",
               created_at: str = "2024-01-01T00:00:00") -> ArchiveEntry:
    return ArchiveEntry(
        archive_id=archive_id,
        session_id=session_id,
        overview=overview,
        abstract=abstract,
        messages=[],
        created_at=created_at,
    )


def make_ctx(**overrides) -> RequestContext:
    defaults = dict(
        account_id="acct-test",
        user_id="u-test",
        agent_id="agent-test",
        session_id="sess-test",
        trace_id="trace-001",
    )
    defaults.update(overrides)
    return RequestContext(**defaults)


def make_read_api(items):
    api = MagicMock()
    by_uri = {item["uri"]: item for item in items}
    def read_memory(uri, ctx, **kwargs):
        item = by_uri[uri]
        return RetrievedBlock(
            uri=uri, score=item["score"], category=item["category"],
            abstract=item["abstract"], content_excerpt=item["abstract"],
        )
    api.read_memory.side_effect = read_memory
    return api


class TestAssembleSystemPrompt:
    """Test that assemble() builds system prompt with stable prefix + dynamic suffix."""

    @patch("server.memory_service._HAS_AGFS", False)
    def test_no_agfs_returns_empty_archives(self):
        """Without AGFS, archives are empty but assemble still works."""
        svc = MemoryService()
        result = svc.compose({
            "messages": [{"role": "user", "content": "hello"}],
            "sessionId": "sess-1",
        })
        assert "messages" in result
        assert result["archiveCount"] == 0
        assert result["archiveIncluded"] is False

    def test_messages_returned_with_original_preserved(self):
        """Original messages are preserved within injected user messages."""
        svc = MemoryService()
        original = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with patch.object(svc, "_read_profile", return_value=""):
            with patch.object(svc, "_collect_archives", return_value=([], [])):
                with patch.object(svc, "_search_working_set", return_value=([], [], {})):
                    result = svc.compose({
                        "messages": original,
                        "prompt": "test query",
                    })

        # No dynamic content → messages should equal original
        assert result["messages"] == original

    def test_archive_injected_as_user_message(self):
        """Archive content goes into messages as user message (episodic layer)."""
        svc = MemoryService()
        latest = [ArchiveRef(archive_id="a1", archive_uri="u1", abstract="ab1",
                             overview="Full overview of session", tokens=10)]
        pre = [ArchiveRef(archive_id="a2", archive_uri="u2", abstract="Old abstract",
                          overview=None, tokens=5)]

        with patch.object(svc, "_read_profile", return_value=""):
            with patch.object(svc, "_collect_archives", return_value=(latest, pre)):
                with patch.object(svc, "_search_working_set", return_value=([], [], {})):
                    result = svc.compose({
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt": "test query",
                    })

        # Legacy fields are empty
        assert result["systemPromptSuffix"] == ""
        assert result["systemPromptAddition"] == ""
        assert result["memoryUserMessage"] == ""

        # Archives injected as user message in messages
        msgs = result["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 1

        archive_msg = [m for m in user_msgs if "Archive History" in m.get("content", "")]
        assert len(archive_msg) >= 1
        assert "Full overview of session" in archive_msg[0]["content"]
        assert "Old abstract" in archive_msg[0]["content"]

    def test_working_set_injected_as_user_message(self):
        """Working set goes into messages as user message after original messages."""
        svc = MemoryService()
        ws = [
            {"uri": "u1", "abstract": "Found memory about X", "score": 0.9, "category": "entity"},
        ]

        with patch.object(svc, "get_read_api", return_value=make_read_api(ws)), patch.object(svc, "_read_profile", return_value=""):
            with patch.object(svc, "_collect_archives", return_value=([], [])):
                with patch.object(svc, "_search_working_set", return_value=(ws, [], {})):
                    result = svc.compose({
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt": "test query",
                    })

        # Legacy fields are empty
        assert result["memoryUserMessage"] == ""
        assert result["systemPromptAddition"] == ""

        # Working set injected as user message in messages
        msgs = result["messages"]
        ws_msgs = [m for m in msgs if m["role"] == "user" and "Retrieved Memories" in m.get("content", "")]
        assert len(ws_msgs) >= 1
        assert "Found memory about X" in ws_msgs[0]["content"]

    def test_stable_before_dynamic_ordering(self):
        """Profile → Archive → [original] → Working Set order in messages."""
        svc = MemoryService()
        latest = [ArchiveRef(archive_id="a1", archive_uri="u1", abstract="ab1",
                             overview="Archive overview", tokens=10)]
        ws = [{"uri": "u1", "abstract": "Working set item", "score": 0.8, "category": "memory"}]

        with patch.object(svc, "get_read_api", return_value=make_read_api(ws)), patch.object(svc, "_read_profile", return_value="User profile text"):
            with patch.object(svc, "_collect_archives", return_value=(latest, [])):
                with patch.object(svc, "_search_working_set", return_value=(ws, [], {})):
                    result = svc.compose({
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt": "test query",
                    })

        # Legacy fields are empty
        assert result["systemPromptAddition"] == ""
        assert result["systemPromptSuffix"] == ""
        assert result["memoryUserMessage"] == ""

        # All content in messages as user messages
        msgs = result["messages"]
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]

        # Find indices
        profile_idx = next((i for i, c in enumerate(user_contents) if "Profile" in c), None)
        archive_idx = next((i for i, c in enumerate(user_contents) if "Archive History" in c), None)
        ws_idx = next((i for i, c in enumerate(user_contents) if "Retrieved Memories" in c), None)

        # Profile and Archive come before Working Set
        assert profile_idx is not None
        assert archive_idx is not None
        assert ws_idx is not None
        assert profile_idx < ws_idx
        assert archive_idx < ws_idx


class TestAssembleBudgetConstraints:
    """Budget constrains archive content correctly."""

    def test_small_budget_truncates_pre_archives(self):
        """Small budget should drop pre-archive entries."""
        entries = [
            make_entry("arc1", abstract="A" * 5000, created_at="2024-01-01T00:00:00"),
            make_entry("arc2", abstract="B" * 5000, created_at="2024-01-02T00:00:00"),
            make_entry("arc3", overview="Recent overview", abstract="Recent abstract",
                       created_at="2024-01-03T00:00:00"),
        ]
        budget = TokenBudget(total=500, archive_ratio=0.5)
        latest, pre = build_archive_refs(entries, budget)

        assert len(latest) == 1
        # archive_limit = 250, pre-archives have 1250 tokens each → all dropped
        assert len(pre) == 0

    def test_archive_count_in_return_value(self):
        """archiveCount reflects total archives found."""
        svc = MemoryService()
        latest = [ArchiveRef(archive_id="a1", archive_uri="u1", abstract="ab1",
                             overview="ov1", tokens=10)]
        pre = [ArchiveRef(archive_id="a2", archive_uri="u2", abstract="ab2",
                          overview=None, tokens=5)]

        with patch.object(svc, "_read_profile", return_value=""):
            with patch.object(svc, "_collect_archives", return_value=(latest, pre)):
                with patch.object(svc, "_search_working_set", return_value=([], [], {})):
                    result = svc.compose({
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt": "test query",
                    })

        assert result["archiveCount"] == 2
        assert result["archiveIncluded"] is True


class TestGracefulDegradation:
    """Graceful degradation when store unavailable."""

    @patch("server.memory_service._HAS_AGFS", True)
    @patch("server.memory_service.AGFSClient", side_effect=Exception("connection failed"))
    def test_collect_archives_failure_returns_empty(self, mock_client):
        """When AGFS connection fails, _collect_archives returns empty lists."""
        svc = MemoryService()
        ctx = make_ctx()
        budget = TokenBudget()
        latest, pre = svc._collect_archives(ctx, budget)
        assert latest == []
        assert pre == []

    @patch("server.memory_service._HAS_AGFS", True)
    @patch("server.memory_service.AGFSClient")
    @patch("server.memory_service.AGFSContextFS")
    def test_list_archives_exception_returns_empty(self, mock_agfs_cls, mock_client_cls):
        """When list_archives raises, _collect_archives returns empty lists."""
        mock_store = MagicMock()
        mock_store.list_archives.side_effect = Exception("storage error")

        with patch("session.SessionArchiveStore", return_value=mock_store):
            svc = MemoryService()
            ctx = make_ctx()
            budget = TokenBudget()
            latest, pre = svc._collect_archives(ctx, budget)

        assert latest == []
        assert pre == []

    def test_profile_failure_preserves_original_messages(self):
        """When pipeline fails, original messages are returned."""
        svc = MemoryService()
        original = [{"role": "user", "content": "hello"}]
        with patch.object(svc, "_read_profile", side_effect=RuntimeError("boom")):
            result = svc.compose({
                "messages": original,
                "prompt": "test query",
            })

        assert result["messages"] == original
        assert result["systemPromptAddition"] == ""

    def test_search_failure_still_returns_archives(self):
        """Working set search failure doesn't prevent archive injection."""
        svc = MemoryService()
        latest = [ArchiveRef(archive_id="a1", archive_uri="u1", abstract="ab1",
                             overview="Archive overview", tokens=10)]

        with patch.object(svc, "_read_profile", return_value=""):
            with patch.object(svc, "_collect_archives", return_value=(latest, [])):
                with patch.object(svc, "_search_working_set", side_effect=Exception("search down")):
                    result = svc.compose({
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt": "test query",
                    })

        # Should still get archive in messages as user message
        msgs = result["messages"]
        archive_msgs = [m for m in msgs if m["role"] == "user" and "Archive overview" in m.get("content", "")]
        assert len(archive_msgs) >= 1
        assert result["archiveCount"] == 1
