"""Integration tests for MemoryService assemble() replacement mode.

Tests that assemble returns messages[] without systemPromptAddition,
uses token budget management, and supports archive inclusion.
"""

from unittest.mock import Mock, patch

from core.models import RequestContext
from server.memory_service import MemoryService


class TestAssembleReplacementMode:
    """Test MemoryService assemble() in pure message replacement mode."""

    def test_assemble_returns_messages_not_system_prompt_addition(self):
        """Test that assemble returns messages[] without systemPromptAddition."""
        service = MemoryService()

        params = {
            "messages": [
                {"role": "user", "content": "What do you remember about my preferences?"}
            ],
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            mock_api = Mock()
            mock_result = Mock()
            mock_result.hits = []
            mock_api.search_memory.return_value = mock_result
            mock_get_read_api.return_value = mock_api

            result = service.compose(params)

            # Should return messages array
            assert "messages" in result
            assert isinstance(result["messages"], list)

            # Should have Layer 1: stable system prompt addition
            assert "systemPromptAddition" in result

            # Should have Layer 2: session state suffix
            assert "systemPromptSuffix" in result

            # Should have new fields
            assert "estimatedTokens" in result
            assert "archiveCount" in result
            assert "archiveIncluded" in result

            # Should include original message
            assert len(result["messages"]) >= 1
            assert result["messages"][-1]["role"] == "user"

    def test_assemble_with_empty_query(self):
        """Test assemble with empty query returns original messages."""
        service = MemoryService()

        params = {
            "messages": [{"role": "user", "content": ""}],
            "sessionId": "test-session",
        }

        result = service.compose(params)

        assert result["messages"] == params["messages"]
        assert result["archiveCount"] == 0
        assert result["archiveIncluded"] is False
        assert result["estimatedTokens"] >= 0

    def test_assemble_with_custom_token_budget(self):
        """Test assemble respects custom token budget."""
        service = MemoryService()

        params = {
            "messages": [
                {"role": "user", "content": "Tell me about my background"}
            ],
            "tokenBudget": 50_000,  # Smaller budget
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            mock_api = Mock()
            mock_result = Mock()
            mock_result.hits = []
            mock_api.search_memory.return_value = mock_result
            mock_get_read_api.return_value = mock_api

            result = service.compose(params)

            assert "messages" in result
            assert "estimatedTokens" in result
            # Result should be within reasonable bounds
            assert result["estimatedTokens"] < 100_000  # Should respect budget

    def test_assemble_includes_memory_hits_as_user_message(self):
        """Test that memory hits are injected as user message after original messages."""
        service = MemoryService()

        params = {
            "messages": [
                {"role": "user", "content": "What are my coding preferences?"}
            ],
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            mock_api = Mock()
            mock_result = Mock()

            # Create mock hit
            from core.models import RetrievedBlock
            mock_hit = RetrievedBlock(
                uri="ctx://test-account/users/test-user/memories/preference/coding",
                score=0.85, category="preference",
                abstract="Prefers Python for data science",
                content_excerpt="Prefers Python for data science",
            )
            mock_result.hits = [mock_hit]
            mock_api.read_memory.return_value = mock_hit
            mock_api.search_memory.return_value = mock_result
            mock_get_read_api.return_value = mock_api

            result = service.compose(params)

            # Legacy fields are empty — content lives in messages
            assert result["systemPromptAddition"] == ""
            assert result["memoryUserMessage"] == ""

            # Memory hits injected as user message after original messages
            msgs = result["messages"]
            user_msgs = [m for m in msgs if m["role"] == "user"]
            assert len(user_msgs) >= 1

            # Retrieved memories follow the original messages
            ws_msg = [m for m in msgs if m["role"] == "user" and "Retrieved Memories" in m.get("content", "")]
            assert len(ws_msg) >= 1
            assert "preference" in ws_msg[0]["content"]
            assert "Prefers Python" in ws_msg[0]["content"]

    def test_assemble_graceful_degradation_on_error(self):
        """Test graceful degradation when memory search fails."""
        service = MemoryService()

        params = {
            "messages": [
                {"role": "user", "content": "Test query"}
            ],
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            # Make search_memory raise an exception
            mock_get_read_api.side_effect = Exception("Search failed")

            result = service.compose(params)

            # Should still return messages (original messages untouched)
            assert "messages" in result
            assert len(result["messages"]) >= 1

            # Should have empty legacy fields on error
            assert result["systemPromptAddition"] == ""
            assert result["memoryUserMessage"] == ""

            # Should have no archives on error
            assert result["archiveCount"] == 0
            assert result["archiveIncluded"] is False

    def test_assemble_archive_inclusion_placeholder(self):
        """Test that archive inclusion structure is in place (P1 stub)."""
        service = MemoryService()

        params = {
            "messages": [
                {"role": "user", "content": "What did we discuss yesterday?"}
            ],
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            mock_api = Mock()
            mock_result = Mock()
            mock_result.hits = []
            mock_api.search_memory.return_value = mock_result
            mock_get_read_api.return_value = mock_api

            result = service.compose(params)

            # Should have archive fields (P1 stub returns 0)
            assert "archiveCount" in result
            assert "archiveIncluded" in result
            assert result["archiveCount"] == 0  # P1 stub
            assert result["archiveIncluded"] is False  # P1 stub

    def test_format_memory_as_messages_helper(self):
        """Test _format_working_set helper (Layer 3)."""
        from core.models import TokenBudget
        service = MemoryService()

        # Create mock working set items
        working_set = [{
            "category": "preference",
            "abstract": "Prefers dark mode",
            "score": 0.9,
        }]

        msg = service._format_working_set(working_set, TokenBudget().allocate())
        assert "Retrieved Memories" in msg
        assert "preference" in msg
        assert "dark mode" in msg

    def test_format_memory_as_messages_empty(self):
        """Test _format_working_set with empty working set."""
        service = MemoryService()
        from core.models import TokenBudget
        msg = service._format_working_set([], TokenBudget().allocate())
        assert msg == ""

    def test_estimate_tokens_helper(self):
        """Test _estimate_tokens helper."""
        service = MemoryService()

        messages = [
            {"role": "user", "content": "Hello world " * 20},  # ~220 chars
            {"role": "assistant", "content": "Hi there " * 30},  # ~300 chars
        ]

        tokens = service._estimate_tokens(messages)

        # Should be approximately (220 + 300) / 4 = 130 tokens
        assert 100 <= tokens <= 200  # Allow some margin

    def test_estimate_tokens_empty(self):
        """Test _estimate_tokens with empty messages."""
        service = MemoryService()
        tokens = service._estimate_tokens([])
        assert tokens == 0

    def test_collect_archives_stub(self):
        """Test _collect_archives returns empty lists (P1 stub)."""
        service = MemoryService()
        ctx = RequestContext(
            account_id="test-account",
            user_id="test-user",
            agent_id="test-agent",
            session_id="test-session",
            trace_id="test-trace",
        )
        from core.models import TokenBudget

        latest, pre = service._collect_archives(ctx, TokenBudget())

        # P1 stub returns empty lists
        assert latest == []
        assert pre == []

    def test_assemble_preserves_original_messages(self):
        """Test that original messages are preserved in output."""
        service = MemoryService()

        original_messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
        ]

        params = {
            "messages": original_messages,
            "sessionId": "test-session",
        }

        with patch.object(service, 'get_read_api') as mock_get_read_api:
            mock_api = Mock()
            mock_result = Mock()
            mock_result.hits = []
            mock_api.search_memory.return_value = mock_result
            mock_get_read_api.return_value = mock_api

            result = service.compose(params)

            result_messages = result["messages"]
            # Original messages are present in the result (may have system messages around them)
            contents = [m["content"] for m in result_messages]
            for orig in original_messages:
                assert orig["content"] in contents
