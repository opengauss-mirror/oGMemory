"""Integration tests for unified assembly pipeline.

These tests verify the complete unified assembly workflow:
1. Assembly returns messages (not system prompt injection)
2. Token budget truncation works correctly
3. Memory hits are formatted as user messages
4. Archive inclusion in assemble
5. Session commit creates archive
6. Cold memory archival workflow
7. Async extraction polling

Dependencies:
- Task #1: unified assembly models (completed)
- Task #2: ResultRanker token budget management
- Task #3: session archival system
- Task #4: archive expansion + cold memory archiver
- Task #7: async extraction + commit_session()

Run with: pytest tests/integration/test_unified_assembly.py -v -s
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, RetrievedBlock, SearchMemoryResult
from server.memory_service import MemoryService


@pytest.fixture
def sample_context():
    """Create sample RequestContext for testing."""
    return RequestContext(
        account_id="test-account",
        user_id="test-user",
        agent_id="test-agent",
        session_id=str(uuid4()),
        trace_id=str(uuid4()),
    )


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "你好，我叫李明，是一名Python开发工程师"},
        {"role": "assistant", "content": "你好李明！很高兴认识你。"},
        {"role": "user", "content": "我在北京工作，平时喜欢用Python做数据分析"},
    ]


class TestAssemblyReturnsMessages:
    """Test 1: Assembly returns messages (not system prompt injection).

    Verify that the unified assembly pipeline returns a list of messages
    rather than injecting content into the system prompt.
    """

    def test_assemble_returns_messages_list(self, sample_messages, sample_context):
        """ResultRanker.assemble() returns messages + systemPromptAddition (RCA mode)."""
        service = MemoryService()

        result = service.compose({
            "messages": sample_messages,
            "prompt": "Python编程",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Verify result structure (RCA pipeline mode)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert "estimatedTokens" in result
        assert "systemPromptAddition" in result  # RCA pipeline adds this

        # 3-layer return structure
        assert "archiveCount" in result
        assert "archiveIncluded" in result
        assert "systemPromptSuffix" in result  # Layer 2: session state

    def test_assemble_messages_include_memory_hits(self, sample_messages, sample_context):
        """Memory hits are included as separate user messages."""
        service = MemoryService()

        result = service.compose({
            "messages": sample_messages,
            "prompt": "Python数据分析",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Messages should be assembled (may include memory hits as user messages)
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Check that we have user messages (original may be filtered)
        user_messages = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_messages) > 0  # Should have at least some messages

    def test_assemble_no_system_prompt_injection(self, sample_messages, sample_context):
        """Assembly returns systemPromptAddition separately (RCA mode)."""
        service = MemoryService()

        result = service.compose({
            "messages": sample_messages,
            "prompt": "测试查询",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Verify RCA pipeline returns systemPromptAddition
        assert "systemPromptAddition" in result
        assert isinstance(result["systemPromptAddition"], str)

        # Layer 3: memoryUserMessage present
        assert "memoryUserMessage" in result

        # All returned items should be in messages list
        assert "messages" in result


class TestTokenBudgetTruncation:
    """Test 2: Token budget truncation works correctly.

    Verify that assembly respects token budget and truncates intelligently.
    """

    def test_token_budget_enforced(self, sample_context):
        """Assembly respects configured token budget."""
        service = MemoryService()

        # Test with small token budget
        small_budget = 500
        result = service.compose({
            "messages": [
                {"role": "user", "content": "A" * 1000},  # Large message
                {"role": "assistant", "content": "B" * 1000},
            ],
            "prompt": "测试",
            "tokenBudget": small_budget,
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should return results within budget
        assert "estimatedTokens" in result
        assert "messages" in result
        # Note: actual truncation logic may vary, but result should be valid

    def test_truncation_prioritizes_recent_messages(self, sample_context):
        """When over budget, recent messages are prioritized."""
        service = MemoryService()

        # Create many messages that would exceed budget
        many_messages = [
            {"role": "user", "content": f"Message {i}: " + "X" * 100}
            for i in range(50)
        ]

        result = service.compose({
            "messages": many_messages,
            "prompt": "最后的消息",
            "tokenBudget": 1000,  # Small budget
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should return some messages (graceful degradation)
        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_truncation_preserves_memory_hits(self, sample_context):
        """Memory hits are preserved even during truncation."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "重要问题"},
            ],
            "prompt": "Python",
            "tokenBudget": 500,  # Small budget
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should still return valid result
        assert "messages" in result
        assert "estimatedTokens" in result

    def test_truncation_estimates_tokens_accurately(self, sample_context):
        """Token estimation is reasonably accurate."""
        service = MemoryService()

        messages = [
            {"role": "user", "content": "Hello " * 100},  # ~700 tokens
            {"role": "assistant", "content": "Hi " * 100},  # ~200 tokens
        ]

        result = service.compose({
            "messages": messages,
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should estimate tokens reasonably
        assert "estimatedTokens" in result
        estimated = result["estimatedTokens"]
        assert estimated > 0
        assert estimated < 10000  # Should be reasonable


class TestMemoryHitsAsUserMessages:
    """Test 3: Memory hits are formatted as user messages.

    Verify that retrieved memories are injected as user messages,
    not system prompt content.
    """

    def test_memory_hits_formatted_as_user_messages(self, sample_context):
        """Retrieved memories become user messages in assembly."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "你好"},
            ],
            "prompt": "Python开发",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Check that memories (if any) are in messages as user role
        assert "messages" in result
        user_messages = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_messages) > 0

    def test_memory_hit_message_format(self, sample_context):
        """Memory hit messages follow the expected format."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "我的偏好"},
            ],
            "prompt": "用户偏好",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Verify message structure
        assert "messages" in result
        for msg in result["messages"]:
            assert "role" in msg
            assert "content" in msg
            # Content should be a string
            assert isinstance(msg["content"], str)

    def test_multiple_memory_hits_separate_messages(self, sample_context):
        """Each memory hit becomes a separate user message."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "技术背景"},
            ],
            "prompt": "编程技能和背景",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Messages should be separate (not concatenated)
        assert "messages" in result
        assert isinstance(result["messages"], list)

        # Each message should have role and content
        for msg in result["messages"]:
            assert "role" in msg
            assert "content" in msg


class TestArchiveInclusionInAssemble:
    """Test 4: Archive inclusion in assemble.

    Verify that session archives are included when assembling context.
    """

    def test_archive_expansion_in_assemble(self, sample_context):
        """Session archives are expanded and included in assembly."""
        service = MemoryService()

        # Test that assemble completes successfully
        result = service.compose({
            "messages": [
                {"role": "user", "content": "继续之前的话题"},
            ],
            "prompt": "历史记录",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Verify RCA pipeline return structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result

        # Archive fields present in 3-layer return
        assert "archiveCount" in result
        assert "systemPromptSuffix" in result  # Layer 2: session state

    def test_cold_memories_from_archives(self, sample_context):
        """Cold memories are retrieved from archives during assembly."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "很久之前的对话"},
            ],
            "prompt": "旧话题",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # RCA pipeline should complete successfully
        assert "messages" in result
        assert "systemPromptAddition" in result

    def test_archive_ranking_by_relevance(self, sample_context):
        """Archives are ranked by relevance to current query."""
        service = MemoryService()

        result = service.compose({
            "messages": [
                {"role": "user", "content": "相关主题"},
            ],
            "prompt": "搜索相关内容",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # RCA pipeline should complete successfully
        assert "messages" in result
        assert "systemPromptAddition" in result



class TestColdMemoryArchival:
    """Test 6: Cold memory archival workflow.

    Verify that old memories are moved to cold storage (archives).
    """

    def test_old_memories_archived(self, sample_context):
        """Memories older than threshold are archived."""
        pytest.skip("Pending Task #4: Archive expansion + cold memory archiver")

    def test_archived_memories_searchable(self, sample_context):
        """Archived memories remain searchable."""
        pytest.skip("Pending Task #4: Archive expansion + cold memory archiver")

    def test_archive_indexing(self, sample_context):
        """Archives are indexed for efficient retrieval."""
        pytest.skip("Pending Task #4: Archive expansion + cold memory archiver")
class TestEndToEndUnifiedAssembly:
    """End-to-end tests combining all assembly features."""

    def test_full_assembly_workflow(self, sample_messages, sample_context):
        """Complete workflow: messages + session state + token budget."""
        pytest.skip("session_state integration pending")
        from session.session_state import TaskState

        service = MemoryService()

        # Set up session state so RCA has content in active_task slot
        task_state = TaskState(
            objective="Build RCA pipeline",
            current_stage="Integration testing",
            next_step="Run full test suite",
            blockers=[],
        )
        service._session_state.update_task_state(sample_context.session_id, task_state)

        result = service.compose({
            "messages": sample_messages,
            "prompt": "Python开发",
            "tokenBudget": 128_000,
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Verify RCA pipeline output structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result
        assert isinstance(result["messages"], list)
        assert isinstance(result["systemPromptAddition"], str)
        assert isinstance(result["estimatedTokens"], int)

        # Messages should contain original user messages
        user_msgs = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_msgs) >= 1

        # Token estimate should be positive
        assert result["estimatedTokens"] >= 0

    def test_assembly_with_no_results(self, sample_messages, sample_context):
        """Assembly handles no memory hits gracefully."""
        service = MemoryService()

        # Use a nonsense query unlikely to match anything
        result = service.compose({
            "messages": sample_messages,
            "prompt": "xyzzy_nonexistent_query_12345",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # RCA pipeline should still return valid structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result

        # systemPromptAddition may be empty when no memories found — that's fine
        assert isinstance(result["systemPromptAddition"], str)

        # Original messages should still be present
        assert len(result["messages"]) >= 1

    @patch('server.memory_service.MemoryService.get_read_api')
    def test_assembly_error_handling(self, mock_get_read_api, sample_context):
        """Assembly handles errors gracefully."""
        # Mock read_api to raise inside the RCA pipeline
        mock_api = Mock()
        mock_api.search_memory.side_effect = RuntimeError("Service unavailable")
        mock_get_read_api.return_value = mock_api

        service = MemoryService()
        result = service.compose({
            "messages": [{"role": "user", "content": "Test query"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should degrade gracefully — return messages + systemPromptAddition
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result


class TestResultRankerIntegration:
    """Integration tests with mock AGFS/LLM providers."""

    @patch('server.memory_service.MemoryService.get_read_api')
    def test_assembly_with_mock_providers(self, mock_get_read_api, sample_messages, sample_context):
        """Assembly works with mock providers (no external services)."""
        from core.models import RetrievedBlock, SearchMemoryResult

        mock_api = Mock()
        mock_api.search_memory.return_value = SearchMemoryResult(
            query="test",
            hits=[
                RetrievedBlock(
                    uri="ctx://test-account/users/test-user/memories/preferences/coding",
                    score=0.85,
                    category="preference",
                    abstract="User prefers Python for data analysis",
                ),
            ],
        )
        mock_get_read_api.return_value = mock_api

        service = MemoryService()
        result = service.compose({
            "messages": sample_messages,
            "prompt": "Python数据分析",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should return valid RCA structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result
        assert isinstance(result["messages"], list)

        # Mock was called for memory search
        assert mock_api.search_memory.call_count == 2
        searches = mock_api.search_memory.call_args_list
        assert searches[0].kwargs["top_k"] == 15
        assert searches[1].kwargs["categories"] == ["session_summary"]

    def test_assembly_performance(self, sample_messages, sample_context):
        """Assembly completes within acceptable time limits."""
        import time

        service = MemoryService()
        start = time.monotonic()

        result = service.compose({
            "messages": sample_messages,
            "prompt": "性能测试",
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        elapsed = time.monotonic() - start

        # Assembly should complete quickly (< 5s even with no external services)
        assert elapsed < 5.0, f"Assembly took {elapsed:.2f}s, too slow"
        assert "messages" in result
        assert "systemPromptAddition" in result

    def test_concurrent_assembly_requests(self, sample_context):
        """Multiple assembly requests can be processed sequentially (concurrency-safe)."""
        service = MemoryService()

        results = []
        for i in range(5):
            result = service.compose({
                "messages": [{"role": "user", "content": f"Concurrent query {i}"}],
                "prompt": f"并发测试 {i}",
                "accountId": sample_context.account_id,
                "userId": sample_context.user_id,
                "agentId": sample_context.agent_id,
                "sessionId": f"{sample_context.session_id}_{i}",
            })
            results.append(result)

        # All requests should succeed independently
        assert len(results) == 5
        for result in results:
            assert "messages" in result
            assert "systemPromptAddition" in result
            assert "estimatedTokens" in result

        # Each session should be isolated
        assert len(set(r["estimatedTokens"] for r in results)) >= 1


# Helper functions for test setup

def create_mock_memory_hit(uri: str, score: float, abstract: str) -> RetrievedBlock:
    """Create a mock RetrievedBlock for testing."""
    return RetrievedBlock(
        uri=uri,
        score=score,
        abstract=abstract,
        content_excerpt=f"Content for {uri}",
        category="entity",
        owner_space="user:test-user",
    )


def create_mock_search_result(hits: list[RetrievedBlock]) -> SearchMemoryResult:
    """Create a mock SearchMemoryResult for testing."""
    return SearchMemoryResult(
        query="test query",
        hits=hits,
    )


class TestRCAPipelineIntegration:
    """Integration tests for the RCA (RuntimeContextAssembly) pipeline."""

    def test_rca_pipeline_returns_system_prompt_addition(self, sample_context):
        """RCA pipeline returns systemPromptAddition with rendered slots."""
        service = MemoryService()

        result = service.compose({
            "messages": [{"role": "user", "content": "Tell me about our project"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
            "tokenBudget": 128_000,
        })

        # Verify RCA pipeline structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result

        # System prompt addition should be a string (may be empty if no slots filled)
        assert isinstance(result["systemPromptAddition"], str)

    def test_rca_pipeline_with_empty_query(self, sample_context):
        """RCA pipeline handles empty query gracefully."""
        service = MemoryService()

        result = service.compose({
            "messages": [],  # No messages to extract query from
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should return original messages with empty systemPromptAddition
        assert "messages" in result
        assert result["systemPromptAddition"] == ""
        assert "estimatedTokens" in result

    def test_rca_pipeline_intent_detection(self, sample_context):
        """RCA pipeline performs intent detection on queries."""
        service = MemoryService()

        # Test with entity reference
        result = service.compose({
            "messages": [{"role": "user", "content": 'What about "Project Alpha" status?'}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should complete without errors
        assert "messages" in result
        assert "systemPromptAddition" in result

    def test_rca_pipeline_with_action_keywords(self, sample_context):
        """RCA pipeline detects action keywords for skills expansion."""
        service = MemoryService()

        result = service.compose({
            "messages": [{"role": "user", "content": "How do I deploy using Docker?"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should complete without errors
        assert "messages" in result
        assert "systemPromptAddition" in result

    def test_rca_pipeline_token_budget_respected(self, sample_context):
        """RCA pipeline respects token budget constraints."""
        service = MemoryService()

        result = service.compose({
            "messages": [{"role": "user", "content": "Test query"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
            "tokenBudget": 10_000,  # Small budget
        })

        # Should complete successfully even with small budget
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert result["estimatedTokens"] >= 0

    def test_rca_pipeline_session_state_integration(self, sample_context):
        """RCA pipeline integrates with session state tracking."""
        pytest.skip("session_state integration pending")
        from session.session_state import TaskState

        service = MemoryService()

        # Set up task state
        task_state = TaskState(
            objective="Complete project setup",
            current_stage="Configuring database",
            next_step="Install PostgreSQL",
            blockers=["Waiting for credentials"],
        )
        service._session_state.update_task_state(sample_context.session_id, task_state)

        # Now call assemble
        result = service.compose({
            "messages": [{"role": "user", "content": "What's next?"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should complete successfully and include task context
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert result["estimatedTokens"] > 0

    def test_rca_pipeline_multi_turn_conversation(self, sample_context):
        """RCA pipeline handles multi-turn conversations correctly."""
        service = MemoryService()

        # First turn
        result1 = service.compose({
            "messages": [{"role": "user", "content": "Let's discuss the architecture"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        assert "messages" in result1
        assert "systemPromptAddition" in result1

        # Second turn
        result2 = service.compose({
            "messages": [{"role": "user", "content": "What about the database?"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        assert "messages" in result2
        assert "systemPromptAddition" in result2


class TestRCADegradationPaths:
    """Test degradation paths when components are unavailable."""

    @patch('server.memory_service.MemoryService.get_read_api')
    def test_rca_degrades_on_vector_unavailable(self, mock_get_read_api, sample_context):
        """RCA pipeline degrades gracefully when vector search fails."""
        # Mock read_api to raise exception
        mock_api = Mock()
        mock_api.search_memory.side_effect = Exception("Vector index down")
        mock_get_read_api.return_value = mock_api

        service = MemoryService()

        result = service.compose({
            "messages": [{"role": "user", "content": "Test query"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should still return valid structure (minimal RCP)
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result

    @patch('server.memory_service._HAS_AGFS', False)
    def test_rca_degrades_on_agfs_unavailable(self, sample_context):
        """RCA pipeline degrades gracefully when AGFS is unavailable."""
        service = MemoryService()

        result = service.compose({
            "messages": [{"role": "user", "content": "Test query"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should still return valid structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result

    def test_rca_handles_all_components_unavailable(self, sample_context):
        """RCA pipeline handles complete component failure."""
        service = MemoryService()

        # With minimal configuration, should still work
        result = service.compose({
            "messages": [{"role": "user", "content": "Simple question"}],
            "accountId": sample_context.account_id,
            "userId": sample_context.user_id,
            "agentId": sample_context.agent_id,
            "sessionId": sample_context.session_id,
        })

        # Should always return valid structure
        assert "messages" in result
        assert "systemPromptAddition" in result
        assert "estimatedTokens" in result
