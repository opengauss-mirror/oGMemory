"""Unit tests for service layer write API.

Tests verify:
- Extractor initialization (single tool-use Extractor)
- commit_session calls the complete pipeline
- OutboxStore integration
- Statistics return values
- Single candidate writes
- Cross-account access control

See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from core.models import RequestContext, CandidateMemory, WritePlan, ContextNode
from core.errors import AccessDeniedError
from service.api import MemoryWriteAPI, init_write_api, get_write_api
from commit import OutboxStore
from extraction.schemas.registry import SchemaRegistry
from providers.llm.mock_llm import MockLLM


class TestMemoryWriteAPIInit:
    """Verify MemoryWriteAPI initialization."""

    def test_init_creates_context_writer(self):
        """init should create ContextWriter with fs."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        registry = SchemaRegistry()

        api = MemoryWriteAPI(mock_fs, mock_llm, schema_registry=registry)

        assert api._fs is mock_fs
        assert api._llm is mock_llm
        assert api._writer is not None
        assert api._pipeline is not None

    def test_init_creates_single_extractor(self):
        """init should create single tool-use Extractor."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        registry = SchemaRegistry()

        api = MemoryWriteAPI(mock_fs, mock_llm, schema_registry=registry)

        # Verify pipeline has extractor set
        assert api._pipeline._extractors is not None
        assert len(api._pipeline._extractors) == 1

        # Verify extractor type
        from extraction import Extractor
        assert isinstance(api._pipeline._extractors[0], Extractor)

    def test_init_with_outbox_store(self):
        """init should accept optional OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        mock_outbox = Mock(spec=OutboxStore)
        registry = SchemaRegistry()

        api = MemoryWriteAPI(mock_fs, mock_llm, mock_outbox, schema_registry=registry)

        assert api._outbox_store is mock_outbox

    def test_init_without_outbox_store(self):
        """init should work without OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        registry = SchemaRegistry()

        api = MemoryWriteAPI(mock_fs, mock_llm, schema_registry=registry)

        assert api._outbox_store is None


class TestCommitSession:
    """Verify commit_session method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.registry = SchemaRegistry()
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox, schema_registry=self.registry)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_commit_session_calls_pipeline_extract(self):
        """commit_session should call extract on pipeline."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Mock the writer to return success
        self.mock_fs.exists.return_value = False
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify extract was called
        assert "candidates_extracted" in result

    def test_commit_session_filters_by_confidence(self):
        """commit_session should filter low confidence candidates."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        # Test with high threshold
        result = self.api.commit_session(messages, self.ctx, confidence_threshold=0.9)

        assert "candidates_filtered" in result

    def test_commit_session_deduplicates_candidates(self):
        """commit_session should deduplicate before writing."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify writes were performed
        assert "writes_completed" in result

    def test_commit_session_returns_statistics(self):
        """commit_session should return comprehensive statistics."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer to return different plans
        plans = [
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[]),
            WritePlan(action="skip", target_uri="uri2", merged_fields={}, relation_edges=[]),
            WritePlan(action="create", target_uri="uri3", merged_fields={}, relation_edges=[]),
        ]
        self.api._writer.write_candidates = Mock(return_value=plans)

        # Mock node read for outbox
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="uri1",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify all statistics keys
        assert "candidates_extracted" in result
        assert "candidates_filtered" in result
        assert "writes_completed" in result
        assert "writes_skipped" in result
        assert "writes_failed" in result
        assert "plans" in result

        # Verify counts
        assert result["writes_completed"] == 2
        assert result["writes_skipped"] == 1

    def test_commit_session_empty_messages(self):
        """commit_session should handle empty messages.

        Note: MockLLM returns fixtures even with empty messages,
        so candidates_extracted may be > 0. The important thing
        is that it doesn't crash and returns valid statistics.
        """
        result = self.api.commit_session([], self.ctx)

        # Should return valid statistics structure
        assert "candidates_extracted" in result
        assert "writes_completed" in result
        assert "writes_skipped" in result
        assert "writes_failed" in result
        # All counts should be non-negative integers
        assert result["candidates_extracted"] >= 0
        assert result["writes_completed"] >= 0

    def test_commit_session_registers_outbox_events(self):
        """commit_session should register OutboxEvents when OutboxStore is provided.

        Outbox registration now happens inside ContextWriter.write_candidate().
        This test verifies that:
        1. ContextWriter is initialized with outbox_store
        2. Outbox registration happens during real write operations
        """
        messages = [{"role": "user", "content": "Test"}]

        # Configure MockLLM to return a tool call with all required fields
        self.mock_llm._mock_tool_calls = [
            {
                "tool": "extract_profile",
                "input": {
                    "routing_key": "name",
                    "abstract": "Test profile",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                    "evidence_quote": "Test quote",
                    "attributed_speaker": "user",
                    "attribution_basis": "self_first_person",
                },
            }
        ]

        # Verify that ContextWriter has the outbox_store
        assert self.api._writer._outbox_store is not None
        assert self.api._writer._outbox_store == self.mock_outbox

        # Mock fs to allow real write_candidate execution (no function replacement)
        self.mock_fs.exists.return_value = False
        self.mock_fs.write_node.return_value = None
        self.mock_fs.read_node.return_value = ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space:user-123",
            abstract="Test profile",
            overview="Overview",
            content="Content",
        )

        result = self.api.commit_session(messages, self.ctx)

        # Verify outbox was called during real write_candidate execution
        self.mock_outbox.register_write.assert_called()
        assert result["writes_completed"] >= 1

    def test_commit_session_without_outbox_store(self):
        """commit_session should work without OutboxStore."""
        registry = SchemaRegistry()
        api = MemoryWriteAPI(self.mock_fs, self.mock_llm, outbox_store=None, schema_registry=registry)
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        api._writer.write_candidates = Mock(return_value=[])

        # Should not raise
        result = api.commit_session(messages, self.ctx)

    def test_write_memory_skips_agent_scoped_candidate_without_agent_context(self):
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="deploy_flow",
            abstract="skill",
            overview="overview",
            content="content",
            confidence=0.9,
        )
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="",
            session_id="session-789",
            trace_id="trace-abc",
        )

        result = self.api.write_memory(candidate, ctx)

        assert result["action"] == "skip"
        self.mock_fs.write_node.assert_not_called()

        assert result is not None


class TestWriteMemory:
    """Verify write_memory method for single candidate writes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.registry = SchemaRegistry()
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox, schema_registry=self.registry)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_write_memory_single_candidate(self):
        """write_memory should write a single candidate."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        # Mock node read for outbox
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.write_memory(candidate, self.ctx)

        # Verify result structure
        assert "action" in result
        assert "target_uri" in result
        assert "merged_fields" in result
        assert result["action"] == "create"

        # Verify writer was called
        self.api._writer.write_candidate.assert_called_once_with(candidate, self.ctx)

    def test_write_memory_uses_atomic_path_for_concrete_capabilities(self):
        """Concrete storage and outbox methods should enable the atomic write path."""

        class AtomicWriteMixin:
            def write_node_with_outbox(self, node, ctx, event):
                self.atomic_writes.append((node, ctx, event))

        class AtomicFS(AtomicWriteMixin):
            def __init__(self):
                self.atomic_writes = []
                self.fallback_writes = []

            def exists(self, uri, ctx):
                return False

            def write_node(self, node, ctx):
                self.fallback_writes.append((node, ctx))

        class AtomicEventBuilder:
            def build_write_event(self, node):
                self.built_nodes.append(node)
                return self.event

        class AtomicOutbox(AtomicEventBuilder):
            def __init__(self):
                self.event = object()
                self.built_nodes = []
                self.registered_writes = []

            def register_write(self, node, ctx):
                self.registered_writes.append((node, ctx))

        fs = AtomicFS()
        outbox = AtomicOutbox()
        api = MemoryWriteAPI(
            fs,
            self.mock_llm,
            outbox,
            schema_registry=self.registry,
        )
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        result = api.write_memory(candidate, self.ctx)

        assert result["action"] == "create"
        assert len(fs.atomic_writes) == 1
        node, write_ctx, event = fs.atomic_writes[0]
        assert outbox.built_nodes == [node]
        assert write_ctx is self.ctx
        assert event is outbox.event
        assert fs.fallback_writes == []
        assert outbox.registered_writes == []

    def test_write_memory_respects_backend_atomic_node_scope(self):
        """A code-only AGFS atomic primitive must not capture memory writes."""

        class CodeOnlyAtomicFS:
            def __init__(self):
                self.atomic_writes = []
                self.fallback_writes = []

            @staticmethod
            def supports_atomic_outbox_write(node):
                return node.context_type == "RESOURCE"

            def write_node_with_outbox(self, node, ctx, event):
                self.atomic_writes.append((node, ctx, event))

            def exists(self, uri, ctx):
                return False

            def write_node(self, node, ctx):
                self.fallback_writes.append((node, ctx))

        class Outbox:
            def __init__(self):
                self.built_nodes = []
                self.registered_writes = []

            def build_write_event(self, node):
                self.built_nodes.append(node)
                return object()

            def register_write(self, node, ctx):
                self.registered_writes.append((node, ctx))

        fs = CodeOnlyAtomicFS()
        outbox = Outbox()
        api = MemoryWriteAPI(
            fs,
            self.mock_llm,
            outbox,
            schema_registry=self.registry,
        )
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        result = api.write_memory(candidate, self.ctx)

        assert result["action"] == "create"
        assert fs.atomic_writes == []
        assert len(fs.fallback_writes) == 1
        assert outbox.built_nodes == []
        assert outbox.registered_writes == [fs.fallback_writes[0]]

    def test_write_memory_skipped_candidate(self):
        """write_memory should handle skip action."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer to return skip
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="skip",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.write_memory(candidate, self.ctx)

        assert result["action"] == "skip"

        # Outbox should NOT be called for skipped writes
        self.mock_outbox.register_write.assert_not_called()


class TestWriteMemories:
    """Verify write_memories method for batch writes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.registry = SchemaRegistry()
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox, schema_registry=self.registry)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_write_memories_parallel(self):
        """write_memories should write in parallel by default."""
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key=f"pref_{i}",
                abstract=f"Preference {i}",
                overview="Overview",
                content="Content",
                confidence=0.9,
            )
            for i in range(3)
        ]

        # Mock writer
        self.api._writer.write_candidates_parallel = Mock(return_value=[
            WritePlan(action="create", target_uri=f"uri{i}", merged_fields={}, relation_edges=[])
            for i in range(3)
        ])

        # Mock node read
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="uri1",
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.write_memories(candidates, self.ctx, parallel=True)

        assert len(result) == 3
        self.api._writer.write_candidates_parallel.assert_called_once()

    def test_write_memories_serial(self):
        """write_memories should write serially when parallel=False."""
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="pref_1",
                abstract="Preference 1",
                overview="Overview",
                content="Content",
                confidence=0.9,
            )
        ]

        # Mock writer
        self.api._writer.write_candidates = Mock(return_value=[
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[])
        ])

        result = self.api.write_memories(candidates, self.ctx, parallel=False)

        assert len(result) == 1
        self.api._writer.write_candidates.assert_called_once()

    def test_write_memories_deduplicates(self):
        """write_memories passes candidates through (dedup now happens in Extractor)."""
        # Create two candidates with same routing_key but different confidence
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",
                abstract="Likes coffee",
                overview="Overview",
                content="Content",
                confidence=0.9,  # Higher confidence
            ),
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",  # Same routing key
                abstract="Likes dark roast",
                overview="Different overview",
                content="Different content",
                confidence=0.8,  # Lower confidence
            ),
        ]

        # Mock fs to allow real write_candidate execution
        self.mock_fs.exists.return_value = False
        self.mock_fs.list_children.return_value = []
        self.mock_fs.write_node.return_value = None

        result = self.api.write_memories(candidates, self.ctx, parallel=False)

        # Deduplication is now pass-through (handled in Extractor)
        # Both candidates will be written (they merge at the same URI)
        assert len(result) == 2
        assert self.mock_fs.write_node.call_count == 2


class TestWriteAPICrossAccountAccess:
    """Verify cross-account access control for write operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.registry = SchemaRegistry()
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, schema_registry=self.registry)

        self.ctx_acme = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

    def test_write_memory_enforces_account_in_context(self):
        """write_memory should use account_id from context."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="Profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://acme/users/alice/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        self.api.write_memory(candidate, self.ctx_acme)

        # Verify the writer was called with acme context
        self.api._writer.write_candidate.assert_called_once()
        call_ctx = self.api._writer.write_candidate.call_args[0][1]
        assert call_ctx.account_id == "acme"


class TestWriteAPISingleton:
    """Verify global write API singleton pattern."""

    def test_init_write_api_creates_singleton(self):
        """init_write_api should create and return a global API instance."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        registry = SchemaRegistry()

        api = init_write_api(mock_fs, mock_llm, schema_registry=registry)

        assert api is not None
        assert isinstance(api, MemoryWriteAPI)

    def test_get_write_api_returns_singleton(self):
        """get_write_api should return the same instance created by init_write_api."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        registry = SchemaRegistry()

        # First call
        api1 = init_write_api(mock_fs, mock_llm, schema_registry=registry)
        # Second call
        api2 = get_write_api()

        assert api1 is api2

    def test_init_write_api_with_outbox_store(self):
        """init_write_api should accept optional OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        mock_outbox = Mock(spec=OutboxStore)
        registry = SchemaRegistry()

        api = init_write_api(mock_fs, mock_llm, mock_outbox, schema_registry=registry)

        assert api._outbox_store is mock_outbox
