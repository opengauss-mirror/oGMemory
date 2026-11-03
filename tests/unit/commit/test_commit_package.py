"""Unit tests for commit package components."""

import hashlib
import json
import re
from unittest.mock import Mock, MagicMock

import pytest

from core.models import RequestContext, CandidateMemory, WritePlan, ContextNode, RelationEdge
from commit.merge_policies import (
    ProfilePolicy,
    AggregateTopicPolicy,
    AppendOnlyPolicy,
    SkillToolPolicy,
)
from commit.policy_router import PolicyRouter
from commit.archive_builder import ArchiveBuilder
from commit.candidate_pipeline import CandidatePipeline
from commit.context_writer import ContextWriter


class TestProfilePolicy:
    """Tests for ProfilePolicy."""

    def setup_method(self):
        """Set up test fixtures."""
        from core.uri_resolver import URIResolver
        from extraction.schemas.registry import SchemaRegistry

        self.mock_fs = Mock()
        self.registry = SchemaRegistry()
        self.uri_resolver = URIResolver(self.registry)
        self.policy = ProfilePolicy(self.mock_fs, self.uri_resolver)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_plan_returns_merge_for_existing_profile(self):
        """Test that existing profile returns merge action."""
        self.mock_fs.exists.return_value = True
        existing_node = ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user_space:user-123",
            abstract="Old profile",
            overview="Old overview",
            content="Old content",
            metadata={"version": 1},
        )
        self.mock_fs.read_node.return_value = existing_node

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "merge"
        assert "test-account" in plan.target_uri
        assert "user-123" in plan.target_uri
        assert plan.target_uri.endswith("/profile")

    def test_plan_returns_create_for_new_profile(self):
        """Test that new profile returns create action."""
        self.mock_fs.exists.return_value = False

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "create"
        assert "test-account" in plan.target_uri


class TestAggregateTopicPolicy:
    """Tests for AggregateTopicPolicy."""

    def setup_method(self):
        """Set up test fixtures."""
        from core.uri_resolver import URIResolver
        from extraction.schemas.registry import SchemaRegistry

        self.mock_fs = Mock()
        self.registry = SchemaRegistry()
        self.uri_resolver = URIResolver(self.registry)
        self.policy = AggregateTopicPolicy(self.mock_fs, self.uri_resolver)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_plan_returns_merge_for_existing_topic(self):
        """Test that existing topic returns merge action."""
        self.mock_fs.exists.return_value = True
        existing_node = ContextNode(
            uri="ctx://test-account/users/user-123/memories/preferences/coffee",
            context_type="MEMORY",
            category="preference",
            level=4,
            owner_space="user_space:user-123",
            abstract="Coffee preference",
            overview="Old overview",
            content="Old content",
            metadata={"version": 1},
        )
        self.mock_fs.read_node.return_value = existing_node

        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee",
            abstract="Coffee preference",
            overview="Likes dark roast",
            content="Full preference info",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "merge"
        assert "beverage" in plan.target_uri  # "coffee" normalized to canonical "beverage"

    def test_plan_returns_create_for_new_topic(self):
        """Test that new topic returns create action."""
        self.mock_fs.exists.return_value = False

        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee",
            abstract="Coffee preference",
            overview="Likes dark roast",
            content="Full preference info",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "create"

    @pytest.mark.parametrize("raw_key,stored_key", [
        ("ProjectAtlas", "projectatlas"),
        ("上下文图谱", "auto_" + hashlib.sha256("上下文图谱".encode()).hexdigest()[:16]),
    ])
    def test_existing_topic_reuses_legacy_uri(self, raw_key, stored_key):
        from core.validation import validate_candidate

        uri = f"ctx://test-account/users/user-123/memories/entities/{stored_key}"
        self.mock_fs.exists.side_effect = lambda target, _ctx: target == uri
        self.mock_fs.read_node.return_value = ContextNode(
            uri=uri, context_type="MEMORY", category="entity", level=4,
            owner_space="user_space:user-123", abstract=raw_key,
            overview="Old overview", content="Old content", metadata={"version": 1},
        )
        candidate = validate_candidate(CandidateMemory(
            category="entity", owner_scope="user", routing_key=raw_key,
            abstract=raw_key, overview="New overview", content="New content",
            confidence=0.9,
        ))

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "merge"
        assert plan.target_uri == uri


class TestAppendOnlyPolicy:
    """Tests for AppendOnlyPolicy."""

    def setup_method(self):
        """Set up test fixtures."""
        from core.uri_resolver import URIResolver
        from extraction.schemas.registry import SchemaRegistry

        self.mock_fs = Mock()
        self.registry = SchemaRegistry()
        self.uri_resolver = URIResolver(self.registry)
        self.policy = AppendOnlyPolicy(self.mock_fs, self.uri_resolver)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_plan_always_returns_create(self):
        """Test that events always return create action."""
        candidate = CandidateMemory(
            category="event",
            owner_scope="user",
            routing_key="visit_20250315",
            abstract="Coffee shop visit",
            overview="Visited downtown coffee shop",
            content="Full event details",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "create"
        # Should include generated event ID
        assert "test-account" in plan.target_uri

    def test_event_uri_appends_single_storage_uniqueness_suffix(self):
        """Event URI adds exactly one storage-side uniqueness suffix."""
        candidate = CandidateMemory(
            category="event",
            owner_scope="user",
            routing_key="visit_20250315",
            abstract="Coffee shop visit",
            overview="Visited downtown coffee shop",
            content="Full event details",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert re.search(r"/visit_20250315_\d{14}_[0-9a-f]{8}$", plan.target_uri)


class TestSkillToolPolicy:
    """Tests for SkillToolPolicy."""

    def setup_method(self):
        """Set up test fixtures."""
        from core.uri_resolver import URIResolver
        from extraction.schemas.registry import SchemaRegistry

        self.mock_fs = Mock()
        self.registry = SchemaRegistry()
        self.uri_resolver = URIResolver(self.registry)
        self.policy = SkillToolPolicy(self.mock_fs, self.uri_resolver)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_plan_returns_merge_for_existing_skill(self):
        """Test that existing skill returns merge action."""
        self.mock_fs.exists.return_value = True

        # Mock read_node to return existing skill node
        from core.models import ContextNode
        existing_node = ContextNode(
            uri="ctx://test-account/agents/agent-456/skills/generated/code_review",
            context_type="SKILL",
            category="skill",
            level=4,
            owner_space="agent_space:agent-456",
            abstract="Code review skill",
            overview="Best practices for code review",
            content="Existing skill documentation",
            metadata={"version": 1, "usage_count": 5},
        )
        self.mock_fs.read_node.return_value = existing_node

        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Best practices for code review",
            content="Full skill documentation",
            confidence=0.9,
        )

        plan = self.policy.plan(candidate, self.ctx)

        assert plan.action == "merge"
        assert "code_review" in plan.target_uri
        assert "skills" in plan.target_uri


class TestPolicyRouter:
    """Tests for PolicyRouter."""

    def setup_method(self):
        """Set up test fixtures."""
        from extraction.schemas.registry import SchemaRegistry

        self.mock_fs = Mock()
        self.registry = SchemaRegistry()
        self.router = PolicyRouter(self.mock_fs, registry=self.registry)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_route_returns_correct_policy(self):
        """Test that correct policy is returned for each category."""
        from commit.merge_policies import ProfilePolicy, AggregateTopicPolicy

        profile_policy = self.router.route(CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="",
            overview="",
            content="",
            confidence=0.9,
        ))
        assert isinstance(profile_policy, ProfilePolicy)  # Profile uses ProfilePolicy

        pref_policy = self.router.route(CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee",
            abstract="",
            overview="",
            content="",
            confidence=0.9,
        ))
        assert isinstance(pref_policy, AggregateTopicPolicy)

    def test_route_returns_none_for_unknown_category(self):
        """Test that unknown category returns None."""
        policy = self.router.route(CandidateMemory(
            category="unknown",
            owner_scope="user",
            routing_key="test",
            abstract="",
            overview="",
            content="",
            confidence=0.9,
        ))
        assert policy is None

    def test_plan_raises_for_unknown_category(self):
        """Test that plan() raises ValueError for unknown category."""
        with pytest.raises(ValueError, match="No policy found for category"):
            self.router.plan(CandidateMemory(
                category="unknown",
                owner_scope="user",
                routing_key="test",
                abstract="",
                overview="",
                content="",
                confidence=0.9,
            ), self.ctx)


class TestArchiveBuilder:
    """Tests for ArchiveBuilder."""

    def setup_method(self):
        """Set up test fixtures."""
        from providers.llm import MockLLM
        self.mock_llm = MockLLM()
        self.builder = ArchiveBuilder(llm=self.mock_llm)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_build_returns_context_node(self):
        """Test that build returns a valid ContextNode."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = self.builder.build(candidate, plan, self.ctx)

        assert isinstance(node, ContextNode)
        assert node.uri == plan.target_uri
        assert node.category == "profile"
        assert node.abstract == candidate.abstract


class TestArchiveBuilderLLMMerge:
    """Tests for ArchiveBuilder LLM semantic merge."""

    def setup_method(self):
        """Set up test fixtures."""
        from providers.llm import MockLLM
        from unittest.mock import Mock
        self.mock_llm_base = MockLLM()
        self.mock_llm = Mock(wraps=self.mock_llm_base)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_skill_merge_calls_llm_complete_json(self):
        """ArchiveBuilder with LLM calls complete_json for semantic merge."""
        # Mock LLM to return merged content
        self.mock_llm.complete_json.return_value = {"content": "merged content"}

        builder = ArchiveBuilder(llm=self.mock_llm)

        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Best practices",
            content="New skill content",
            confidence=0.9,
        )

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/agents/agent-456/skills/generated/code_review",
            merged_fields={
                "skill_merge": True,
                "existing_content": "Existing content",
                "new_content": "New content",
            },
            relation_edges=[],
        )

        node = builder.build(candidate, plan, self.ctx)

        # Verify result contains merged content
        assert node.content == "merged content"

    def test_skill_merge_empty_existing_returns_new(self):
        """ArchiveBuilder returns new content when existing_content is empty."""
        builder = ArchiveBuilder(llm=self.mock_llm)

        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Best practices",
            content="New content",  # This will be used by merged_fields.get("new_content", candidate.content)
            confidence=0.9,
        )

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/agents/agent-456/skills/generated/code_review",
            merged_fields={
                "skill_merge": True,
                "existing_content": "",  # Empty existing
                "new_content": "New content",
            },
            relation_edges=[],
        )

        node = builder.build(candidate, plan, self.ctx)

        # Should return new_content without calling LLM
        assert node.content == "New content"


class TestCandidatePipeline:
    """Tests for CandidatePipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_extractor = Mock()
        self.mock_extractor.extract.return_value = [
            CandidateMemory(
                category="profile",
                owner_scope="user",
                routing_key="user_profile",
                abstract="",
                overview="",
                content="",
                confidence=0.9,
            )
        ]
        self.pipeline = CandidatePipeline([self.mock_extractor])
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_extract_calls_extractor(self):
        """Test that extract calls the configured extractors."""
        messages = [{"role": "user", "content": "test"}]
        candidates = self.pipeline.extract(messages, self.ctx)

        # Mock extractor may return empty list or candidates
        # Just verify it was called
        self.mock_extractor.extract.assert_called_once()

    @pytest.mark.parametrize("parallel", [False, True])
    def test_extractor_failure_is_not_successful_empty_extraction(self, parallel):
        error = ValueError("No valid JSON found in LLM output")
        self.mock_extractor.extract.side_effect = error
        with pytest.raises(ValueError) as caught:
            self.pipeline.extract(
                [{"role": "user", "content": "Remember my birthday"}],
                self.ctx, parallel=parallel,
            )
        assert caught.value is error

    @pytest.mark.parametrize("parallel", [False, True])
    def test_partial_extraction_failure_is_not_success(self, parallel):
        successful = Mock()
        successful.extract.return_value = []
        self.mock_extractor.extract.side_effect = RuntimeError("model failed")
        pipeline = CandidatePipeline([successful, self.mock_extractor])
        with pytest.raises(RuntimeError, match="model failed"):
            pipeline.extract([], self.ctx, parallel=parallel)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_legitimate_empty_extraction_remains_success(self, parallel):
        self.mock_extractor.extract.return_value = []
        assert self.pipeline.extract([], self.ctx, parallel=parallel) == []
        assert CandidatePipeline([]).extract([], self.ctx, parallel=parallel) == []

    def test_parallel_extraction_preserves_usage_context(self):
        from contextvars import ContextVar

        owner = ContextVar("test_extraction_owner", default=None)
        observed = []
        self.mock_extractor.extract.side_effect = lambda *args: (
            observed.append(owner.get()) or []
        )
        token = owner.set("test-account")
        try:
            assert self.pipeline.extract([], self.ctx) == []
        finally:
            owner.reset(token)
        assert observed == ["test-account"]

    def test_extract_uses_stable_extractor_contract(self):
        candidates = self.pipeline.extract(
            [{"role": "user", "content": "test"}],
            self.ctx,
            parallel=False,
        )

        assert len(candidates) == 1
        args, kwargs = self.mock_extractor.extract.call_args
        assert len(args) == 6
        assert kwargs == {}

    @pytest.mark.parametrize("parallel", [False, True])
    def test_extract_retains_candidate_with_unresolved_name(self, parallel):
        self.mock_extractor.extract.return_value = [CandidateMemory(
            category="preference", owner_scope="user", routing_key="_md",
            abstract="Keep diagrams in Markdown", overview="", content="",
            confidence=0.9,
        )]

        candidates = self.pipeline.extract(
            [{"role": "user", "content": "Keep diagrams in Markdown"}],
            self.ctx, parallel=parallel,
        )

        assert len(candidates) == 1
        assert candidates[0].routing_key == "_md"

    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        candidates = [
            CandidateMemory(
                category="profile",
                owner_scope="user",
                routing_key="profile",
                abstract="",
                overview="",
                content="",
                confidence=0.3,
            ),
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",
                abstract="",
                overview="",
                content="",
                confidence=0.8,
            ),
        ]

        filtered = self.pipeline.filter_by_confidence(candidates, threshold=0.5)

        assert len(filtered) == 1
        assert filtered[0].category == "preference"


class TestContextWriter:
    """Tests for ContextWriter."""

    def setup_method(self):
        """Set up test fixtures."""
        from providers.llm import MockLLM
        from extraction.schemas.registry import SchemaRegistry
        from commit.policy_router import PolicyRouter

        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.registry = SchemaRegistry()
        self.policy_router = PolicyRouter(self.mock_fs, registry=self.registry)
        self.writer = ContextWriter(self.mock_fs, llm=self.mock_llm, policy_router=self.policy_router)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_write_candidate_calls_fs_write_node(self):
        """Test that write_candidate calls ContextFS.write_node."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock exists to return False (create new)
        self.mock_fs.exists.return_value = False

        plan = self.writer.write_candidate(candidate, self.ctx)

        assert plan.action == "create"
        # Verify write_node was called
        self.mock_fs.write_node.assert_called_once()

    def test_write_candidate_triggers_directory_event_for_preference(self):
        """Test that directory event is triggered for non-profile categories."""
        from unittest.mock import Mock as MockMock
        from providers.llm import MockLLM
        from extraction.schemas.registry import SchemaRegistry
        from commit.policy_router import PolicyRouter

        mock_outbox = MockMock(spec=['register_write', 'register_directory'])
        registry = SchemaRegistry()
        policy_router = PolicyRouter(self.mock_fs, registry=registry)
        writer = ContextWriter(
            self.mock_fs,
            llm=MockLLM(),
            policy_router=policy_router,
            outbox_store=mock_outbox,
        )

        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee",
            abstract="Likes coffee",
            overview="",
            content="",
            confidence=0.9,
        )

        self.mock_fs.exists.return_value = False

        plan = writer.write_candidate(candidate, self.ctx)

        assert plan.action == "create"
        # Verify both register_write and register_directory were called
        mock_outbox.register_write.assert_called_once()
        mock_outbox.register_directory.assert_called_once()

    def test_write_candidate_skips_directory_event_for_profile(self):
        """Test that directory event is skipped for profile category."""
        from unittest.mock import Mock as MockMock
        from providers.llm import MockLLM
        from extraction.schemas.registry import SchemaRegistry
        from commit.policy_router import PolicyRouter

        mock_outbox = MockMock(spec=['register_write', 'register_directory'])
        registry = SchemaRegistry()
        policy_router = PolicyRouter(self.mock_fs, registry=registry)
        writer = ContextWriter(
            self.mock_fs,
            llm=MockLLM(),
            policy_router=policy_router,
            outbox_store=mock_outbox,
        )

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="",
            content="",
            confidence=0.9,
        )

        self.mock_fs.exists.return_value = False

        plan = writer.write_candidate(candidate, self.ctx)

        assert plan.action == "create"
        # register_write should be called, but not register_directory
        mock_outbox.register_write.assert_called_once()
        mock_outbox.register_directory.assert_not_called()
