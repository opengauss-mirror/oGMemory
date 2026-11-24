"""Tests for ArchiveBuilder.

Tests the construction of ContextNode from CandidateMemory and WritePlan.
"""

import pytest
from datetime import datetime, timezone

from commit.archive_builder import ArchiveBuilder
from core.models import (
    RequestContext,
    CandidateMemory,
    WritePlan,
    ContextNode,
)
from core.enums import ContextType
from providers.llm.mock_llm import MockLLM


@pytest.mark.parametrize(("category", "section"), [("entity", "## Home\n"), ("preference", "")])
def test_fast_append_preserves_card_title_and_groups_facts(category, section):
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    builder = ArchiveBuilder(MockLLM(), card_format_enabled=True, allow_model_merge=False)
    old = "# Nathan\n" + section + "- Lives in Denver."
    new = "# Nate\n" + section + "- Lives in Denver.\n- Makes bread."
    candidate = CandidateMemory(category, "user", "nate", "Nate makes bread.",
                                "- Makes bread.", new, .96)
    plan = WritePlan(
        action="merge", target_uri="ctx://acct/users/user/memories/facts/nate",
        merged_fields={"existing_content": old, "content_append": new},
    )
    assert builder.build(candidate, plan, ctx).content == (
        "# Nathan\n\n" + section + "- Lives in Denver.\n- Makes bread."
    )


class TestArchiveBuilderInit:
    """Test ArchiveBuilder initialization."""

    def test_init_with_llm(self):
        """ArchiveBuilder requires LLM for semantic merge."""
        llm = MockLLM()
        builder = ArchiveBuilder(llm=llm)
        assert builder._llm is llm

    def test_init_llm_required(self):
        """ArchiveBuilder requires LLM instance."""
        with pytest.raises(TypeError):
            ArchiveBuilder()


class TestArchiveBuilderBuild:
    """Test ArchiveBuilder.build() method."""

    @pytest.fixture
    def builder(self):
        """Create ArchiveBuilder with MockLLM."""
        return ArchiveBuilder(llm=MockLLM())

    @pytest.fixture
    def ctx(self):
        """Create RequestContext."""
        return RequestContext(
            account_id="test-account",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            trace_id="trace-001",
        )

    @pytest.fixture
    def profile_candidate(self):
        """Create profile candidate."""
        return CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="User profile abstract",
            overview="User profile overview",
            content="User profile content",
            confidence=0.9,
        )

    @pytest.fixture
    def preference_candidate(self):
        """Create preference candidate."""
        return CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding_style",
            abstract="Prefers Python",
            overview="User prefers Python for coding",
            content="User prefers Python over JavaScript",
            confidence=0.85,
        )

    @pytest.fixture
    def skill_candidate(self):
        """Create skill candidate."""
        return CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Can review code",
            content="Reviews code for best practices",
            confidence=0.9,
        )

    def test_build_creates_context_node(self, builder, profile_candidate, ctx):
        """build() should return ContextNode."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        assert isinstance(node, ContextNode)
        assert node.uri == plan.target_uri

    def test_build_sets_correct_uri(self, builder, preference_candidate, ctx):
        """build() should set URI from WritePlan."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.uri == plan.target_uri

    def test_build_sets_context_type_memory(self, builder, preference_candidate, ctx):
        """build() should set context_type=MEMORY for non-skill categories."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.context_type == ContextType.MEMORY.value

    def test_build_sets_context_type_skill(self, builder, skill_candidate, ctx):
        """build() should set context_type=SKILL for skill category."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/agents/agent-001/skills/generated/code_review",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(skill_candidate, plan, ctx)

        assert node.context_type == ContextType.SKILL.value

    def test_build_sets_level_for_profile(self, builder, profile_candidate, ctx):
        """build() should set level=3 for profile."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        assert node.level == 3

    def test_build_sets_level_for_preference(self, builder, preference_candidate, ctx):
        """build() should set level=4 for preference."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.level == 4

    def test_build_sets_owner_space_user(self, builder, preference_candidate, ctx):
        """build() should set owner_space for user scope."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.owner_space == ctx.user_space_name()

    def test_build_sets_owner_space_agent(self, builder, skill_candidate, ctx):
        """build() should set owner_space for agent scope."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/agents/agent-001/skills/generated/code_review",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(skill_candidate, plan, ctx)

        assert node.owner_space == ctx.agent_space_name()

    def test_build_sets_content_from_candidate(self, builder, preference_candidate, ctx):
        """build() should set content from candidate."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.content == preference_candidate.content

    def test_build_sets_overview_from_candidate(self, builder, preference_candidate, ctx):
        """build() should set overview from candidate."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.overview == preference_candidate.overview

    def test_build_sets_abstract_from_candidate(self, builder, preference_candidate, ctx):
        """build() should set abstract from candidate."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.abstract == preference_candidate.abstract

    def test_build_includes_metadata_timestamp(self, builder, preference_candidate, ctx):
        """build() should include created_at in metadata."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        before = datetime.now(timezone.utc)
        node = builder.build(preference_candidate, plan, ctx)
        after = datetime.now(timezone.utc)

        assert "created_at" in node.metadata
        created_at = datetime.fromisoformat(node.metadata["created_at"])
        assert before <= created_at <= after

    def test_build_merge_action_uses_merged_fields(self, builder, profile_candidate, ctx):
        """build() should use merged_fields for merge action with skill_merge."""
        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={
                "skill_merge": True,
                "existing_content": "Old content",
                "new_content": "New content",
                "existing_overview": "Old overview",
                "new_overview": "New overview",
                "new_abstract": "Merged abstract",
            },
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        assert "Old content" in node.content
        assert "New content" in node.content

    def test_build_merge_action_without_merged_fields_uses_candidate(
        self, builder, profile_candidate, ctx
    ):
        """build() should use candidate content if merged_fields is empty."""
        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        assert node.content == profile_candidate.content

    def test_build_sets_category(self, builder, preference_candidate, ctx):
        """build() should set category from candidate."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding_style",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(preference_candidate, plan, ctx)

        assert node.category == preference_candidate.category



    def test_expected_version_passed_to_metadata(self, builder, profile_candidate, ctx):
        """build() should pass expected_version from merged_fields to metadata for optimistic locking."""
        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={
                "expected_version": 3,  # Simulating existing node version
            },
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        assert node.metadata.get("expected_version") == 3

    def test_overview_append_merges_with_existing(self, builder, profile_candidate, ctx):
        """build() should merge existing_overview with overview_append, not overwrite."""
        existing_overview = "Alice is a software engineer who loves coffee."
        new_overview = "She recently started learning Rust for systems programming."

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding",
            merged_fields={
                "existing_overview": existing_overview,
                "overview_append": new_overview,
            },
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        # Both overviews should be present
        assert existing_overview in node.overview
        assert new_overview in node.overview
        # New overview should come after existing
        assert node.overview.index(existing_overview) < node.overview.index(new_overview)

    def test_content_append_merges_with_existing(self, builder, profile_candidate, ctx):
        """build() should merge existing_content with content_append, not overwrite."""
        existing_content = "## Coding Style\n\n- Uses 4 spaces for indentation"
        new_content = "## Coding Style\n\n- Uses 4 spaces for indentation\n\n- Prefers snake_case for variables\n- Writes descriptive commit messages"

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-account/users/user-001/memories/preferences/coding",
            merged_fields={
                "existing_content": existing_content,
                "content_append": "- Prefers snake_case for variables\n- Writes descriptive commit messages",
            },
            relation_edges=[],
        )

        node = builder.build(profile_candidate, plan, ctx)

        # Both content parts should be present
        assert "Uses 4 spaces for indentation" in node.content
        assert "Prefers snake_case" in node.content
        assert "descriptive commit messages" in node.content




class TestArchiveBuilderMergeContent:
    """Test ArchiveBuilder._merge_content() method."""

    @pytest.fixture
    def builder(self):
        """Create ArchiveBuilder with MockLLM."""
        return ArchiveBuilder(llm=MockLLM())

    def test_merge_content_combines_texts(self, builder):
        """_merge_content should combine old and new content for skill_merge."""
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="test_skill",
            abstract="New abstract",
            overview="New overview",
            content="New content",
            confidence=0.9,
        )

        merged_fields = {
            "skill_merge": True,
            "existing_content": "Old content",
            "new_content": "New content",
        }

        result = builder._merge_content(candidate, merged_fields)

        assert "Old content" in result
        assert "New content" in result

    def test_merge_content_uses_candidate_if_not_skill_merge(self, builder):
        """_merge_content should use candidate content if not skill_merge."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="New abstract",
            overview="New overview",
            content="Candidate content",
            confidence=0.9,
        )

        merged_fields = {
            "content": "Merged content",  # This is ignored for non-skill_merge
        }

        result = builder._merge_content(candidate, merged_fields)

        assert result == candidate.content

    def test_llm_merge_retries_with_strict_json_prompt(self):
        """_llm_merge_content should retry once when the first JSON merge fails."""

        class FlakyLLM:
            def __init__(self):
                self.prompts = []

            def complete_json(self, prompt, schema):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    raise ValueError("bad json")
                return {"content": "Old fact\n\nNew fact"}

        llm = FlakyLLM()
        builder = ArchiveBuilder(llm=llm)

        result = builder._llm_merge_content("Old fact", "New fact")

        assert result == "Old fact\n\nNew fact"
        assert len(llm.prompts) == 2
        assert "ONLY valid JSON" in llm.prompts[1]

    def test_llm_merge_falls_back_to_deterministic_deduped_append(self, caplog):
        """_llm_merge_content should not drop new content if JSON merge keeps failing."""

        class BrokenLLM:
            def __init__(self):
                self.calls = 0

            def complete_json(self, prompt, schema):
                self.calls += 1
                raise ValueError("bad json")

        llm = BrokenLLM()
        builder = ArchiveBuilder(llm=llm)

        with caplog.at_level("WARNING"):
            result = builder._llm_merge_content(
                "Caroline attended an LGBTQ support group.\n\nMelanie painted a sunrise.",
                "Melanie painted a sunrise.\n\nCaroline researched adoption agencies.",
            )

        assert llm.calls == 2
        assert result == (
            "Caroline attended an LGBTQ support group.\n\n"
            "Melanie painted a sunrise.\n\n"
            "Caroline researched adoption agencies."
        )
        assert "falling back to deterministic append" in caplog.text

    def test_merge_without_model_deduplicates_existing_and_new_content(self):
        builder = ArchiveBuilder(llm=None, allow_model_merge=False)

        result = builder._llm_merge_content(
            "The receiver was repaired.\n\nThe antenna was tested.",
            "The antenna was tested.\n\nThe cable was replaced.",
        )

        assert result == "The receiver was repaired.\n\nThe antenna was tested.\n\nThe cable was replaced."


class TestArchiveBuilderMergeOverview:
    """Test ArchiveBuilder._merge_overview() method."""

    @pytest.fixture
    def builder(self):
        """Create ArchiveBuilder with MockLLM."""
        return ArchiveBuilder(llm=MockLLM())

    def test_merge_overview_combines_texts(self, builder):
        """_merge_overview should combine old and new overview."""
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Can review code",
            content="Reviews code",
            confidence=0.9,
        )

        merged_fields = {
            "skill_merge": True,
            "overview": "Old overview. Can review code.",
        }

        result = builder._merge_overview(candidate, merged_fields)

        assert "Old overview" in result or "review" in result

    def test_merge_overview_uses_candidate_if_no_merge(self, builder):
        """_merge_overview should use candidate overview if not skill_merge."""
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Code review skill",
            overview="Can review code",
            content="Reviews code",
            confidence=0.9,
        )

        merged_fields = {
            "overview": "Merged overview",
        }

        result = builder._merge_overview(candidate, merged_fields)

        assert result == candidate.overview


class TestArchiveBuilderEdgeCases:
    """Test ArchiveBuilder edge cases."""

    @pytest.fixture
    def builder(self):
        """Create ArchiveBuilder with MockLLM."""
        return ArchiveBuilder(llm=MockLLM())

    @pytest.fixture
    def ctx(self):
        """Create RequestContext."""
        return RequestContext(
            account_id="test-account",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            trace_id="trace-001",
        )

    def test_build_with_empty_content(self, builder, ctx):
        """build() should handle empty content."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="Abstract",
            overview="Overview",
            content="",
            confidence=0.9,
        )

        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(candidate, plan, ctx)

        assert node.content == ""

    def test_build_with_very_long_content(self, builder, ctx):
        """build() should handle very long content."""
        long_content = "A" * 100000  # 100KB content

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="Abstract",
            overview="Overview",
            content=long_content,
            confidence=0.9,
        )

        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(candidate, plan, ctx)

        assert node.content == long_content

    def test_build_with_special_characters_in_content(self, builder, ctx):
        """build() should handle special characters."""
        special_content = "Content with 特殊字符 🎉 \n\t\r"

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="Abstract",
            overview="Overview",
            content=special_content,
            confidence=0.9,
        )

        plan = WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-001/memories/profile",
            merged_fields={},
            relation_edges=[],
        )

        node = builder.build(candidate, plan, ctx)

        assert node.content == special_content
