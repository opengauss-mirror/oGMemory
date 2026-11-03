"""End-to-end integration tests for ContextEngine.

These tests verify the complete data flow from commit_session through
to vector indexing. They require a running AGFS server or MockAGFS.

Run with: pytest tests/integration/ -v -m integration
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import TypedQuery, user_space_name
from index.index_record_builder import build_index_records
from providers.vector_index import InMemoryVectorIndex
from service.api import MemoryWriteAPI


@pytest.fixture
def mock_llm(mock_llm):
    mock_llm._mock_tool_calls = [{
        "tool": "extract_profile",
        "input": {
            "abstract": "Alice likes TypeScript",
            "overview": "## Profile\nName: Alice\nPreference: TypeScript",
            "content": "我是Alice，喜欢TypeScript",
            "confidence": 0.9,
            "evidence_quote": "我是Alice",
            "attributed_speaker": "user",
            "attribution_basis": "self_first_person",
        },
    }]
    return mock_llm


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================

integration_mark = pytest.mark.integration


@pytest.mark.integration
class TestFullPipeline:
    """Test complete write pipeline from commit_session to indexing."""

    def test_commit_session_creates_context_node(
        self, agfs, sample_context, mock_llm
    ):
        """Verify commit_session creates a ContextNode in storage."""
        api = MemoryWriteAPI(fs=agfs, llm=mock_llm)

        # Configure mock to return profile extraction
        mock_llm._mock_tool_calls = [
            {
                "tool": "extract_profile",
                "input": {
                    "abstract": "软件工程师张三",
                    "overview": "## Profile\n\nName: 张三\nRole: 软件工程师",
                    "content": "我叫张三，是一名软件工程师",
                    "evidence_quote": "我叫张三，是一名软件工程师",
                    "attributed_speaker": "user",
                    "attribution_basis": "self_first_person",
                    "confidence": 0.9,
                },
            }
        ]

        messages = [
            {"role": "user", "content": "我叫张三，是一名软件工程师"},
            {"role": "assistant", "content": "你好张三！"}
        ]

        result = api.commit_session(messages, sample_context)

        assert result["candidates_extracted"] >= 1
        assert result["writes_completed"] >= 1

        # Verify node was written
        profile_uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/profile"
        assert agfs.exists(profile_uri, sample_context)

    def test_vector_index_receives_records(
        self, agfs, sample_context, mock_llm, mock_embedder
    ):
        """Verify IndexRecords are created and upserted to vector index."""
        # First write a node
        api = MemoryWriteAPI(fs=agfs, llm=mock_llm)

        messages = [
            {"role": "user", "content": "我是Alice，喜欢TypeScript"}
        ]

        api.commit_session(messages, sample_context)

        # Read the node back
        profile_uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/profile"
        node = agfs.read_node(profile_uri, sample_context)

        # Build index records from the node
        records = build_index_records(node)

        # Verify records were created
        assert len(records) >= 1
        assert all(r.filters["account_id"] == sample_context.account_id for r in records)
        assert all(r.filters["owner_space"] == user_space_name(sample_context.user_id) for r in records)

        # Upsert to vector index
        vector_index = InMemoryVectorIndex(dimension=mock_embedder._dimension)
        vector_index.upsert(records)

        # Verify search works
        query = TypedQuery(
            text="Alice TypeScript",
            context_type="MEMORY",
            categories=["profile"],
            account_id=sample_context.account_id,
            owner_space=user_space_name(sample_context.user_id),
        )

        results = vector_index.search(query)

        assert len(results) >= 1
        assert results[0].uri.removesuffix("/content.md") == profile_uri


@pytest.mark.integration
class TestMultiTenantIsolation:
    """Test multi-tenant isolation in the full pipeline."""

    def test_separate_accounts_isolated(
        self, agfs, sample_context_a, sample_context_b, mock_llm
    ):
        """Verify users from different accounts cannot access each other's data."""
        api = MemoryWriteAPI(fs=agfs, llm=mock_llm)

        # Write memory for user A
        api.commit_session(
            [{"role": "user", "content": "我是Alice，喜欢TypeScript。这是用户A的私密信息"}],
            sample_context_a
        )

        # Write memory for user B
        api.commit_session(
            [{"role": "user", "content": "我是Alice，喜欢TypeScript。这是用户B的私密信息"}],
            sample_context_b
        )

        assert agfs.exists(f"ctx://{sample_context_a.account_id}/users/{sample_context_a.user_id}/memories/profile", sample_context_a)
        assert agfs.exists(f"ctx://{sample_context_b.account_id}/users/{sample_context_b.user_id}/memories/profile", sample_context_b)

        # Verify A cannot read B's data
        profile_uri_b = f"ctx://{sample_context_b.account_id}/users/{sample_context_b.user_id}/memories/profile"

        # A trying to access B's profile should fail
        with pytest.raises(Exception):  # AccessDeniedError or NodeNotFoundError
            agfs.read_node(profile_uri_b, sample_context_a)


@pytest.mark.integration
class TestOwnerSpaceFormat:
    """Test owner_space format consistency across the pipeline."""

    def test_owner_space_uses_colon_format(
        self, agfs, sample_context, mock_llm
    ):
        """Verify owner_space uses user:id format, not user_id format."""
        api = MemoryWriteAPI(fs=agfs, llm=mock_llm)

        messages = [
            {"role": "user", "content": "我是Alice，喜欢TypeScript。测试 owner_space 格式"}
        ]

        result = api.commit_session(messages, sample_context)
        assert result["writes_completed"] == 1

        # Read back the node
        profile_uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/profile"
        node = agfs.read_node(profile_uri, sample_context)

        # Verify format is "user:id" not "user_id"
        expected_owner_space = user_space_name(sample_context.user_id)
        assert node.owner_space == expected_owner_space
        assert ":" in node.owner_space

    def test_index_records_use_same_owner_space(
        self, agfs, sample_context, mock_llm
    ):
        """Verify IndexRecords use the same owner_space format as ContextNode."""
        api = MemoryWriteAPI(fs=agfs, llm=mock_llm)

        messages = [
            {"role": "user", "content": "我是Alice，喜欢TypeScript。测试索引格式"}
        ]

        api.commit_session(messages, sample_context)

        profile_uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/profile"
        node = agfs.read_node(profile_uri, sample_context)

        records = build_index_records(node)

        expected_owner_space = user_space_name(sample_context.user_id)

        # All records should have the same owner_space as the node
        for record in records:
            assert record.filters["owner_space"] == expected_owner_space
            assert record.filters["owner_space"] == node.owner_space
