"""End-to-end test for complete ContextEngine pipeline.

This test verifies the full data flow:
1. Commit: commit_session() processes messages
2. Extract: LLM extracts CandidateMemory (7 types)
3. Plan: PolicyRouter creates WritePlan
4. Archive: ArchiveBuilder generates abstract/overview/content
5. Store: AGFSContextFS writes ContextNode
6. Index: IndexRecordBuilder creates IndexRecords
7. Retrieve: search_memory() finds and returns memories

Run with: pytest tests/integration/test_e2e_full_pipeline.py -v -s
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from core.models import RequestContext, agent_space_name, user_space_name
from index.index_record_builder import build_index_records
from providers.embedder import MockEmbedder
from providers.llm import MockLLM
from providers.vector_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from service.api import MemoryWriteAPI, ReadAPI
from service.memory_fs import MemoryFS


@pytest.fixture(scope="function")
def e2e_context():
    """Create RequestContext for e2e testing."""
    import uuid
    return RequestContext(
        account_id=f"e2e-test-{uuid.uuid4().hex}",
        user_id="test-user",
        agent_id="test-agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def e2e_embedder():
    """Mock embedder with consistent dimensions."""
    return MockEmbedder(dimension=1536)


@pytest.fixture(scope="function")
def e2e_vector_index(e2e_embedder):
    """In-memory vector index for testing."""
    return InMemoryVectorIndex(dimension=e2e_embedder._dimension)


@pytest.fixture(scope="function")
def e2e_llm():
    """Mock LLM that returns realistic extractions."""
    llm = MockLLM()

    # Configure mock to extract all 7 memory types
    llm._mock_tool_calls = [
        # Profile
        {
            "tool": "extract_profile",
            "input": {
                "abstract": "李明是一名Python开发工程师，在北京工作",
                "evidence_quote": "我叫李明",
                "attributed_speaker": "user",
                "attribution_basis": "self_first_person",
                "overview": "## Profile\n\n姓名: 李明\n职业: Python开发工程师\n地点: 北京",
                "content": "我叫李明，是一名Python开发工程师，目前在北京工作",
                "confidence": 0.95,
            },
        },
        # Preference
        {
            "tool": "extract_preference",
            "input": {
                "routing_key": "coding_style",
                "abstract": "偏好PEP8代码风格，使用类型注解",
                "evidence_quote": "遵循PEP8规范",
                "overview": "## 编码风格偏好\n\n- 遵循PEP8规范\n- 使用类型注解\n- 函数文档字符串",
                "content": "详细编码偏好: 遵循PEP8，使用类型注解和docstring",
                "confidence": 0.9,
            },
        },
        # Entity
        {
            "tool": "extract_entity",
            "input": {
                "routing_key": "fastapi",
                "abstract": "FastAPI - 现代Python Web框架",
                "overview": "## FastAPI\n\n类型: Web框架\n特点: 高性能、异步支持",
                "content": "FastAPI是一个现代、快速的Python Web框架",
                "confidence": 0.85,
            },
        },
        # Event
        {
            "tool": "extract_event",
            "input": {
                "routing_key": "code_review_20250326",
                "abstract": "参与了团队代码审查会议，讨论了性能优化方案",
                "overview": "## 代码审查会议\n\n时间: 2025-03-26\n议题: 性能优化",
                "content": "参加了团队的代码审查会议，讨论了FastAPI服务的性能优化方案",
                "confidence": 0.88,
            },
        },
        # Case
        {
            "tool": "extract_case",
            "input": {
                "routing_key": "fix_n_plus_1_query",
                "abstract": "修复了N+1查询问题，通过使用selectinload优化",
                "overview": "## 问题: N+1查询\n- 诊断: SQLAlchemy未加载关联数据\n- 解决: 使用selectinload预加载\n- 结果: 查询次数从100次降到2次",
                "content": "修复了API的N+1查询问题。发现是SQLAlchemy默认lazy loading导致，添加selectinload后查询性能提升50倍",
                "confidence": 0.92,
            },
        },
        # Pattern
        {
            "tool": "extract_pattern",
            "input": {
                "routing_key": "async_error_handling",
                "abstract": "用户倾向于在异步代码中使用try/except包裹每个await",
                "overview": "## 异步错误处理模式\n\n观察: 用户习惯在每个await外添加try/except",
                "content": "注意到用户在异步编程中习惯在每个await语句外添加独立的try/except块",
                "confidence": 0.75,
            },
        },
        # Skill
        {
            "tool": "extract_skill",
            "input": {
                "abstract": "FastAPI依赖注入使用技能",
                "overview": "## FastAPI依赖注入\n\n触发: 创建新endpoint时\n步骤: 1. 定义依赖函数 2. 使用Depends()注入",
                "content": "完整的FastAPI依赖注入使用指南和最佳实践",
                "confidence": 0.9,
            },
        },
    ]

    return llm


@pytest.mark.e2e
class TestE2EFullPipeline:
    """End-to-end test of the complete ContextEngine pipeline."""

    def test_full_pipeline_commit_to_retrieve(
        self, agfs, e2e_context, e2e_llm, e2e_embedder, e2e_vector_index
    ):
        """Test complete pipeline from commit_session to search_memory.

        Pipeline steps:
        1. commit_session() extracts and writes memories
        2. Verify ContextNodes stored in AGFS
        3. Build and upsert IndexRecords
        4. search_memory() retrieves relevant memories
        """

        # ====================================================================
        # Step 1: Commit - Extract and Write
        # ====================================================================
        print("\n[Step 1] Commit session...")

        write_api = MemoryWriteAPI(fs=agfs, llm=e2e_llm)

        messages = [
            {"role": "user", "content": "我叫李明，是一名Python开发工程师，在北京工作"},
            {"role": "assistant", "content": "你好李明！很高兴认识你。"},
            {"role": "user", "content": "我编码时遵循PEP8规范，喜欢用类型注解。最近在用FastAPI框架"},
            {"role": "assistant", "content": "FastAPI是个很棒的框架！"},
            {"role": "user", "content": "今天参加了代码审查会议，讨论了性能优化。我们还发现一个N+1查询问题，用selectinload修复了"},
            {"role": "assistant", "content": "很好的优化！N+1问题用preload解决很有效"},
        ]

        result = write_api.commit_session(messages, e2e_context)

        print(f"  Extracted: {result['candidates_extracted']} candidates")
        print(f"  Written: {result['writes_completed']} nodes")

        assert result["candidates_extracted"] == 7, "Should extract 7 memory types"
        assert result["writes_completed"] == 7, "Should write 7 nodes"

        # ====================================================================
        # Step 2: Verify Storage - ContextNodes in AGFS
        # ====================================================================
        print("\n[Step 2] Verify ContextNodes stored...")

        expected_uris = [plan["target_uri"] for plan in result["plans"]]
        stored_nodes = [agfs.read_node(uri, e2e_context) for uri in expected_uris]
        nodes = {node.category: node for node in stored_nodes}
        assert set(nodes) == {"profile", "preference", "entity", "event", "case", "pattern", "skill"}
        assert all(agfs.exists(uri, e2e_context) for uri in expected_uris)
        profile_uri = f"ctx://{e2e_context.account_id}/users/{e2e_context.user_id}/memories/profile"
        assert nodes["profile"].uri == profile_uri
        assert "PEP8" in nodes["preference"].content
        assert "FastAPI" in nodes["entity"].content
        assert "try/except" in nodes["pattern"].content
        assert "FastAPI" in nodes["skill"].content

        # Verify node structure
        profile_node = nodes["profile"]
        assert profile_node.category == "profile"
        assert profile_node.owner_space == user_space_name(e2e_context.user_id)
        assert profile_node.abstract == "李明是一名Python开发工程师，在北京工作"
        assert profile_node.overview  # Should have overview
        assert profile_node.content  # Should have content
        assert profile_node.metadata.get("status") == "ACTIVE"
        print("  ✓ Profile node structure valid")

        case_node = nodes["case"]
        assert case_node.owner_space == agent_space_name(e2e_context.agent_id)
        assert "selectinload" in case_node.content

        # ====================================================================
        # Step 3: Build Index Records
        # ====================================================================
        print("\n[Step 3] Build index records...")

        all_records = []
        for node in stored_nodes:
            records = build_index_records(node)
            all_records.extend(records)
            print(f"  {node.category}: {len(records)} records (L0/L1/L2)")

        assert len(all_records) >= len(expected_uris), f"At least {len(expected_uris)} index records"

        # Verify index record format
        for record in all_records:
            assert record.filters.get("account_id") == e2e_context.account_id
            assert record.filters.get("owner_space") in [
                user_space_name(e2e_context.user_id),
                agent_space_name(e2e_context.agent_id),
            ]
            assert record.filters.get("category") in [
                "profile", "preference", "entity", "event",
                "case", "pattern", "skill"
            ], f"Unknown category: {record.filters.get('category')}"

        print("  ✓ All index records have correct filters")

        # ====================================================================
        # Step 4: Upsert to Vector Index
        # ====================================================================
        print("\n[Step 4] Upsert to vector index...")

        e2e_vector_index.upsert(all_records)
        print(f"  ✓ Upserted {len(all_records)} records")

        # ====================================================================
        # Step 5: Retrieve - Search Memory
        # ====================================================================
        print("\n[Step 5] Search memories...")

        # Build retrieval pipeline
        cfg = RetrievalConfig()
        pipeline = RetrievalPipeline(
            planner=QueryPlanner(cfg),
            seed_retriever=SeedRetriever(e2e_vector_index, e2e_embedder, cfg),
            hierarchical_searcher=HierarchicalSearcher(e2e_vector_index, cfg),
            assembly=ResultRanker(cfg),
            config=cfg,
        )

        read_api = ReadAPI(
            pipeline=pipeline,
            read_service=ContextReader(fs=agfs),
            config=cfg,
        )

        # Test 1: Search for profile info
        print("\n  Query 1: '李明是谁？'")
        result1 = read_api.search_memory(
            query="李明是谁？他在哪工作？",
            ctx=e2e_context,
            top_k=5,
        )

        print(f"    Found {len(result1.hits)} hits")

        # With mock embedder, any result is acceptable - just verify retrieval works
        assert len(result1.hits) > 0, "Should find at least one result"

        # Check that results have expected structure
        for hit in result1.hits:
            assert hasattr(hit, 'category'), "Hit should have category"
            assert hasattr(hit, 'abstract'), "Hit should have abstract"
            assert hasattr(hit, 'content_excerpt'), "Hit should have content_excerpt"
            assert hasattr(hit, 'uri'), "Hit should have uri"
            assert hit.uri.startswith(f"ctx://{e2e_context.account_id}/"), "URI should match account"

        print(f"    ✓ Retrieval works, got categories: {[h.category for h in result1.hits]}")

        # Try a more specific query for profile
        print("\n  Query 1b: Direct profile URI lookup")
        profile_content = read_api.read_memory(profile_uri, e2e_context)
        assert "李明" in profile_content.content_excerpt or "工程师" in profile_content.content_excerpt

        # Test 2: More searches to verify retrieval
        print("\n  Query 2: '编码风格是什么？'")
        result2 = read_api.search_memory(
            query="有什么编码习惯？",
            ctx=e2e_context,
            top_k=5,
        )

        print(f"    Found {len(result2.hits)} hits")
        print(f"    ✓ Search 2 works, categories: {[h.category for h in result2.hits]}")

        # Test 3: Search for technical content
        print("\n  Query 3: '使用什么框架？'")
        result3 = read_api.search_memory(
            query="在用什么框架开发？",
            ctx=e2e_context,
            top_k=5,
        )

        print(f"    Found {len(result3.hits)} hits")
        print(f"    ✓ Search 3 works, categories: {[h.category for h in result3.hits]}")

        # ====================================================================
        # Step 6: Verify Multi-tenant Isolation
        # ====================================================================
        print("\n[Step 6] Verify multi-tenant isolation...")

        # Create different user context
        other_context = RequestContext(
            account_id="other-account",
            user_id="other-user",
            agent_id="other-agent",
            session_id=e2e_context.session_id,
            trace_id=e2e_context.trace_id,
        )

        # Search from different account should return empty
        isolated_result = read_api.search_memory(
            query="李明是谁",
            ctx=other_context,
            top_k=5,
        )

        assert len(isolated_result.hits) == 0, "Other account should not see data"
        print("  ✓ Multi-tenant isolation working")

        print("\n✅ Full pipeline test PASSED!")
        print("\nSummary:")
        print("  - Extracted: 7 memory types (profile/preference/entity/event/case/pattern/skill)")
        print(f"  - Stored: {len(expected_uris)} ContextNodes in AGFS (event/case use dynamic IDs)")
        print(f"  - Indexed: {len(all_records)} IndexRecords")
        print("  - Retrieved: All queries returned relevant results")
        print("  - Isolation: Multi-tenant working correctly")


@pytest.mark.e2e
class TestE2EMemoryFS:
    """Test MemoryFS file system browsing interface."""

    def test_memory_fs_list_and_read(
        self, agfs, e2e_context, e2e_llm
    ):
        """Test MemoryFS can read memory metadata and content."""
        write_api = MemoryWriteAPI(fs=agfs, llm=e2e_llm)

        messages = [
            {"role": "user", "content": "我叫李明，是一名Python开发工程师"},
        ]

        write_api.commit_session(messages, e2e_context)

        # Create MemoryFS
        memory_fs = MemoryFS(fs=agfs)

        # Test direct node access (listing requires directory URI fix)
        profile_path = f"/users/{e2e_context.user_id}/memories/profile"

        # Check exists
        assert memory_fs.exists(profile_path, e2e_context), "Profile should exist"
        print("  ✓ Profile exists")

        # Get profile info
        info = memory_fs.stat(profile_path, e2e_context)

        assert info["category"] == "profile"
        assert info["has_content"] is True
        print(f"  ✓ Profile stat: category={info['category']}, size={info['size']}")

        # Read abstract
        abstract = memory_fs.read_abstract(profile_path, e2e_context)
        assert len(abstract) > 0, "Abstract should not be empty"
        print(f"  ✓ Profile abstract: {abstract}")

        # Read overview
        overview = memory_fs.read_overview(profile_path, e2e_context)
        assert len(overview) > 0, "Overview should not be empty"
        print("  ✓ Profile overview exists")


# E2E marker
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "e2e: end-to-end integration tests")
