#!/usr/bin/env python3
"""
ContextEngine Bridge Test Script

直接测试 OpenClaw 插件的 bridge/memory_api.py 功能，
无需安装 OpenClaw。
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path (script is in scripts/, so go up one level)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add plugin directory and its parent to path
plugin_dir = project_root / "plugins/openclaw"
sys.path.insert(0, str(plugin_dir))
sys.path.insert(0, str(plugin_dir / "bridge"))

# Set up environment
os.environ.setdefault('AGFS_BASE_URL', 'http://localhost:1833')
os.environ.setdefault('CONTEXTENGINE_PROVIDER', 'mock')  # Default to mock for testing

# Import functions directly from the module
# The memory_api.py module sets up sys.path internally, so we can exec it
import importlib.util
spec = importlib.util.spec_from_file_location("memory_api", plugin_dir / "bridge" / "memory_api.py")
memory_api = importlib.util.module_from_spec(spec)
sys.modules['memory_api'] = memory_api
spec.loader.exec_module(memory_api)

commit_messages_to_memory = memory_api.commit_messages_to_memory
_extract_query = memory_api._extract_query
_format_memory_addition = memory_api._format_memory_addition


def test_commit_session():
    """Test commit_session functionality."""
    print("\n" + "="*60)
    print("测试 1: commit_session - 写入对话记忆")
    print("="*60)

    messages = [
        {"role": "user", "content": "我叫张三，是一名Python开发者，住在北京"},
        {"role": "assistant", "content": "你好张三！很高兴认识你。"},
        {"role": "user", "content": "我最近在学习FastAPI框架，觉得它很强大"},
        {"role": "assistant", "content": "FastAPI确实是个很棒的现代Web框架！"},
    ]

    session_id = "test-session-001"

    print(f"\n输入消息 ({len(messages)} 条):")
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:50]
        print(f"  [{i+1}] {role}: {content}...")

    try:
        result = commit_messages_to_memory(messages, session_id)
        print(f"\n✅ 写入成功:")
        print(f"  - 提取候选: {result.get('candidates_extracted', 0)}")
        print(f"  - 完成写入: {result.get('writes_completed', 0)}")
        print(f"  - 写入失败: {result.get('writes_failed', 0)}")
        return True
    except Exception as e:
        print(f"\n❌ 写入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_memory():
    """Test search_memory functionality."""
    print("\n" + "="*60)
    print("测试 2: search_memory - 语义检索")
    print("="*60)

    messages = [
        {"role": "user", "content": "张三是做什么工作的？"},
        {"role": "assistant", "content": "让我查一下..."},
    ]

    query = _extract_query(messages)
    print(f"\n检索查询: {query}")

    # Import ReadAPI to do actual search
    try:
        from core.models import RequestContext
        from service.api import ReadAPI
        from pyagfs import AGFSClient
        from fs.agfs_adapter import AGFSContextFS
        from retrieval.pipeline import RetrievalPipeline, RetrievalConfig
        from retrieval.query_planner import QueryPlanner
        from retrieval.seed_retriever import SeedRetriever
        from retrieval.hierarchical_searcher import HierarchicalSearcher
        from retrieval.result_ranker import ResultRanker
        from retrieval.context_reader import ContextReader
        from providers.embedder.mock_embedder import MockEmbedder
        from providers.vector_index.in_memory_index import InMemoryVectorIndex

        # Initialize components
        agfs_client = AGFSClient(api_base_url=os.environ.get('AGFS_BASE_URL'))
        agfs = AGFSContextFS(client=agfs_client, mount_prefix="/local/plugin")

        cfg = RetrievalConfig()
        embedder = MockEmbedder()
        vector_index = InMemoryVectorIndex(dimension=384)

        pipeline = RetrievalPipeline(
            planner=QueryPlanner(cfg),
            seed_retriever=SeedRetriever(vector_index, embedder, cfg),
            hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
            assembly=ResultRanker(cfg),
            config=cfg,
        )

        read_api = ReadAPI(
            pipeline=pipeline,
            read_service=ContextReader(fs=agfs),
            config=cfg,
        )

        ctx = RequestContext(
            account_id="default",
            user_id="default-user",
            agent_id="main",
            session_id="test-session-001",
            trace_id="test-trace",
        )

        result = read_api.search_memory(
            query=query,
            ctx=ctx,
            top_k=5,
        )

        print(f"\n✅ 检索成功:")
        print(f"  - 返回结果: {len(result.hits)} 条")

        if result.hits:
            print(f"\n  检索结果:")
            for i, hit in enumerate(result.hits[:3], 1):
                category = hit.category or "memory"
                abstract = hit.abstract or "无摘要"
                score = hit.score or 0
                print(f"    [{i}] [{category}] {abstract} (相关度: {score:.2f})")

        # Test format_memory_addition
        formatted = _format_memory_addition(result.hits)
        print(f"\n  系统提示注入示例:\n{formatted}")

        return True
    except Exception as e:
        print(f"\n❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check():
    """Test health check."""
    print("\n" + "="*60)
    print("测试 3: health_check - 健康检查")
    print("="*60)

    try:
        from pyagfs import AGFSClient

        client = AGFSClient(api_base_url=os.environ.get('AGFS_BASE_URL'))
        client.ls("/")

        print(f"\n✅ AGFS 连接正常")
        print(f"  - URL: {os.environ.get('AGFS_BASE_URL')}")
        return True
    except Exception as e:
        print(f"\n❌ AGFS 连接失败: {e}")
        print(f"\n提示: 请先启动 AGFS 服务")
        print(f"  cd agfs/agfs-server")
        print(f"  go build -o build/agfs-server")
        print(f"  ./build/agfs-server -c config.yaml")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ContextEngine Bridge 功能测试")
    print("="*60)
    print(f"\n环境配置:")
    print(f"  AGFS_BASE_URL: {os.environ.get('AGFS_BASE_URL')}")
    print(f"  CONTEXTENGINE_PROVIDER: {os.environ.get('CONTEXTENGINE_PROVIDER', 'not set')}")
    print(f"  OGMEM_API_KEY: {'已设置' if os.environ.get('OGMEM_API_KEY') else '未设置 (将使用 mock)'}")

    results = []

    # Run tests
    results.append(("健康检查", test_health_check()))
    results.append(("写入记忆", test_commit_session()))
    results.append(("检索记忆", test_search_memory()))

    # Summary
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过！ContextEngine 集成正常工作。")
        print("\n下一步:")
        print("  1. 安装 OpenClaw: npm install -g @openclaw/cli")
        print("  2. 配置插件: openclaw plugins install -l ./plugins/openclaw")
        print("  3. 设置 contextEngine 插件为 og-memory-context-engine")
    else:
        print("\n⚠️ 部分测试失败，请检查 AGFS 服务是否运行。")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
