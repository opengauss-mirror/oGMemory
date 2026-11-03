#!/usr/bin/env python3
"""测试 assemble prompt 拼接逻辑。

无需 Docker 镜像，直接从代码目录运行：
  cd /data1/sundechao/omv2/oG-Memory_7755
  python3 tests/test_assemble_prompt.py

该脚本模拟 compose() 各层的输出，展示最终发送给 LLM 的完整 prompt 结构。
需要连接 AGFS 和向量数据库才能看到真实检索结果（否则各层降级为空）。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CONTEXTENGINE_PROVIDER", "mock")
os.environ.setdefault("AGFS_BASE_URL", "http://127.0.0.1:1833")

def test_compose_structure():
    """调用 MemoryService.compose()，打印完整返回结构。"""
    from server.memory_service import MemoryService

    svc = MemoryService()

    params = {
        "messages": [
            {"role": "user", "content": "What pet does Caroline have?"},
        ],
        "sessionId": "test-session-001",
        "accountId": "acct-demo",
        "userId": "u-alice",
        "agentId": "main",
        "tokenBudget": 128000,
    }

    print("=" * 70)
    print("调用 compose() ...")
    print("=" * 70)

    result = svc.compose(params)

    print()
    print(">>> compose() 返回的完整字段 <<<")
    print("-" * 50)
    for key in sorted(result.keys()):
        val = result[key]
        if isinstance(val, str):
            preview = val[:200] + "..." if len(val) > 200 else val
            print(f"  {key}: ({len(val)} chars) {repr(preview)}")
        elif isinstance(val, list):
            print(f"  {key}: [{len(val)} items]")
        elif isinstance(val, dict):
            print(f"  {key}: {json.dumps(val, ensure_ascii=False)}")
        else:
            print(f"  {key}: {val}")

    print()
    print(">>> 各语义槽内容 <<<")
    print("-" * 50)

    slots = [
        ("identityContext", "Layer 1 - Profile"),
        ("episodicContext", "Layer 1b - Archive History"),
        ("sessionContext", "Layer 2 - Session State"),
        ("retrievedEvidence", "Layer 3 - Working Set"),
    ]

    for field, label in slots:
        content = result.get(field, "")
        if content:
            print(f"\n[{label}] ({len(content)} chars)")
            print(content[:500])
            if len(content) > 500:
                print(f"  ... (truncated, total {len(content)} chars)")
        else:
            print(f"\n[{label}] (empty)")

    print()
    print(">>> messages 数组（注入后）<<<")
    print("-" * 50)

    messages = result.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            preview = content[:150]
            print(f"\n  [{i}] role={role} ({len(content)} chars)")
            print(f"      {preview}")
        elif isinstance(content, list):
            total = sum(len(b.get("text", "")) if isinstance(b, dict) else len(b) for b in content)
            print(f"\n  [{i}] role={role} (blocks={len(content)}, total_len={total})")

    print()
    print("=" * 70)
    print(">>> 最终 LLM 看到的 prompt 拼接顺序 <<<")
    print("=" * 70)

    print("""
当前拼接逻辑（修改后）：

1. systemPromptAddition = "" (空，不拼任何东西)
2. messages 数组（后端已组装好，插件直接透传）：
   [user] Profile            ← 最前面
   [user/assistant] 原始对话  ← 中间
   [user] Working Set        ← 最后面（向量检索）

最终 LLM 收到的 messages:
""")

    messages = result.get("messages", [])
    for idx, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            print(f"  [{idx}] [{role}] {repr(content[:100])}")
        elif isinstance(content, list):
            print(f"  [{idx}] [{role}] (blocks={len(content)})")


def test_compose_with_mock_layers():
    """模拟各层都有内容的情况，展示完整结构。"""
    from server.memory_service import MemoryService, ComposedContext

    print()
    print("=" * 70)
    print("模拟各层都有内容的完整 prompt 结构")
    print("=" * 70)

    svc = MemoryService()

    mock_result = ComposedContext(
        identity_context="## Profile\nUser is Alice, an AI researcher. Prefers concise English answers.",
        episodic_context=(
            "## Archive History\n"
            "### Latest Session\n"
            "Caroline and Melanie discussed adoption plans, shared paintings, and talked about pets.\n"
            "Caroline mentioned she has a guinea pig named Oscar. Melanie has 3 kids.\n"
            "### Previous Sessions\n"
            "- archive_01: Melanie ran a charity race in May 2023\n"
            "- archive_02: Caroline attended LGBTQ+ counseling workshop\n"
            "- archive_03: Both went camping in June 2023"
        ),
        session_context=(
            "## Active Task\nAnswering QA about Caroline and Melanie's life\n\n"
            "## Recent Decisions\n- Looking up memory for pet information\n\n"
            "## Recent Session Summary\n"
            "User asked 5 questions about Caroline and Melanie. 3 answered correctly."
        ),
        task_context="",
        retrieved_evidence=(
            "## Working Set (relevant memories)\n"
            "- [entity] Caroline is a transgender woman who shared her journey (相关度: 92%)\n"
            "- [entity] Caroline has a guinea pig named Oscar (相关度: 88%)\n"
            "- [event] Melanie went camping with family in July 2023 (相关度: 78%)\n"
            "- [entity] Melanie has 3 children and pets named Oliver, Luna, Bailey (相关度: 75%)\n"
            "- [preference] Caroline enjoys abstract art and painting (相关度: 65%)"
        ),
        messages=[
            {"role": "user", "content": "Current date: 2023-10-22. Answer the question directly: What pet does Caroline have?"},
        ],
        estimated_tokens=15000,
        archive_count=3,
        archive_included=True,
    )

    output = svc._to_response(mock_result)

    print("\n>>> ogmem compose() 返回给 OpenClaw 插件的字段 <<<\n")
    for key in ["identityContext", "episodicContext", "sessionContext", "retrievedEvidence"]:
        val = output.get(key, "")
        if val:
            print(f"[{key}] ({len(val)} chars):")
            print(val)
            print()

    print("\n>>> 最终 LLM 收到的 messages（插件直接透传后端 messages）<<<\n")

    print("  systemPromptAddition = '' (空)")
    print()
    msgs = output.get("messages", [])
    for idx, msg in enumerate(msgs):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            print(f"  messages[{idx}] = {{role: '{role}', content: '{content[:80]}...'}}")


if __name__ == "__main__":
    print("Test 1: 调用真实 compose() (可能各层为空，取决于 AGFS/DB 连接)")
    test_compose_structure()

    print("\n\nTest 2: 模拟完整各层内容")
    test_compose_with_mock_layers()
