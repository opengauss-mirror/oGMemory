# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""ContextEngine 与 Claude Code/OpenClaw 集成示例。

演示如何将 ContextEngine 作为 Agent 的记忆系统。
注意: dev 分支只包含写入路径，读取功能在 phase1 分支。
"""

import os
from typing import List, Dict, Any, Optional
from service.api import MemoryWriteAPI, ReadAPI, init_api
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import OpenAILLM
from providers.vector_index.in_memory_index import InMemoryVectorIndex


class AgentMemory:
    """Agent 记忆管理器 - 为 AI Agent 提供持久化记忆功能 (仅写入).

    用法:
        memory = AgentMemory(account_id="my-account", user_id="user-123")
        memory.remember_conversation([
            {"role": "user", "content": "我偏好 Python 编程"},
            {"role": "assistant", "content": "记住了"}
        ])

    注意: 搜索功能需要切换到 phase1 分支
    """

    def __init__(
        self,
        account_id: str,
        user_id: str,
        agent_id: str = "agent-001",
        agfs_url: str = "http://localhost:1833",
    ):
        """初始化记忆系统。

        Args:
            account_id: 租户ID
            user_id: 用户ID
            agent_id: Agent ID
            agfs_url: AGFS 服务地址
        """
        self.account_id = account_id
        self.user_id = user_id
        self.agent_id = agent_id

        # 初始化依赖
        client = AGFSClient(api_base_url=agfs_url)
        fs = AGFSContextFS(client=client, mount_prefix="/local")

        api_key = os.environ.get("OGMEM_API_KEY")
        if not api_key:
            raise ValueError("OGMEM_API_KEY 环境变量未设置")

        llm = OpenAILLM(api_key=api_key, model="gpt-4")
        vector_index = InMemoryVectorIndex(dimension=384)

        # init_api 返回 (read_api, write_api) 元组
        self._read_api, self._write_api = init_api(fs=fs, vector_index=vector_index, llm=llm, outbox_store=None)

    def _create_context(self, session_id: str) -> RequestContext:
        """创建请求上下文"""
        import uuid
        return RequestContext(
            account_id=self.account_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
        )

    def remember_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """记住对话内容。

        Args:
            messages: 对话消息列表
            session_id: 会话ID（可选）

        Returns:
            写入结果统计
        """
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())

        ctx = self._create_context(session_id)

        result = self._write_api.commit_session(messages, ctx)
        return {
            "session_id": session_id,
            "candidates_written": result.get("writes_completed", 0),
            "candidates_filtered": result.get("candidates_filtered", 0),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """搜索相关记忆。

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            category: 限制搜索的类别（可选）
            session_id: 会话ID（可选）

        Returns:
            搜索结果列表，每项包含 uri, abstract, score 等
        """
        ctx = self._create_context(session_id or "search")

        results = self._read_api.search_memory(query, top_k=top_k, category=category, ctx=ctx)

        # search_memory 返回的是 dict 列表，不是对象
        return results

    def remember_fact(
        self,
        content: str,
        category: str = "entity",
        confidence: float = 0.9,
    ) -> Dict[str, Any]:
        """记住一条事实（快捷方式）。

        Args:
            content: 事实内容
            category: 类别 (entity/preference/pattern 等)
            confidence: 置信度

        Returns:
            写入结果
        """
        import uuid

        # 将单条事实包装成对话格式
        messages = [
            {"role": "user", "content": content}
        ]
        return self.remember_conversation(messages, str(uuid.uuid4()))


# ========== 使用示例 ==========

def demo_claude_code_integration():
    """Claude Code 集成示例"""

    # 初始化记忆系统
    memory = AgentMemory(
        account_id="claude-code-user",
        user_id="user-001",
        agent_id="claude-code",
    )

    # 记住对话
    print("=== 记忆对话 ===")
    result = memory.remember_conversation([
        {"role": "user", "content": "我的名字是李四，是一名前端开发工程师"},
        {"role": "assistant", "content": "好的李四，我记住了"},
    ])
    print(f"写入完成: {result['candidates_written']} 条记忆")

    # 记住偏好
    print("\n=== 记住偏好 ===")
    memory.remember_fact("用户喜欢使用 TypeScript 进行开发", category="preference")

    print("\n注意: 搜索功能在 phase1 分支可用")
    print("切换到 phase1 分支获取完整的读写功能:")
    print("  git checkout phase1")


if __name__ == "__main__":
    # 运行示例
    print("=" * 50)
    print("ContextEngine - Claude Code/OpenClaw 集成示例")
    print("=" * 50)
    demo_claude_code_integration()
    print("\n" + "=" * 50 + "\n")
