# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""ContextEngine 基础使用示例。

演示如何使用 ContextEngine 为 AI Agent 存储和检索记忆。
"""

import os
from service.api import MemoryWriteAPI, ReadAPI, init_api
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import OpenAILLM
from providers.embedder import OpenAIEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex


def main():
    """基础使用示例：写入和读取记忆"""

    # 1. 配置 AGFS 连接
    agfs_url = os.environ.get("AGFS_BASE_URL", "http://localhost:1833")
    client = AGFSClient(api_base_url=agfs_url)
    fs = AGFSContextFS(client=client, mount_prefix="/local")

    # 2. 配置 LLM (用于抽取)
    api_key = os.environ.get("OGMEM_API_KEY")
    if not api_key:
        print("请设置 OGMEM_API_KEY 环境变量")
        return

    llm = OpenAILLM(api_key=api_key, model="gpt-4")

    # 3. 配置向量索引
    vector_index = InMemoryVectorIndex(dimension=384)

    # 4. 初始化 API (返回 read_api 和 write_api)
    read_api, write_api = init_api(fs=fs, vector_index=vector_index, llm=llm, outbox_store=None)

    # 5. 创建请求上下文
    ctx = RequestContext(
        account_id="demo-account",
        user_id="alice",
        agent_id="assistant",
        session_id="demo-session",
        trace_id="demo-trace"
    )

    # 6. 写入对话记忆
    messages = [
        {"role": "user", "content": "我叫张三，住在北京，是一名后端工程师"},
        {"role": "assistant", "content": "你好张三！很高兴认识你。"}
    ]

    write_result = write_api.commit_session(messages, ctx)
    print(f"✅ 写入完成: {write_result}")

    # 7. 检索记忆
    query = "张三是做什么工作的？"
    search_result = read_api.search_memory(query, top_k=5, ctx=ctx)
    print(f"✅ 检索结果: {len(search_result)} 条相关记忆")

    for hit in search_result:
        print(f"  - {hit['uri']}: {hit['abstract']}")


if __name__ == "__main__":
    main()
