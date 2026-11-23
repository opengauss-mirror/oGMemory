# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Run from the repository after installing .[deepagents] and setting OGMEM_*.

Usage: python -m examples.deepagents_memory
Configure ANTHROPIC_API_KEY for the Deep Agents default model, or pass another
configured LangChain chat model to create_deep_agent(model=...).
"""

from uuid import uuid4

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from integrations.deepagents import OGMemoryMiddleware


def main() -> None:
    memory = OGMemoryMiddleware.from_env()
    agent = create_deep_agent(middleware=[memory], checkpointer=InMemorySaver())
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        while prompt := input("You (empty to exit): ").strip():
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}, config=config
            )
            print(result["messages"][-1].content)
    finally:
        memory.dispose(thread_id)


if __name__ == "__main__":
    main()
