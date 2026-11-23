import json

import pytest
import yaml

from core.models import ContextNode
from memorybot.models import ChatInput
from memorybot.templates import TemplateRegistry
from tests.integration.test_memorybot_runtime import execute, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


def test_template_scope_selects_native_user_agent_and_resource_materials(runtime):
    ctx, fs = runtime.ctx, runtime.memory._get_context_fs()
    nodes = [
        ContextNode(
            f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/release",
            "MEMORY",
            "entity",
            2,
            ctx.user_space_name(),
            "用户发布安排",
            "用户发布安排",
            "林悦负责11月12日青禾发布。",
            {},
        ),
        ContextNode(
            f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/cases/release",
            "MEMORY",
            "case",
            2,
            ctx.agent_space_name(),
            "Agent发布经验",
            "Agent发布经验",
            "青禾发布需完成交接确认。",
            {},
        ),
        ContextNode(
            f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/documents/notes",
            "RESOURCE",
            "document",
            2,
            ctx.user_space_name(),
            "用户发布资料",
            "用户发布资料",
            "青禾交接由林悦签收。",
            {},
        ),
        ContextNode(
            f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/documents/notes",
            "RESOURCE",
            "document",
            2,
            ctx.agent_space_name(),
            "Agent发布资料",
            "Agent发布资料",
            "青禾发布检查共三项。",
            {},
        ),
    ]
    for node in nodes:
        fs.write_node(node, ctx)
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.update(
            "bot_session",
            {"session_id": ctx.session_id},
            {"scope": {"memory_scope": "authorized", "resource_ids": [node.uri for node in nodes]}},
        )
    declaration = TemplateRegistry().get("weekly_report")
    declaration.update(
        id="native_ranges",
        name="发布经验与资料",
        tools=["read_memory"],
        permission_requirements=[
            {"action": "memory.read", "scope": ctx.agent_space_name()},
            {"action": "resource.read", "scope": "user"},
            {"action": "resource.read", "scope": nodes[3].uri},
            {"action": "file.write", "scope": "user"},
            {"action": "tool.execute", "scope": "read_memory"},
        ],
    )
    runtime.memory.add_managed_skill(
        ctx,
        "---\n"
        + yaml.safe_dump(
            {
                "name": "native-ranges",
                "description": "整理青禾发布经验与资料",
                "memorybot_template": declaration,
            },
            allow_unicode=True,
        )
        + "---\n整理发布经验与资料，注明来源。\n",
    )
    for template_id, expected in (
        ("weekly_report", {nodes[0].uri, nodes[2].uri}),
        ("native_ranges", {nodes[1].uri, nodes[2].uri, nodes[3].uri}),
    ):
        completed = execute(
            runtime,
            task(
                runtime,
                "chat",
                ChatInput(text="整理青禾发布", template_id=template_id).model_dump(mode="json"),
            ),
        )
        assert completed["result"]["file_ids"]
        materials = next(
            json.loads(message["content"].split("\n", 1)[1])
            for message in runtime.llm.messages[-1]
            if message["content"].startswith("以下是本轮可参考的资料与来源：\n")
        )
        assert {source["uri"] for source in materials["sources"]} == expected
        steps = runtime.store.list(ctx.account_id, "bot_step", {"task_id": completed["task_id"]})
        retrieved = next(step for step in steps if "scope" in step["input"])
        assert retrieved["input"]["scope"]["memory_scope"] == "authorized"
        assert {source["uri"] for source in retrieved["output"]["sources"]} == expected
