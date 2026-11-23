from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.models import ContextNode
from memorybot.models import ChatInput
from memorybot.templates import TemplateRegistry
from tests.integration.test_memorybot_runtime import execute, memory_node, source, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("memory_owner", ["user", "agent:memorybot"])
def test_template_owner_range_drives_model_materials_and_read_tool(runtime, memory_owner):
    ctx = replace(
        runtime.ctx,
        visible_owner_spaces=(runtime.ctx.user_space_name(), runtime.ctx.agent_space_name()),
    )
    runtime.ctx = ctx
    runtime.gateway.context_resolver = lambda item: replace(
        ctx, session_id=item["accepted_authz"].get("memory_session_id", item["task_id"])
    )
    user_uri = memory_node(runtime, "青禾用户计划：周五完成发布验收。")
    agent_uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/experiences/qinghe"
    agent = ContextNode(
        agent_uri,
        "MEMORY",
        "experience",
        4,
        ctx.agent_space_name(),
        "青禾操作经验",
        "青禾发布前完成回滚演练。",
        "青禾操作经验：发布前先完成回滚演练。",
        {"routing_key": "qinghe"},
    )
    fs = runtime.memory._get_context_fs()
    fs.write_node(agent, ctx)
    runtime.memory._get_outbox_store(fs).register_write(agent, ctx)
    runtime.memory.drain_outbox_sync(ctx.account_id)
    document = source(runtime, "青禾资料.md", "青禾本次发布负责人为林悦。")
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.update(
            "bot_session",
            {"session_id": ctx.session_id},
            {"scope": {"memory_scope": "authorized", "resource_ids": []}},
        )
    declaration = TemplateRegistry().get("weekly_report")
    declaration.update(
        id="scoped_report",
        tools=["read_memory", "search_memory"],
        permission_requirements=[
            {"action": "memory.read", "scope": memory_owner},
            {"action": "resource.read", "scope": "user"},
            {"action": "file.write", "scope": "user"},
            {"action": "tool.execute", "scope": "search_memory"},
        ],
    )
    runtime.worker.templates = TemplateRegistry([declaration])
    runtime.llm.planned_calls.append({"tool": "search_memory", "input": {"query": "青禾"}})
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(
                text="整理青禾发布信息，查询对应记忆并保留资料负责人。",
                template_id="scoped_report",
                file_ids=[document["file_id"]],
            ).model_dump(mode="json"),
        ),
    )
    expected_uri = user_uri if memory_owner == "user" else agent_uri
    expected_owner = ctx.user_space_name() if memory_owner == "user" else ctx.agent_space_name()
    materials = json.loads(runtime.llm.messages[0][1]["content"].split("\n", 1)[1])
    memories = [item for item in materials["sources"] if item["context_type"] == "MEMORY"]
    resources = [item for item in materials["sources"] if item["context_type"] == "RESOURCE"]
    assert {item["uri"] for item in memories} == {expected_uri}
    assert {item["owner_space"] for item in memories} == {expected_owner}
    assert {item["uri"] for item in resources} == {
        item["uri"] for item in document["metadata"]["memory_sources"]
    }
    assert materials["documents"][0]["text"] == "青禾本次发布负责人为林悦。"
    steps = runtime.store.list(ctx.account_id, "bot_step", {"task_id": result["task_id"]})
    frozen = next(step for step in steps if step["input"].get("template_id") == "scoped_report")
    retrieved = next(step for step in steps if "authorization_fingerprint" in step["input"])
    assert frozen["step_no"] < retrieved["step_no"]
    assert retrieved["input"]["scope"]["memory_owners"] == [expected_owner]
    model_steps = [step for step in steps if step["kind"] == "model"]
    assert all(
        [tool["name"] for tool in step["input"]["tools"]] == ["search_memory"]
        for step in model_steps
    )
    tool = next(step for step in steps if step["input"].get("tool") == "search_memory")
    assert {item["uri"] for item in tool["output"]["sources"]} == {expected_uri}
