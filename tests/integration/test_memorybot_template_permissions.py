import pytest
import yaml

from core.models import ContextNode
from memorybot.models import ChatInput, SessionCreate
from memorybot.service import BotService
from memorybot.templates import TemplateRegistry
from tests.integration.test_memorybot_runtime import execute
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


def test_native_published_template_permissions_cover_list_admission_and_execution(runtime):
    memory = runtime.memory
    memory._cfg.role_control_enabled = True
    memory._cfg.root_api_key = "integration-root-" + runtime.ctx.account_id
    service = BotService(
        runtime.store, memory, runtime.files, runtime.worker.config, gateway=runtime.gateway
    )
    ctx = service.authenticate({"X-API-Key": runtime.api_key})
    uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/documents/release-permissions"
    memory._get_context_fs().write_node(
        ContextNode(
            uri,
            "RESOURCE",
            "document",
            2,
            ctx.user_space_name(),
            "青禾发布资料",
            "林悦负责",
            "青禾负责人林悦，11月12日发布。",
            {"title": "青禾发布资料"},
        ),
        ctx,
    )
    declaration = TemplateRegistry().get("weekly_report")
    declaration.update(
        id="permission_review",
        name="授权资料检查",
        tools=["read_memory"],
        permission_requirements=[
            {"action": "memory.read", "scope": "user"},
            {"action": "resource.read", "scope": uri},
            {"action": "file.write", "scope": "user"},
            {"action": "tool.execute", "scope": "read_memory"},
        ],
    )
    memory.add_managed_skill(
        ctx,
        "---\n"
        + yaml.safe_dump(
            {
                "name": "permission-review",
                "description": "按授权资料整理发布信息",
                "memorybot_template": declaration,
            },
            allow_unicode=True,
        )
        + "---\n读取指定资料，列出负责人和发布日期，注明来源。\n",
    )
    scope = {"memory_scope": "user", "resource_ids": [uri]}
    rows = service.template_list(ctx, scope)
    visible = next(row for row in rows if row["id"] == "permission_review")
    assert visible["permission_requirements"] == declaration["permission_requirements"]
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        _, session = service.create_session(
            ctx, tx, SessionCreate(title="授权模板", scope=scope).model_dump(mode="json")
        )
    data = ChatInput(text="整理发布资料", template_id="permission_review").model_dump(mode="json")
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        status, accepted = service.create_chat(
            ctx, tx, session["session_id"], data, "template-review"
        )
    assert status == 202
    runtime.llm.planned_calls.append(
        {"id": "call_source", "tool": "read_memory", "input": {"uri": uri}}
    )
    result = execute(runtime, accepted)
    assert result["result"]["file_ids"]
    steps = runtime.store.list(ctx.account_id, "bot_step", {"task_id": accepted["task_id"]})
    frozen = next(
        step["output"]
        for step in steps
        if (step.get("output") or {}).get("id") == "permission_review"
    )
    assert frozen["permission_requirements"] == declaration["permission_requirements"]
    assert frozen["source_uri"].endswith("/skills/imported/permission-review")
    assert any(
        step["kind"] == "tool" and step["input"].get("tool") == "read_memory" for step in steps
    )
    artifact = runtime.store.get(
        ctx.account_id, "bot_file", {"file_id": result["result"]["file_ids"][0]}
    )
    assert artifact["state"] == "ready"
    assert "青禾" in runtime.files.read(ctx.account_id, artifact["storage_key"]).decode()
