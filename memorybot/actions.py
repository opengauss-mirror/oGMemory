"""Explicit user actions and schedule lookup for conversations."""

import re
from datetime import timedelta
from typing import Literal

from pydantic import Field

from memorybot.models import BotError, Contract, MemoryAction, ScheduleCreate, SchedulePatch


class RelativeReminderRule(Contract):
    type: Literal["after"]
    delay_seconds: float = Field(gt=0, le=31_536_000, strict=True, allow_inf_nan=False)


def action_tools(text):
    allowed_actions = {}
    if re.search(r"提醒|定时|每(?:天|周|月|隔)|remind|schedule", text, re.IGNORECASE):
        allowed_actions["create_schedule"] = []
    changes = {
        "pause": r"暂停|pause",
        "resume": r"恢复|继续|resume",
        "cancel": r"取消|cancel",
        "update": r"更改|修改|调整|改为|更新|change|update",
    }
    actions = [name for name, pattern in changes.items() if re.search(pattern, text, re.IGNORECASE)]
    if actions and re.search(r"提醒|计划|定时|schedule|reminder", text, re.IGNORECASE):
        allowed_actions["update_schedule"] = actions
    memory = {
        "remember": r"记住|记下|remember",
        "correct": r"更正|纠正|更改.*记忆|更新.*记忆|correct",
        "forget": r"忘记|遗忘|删除.*记忆|forget",
    }
    actions = [name for name, pattern in memory.items() if re.search(pattern, text, re.IGNORECASE)]
    if actions:
        allowed_actions["memory_action"] = actions
    schemas = {
        "create_schedule": ScheduleCreate.model_json_schema(),
        "update_schedule": SchedulePatch.model_json_schema(),
        "memory_action": MemoryAction.model_json_schema(),
    }
    create_schema = schemas["create_schedule"]
    create_schema["required"].remove("session_id")
    create_schema["required"].remove("timezone")
    create_schema["$defs"]["RelativeReminderRule"] = RelativeReminderRule.model_json_schema()
    relative_ref = "#/$defs/RelativeReminderRule"
    rule_schema = create_schema["properties"]["rule"]
    rule_schema["oneOf"].append({"$ref": relative_ref})
    rule_schema["discriminator"]["mapping"]["after"] = relative_ref
    schemas["update_schedule"]["properties"]["schedule_id"] = {
        "type": "string",
        "maxLength": 64,
    }
    schemas["update_schedule"]["required"].append("schedule_id")
    descriptions = {
        "create_schedule": (
            "按用户明确要求创建提醒或周期任务。默认当前会话和 Asia/Shanghai 时区。"
            "每项提醒单独调用：相对时间使用 rule.type=after 和 delay_seconds，"
            "绝对时间使用 rule.type=once 和 at。"
        ),
        "update_schedule": "按用户明确要求调整、暂停、恢复或取消已有计划，使用真实计划标识及修订号。",
        "memory_action": "按用户明确要求记住、更正或遗忘记忆。更正和遗忘使用读取的真实URI；返回排队受理结果。",
    }
    return allowed_actions, [
        {
            "name": name,
            "description": descriptions[name],
            "input_schema": schemas[name],
        }
        for name in allowed_actions
    ]


def managed_tool(execution, value, operation_id, waiting, tx, allowed):
    from memorybot.service import BotService

    name, arguments = value["tool"], dict(value["input"])
    if name not in allowed or (
        name != "create_schedule" and arguments.get("action") not in allowed[name]
    ):
        raise BotError("PERMISSION_DENIED", "操作超出本条用户明确要求", 403)
    ctx = execution.gateway.context(execution.task)
    app = BotService(
        execution.store,
        execution.gateway.service,
        execution.files,
        execution.config,
        gateway=execution.gateway,
    )
    if name == "create_schedule":
        arguments.setdefault("session_id", execution.task["session_id"])
        arguments.setdefault("timezone", "Asia/Shanghai")
        delivery = execution.task["input"].get("delivery") or {"channel": "web"}
        arguments.setdefault("task_input", {}).setdefault(
            "delivery", {key: delivery[key] for key in ("channel", "target_id") if key in delivery}
        )
        rule = arguments.get("rule")
        if isinstance(rule, dict) and rule.get("type") == "after":
            delay = RelativeReminderRule.model_validate(rule).delay_seconds
            arguments["rule"] = {
                "type": "once",
                "at": (tx.now() + timedelta(seconds=delay)).isoformat(),
            }
        data = ScheduleCreate.model_validate(arguments).model_dump(mode="json")
        status, result = app.create_schedule(ctx, tx, data)
    elif name == "update_schedule":
        schedule_id = arguments.pop("schedule_id")
        data = SchedulePatch.model_validate(arguments).model_dump(mode="json", exclude_unset=True)
        status, result = app.patch_schedule(ctx, tx, schedule_id, data)
    else:
        data = MemoryAction.model_validate(arguments).model_dump(mode="json")
        status, result = app.memory_action(ctx, tx, data, operation_id)
        # Link the child so memory and indexing updates reach the parent task.
        tx.update(
            "bot_task",
            {"task_id": result["task"]["task_id"]},
            {"parent_task_id": execution.task["task_id"]},
        )
        execution.store.set_memory_status(tx, execution.task["task_id"], "pending")
    return {"operation": name, "status": status, "data": result}


def list_schedules(execution):
    rows = execution.rows("bot_schedule", order_by="updated_at DESC", limit=101)
    return {
        "schedules": [
            {
                key: row[key]
                for key in (
                    "schedule_id",
                    "session_id",
                    "name",
                    "state",
                    "next_run_at",
                    "timezone",
                    "revision",
                )
            }
            for row in rows[:100]
        ],
        "has_more": len(rows) > 100,
    }
