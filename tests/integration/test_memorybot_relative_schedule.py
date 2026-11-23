from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from memorybot.models import ChatInput
from memorybot.reminders import relative_reminder_seconds
from tests.integration.test_memorybot_runtime import execute, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("请在一分钟后提醒我检查验收报告", 60),
        ("提醒我在1分钟后检查验收报告", 60),
        ("半小时后提醒我检查验收报告", 1800),
        ("一个半小时后提醒我检查验收报告", 5400),
        ("一小时三十分钟后提醒我检查验收报告", 5400),
        ("提醒我在1小时30分钟后检查验收报告", 5400),
        ("请在两天三小时十五分钟后执行验收检查", 184500),
        ("二十五分钟后提醒我检查验收报告", 1500),
        ("两天后执行验收检查", 172800),
        ("0.5分钟后提醒我检查验收报告", 30),
    ],
)
def test_explicit_relative_reminder_duration(text, seconds):
    assert relative_reminder_seconds(text) == seconds


def test_one_minute_reminder_uses_service_clock_and_runs(runtime):
    runtime.llm.planned_calls.append(
        {
            "tool": "create_schedule",
            "input": {
                "name": "验收报告提醒",
                "task_input": {"text": "请检查验收报告。"},
                "rule": {"type": "after", "delay_seconds": 60},
            },
        }
    )
    with runtime.store.transaction(runtime.ctx.account_id) as tx:
        started = tx.now()
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="请在一分钟后提醒我检查验收报告。").model_dump(mode="json"),
        ),
    )
    schedule = runtime.store.list(runtime.ctx.account_id, "bot_schedule")[0]
    due = datetime.fromisoformat(schedule["rule"]["at"])
    with runtime.store.transaction(runtime.ctx.account_id) as tx:
        completed = tx.now()
    assert started + timedelta(seconds=60) <= due <= completed + timedelta(seconds=60)
    assert schedule["session_id"] == result["session_id"]
    system = runtime.llm.messages[0][0]["content"]
    assert "当前 Asia/Shanghai 时间：" in system
    with runtime.store.transaction(runtime.ctx.account_id) as tx:
        arrived = tx.now() - timedelta(seconds=1)
        tx.update(
            "bot_schedule",
            {"schedule_id": schedule["schedule_id"]},
            {"next_run_at": arrived, "rule": {"type": "once", "at": arrived.isoformat()}},
        )
    runtime.worker.run_once()
    run = runtime.store.list(
        runtime.ctx.account_id, "bot_task", {"schedule_id": schedule["schedule_id"]}
    )[0]
    finished = execute(runtime, run)
    assert finished["state"] == "succeeded"
    assert finished["session_id"] == result["session_id"]
