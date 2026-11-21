from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from memorybot.actions import action_tools, managed_tool
from memorybot.config import BotConfig
from memorybot.service import BotService


@pytest.mark.parametrize(
    ("text", "second_rule", "second_delay"),
    [
        ("十分钟后提醒我喝水和三十分钟后开会", {"type": "after", "delay_seconds": 1800}, 1800),
        (
            "十分钟后提醒我喝水和下周一上午九时开会",
            {"type": "once", "at": "2026-11-23T01:00:00Z"},
            172800,
        ),
    ],
    ids=["two-relative-times", "relative-and-absolute-time"],
)
def test_managed_schedule_uses_each_reminders_own_rule(monkeypatch, text, second_rule, second_delay):
    now = datetime(2026, 11, 21, 1, tzinfo=UTC)
    execution = SimpleNamespace(
        task={"session_id": "calendar", "input": {"text": text}},
        gateway=SimpleNamespace(context=lambda task: object(), service=None),
        store=None,
        files=None,
        config=BotConfig(),
    )
    monkeypatch.setattr(BotService, "create_schedule", lambda self, ctx, tx, data: (201, data))
    tx = SimpleNamespace(now=lambda: now)
    allowed, tools = action_tools(text)
    schema = next(tool["input_schema"] for tool in tools if tool["name"] == "create_schedule")
    validator = Draft202012Validator(schema)
    for name, rule, delay in (
        ("喝水", {"type": "after", "delay_seconds": 600}, 600),
        ("开会", second_rule, second_delay),
    ):
        payload = {"name": name, "rule": rule, "task_input": {"text": name}}
        validator.validate(payload)
        result = managed_tool(
            execution, {"tool": "create_schedule", "input": payload}, "test", None, tx, allowed
        )
        assert result["status"] == 201
        assert datetime.fromisoformat(result["data"]["rule"]["at"]) == now + timedelta(seconds=delay)
