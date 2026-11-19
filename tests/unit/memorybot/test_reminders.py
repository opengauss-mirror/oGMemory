import pytest

from memorybot.models import ChatInput
from memorybot.reminders import relative_reminder_seconds


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("一千二百分钟后提醒我喝水", 72000),
        ("提醒我再过一千零二分钟后喝水", 60120),
        ("通知我在半小时后开始", 1800),
        ("一 个 半 小时 30 分钟以后再执行", 7200),
        ("1.5小时二十分钟之后运行", 6600),
        ("一万二百分钟后提醒我。两分钟后提醒我整理。", 120),
    ],
)
def test_explicit_relative_reminder(text, seconds):
    assert relative_reminder_seconds(ChatInput(text=text).text) == seconds


def test_relative_reminder_after_pasted_note():
    text = "已保存记录 1" + " " * 384 + "。两分钟后提醒我归档。"

    assert relative_reminder_seconds(ChatInput(text=text).text) == 120
