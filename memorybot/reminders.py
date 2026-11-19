"""Resolve explicit relative reminder durations against the service clock."""

from __future__ import annotations

import re

_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
_PLACE_VALUES = {"十": 10, "百": 100, "千": 1000}
_UNIT_SECONDS = {"秒钟": 1, "秒": 1, "分钟": 60, "小时": 3600, "天": 86400, "周": 604800}
_CHINESE_NUMERALS = "".join(_DIGITS) + "".join(_PLACE_VALUES) + "万萬亿億兆壹贰貳叁參肆伍陆陸柒捌玖拾佰仟点點"
_DIGIT_CHARS = rf"\d{_CHINESE_NUMERALS}"
_NUMBER = rf"\d++(?:\.\d++)?|[{_CHINESE_NUMERALS}]++|半"
_PART = re.compile(
    rf"(?<![{_DIGIT_CHARS}])(?<![{_DIGIT_CHARS}]\.)(?P<amount>{_NUMBER})"
    rf"(?:\s*+个)?(?:\s*+(?P<half>半))?\s*+(?P<unit>{'|'.join(_UNIT_SECONDS)})"
)
_DURATION_END = re.compile(r"\s*+(?:之后|以后|后)")
_INTENT_BEFORE = re.compile(r"(?:提醒|通知)我\s*+(?:(?:在|再过|过)\s*+)?")
_INTENT_AFTER = re.compile(r"\s*+(?:再)?(?:提醒|通知|执行|运行)")


def _amount_value(amount: str) -> float | None:
    if amount == "半":
        return 0.5
    if amount[0].isdecimal():
        return float(amount)

    value, digit = 0, 0
    for char in amount:
        if char in _PLACE_VALUES:
            value += (digit or 1) * _PLACE_VALUES[char]
            digit = 0
        elif char in _DIGITS:
            digit = _DIGITS[char]
        else:
            return None
    return value + digit


def relative_reminder_seconds(text: str) -> float | None:
    """Read a duration attached to an explicit reminder or execution request."""
    intent_starts = {intent.end() for intent in _INTENT_BEFORE.finditer(text)}
    duration_start, duration_end = 0, 0
    seconds: float | None = 0.0

    for part in _PART.finditer(text):
        gap = text[duration_end : part.start()]
        if not duration_end or (gap and not gap.isspace()):
            duration_start, seconds = part.start(), 0.0
        duration_end = part.end()

        value = _amount_value(part["amount"])
        if value is None:
            seconds = None
        elif seconds is not None:
            seconds += (value + (0.5 if part["half"] else 0)) * _UNIT_SECONDS[part["unit"]]

        if seconds is not None and 0 < seconds <= 31536000:
            ending = _DURATION_END.match(text, duration_end)
            if ending and (
                duration_start in intent_starts or _INTENT_AFTER.match(text, ending.end())
            ):
                return seconds

    return None
