from datetime import UTC, datetime

from memorybot.scheduling import due_run, next_run
from memorybot.store import json_value


def test_interval_uses_original_anchor_and_coalesces_backlog():
    rule = {"type": "interval", "interval_seconds": 120}
    anchor = datetime(2026, 11, 3, 8, 0, tzinfo=UTC)
    after = datetime(2026, 11, 3, 8, 7, tzinfo=UTC)
    assert next_run(rule, "Asia/Shanghai", after, anchor).isoformat() == "2026-11-03T08:08:00+00:00"
    assert (
        due_run(rule, "Asia/Shanghai", "2026-11-03T08:02:00Z", after, anchor).isoformat()
        == "2026-11-03T08:06:00+00:00"
    )


def test_weekly_uses_local_time_and_first_fall_back_occurrence():
    rule = {"type": "weekly", "weekdays": [7], "local_time": "01:30:00"}
    assert (
        next_run(rule, "America/New_York", "2026-11-01T00:00:00Z").isoformat()
        == "2026-11-01T05:30:00+00:00"
    )
    assert (
        next_run(rule, "America/New_York", "2026-11-01T05:30:00Z").isoformat()
        == "2026-11-08T06:30:00+00:00"
    )


def test_weekly_spring_gap_moves_to_first_valid_local_time():
    rule = {"type": "weekly", "weekdays": [7], "local_time": "02:30:00"}
    assert (
        next_run(rule, "America/New_York", "2026-03-08T00:00:00Z").isoformat()
        == "2026-03-08T07:00:00+00:00"
    )


def test_once_keeps_original_due_time_after_restart():
    rule = {"type": "once", "at": "2026-11-03T16:00:00+08:00"}
    assert (
        next_run(rule, "Asia/Shanghai", "2026-11-03T07:00:00Z").isoformat()
        == "2026-11-03T08:00:00+00:00"
    )
    assert (
        due_run(
            rule, "Asia/Shanghai", rule["at"], "2026-11-04T00:00:00Z", "2026-11-01T00:00:00Z"
        ).isoformat()
        == "2026-11-03T08:00:00+00:00"
    )


def test_persistent_response_serializes_nested_utc_times():
    now = datetime(2026, 11, 3, 8, 0, tzinfo=UTC)
    assert json_value({"items": [{"created_at": now}]}) == {
        "items": [{"created_at": "2026-11-03T08:00:00Z"}]
    }
