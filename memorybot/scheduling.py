"""UTC schedule expansion with local weekly calendar rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

UTC = UTC


def as_utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("schedule timestamps require a timezone")
    return value.astimezone(UTC)


def _local_instant(local: datetime, zone: ZoneInfo) -> datetime:
    def valid(candidate: datetime) -> bool:
        return (
            candidate.replace(tzinfo=zone, fold=0)
            .astimezone(UTC)
            .astimezone(zone)
            .replace(tzinfo=None)
            == candidate
        )

    if not valid(local):
        low, high = 0, 172800
        while low < high:
            middle = (low + high) // 2
            if valid(local + timedelta(seconds=middle)):
                high = middle
            else:
                low = middle + 1
        local += timedelta(seconds=low)
    return local.replace(tzinfo=zone, fold=0).astimezone(UTC)


def next_run(
    rule: dict, timezone: str, after: datetime | str, created_at: datetime | str | None = None
) -> datetime | None:
    """Return the first scheduled instant strictly after ``after``."""
    zone, after = ZoneInfo(timezone), as_utc(after)
    kind = rule["type"]
    if kind == "once":
        instant = as_utc(rule["at"])
        return instant if instant > after else None
    if kind == "interval":
        seconds = rule["interval_seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int)
            or not 60 <= seconds <= 31536000
        ):
            raise ValueError("interval_seconds must be between 60 and 31536000")
        anchor = as_utc(created_at) if created_at is not None else after
        count = max(1, int((after - anchor).total_seconds() // seconds) + 1)
        return anchor + timedelta(seconds=count * seconds)
    if kind == "weekly":
        weekdays = set(rule["weekdays"])
        if not weekdays or not weekdays <= set(range(1, 8)):
            raise ValueError("weekdays must use ISO weekdays 1 through 7")
        hour, minute, second = map(int, rule["local_time"].split(":"))
        date = after.astimezone(zone).date()
        for offset in range(15):
            day = date + timedelta(days=offset)
            if day.isoweekday() in weekdays:
                instant = _local_instant(
                    datetime(day.year, day.month, day.day, hour, minute, second), zone
                )
                if instant > after:
                    return instant
    raise ValueError("schedule rule must be once, interval, or weekly")


def due_run(
    rule: dict,
    timezone: str,
    next_run_at: datetime | str,
    now: datetime | str,
    created_at: datetime | str,
) -> datetime | None:
    """Coalesce periodic backlog to its latest due instant."""
    due, now = as_utc(next_run_at), as_utc(now)
    if due > now:
        return None
    if rule["type"] == "once":
        return due
    if rule["type"] == "interval":
        seconds = rule["interval_seconds"]
        return due + timedelta(seconds=int((now - due).total_seconds() // seconds) * seconds)
    zone = ZoneInfo(timezone)
    hour, minute, second = map(int, rule["local_time"].split(":"))
    local_date = now.astimezone(zone).date()
    for offset in range(15):
        day = local_date - timedelta(days=offset)
        if day.isoweekday() in rule["weekdays"]:
            instant = _local_instant(
                datetime(day.year, day.month, day.day, hour, minute, second), zone
            )
            if due <= instant <= now:
                return instant
    return due
