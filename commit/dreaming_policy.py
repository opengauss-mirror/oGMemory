"""Deterministic checks shared by Dreaming previews and writes."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Literal


@dataclass(frozen=True)
class Evidence:
    occurrence_id: str
    weight: Literal[0, 2, 4]
    supports: bool = True


def _occurrence_weights(evidence: Iterable[Evidence]) -> dict[str, tuple[int, int]]:
    weights: dict[str, tuple[int, int]] = {}
    for item in evidence:
        if not item.occurrence_id or item.weight == 0:
            continue
        support, refutation = weights.get(item.occurrence_id, (0, 0))
        if item.supports:
            support = max(support, item.weight)
        else:
            refutation = max(refutation, item.weight)
        weights[item.occurrence_id] = (support, refutation)
    return weights


def evidence_score(evidence: Iterable[Evidence]) -> float | None:
    weights = _occurrence_weights(evidence)
    if not weights:
        return None
    total = sum(support - refutation for support, refutation in weights.values())
    log_odds = min(8, max(-8, -2 + total))
    return 1 / (1 + 2 ** (-log_odds))


def eligible_for_promotion(evidence: Iterable[Evidence]) -> bool:
    """Require two independent observations that support the same claim."""
    weights = _occurrence_weights(evidence)
    return len(weights) >= 2 and all(
        support > 0 and refutation == 0 for support, refutation in weights.values()
    )


def valid_at(at: datetime, *, valid_from: datetime | None = None, valid_to: datetime | None = None) -> bool:
    return (valid_from is None or valid_from <= at) and (valid_to is None or at < valid_to)


@dataclass(frozen=True)
class Lifecycle:
    tier: Literal["hot", "warm", "cold"] = "warm"
    eligibility: Literal["active", "retiring", "retired"] = "active"
    cold_since: datetime | None = None
    retiring_since: datetime | None = None
    last_heat_day: date | None = None
    low_windows: int = 0


def advance_lifecycle(
    state: Lifecycle,
    *,
    now: datetime,
    first_seen: datetime,
    heat: float,
    low_value: bool = False,
    protected: bool = False,
    adopted: bool = False,
    restore: bool = False,
) -> Lifecycle:
    """Advance recall eligibility and temperature using UTC daily observations."""
    day = now.astimezone(UTC).date()
    if restore or adopted:
        return Lifecycle(tier="hot" if adopted else "warm", last_heat_day=day)
    if state.eligibility == "retired":
        return state

    if state.last_heat_day is None or state.last_heat_day < day:
        state = _advance_temperature(state, now, first_seen, day, heat)
    if (
        protected or not low_value or state.cold_since is None
        or now - state.cold_since < timedelta(days=90)
    ):
        if state.eligibility == "retiring":
            return replace(state, eligibility="active", retiring_since=None)
        return state
    if state.retiring_since is None:
        return replace(state, eligibility="retiring", retiring_since=now)
    if now - state.retiring_since >= timedelta(days=7):
        return replace(state, eligibility="retired")
    return state


def _advance_temperature(
    state: Lifecycle, now: datetime, first_seen: datetime, day: date, heat: float,
) -> Lifecycle:
    if heat >= 0.45:
        return Lifecycle(tier="hot", last_heat_day=day)
    consecutive = state.last_heat_day == day - timedelta(days=1)
    previous_windows = state.low_windows if consecutive else 0
    windows = previous_windows + 1 if heat < 0.15 else 0
    if state.tier == "hot":
        return replace(state, tier="warm", last_heat_day=day, low_windows=windows)
    if windows >= 2 and now - first_seen >= timedelta(days=7) and state.tier == "warm":
        return replace(state, tier="cold", cold_since=now, last_heat_day=day, low_windows=windows)
    return replace(state, last_heat_day=day, low_windows=windows)
