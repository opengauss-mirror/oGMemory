"""Hotness scoring for cold/hot memory lifecycle management.

Pure function — ``score = log1p(active_count)/(log1p(active_count)+1) * exp_decay(age)``.

freq range is [0.0, 1.0): 0 when never accessed, approaches 1 as access count grows.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

DEFAULT_HALF_LIFE_DAYS: float = 7.0


def blend_hotness(score: float, active_count: int, updated_at, config) -> float:
    """Apply the existing per-node blend without hierarchy propagation."""
    if isinstance(updated_at, str):
        try:
            updated_at = datetime.fromisoformat(updated_at)
        except (ValueError, TypeError):
            updated_at = None
    if not isinstance(updated_at, datetime):
        updated_at = None
    heat = hotness_score(active_count, updated_at, half_life_days=config.hotness_half_life_days)
    return (1 - config.hotness_alpha) * score + config.hotness_alpha * heat


def hotness_score(
    active_count: int,
    updated_at: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)

    freq = math.log1p(max(active_count, 0)) / (math.log1p(max(active_count, 0)) + 1)

    if updated_at is None:
        return 0.0

    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = max((now - updated_at).total_seconds() / 86_400.0, 0.0)
    decay_rate = math.log(2) / max(half_life_days, 1e-9)
    recency = math.exp(-decay_rate * age_days)

    return freq * recency
