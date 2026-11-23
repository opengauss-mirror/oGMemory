from datetime import UTC, datetime, timedelta

import pytest

from commit.dreaming_policy import (
    Evidence,
    Lifecycle,
    advance_lifecycle,
    eligible_for_promotion,
    evidence_score,
)


@pytest.mark.parametrize("evidence, score, independent", [
    ([Evidence("meeting-1", 4), Evidence("meeting-1", 2)], 0.8, False),
    ([Evidence("meeting-1", 2), Evidence("meeting-2", 2)], 0.8, True),
    ([Evidence("first-report", 4), Evidence("amended-report", 4, supports=False)], 0.2, False),
])
def test_evidence_is_scored_by_occurrence(evidence, score, independent):
    assert evidence_score(evidence) == pytest.approx(score)
    if independent:
        assert eligible_for_promotion(evidence)


def test_unused_memory_cools_retires_and_returns_after_restore():
    first_seen = datetime(2026, 7, 1, tzinfo=UTC)
    first_sweep = first_seen + timedelta(days=7)
    cold_day = first_sweep + timedelta(days=1)
    retiring_day = cold_day + timedelta(days=90)
    retired_day = retiring_day + timedelta(days=7)
    state = Lifecycle()

    for now in [first_sweep, cold_day, retiring_day, retired_day]:
        state = advance_lifecycle(state, now=now, first_seen=first_seen, heat=0, low_value=True)

    assert (state.tier, state.eligibility, state.cold_since) == ("cold", "retired", cold_day)

    restored = advance_lifecycle(state, now=retired_day, first_seen=first_seen, heat=0, restore=True)

    assert (restored.tier, restored.eligibility, restored.cold_since) == ("warm", "active", None)


def test_protected_memory_stays_active_during_cold_storage():
    first_seen = datetime(2026, 7, 1, tzinfo=UTC)
    cold_since = first_seen + timedelta(days=10)
    state = Lifecycle(tier="cold", cold_since=cold_since)

    result = advance_lifecycle(
        state, now=cold_since + timedelta(days=100), first_seen=first_seen, heat=0,
        low_value=True, protected=True,
    )

    assert (result.tier, result.eligibility) == ("cold", "active")
