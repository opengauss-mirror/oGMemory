"""Deterministic checks shared by Dreaming previews and writes."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
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
