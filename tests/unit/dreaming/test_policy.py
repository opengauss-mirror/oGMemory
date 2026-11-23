import pytest

from commit.dreaming_policy import Evidence, eligible_for_promotion, evidence_score


@pytest.mark.parametrize("evidence, score, independent", [
    ([Evidence("meeting-1", 4), Evidence("meeting-1", 2)], 0.8, False),
    ([Evidence("meeting-1", 2), Evidence("meeting-2", 2)], 0.8, True),
    ([Evidence("first-report", 4), Evidence("amended-report", 4, supports=False)], 0.2, False),
])
def test_evidence_is_scored_by_occurrence(evidence, score, independent):
    assert evidence_score(evidence) == pytest.approx(score)
    if independent:
        assert eligible_for_promotion(evidence)
