"""Recall truncation boundaries for the first optimization batch."""

from server.memory_service import MemoryService


def test_default_gap_trims_at_ten_percent():
    items = [{"score": score} for score in [0.90, 0.89, 0.88, 0.78, 0.77]]
    assert MemoryService._truncate_by_score_gap(items) == items[:3]
    assert MemoryService._truncate_by_score_gap(items, gap_ratio=0.15) == items


def test_recall_keeps_three_items_despite_an_earlier_gap():
    items = [{"score": score} for score in [0.9, 0.4, 0.39]]
    assert MemoryService._truncate_by_score_gap(items) == items


def test_small_gaps_and_zero_scores_keep_all_items():
    for scores in ([0.9, 0.89, 0.88, 0.87], [0, 0, 0, 0]):
        items = [{"score": score} for score in scores]
        assert MemoryService._truncate_by_score_gap(items) == items
