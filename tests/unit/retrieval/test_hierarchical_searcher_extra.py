"""Additional tests for retrieval.hierarchical_searcher."""

from __future__ import annotations

from datetime import datetime, timezone

from core.models import RetrievalConfig, SeedHit, SeedResult, TypedQuery
from retrieval.hierarchical_searcher import HierarchicalSearcher
from tests.unit.retrieval.conftest import make_ctx


class FakeVectorIndex:
    def __init__(self, mapping: dict[str, list[SeedHit]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, dict, int]] = []

    def search_children(self, parent_uri, query_vector, filters, top_k):
        self.calls.append((parent_uri, filters, top_k))
        return self.mapping.get(parent_uri, [])


def test_expand_walks_hierarchy_and_returns_sorted_leaf_hits():
    mapping = {
        "ctx://acct-1/users/u1/memories/preferences": [
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/preferences/coffee",
                score=0.9,
                level=2,
                parent_uri="ctx://acct-1/users/u1/memories/preferences",
                category="preferences",
                owner_space="user:u1",
                abstract="coffee",
                has_overview=True,
                has_content=True,
            ),
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/preferences/nested",
                score=0.8,
                level=1,
                parent_uri="ctx://acct-1/users/u1/memories/preferences",
                category="preferences",
                owner_space="user:u1",
            ),
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/preferences/ignore",
                score=0.1,
                level=2,
                parent_uri="ctx://acct-1/users/u1/memories/preferences",
                category="preferences",
                owner_space="user:u1",
            ),
        ],
        "ctx://acct-1/users/u1/memories/preferences/nested": [
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/preferences/nested/tea",
                score=0.95,
                level=2,
                parent_uri="ctx://acct-1/users/u1/memories/preferences/nested",
                category="preferences",
                owner_space="user:u1",
                abstract="tea",
                has_content=True,
            )
        ],
    }
    vector_index = FakeVectorIndex(mapping)
    searcher = HierarchicalSearcher(
        vector_index,
        RetrievalConfig(
            max_convergence_rounds=2,
            score_propagation_alpha=0.5,
            hotness_alpha=0.0,
            default_score_threshold=0.5,
        ),
    )
    seed_result = SeedResult(
        query_vector=[0.1, 0.2],
        initial_candidates=[
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/profile",
                score=0.7,
                level=2,
                category="profile",
                owner_space="user:u1",
                abstract="profile",
            )
        ],
        starting_points=[
            SeedHit(
                uri="ctx://acct-1/users/u1/memories/preferences",
                score=0.8,
                level=1,
                parent_uri="ctx://acct-1/users/u1/memories",
            )
        ],
    )

    results = searcher.expand(
        TypedQuery(text="coffee", context_type="MEMORY", categories=["preferences"], owner_space="user:u1"),
        seed_result,
        make_ctx(),
        limit=3,
        score_threshold=0.5,
    )

    uris = [hit.uri for hit in results]
    assert "ctx://acct-1/users/u1/memories/preferences/ignore" not in uris
    assert "ctx://acct-1/users/u1/memories/preferences/coffee" in uris
    assert "ctx://acct-1/users/u1/memories/preferences/nested/tea" in uris
    assert results[0].uri == "ctx://acct-1/users/u1/memories/preferences/nested/tea"
    assert vector_index.calls[0][1] == {
        "account_id": "acct-1",
        "context_type": "MEMORY",
        "owner_space": "user:u1",
        "category": ["preferences"],
    }


def test_recursive_search_stops_after_convergence():
    repeated_leaf = SeedHit(
        uri="ctx://acct-1/users/u1/memories/preferences/coffee",
        score=0.9,
        level=2,
        parent_uri="ctx://acct-1/users/u1/memories/preferences",
    )
    vector_index = FakeVectorIndex(
        {
            "ctx://acct-1/users/u1/memories/preferences": [repeated_leaf],
            "ctx://acct-1/users/u1/memories/preferences_alt": [repeated_leaf],
        }
    )
    searcher = HierarchicalSearcher(
        vector_index,
        RetrievalConfig(max_convergence_rounds=1, hotness_alpha=0.0),
    )

    results = searcher._recursive_search(
        query_vector=[0.1],
        starting_points=[
            SeedHit(uri="ctx://acct-1/users/u1/memories/preferences", score=0.8, level=1),
            SeedHit(uri="ctx://acct-1/users/u1/memories/preferences_alt", score=0.7, level=1),
        ],
        initial_candidates=[],
        typed_query=TypedQuery(text="coffee", context_type="MEMORY", categories=[]),
        ctx=make_ctx(),
        limit=1,
        threshold=None,
    )

    assert len(results) == 1
    assert len(vector_index.calls) == 2


def test_convert_results_blends_hotness_and_tolerates_invalid_timestamps():
    searcher = HierarchicalSearcher(
        FakeVectorIndex({}),
        RetrievalConfig(hotness_alpha=0.5, hotness_half_life_days=7.0),
    )

    results = searcher._convert_results(
        [
            {
                "uri": "one",
                "level": 2,
                "_final_score": 0.4,
                "active_count": 10,
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "uri": "two",
                "level": 2,
                "_final_score": 0.9,
                "active_count": 0,
                "updated_at": "not-a-date",
            },
        ],
        limit=2,
    )

    assert {hit.uri for hit in results} == {"one", "two"}
    assert results[0].score >= results[1].score
