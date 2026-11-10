"""Typed hierarchy traversal selects matching children before its result limit."""

import pytest

from core.models import IndexRecord, RequestContext, RetrievalConfig, SeedHit, SeedResult, TypedQuery
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.hierarchical_searcher import HierarchicalSearcher


@pytest.mark.parametrize("category,directory", [("entity", "entities"), ("preference", "preferences")])
def test_hierarchy_selects_requested_type_from_mixed_parent(category, directory):
    root = "ctx://acct-1/users/u1/memories"
    index = InMemoryVectorIndex(dimension=2)
    index.upsert([
        IndexRecord(
            id=f"{name}-{level}", uri=f"{root}/{folder}" + ("/deploy" if level == 2 else ""),
            level=level, text="deploy",
            filters={"account_id": "acct-1", "owner_space": "user:u1"},
            metadata={"context_type": "MEMORY", "category": name,
                      "_embedding": [1, 0] if level == 2 else [0.6, 0.8]},
        ) for name, folder in [("entity", "entities"), ("preference", "preferences")]
        for level in (1, 2)
    ])
    ctx = RequestContext("acct-1", "u1", "a1", "s1", "t1", visible_owner_spaces=("user:u1",))
    searcher = HierarchicalSearcher(index, RetrievalConfig(hotness_alpha=0))
    seed = SeedResult(starting_points=[SeedHit(root, 0.0, level=0)], query_vector=[1, 0])

    hits = searcher.expand(TypedQuery("deploy", "MEMORY", [category], top_k=1), seed, ctx, limit=1)

    assert [(hit.uri, hit.category) for hit in hits] == [(f"{root}/{directory}/deploy", category)]
