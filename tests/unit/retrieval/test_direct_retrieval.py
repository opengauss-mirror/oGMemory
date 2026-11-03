"""Issue 70: disabled hierarchy searches L2 directly through the original assembly."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.lexical import LexicalHit
from core.models import IndexRecord, RetrievalConfig, SeedHit, TypedQuery
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from tests.unit.retrieval.conftest import make_ctx

ROOT = "ctx://acct-1/users/u1/memories/"
NODE = ROOT + "preferences/coding"
QUERY = "coding preferences"


def build(*, expand=False, layers=(0, 1, 2), lexical=None, **kwargs):
    cfg = RetrievalConfig(expand_hierarchy=expand, hotness_alpha=0,
                          default_score_threshold=-2, **kwargs)
    embedder = MockEmbedder(dimension=8)
    vector = embedder.embed_texts([QUERY])[0]
    index = InMemoryVectorIndex(dimension=8)
    index.upsert([
        IndexRecord(id=f"record-{level}", uri=NODE + ("/content.md" if level == 2 else "/"),
                    level=level, text=f"L{level} evidence",
                    filters={"account_id": "acct-1", "owner_space": "user_u1"},
                    metadata={"context_type": "MEMORY", "category": "preference",
                              "parent_uri": ROOT, "_embedding": vector})
        for level in layers
    ])
    index.search_by_vector = Mock(wraps=index.search_by_vector)
    index.search_children = Mock(wraps=index.search_children)
    searcher = HierarchicalSearcher(index, cfg)
    searcher.expand = Mock(wraps=searcher.expand)
    backfill = SimpleNamespace(ensure_ready=lambda *args: True)
    seed = SeedRetriever(index, embedder, cfg, lexical, backfill)
    pipe = RetrievalPipeline(QueryPlanner(cfg), seed, searcher, ResultRanker(cfg), cfg)
    return pipe, seed, searcher, index


@pytest.mark.parametrize("mode", ["quick", "thinking"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_false_searches_only_l2_without_expansion(mode, hybrid):
    pipe, seed, searcher, index = build(hybrid_enabled=hybrid, lexical_weight=0)
    result = pipe.run(QUERY, make_ctx(), top_k=5, mode=mode)
    assert [(h.uri, h.level_hit, h.abstract) for h in result.hits] == [
        (NODE + "/content.md", "L2", "L2 evidence")]
    assert "direct_retrieval" not in result.hits[0].metadata
    assert "matched_levels" not in result.hits[0].metadata
    assert index.search_by_vector.call_args.kwargs["filters"]["level"] == 2
    sr = seed.search(TypedQuery(QUERY, "MEMORY", [], top_k=5), make_ctx())
    assert sr.root_uris == sr.starting_points == []
    assert [h.level for h in sr.initial_candidates] == [2]
    assert searcher.expand.call_count == index.search_children.call_count == 0
    assert "hierarchical" not in [s.stage for s in result.trace.stages]


@pytest.mark.parametrize("layers", [(), (0,), (1,), (0, 1)])
@pytest.mark.parametrize("mode", ["quick", "thinking"])
def test_false_empty_l2_never_falls_back_to_summaries_or_expansion(layers, mode):
    pipe, _, searcher, index = build(layers=layers)
    assert pipe.run(QUERY, make_ctx(), mode=mode).hits == []
    assert searcher.expand.call_count == index.search_children.call_count == 0


def test_default_true_retains_mixed_search_and_expansion():
    pipe, _, searcher, index = build(expand=True)
    assert pipe.run(QUERY, make_ctx()).hits
    assert index.search_by_vector.call_args.kwargs["filters"]["level"] == [0, 1, 2]
    assert searcher.expand.call_count > 0
    assert index.search_children.call_count > 0


class Lexical:
    supported_filters = frozenset({"account_id", "owner_space", "level", "context_type", "category"})
    doc_count = 3

    def __init__(self, support_level=True):
        self.filters = None
        if not support_level:
            self.supported_filters = self.supported_filters - {"level"}

    def dirty_generations(self, *args):
        return {}

    def search(self, query, top_k, filters):
        self.filters = filters
        return [LexicalHit(f"lexical-{level}", 5.0, {
            "uri": NODE + ("/content.md" if level == 2 else "/"), "level": level,
            "account_id": "acct-1", "owner_space": "user_u1", "context_type": "MEMORY",
            "category": "preference", "abstract": f"L{level} abstract", "has_content": level == 2,
        }) for level in (0, 1, 2) if "level" not in filters or filters["level"] == level]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("semantic_weight", [0, 1])
@pytest.mark.parametrize("support_level", [False, True])
def test_false_lexical_path_requires_initial_l2_filter(hybrid, semantic_weight, support_level):
    lexical = Lexical(support_level)
    pipe, seed, searcher, index = build(layers=(), lexical=lexical,
                                      hybrid_enabled=hybrid, semantic_weight=semantic_weight)
    if not support_level:
        with pytest.raises(RuntimeError, match="level filter"):
            seed.search(TypedQuery(QUERY, "MEMORY", [], top_k=5), make_ctx())
        assert lexical.filters is None
        return
    sr = seed.search(TypedQuery(QUERY, "MEMORY", [], top_k=5), make_ctx())
    assert sr.starting_points == sr.root_uris == []
    assert [h.level for h in sr.initial_candidates] == [2]
    if support_level:
        assert lexical.filters["level"] == 2
    result = pipe.run(QUERY, make_ctx())
    assert [(h.uri, h.level_hit, h.abstract) for h in result.hits] == [(NODE + "/content.md", "L2", "L2 abstract")]
    assert searcher.expand.call_count == index.search_children.call_count == 0


@pytest.mark.parametrize("threshold", [.9, 1, 1.1])
def test_false_threshold_cannot_be_bypassed_by_assembly_fallback(threshold):
    pipe, _, searcher, index = build(layers=(2,))
    result = pipe.run(QUERY, make_ctx(), score_threshold=threshold)
    assert len(result.hits) == (1 if threshold < 1 else 0)
    assert searcher.expand.call_count == index.search_children.call_count == 0


def test_false_semantic_keeps_local_hotness_without_parent_propagation():
    pipe, seed, searcher, index = build(layers=(2,))
    pipe.cfg.hotness_alpha = .2
    seed._global_vector_search = Mock(return_value=[
        SeedHit(NODE + "/content.md", 1, 2, abstract="body", active_count=0,
                updated_at=datetime.now(timezone.utc).isoformat())])
    result = pipe.run(QUERY, make_ctx())
    assert result.hits[0].score == pytest.approx(.8)
    assert searcher.expand.call_count == index.search_children.call_count == 0


def test_false_lexical_only_keeps_score_without_heat_or_embedding():
    pipe, seed, _, _ = build(layers=(), lexical=Lexical(), hybrid_enabled=True, semantic_weight=0)
    pipe.cfg.hotness_alpha = .9
    seed.embedder.embed_texts = Mock(side_effect=AssertionError("lexical-only must not embed"))
    result = pipe.run(QUERY, make_ctx())
    assert result.hits[0].score == pytest.approx(1)
