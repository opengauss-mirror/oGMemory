"""Integration-level unit tests for retrieval.pipeline.RetrievalPipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.models import (
    IndexRecord,
    RetrievalConfig,
    RequestContext,
    SearchMemoryResult,
    SeedHit,
    SeedResult,
    TypedQuery,
    RetrievedBlock,
)
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder
from retrieval.result_ranker import ResultRanker
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.trace import RetrievalTrace
from tests.unit.retrieval.conftest import make_ctx


def _build_pipeline(
    idx: InMemoryVectorIndex,
    emb: MockEmbedder,
    cfg: RetrievalConfig,
    *,
    with_expand: bool = True,
) -> RetrievalPipeline:
    planner = QueryPlanner(cfg)
    seed = SeedRetriever(idx, emb, cfg)
    expand = HierarchicalSearcher(idx, cfg) if with_expand else None
    assembly = ResultRanker(cfg)
    return RetrievalPipeline(planner, seed, expand, assembly, cfg)


def _seed_records(idx: InMemoryVectorIndex) -> None:
    records = [
        IndexRecord(
            id="a0", uri="ctx://acct-1/users/u1/memories", level=0,
            text="user memory root with profile and preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile"},
        ),
        IndexRecord(
            id="a1", uri="ctx://acct-1/users/u1/memories/profile", level=1,
            text="user profile overview with coding preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories"},
        ),
        IndexRecord(
            id="a2", uri="ctx://acct-1/users/u1/memories/profile/detail", level=2,
            text="detailed user profile about coding habits and preference for Python",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories/profile"},
        ),
        IndexRecord(
            id="a3", uri="ctx://acct-1/users/u1/memories/entities/openai", level=2,
            text="OpenAI is a leading AI research company",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "entities", "parent_uri": "ctx://acct-1/users/u1/memories/entities"},
        ),
    ]
    idx.upsert(records)


@pytest.fixture()
def cfg():
    return RetrievalConfig(
        global_search_topk=20,
        default_top_k=5,
        max_convergence_rounds=2,
        score_propagation_alpha=0.5,
        hotness_alpha=0.0,
    )


@pytest.fixture()
def ctx():
    return make_ctx()


@pytest.fixture()
def idx():
    vi = InMemoryVectorIndex(dimension=8)
    _seed_records(vi)
    return vi


@pytest.fixture()
def emb():
    return MockEmbedder(dimension=8)


class TestPipeline:
    def test_returns_search_memory_result(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("用户的偏好是什么", ctx)
        assert isinstance(result, SearchMemoryResult)

    def test_hits_are_l2_only(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("用户的偏好是什么", ctx)
        for block in result.hits:
            assert block.level_hit == "L2"

    def test_without_expand(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg, with_expand=False)
        result = pipe.run("用户的偏好是什么", ctx)
        assert isinstance(result, SearchMemoryResult)

    def test_trace_contains_stages(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("tell me about coding", ctx)
        assert result.trace is not None
        stages = [s.stage for s in result.trace.stages]
        assert "planner" in stages
        assert "seed_retrieval" in stages

    def test_tenant_isolation(self, emb, cfg):
        vi = InMemoryVectorIndex(dimension=8)
        vi.upsert([
            IndexRecord(
                id="other", uri="ctx://other/users/x/memories/y", level=2,
                text="secret data",
                filters={"account_id": "other-acct", "owner_space": "user_x"},
                metadata={"context_type": "MEMORY"},
            ),
        ])
        pipe = _build_pipeline(vi, emb, cfg)
        ctx_mine = make_ctx(account_id="acct-1")
        result = pipe.run("secret data", ctx_mine)
        for block in result.hits:
            assert "other" not in block.uri

    def test_hierarchical_expansion_uses_internal_candidate_pool(self, ctx):
        class Planner:
            def plan(self, query, ctx, **kwargs):
                return [
                    TypedQuery(
                        text=query,
                        context_type="MEMORY",
                        categories=[],
                        top_k=5,
                        account_id=ctx.account_id,
                    )
                ]

        class Seed:
            def search(self, typed_query, ctx, *, mode):
                return SeedResult(
                    starting_points=[SeedHit(uri="ctx://acct-1/users/u1/memories/", score=0.9, level=0)],
                    query_vector=[0.1],
                )

        class Expand:
            def __init__(self) -> None:
                self.limit: int | None = None

            def expand(self, typed_query, seed_result, ctx, *, limit, mode, score_threshold):
                self.limit = limit
                return []

        cfg = RetrievalConfig(
            global_search_topk=30,
            default_top_k=5,
            candidate_pool_enabled=True,
            candidate_pool_multiplier=3,
            min_candidate_pool_size=15,
        )
        expand = Expand()
        pipe = RetrievalPipeline(
            Planner(),
            Seed(),
            expand,
            ResultRanker(cfg),
            cfg,
        )

        pipe.run("profile", ctx, top_k=5)

        assert expand.limit == 15

    def test_hierarchical_expansion_uses_final_top_k_when_candidate_pool_disabled(self, ctx):
        class Planner:
            def plan(self, query, ctx, **kwargs):
                return [
                    TypedQuery(
                        text=query,
                        context_type="MEMORY",
                        categories=[],
                        top_k=5,
                        account_id=ctx.account_id,
                    )
                ]

        class Seed:
            def search(self, typed_query, ctx, *, mode):
                return SeedResult(
                    starting_points=[SeedHit(uri="ctx://acct-1/users/u1/memories/", score=0.9, level=0)],
                    query_vector=[0.1],
                )

        class Expand:
            def __init__(self) -> None:
                self.limit: int | None = None

            def expand(self, typed_query, seed_result, ctx, *, limit, mode, score_threshold):
                self.limit = limit
                return []

        cfg = RetrievalConfig(global_search_topk=30, default_top_k=5)
        expand = Expand()
        pipe = RetrievalPipeline(
            Planner(),
            Seed(),
            expand,
            ResultRanker(cfg),
            cfg,
        )

        pipe.run("profile", ctx, top_k=5)

        assert expand.limit == 5

    def test_multiple_typed_queries_do_not_exceed_final_top_k(self, ctx):
        class Planner:
            def plan(self, query, ctx, **kwargs):
                return [
                    TypedQuery(text="memory", context_type="MEMORY", categories=[], top_k=2),
                    TypedQuery(text="skill", context_type="SKILL", categories=[], top_k=2),
                ]

        class Seed:
            def search(self, typed_query, ctx, *, mode):
                return SeedResult(
                    initial_candidates=[
                        SeedHit(uri=f"{typed_query.text}-1", score=0.9, level=2),
                        SeedHit(uri=f"{typed_query.text}-2", score=0.8, level=2),
                    ],
                    query_vector=[0.1],
                )

        class Assembly:
            def assemble(self, typed_query, leaf_hits, seed_result, **kwargs):
                if typed_query.text == "memory":
                    return [
                        RetrievedBlock(uri="memory-low-1", score=0.9, ranking_score=0.1),
                        RetrievedBlock(uri="memory-low-2", score=0.8, ranking_score=0.2),
                    ]
                return [
                    RetrievedBlock(uri="skill-high-1", score=0.7, ranking_score=0.9),
                    RetrievedBlock(uri="skill-high-2", score=0.6, ranking_score=0.8),
                ]

        cfg = RetrievalConfig(
            global_search_topk=10,
            default_top_k=2,
            global_result_rerank_enabled=True,
        )
        pipe = RetrievalPipeline(
            Planner(),
            Seed(),
            None,
            Assembly(),
            cfg,
        )

        result = pipe.run("combined query", ctx, top_k=2)

        assert len(result.hits) == 2
        assert [hit.uri for hit in result.hits] == ["skill-high-1", "skill-high-2"]

    def test_multiple_typed_queries_rank_globally_with_default_config(self, ctx):
        class Planner:
            def plan(self, query, ctx, **kwargs):
                return [
                    TypedQuery(text="memory", context_type="MEMORY", categories=[], top_k=2),
                    TypedQuery(text="skill", context_type="SKILL", categories=[], top_k=2),
                ]

        class Seed:
            def search(self, typed_query, ctx, *, mode):
                return SeedResult(
                    initial_candidates=[SeedHit(uri=f"{typed_query.text}-1", score=0.9, level=2)],
                    query_vector=[0.1],
                )

        class Assembly:
            def assemble(self, typed_query, leaf_hits, seed_result, **kwargs):
                if typed_query.text == "memory":
                    return [
                        RetrievedBlock(uri="memory-low-1", score=0.9, ranking_score=0.1),
                        RetrievedBlock(uri="memory-low-2", score=0.8, ranking_score=0.2),
                    ]
                return [
                    RetrievedBlock(uri="skill-high-1", score=0.7, ranking_score=0.9),
                    RetrievedBlock(uri="skill-high-2", score=0.6, ranking_score=0.8),
                ]

        cfg = RetrievalConfig(global_search_topk=10, default_top_k=2)
        pipe = RetrievalPipeline(
            Planner(),
            Seed(),
            None,
            Assembly(),
            cfg,
        )

        result = pipe.run("combined query", ctx, top_k=2)

        assert [hit.uri for hit in result.hits] == [
            "skill-high-1",
            "skill-high-2",
        ]

    def test_combined_queries_keep_best_ranked_memory_units(self, ctx):
        uri = "ctx://acct-1/agents/a1/memories/tools/deploy"
        groups = {
            "first": [
                RetrievedBlock(uri=uri + "/content.md", score=0.6, ranking_score=0.7,
                               metadata={"unit_key": "first"}),
                RetrievedBlock(uri=uri, score=0.7, ranking_score=0.8,
                               metadata={"unit_key": "second"}),
            ],
            "second": [
                RetrievedBlock(uri=uri, score=0.5, ranking_score=0.95,
                               metadata={"unit_key": "first"}),
                RetrievedBlock(uri="ctx://acct-1/users/u1/memories/preferences/cli",
                               score=0.9, ranking_score=0.9),
            ],
        }
        queries = [TypedQuery(text, "MEMORY", [], top_k=3) for text in groups]
        assembly = SimpleNamespace(assemble=lambda tq, *args, **kwargs: groups[tq.text])
        pipe = RetrievalPipeline(None, None, None, assembly)

        hits = pipe._run_assembly(queries, [], None, RetrievalTrace(), ctx)

        assert [(hit.ranking_score, hit.metadata.get("unit_key")) for hit in hits] == [
            (0.95, "first"), (0.9, None), (0.8, "second"),
        ]
