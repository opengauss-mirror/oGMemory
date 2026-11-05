"""Unit tests for retrieval.seed_retriever.SeedRetriever."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.lexical import LexicalHit
from core.models import IndexRecord, RequestContext, RetrievalConfig, SeedHit, TypedQuery
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.seed_retriever import SeedRetriever


def _make_ctx():
    return RequestContext(
        account_id="acct-1", user_id="u1", agent_id="a1",
        session_id="s1", trace_id="t1",
    )


def _make_cfg():
    return RetrievalConfig(
        global_search_topk=20, default_top_k=5,
        max_convergence_rounds=2, score_propagation_alpha=0.5, hotness_alpha=0.0,
    )


def _make_index() -> InMemoryVectorIndex:
    idx = InMemoryVectorIndex(dimension=8)
    records = [
        IndexRecord(
            id="r0", uri="ctx://acct-1/users/u1/memories", level=0,
            text="user memory root",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r1", uri="ctx://acct-1/users/u1/memories/profile", level=1,
            text="user profile overview",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "parent_uri": "ctx://acct-1/users/u1/memories",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r2", uri="ctx://acct-1/users/u1/memories/profile/detail", level=2,
            text="user profile full content about coding preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "parent_uri": "ctx://acct-1/users/u1/memories/profile",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r3", uri="ctx://acct-1/users/u1/memories/entities/openai", level=2,
            text="OpenAI is an AI company",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "entities",
                       "parent_uri": "ctx://acct-1/users/u1/memories/entities",
                       "has_overview": True, "has_content": True},
        ),
    ]
    idx.upsert(records)
    return idx


@pytest.fixture()
def cfg():
    return _make_cfg()


@pytest.fixture()
def ctx():
    return _make_ctx()


@pytest.fixture()
def embedder():
    return MockEmbedder(dimension=8)


@pytest.fixture()
def index_with_data():
    return _make_index()


@pytest.fixture()
def retriever(index_with_data, embedder, cfg):
    return SeedRetriever(index_with_data, embedder, cfg)


@pytest.fixture()
def tq():
    return TypedQuery(
        text="user profile",
        context_type="MEMORY",
        categories=[],
        top_k=10,
        account_id="acct-1",
        owner_space="user_u1",
    )


class _VecIdx:
    """Minimal vector index stub for fail-fast config tests."""

    def search_by_vector(self, query_vector, filters, top_k):
        return []


class TestSeedRetriever:
    def test_returns_seed_result(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        total = len(result.starting_points) + len(result.initial_candidates)
        assert total > 0

    def test_query_vector_populated(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        assert len(result.query_vector) > 0

    def test_split_levels(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        for sp in result.starting_points:
            assert sp.level < 2 or sp.score == 0.0
        for ic in result.initial_candidates:
            assert ic.level == 2

    def test_empty_collection(self, embedder, cfg, ctx):
        empty_idx = InMemoryVectorIndex(dimension=8)
        r = SeedRetriever(empty_idx, embedder, cfg)
        tq = TypedQuery(text="anything", context_type="MEMORY", categories=[], top_k=5, account_id="acct-1")
        result = r.search(tq, ctx)
        assert len(result.initial_candidates) == 0

    def test_owner_space_list_filters_visible_spaces(self, retriever, ctx):
        tq = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=10,
            account_id="acct-1",
            owner_space=["user_u1"],
        )

        result = retriever.search(tq, ctx)

        assert result.initial_candidates
        assert all(hit.owner_space == "user_u1" for hit in result.initial_candidates)

    def test_owner_space_defaults_to_visible_spaces(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="visible",
                    uri="ctx://acct-1/users/u1/memories/visible",
                    level=2,
                    text="shared visible support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u1"},
                    metadata={"context_type": "MEMORY", "category": "event"},
                ),
                IndexRecord(
                    id="hidden",
                    uri="ctx://acct-1/users/u2/memories/hidden",
                    level=2,
                    text="shared visible support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u2"},
                    metadata={"context_type": "MEMORY", "category": "event"},
                ),
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )
        query = TypedQuery(
            text="support group",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space=None,
        )

        result = retriever.search(query, ctx)

        assert {hit.owner_space for hit in result.initial_candidates} == {"user:u1"}

    def test_explicit_owner_space_is_intersected_with_visible_spaces(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="visible",
                    uri="ctx://acct-1/users/u1/memories/visible",
                    level=2,
                    text="support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u1"},
                    metadata={"context_type": "MEMORY"},
                ),
                IndexRecord(
                    id="hidden",
                    uri="ctx://acct-1/users/u2/memories/hidden",
                    level=2,
                    text="support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u2"},
                    metadata={"context_type": "MEMORY"},
                ),
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )

        hidden_result = retriever.search(
            TypedQuery(
                text="support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space="user:u2",
            ),
            ctx,
        )
        mixed_result = retriever.search(
            TypedQuery(
                text="support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space=["user:u1", "user:u2"],
            ),
            ctx,
        )

        assert hidden_result.initial_candidates == []
        assert {hit.owner_space for hit in mixed_result.initial_candidates} == {"user:u1"}

    def test_owner_space_intersection_accepts_legacy_alias(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="legacy",
                    uri="ctx://acct-1/users/u1/memories/legacy",
                    level=2,
                    text="legacy support group",
                    filters={"account_id": "acct-1", "owner_space": "user_u1"},
                    metadata={"context_type": "MEMORY"},
                )
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )

        result = retriever.search(
            TypedQuery(
                text="legacy support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space="user_u1",
            ),
            ctx,
        )

        assert {hit.owner_space for hit in result.initial_candidates} == {"user_u1"}

    def test_resource_query_applies_visible_owner_space_filter(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.filters = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.filters = filters
                return []

        idx = RecordingIndex()
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )
        query = TypedQuery(
            text="resource reference",
            context_type="RESOURCE",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space=["user:u1", "agent:a1"],
        )

        retriever.search(query, ctx)

        assert idx.filters["owner_space"] == ["user:u1", "user_u1", "agent:a1", "agent_a1"]

    def test_root_uris_include_visible_shared_agents(self, retriever, ctx):
        shared_ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1", "agent:a1", "agent:a2"),
        )

        result = retriever.search(
            TypedQuery(
                text="user profile",
                context_type="MEMORY",
                categories=[],
                top_k=10,
                account_id="acct-1",
                owner_space=["user_u1"],
            ),
            shared_ctx,
        )

        assert "ctx://acct-1/agents/a1/memories/" in result.root_uris
        assert "ctx://acct-1/agents/a2/memories/" in result.root_uris

    def test_root_uris_include_legacy_visible_shared_agents(self, retriever):
        shared_ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user_u1", "agent_a2"),
        )

        result = retriever.search(
            TypedQuery(
                text="user profile",
                context_type="MEMORY",
                categories=[],
                top_k=10,
                account_id="acct-1",
                owner_space=["user_u1"],
            ),
            shared_ctx,
        )

        assert "ctx://acct-1/agents/a2/memories/" in result.root_uris

    def test_vector_search_uses_internal_candidate_pool(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(
            global_search_topk=30,
            default_top_k=5,
            candidate_pool_enabled=True,
            candidate_pool_multiplier=3,
            min_candidate_pool_size=15,
        )
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 15

    def test_candidate_pool_does_not_cap_below_requested_top_k(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(
            global_search_topk=30,
            default_top_k=5,
            candidate_pool_enabled=True,
            candidate_pool_multiplier=3,
            min_candidate_pool_size=15,
        )
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="large result set",
            context_type="MEMORY",
            categories=[],
            top_k=40,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 40

    def test_vector_search_uses_global_topk_when_candidate_pool_disabled(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(global_search_topk=30, default_top_k=5)
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 30

    def test_weighted_rrf_rewards_hits_returned_by_both_paths(self):
        shared = SeedHit(uri="ctx://acct-1/shared", score=0.9, level=2)
        vector_only = SeedHit(uri="ctx://acct-1/vector", score=0.8, level=2)
        lexical_only = SeedHit(uri="ctx://acct-1/lexical", score=0.0, level=2)

        fused = SeedRetriever._weighted_rrf(
            [shared, vector_only],
            [shared, lexical_only],
            semantic_weight=1.0,
            lexical_weight=1.0,
            rrf_k=60,
        )

        assert fused[0].uri == shared.uri
        assert fused[0].metadata["_fusion_strategy"] == "weighted_rrf"
        assert fused[0].metadata["_retrieval_ranks"] == {
            "semantic": 1,
            "lexical": 1,
        }

        weighted = SeedRetriever._weighted_rrf(
            [vector_only],
            [lexical_only],
            semantic_weight=3.0,
            lexical_weight=1.0,
            rrf_k=60,
        )
        assert weighted[0].uri == vector_only.uri

    def test_weighted_rrf_omits_candidates_from_zero_weight_path(self):
        vector = SeedHit(uri="ctx://acct-1/vector", score=0.8, level=2)
        lexical = SeedHit(uri="ctx://acct-1/lexical", score=1.0, level=2)

        fused = SeedRetriever._weighted_rrf(
            [vector],
            [lexical],
            semantic_weight=1.0,
            lexical_weight=0.0,
            rrf_k=60,
        )

        assert [hit.uri for hit in fused] == [vector.uri]

    def test_hybrid_search_uses_configured_provider_limits_and_owner_filter(
        self, embedder, ctx
    ):
        class VectorIndex:
            def search_by_vector(self, query_vector, filters, top_k):
                self.filters, self.top_k = filters, top_k
                return []

        class LexicalIndex:
            supported_filters = frozenset({"account_id", "owner_space"})

            def search(self, query, top_k, filters):
                self.filters, self.top_k = filters, top_k
                return []

        vector, lexical = VectorIndex(), LexicalIndex()
        cfg = RetrievalConfig(hybrid_enabled=True, vector_top_k=7, lexical_top_k=3)
        query = TypedQuery(
            text="hybrid",
            context_type="MEMORY",
            categories=[],
            top_k=2,
            account_id="acct-1",
            owner_space="user_u1",
        )

        backfill = SimpleNamespace(ensure_ready=lambda _ctx, _scopes: True)
        SeedRetriever(vector, embedder, cfg, lexical, backfill).search(query, ctx)

        assert vector.top_k == 7
        assert lexical.top_k == 3
        assert lexical.filters["account_id"] == "acct-1"
        assert "user_u1" in lexical.filters["owner_space"]

    def test_hybrid_without_backfill_coordinator_fails_fast(self, embedder, ctx):
        """P1-2 regression: missing lexical backfill coordinator is a
        MISCONFIGURATION — it must raise, not silently degrade to
        vector-only results."""

        class LexicalIndex:
            supported_filters = frozenset({"account_id", "owner_space"})

            def search(self, *a, **k):
                return []

        cfg = RetrievalConfig(hybrid_enabled=True, lexical_weight=0.5)
        query = TypedQuery(
            text="hybrid", context_type="MEMORY", categories=[], top_k=2,
            account_id="acct-1", owner_space="user_u1",
        )
        r = SeedRetriever(_VecIdx(), embedder, cfg, LexicalIndex())
        with pytest.raises(RuntimeError, match="lexical backfill coordinator"):
            r.search(query, ctx)

    def test_hybrid_backfill_not_ready_fails_fast(self, embedder, ctx):
        """P1-2 regression: backfill refusing ready-state is a
        misconfiguration signal — raise, don't degrade."""

        class LexicalIndex:
            supported_filters = frozenset({"account_id", "owner_space"})

            def search(self, *a, **k):
                return []

        cfg = RetrievalConfig(hybrid_enabled=True, lexical_weight=0.5)
        query = TypedQuery(
            text="hybrid", context_type="MEMORY", categories=[], top_k=2,
            account_id="acct-1", owner_space="user_u1",
        )
        backfill = SimpleNamespace(ensure_ready=lambda _ctx, _scopes: False)
        r = SeedRetriever(_VecIdx(), embedder, cfg, LexicalIndex(), backfill)
        with pytest.raises(RuntimeError, match="backfill is not ready"):
            r.search(query, ctx)

    def test_hybrid_provider_missing_security_filters_fails_fast(self, embedder, ctx):
        """P1-2 regression: a lexical provider without account/owner-space
        security filters is a misconfiguration — raise, don't degrade."""

        class UnsafeLexicalIndex:
            supported_filters = frozenset({"account_id"})

            def search(self, *a, **k):
                return []

        cfg = RetrievalConfig(hybrid_enabled=True, lexical_weight=0.5)
        query = TypedQuery(
            text="hybrid", context_type="MEMORY", categories=[], top_k=2,
            account_id="acct-1", owner_space="user_u1",
        )
        backfill = SimpleNamespace(ensure_ready=lambda _ctx, _scopes: True)
        r = SeedRetriever(_VecIdx(), embedder, cfg, UnsafeLexicalIndex(), backfill)
        with pytest.raises(RuntimeError, match="security filters"):
            r.search(query, ctx)

    def test_zero_lexical_weight_skips_backfill_and_bm25(self, embedder, ctx):
        class VectorIndex:
            def search_by_vector(self, query_vector, filters, top_k):
                return [
                    SeedHit(
                        uri="ctx://acct-1/users/u1/memories/vector/content.md",
                        score=0.9,
                        level=2,
                    )
                ]

        class LexicalIndex:
            def search(self, **_kwargs):
                raise AssertionError("zero-weight lexical search must be skipped")

        backfill = SimpleNamespace(
            ensure_ready=lambda *_args: (_ for _ in ()).throw(
                AssertionError("zero-weight lexical backfill must be skipped")
            )
        )
        cfg = RetrievalConfig(
            hybrid_enabled=True,
            semantic_weight=1.0,
            lexical_weight=0.0,
        )
        query = TypedQuery(
            text="semantic only",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        result = SeedRetriever(
            VectorIndex(), embedder, cfg, LexicalIndex(), backfill
        ).search(query, ctx)

        assert [hit.uri for hit in result.initial_candidates] == [
            "ctx://acct-1/users/u1/memories/vector/content.md"
        ]

    def test_zero_semantic_weight_skips_embedding_and_vector_search(self, ctx):
        class Embedder:
            def embed_texts(self, _texts):
                raise AssertionError("zero-weight embedding must be skipped")

        class VectorIndex:
            def search_by_vector(self, **_kwargs):
                raise AssertionError("zero-weight vector search must be skipped")

        class LexicalIndex:
            supported_filters = frozenset({"account_id", "owner_space", "level"})

            def search(self, query, top_k, filters):
                assert filters["level"] == 2
                return [
                    LexicalHit(
                        id="lexical-leaf",
                        score=5.0,
                        metadata={
                            "uri": "ctx://acct-1/users/u1/memories/lexical/content.md",
                            "level": 2,
                            "account_id": "acct-1",
                            "owner_space": "user_u1",
                        },
                    )
                ]

        cfg = RetrievalConfig(
            hybrid_enabled=True,
            semantic_weight=0.0,
            lexical_weight=1.0,
        )
        query = TypedQuery(
            text="lexical only",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )
        backfill = SimpleNamespace(ensure_ready=lambda _ctx, _scopes: True)

        result = SeedRetriever(
            VectorIndex(), Embedder(), cfg, LexicalIndex(), backfill
        ).search(query, ctx)

        assert result.query_vector == []
        assert result.root_uris == []
        assert result.starting_points == []
        assert [hit.uri for hit in result.initial_candidates] == [
            "ctx://acct-1/users/u1/memories/lexical/content.md"
        ]

    def test_hybrid_lexical_results_are_revalidated_against_tenant_scope(
        self, index_with_data, embedder, ctx
    ):
        class LexicalIndex:
            supported_filters = frozenset({"account_id", "owner_space"})

            def search(self, query, top_k, filters):
                return [
                    LexicalHit(
                        id="wrong-owner",
                        score=8.0,
                        metadata={
                            "uri": "ctx://acct-1/users/u2/memories/leak",
                            "level": 2,
                            "account_id": "acct-1",
                            "owner_space": "user_u2",
                        },
                    ),
                    LexicalHit(
                        id="bad",
                        score=10.0,
                        metadata={
                            "uri": "ctx://other/users/u1/memories/leak",
                            "level": 2,
                            "account_id": "other",
                            "owner_space": "user_u1",
                        },
                    ),
                    LexicalHit(
                        id="good",
                        score=5.0,
                        metadata={
                            "uri": "ctx://acct-1/users/u1/memories/good",
                            "level": 2,
                            "account_id": "acct-1",
                            "owner_space": "user_u1",
                        },
                    ),
                ]

        hybrid_cfg = RetrievalConfig(hybrid_enabled=True)
        backfill = SimpleNamespace(ensure_ready=lambda _ctx, _scopes: True)
        retriever = SeedRetriever(
            index_with_data, embedder, hybrid_cfg, LexicalIndex(), backfill
        )
        query = TypedQuery(
            text="good",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        result = retriever.search(query, ctx)

        uris = {hit.uri for hit in result.initial_candidates}
        assert "ctx://other/users/u1/memories/leak" not in uris
        assert "ctx://acct-1/users/u2/memories/leak" not in uris
        assert "ctx://acct-1/users/u1/memories/good" in uris


class TestDegradeOnEmbedFailure:
    """WS 3.4: embed failures degrade to BM25-only retrieval, never raise."""

    def _bm25_index(self):
        """A minimal BM25 index stub matching the .search() interface."""

        def _hit(uri, level, parent, score):
            return SimpleNamespace(
                score=score,
                metadata={
                    "uri": uri,
                    "level": level,
                    "parent_uri": parent,
                    "context_type": "MEMORY",
                    "category": "profile",
                    "owner_space": "user_u1",
                    "account_id": "acct-1",
                },
            )

        class _Idx:
            doc_count = 3
            supported_filters = {"account_id", "owner_space"}

            def search(self, query, top_k, filters):
                return [
                    _hit(
                        "ctx://acct-1/users/u1/memories/profile", 1,
                        "ctx://acct-1/users/u1/memories", 1.5,
                    ),
                    _hit(
                        "ctx://acct-1/users/u1/memories/profile/detail", 2,
                        "ctx://acct-1/users/u1/memories/profile", 1.2,
                    ),
                ]

        return _Idx()

    def test_embed_failure_degrades_to_bm25_only(self, cfg, ctx, tq):
        from unittest.mock import Mock

        class ExplodingEmbedder:
            dimension = 8

            def embed_texts(self, texts, **kwargs):
                raise RuntimeError("embedder down")

        r = SeedRetriever(
            Mock(), ExplodingEmbedder(), cfg, bm25_index=self._bm25_index()
        )
        result = r.search(tq, ctx)

        # No raise; BM25 level-2 hits survive as candidates.
        assert result.query_vector == []
        assert result.initial_candidates
        uris = {h.uri for h in result.initial_candidates}
        assert "ctx://acct-1/users/u1/memories/profile/detail" in uris
        # Review item 7: with no query vector there must be NO hierarchy
        # expansion — starting points would drive search_children into
        # cosine SQL against []::vector, which pgvector cannot execute.
        # Lexical L2 hits only, no semantic traversal.
        assert result.starting_points == []
        assert result.root_uris == []

    def test_embed_and_bm25_failure_returns_empty_not_raise(self, cfg, ctx, tq):
        from unittest.mock import Mock

        class ExplodingEmbedder:
            dimension = 8

            def embed_texts(self, texts, **kwargs):
                raise RuntimeError("embedder down")

        class ExplodingBM25:
            doc_count = 3

            def search(self, *args, **kwargs):
                raise RuntimeError("bm25 down")

        r = SeedRetriever(
            Mock(), ExplodingEmbedder(), cfg, bm25_index=ExplodingBM25()
        )
        result = r.search(tq, ctx)
        assert result.initial_candidates == []
        # Total outage: empty result, no raise — and no starting points
        # (review item 7: score-0 root starting points are unusable
        # without a query vector; they would only push []::vector into
        # the SQL hierarchy search).
        assert result.starting_points == []
        assert result.root_uris == []

    def test_pipeline_run_seed_falls_back_to_root_uris(self, cfg, ctx, tq):
        """RetrievalPipeline._run_seed must not re-raise when search fails."""
        from unittest.mock import Mock

        from retrieval.pipeline import RetrievalPipeline

        class ExplodingRetriever:
            def search(self, tq, ctx, mode="quick"):
                raise RuntimeError("total outage")

        pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
        trace = Mock()

        result = pipeline._run_seed(tq, ctx, trace, mode="quick")

        assert result.query_vector == []
        assert result.initial_candidates == []
        assert result.root_uris  # rooted at the typed query's roots
        for hit in result.starting_points:
            assert hit.score == 0.0
        assert all(
            hit.uri in result.root_uris for hit in result.starting_points
        )
