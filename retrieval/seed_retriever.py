"""Stage 2 — Seed retriever.

Performs a global L0/L1/L2 vector search, or L2-only without expansion, optionally
augmented with BM25 keyword search via Vector-Anchored Fusion.
Splits results into starting_points (L0/L1) and initial_candidates (L2).
"""

from __future__ import annotations

import logging
from typing import Any

from core.enums import ContextType
from core.interfaces import Embedder, VectorIndex
from core.models import (
    RequestContext,
    RetrievalConfig,
    RetrieverMode,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.candidate_pool import candidate_top_k
from retrieval.identity import memory_identity
from retrieval.owner_scope import allowed_owner_space_filter, visible_agent_ids

logger = logging.getLogger(__name__)


class SeedRetriever:
    """Global search with optional BM25 hybrid fusion and direct L2 mode."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        bm25_index: Any | None = None,
        lexical_backfill: Any | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.bm25_index = bm25_index
        self.lexical_backfill = lexical_backfill

    def search(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
        *,
        mode: str = RetrieverMode.QUICK,
        query_vector: list[float] | None = None,
    ) -> SeedResult:
        semantic_enabled = (
            not self.config.hybrid_enabled or self.config.semantic_weight > 0
        )
        if semantic_enabled:
            if query_vector is None:
                query_vector = self.embedder.embed_texts([typed_query.text])[0]
            root_uris = (
                self._get_root_uris(typed_query.context_type, ctx)
                if self.config.expand_hierarchy else []
            )
            global_results = self._global_vector_search(
                query_vector, typed_query, ctx
            )
        else:
            query_vector = []
            root_uris = []
            global_results = []

        # Database-native hybrid retrieval compares ranks rather than provider-
        # specific score scales. The legacy in-memory path remains disabled
        # unless an index is explicitly supplied, preserving default behavior.
        lexical_enabled = (
            not self.config.hybrid_enabled or self.config.lexical_weight > 0
        )
        if self.bm25_index and lexical_enabled and (
            self.config.hybrid_enabled
            or getattr(self.bm25_index, "doc_count", 0) > 0
        ):
            if self.config.hybrid_enabled:
                owner_spaces = self._hybrid_owner_spaces(typed_query, ctx)
                if not owner_spaces:
                    bm25_results = []
                else:
                    if self.lexical_backfill is None:
                        raise RuntimeError(
                            "hybrid retrieval requires a lexical backfill coordinator"
                        )
                    if not self.lexical_backfill.ensure_ready(ctx, owner_spaces):
                        raise RuntimeError("required lexical backfill is not ready")
                    bm25_results = self._bm25_search(typed_query, ctx)
            else:
                bm25_results = self._bm25_search(typed_query, ctx)
            if bm25_results:
                if self.config.hybrid_enabled:
                    global_results = self._weighted_rrf(
                        global_results,
                        bm25_results,
                        semantic_weight=self.config.semantic_weight,
                        lexical_weight=self.config.lexical_weight,
                        rrf_k=self.config.rrf_k,
                    )
                else:
                    global_results = self._fuse_results(
                        global_results, bm25_results
                    )

        if not self.config.expand_hierarchy:
            return SeedResult(
                initial_candidates=[hit for hit in global_results if hit.level == 2],
                query_vector=query_vector,
            )

        starting_points, initial_candidates = self._merge_starting_points(
            root_uris, global_results,
        )

        return SeedResult(
            starting_points=starting_points,
            initial_candidates=initial_candidates,
            query_vector=query_vector,
            root_uris=root_uris,
        )

    @staticmethod
    def _get_root_uris(context_type: str | None, ctx: RequestContext) -> list[str]:
        # SECURITY: URI format per CLAUDE.md §1:
        # - ctx://{account}/users/{user}/memories/
        # - ctx://{account}/agents/{agent}/memories/
        # - ctx://{account}/agents/{agent}/skills/
        account = ctx.account_id
        agent_ids = visible_agent_ids(ctx)
        if not agent_ids and ctx.agent_id:
            agent_ids = [ctx.agent_id]
        if not context_type:
            roots = [
                f"ctx://{account}/users/{ctx.user_id}/memories/",
                f"ctx://{account}/resources/",
            ]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in agent_ids)
            roots.extend(f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in agent_ids)
            return roots
        ct = context_type.upper()
        if ct == ContextType.MEMORY.value:
            roots = [f"ctx://{account}/users/{ctx.user_id}/memories/"]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in agent_ids)
            return roots
        if ct == ContextType.RESOURCE.value:
            return [f"ctx://{account}/resources/"]
        if ct == ContextType.SKILL.value:
            return [f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in agent_ids]
        return []

    def _global_vector_search(
        self,
        query_vector: list[float],
        typed_query: TypedQuery,
        ctx: RequestContext,
    ) -> list[SeedHit]:
        filters = {
            "level": [0, 1, 2] if self.config.expand_hierarchy else 2,
            **self._search_filters(typed_query, ctx),
        }

        return self.vector_index.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=(
                self.config.vector_top_k
                if self.config.hybrid_enabled
                else candidate_top_k(
                    self.config,
                    typed_query.top_k,
                    fallback_top_k=self.config.global_search_topk,
                )
            ),
        )

    def _search_filters(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {"account_id": ctx.account_id}
        if typed_query.context_type:
            filters["context_type"] = typed_query.context_type
        if typed_query.resource_type:
            filters["resource_type"] = typed_query.resource_type
        owner_space = allowed_owner_space_filter(
            typed_query,
            ctx,
            enforce_visible_scope=(
                True
                if self.config.hybrid_enabled
                else getattr(self.config, "owner_scope_filter_enabled", False)
            ),
        )
        if owner_space:
            filters["owner_space"] = owner_space
        if typed_query.categories:
            filters["category"] = typed_query.categories
        if typed_query.memory_directories is not None:
            filters["memory_directory"] = typed_query.memory_directories
        return filters

    def _bm25_search(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
    ) -> list[SeedHit]:
        """BM25 keyword search with metadata filtering."""
        filters = self._search_filters(typed_query, ctx)
        lexical_only = (
            self.config.hybrid_enabled and self.config.semantic_weight <= 0
        )
        l2_only = lexical_only or not self.config.expand_hierarchy
        if l2_only:
            filters["level"] = 2
        elif str(typed_query.context_type).upper() == "MEMORY":
            filters["level"] = [0, 1, 2]

        supported_filters = frozenset(
            getattr(self.bm25_index, "supported_filters", filters)
        )
        if "memory_directory" in filters and "memory_directory" not in supported_filters:
            raise RuntimeError("typed memory retrieval requires a lexical provider with a memory_directory filter")
        if not self.config.expand_hierarchy and "level" not in supported_filters:
            raise RuntimeError("direct L2 retrieval requires a lexical provider with a level filter")
        if self.config.hybrid_enabled and not {
            "account_id", "owner_space"
        }.issubset(supported_filters):
            raise RuntimeError(
                "lexical provider must support account_id and owner_space security filters"
            )
        provider_filters = {
            key: value for key, value in filters.items() if key in supported_filters
        }
        bm25_hits = self.bm25_index.search(
            query=typed_query.text,
            top_k=(
                self.config.lexical_top_k
                if self.config.hybrid_enabled
                else candidate_top_k(
                    self.config,
                    typed_query.top_k,
                    fallback_top_k=self.config.global_search_topk,
                )
            ),
            filters=provider_filters,
        )
        hits: list[SeedHit] = []
        for h in bm25_hits:
            meta = h.metadata
            metadata = dict(meta or {})
            if int(metadata.get("level", 2)) not in (0, 1, 2) or (l2_only and int(metadata.get("level", 2)) != 2):
                continue
            if self.config.hybrid_enabled:
                if metadata.get("account_id") != ctx.account_id:
                    logger.warning("Discarded lexical result outside account scope")
                    continue
                allowed_spaces = filters.get("owner_space")
                if allowed_spaces:
                    allowed = (
                        {str(value) for value in allowed_spaces}
                        if isinstance(allowed_spaces, (list, tuple, set, frozenset))
                        else {str(allowed_spaces)}
                    )
                    if str(metadata.get("owner_space", "")) not in allowed:
                        logger.warning("Discarded lexical result outside owner scope")
                        continue
            metadata["_bm25_raw_score"] = h.score
            hits.append(SeedHit(
                uri=meta.get("uri", ""),
                score=0.0,  # score set by _fuse_results
                level=int(meta.get("level", 2)),
                parent_uri=meta.get("parent_uri"),
                context_type=meta.get("context_type", ""),
                category=meta.get("category", ""),
                owner_space=meta.get("owner_space", ""),
                abstract=meta.get("abstract", ""),
                has_overview=bool(meta.get("has_overview")),
                has_content=bool(meta.get("has_content")),
                active_count=int(meta.get("active_count", 0)),
                updated_at=meta.get("updated_at"),
                metadata=metadata,
            ))
        return hits

    @staticmethod
    def _hybrid_owner_spaces(
        typed_query: TypedQuery, ctx: RequestContext
    ) -> list[str]:
        value = allowed_owner_space_filter(
            typed_query, ctx, enforce_visible_scope=True
        )
        if not value or value == "__inaccessible__":
            return []
        return list(value) if isinstance(value, list) else [str(value)]

    @staticmethod
    def _fuse_results(
        vector_hits: list[SeedHit],
        bm25_hits: list[SeedHit],
        alpha: float = 0.7,
        saturation_k: float = 5.0,
    ) -> list[SeedHit]:
        """Vector-Anchored Fusion: combine vector and BM25 results.

        Formula:
            sat_bm25 = raw_bm25 / (raw_bm25 + saturation_k)
            final_score = alpha * vec_score + (1 - alpha) * sat_bm25

        Args:
            vector_hits: Results from vector search (score = cosine similarity)
            bm25_hits: Results from BM25 search (score = raw BM25 score)
            alpha: Weight for vector score (default 0.7 = 70% vector)
            saturation_k: BM25 saturation constant (default 5.0)
        """
        # Fuse matching nodes or units within the same layer.
        fused: dict[tuple[str, str, int], SeedHit] = {}
        vec_scores: dict[tuple[str, str, int], float] = {}
        bm25_scores: dict[tuple[str, str, int], float] = {}

        for hit in vector_hits:
            key = (*memory_identity(hit.uri, hit.metadata), hit.level)
            fused[key] = hit
            vec_scores[key] = hit.score

        for hit in bm25_hits:
            key = (*memory_identity(hit.uri, hit.metadata), hit.level)
            bm25_scores[key] = hit.metadata.get("_bm25_raw_score", 0.0)
            if key not in fused:
                fused[key] = hit

        if not fused:
            return []

        # Compute floor defaults (minimum score from each path)
        min_vec = min(vec_scores.values()) if vec_scores else 0.0
        min_bm25_sat = min(
            s / (s + saturation_k) for s in bm25_scores.values()
        ) if bm25_scores else 0.0

        # Compute fused scores
        results: list[SeedHit] = []
        for uri, hit in fused.items():
            vec_s = vec_scores.get(uri, min_vec)
            raw_bm25 = bm25_scores.get(uri, 0.0)
            sat_bm25 = raw_bm25 / (raw_bm25 + saturation_k) if raw_bm25 > 0 else min_bm25_sat
            fused_score = alpha * vec_s + (1 - alpha) * sat_bm25

            results.append(SeedHit(
                uri=hit.uri,
                score=fused_score,
                level=hit.level,
                parent_uri=hit.parent_uri,
                context_type=hit.context_type,
                category=hit.category,
                owner_space=hit.owner_space,
                abstract=hit.abstract,
                has_overview=hit.has_overview,
                has_content=hit.has_content,
                active_count=hit.active_count,
                updated_at=hit.updated_at,
                metadata=hit.metadata,
            ))

        results.sort(key=lambda h: h.score, reverse=True)
        return results

    @staticmethod
    def _weighted_rrf(
        vector_hits: list[SeedHit],
        lexical_hits: list[SeedHit],
        *,
        semantic_weight: float,
        lexical_weight: float,
        rrf_k: float,
    ) -> list[SeedHit]:
        """Fuse heterogeneous result rankings with weighted reciprocal rank."""
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        hits: dict[tuple[str, str, int], SeedHit] = {}
        scores: dict[tuple[str, str, int], float] = {}
        ranks: dict[tuple[str, str, int], dict[str, int]] = {}
        for path, path_hits, weight in (
            ("semantic", vector_hits, semantic_weight),
            ("lexical", lexical_hits, lexical_weight),
        ):
            if weight <= 0:
                continue
            seen: set[tuple[str, str, int]] = set()
            for rank, hit in enumerate(path_hits, start=1):
                key = (*memory_identity(hit.uri, hit.metadata), hit.level)
                if not hit.uri or key in seen:
                    continue
                seen.add(key)
                hits.setdefault(key, hit)
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
                ranks.setdefault(key, {})[path] = rank

        max_score = (semantic_weight + lexical_weight) / (rrf_k + 1.0)
        results: list[SeedHit] = []
        for key, raw_score in scores.items():
            hit = hits[key]
            metadata = dict(hit.metadata or {})
            metadata["_fusion_strategy"] = "weighted_rrf"
            metadata["_retrieval_ranks"] = ranks[key]
            results.append(SeedHit(
                uri=hit.uri,
                score=raw_score / max_score if max_score > 0 else 0.0,
                level=hit.level,
                parent_uri=hit.parent_uri,
                context_type=hit.context_type,
                category=hit.category,
                owner_space=hit.owner_space,
                abstract=hit.abstract,
                has_overview=hit.has_overview,
                has_content=hit.has_content,
                active_count=hit.active_count,
                updated_at=hit.updated_at,
                metadata=metadata,
            ))
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results

    @staticmethod
    def _merge_starting_points(
        root_uris: list[str],
        global_results: list[SeedHit],
    ) -> tuple[list[SeedHit], list[SeedHit]]:
        initial_candidates: list[SeedHit] = []
        points: list[SeedHit] = []
        seen: set[str] = set()

        for hit in global_results:
            if hit.level == 2:
                initial_candidates.append(hit)
            else:
                if hit.uri not in seen:
                    points.append(hit)
                    seen.add(hit.uri)

        for uri in root_uris:
            if uri not in seen:
                points.append(SeedHit(uri=uri, score=0.0, level=0))
                seen.add(uri)

        return points, initial_candidates
