"""Stage 3 — Hierarchical recursive searcher.

Priority-queue driven recursive expansion:
  - L0/L1 nodes are pushed back into the queue
  - L2 nodes are terminal hits
  - Score propagation: final = alpha * child + (1-alpha) * parent
  - Convergence detection stops after top-k unchanged for N rounds
"""

from __future__ import annotations

import heapq
import logging
from datetime import datetime
from typing import Any

from core.interfaces import VectorIndex
from core.models import (
    LeafHit,
    RequestContext,
    RetrievalConfig,
    RetrieverMode,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.hotness import hotness_score
from retrieval.identity import memory_identity
from retrieval.owner_scope import allowed_owner_space_filter

logger = logging.getLogger(__name__)


class HierarchicalSearcher:
    def __init__(
        self,
        vector_index: VectorIndex,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.cfg = config or RetrievalConfig()

    def expand(
        self,
        typed_query: TypedQuery,
        seed_result: SeedResult,
        ctx: RequestContext,
        *,
        limit: int = 5,
        mode: str = RetrieverMode.THINKING,
        score_threshold: float | None = None,
    ) -> list[LeafHit]:
        candidates = self._recursive_search(
            query_vector=seed_result.query_vector,
            starting_points=seed_result.starting_points,
            initial_candidates=seed_result.initial_candidates,
            typed_query=typed_query,
            ctx=ctx,
            limit=limit,
            threshold=score_threshold,
        )
        return self._convert_results(candidates, limit)

    def _recursive_search(
        self,
        query_vector: list[float],
        starting_points: list[SeedHit],
        initial_candidates: list[SeedHit],
        typed_query: TypedQuery,
        ctx: RequestContext,
        limit: int,
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        effective_threshold = threshold if threshold is not None else self.cfg.default_score_threshold
        alpha = self.cfg.score_propagation_alpha
        collected_by_uri: dict[tuple[str, str, int], dict[str, Any]] = {}
        dir_queue: list[tuple[float, str]] = []
        visited: set[str] = set()
        prev_topk_uris: set[tuple[str, str, int]] = set()
        convergence_rounds = 0

        for hit in initial_candidates:
            if hit.level == 2 and hit.uri:
                identity = (*memory_identity(hit.uri, hit.metadata), hit.level)
                previous = collected_by_uri.get(identity)
                if previous is None or hit.score > previous["_final_score"]:
                    collected_by_uri[identity] = self._hit_to_dict(hit, hit.score)

        for hit in starting_points:
            heapq.heappush(dir_queue, (-hit.score, hit.uri))

        base_filters: dict[str, Any] = {"account_id": ctx.account_id}
        if typed_query.context_type:
            base_filters["context_type"] = typed_query.context_type
        if typed_query.resource_type:
            base_filters["resource_type"] = typed_query.resource_type
        if typed_query.memory_directories is not None:
            base_filters["memory_directory"] = typed_query.memory_directories
        owner_space = allowed_owner_space_filter(
            typed_query,
            ctx,
            enforce_visible_scope=getattr(self.cfg, "owner_scope_filter_enabled", False),
        )
        if owner_space:
            base_filters["owner_space"] = owner_space

        while dir_queue:
            neg_score, current_uri = heapq.heappop(dir_queue)
            current_score = -neg_score
            if current_uri in visited:
                continue
            visited.add(current_uri)

            pre_filter_limit = max(limit * 2, 20)
            children = self.vector_index.search_children(
                parent_uri=current_uri,
                query_vector=query_vector,
                filters=base_filters,
                top_k=pre_filter_limit,
            )
            if not children:
                continue

            for child in children:
                final_score = (
                    alpha * child.score + (1 - alpha) * current_score
                    if current_score
                    else child.score
                )

                if final_score <= effective_threshold:
                    continue

                uri = child.uri
                identity = (*memory_identity(uri, child.metadata), child.level)
                prev = collected_by_uri.get(identity)
                if prev is None or final_score > prev.get("_final_score", 0):
                    collected_by_uri[identity] = self._hit_to_dict(child, final_score)

                if uri not in visited and child.level != 2:
                    heapq.heappush(dir_queue, (-final_score, uri))

            current_topk = sorted(
                collected_by_uri.values(),
                key=lambda x: x.get("_final_score", 0),
                reverse=True,
            )[:limit]
            current_topk_uris = {(*memory_identity(c["uri"], c["metadata"]), c["level"]) for c in current_topk}

            if current_topk_uris == prev_topk_uris and len(current_topk_uris) >= limit:
                convergence_rounds += 1
                if convergence_rounds >= self.cfg.max_convergence_rounds:
                    break
            else:
                convergence_rounds = 0
                prev_topk_uris = current_topk_uris

        return sorted(
            collected_by_uri.values(),
            key=lambda x: x.get("_final_score", 0),
            reverse=True,
        )[:limit]

    def _convert_results(self, candidates: list[dict[str, Any]], limit: int) -> list[LeafHit]:
        results: list[LeafHit] = []
        hotness_alpha = self.cfg.hotness_alpha

        for c in candidates:
            semantic_score: float = c.get("_final_score", 0.0)

            updated_at_raw = c.get("updated_at")
            updated_at_val: datetime | None = None
            if isinstance(updated_at_raw, str):
                try:
                    updated_at_val = datetime.fromisoformat(updated_at_raw)
                except (ValueError, TypeError):
                    pass
            elif isinstance(updated_at_raw, datetime):
                updated_at_val = updated_at_raw

            h_score = hotness_score(
                active_count=c.get("active_count", 0),
                updated_at=updated_at_val,
                half_life_days=self.cfg.hotness_half_life_days,
            )

            blended = (1 - hotness_alpha) * semantic_score + hotness_alpha * h_score

            results.append(
                LeafHit(
                    uri=c.get("uri", ""),
                    score=blended,
                    level=c.get("level", 2),
                    parent_uri=c.get("parent_uri"),
                    category=c.get("category", ""),
                    owner_space=c.get("owner_space", ""),
                    abstract=c.get("abstract", ""),
                    has_overview=c.get("has_overview", False),
                    has_content=c.get("has_content", False),
                    active_count=c.get("active_count", 0),
                    updated_at=c.get("updated_at"),
                    metadata=dict(c.get("metadata") or {}),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _hit_to_dict(hit: SeedHit, final_score: float) -> dict[str, Any]:
        return {
            "uri": hit.uri,
            "level": hit.level,
            "parent_uri": hit.parent_uri,
            "category": hit.category,
            "owner_space": hit.owner_space,
            "abstract": hit.abstract,
            "has_overview": hit.has_overview,
            "has_content": hit.has_content,
            "active_count": hit.active_count,
            "updated_at": hit.updated_at,
            "metadata": dict(hit.metadata or {}),
            "_final_score": final_score,
        }
