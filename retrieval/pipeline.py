"""Retrieval pipeline orchestrator.

Runs 4 stages sequentially — errors are propagated, not silently degraded:
  Stage 1  QueryPlanner        — classify query into TypedQuery[]
  Stage 2  SeedRetriever       — global L0/L1/L2 mixed vector search
  Stage 3  HierarchicalSearcher — auto-expand L0/L1 to L2 leaves
  Stage 4  ResultRanker     — keep only L2, sort, top-k -> RetrievedBlock[]

Returns a pure structured SearchMemoryResult.
"""

from __future__ import annotations

import logging
import time

from core.errors import RetrievalError
from core.models import (
    LeafHit,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.candidate_pool import candidate_top_k
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker, _vec_to_leaf
from retrieval.seed_retriever import SeedRetriever
from retrieval.trace import RetrievalTrace, TraceTimer

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        planner: QueryPlanner,
        seed_retriever: SeedRetriever,
        hierarchical_searcher: HierarchicalSearcher | None,
        assembly: ResultRanker,
        config: RetrievalConfig | None = None,
        context_reader=None,
        access_tracker=None,
    ) -> None:
        self.planner = planner
        self.seed_retriever = seed_retriever
        self.hierarchical_searcher = hierarchical_searcher
        self.assembly = assembly
        self.cfg = config or RetrievalConfig()
        self._context_reader = context_reader
        self._access_tracker = access_tracker

    def run(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        session_archive: dict | None = None,
        hints: dict | None = None,
        score_threshold: float | None = None,
        mode: str = RetrieverMode.QUICK,
        fill_content_for_top_k: int = 0,
        context_type: str | None = None,
        resource_type: str | None = None,
    ) -> SearchMemoryResult:
        t0 = time.monotonic()
        trace = RetrievalTrace()

        # Stage 1
        typed_queries = self._run_planner(
            query, ctx, trace,
            top_k=top_k, categories=categories,
            hints=hints, session_archive=session_archive,
            context_type=context_type,
            resource_type=resource_type,
        )

        # Stage 2
        all_seed_results: list[SeedResult] = []
        level_counts: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0}
        for tq in typed_queries:
            sr = self._run_seed(tq, ctx, trace, mode=mode)
            all_seed_results.append(sr)
            for h in sr.starting_points:
                level_counts[f"L{h.level}"] = level_counts.get(f"L{h.level}", 0) + 1
            for h in sr.initial_candidates:
                level_counts[f"L{h.level}"] = level_counts.get(f"L{h.level}", 0) + 1
        trace.level_histogram = level_counts

        # Stage 3
        all_leaf_hits: list[LeafHit] = []
        for tq, sr in zip(typed_queries, all_seed_results):
            leaves = self._run_expand(
                tq, sr, ctx, trace,
                limit=candidate_top_k(self.cfg, tq.top_k, fallback_top_k=tq.top_k), mode=mode,
                score_threshold=score_threshold,
            )
            all_leaf_hits.extend(leaves)

        # Stage 4
        merged_seed = self._merge_seeds(all_seed_results)
        blocks = self._run_assembly(typed_queries, all_leaf_hits, merged_seed, trace, ctx, fill_content_for_top_k=fill_content_for_top_k)

        trace.total_ms = (time.monotonic() - t0) * 1000

        # Track access hits
        if self._access_tracker is not None:
            hit_uris = [b.uri for b in blocks if hasattr(b, 'uri')]
            self._access_tracker.record_hits(hit_uris)

        return SearchMemoryResult(
            request_id=trace.request_id,
            query=query,
            typed_queries=typed_queries,
            hits=blocks,
            trace=trace,
        )

    # -- Stage runners with degradation ------------------------------------

    def _run_planner(
        self,
        query,
        ctx,
        trace,
        *,
        top_k,
        categories,
        hints,
        session_archive,
        context_type,
        resource_type,
    ) -> list[TypedQuery]:
        with TraceTimer("planner") as st:
            st.input_count = 1
            try:
                queries = self.planner.plan(
                    query, ctx,
                    session_archive=session_archive,
                    hints=hints,
                    categories=categories,
                    top_k=top_k,
                    context_type=context_type,
                    resource_type=resource_type,
                )
            except RetrievalError:
                raise
            except Exception as exc:
                logger.error("[planner] failed: %s", exc, exc_info=True)
                raise
            st.output_count = len(queries)
        trace.add_stage(st)
        return queries

    def _run_seed(self, tq, ctx, trace, *, mode) -> SeedResult:
        with TraceTimer("seed_retrieval") as st:
            st.input_count = 1
            try:
                result = self.seed_retriever.search(tq, ctx, mode=mode)
            except Exception as exc:
                # Do not re-raise: during embedder/vector-index outages
                # every turn would otherwise fail. Fall back to a seed
                # rooted at the typed query's root URIs (score 0.0) so
                # hierarchical expansion / fallback L2 still has
                # something to walk and the search run continues.
                logger.error(
                    "[seed_retrieval] failed: %s — degrading to "
                    "root-URI-only seed", exc, exc_info=True,
                )
                root_uris = SeedRetriever._get_root_uris(
                    getattr(tq, "context_type", None), ctx,
                )
                result = SeedResult(
                    starting_points=[
                        SeedHit(uri=uri, score=0.0)
                        for uri in root_uris
                    ],
                    initial_candidates=[],
                    query_vector=[],
                    root_uris=root_uris,
                )
            st.output_count = len(result.starting_points) + len(result.initial_candidates)
        trace.add_stage(st)
        return result

    def _run_expand(self, tq, sr, ctx, trace, *, limit, mode, score_threshold) -> list[LeafHit]:
        if not sr.starting_points and not sr.initial_candidates:
            return []

        has_dirs = any(v.level < 2 for v in sr.starting_points)

        if has_dirs and self.hierarchical_searcher:
            with TraceTimer("hierarchical") as st:
                st.input_count = len(sr.starting_points) + len(sr.initial_candidates)
                try:
                    leaves = self.hierarchical_searcher.expand(
                        tq, sr, ctx,
                        limit=limit, mode=mode, score_threshold=score_threshold,
                    )
                except Exception as exc:
                    logger.error("[hierarchical] expand failed: %s", exc, exc_info=True)
                    raise
                st.output_count = len(leaves)
            trace.add_stage(st)
            return leaves

        return self._fallback_l2_from_seed(sr)

    def _run_assembly(self, typed_queries, leaf_hits, merged_seed, trace, ctx=None, fill_content_for_top_k=0) -> list[RetrievedBlock]:
        with TraceTimer("assembly") as st:
            st.input_count = len(leaf_hits)
            all_blocks = []
            seen_uris = set()

            # Run assembly for each typed query (e.g., memory/skill/resource)
            # and deduplicate results by URI
            for tq in (typed_queries or [TypedQuery(text="", context_type="")]):
                try:
                    blocks = self.assembly.assemble(
                        tq, leaf_hits, merged_seed, ctx=ctx,
                        fill_content_for_top_k=fill_content_for_top_k,
                        context_reader=self._context_reader,
                    )
                except Exception as exc:
                    logger.error("[assembly] assemble failed: %s", exc, exc_info=True)
                    raise

                for b in blocks:
                    uri = getattr(b, "uri", None) or getattr(b, "source_uri", None)
                    if uri and uri in seen_uris:
                        continue
                    if uri:
                        seen_uris.add(uri)
                    all_blocks.append(b)

            final_limit = max((getattr(tq, "top_k", 0) for tq in typed_queries), default=0)
            if getattr(self.cfg, "global_result_rerank_enabled", False) and final_limit > 0:
                all_blocks.sort(
                    key=lambda block: (
                        getattr(block, "ranking_score", None)
                        if getattr(block, "ranking_score", None) is not None
                        else getattr(block, "score", 0.0)
                    ),
                    reverse=True,
                )
                all_blocks = all_blocks[:final_limit]

            st.output_count = len(all_blocks)
        trace.add_stage(st)
        return all_blocks

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _fallback_l2_from_seed(sr: SeedResult) -> list[LeafHit]:
        """Extract L2 hits from seed result using the shared _vec_to_leaf utility."""
        out: list[LeafHit] = []
        for v in sr.initial_candidates:
            if v.level == 2:
                out.append(_vec_to_leaf(v))
        return out

    @staticmethod
    def _merge_seeds(seed_results: list[SeedResult]) -> SeedResult | None:
        if not seed_results:
            return None
        if len(seed_results) == 1:
            return seed_results[0]
        merged = SeedResult()
        seen_sp: set[str] = set()
        seen_ic: set[str] = set()
        for sr in seed_results:
            for v in sr.starting_points:
                if v.uri not in seen_sp:
                    merged.starting_points.append(v)
                    seen_sp.add(v.uri)
            for v in sr.initial_candidates:
                if v.uri not in seen_ic:
                    merged.initial_candidates.append(v)
                    seen_ic.add(v.uri)
            if not merged.query_vector and sr.query_vector:
                merged.query_vector = sr.query_vector
            merged.root_uris.extend(sr.root_uris)
        return merged
