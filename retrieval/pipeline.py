"""Retrieval pipeline orchestrator.

Runs 4 stages sequentially — errors are propagated, not silently degraded:
  Stage 1  QueryPlanner        — classify query into TypedQuery[]
  Stage 2  SeedRetriever       — mixed-level search, or direct L2 search
  Stage 3  HierarchicalSearcher — expand to L2 leaves, or select direct L2 hits
  Optional SceneRetrievalStage — supplement and rank scoped memory evidence
  Stage 4  ResultRanker     — keep only L2, sort, top-k -> RetrievedBlock[]

Returns a pure structured SearchMemoryResult.
"""

from __future__ import annotations

import logging
import asyncio
import time

from core.errors import RetrievalError
from core.models import (
    LeafHit,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
    SeedResult,
    SeedHit,
    TypedQuery,
)
from retrieval.candidate_pool import candidate_top_k
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.hotness import blend_hotness
from retrieval.identity import memory_identity
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker, _vec_to_leaf
from retrieval.scene_stage import SceneRetrievalStage
from retrieval.seed_retriever import SeedRetriever
from retrieval.trace import RetrievalTrace, TraceTimer
from scene.retrieval import SceneRanker, SceneSearchSource, select_scene_evidence

logger = logging.getLogger(__name__)
from core.uri_parser import strip_l2_suffix
from retrieval.graph_retriever import GraphRetrievalReport, OneHopRetriever
from retrieval.edge_retrieval_contract import GraphExpansionOutcome, RetrievalOrigin


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
        scene_ranker: SceneRanker | None = None,
        scene_source: SceneSearchSource | None = None,
        graph_retriever=None,
        graph_supplement=None,
    ) -> None:
        self.planner = planner
        self.seed_retriever = seed_retriever
        self.hierarchical_searcher = hierarchical_searcher
        self.assembly = assembly
        self.cfg = config or RetrievalConfig()
        self._context_reader = context_reader
        self._access_tracker = access_tracker
        self._scene_stage = (
            SceneRetrievalStage(scene_source, scene_ranker)
            if scene_source is not None and scene_ranker is not None else None
        )
        self._graph_retriever = graph_retriever
        self._graph_supplement = graph_supplement

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
        memory_directories: list[str] | None = None,
        collect_graph_supplement: bool = False,
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
            tq.memory_directories = memory_directories
            sr = self._run_seed(tq, ctx, trace, mode=mode)
            all_seed_results.append(sr)
            for h in sr.starting_points:
                level_counts[f"L{h.level}"] = level_counts.get(f"L{h.level}", 0) + 1
            for h in sr.initial_candidates:
                level_counts[f"L{h.level}"] = level_counts.get(f"L{h.level}", 0) + 1
        trace.level_histogram = level_counts

        # Stage 3
        all_leaf_hits: list[LeafHit] = []
        query_results = []
        scene_stage = self._scene_stage.bind(ctx, typed_queries, trace) if self._scene_stage else None
        for tq, sr in zip(typed_queries, all_seed_results):
            reserve = None
            leaves = self._run_expand(
                tq, sr, ctx, trace,
                limit=candidate_top_k(self.cfg, tq.top_k, fallback_top_k=tq.top_k), mode=mode,
                score_threshold=score_threshold,
            )
            if scene_stage is not None:
                is_visible = getattr(self._context_reader, "is_visible", None)
                if callable(is_visible):
                    leaves = [hit for hit in leaves if is_visible(hit.uri, ctx)]
                candidates = scene_stage.apply(leaves, tq, sr.query_vector, ctx, trace)
                leaves, reserve = candidates.hits, candidates.direct_evidence_reserve
            query_results.append((tq, leaves, sr if self.cfg.expand_hierarchy else None, reserve))
            all_leaf_hits.extend(leaves)

        pregraph_hits = list(all_leaf_hits)
        # Graph additions must obey the same effective category scope as seeds.
        # An empty category list means unrestricted, as in the ordinary path.
        graph_categories = set(categories) if categories else None
        if graph_categories is None and typed_queries and all(tq.categories for tq in typed_queries):
            graph_categories = {c for tq in typed_queries for c in tq.categories}
        # Graph memory never expands resource, trace or skill-only rounds.
        memory_round = all(str(tq.context_type).upper() == "MEMORY" for tq in typed_queries) and bool(typed_queries)
        memory_round = memory_round and categories != ["session_summary"]
        if self.cfg.enable_graph_expansion and self._graph_retriever and memory_round:
            vector = next((sr.query_vector for tq, sr in zip(typed_queries, all_seed_results)
                           if tq.text == query and sr.query_vector), None)
            all_leaf_hits = self._expand_by_graph(query, all_leaf_hits, ctx, trace,
                                                 query_vector=vector, categories=graph_categories,
                                                 memory_directories=memory_directories)
        if (collect_graph_supplement and memory_round and self.cfg.enable_graph_expansion
                and self.cfg.graph_supplement_enabled and self._graph_supplement):
            try:
                vector = next((sr.query_vector for tq, sr in zip(typed_queries, all_seed_results)
                               if tq.text == query), None)
                trace.graph_expansion["two_hop_supplement"] = self._graph_supplement.select(
                    query, pregraph_hits, all_leaf_hits, ctx, query_vector=vector,
                    **({"categories": graph_categories} if graph_categories is not None else {}),
                    **({"memory_directories": memory_directories} if memory_directories is not None else {}))
            except Exception as exc:
                logger.warning("Graph supplement failed; preserving baseline: %s", exc)
                trace.graph_expansion["two_hop_supplement_error"] = type(exc).__name__

        # Stage 4
        # Direct mode must not reintroduce candidates rejected by its threshold.
        merged_seed = self._merge_seeds(all_seed_results) if self.cfg.expand_hierarchy else None
        blocks = self._run_assembly(
            typed_queries, all_leaf_hits, merged_seed, trace, ctx,
            fill_content_for_top_k=fill_content_for_top_k,
            query_results=query_results if self._scene_stage is not None else None,
        )

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

    def _expand_by_graph(
        self,
        query: str,
        candidates: list[LeafHit],
        ctx: RequestContext,
        trace: RetrievalTrace,
        *,
        query_vector: list[float] | None = None,
        categories: set[str] | None = None,
        memory_directories: list[str] | None = None,
    ) -> list[LeafHit]:
        """Expand candidates via the configured graph retriever (serial).

        Runs after hierarchical expansion: the top-K candidates are used as
        seeds, the retriever returns ``GraphHit[]`` (seeds + neighbors with
        inherited scores), and non-seed hits are converted to ``LeafHit`` and
        appended. The injected OneHopRetriever uses the existing relation reader.

        Returns the original candidates plus the graph-expanded neighbors.
        """
        if not candidates:
            return candidates

        with TraceTimer("graph_expansion") as st:
            st.input_count = len(candidates)

            # Take top-k seed nodes for expansion
            seeds = candidates[: self.cfg.graph_expand_top_k]
            seed_hits = [
                SeedHit(
                    uri=c.uri,
                    score=c.score,
                    level=c.level,
                    category=c.category,
                    abstract=c.abstract,
                )
                for c in seeds
            ]

            try:
                search_method = getattr(
                    self._graph_retriever,
                    "search_with_report",
                    self._graph_retriever.search,
                )
                # Only the native retriever opts into this additive contract.
                # Third-party retrievers keep their existing signatures.
                extra = {}
                if type(self._graph_retriever) is OneHopRetriever and categories is not None:
                    extra["categories"] = categories
                if (type(self._graph_retriever) is OneHopRetriever
                        and self._graph_retriever._embedder is self.seed_retriever.embedder):
                    extra["query_vector"] = query_vector
                search_result = self._run_async(
                    search_method(
                        query=query,
                        ctx=ctx,
                        seed_hits=seed_hits,
                        top_k=self.cfg.graph_expand_limit * max(1, len(seeds)),
                        **extra,
                    )
                )
                if (
                    isinstance(search_result, tuple)
                    and len(search_result) == 2
                    and isinstance(search_result[1], GraphRetrievalReport)
                ):
                    graph_hits, report = search_result
                else:
                    graph_hits = search_result
                    report = GraphRetrievalReport(
                        outcome=(
                            GraphExpansionOutcome.ADDED_NEIGHBORS
                            if any(hit.hop_distance > 0 for hit in graph_hits)
                            else GraphExpansionOutcome.NO_EDGES
                        ),
                        anchor_count=len(seeds),
                        graph_added_node_count=sum(hit.hop_distance > 0 for hit in graph_hits),
                    )
            except Exception as exc:
                logger.debug("[graph_expansion] search failed: %s", exc)
                st.output_count = len(candidates)
                report = GraphRetrievalReport(
                    outcome=GraphExpansionOutcome.GRAPH_FAILURE,
                    anchor_count=len(seeds),
                    graph_lookup_failure_count=1,
                )
                st.metrics = report.to_dict()
                trace.graph_expansion = dict(st.metrics)
                trace.add_stage(st)
                return candidates

            # Normalise URIs before dedup: vector hits carry /content.md,
            # graph neighbour URIs do not — without stripping, a node that
            # is both a seed and a neighbour would appear twice.
            candidate_uris = {strip_l2_suffix(c.uri) for c in candidates}
            expanded_hits: list[LeafHit] = []
            for hit in graph_hits:
                norm = strip_l2_suffix(hit.uri)
                if norm in candidate_uris:
                    if hit.hop_distance > 0:
                        report.deduplicated_target_count += 1
                    continue
                # Traversal output is only a candidate reference. Hydrate it
                # through the same ACL-aware ContextFS path used by explicit
                # reads before it can enter prompt assembly.
                if self._context_reader is None:
                    report.hydration_rejection_count += 1
                    continue
                try:
                    hydrated = self._context_reader.read_authorized(hit.uri, ctx)
                except Exception as exc:
                    logger.debug("[graph_expansion] hydration rejected %s: %s", hit.uri, exc)
                    report.hydration_rejection_count += 1
                    continue
                if categories is not None and hydrated.category not in categories:
                    report.hydration_rejection_count += 1
                    continue
                from core.memory_names import memory_directory
                if memory_directories is not None and memory_directory(hydrated.uri) not in memory_directories:
                    report.hydration_rejection_count += 1
                    continue
                candidate_uris.add(norm)
                expanded_hits.append(
                    LeafHit(
                        uri=hydrated.uri,
                        score=hit.score,
                        level=2,  # Graph-expanded nodes are treated as L2
                        category=hydrated.category,
                        owner_space=hydrated.owner_space,
                        abstract="",
                        has_overview=bool(hydrated.overview),
                        has_content=bool(hydrated.content_excerpt),
                        metadata={
                            **dict(hydrated.metadata or {}),
                            "retrieval_origin": RetrievalOrigin.GRAPH_NEIGHBOR.value,
                            "graph_anchor_uri": hit.anchor_uri,
                            "graph_target_uri": hit.uri,
                            "graph_relation": (
                                hit.relation_types[0] if hit.relation_types else None
                            ),
                            "graph_edge_similarity": hit.edge_similarity,
                            "graph_edge_weight": hit.edge_weight,
                            "graph_score": hit.score,
                            "graph_hop_distance": hit.hop_distance,
                            "graph_explanation": hit.explanation,
                        },
                        overview=hydrated.overview,
                        content_excerpt=hydrated.content_excerpt,
                    )
                )

            st.output_count = len(candidates) + len(expanded_hits)
            report.graph_added_node_count = len(expanded_hits)
            if report.hydration_rejection_count and not expanded_hits:
                report.outcome = GraphExpansionOutcome.HYDRATION_REJECTION
            st.metrics = report.to_dict()
            trace.graph_expansion = dict(st.metrics)
        trace.add_stage(st)

        return candidates + expanded_hits

    @staticmethod
    def _run_async(coro):
        """Drive an async coroutine from sync context.

        The retrieval pipeline is synchronous, but graph retrievers expose an
        async ``search``. We run the coroutine in a throwaway event loop, or
        in a worker thread if a loop is already running on this thread.
        """
        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if not in_running_loop:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        # Running loop on this thread — execute in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()

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
                logger.error("[seed_retrieval] failed: %s", exc, exc_info=True)
                raise
            st.output_count = len(result.starting_points) + len(result.initial_candidates)
        trace.add_stage(st)
        return result

    def _run_expand(self, tq, sr, ctx, trace, *, limit, mode, score_threshold) -> list[LeafHit]:
        if not self.cfg.expand_hierarchy:
            with TraceTimer("direct_retrieval") as st:
                st.input_count = len(sr.initial_candidates)
                threshold = self.cfg.default_score_threshold if score_threshold is None else score_threshold
                hits = [hit for hit in self._fallback_l2_from_seed(sr) if hit.score > threshold]
                # Use the existing candidate pool before local semantic heat.
                hits.sort(key=lambda hit: hit.score, reverse=True)
                hits = hits[:limit]
                if sr.query_vector:
                    for hit in hits:
                        hit.score = blend_hotness(hit.score, hit.active_count, hit.updated_at, self.cfg)
                    hits.sort(key=lambda hit: hit.score, reverse=True)
                st.output_count = len(hits)
            trace.add_stage(st)
            return hits

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

    def _run_assembly(
        self, typed_queries, leaf_hits, merged_seed, trace, ctx=None,
        fill_content_for_top_k=0, query_results=None,
    ) -> list[RetrievedBlock]:
        with TraceTimer("assembly") as st:
            st.input_count = len(leaf_hits)
            all_blocks = []
            seen_uris = set()

            # Run assembly for each typed query (e.g., memory/skill/resource)
            # and deduplicate results by URI
            inputs = query_results if query_results is not None else [
                (tq, leaf_hits, merged_seed, None)
                for tq in (typed_queries or [TypedQuery(text="", context_type="", categories=[])])
            ]
            for tq, query_hits, query_seed, reserve in inputs:
                if query_results is not None and str(tq.context_type).upper() == "MEMORY":
                    # Scene candidates keep their per-query reserve. Graph
                    # additions still reach assembly without replacing them.
                    known = {memory_identity(h.uri, h.metadata) for h in query_hits}
                    query_hits = query_hits + [h for h in leaf_hits
                        if h.metadata.get("retrieval_origin") == "graph_neighbor"
                        and memory_identity(h.uri, h.metadata) not in known
                        and (not tq.categories or h.category in tq.categories)]
                try:
                    blocks = self.assembly.assemble(
                        tq, query_hits, query_seed, ctx=ctx,
                        fill_content_for_top_k=fill_content_for_top_k,
                        context_reader=self._context_reader,
                        **({"direct_evidence_reserve": reserve} if reserve is not None else {}),
                    )
                except Exception as exc:
                    logger.error("[assembly] assemble failed: %s", exc, exc_info=True)
                    raise

                for b in blocks:
                    uri = getattr(b, "uri", None) or getattr(b, "source_uri", None)
                    identity = memory_identity(uri, getattr(b, "metadata", None)) if uri else None
                    if identity and identity in seen_uris:
                        continue
                    if identity:
                        seen_uris.add(identity)
                    all_blocks.append(b)

            final_limit = max((getattr(tq, "top_k", 0) for tq in typed_queries), default=0)
            if getattr(self.cfg, "global_result_rerank_enabled", False) and final_limit > 0:
                graph_blocks = [b for b in all_blocks if b.metadata.get("retrieval_origin") == "graph_neighbor"]
                all_blocks = [b for b in all_blocks if b.metadata.get("retrieval_origin") != "graph_neighbor"]
                all_blocks.sort(
                    key=lambda block: (
                        getattr(block, "ranking_score", None)
                        if getattr(block, "ranking_score", None) is not None
                        else getattr(block, "score", 0.0)
                    ),
                    reverse=True,
                )
                reserve = next((reserve for _, _, _, reserve in inputs if reserve is not None), None)
                selected = select_scene_evidence(
                    all_blocks, final_limit, reserve,
                    lambda block: bool(block.metadata.get("scene_expanded")),
                )
                selected_ids = {id(block) for block in selected}
                all_blocks = [block for block in all_blocks if id(block) in selected_ids]
                all_blocks.extend(graph_blocks)

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
        seen_ic: set[tuple[str, str]] = set()
        for sr in seed_results:
            for v in sr.starting_points:
                if v.uri not in seen_sp:
                    merged.starting_points.append(v)
                    seen_sp.add(v.uri)
            for v in sr.initial_candidates:
                identity = memory_identity(v.uri, v.metadata)
                if identity not in seen_ic:
                    merged.initial_candidates.append(v)
                    seen_ic.add(identity)
            if not merged.query_vector and sr.query_vector:
                merged.query_vector = sr.query_vector
            merged.root_uris.extend(sr.root_uris)
        return merged
