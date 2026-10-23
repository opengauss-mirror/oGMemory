"""One-hop semantic retrieval over the existing GraphService relation reader.

Only stored compatible edge vectors are scored. No LLM, graph database copy,
or new node embedding is used at retrieval time.
"""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.models import RelationNeighbor, RequestContext, SeedHit
from core.uri_parser import strip_l2_suffix
from core.vector_math import cosine_similarity
from retrieval.edge_retrieval_contract import GraphExpansionOutcome

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Modes
# ============================================================================


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class GraphHit:
    """A graph-retrieved hit with additional metadata."""

    uri: str
    score: float
    hop_distance: int = 0
    path: list[str] | None = None  # Path from seed to this node
    relation_types: list[str] | None = None  # Relations traversed
    effective_weight: float | None = None  # Time-decayed weight
    anchor_uri: str | None = None
    edge_similarity: float | None = None
    edge_weight: float | None = None
    explanation: str | None = None


@dataclass
class GraphRetrievalReport:
    """Bounded, content-free measurements for one graph search."""

    outcome: str = GraphExpansionOutcome.NO_EDGES
    anchor_count: int = 0
    candidate_edge_count: int = 0
    accepted_edge_count: int = 0
    threshold_rejection_count: int = 0
    compatibility_rejection_count: int = 0
    deduplicated_target_count: int = 0
    graph_lookup_failure_count: int = 0
    graph_added_node_count: int = 0
    hydration_rejection_count: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            key: value.value if isinstance(value, StrEnum) else value
            for key, value in self.__dict__.items()
        }


# ============================================================================
# Base Graph Retriever
# ============================================================================


class BaseGraphRetriever(ABC):
    """Base class for graph-based retrievers.

    Args:
        graph_store: Existing GraphService exposing canonical relation reads.
        config: Optional configuration dict.
    """

    def __init__(
        self,
        graph_store,
        config: dict[str, Any] | None = None,
        *,
        embedder=None,
        embedding_profile: dict[str, Any] | None = None,
    ):
        self._graph_store = graph_store
        self._config = config or {}
        # S8R-02 wires the already-shared embedder and its compatibility
        # identity. S8R-03 will use them for query-once semantic scoring.
        self._embedder = embedder
        self._embedding_profile = dict(embedding_profile or {})

    @abstractmethod
    async def search(
        self,
        query: str,
        ctx: RequestContext,
        seed_hits: list[SeedHit],
        top_k: int = 10,
        query_time: datetime | None = None,
    ) -> list[GraphHit]:
        """Search using graph structure.

        Args:
            query: Query text
            ctx: Request context
            seed_hits: Initial seed hits from vector search
            top_k: Number of results to return
            query_time: Optional time point for validity check

        Returns:
            List of GraphHit sorted by relevance
        """
        ...

    # -- shared helpers -----------------------------------------------------

    def _read_relation_neighbors(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        limit: int,
        min_weight: float = 0.0,
    ) -> list[RelationNeighbor]:
        """Read typed neighbors through the graph-service contract."""
        reader = getattr(self._graph_store, "get_relation_neighbors", None)
        if reader is not None:
            values = reader(uri, ctx, limit=limit, min_weight=min_weight)
        else:
            # Compatibility for third-party/test stores. Production
            # composition supplies GraphService and never takes this branch.
            values = self._graph_store.get_one_hop(
                uri, ctx, limit=limit, min_weight=min_weight
            )
        return [
            value
            if isinstance(value, RelationNeighbor)
            else RelationNeighbor(
                uri=value.get("uri", ""),
                abstract=value.get("abstract", ""),
                overview=value.get("overview", ""),
                category=value.get("category", ""),
                relation_type=value.get("relation_type", ""),
                weight=float(value.get("weight", 0.0)),
                reason=value.get("reason", ""),
                edge_projection=value.get("edge_projection"),
                source_revision=value.get("source_revision"),
                target_revision=value.get("target_revision"),
            )
            for value in values
        ]

    async def search_with_report(self, *args, **kwargs):
        hits = await self.search(*args, **kwargs)
        report = GraphRetrievalReport(
            outcome=(
                GraphExpansionOutcome.ADDED_NEIGHBORS
                if any(hit.hop_distance > 0 for hit in hits)
                else GraphExpansionOutcome.NO_EDGES
            ),
            anchor_count=len(kwargs.get("seed_hits", [])),
            graph_added_node_count=sum(hit.hop_distance > 0 for hit in hits),
        )
        return hits, report


# ============================================================================
# 1-Hop Retriever (default — the original _expand_by_graph logic)
# ============================================================================


class OneHopRetriever(BaseGraphRetriever):
    """Simple 1-hop neighbor expansion.

    Takes the top-K ordinary anchors and reads outgoing canonical relations.
    Scores compatible stored edge vectors with query cosine times edge weight;
    the original anchors are retained on failures or empty graph results.
    """

    def __init__(
        self,
        graph_store,
        config: dict[str, Any] | None = None,
        *,
        embedder=None,
        embedding_profile: dict[str, Any] | None = None,
    ):
        super().__init__(
            graph_store,
            config,
            embedder=embedder,
            embedding_profile=embedding_profile,
        )
        self._expansion_limit = self._config.get("expansion_limit", 5)
        self._edges_per_node = self._config.get("edges_per_node", 3)
        self._min_weight = self._config.get("min_weight", 0.0)
        self._similarity_threshold = self._config.get("edge_similarity_threshold", 0.35)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return cosine_similarity(left, right, strict_dimensions=True)

    def _compatible_vector(self, neighbor: RelationNeighbor) -> list[float] | None:
        projection = neighbor.edge_projection
        if not isinstance(projection, dict):
            return None
        vector = projection.get("vector")
        profile = self._embedding_profile
        expected = {
            "embedding_profile": profile.get("profile_id"),
            "model": profile.get("model"),
            "dimension": profile.get("dimension"),
            "representation_version": profile.get("representation_version"),
        }
        for key, value in expected.items():
            if value is not None and projection.get(key) != value:
                return None
        if projection.get("source_revision") != neighbor.source_revision:
            return None
        if projection.get("target_revision") != neighbor.target_revision:
            return None
        if not isinstance(vector, (list, tuple)):
            return None
        if projection.get("dimension") != len(vector):
            return None
        try:
            values = [float(value) for value in vector]
        except (TypeError,ValueError):
            return None
        return values if values and all(math.isfinite(value) for value in values) else None

    def _relation_neighbors(self, uri: str, ctx: RequestContext) -> list[RelationNeighbor]:
        return self._read_relation_neighbors(
            uri, ctx, limit=self._edges_per_node, min_weight=self._min_weight
        )

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        seed_hits: list[SeedHit],
        top_k: int = 10,
        query_time: datetime | None = None,
    ) -> list[GraphHit]:
        hits, _ = await self._search_with_report(query, ctx, seed_hits, top_k, query_time)
        return hits

    async def search_with_report(
        self,
        query: str,
        ctx: RequestContext,
        seed_hits: list[SeedHit],
        top_k: int = 10,
        query_time: datetime | None = None,
        *,
        query_vector: list[float] | None = None,
        categories: set[str] | None = None,
    ) -> tuple[list[GraphHit], GraphRetrievalReport]:
        return await self._search_with_report(query, ctx, seed_hits, top_k, query_time,
                                              query_vector=query_vector, categories=categories)

    async def _search_with_report(
        self,
        query: str,
        ctx: RequestContext,
        seed_hits: list[SeedHit],
        top_k: int,
        query_time: datetime | None,
        *,
        query_vector: list[float] | None = None,
        categories: set[str] | None = None,
    ) -> tuple[list[GraphHit], GraphRetrievalReport]:
        """Search with 1-hop expansion. Returns seed hits + expanded neighbors."""
        del query_time
        report = GraphRetrievalReport(anchor_count=len(seed_hits))
        # Seed hits pass through unchanged and in their original order. Keys are
        # L2-suffix-normalised so a neighbour (no suffix) that is also a
        # seed (with /content.md) is recognised as the same node.
        anchors = [GraphHit(uri=hit.uri, score=hit.score, hop_distance=0) for hit in seed_hits]
        anchor_keys = {strip_l2_suffix(hit.uri) for hit in seed_hits}
        graph_hits: dict[str, GraphHit] = {}

        if not query or (query_vector is None and self._embedder is None):
            report.outcome = GraphExpansionOutcome.QUERY_EMBEDDING_FAILURE
            return anchors, report
        try:
            if query_vector is None:
                query_vectors = await asyncio.to_thread(self._embedder.embed_texts, [query])
                if len(query_vectors) != 1:
                    report.outcome = GraphExpansionOutcome.QUERY_EMBEDDING_FAILURE
                    return anchors, report
                query_vector = query_vectors[0]
            query_vector = [float(value) for value in query_vector]
            if not query_vector or not all(math.isfinite(v) for v in query_vector):
                raise ValueError("Non-finite or empty graph query vector")
        except Exception as exc:
            logger.debug("graph query embedding failed: %s", exc)
            report.outcome = GraphExpansionOutcome.QUERY_EMBEDDING_FAILURE
            return anchors, report

        for hit in seed_hits[: self._expansion_limit]:
            try:
                neighbors = self._relation_neighbors(strip_l2_suffix(hit.uri), ctx)
            except Exception as exc:
                logger.debug("graph expansion failed for %s: %s", hit.uri, exc)
                report.graph_lookup_failure_count += 1
                continue

            report.candidate_edge_count += len(neighbors)

            for nb in neighbors:
                if categories is not None and nb.category not in categories:
                    report.hydration_rejection_count += 1
                    continue
                nb_uri = nb.uri
                if not nb_uri:
                    continue
                normalized = strip_l2_suffix(nb_uri)
                if normalized in anchor_keys:
                    report.deduplicated_target_count += 1
                    continue
                weight = float(nb.weight)
                if not math.isfinite(weight) or weight < self._min_weight:
                    report.threshold_rejection_count += 1
                    continue
                vector = self._compatible_vector(nb)
                if vector is None:
                    report.compatibility_rejection_count += 1
                    continue
                try:
                    similarity = self._cosine(query_vector, vector)
                except ValueError:
                    report.compatibility_rejection_count += 1
                    continue
                if similarity < self._similarity_threshold:
                    report.threshold_rejection_count += 1
                    continue
                report.accepted_edge_count += 1
                graph_score = similarity * weight
                existing = graph_hits.get(normalized)
                if existing is not None:
                    report.deduplicated_target_count += 1
                if existing is None or graph_score > existing.score:
                    graph_hits[normalized] = GraphHit(
                        uri=nb_uri,
                        score=graph_score,
                        hop_distance=1,
                        relation_types=[nb.relation_type]
                        if nb.relation_type
                        else None,
                        path=[hit.uri, nb_uri],
                        anchor_uri=hit.uri,
                        edge_similarity=similarity,
                        edge_weight=weight,
                        explanation=nb.reason or None,
                    )

        ranked_neighbors = sorted(graph_hits.values(), key=lambda item: (-item.score, item.uri))
        # ``top_k`` bounds graph additions; anchors are an additive mandatory
        # stream and therefore do not consume graph-result slots.
        added = ranked_neighbors[:top_k]
        report.graph_added_node_count = len(added)
        if added:
            report.outcome = GraphExpansionOutcome.ADDED_NEIGHBORS
        elif report.graph_lookup_failure_count and not report.candidate_edge_count:
            report.outcome = GraphExpansionOutcome.GRAPH_FAILURE
        elif report.candidate_edge_count:
            report.outcome = GraphExpansionOutcome.ALL_REJECTED
        else:
            report.outcome = GraphExpansionOutcome.NO_EDGES
        return anchors + added, report
