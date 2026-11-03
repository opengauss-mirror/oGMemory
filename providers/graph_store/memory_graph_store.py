"""In-memory implementation of GraphStore for testing.

This implementation stores all nodes and edges in memory dictionaries.
Useful for unit testing without a PostgreSQL dependency.

Usage::

    from providers.graph_store import MemoryGraphStore
    from core.graph_models import create_raw_node

    store = MemoryGraphStore()
    node = create_raw_node(RawDataType.PROMPT, "Hello world", "acc1")
    store.create_node(node.serialize(), ctx)
"""

from __future__ import annotations

import math
from typing import Any

from core.errors import AccessDeniedError, NodeNotFoundError
from core.graph_enums import (
    GraphEdgeType,
    GraphNodeType,
    RawDataType,
)
from core.graph_models import (
    ExtractedNode,
    GraphEdge,
    GraphNode,
    GraphStats,
    MemoryNode,
    RawNode,
    SubgraphResult,
    SupportingChain,
    node_from_payload,
)
from core.interfaces import GraphStore
from core.models import RequestContext
from core.temporal import seconds, within_time


class MemoryGraphStore(GraphStore):
    """In-memory implementation of GraphStore.

    Thread-unsafe, for testing only.
    """

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        # Index for faster lookups
        self._nodes_by_type: dict[GraphNodeType, list[str]] = {
            GraphNodeType.RAW: [],
            GraphNodeType.EXTRACTED: [],
            GraphNodeType.MEMORY: [],
        }
        self._edges_by_source: dict[str, list[str]] = {}
        self._edges_by_target: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_access(self, node_account_id: str, ctx: RequestContext) -> None:
        """Check if the request context has access to the node."""
        if node_account_id != ctx.account_id:
            raise AccessDeniedError(
                f"graph_node:{node_account_id}",
                ctx.account_id,
                f"Node belongs to account '{node_account_id}'",
            )

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Create a new graph node."""
        node = node_from_payload(payload)
        self._check_access(node.account_id, ctx)
        if node.node_id in self._nodes:
            self._check_access(self._nodes[node.node_id].account_id, ctx)
        self._nodes[node.node_id] = node
        self._nodes_by_type[node.node_type].append(node.node_id)
        return node

    def get_node(self, node_id: str, ctx: RequestContext) -> GraphNode | None:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node and node.account_id != ctx.account_id:
            return None
        return node

    def update_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Update an existing node."""
        node = node_from_payload(payload)
        if node.node_id not in self._nodes:
            raise NodeNotFoundError(node.node_id)
        self._check_access(self._nodes[node.node_id].account_id, ctx)
        self._check_access(node.account_id, ctx)
        self._nodes[node.node_id] = node
        return node

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all its edges."""
        if node_id not in self._nodes:
            return

        node = self._nodes[node_id]
        self._check_access(node.account_id, ctx)

        # Remove from type index
        self._nodes_by_type[node.node_type].remove(node_id)

        # Remove connected edges
        for edge_id in list(self._edges.keys()):
            edge = self._edges[edge_id]
            if edge.source_id == node_id or edge.target_id == node_id:
                self._remove_edge(edge_id)

        del self._nodes[node_id]

    def list_nodes(
        self,
        node_type: GraphNodeType | None,
        ctx: RequestContext,
        limit: int = 4096,
        offset: int = 0,
        raw_type: RawDataType | None = None,
        extractor_type: str | None = None,
        parent_audio_node_ids: list[str] | None = None,
    ) -> list[GraphNode]:
        """List nodes of a specific type, optionally filtered further by
        ``raw_type`` (only applies when ``node_type == RAW``).

        Pushing the ``raw_type`` filter into the store layer matters:
        previously ``GraphService.list_raw_nodes`` paginated the store
        with ``node_type=RAW`` and then filtered by ``raw_type`` in
        Python AFTER pagination. With a default ``limit=100`` and many
        VIDEO RawNodes, the first 100 returned were all VIDEO and the
        AUDIO children were silently dropped — voice_asr saw 0 AUDIO
        nodes even when the graph had 100+ of them.

        ``extractor_type`` and ``parent_audio_node_ids`` are
        ``EXTRACTED``-only filters; for the in-memory store they are
        post-evaluated in Python after the per-type / per-tenant
        scan (no SQL push-down is needed because there's no
        pagination to leak through). ``parent_audio_node_ids``
        matches the ``extracted_content.source_audio_node_id`` slot
        that ``voice_asr`` writes, so the same call pattern works
        on both backends.
        """
        if node_type:
            node_ids = self._nodes_by_type.get(node_type, [])
        else:
            node_ids = list(self._nodes.keys())

        result = []
        for node_id in node_ids:
            node = self._nodes.get(node_id)
            if node is None or node.account_id != ctx.account_id:
                continue
            # Push raw_type filter into the store so pagination cannot
            # silently drop nodes of a different type.
            if raw_type is not None and node_type == GraphNodeType.RAW:
                if getattr(node, "raw_type", None) != raw_type:
                    continue
            # ``extractor_type`` / ``parent_audio_node_ids`` filters
            # for EXTRACTED nodes — same push-down principle as
            # ``raw_type`` above.
            if node_type == GraphNodeType.EXTRACTED:
                if (
                    extractor_type is not None
                    and getattr(node, "extractor_type", None) != extractor_type
                ):
                    continue
                if parent_audio_node_ids:
                    ec = getattr(node, "extracted_content", None) or {}
                    audio_id = ec.get("source_audio_node_id")
                    if audio_id not in parent_audio_node_ids:
                        continue
            result.append(node)

        result.sort(key=lambda n: n.updated_at, reverse=True)
        if limit is None:
            return result[offset:]
        return result[offset:offset + limit]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def create_edge(self, edge: GraphEdge, ctx: RequestContext) -> GraphEdge:
        """Create a new edge between nodes."""
        self._check_access(edge.account_id, ctx)
        for node_id in (edge.source_id, edge.target_id):
            if self.get_node(node_id, ctx) is None:
                raise NodeNotFoundError(node_id)
        if edge.edge_id in self._edges:
            self._check_access(self._edges[edge.edge_id].account_id, ctx)
            self._remove_edge(edge.edge_id)
        self._edges[edge.edge_id] = edge

        # Update indexes
        if edge.source_id not in self._edges_by_source:
            self._edges_by_source[edge.source_id] = []
        self._edges_by_source[edge.source_id].append(edge.edge_id)

        if edge.target_id not in self._edges_by_target:
            self._edges_by_target[edge.target_id] = []
        self._edges_by_target[edge.target_id].append(edge.edge_id)

        # Incremental PageRank update
        self._update_pagerank_incremental(edge.target_id, edge.weight, ctx)

        return edge

    def list_identity_pairs(
        self, source_ids: list[str], ctx: RequestContext,
    ) -> list[tuple[str, str]]:
        sources = set(source_ids)
        return [
            (e.source_id, e.target_id) for e in self._edges.values()
            if self.get_edge(e.edge_id, ctx) is not None and e.source_id in sources
            and e.edge_type == GraphEdgeType.SAME_IDENTITY
        ]

    def ensure_identity_edges(self, edges: list[GraphEdge], ctx: RequestContext) -> int:
        # Validate the entire batch before mutating anything.
        for edge in edges:
            if edge.account_id != ctx.account_id or edge.edge_type != GraphEdgeType.SAME_IDENTITY:
                raise ValueError("identity edge batch has wrong account or type")
            for node_id in (edge.source_id, edge.target_id):
                node = self._nodes.get(node_id)
                if node is None or node.account_id != ctx.account_id:
                    raise ValueError("identity endpoint missing or outside account")
        existing = set(self.list_identity_pairs([e.source_id for e in edges], ctx))
        inserted = 0
        for edge in edges:
            pair = (edge.source_id, edge.target_id)
            if pair not in existing:
                self.create_edge(edge, ctx)
                existing.add(pair)
                inserted += 1
        return inserted

    def _update_pagerank_incremental(
        self, target_node_id: str, incoming_weight: float, ctx: RequestContext
    ) -> None:
        """Incrementally update PageRank for a node based on incoming edge."""
        target = self.get_node(target_node_id, ctx)
        if not target:
            return

        # Calculate total incoming weight
        total_incoming = 0.0
        for edge_id in self._edges_by_target.get(target_node_id, []):
            edge = self.get_edge(edge_id, ctx)
            if edge:
                total_incoming += edge.weight

        target.pagerank = total_incoming

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge by ID."""
        if self.get_edge(edge_id, ctx) is not None:
            self._remove_edge(edge_id)

    def _remove_edge(self, edge_id: str) -> None:
        """Remove an edge from storage."""
        edge = self._edges.pop(edge_id, None)
        if edge:
            if edge.edge_id in self._edges_by_source.get(edge.source_id, []):
                self._edges_by_source[edge.source_id].remove(edge.edge_id)
            if edge.edge_id in self._edges_by_target.get(edge.target_id, []):
                self._edges_by_target[edge.target_id].remove(edge.edge_id)

    def get_edge(self, edge_id: str, ctx: RequestContext) -> GraphEdge | None:
        """Get an edge by ID."""
        edge = self._edges.get(edge_id)
        if (edge is None or edge.account_id != ctx.account_id
                or self.get_node(edge.source_id, ctx) is None
                or self.get_node(edge.target_id, ctx) is None):
            return None
        return edge

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges."""
        results = []
        if self.get_node(node_id, ctx) is None:
            return results

        if direction in ("outgoing", "both"):
            for edge_id in self._edges_by_source.get(node_id, []):
                edge = self.get_edge(edge_id, ctx)
                if edge and (edge_types is None or edge.edge_type in edge_types):
                    neighbor = self.get_node(edge.target_id, ctx)
                    if neighbor:
                        results.append((neighbor, edge, "outgoing"))

        if direction in ("incoming", "both"):
            for edge_id in self._edges_by_target.get(node_id, []):
                edge = self.get_edge(edge_id, ctx)
                if edge and (edge_types is None or edge.edge_type in edge_types):
                    neighbor = self.get_node(edge.source_id, ctx)
                    if neighbor:
                        results.append((neighbor, edge, "incoming"))

        return results

    def traverse(
        self, start_node_id: str, depth: int, ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        from .traversal import traverse
        return traverse(self, start_node_id, depth, ctx, edge_types)

    def get_supporting_chain(self, memory_id: str, ctx: RequestContext) -> SupportingChain:
        from .traversal import get_supporting_chain
        return get_supporting_chain(self, memory_id, ctx)

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores for nodes."""
        nodes_to_compute = [nid for nid in (node_ids or list(self._nodes))
                            if self.get_node(nid, ctx) is not None]
        if not nodes_to_compute:
            return {}

        # Build adjacency
        incoming: dict[str, list] = {nid: [] for nid in nodes_to_compute}
        outgoing_count: dict[str, int] = {nid: 0 for nid in nodes_to_compute}

        for edge in self._edges.values():
            if (self.get_edge(edge.edge_id, ctx) is not None
                    and edge.source_id in nodes_to_compute and edge.target_id in nodes_to_compute):
                incoming[edge.target_id].append((edge.source_id, edge.weight))
                outgoing_count[edge.source_id] += 1

        # Iterative PageRank
        N = len(nodes_to_compute)
        scores = {nid: 1.0 / N for nid in nodes_to_compute}

        for _ in range(10):  # Max iterations
            new_scores = {}
            for nid in nodes_to_compute:
                sum_contrib = sum(
                    scores[src] * w / max(outgoing_count[src], 1)
                    for src, w in incoming[nid]
                )
                new_scores[nid] = (1 - damping) / N + damping * sum_contrib

            scores = new_scores

        # Update nodes
        for nid, score in scores.items():
            if nid in self._nodes:
                self._nodes[nid].pagerank = score

        return scores

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a specific node."""
        node = self.get_node(node_id, ctx)
        return node.pagerank if node else 0.0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memories; tags requires all specified tags (empty means no filter)."""
        results = []

        for node in self._nodes.values():
            if not isinstance(node, MemoryNode):
                continue
            if node.account_id != ctx.account_id:
                continue
            if node.pagerank < min_confidence:
                continue
            if topic and node.topic != topic:
                continue
            if tags and not set(tags).issubset(node.tags or []):
                continue
            if query:
                if query.lower() not in node.summary.lower() and query.lower() not in node.content.lower():
                    continue

            results.append(node)

        results.sort(key=lambda n: n.pagerank, reverse=True)
        return results[:top_k]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by type and optional text query."""
        results = []

        for node in self._nodes.values():
            if not isinstance(node, RawNode):
                continue
            if node.account_id != ctx.account_id:
                continue
            if raw_type and node.raw_type != raw_type:
                continue
            if query and query.lower() not in node.content.lower():
                continue

            results.append(node)

        results.sort(key=lambda n: n.updated_at, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two equal-length non-zero vectors.

        Returns 0.0 when either vector has zero norm or the lengths differ
        rather than raising — search stays robust against dirty data.
        """
        from core.vector_math import cosine_similarity
        return cosine_similarity(a, b)

    def search_by_embedding(
        self,
        query_vector: list[float],
        ctx: RequestContext,
        *,
        node_types: tuple[str, ...] = ("extracted",),
        extractor_types: tuple[str, ...] | None = None,
        video_id: str | None = None,
        before_clip: int | None = None,
        before_sec: float | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search over node embeddings.

        Walks every account-scoped node of the requested ``node_types``
        (only ``"extracted"`` carries embeddings today; other types are
        skipped), filters on the optional clip / video / extractor
        constraints, and returns the top_k ``(node_id, score)`` pairs sorted
        by score desc.

        Nodes missing an embedding, with a dimension mismatch, or with
        malformed clip metadata are excluded silently — the search primitive
        is best-effort by design so a single bad row cannot poison results.

        Args:
            query_vector: 1-D query vector; matches dimension of stored
                embeddings or is skipped.
            ctx: Request context for tenant isolation.
            node_types: Tuple of :class:`GraphNodeType` string values
                (``"raw"``, ``"extracted"``, ``"memory"``). Unknown values
                are ignored.
            extractor_types: Optional whitelist of ``extractor_type`` on
                ExtractedNode. ``None`` means no filter.
            video_id: Optional ``extracted_content["_video_id"]`` match.
            before_clip: Optional ordinal upper bound on
                ``extracted_content["_clip_id"]``. Rows missing the key or
                exceeding the bound are excluded.
            top_k: Maximum results to return.

        Returns:
            ``[(node_id, score), ...]`` sorted by score desc, capped at
            ``top_k``. Empty when ``top_k <= 0`` or no candidates pass.
        """
        if before_sec is not None:
            seconds(before_sec)
        if top_k <= 0 or not query_vector:
            return []

        allowed_types: set[GraphNodeType] = set()
        for nt in node_types:
            try:
                allowed_types.add(GraphNodeType(nt))
            except ValueError:
                continue
        if not allowed_types:
            return []

        ranked: list[tuple[str, float]] = []
        for node in self._nodes.values():
            if node.account_id != ctx.account_id:
                continue
            if node.node_type not in allowed_types:
                continue
            # Today only ExtractedNode carries embeddings.
            if not isinstance(node, ExtractedNode):
                continue

            embedding = node.get_embedding()
            if not embedding or len(embedding) != len(query_vector):
                continue

            if (
                extractor_types is not None
                and node.extractor_type not in extractor_types
            ):
                continue

            content = node.extracted_content or {}
            if not within_time(content, before_sec):
                continue
            if video_id is not None and content.get("source_video_id", content.get("_video_id")) != video_id:
                continue
            if before_clip is not None:
                clip_id_value = content.get("source_clip_id", content.get("_clip_id"))
                if clip_id_value is None:
                    continue
                try:
                    if int(clip_id_value) > before_clip:
                        continue
                except (TypeError, ValueError):
                    continue

            ranked.append((node.node_id, self._cosine(query_vector, embedding)))

        # Stable, deterministic tie-break: on identical scores the
        # control loop must see the same node_id order from both the
        # Memory and SQL backends. Sort by (-score, node_id) so the
        # node_id is a strict ascending tiebreaker (matches SQL).
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph."""
        total = raw_count = extracted_count = memory_count = 0
        total_edges = sum(self.get_edge(eid, ctx) is not None for eid in self._edges)
        pageranks = []

        for node in self._nodes.values():
            if node.account_id != ctx.account_id:
                continue
            total += 1
            pageranks.append(node.pagerank)

            if isinstance(node, RawNode):
                raw_count += 1
            elif isinstance(node, ExtractedNode):
                extracted_count += 1
            elif isinstance(node, MemoryNode):
                memory_count += 1

        avg_pr = sum(pageranks) / len(pageranks) if pageranks else 0.0

        # Top memories
        memories = [
            (n.node_id, n.summary, n.pagerank)
            for n in self._nodes.values()
            if isinstance(n, MemoryNode) and n.account_id == ctx.account_id
        ]
        memories.sort(key=lambda x: x[2], reverse=True)
        top_memories = [
            {"node_id": nid, "summary": summary, "pagerank": pr}
            for nid, summary, pr in memories[:5]
        ]

        return GraphStats(
            total_nodes=total,
            raw_nodes=raw_count,
            extracted_nodes=extracted_count,
            memory_nodes=memory_count,
            total_edges=total_edges,
            avg_pagerank=avg_pr,
            top_memories=top_memories,
        )
