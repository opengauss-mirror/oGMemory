"""Graph service for heterogeneous multimodal memory system.

Provides high-level operations for creating and managing graph nodes,
edges, and traversing the evidence chain for memories.

Usage::

    from service.graph_service import GraphService
    from core.models import RequestContext
from core.transcript_lexical import transcript_score, validate_transcript_request
    from core.graph_enums import RawDataType, GraphEdgeType
    from core.graph_models import create_raw_node, create_graph_edge

    service = GraphService(graph_store)
    ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                         session_id="s1", trace_id="t1")

    # Create a raw prompt node
    raw_node = create_raw_node(
        raw_type=RawDataType.PROMPT,
        content="Write a story about a dragon",
        account_id="acc1"
    )
    service.create_raw_node(raw_node, ctx)

    # Create extracted node from raw
    extracted = service.create_extracted_node(
        source_raw_id=raw_node.node_id,
        extractor_type="prompt_analyzer",
        extraction_schema="prompt_structure_v1",
        extracted_content={"intent": "creative_writing", "subject": "dragon"},
        ctx=ctx
    )

    # Create memory from extracted
    memory = service.create_memory_node(
        summary="User wants to write a dragon story",
        supporting_node_ids=[extracted.node_id],
        ctx=ctx
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from core.graph_enums import (
    ExtractionStatus,
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
    create_extracted_node,
    create_graph_edge,
    create_memory_node,
    create_raw_node,
)
from core.interfaces import GraphStore
from core.models import RequestContext
from core.temporal import (
    boundary_transcript,
    contains_time,
    current_clip_key,
    seconds,
    temporal_metadata,
    within_scope,
)
from core.transcript_lexical import transcript_score, validate_transcript_request

if TYPE_CHECKING:
    from core.errors import ContextEngineError
    from core.interfaces import Embedder, VectorIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control-loop value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipScope:
    """Read-side filter handle for control-loop reads.

    Carries the QA-evaluation boundary (which video, which clip ordinal is
    the last "seen" clip) so subsequent read methods can apply it without
    the caller restating the filter. Mirrors M3-Agent's
    ``VideoGraph.truncate_memory_by_clip`` semantics but is non-destructive
    — never mutates the underlying store, just threads a filter object.

    All fields are optional; ``None`` means "no constraint on this
    dimension." Construct directly::

        scope = ClipScope(before_clip=5, video_id="vid1")
        hits = service.search_by_embedding(query, ctx, scope=scope)
    """

    video_id: str | None = None
    before_clip: int | None = None
    # Absolute video seconds; only complete source intervals ending <= this pass.
    before_sec: float | None = None

    def __post_init__(self):
        if self.before_sec is not None:
            seconds(self.before_sec)


@dataclass(frozen=True)
class EvidenceHit:
    """One retrieval hit returned by :meth:`GraphService.search_by_embedding`.

    Bundles the typed :class:`ExtractedNode` with the cosine score plus
    the ``source_video_id`` / ``source_clip_id`` provenance (or, for
    older writers, the legacy ``_video_id`` / ``_clip_id`` keys)
    lifted from ``extracted_content``. Callers that just need the
    score + node can ignore the ``video_id`` / ``clip_id`` fields;
    callers that need provenance (e.g. the M3 control loop) read them
    directly.
    """

    node: ExtractedNode
    score: float
    video_id: str | None = None
    clip_id: int | None = None


class GraphService:
    """High-level service for heterogeneous graph memory operations.

    Provides convenience methods that combine multiple store operations
    and handle edge cases like creating supporting edges automatically.
    """

    def __init__(
        self,
        graph_store: GraphStore | None,
        vector_index: "VectorIndex | None" = None,
        *,
        relation_reader=None,
    ):
        """Initialize the graph service.

        Args:
            graph_store: Underlying graph storage implementation.
            vector_index: Optional :class:`VectorIndex` for ANN search.
                When supplied, :meth:`search_by_embedding` routes
                through it (HNSW + scope columns) instead of falling
                back to the store's in-Python cosine over JSONB. The
                store still owns the node provenance, so we re-read
                ``ExtractedNode`` via ``get_node`` after the index
                returns candidate ``node_id``s.
        """
        self._store = graph_store
        # Typed as Any to avoid an import cycle with
        # ``core.interfaces``. The VectorIndex Protocol is structural;
        # see ``core/interfaces.py:175``.
        self._vector_index = vector_index
        self._relation_reader = relation_reader

    @property
    def supports_relation_reads(self) -> bool:
        """ContextNode URI relations are separate from the UUID/M3 graph."""
        return self._relation_reader is not None

    def get_relation_neighbors(self, uri, ctx, *, limit=10, min_weight=0.0):
        if self._relation_reader is None:
            raise RuntimeError("semantic relation reader is not configured")
        return self._relation_reader.get_relation_neighbors(uri, ctx, limit=limit, min_weight=min_weight)

    def get_traversal_neighbors(self, uri, ctx, *, limit=128, embedding_profile=None):
        if self._relation_reader is None:
            raise RuntimeError("semantic relation reader is not configured")
        return self._relation_reader.get_traversal_neighbors(uri, ctx, limit=limit, embedding_profile=embedding_profile)

    def _create_node_with_edges(self, node, edges, ctx):
        create = getattr(self._store, "create_node_with_edges", None)
        if callable(create):
            return create(node.serialize(), edges, ctx)
        return GraphStore.create_node_with_edges(self._store, node.serialize(), edges, ctx)

    # ====================================================================
    # Raw Node Operations
    # ====================================================================

    def create_raw_node(
        self,
        raw_type: RawDataType,
        content: str,
        ctx: RequestContext,
        metadata: dict | None = None,
        node_id: str | None = None,
    ) -> RawNode:
        """Create a raw multimodal data node.

        Args:
            raw_type: Type of raw data (prompt, image, video, etc.)
            content: The raw content
            ctx: RequestContext for access control
            metadata: Optional metadata dict
            node_id: Optional pre-generated node ID

        Returns:
            The created RawNode
        """
        node = create_raw_node(
            raw_type=raw_type,
            content=content,
            account_id=ctx.account_id,
            metadata=metadata,
            node_id=node_id,
        )
        return self._store.create_node(node.serialize(), ctx)

    def get_raw_node(
        self, node_id: str, ctx: RequestContext
    ) -> RawNode | None:
        """Get a raw node by ID.

        Args:
            node_id: UUID of the raw node
            ctx: RequestContext for access control

        Returns:
            RawNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, RawNode):
            return node
        return None

    def list_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        limit: int = 4096,
        offset: int = 0,
    ) -> list[RawNode]:
        """List raw nodes optionally filtered by type.

        Args:
            raw_type: Filter by raw data type (None for all).
                Pushed into the store layer's WHERE clause so the
                LIMIT applies to the requested type only — no more
                silent drop of AUDIO children when the graph has
                many VIDEO RawNodes.
            ctx: RequestContext for access control
            limit: Max results (default 4096; was 100). Generous so
                callers iterating ALL of one type don't paginate
                unnecessarily. Pass an explicit smaller ``limit`` if
                you need a true "latest N" view.
            offset: Skip count

        Returns:
            List of RawNodes
        """
        # Push raw_type into the store so pagination cannot silently
        # drop nodes of a different type. Pre-fix, this method
        # paginated with node_type=RAW and filtered by raw_type in
        # Python AFTER pagination — with the previous default
        # limit=100 and many VIDEO nodes, AUDIO children were dropped.
        nodes = self._store.list_nodes(
            GraphNodeType.RAW, ctx, limit, offset, raw_type=raw_type,
        )
        return [n for n in nodes if isinstance(n, RawNode)]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        query: str,
        ctx: RequestContext,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by content.

        Args:
            raw_type: Filter by type
            query: Search query
            ctx: RequestContext for access control
            top_k: Max results

        Returns:
            List of matching RawNodes
        """
        return self._store.search_raw_nodes(raw_type, ctx, query, top_k)

    # ====================================================================
    # Extracted Node Operations
    # ====================================================================

    def create_extracted_node(
        self,
        source_raw_id: str,
        extractor_type: str,
        extraction_schema: str,
        ctx: RequestContext,
        extracted_content: dict | None = None,
        node_id: str | None = None,
    ) -> ExtractedNode:
        """Create an extracted node from a raw node.

        Automatically creates the extracted_from edge.

        Args:
            source_raw_id: ID of the source raw node
            extractor_type: Type/name of the extractor
            extraction_schema: Schema used for extraction
            ctx: RequestContext for access control
            extracted_content: The extraction results
            node_id: Optional pre-generated node ID

        Returns:
            The created ExtractedNode
        """
        node = create_extracted_node(
            source_raw_id=source_raw_id,
            extractor_type=extractor_type,
            extraction_schema=extraction_schema,
            account_id=ctx.account_id,
            extracted_content=extracted_content,
            node_id=node_id,
        )

        # Create extracted_from edge
        edge = create_graph_edge(
            source_id=node.node_id,
            target_id=source_raw_id,
            edge_type=GraphEdgeType.EXTRACTED_FROM,
            account_id=ctx.account_id,
            weight=1.0,
        )
        self._create_node_with_edges(node, [edge], ctx)

        return node

    def get_extracted_node(
        self, node_id: str, ctx: RequestContext
    ) -> ExtractedNode | None:
        """Get an extracted node by ID.

        Args:
            node_id: UUID of the extracted node
            ctx: RequestContext for access control

        Returns:
            ExtractedNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, ExtractedNode):
            return node
        return None

    def list_extracted_nodes(
        self,
        extractor_type: str | None,
        ctx: RequestContext,
        limit: int = 4096,
        offset: int = 0,
        parent_audio_node_ids: list[str] | None = None,
    ) -> list[ExtractedNode]:
        """List extracted nodes optionally filtered by extractor type
        and / or by the audio raw node they descend from.

        Args:
            extractor_type: Filter by ``extractor_type``. Pushed to the
                store layer's WHERE clause so the LIMIT budget isn't
                shared with face / episodic_memory / etc. — without
                this push-down, a 4096-limit query against a 100k+
                graph silently drops the oldest extractor_type's
                children.
            ctx: RequestContext for access control
            limit: Max results (default 4096; was 100. Bumped so a long
                video with many voice_asr / episodic_memory ExtractedNodes
                doesn't silently truncate the lookup. The underlying
                store sorts by updated_at DESC, so callers that need a
                true "latest" view should pass a smaller limit
                explicitly. ``None`` is honored by the stores as
                "no LIMIT clause" — use only after pre-narrowing via
                ``extractor_type`` or ``parent_audio_node_ids``.
            offset: Skip count
            parent_audio_node_ids: Optional list of audio raw node ids.
                The store pushes this into
                ``extracted_content->>'source_audio_node_id' = ANY(...)``
                so a video-scoped ASR lookup doesn't have to share its
                LIMIT budget with the rest of the voice_asr corpus.
                This is the path that fixes the 100-video M3-Bench
                transcript-drop bug: previously the function returned
                only the newest 4096 voice_asr nodes (most videos had
                no children among them), so 20/23 videos received
                ``speech_transcript: []``.

        Returns:
            List of ExtractedNodes
        """
        nodes = self._store.list_nodes(
            GraphNodeType.EXTRACTED,
            ctx,
            limit,
            offset,
            extractor_type=extractor_type,
            parent_audio_node_ids=parent_audio_node_ids,
        )
        return [n for n in nodes if isinstance(n, ExtractedNode)]

    def update_extracted_node(
        self, node: ExtractedNode, ctx: RequestContext
    ) -> ExtractedNode:
        """Persist a modified :class:`ExtractedNode` back to the store.

        Use this when the caller already holds a live ExtractedNode
        (e.g. from ``get_extracted_node``) and has mutated it in place —
        such as calling :meth:`ExtractedNode.set_embedding` after
        computing an embedding out-of-band. The node is re-serialised
        and the store is asked to overwrite the existing entry; the
        caller receives the persisted node back for chaining.

        Raises ``ValueError`` if the node id is missing from the
        store (callers should use ``mark_extraction_completed`` /
        ``mark_extraction_failed`` for status transitions; this method
        is for arbitrary field updates).
        """
        if not isinstance(node, ExtractedNode):
            raise ValueError(
                f"update_extracted_node requires an ExtractedNode, "
                f"got {type(node).__name__}"
            )
        existing = self._store.get_node(node.node_id, ctx)
        if existing is None:
            raise ValueError(f"Extracted node not found: {node.node_id}")
        return self._store.update_node(node.serialize(), ctx)

    def mark_extraction_completed(
        self,
        node_id: str,
        extracted_content: dict,
        confidence: float,
        ctx: RequestContext,
    ) -> ExtractedNode:
        """Mark an extraction as completed with results.

        Args:
            node_id: ID of the extracted node
            extracted_content: The extraction results
            confidence: Confidence score (0-1)
            ctx: RequestContext for access control

        Returns:
            The updated ExtractedNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, ExtractedNode):
            raise ValueError(f"Extracted node not found: {node_id}")

        # In-memory stores return live objects. A failed write must not expose
        # a completed status/cache receipt that never successfully persisted.
        node = replace(node)
        node.mark_completed(extracted_content, confidence)
        return self._store.update_node(node.serialize(), ctx)

    def mark_extraction_failed(
        self, node_id: str, error: str, ctx: RequestContext
    ) -> ExtractedNode:
        """Mark an extraction as failed.

        Args:
            node_id: ID of the extracted node
            error: Error message
            ctx: RequestContext for access control

        Returns:
            The updated ExtractedNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, ExtractedNode):
            raise ValueError(f"Extracted node not found: {node_id}")

        node.mark_failed(error)
        return self._store.update_node(node.serialize(), ctx)

    # ====================================================================
    # Memory Node Operations
    # ====================================================================

    def create_memory_node(
        self,
        summary: str,
        ctx: RequestContext,
        content: str = "",
        supporting_node_ids: list[str] | None = None,
        topic: str = "",
        tags: list[str] | None = None,
        parent_memory_id: str | None = None,
        node_id: str | None = None,
    ) -> MemoryNode:
        """Create a memory node with supporting nodes.

        Automatically creates supports edges for each supporting node.

        Args:
            summary: Memory summary
            ctx: RequestContext for access control
            content: Optional full memory content
            supporting_node_ids: IDs of supporting nodes
            topic: Optional topic
            tags: Optional tags
            parent_memory_id: ID of parent memory (for evolution)
            node_id: Optional pre-generated node ID

        Returns:
            The created MemoryNode
        """
        supporting_node_ids = supporting_node_ids or []
        node = create_memory_node(
            summary=summary,
            account_id=ctx.account_id,
            content=content,
            supporting_node_ids=supporting_node_ids,
            topic=topic,
            tags=tags,
            node_id=node_id,
            parent_memory_id=parent_memory_id,
        )
        edges = []

        # Create supports edges
        for supporting_id in supporting_node_ids:
            edge = create_graph_edge(
                source_id=supporting_id,
                target_id=node.node_id,
                edge_type=GraphEdgeType.SUPPORTS,
                account_id=ctx.account_id,
                weight=1.0,
            )
            edges.append(edge)

        # Create evolves edge if parent exists
        if parent_memory_id:
            edge = create_graph_edge(
                source_id=parent_memory_id,
                target_id=node.node_id,
                edge_type=GraphEdgeType.EVOLVES,
                account_id=ctx.account_id,
                weight=1.0,
            )
            edges.append(edge)

        self._create_node_with_edges(node, edges, ctx)
        return node

    def get_memory_node(
        self, node_id: str, ctx: RequestContext
    ) -> MemoryNode | None:
        """Get a memory node by ID.

        Args:
            node_id: UUID of the memory node
            ctx: RequestContext for access control

        Returns:
            MemoryNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, MemoryNode):
            return node
        return None

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memory nodes.

        Args:
            query: Search query
            ctx: RequestContext for access control
            min_confidence: Minimum confidence threshold
            topic: Filter by topic
            tags: Filter by tags (OR logic)
            top_k: Max results

        Returns:
            List of matching MemoryNodes sorted by confidence
        """
        return self._store.search_memories(
            query, ctx, min_confidence, topic, tags, top_k
        )

    def update_memory(
        self,
        node_id: str,
        ctx: RequestContext,
        summary: str | None = None,
        content: str | None = None,
        topic: str | None = None,
        tags: list[str] | None = None,
        supporting_node_ids: list[str] | None = None,
    ) -> MemoryNode:
        """Update a memory node.

        Args:
            node_id: ID of the memory node
            ctx: RequestContext for access control
            summary: New summary
            content: New content
            topic: New topic
            tags: New tags
            supporting_node_ids: Replace the source ID list when supplied.
                This updates node data only; callers that replace provenance
                must reconcile SUPPORTS/EVOLVES edges separately.

        Returns:
            The updated MemoryNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, MemoryNode):
            raise ValueError(f"Memory node not found: {node_id}")

        # MemoryGraphStore returns a live object. Do not mutate stored data
        # before update_node succeeds (SQL returns a detached object).
        node = replace(node)
        if summary is not None:
            node.summary = summary
        if content is not None:
            node.content = content
        if topic is not None:
            node.topic = topic
        if tags is not None:
            node.tags = tags
        if supporting_node_ids is not None:
            node.supporting_node_ids = list(supporting_node_ids)

        node.touch()
        return self._store.update_node(node.serialize(), ctx)

    def add_supporting_node(
        self, memory_id: str, supporting_id: str, ctx: RequestContext
    ) -> MemoryNode:
        """Add a supporting node to a memory.

        Args:
            memory_id: ID of the memory node
            supporting_id: ID of the supporting node
            ctx: RequestContext for access control

        Returns:
            The updated MemoryNode
        """
        memory = self._store.get_node(memory_id, ctx)
        if not memory or not isinstance(memory, MemoryNode):
            raise ValueError(f"Memory node not found: {memory_id}")

        memory.add_support(supporting_id)

        # Create the supports edge
        edge = create_graph_edge(
            source_id=supporting_id,
            target_id=memory_id,
            edge_type=GraphEdgeType.SUPPORTS,
            account_id=ctx.account_id,
            weight=1.0,
        )
        self._store.create_edge(edge, ctx)

        return self._store.update_node(memory.serialize(), ctx)

    # ====================================================================
    # Graph Traversal
    # ====================================================================

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges.

        Args:
            node_id: UUID of the center node
            ctx: RequestContext for access control
            edge_types: Filter by edge types
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of (neighbor_node, edge, direction) tuples
        """
        return self._store.get_neighbors(node_id, ctx, edge_types, direction)

    def traverse(
        self,
        start_node_id: str,
        depth: int,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        """BFS traversal from a starting node.

        Args:
            start_node_id: UUID of the starting node
            depth: Maximum traversal depth
            ctx: RequestContext for access control
            edge_types: Filter by edge types

        Returns:
            SubgraphResult with nodes and edges within depth limit
        """
        return self._store.traverse(start_node_id, depth, ctx, edge_types)

    def get_supporting_chain(
        self, memory_id: str, ctx: RequestContext
    ) -> SupportingChain:
        """Get the evidence chain for a memory node.

        Returns the memory and all supporting raw/extracted nodes
        with their relationships.

        Args:
            memory_id: UUID of the memory node
            ctx: RequestContext for access control

        Returns:
            SupportingChain with memory, supporting nodes, and edges
        """
        return self._store.get_supporting_chain(memory_id, ctx)

    # ====================================================================
    # Edge Operations
    # ====================================================================

    def list_identity_pairs(
        self, source_ids: list[str], ctx: RequestContext,
    ) -> list[tuple[str, str]]:
        return self._store.list_identity_pairs(source_ids, ctx)

    def ensure_identity_edges(
        self, pairs: list[tuple[str, str, float]], ctx: RequestContext,
    ) -> int:
        edges = [
            create_graph_edge(
                source_id=source, target_id=target,
                edge_type=GraphEdgeType.SAME_IDENTITY,
                account_id=ctx.account_id, properties={"score": score},
            )
            for source, target, score in pairs
        ]
        return self._store.ensure_identity_edges(edges, ctx)

    def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: GraphEdgeType,
        ctx: RequestContext,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> GraphEdge:
        """Create an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            ctx: RequestContext for access control
            weight: Edge weight
            properties: Additional edge properties

        Returns:
            The created GraphEdge
        """
        edge = create_graph_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            account_id=ctx.account_id,
            weight=weight,
            properties=properties,
        )
        return self._store.create_edge(edge, ctx)

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge.

        Args:
            edge_id: UUID of the edge to delete
            ctx: RequestContext for access control
        """
        self._store.delete_edge(edge_id, ctx)

    # ====================================================================
    # PageRank
    # ====================================================================

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores.

        Args:
            ctx: RequestContext for access control
            node_ids: Specific nodes to compute for (None = all)
            damping: Damping factor

        Returns:
            Dict mapping node_id to PageRank score
        """
        return self._store.compute_pagerank(ctx, node_ids, damping)

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a node.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            PageRank score
        """
        return self._store.get_node_pagerank(node_id, ctx)

    # ====================================================================
    # Statistics
    # ====================================================================

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph.

        Args:
            ctx: RequestContext for access control

        Returns:
            GraphStats with node and edge counts, top memories
        """
        return self._store.get_graph_stats(ctx)

    # ====================================================================
    # Node Deletion
    # ====================================================================

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all connected edges.

        Args:
            node_id: UUID of the node to delete
            ctx: RequestContext for access control
        """
        self._store.delete_node(node_id, ctx)

    # ====================================================================
    # Control-Loop Primitives
    # ====================================================================

    def search_by_embedding(
        self,
        query: "str | list[float]",
        ctx: RequestContext,
        *,
        embedder: "Embedder | None" = None,
        scope: ClipScope | None = None,
        node_types: tuple[str, ...] = ("extracted",),
        extractor_types: tuple[str, ...] | None = None,
        top_k: int = 10,
    ) -> list[EvidenceHit]:
        """Cosine-similarity search over node embeddings.

        Wraps :meth:`GraphStore.search_by_embedding` with two
        service-level conveniences:

        - Accepts ``query`` as either a ``str`` (in which case
          ``embedder`` is required) or a precomputed ``list[float]``.
          The service embeds the text and delegates the vector.
        - Returns :class:`EvidenceHit` wrappers instead of raw
          ``(node_id, score)`` pairs, with ``_video_id`` /
          ``_clip_id`` provenance lifted from ``extracted_content`` for
          caller convenience.

        See :meth:`GraphStore.search_by_embedding` for the underlying
        filter semantics (tenant isolation, best-effort skip on bad
        rows, sort order). ``scope`` is translated into ``video_id``
        and ``before_clip`` kwargs on this call — it does not persist
        in the store.

        Args:
            query: ``str`` (requires ``embedder``) or ``list[float]``.
            ctx: Tenant context.
            embedder: Required when ``query`` is a ``str``.
            scope: Optional :class:`ClipScope` to filter by video / clip.
            node_types: Whitelist of node-type strings; only
                ``"extracted"`` is supported today.
            extractor_types: Optional whitelist of ``extractor_type``.
            top_k: Maximum results.

        Returns:
            ``list[EvidenceHit]`` ordered by score desc.

        Raises:
            ValueError: If ``query`` is a ``str`` but no ``embedder``
                was supplied.
            TypeError: If ``query`` is neither a ``str`` nor a
                list-like of floats.
        """
        # Resolve query → vector.
        if isinstance(query, str):
            if embedder is None:
                raise ValueError(
                    "embedder is required when query is a string; "
                    "pass an Embedder implementation or supply a "
                    "precomputed list[float] vector instead"
                )
            vectors = embedder.embed_texts([query])
            if not vectors or not vectors[0]:
                return []
            query_vector: list[float] = list(vectors[0])
        elif isinstance(query, (list, tuple)):
            query_vector = [float(x) for x in query]
        else:
            raise TypeError(
                f"query must be str or list[float], got {type(query).__name__}"
            )

        # Translate scope into filter args (used by BOTH paths below).
        video_id = scope.video_id if scope is not None else None
        before_clip = scope.before_clip if scope is not None else None
        before_sec = scope.before_sec if scope is not None else None

        # --- VectorIndex fast path -----------------------------------------
        # When a VectorIndex has been injected (M3's OpenGaussVectorIndex
        # is the canonical example), route through it. The index returns
        # SeedHit objects that carry the originating ``extracted_node_id``
        # in their metadata (the M3 driver stamps it on upsert); we use
        # that to re-fetch the ExtractedNode from the store and reuse
        # the same provenance-lift logic as the legacy path. Stale
        # index entries that no longer point at a live graph node are
        # silently dropped — defense in depth against partial writes
        # during indexing.
        if self._vector_index is not None:
            vi_filters: dict = {"account_id": ctx.account_id}
            if video_id is not None:
                vi_filters["video_id"] = video_id
            if before_clip is not None:
                vi_filters["before_clip"] = before_clip
            if before_sec is not None:
                vi_filters["before_sec"] = before_sec
            if extractor_types is not None:
                # The VectorIndex filter accepts a list (ANY semantics).
                vi_filters["extractor_type"] = list(extractor_types)
            seed_hits = self._vector_index.search_by_vector(
                query_vector=query_vector,
                filters=vi_filters,
                top_k=top_k,
            )
            hits: list[EvidenceHit] = []
            for seed in seed_hits:
                ext_id = seed.metadata.get("extracted_node_id") if seed.metadata else None
                if not ext_id:
                    # Index entry doesn't carry the forward pointer —
                    # try the SeedHit.uri as a last-resort hint, then
                    # give up on this hit. Should not happen with M3
                    # records written by m3.index_episodic but a future
                    # writer might omit it.
                    continue
                node = self._store.get_node(ext_id, ctx)
                if not isinstance(node, ExtractedNode):
                    # Stale index row — drop quietly so a partial
                    # write during indexing doesn't surface as a
                    # visible failure.
                    logger.debug(
                        "VectorIndex hit %s references missing node %s; skipping",
                        seed.uri, ext_id,
                    )
                    continue
                content = node.extracted_content or {}
                if not within_scope(content, video_id, before_clip, before_sec):
                    continue
                prov_video = content.get("source_video_id") or content.get("_video_id")
                prov_clip = content.get("source_clip_id")
                if prov_clip is None:
                    prov_clip = content.get("_clip_id")
                if isinstance(prov_clip, str) and prov_clip.isdigit():
                    prov_clip = int(prov_clip)
                hits.append(
                    EvidenceHit(
                        node=node, score=seed.score,
                        video_id=prov_video, clip_id=prov_clip,
                    )
                )
            return hits

        # --- Legacy store path ----------------------------------------------
        # Backward compat for callers that haven't migrated to a
        # VectorIndex yet (tests, non-M3 consumers). The store still
        # reads ``extracted_content["embedding"]`` which is the slot
        # M3 used to populate; non-M3 writers can keep using this
        # path unchanged.
        raw_hits = self._store.search_by_embedding(
            query_vector=query_vector,
            ctx=ctx,
            node_types=node_types,
            extractor_types=extractor_types,
            video_id=video_id,
            before_clip=before_clip,
            **({"before_sec": before_sec} if before_sec is not None else {}),
            top_k=top_k,
        )

        hits = []
        for node_id, score in raw_hits:
            node = self._store.get_node(node_id, ctx)
            if not isinstance(node, ExtractedNode):
                # Defense in depth: the store's `search_by_embedding`
                # should never return non-extracted nodes when
                # `node_types=("extracted",)`, but if a custom store
                # implementation drifts we don't want a hard error —
                # just drop the hit.
                continue
            content = node.extracted_content or {}
            if not within_scope(content, video_id, before_clip, before_sec):
                continue
            # Provenance: prefer source_video_id / source_clip_id (the
            # keys current M3 writers stamp); fall back to legacy
            # _video_id / _clip_id. Explicit None-check for clip_id —
            # 0 is a valid clip ordinal. Same pattern as
            # m3/semantic_memory_consolidator.py.
            prov_video = content.get("source_video_id") or content.get("_video_id")
            clip_id = content.get("source_clip_id")
            if clip_id is None:
                clip_id = content.get("_clip_id")
            if isinstance(clip_id, str) and clip_id.isdigit():
                clip_id = int(clip_id)  # EvidenceHit.clip_id is int | None
            hits.append(
                EvidenceHit(node=node, score=score, video_id=prov_video, clip_id=clip_id)
            )
        return hits

    def search_by_transcript(
        self, query: str, ctx: RequestContext, *, scope: ClipScope,
        extractor_types: tuple[str, ...] | None = None, top_k: int = 10,
    ) -> list[EvidenceHit]:
        """Independent, opt-in ASR recall with live graph provenance checks.

        No graph-store scan fallback: an index with transcript capability is
        required. Deleted/stale, foreign-account and out-of-scope nodes are
        excluded, and lexical matches are rechecked against the current ASR.
        """
        filters = {"account_id": ctx.account_id, "video_id": scope.video_id}
        if scope.before_clip is not None:
            filters["before_clip"] = scope.before_clip
        if scope.before_sec is not None:
            filters["before_sec"] = scope.before_sec
        if extractor_types is not None:
            filters["extractor_type"] = list(extractor_types)
        validate_transcript_request(query, filters, top_k)
        search = getattr(self._vector_index, "search_by_transcript", None)
        if not callable(search):
            raise NotImplementedError("Transcript recall requires a transcript-capable index (memory or SQL)")
        hits = []
        for seed in search(query=query, filters=filters, top_k=top_k):
            ext_id = (seed.metadata or {}).get("extracted_node_id")
            if not ext_id:
                continue
            node = self._store.get_node(ext_id, ctx)
            if not isinstance(node, ExtractedNode) or node.account_id != ctx.account_id:
                continue
            if extractor_types is not None and node.extractor_type not in extractor_types:
                continue
            content = node.extracted_content or {}
            if not within_scope(content, scope.video_id, scope.before_clip, scope.before_sec):
                continue
            score = transcript_score(query, content)
            if score <= 0:
                continue
            clip = content.get("source_clip_id")
            if clip is None:
                clip = content.get("_clip_id")
            if isinstance(clip, str) and clip.isdigit():
                clip = int(clip)
            hits.append(EvidenceHit(
                node=node, score=score,
                video_id=content.get("source_video_id") or content.get("_video_id"),
                clip_id=clip,
            ))
        hits.sort(key=lambda hit: (-hit.score, hit.node.node_id))
        return hits[:top_k]

    def _get_temporal_nodes(
        self, ctx: RequestContext, *, scope: ClipScope, timestamp: float,
        extractor_types: tuple[str, ...] | None = None,
    ) -> list[ExtractedNode]:
        """Resolve indexed source intervals, rechecking live tenant/provenance."""
        from core.clip_lookup import clip_ordinal, validate_clip_lookup
        timestamp = seconds(timestamp)
        filters = {"account_id": ctx.account_id, "video_id": scope.video_id}
        validate_clip_lookup(filters, [])
        if scope.before_clip is not None:
            filters["before_clip"] = scope.before_clip
        if extractor_types is not None:
            filters["extractor_type"] = list(extractor_types)
        lookup = getattr(self._vector_index, "get_records_at_time", None)
        if not callable(lookup):
            raise NotImplementedError("Temporal QA requires an interval-capable index")
        nodes = {}
        for record in lookup(filters=filters, timestamp=timestamp):
            ext_id = record.metadata.get("extracted_node_id")
            if not ext_id:
                continue
            node = self._store.get_node(ext_id, ctx)
            if not isinstance(node, ExtractedNode) or node.account_id != ctx.account_id:
                continue
            if extractor_types is not None and node.extractor_type not in extractor_types:
                continue
            content = node.extracted_content or {}
            if (not within_scope(content, scope.video_id, scope.before_clip)
                    or not contains_time(content, timestamp)
                    or clip_ordinal(content.get("source_clip_id", content.get("_clip_id"))) is None):
                continue
            nodes[node.node_id] = node
        return sorted(nodes.values(), key=lambda n: (current_clip_key(n.extracted_content), n.node_id))

    def get_clip_at_time(
        self, ctx: RequestContext, *, video_id: str, timestamp: float,
        extractor_types: tuple[str, ...] = ("episodic_memory_generator",),
    ) -> dict | None:
        """Return live provenance only, for resolving a QA policy once."""
        nodes = self._get_temporal_nodes(
            ctx, scope=ClipScope(video_id), timestamp=timestamp, extractor_types=extractor_types,
        )
        if not nodes:
            return None
        content = nodes[0].extracted_content
        return {
            **temporal_metadata(content),
            "clip_id": int(content.get("source_clip_id", content.get("_clip_id"))),
        }

    def get_boundary_transcript(
        self, ctx: RequestContext, *, scope: ClipScope,
        extractor_types: tuple[str, ...] | None = None,
    ) -> list[EvidenceHit]:
        """Read the timestamp-containing clip without embedding or future text.

        The exact lookup retains tenant/video/clip/type filters. Time is
        applied to the live ASR segments, not to the entire index record.
        No full-clip vector or lexical score participates in this branch.
        """
        if scope.before_sec is None:
            return []
        nodes = self._get_temporal_nodes(
            ctx, scope=scope, timestamp=scope.before_sec, extractor_types=extractor_types,
        )
        hits = {}
        for node in nodes[:1]:
            content = node.extracted_content or {}
            projected = boundary_transcript(content, scope.before_sec)
            if projected:
                hits[node.node_id] = EvidenceHit(
                    node=replace(node, extracted_content=projected), score=0.,
                    video_id=scope.video_id, clip_id=int(projected["source_clip_id"]),
                )
        # One exact boundary clip, one independently budgeted evidence item.
        return [hits[key] for key in sorted(hits)[:1]]

    def get_evidence_by_clips(
        self, clip_ids: list[int], ctx: RequestContext, *, scope: ClipScope,
        extractor_types: tuple[str, ...] | None = None,
    ) -> list[EvidenceHit]:
        """Exact indexed context read, followed by live graph scope checks."""
        from core.clip_lookup import clip_ordinal, validate_clip_lookup
        filters = {"account_id": ctx.account_id, "video_id": scope.video_id}
        clips = set(validate_clip_lookup(filters, clip_ids))
        if scope.before_clip is not None:
            filters["before_clip"] = scope.before_clip
        if scope.before_sec is not None:
            filters["before_sec"] = scope.before_sec
        if extractor_types is not None:
            filters["extractor_type"] = list(extractor_types)
        lookup = getattr(self._vector_index, "get_records_by_clips", None)
        if not callable(lookup):
            raise NotImplementedError("Exact context requires a clip-capable index (memory or SQL)")
        hits = {}
        for record in lookup(filters=filters, clip_ids=sorted(clips)):
            ext_id = record.metadata.get("extracted_node_id")
            if not ext_id:
                continue
            node = self._store.get_node(ext_id, ctx)
            if not isinstance(node, ExtractedNode) or node.account_id != ctx.account_id:
                continue
            if extractor_types is not None and node.extractor_type not in extractor_types:
                continue
            content = node.extracted_content or {}
            clip = clip_ordinal(content.get("source_clip_id", content.get("_clip_id")))
            if clip not in clips or not within_scope(content, scope.video_id, scope.before_clip, scope.before_sec):
                continue
            hits[node.node_id] = EvidenceHit(node=node, score=0., video_id=scope.video_id, clip_id=clip)
        return sorted(hits.values(), key=lambda hit: (hit.clip_id, hit.node.node_id))

    def get_clips_summary(
        self,
        video_id: str,
        ctx: RequestContext,
    ) -> list[dict]:
        """Return per-clip extracted-node summary for ``video_id``.

        Walks every ExtractedNode in the account, filters by
        ``_video_id`` and ``_clip_id`` provenance, and returns a
        ``[{clip_id, extracted_count, episodic_id}, ...]`` list
        sorted by ``clip_id`` ascending. ``episodic_id`` is the
        node_id of the first ExtractedNode with
        ``extractor_type == "episodic_memory_generator"`` for that
        clip, or ``None`` if no episodic node exists yet.

        Used by the future ``scripts/m3_eval.py`` harness for
        per-clip coverage reporting.

        Args:
            video_id: The M3 video_id to summarise.
            ctx: Tenant context.

        Returns:
            List of summary dicts sorted by ``clip_id``.
        """
        # 10k is a generous limit for the M3-Bench corpus (~3k nodes);
        # pagination is a v2 concern.
        all_nodes = self._store.list_nodes(
            GraphNodeType.EXTRACTED, ctx, limit=10_000, offset=0
        )
        by_clip: dict[int, list[ExtractedNode]] = {}
        for node in all_nodes:
            if not isinstance(node, ExtractedNode):
                continue
            content = node.extracted_content or {}
            # Same fallback as search_by_embedding: current M3 writers
            # stamp source_video_id; older writers stamped _video_id.
            if content.get("source_video_id") != video_id and content.get("_video_id") != video_id:
                continue
            clip_id_value = content.get("source_clip_id")
            if clip_id_value is None:
                clip_id_value = content.get("_clip_id")
            if clip_id_value is None:
                continue
            try:
                clip_key = int(clip_id_value)
            except (TypeError, ValueError):
                continue
            by_clip.setdefault(clip_key, []).append(node)

        out: list[dict] = []
        for clip_key in sorted(by_clip.keys()):
            nodes = by_clip[clip_key]
            episodic = next(
                (
                    n
                    for n in nodes
                    if n.extractor_type == "episodic_memory_generator"
                ),
                None,
            )
            out.append(
                {
                    "clip_id": clip_key,
                    "extracted_count": len(nodes),
                    "episodic_id": episodic.node_id if episodic else None,
                }
            )
        return out
