"""Core protocol interfaces for ContextEngine.

Phase 0 frozen — these protocols define the contracts between layers.
All protocols use typing.Protocol for structural subtyping.
"""

from datetime import datetime
from typing import Any, Protocol

from typing_extensions import runtime_checkable

from core.graph_enums import (
    GraphEdgeType,
    GraphNodeType,
    RawDataType,
)
from core.graph_models import (
    GraphEdge,
    GraphNode,
    GraphStats,
    MemoryNode,
    RawNode,
    SubgraphResult,
    SupportingChain,
)
from core.models import (
    CandidateMemory,
    ContextNode,
    IndexRecord,
    OutboxEvent,
    RelationEdge,
    RequestContext,
    SeedHit,
    WritePlan,
)


@runtime_checkable
class ContextFS(Protocol):
    """File system abstraction for context persistence.

    All methods enforce RequestContext-based access control via _is_accessible.
    Implementations MUST validate account_id prefix in URI matches RequestContext.

    AGFS implementation writes nodes to:
    /accounts/{account}/users/{user}/memories/{category}/{slug}/
    With files: content.md, .relations.json, .abstract.md, .overview.md, .meta.json

    See CLAUDE.md §2 AGFS Directory Spec for write order semantics.
    """

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node to AGFS with atomic 4-step write order.

        Write order (strictly enforced, Repair Job depends on this):
        ① content.md (largest file, write first)
        ② .relations.json
        ③ .abstract.md, .overview.md (parallel)
        ④ .meta.json (commit point: status=ACTIVE)

        Step ④ completion triggers OutboxEvent registration.
        """
        ...

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from AGFS.

        Raises AccessDeniedError if URI account prefix != ctx.account_id.
        Raises NodeNotFoundError if node does not exist or status != ACTIVE.
        """
        ...

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node from AGFS.

        Raises AccessDeniedError if URI account prefix != ctx.account_id.
        """
        ...

    def archive_node(self, uri: str, ctx: RequestContext) -> None:
        """Archive a node (soft delete). Marks status as ARCHIVED.

        Args:
            uri: URI of the node to archive
            ctx: RequestContext for this operation

        Raises:
            AccessDeniedError: If URI account prefix != ctx.account_id
            NodeNotFoundError: If node does not exist
        """
        ...

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        """Move/rename a node in AGFS.

        Raises AccessDeniedError if either URI account prefix != ctx.account_id.
        """
        ...

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List child URIs under a given URI.

        Returns only URIs where the caller has access.
        """
        ...

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if a node exists and is ACTIVE.

        Returns True only if .meta.json exists AND status == ACTIVE.
        PENDING nodes are not visible to upper layers.
        """
        ...


@runtime_checkable
class ArchiveLocator(Protocol):
    """Optional canonical archive lookup scoped to the caller's account and owner."""

    def find_archive_uri(self, archive_id: str, ctx: RequestContext) -> str | None: ...


class RelationStore(Protocol):
    """Graph relation storage.

    v1 implementation: Read/write .relations.json in AGFS node directories.
    Phase 2+ implementation: May migrate to graph database, interface unchanged.

    Edges are directed and weighted.
    """

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        """Get all outgoing edges from a node.

        Returns empty list if node has no relations or does not exist.
        """
        ...

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        """Add or update relation edges.

        Idempotent: same edge (from_uri, to_uri, relation_type) is updated.
        """
        ...

    def get_one_hop(self, uri: str, ctx: RequestContext, limit: int = 3) -> list[RelationEdge]:
        """Get top-N outgoing edges sorted by weight (descending).

        Returns the highest-weighted edges from a node, up to `limit`.
        Returns empty list if node has no relations or doesn't exist.
        """
        ...


class CandidateExtractor(Protocol):
    """Extract candidate memories from message history.

    Four implementations (one per memory category):
    - ProfileExtractor: User profile and state
    - PreferenceExtractor: User preferences and opinions
    - EntityEventExtractor: Entities and events
    - ProcedureExtractor: Skills and procedures (agent scope)
    """

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract candidate memories from a conversation.

        Messages format: {"role": "user|assistant", "content": "..."}
        Returns list of candidates with confidence scores.
        """
        ...


class MergePolicy(Protocol):
    """Determine write action for a candidate memory.

    Four implementations (one per memory type):
    - ProfilePolicy: Always merge, fixed URI
    - AggregateTopicPolicy: Merge by slug (preference, entity, pattern)
    - AppendOnlyPolicy: Always create, unique event_id (event, case)
    - SkillToolPolicy: Fixed URI, accumulate best practices (skill)
    """

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for a candidate.

        Returns WritePlan with action: create|merge|append|skip
        """
        ...


class TranscriptIndex(Protocol):
    """Optional transcript-only recall capability, independent of vectors.

    Implementations must require account_id and video_id and apply all scope
    filters before limiting candidates. Unsupported backends must not silently
    substitute vector reranking or search descriptions.
    """

    def search_by_transcript(
        self, query: str, filters: dict[str, Any], top_k: int,
    ) -> list[SeedHit]:
        ...


class ClipRecordIndex(Protocol):
    """Optional scoped batch read of complete records at exact clip ordinals."""

    def get_records_by_clips(
        self, filters: dict[str, Any], clip_ids: list[int],
    ) -> list[IndexRecord]:
        ...


class TemporalRecordIndex(Protocol):
    """Optional tenant/video-scoped interval lookup, independent of ranking."""

    def get_records_at_time(
        self, filters: dict[str, Any], timestamp: float,
    ) -> list[IndexRecord]:
        ...


class VectorIndex(Protocol):
    """Vector similarity search interface.

    Implementation layer handles account_id filtering transparently.
    Upper layers (retrieval) are unaware of tenant isolation.

    Upsert is idempotent: same IndexRecord.id overwrites safely.
    """

    def get_embeddings(
        self,
        uris: list[str],
        filters: dict[str, Any],
        level: int = 0,
        *, expected_texts: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        """Read scoped vectors, matching expected source text in the same index read."""
        ...

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Idempotent: duplicate IndexRecord.id overwrites.
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index.

        No-op for non-existent ids.
        """
        ...

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Low-level vector search returning full SeedHit with level info.

        Used by SeedRetriever for L0/L1/L2 mixed recall.
        Default implementation raises NotImplementedError.
        """
        ...

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Search immediate children of *parent_uri*.

        Used by HierarchicalSearcher for recursive expansion.
        Must be implemented — hierarchical expansion is mandatory.
        """
        ...

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count index records, optionally constrained by tenant filters."""
        ...

    def delete_account_data(self, account_id: str) -> int:
        """Delete all index records for an account.

        Returns count of deleted records.
        Used by NamespaceManager.delete_account().
        """
        ...

    def get_record(self, record_id: str) -> "IndexRecord | None":
        """Fetch one :class:`IndexRecord` by id (round-trip).

        Returns ``None`` when no row matches. ``search_by_vector``
        only returns a 200-char abstract to keep the SeedHit payload
        small; callers that need the full ``text`` field (M3 QA's
        :class:`EpisodicSearcher`) use this method to recover it.

        Default implementation raises :class:`NotImplementedError`;
        backends with cheap by-id lookup (InMemoryVectorIndex,
        OpenGaussVectorIndex, ChromaVectorIndex) override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_record; "
            "either override it or use a backend that does."
        )

    def delete_by_owner_space(
        self, account_id: str, owner_space: str
    ) -> int:
        """Delete all records matching account_id + owner_space.

        owner_space must be colon format: "user:{id}" or "agent:{id}".
        Returns count of deleted records.
        Used by NamespaceManager.delete_user() / delete_agent().
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text embedding interface.

    Converts text to fixed-length vectors for vector indexing.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns list of vectors, same length as input.
        All vectors have same dimension (implementation-specific).
        """
        ...


@runtime_checkable
class LLM(Protocol):
    """Large language model interface for JSON completion.

    Used by Extractor and Planner for structured output.
    """

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete a prompt and return JSON matching schema.

        Schema format: {"type": "object", "properties": {...}}
        Returns parsed JSON dict.
        """
        ...

    def complete_multimodal(self, messages: list[dict], schema: dict) -> dict:
        """Complete a multimodal message list and return JSON matching schema.

        Each message dict has ``role`` (``system`` | ``user`` | ``assistant``)
        and ``content``. ``content`` may be:

        - ``str`` — plain text
        - ``list[dict]`` — multimodal content parts. Each part has a
          ``type`` discriminator and a type-specific payload. Recognised
          types follow OpenAI's Chat Completions API:
            * ``{"type": "text", "text": str}``
            * ``{"type": "image_url", "image_url": {"url": str,
              "detail": "low"|"high"|"auto"}}`` — ``url`` may be a
              ``data:image/...;base64,...`` URI for inline images.
            * ``{"type": "input_audio", "input_audio": {"data": str,
              "format": "wav"|"mp3"}}}`` — ``data`` is base64 audio.

        Schema format: ``{"type": "object", "properties": {...}}``.
        Returns parsed JSON dict.

        Implementations should accept arbitrary-length messages and
        multimodal content; providers without multimodal capability
        should raise ``NotImplementedError`` rather than silently
        dropping content.
        """
        ...

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> list[dict]:
        """Run LLM with tool use and return all tool invocations.

        Args:
            prompt: The input prompt
            tools: List of tool definitions with name, description, input_schema
            tool_choice: "auto" (let LLM decide) or specific tool name

        Returns:
            List of tool call results: [{"tool": str, "input": dict}, ...]
            Empty list if LLM decides no tools should be called.

        Tool definition format:
            {
                "name": str,
                "description": str,
                "input_schema": dict (JSON Schema)
            }
        """
        ...

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        """Run LLM with full message history and optional tool use.

        Unlike complete_with_tools (flat prompt), this accepts a full
        message list for multi-turn ReAct conversations.

        Args:
            messages: List of message dicts with "role" and "content".
            tools: Optional tool definitions. None = no tools available.
            tool_choice: "auto", "required", "none", or specific tool name.

        Returns:
            Tuple of (tool_calls, content):
            - tool_calls: list of {"tool": str, "input": dict}
            - content: raw text content (empty string if only tool calls)
        """
        ...


class OutboxStore(Protocol):
    """Outbox event storage for async index synchronization.

    Persists OutboxEvents to storage for processing by OutboxWorker.
    Events are durable and survive process restarts.

    Implementations must support:
    - Registering events after successful node writes
    - Listing pending events for processing
    - Marking events as done or failed
    - Moving failed events to DLQ after max retries
    """

    def register_write(self, node: ContextNode, ctx: RequestContext) -> OutboxEvent:
        """Register a node write for index synchronization.

        Called after successful ContextFS.write_node().

        Args:
            node: ContextNode that was written
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        ...

    def register_delete(self, uri: str, ctx: RequestContext) -> OutboxEvent:
        """Register a node deletion for index synchronization.

        Args:
            uri: URI of deleted node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        ...

    def register_archive(self, uri: str, ctx: RequestContext) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Called after successful archive (soft delete).  The worker
        removes vector index records so the node no longer appears
        in search results.

        Args:
            uri: URI of archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        ...

    def register_directory(
        self, node: ContextNode, ctx: RequestContext
    ) -> OutboxEvent:
        """Register UPSERT_DIRECTORY event for node's parent directory.

        Collects sibling abstracts (max 20) and writes OutboxEvent
        with payload self-contained — indexer does NOT access AGFS.

        Args:
            node: ContextNode that was written (child of the directory)
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        ...

    def list_pending(self, account_id: str) -> list[tuple[str, OutboxEvent]]:
        """List pending events for processing.

        Args:
            account_id: Tenant ID to filter events

        Returns:
            List of (node_uri, OutboxEvent) tuples
        """
        ...

    def count_unfinished(self, account_id: str | None = None) -> dict[str, int]:
        """Count live pending and processing events without claiming them."""
        ...

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as being processed.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
        """
        ...

    def mark_done(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as completed and delete it.

        Args:
            event: OutboxEvent to complete
            node_uri: URI of the associated node
        """
        ...

    def move_to_dlq(self, event: OutboxEvent, node_uri: str) -> None:
        """Move a failed event to dead letter queue.

        Called when retry_count exceeds MAX_RETRY.

        Args:
            event: OutboxEvent that failed
            node_uri: URI of the associated node
        """
        ...

    def increment_retry(self, event: OutboxEvent, node_uri: str, next_retry_at: datetime | None = None) -> None:
        """Increment retry count for a failed event.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
        """
        ...

    def try_acquire(
        self,
        event: OutboxEvent,
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Atomically claim an event for processing.

        Creates .outbox/{event_id}.processing lock file.
        Returns False if already locked by another worker within timeout.

        Args:
            event: OutboxEvent to acquire
            node_uri: URI of the associated node
            worker_id: Unique identifier for the processing worker
            timeout_seconds: Seconds before a stale lock is considered expired

        Returns:
            True if lock was acquired, False if already locked
        """
        ...

    def release(self, event: OutboxEvent, node_uri: str) -> None:
        """Release the processing lock. Called on success or exception.

        Removes the .outbox/{event_id}.processing lock file.

        Args:
            event: OutboxEvent to release
            node_uri: URI of the associated node
        """
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Heterogeneous graph storage for multimodal memory system.

    Provides CRUD operations for raw nodes, extracted nodes, memory nodes,
    and edges between them. Implements incremental PageRank updates.

    Multi-tenant isolation is enforced via account_id in RequestContext.
    """

    # === Node CRUD ===

    def create_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Create a new graph node from a serialized node payload.

        Args:
            payload: JSON-ready node payload. Callers with a typed GraphNode
                should pass ``node.serialize()``. Custom callers may pass any
                dict with the required base fields (``node_type``, ``account_id``;
                ``node_id`` is generated when absent) plus node-type-specific
                fields such as ``raw_type``/``content``, ``extracted_content``,
                or ``summary``.
            ctx: RequestContext for access control

        Returns:
            The created node with generated ID

        Raises:
            AccessDeniedError: If account_id mismatch
            ValidationError: If node data is invalid
        """
        ...

    def create_node_with_edges(
        self, payload: dict[str, Any], edges: list[GraphEdge], ctx: RequestContext,
    ) -> GraphNode:
        """Create a new node and its provenance edges as one logical operation.

        SQL stores override this with a single transaction. The default is a
        compensating rollback for simple stores; existing nodes are never
        overwritten/deleted by this operation.
        """
        from core.errors import AccessDeniedError, NodeNotFoundError
        node_id = payload["node_id"]
        if self.get_node(node_id, ctx) is not None:
            raise FileExistsError(f"Graph node already exists: {node_id}")
        endpoints = {}
        for edge in edges:
            if edge.account_id != ctx.account_id:
                raise AccessDeniedError(edge.edge_id, ctx.account_id)
            if node_id not in (edge.source_id, edge.target_id):
                raise ValueError("Provenance edge must connect the new node")
            for endpoint in (edge.source_id, edge.target_id):
                if endpoint == node_id:
                    continue
                node = self.get_node(endpoint, ctx)
                if node is None:
                    raise NodeNotFoundError(endpoint)
                endpoints[endpoint] = (node, node.pagerank)
        node = self.create_node(payload, ctx)
        try:
            for edge in edges:
                self.create_edge(edge, ctx)
        except Exception:
            self.delete_node(node.node_id, ctx)
            for existing, pagerank in endpoints.values():
                existing.pagerank = pagerank
                self.update_node(existing.serialize(), ctx)
            raise
        return node

    def get_node(self, node_id: str, ctx: RequestContext) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            GraphNode if found and accessible, None otherwise
        """
        ...

    def update_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Update an existing node from a serialized node payload.

        Args:
            payload: JSON-ready node payload. Must include ``node_id`` for the
                node to update. Callers with a typed GraphNode should pass
                ``node.serialize()``.
            ctx: RequestContext for access control

        Returns:
            The updated node
        """
        ...

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all its edges.

        Args:
            node_id: UUID of the node to delete
            ctx: RequestContext for access control
        """
        ...

    def list_nodes(
        self,
        node_type: GraphNodeType | None,
        ctx: RequestContext,
        limit: int = 100,
        offset: int = 0,
        raw_type: "RawDataType | None" = None,
        extractor_type: str | None = None,
        parent_audio_node_ids: "list[str] | None" = None,
    ) -> list[GraphNode]:
        """List nodes of a specific type.

        Args:
            node_type: Filter by node type (None for all types)
            ctx: RequestContext for access control
            limit: Maximum number of nodes to return. ``None`` means
                "no LIMIT clause" — use only when the caller has
                pre-narrowed the result set via ``extractor_type``
                or ``parent_audio_node_ids`` so the unfiltered
                scan is bounded.
            offset: Number of nodes to skip
            raw_type: For ``node_type == RAW``, filter by raw data
                type. Pushed into the SQL WHERE so pagination does
                not silently drop nodes of a different type.
            extractor_type: For ``node_type == EXTRACTED``, filter by
                ``extractor_type``. Pushed into the SQL WHERE for
                the same reason — without it, a generic
                ``node_type='extracted'`` query has to share its
                LIMIT budget across face / voice_asr /
                episodic_memory / etc., and the oldest extractor
                type silently disappears once the table grows past
                ``limit``.
            parent_audio_node_ids: For ``node_type == EXTRACTED``,
                filter to rows whose ``extracted_content.source_audio_node_id``
                matches one of the given ids. Pushed into the SQL
                WHERE as ``extracted_content->>'source_audio_node_id'
                = ANY(...)``. Use this when you need the children of
                a known set of audio raw nodes (e.g. one video's
                ASR segments) without traversing the whole
                ``extractor_type='voice_asr'`` corpus.

        Returns:
            List of matching nodes
        """
        ...

    # === Edge CRUD ===

    def list_identity_pairs(
        self, source_ids: list[str], ctx: RequestContext,
    ) -> list[tuple[str, str]]:
        """Account-scoped identity endpoints, without loading neighboring nodes."""
        ...

    def ensure_identity_edges(self, edges: list[GraphEdge], ctx: RequestContext) -> int:
        """Insert missing identity pairs in bulk; existing edges are unchanged."""
        ...

    def create_edge(self, edge: GraphEdge, ctx: RequestContext) -> GraphEdge:
        """Create a new edge between nodes.

        Automatically triggers incremental PageRank update for affected nodes.

        Args:
            edge: GraphEdge to create
            ctx: RequestContext for access control

        Returns:
            The created edge
        """
        ...

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge by ID.

        Args:
            edge_id: UUID of the edge to delete
            ctx: RequestContext for access control
        """
        ...

    def get_edge(self, edge_id: str, ctx: RequestContext) -> GraphEdge | None:
        """Get an edge by ID.

        Args:
            edge_id: UUID of the edge
            ctx: RequestContext for access control

        Returns:
            GraphEdge if found, None otherwise
        """
        ...

    # === Graph Traversal ===

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",  # "incoming", "outgoing", "both"
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges.

        Args:
            node_id: UUID of the center node
            ctx: RequestContext for access control
            edge_types: Filter by edge types (None for all)
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of (neighbor_node, edge, direction) tuples
        """
        ...

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
            edge_types: Filter by edge types (None for all)

        Returns:
            SubgraphResult with nodes and edges within depth limit
        """
        ...

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
        ...

    # === PageRank ===

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores for nodes.

        Args:
            node_ids: Specific nodes to compute for (None = all nodes)
            ctx: RequestContext for access control
            damping: Damping factor (default 0.85)

        Returns:
            Dict mapping node_id to PageRank score
        """
        ...

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a specific node.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            PageRank score
        """
        ...

    # === Search ===

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memory nodes by text and filters.

        Args:
            query: Text search query
            ctx: RequestContext for access control
            min_confidence: Minimum confidence threshold
            topic: Filter by topic
            tags: Filter by tags (OR logic)
            top_k: Maximum number of results

        Returns:
            List of matching memory nodes sorted by confidence
        """
        ...

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

        Today only :class:`ExtractedNode` carries an embedding
        (under ``extracted_content["embedding"]``). Implementations MUST:

        - Filter by ``account_id`` from ``ctx`` (tenant isolation).
        - Skip nodes whose embedding is missing, malformed, or has the
          wrong dimension relative to ``query_vector``.
        - Apply the clip / video / extractor / node-type filters.
        - If before_sec is set (absolute video seconds), exclude complete
          source intervals ending later or lacking valid temporal provenance,
          before ranking/top_k.
        - Sort by cosine similarity desc and return at most ``top_k``
          ``(node_id, score)`` pairs.

        Implementations SHOULD be best-effort: a single malformed row
        must not poison the result set.

        Args:
            query_vector: 1-D query vector.
            ctx: Request context (provides ``account_id`` for isolation).
            node_types: Tuple of node-type strings (``"raw"``, ``"extracted"``,
                ``"memory"``). Unknown values are silently ignored.
            extractor_types: Optional whitelist of ``extractor_type``.
            video_id: Optional ``extracted_content["_video_id"]`` match.
            before_clip: Optional ordinal upper bound on
                ``extracted_content["_clip_id"]``. Rows missing the key
                or exceeding the bound are excluded.
            top_k: Maximum number of results to return.

        Returns:
            ``[(node_id, score), ...]`` sorted by score desc, capped at
            ``top_k``. Empty when ``top_k <= 0`` or no candidates pass.
        """
        ...

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by type and optional text query.

        Args:
            raw_type: Filter by raw data type
            ctx: RequestContext for access control
            query: Optional text search
            top_k: Maximum number of results

        Returns:
            List of matching raw nodes
        """
        ...

    # === Statistics ===

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph.

        Args:
            ctx: RequestContext for access control

        Returns:
            GraphStats with node and edge counts, top memories
        """
        ...
