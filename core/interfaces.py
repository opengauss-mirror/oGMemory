# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Core protocol interfaces for ContextEngine.

Phase 0 frozen — these protocols define the contracts between layers.
All protocols use typing.Protocol for structural subtyping.
"""

from datetime import datetime
from typing import Any, Protocol
from typing_extensions import runtime_checkable


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
