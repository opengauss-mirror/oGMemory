"""Service API layer - write operations for ContextEngine.

This is the ONLY layer where RequestContext is mandatory and account_id is injected.
All external calls must provide a RequestContext for multi-tenant isolation.

Note: dev branch is WRITE-ONLY. Read operations (ReadAPI) are in phase1 branch.
See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

import threading
import uuid
from dataclasses import dataclass
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)
from core.interfaces import ContextFS, LLM, CandidateExtractor
from core.models import RequestContext, CandidateMemory, WritePlan
from commit import ContextWriter, CandidatePipeline, OutboxStore
from extraction import Extractor


class MemoryWriteAPI:
    """Public API for memory write operations.

    All methods enforce multi-tenant isolation via RequestContext.
    Writes are orchestrated through the commit pipeline:
    1. Extract candidates via CandidateExtractor
    2. Plan write actions via MergePolicy
    3. Build ContextNode via ArchiveBuilder
    4. Write to storage via ContextFS
    5. Register OutboxEvents for async indexing
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        outbox_store: Optional[OutboxStore] = None,
        schema_registry=None,
        vector_index=None,
        embedder=None,
        uri_resolver=None,
internal_tool_usage_tracker=None,
        extraction_mode: str = "eager",
        directory_summary_enabled: bool = True,
        graph_enabled: bool = False,
        relation_store=None,
        relation_candidate_service=None,
    ):
        """Initialize the API with required dependencies.

        Args:
            fs: ContextFS implementation for persisting nodes
            llm: LLM instance for extraction
            outbox_store: OutboxStore for registering index events (optional)
            schema_registry: SchemaRegistry for dynamic tool generation (auto-created if None)
            vector_index: Optional VectorIndex for prefetching existing memories
            embedder: Optional Embedder for prefetching existing memories
            uri_resolver: Optional URIResolver for prefetching existing memories
internal_tool_usage_tracker: Optional tracker for oGMem internal tool calls
            extraction_mode: Extraction mode — "eager" or "lazy". Defaults to "eager".
            directory_summary_enabled: Whether to register directory summary events
        """
        self._fs = fs
        self._llm = llm
        self._outbox_store = outbox_store
        self._vector_index = vector_index
        self._embedder = embedder
        self._uri_resolver = uri_resolver
        self._internal_tool_usage_tracker = internal_tool_usage_tracker
        self._extraction_mode = extraction_mode
        self._graph_enabled = graph_enabled
        self._relation_store = relation_store
        self._relation_candidate_service = relation_candidate_service
        if graph_enabled and relation_store is None:
            raise ValueError("Graph ingestion requires canonical relation storage")

        # Auto-create SchemaRegistry if not provided
        if schema_registry is None:
            from extraction.schemas.registry import SchemaRegistry
            schema_registry = SchemaRegistry()
        self._schema_registry = schema_registry

        # Initialize write components
        from commit.policy_router import PolicyRouter
        policy_router = PolicyRouter(fs, registry=schema_registry, uri_resolver=uri_resolver)
        self._writer = ContextWriter(
            fs,
            llm=self._llm,
            outbox_store=outbox_store,
            policy_router=policy_router,
            directory_summary_enabled=directory_summary_enabled,
        )
        self._pipeline = CandidatePipeline()
        self._pipeline.set_extractors(self._create_extractors())
        self._tasks: dict[str, dict] = {}
        self._tasks_lock = threading.Lock()

    def _create_extractors(self) -> list[CandidateExtractor]:
        """Create default extractors for the pipeline.

        Returns:
            List of CandidateExtractor instances (single tool-use Extractor)
        """
        try:
            from extraction.prompts import PromptManager
            pm = PromptManager()
            return [Extractor(
                self._llm,
                prompt_manager=pm,
                mode=self._extraction_mode,
                schema_registry=self._schema_registry,
                fs=self._fs,
                vector_index=self._vector_index,
                embedder=self._embedder,
                uri_resolver=self._uri_resolver,
                internal_tool_usage_tracker=self._internal_tool_usage_tracker,
                activation_enabled=self._graph_enabled,
            )]
        except Exception:
            # Fall back to default prompt construction if the template system is unavailable.
            return [Extractor(
                self._llm,
                mode=self._extraction_mode,
                schema_registry=self._schema_registry,
                fs=self._fs,
                vector_index=self._vector_index,
                embedder=self._embedder,
                uri_resolver=self._uri_resolver,
                internal_tool_usage_tracker=self._internal_tool_usage_tracker,
                activation_enabled=self._graph_enabled,
            )]

    def commit_session(
        self,
        messages: list[dict],
        ctx: RequestContext,
        confidence_threshold: float = 0.5,
        wait: bool = True,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
        archive_id: str | None = None,
    ) -> dict:
        """Commit a conversation session to memory.

        This is the main entry point for writing memories.
        Extracts candidates from messages, filters by confidence,
        and writes to storage.

        Args:
            messages: List of message dicts with "role" and "content"
                      Example: [{"role": "user", "content": "..."}, ...]
            ctx: RequestContext for this operation
            confidence_threshold: Minimum confidence for writing (default 0.5)
            wait: If True, block until extraction completes (default).
                  If False, return immediately with task_id for async processing.
            session_time: Optional datetime for temporal resolution (defaults to now).
            session_summary: Optional summary of previously extracted content.
            tool_stats_text: Optional tool usage statistics text.
            archive_id: Optional archive_id for provenance tracking.

        Returns:
            Dict with write results:
            {
                "candidates_extracted": int,
                "candidates_filtered": int,
                "writes_completed": int,
                "writes_skipped": int,
                "writes_failed": int,
                "plans": list[WritePlan dict],
                "task_id": str (only if wait=False),
                "status": "processing" (only if wait=False)
            }
        """
        # Step 0: Auto-generate archive_id for provenance if not provided
        if archive_id is None:
            from session.session_manager import generate_archive_id
            archive_id = generate_archive_id()

        # Step 1: Extract candidates
        candidates = self._pipeline.extract(
            messages, ctx, session_time=session_time,
            session_summary=session_summary,
            tool_stats_text=tool_stats_text,
            archive_id=archive_id,
        )

        # Step 2: Filter by confidence
        filtered = self._pipeline.filter_by_confidence(candidates, confidence_threshold)

        # Step 3: Deduplicate (disabled — dedup causes -12% accuracy, see commit 85e0ef3)
        deduplicated = filtered

        # Proposals are transient ingestion inputs, never persisted as node metadata.
        from extraction.relation_proposals import take
        extraction_proposals = take(deduplicated)
        # Step 4: Write candidates (ContextWriter handles outbox registration internally)
        plans = self._writer.write_candidates(deduplicated, ctx)

        graph_diagnostics = None
        if self._graph_enabled:
            from extraction.cross_prompt_relations import run_integrated_relations
            from extraction.relation_proposals import resolve_edges, union_edges
            native_edges, native_rejections = resolve_edges(
                extraction_proposals, deduplicated, plans, ctx, fs=self._fs,
                uri_resolver=self._uri_resolver)
            historical, diagnostics = [], None
            lookup_error = None
            if len(deduplicated) == len(plans) and self._relation_candidate_service is not None:
                try:
                    historical, diagnostics = self._relation_candidate_service.recommend(deduplicated, plans, ctx)
                except Exception as exc:
                    lookup_error = type(exc).__name__
            edges, graph_diagnostics = run_integrated_relations(
                messages,deduplicated,plans,ctx,fs=self._fs,vector_index=self._vector_index,
                embedder=self._embedder,llm=self._llm,historical_candidates=historical,
                historical_diagnostics=diagnostics)
            edges = union_edges(native_edges, edges)
            graph_diagnostics.update(node_extraction_relation_proposals=len(extraction_proposals),
                                     native_relation_accepted=len(native_edges),
                                     native_relation_rejections=native_rejections)
            if lookup_error:
                graph_diagnostics["historical_recommendation_failure"] = lookup_error
            try:
                self._relation_store.upsert_edges(edges,ctx)
                graph_diagnostics["relation_persisted"] = len(edges)
            except Exception as exc:
                graph_diagnostics.update(status="persistence_failed",relation_persisted=0,
                                         persistence_error=type(exc).__name__)
                logger.exception("Graph relation persistence failed")

        # Compile results
        writes_completed = sum(1 for p in plans if p.action != "skip")
        writes_skipped = sum(1 for p in plans if p.action == "skip")
        writes_failed = len(deduplicated) - writes_completed - writes_skipped

        result = {
            "archive_id": archive_id,
            "candidates_extracted": len(candidates),
            "candidates_filtered": len(candidates) - len(filtered),
            "writes_completed": writes_completed,
            "writes_skipped": writes_skipped,
            "writes_failed": writes_failed,
            "plans": [
                {
                    "action": p.action,
                    "target_uri": p.target_uri,
                    "merged_fields": p.merged_fields,
                }
                for p in plans
            ],
        }

        if graph_diagnostics is not None:
            result["graph"] = graph_diagnostics

        # For async mode, return task_id
        if not wait:
            task_id = str(uuid.uuid4())
            with self._tasks_lock:
                self._tasks[task_id] = {"status": "completed", "result": result}
            result["task_id"] = task_id
            result["status"] = "completed"

        return result

    def write_raw_chunk(
        self,
        messages: list[dict],
        ctx: RequestContext,
        chunk_index: int = 0,
        session_id: str = "",
        session_time=None,
    ) -> dict:
        """Write raw chunk text as a session_archive node for retrieval.

        Stores the original conversation text so that details missed by
        structured extraction can still be found via vector search.

        Args:
            messages: List of message dicts in this chunk
            ctx: RequestContext for this operation
            chunk_index: Index of this chunk within the session
            session_id: Session identifier for unique routing key
            session_time: Optional datetime for temporal context

        Returns:
            Dict with write result
        """
        if not messages:
            return {"action": "skip", "reason": "empty messages"}

        # Format messages into plain text
        lines = []
        participants = set()
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("text")
                )
            if content:
                lines.append(f"{role}: {content}")
                if role not in ("system", "assistant"):
                    participants.add(role)

        raw_text = "\n".join(lines)
        if not raw_text.strip():
            return {"action": "skip", "reason": "no text content"}

        # Build unique routing key
        sid = session_id[:8] if session_id else "unknown"
        time_str = ""
        if session_time:
            try:
                from datetime import datetime
                if isinstance(session_time, datetime):
                    time_str = session_time.strftime("%Y%m%d")
                elif isinstance(session_time, str):
                    time_str = session_time[:10].replace("-", "")
            except Exception:
                pass
        routing_key = f"chunk_{time_str}_{sid}_{chunk_index}"

        # Build abstract from first ~200 chars
        abstract = raw_text[:200].replace("\n", " ")
        if len(raw_text) > 200:
            abstract += "..."

        # Build overview from first few messages
        overview_lines = lines[:5]
        overview = "\n".join(overview_lines)
        if len(lines) > 5:
            overview += f"\n... ({len(lines) - 5} more messages)"

        candidate = CandidateMemory(
            category="session_archive",
            owner_scope="user",
            routing_key=routing_key,
            abstract=abstract,
            overview=overview,
            content=raw_text,
            confidence=1.0,
            when=str(session_time) if session_time else None,
            who=", ".join(participants) if participants else None,
        )

        plan = self._writer.write_candidate(candidate, ctx)
        return {
            "action": plan.action,
            "target_uri": plan.target_uri,
        }

    def commit_session_async(
        self,
        messages: list[dict],
        ctx: RequestContext,
        confidence_threshold: float = 0.5,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> str:
        """Fire-and-forget version of commit_session.

        Dispatches extraction + write to a background thread and returns
        a task_id immediately.  The caller can poll get_task_status(task_id)
        for the result.

        Returns:
            task_id string for tracking the background job.
        """
        task_id = str(uuid.uuid4())
        with self._tasks_lock:
            self._tasks[task_id] = {"status": "processing", "result": None}

        def _run():
            try:
                result = self.commit_session(
                    messages=messages,
                    ctx=ctx,
                    confidence_threshold=confidence_threshold,
                    wait=True,
                    session_time=session_time,
                    session_summary=session_summary,
                    tool_stats_text=tool_stats_text,
                )
                with self._tasks_lock:
                    self._tasks[task_id] = {"status": "completed", "result": result}
            except Exception as exc:
                logger.error("commit_session_async failed for task %s: %s", task_id, exc, exc_info=True)
                with self._tasks_lock:
                    self._tasks[task_id] = {"status": "failed", "error": str(exc)}

        t = threading.Thread(target=_run, daemon=True, name=f"commit-{task_id[:8]}")
        t.start()
        return task_id

    def get_task_status(self, task_id: str) -> dict | None:
        """Check status of an async commit_session task."""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def write_memory(
        self,
        candidate: CandidateMemory,
        ctx: RequestContext,
    ) -> dict:
        """Write a single candidate memory.

        Bypasses extraction - use when you already have a CandidateMemory.

        Args:
            candidate: CandidateMemory to write
            ctx: RequestContext for this operation

        Returns:
            Dict with write result:
            {
                "action": str,
                "target_uri": str,
                "merged_fields": dict,
            }
        """
        plan = self._writer.write_candidate(candidate, ctx)

        return {
            "action": plan.action,
            "target_uri": plan.target_uri,
            "merged_fields": plan.merged_fields,
        }

    def write_memories(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext,
        parallel: bool = True,
    ) -> list[dict]:
        """Write multiple candidate memories.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations
            parallel: If True, write in parallel (default True)

        Returns:
            List of write result dicts
        """
        # Deduplicate (disabled — dedup causes -12% accuracy, see commit 85e0ef3)
        deduplicated = candidates

        # Write (ContextWriter handles outbox registration internally)
        if parallel:
            plans = self._writer.write_candidates_parallel(deduplicated, ctx)
        else:
            plans = self._writer.write_candidates(deduplicated, ctx)

        return [
            {
                "action": p.action,
                "target_uri": p.target_uri,
                "merged_fields": p.merged_fields,
            }
            for p in plans
        ]


# Singleton instances for simple usage
# In production, use dependency injection
_default_write_api: Optional[MemoryWriteAPI] = None


def init_write_api(
    fs: ContextFS,
    llm: LLM,
    outbox_store: Optional[OutboxStore] = None,
    schema_registry=None,
    vector_index=None,
    embedder=None,
    uri_resolver=None,
    internal_tool_usage_tracker=None,
) -> MemoryWriteAPI:
    """Initialize the global write API instance.

    Args:
        fs: ContextFS implementation
        llm: LLM instance for extraction
        outbox_store: Optional OutboxStore for async indexing
        schema_registry: Optional SchemaRegistry for dynamic tool generation
        vector_index: Optional VectorIndex for prefetching existing memories
        embedder: Optional Embedder for prefetching existing memories
        uri_resolver: Optional URIResolver for prefetching existing memories
        internal_tool_usage_tracker: Optional tracker for oGMem internal tool calls

    Returns:
        Configured MemoryWriteAPI instance
    """
    global _default_write_api
    _default_write_api = MemoryWriteAPI(
        fs, llm, outbox_store, schema_registry,
        vector_index, embedder, uri_resolver,
        internal_tool_usage_tracker,
    )
    return _default_write_api


def get_write_api() -> Optional[MemoryWriteAPI]:
    """Get the global write API instance.

    Returns:
        MemoryWriteAPI if initialized, None otherwise
    """
    return _default_write_api


# ---------------------------------------------------------------------------
# Read / Search API
# ---------------------------------------------------------------------------

from core.errors import AccessDeniedError, ValidationError as CoreValidationError
from core.models import (
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
)
from retrieval.pipeline import RetrievalPipeline
from retrieval.context_reader import ContextReader


class ReadAPI:
    """Public API for memory search and read operations.

    Exposes two tools consumed by AI agents:
      - search_memory: semantic retrieval -> structured SearchMemoryResult
      - read_memory: URI-based read -> RetrievedBlock with full content
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        read_service: ContextReader | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._read_service = read_service
        self._cfg = config or RetrievalConfig()

    def append_graph_supplement(self, items, query, stats, ctx, budget, formatter):
        supplement = getattr(self._pipeline,"_graph_supplement",None)
        if not (self._cfg.enable_graph_expansion and self._cfg.graph_supplement_enabled and supplement):
            return items
        try:
            return supplement.deliver(items,query,stats,ctx,budget,formatter)
        except Exception as exc:
            logger.warning("Graph delivery failed; preserving baseline: %s",exc)
            stats["graph_supplement_delivery"] = dict(error=type(exc).__name__,added_uris=[],
                                                      baseline_prefix_unchanged=True)
            return items

    def search_memory(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int = 10,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        session_archive: dict | None = None,
        score_threshold: float | None = None,
        include_debug: bool = False,
        mode: str = RetrieverMode.QUICK,
        fill_content_for_top_k: int = 0,
        context_type: str | None = None,
        resource_type: str | None = None,
        collect_graph_supplement: bool = False,
    ) -> SearchMemoryResult:
        if not (query or "").strip():
            raise CoreValidationError("query", "query must not be empty")
        if top_k > self._cfg.max_top_k:
            raise CoreValidationError("top_k", f"top_k={top_k} exceeds max {self._cfg.max_top_k}")

        if target_uri:
            prefix = f"ctx://{ctx.account_id}/"
            if target_uri.startswith("ctx://") and not target_uri.startswith(prefix):
                raise AccessDeniedError(target_uri, ctx.account_id, "target_uri account mismatch")

        result = self._pipeline.run(
            query, ctx,
            top_k=top_k,
            categories=categories,
            target_uri=target_uri,
            session_archive=session_archive,
            score_threshold=score_threshold,
            mode=mode,
            fill_content_for_top_k=fill_content_for_top_k,
            context_type=context_type,
            resource_type=resource_type,
            **({"collect_graph_supplement": True} if collect_graph_supplement else {}),
        )

        if not include_debug:
            result.trace = None

        return result

    def read_memory(
        self,
        uri: str,
        ctx: RequestContext,
    ) -> RetrievedBlock:
        """Read L2 md file content by URI.

        Since search_memory already returns abstract in results,
        read_memory only reads the actual md file content.
        """
        return self._read_service.read(uri, ctx=ctx)
