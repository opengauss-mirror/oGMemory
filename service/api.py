"""Service API layer - write operations for ContextEngine.

This is the ONLY layer where RequestContext is mandatory and account_id is injected.
All external calls must provide a RequestContext for multi-tenant isolation.

Note: dev branch is WRITE-ONLY. Read operations (ReadAPI) are in phase1 branch.
See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

import threading
import uuid
from dataclasses import replace
from typing import Optional

from commit import CandidatePipeline, ContextWriter, OutboxStore
from commit.write_receipts import WriteDeadlineExceededError, WriteNotStartedError, receipt_plan
from core.errors import ConcurrentModificationError, ExtractionIncompleteError
from core.interfaces import LLM, CandidateExtractor, ContextFS
from core.logging_config import get_logger
from core.models import CandidateMemory, RequestContext, WritePlan
from core.operation_deadline import check_deadline, remaining_seconds
from extraction import Extractor
from extraction.node_processing import NodeProcessor, serialized_topics

logger = get_logger(__name__)


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
        source_occurrence_reconciliation_enabled: bool = False,
        card_format_enabled: bool = False,
        dual_pass_enabled: bool = True,
        node_processing_enabled: bool = False,
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
        if type(source_occurrence_reconciliation_enabled) is not bool:
            raise ValueError("source_occurrence_reconciliation_enabled must be a boolean")
        self._source_occurrence_reconciliation_enabled = source_occurrence_reconciliation_enabled
        self._card_format_enabled = card_format_enabled
        self._dual_pass_enabled = dual_pass_enabled
        self._node_processing_enabled = node_processing_enabled

        # Auto-create SchemaRegistry if not provided
        if schema_registry is None:
            from extraction.schemas.registry import SchemaRegistry
            schema_registry = SchemaRegistry()
        self._schema_registry = schema_registry

        # Initialize write components
        from commit.identity_resolver import MemoryIdentityResolver
        from commit.policy_router import PolicyRouter
        from core.uri_resolver import URIResolver
        write_uris = uri_resolver or URIResolver(schema_registry)
        policy_router = PolicyRouter(fs, registry=schema_registry, uri_resolver=write_uris)
        from commit.archive_builder import ArchiveBuilder

        self._writer = ContextWriter(
            fs,
            llm=self._llm,
            outbox_store=outbox_store,
            policy_router=policy_router,
            archive_builder=ArchiveBuilder(self._llm, schema_registry=schema_registry,
                                          card_format_enabled=card_format_enabled),
            directory_summary_enabled=directory_summary_enabled,
            identity_resolver=MemoryIdentityResolver(
                fs, write_uris, schema_registry=schema_registry, policy_router=policy_router,
            ),
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
                source_occurrence_reconciliation_enabled=self._source_occurrence_reconciliation_enabled,
                dual_pass_enabled=self._dual_pass_enabled,
                node_processing_enabled=self._node_processing_enabled,
                schema_registry=self._schema_registry,
                fs=self._fs,
                vector_index=self._vector_index,
                embedder=self._embedder,
                uri_resolver=self._uri_resolver,
                internal_tool_usage_tracker=self._internal_tool_usage_tracker,
                occurrence_update_eligible=lambda candidate: bool(
                    self._writer._identity_resolver
                    and self._writer._identity_resolver.handles(candidate)
                ),
            )]
        except Exception:
            # Fall back to default prompt construction if the template system is unavailable.
            return [Extractor(
                self._llm,
                mode=self._extraction_mode,
                source_occurrence_reconciliation_enabled=self._source_occurrence_reconciliation_enabled,
                dual_pass_enabled=self._dual_pass_enabled,
                node_processing_enabled=self._node_processing_enabled,
                schema_registry=self._schema_registry,
                fs=self._fs,
                vector_index=self._vector_index,
                embedder=self._embedder,
                uri_resolver=self._uri_resolver,
                internal_tool_usage_tracker=self._internal_tool_usage_tracker,
                occurrence_update_eligible=lambda candidate: bool(
                    self._writer._identity_resolver
                    and self._writer._identity_resolver.handles(candidate)
                ),
            )]

    @serialized_topics
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
        candidate_checkpoint=None,
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
        if candidate_checkpoint is not None:
            from service.extraction_checkpoint import RecoveryNeedsConfirmationError

            checkpoint_archive_id = candidate_checkpoint.journal.batch["archive_id"]
            if archive_id is not None and archive_id != checkpoint_archive_id:
                raise RecoveryNeedsConfirmationError("archive_id differs from the extraction checkpoint")
            archive_id = checkpoint_archive_id
        if archive_id is None:
            from session.session_manager import generate_archive_id
            archive_id = generate_archive_id()

        def extract():
            thread_bound = getattr(type(self._fs), "requires_thread_bound_access", None)
            execution_options = {"parallel": False} if (
                remaining_seconds() is not None or callable(thread_bound) and thread_bound(self._fs)) else {}
            return self._pipeline.extract(
                messages, ctx, session_time=session_time,
                session_summary=session_summary,
                tool_stats_text=tool_stats_text,
                archive_id=archive_id,
                **execution_options,
            )

        def refresh_profile(candidate):
            refreshed = [entry for entry in self._pipeline.filter_by_confidence(extract(), confidence_threshold)
                         if entry.category == "profile" and entry.owner_scope == candidate.owner_scope
                         and entry.target_uri in ("", candidate.target_uri)]
            if len(refreshed) != 1:
                raise ExtractionIncompleteError("profile refresh requires one complete source candidate")
            return refreshed[0]

        check_deadline()
        accepted = candidate_checkpoint.load() if candidate_checkpoint is not None else None
        if accepted is None:
            # Extraction must finish before the first accepted candidate is written.
            candidates = (candidate_checkpoint.load_drafts() if self._node_processing_enabled
                          and candidate_checkpoint is not None else None)
            if candidates is None:
                candidates = extract()
                if self._node_processing_enabled and candidate_checkpoint is not None:
                    candidates = [replace(c, node_draft_indices=[i]) if c.category in {"entity", "preference"}
                                  else c for i, c in enumerate(candidates)]
                    candidate_checkpoint.save_drafts(candidates, messages)
            check_deadline()
            filtered = self._pipeline.filter_by_confidence(candidates, confidence_threshold)
            extracted_count, filtered_count = len(candidates), len(candidates) - len(filtered)
            if self._node_processing_enabled:
                filtered = self._node_processor().process(filtered, messages, ctx)
            else:
                from commit.topic_aliases import reconcile_topic_aliases
                filtered = reconcile_topic_aliases(
                    filtered, messages, ctx, self._writer._identity_resolver, self._llm)
            # Read-bound snapshots must consume their observed versions before
            # unbound source facts can merge into those nodes. Freeze this order
            # with the new checkpoint; accepted batches retain their exact order.
            filtered = sorted(filtered, key=lambda candidate: not bool(candidate.target_uri))
            if self._card_format_enabled:
                from extraction.card_format import validate_memory_card
                for candidate in filtered:
                    validate_memory_card(candidate)
            if candidate_checkpoint is not None:
                resolver = self._writer._identity_resolver
                if resolver is not None:
                    resolver.preflight_new_batch(
                        [candidate for candidate in filtered
                         if self._writer.prewrite_skip(candidate, ctx) is None], ctx)
                check_deadline()
                filtered = candidate_checkpoint.accept(filtered, extracted_count, filtered_count)
        else:
            filtered, extracted_count, filtered_count = accepted

        # Step 3: Deduplicate (disabled — dedup causes -12% accuracy, see commit 85e0ef3)
        deduplicated = filtered

        # Step 4: Write candidates (ContextWriter handles outbox registration internally)
        plans = (self._write_checkpointed_candidates(
                     deduplicated, ctx, candidate_checkpoint, refresh_profile=refresh_profile)
                 if candidate_checkpoint is not None else self._writer.write_candidates(deduplicated, ctx))

        # Compile results
        writes_completed = sum(1 for p in plans if p.action != "skip")
        writes_skipped = sum(1 for p in plans if p.action == "skip")
        writes_failed = len(deduplicated) - writes_completed - writes_skipped

        result = {
            "archive_id": archive_id,
            "candidates_extracted": extracted_count,
            "candidates_filtered": filtered_count,
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

        # For async mode, return task_id
        if not wait:
            task_id = str(uuid.uuid4())
            with self._tasks_lock:
                self._tasks[task_id] = {"status": "completed", "result": result}
            result["task_id"] = task_id
            result["status"] = "completed"

        return result

    def _node_processor(self):
        from extraction.prefetch import MemoryPrefetcher
        resolver = self._writer._identity_resolver
        prefetcher = (MemoryPrefetcher(self._fs, self._vector_index, self._embedder,
                                     self._schema_registry, resolver.uris)
                      if self._vector_index is not None and self._embedder is not None else None)
        return NodeProcessor(resolver, self._llm, prefetcher)

    def _write_checkpointed_candidates(self, candidates, ctx, checkpoint, *, refresh_profile=None,
                                       source_only=False):
        """Reuse durable drafts and receipts; never guess whether a write committed."""
        from service.extraction_checkpoint import RecoveryNeedsConfirmationError
        plans = []
        for index, candidate in enumerate(candidates):
            check_deadline()
            completed = checkpoint.completed(index)
            if completed is not None:
                plans.append(WritePlan(**completed))
                continue
            # The frozen inputs prove these candidates cannot reach storage,
            # including old checkpoints that marked them 'writing' before skip.
            skipped = self._writer.prewrite_skip(candidate, ctx)
            if skipped is not None:
                checkpoint.complete(index, skipped)
                plans.append(skipped)
                continue
            if checkpoint.outcome_uncertain(index):
                resolver = self._writer._identity_resolver
                normalized = resolver._normalize_subject(candidate, ctx)
                committed = receipt_plan(self._fs, resolver.uris, normalized, ctx)
                if committed is None:
                    raise RecoveryNeedsConfirmationError(
                        f"write outcome needs confirmation for operation {candidate.write_operation_id}")
                checkpoint.complete(index, committed)
                plans.append(committed)
                continue
            if source_only:
                plan = WritePlan(action="skip", target_uri=candidate.target_uri,
                                 merged_fields={"status": "source_preserved"})
                checkpoint.complete(index, plan)
                plans.append(plan)
                continue
            for attempt in range(3):
                check_deadline()
                checkpoint.begin(index)
                try:
                    plan = self._writer.write_candidate(candidate, ctx)
                except WriteDeadlineExceededError as exc:
                    checkpoint.reject(index, str(exc))
                    raise
                except WriteNotStartedError as exc:
                    checkpoint.reject(index, str(exc))
                    from core.errors import LLMTransientError, MemoryCardFormatError
                    from fs.sql_adapter.session_mutation import SessionMutationBusyError
                    if (refresh_profile is not None
                            and candidate.category == "profile" and candidate.target_uri
                            and isinstance(exc.__cause__, ConcurrentModificationError)):
                        if attempt == 2:
                            raise ExtractionIncompleteError("profile update remains concurrently modified") from exc
                        check_deadline()
                        candidate = checkpoint.revise_unwritten(index, refresh_profile(candidate))
                        logger.info("Refreshed concurrently updated profile: %s", candidate.target_uri)
                        continue
                    if (self._node_processing_enabled and candidate.category in {"entity", "preference"}
                            and isinstance(exc.__cause__, ConcurrentModificationError)):
                        revised = self._node_processor().process(
                            checkpoint.reprocessing_drafts(candidate),
                            checkpoint.draft_source(), ctx)
                        if (len(revised) != 1 or candidate.target_uri
                                and revised[0].target_uri != candidate.target_uri):
                            raise MemoryCardFormatError(
                                "node revision must preserve one accepted target: " + candidate.target_uri)
                        candidate = checkpoint.revise_unwritten(index, revised[0])
                        checkpoint.begin(index)
                        try:
                            plan = self._writer.write_candidate(candidate, ctx)
                        except WriteNotStartedError as retry_error:
                            checkpoint.reject(index, str(retry_error))
                            if isinstance(retry_error.__cause__,
                                          (MemoryCardFormatError, LLMTransientError, SessionMutationBusyError)):
                                raise retry_error.__cause__
                            raise
                        checkpoint.complete(index, plan)
                        plans.append(plan)
                        break
                    if isinstance(exc.__cause__,
                                  (MemoryCardFormatError, LLMTransientError, SessionMutationBusyError)):
                        raise exc.__cause__
                    logger.error("Accepted candidate rejected before writing: %s", exc)
                    break
                checkpoint.complete(index, plan)
                plans.append(plan)
                break
        return plans

    def write_raw_chunk(
        self,
        messages: list[dict],
        ctx: RequestContext,
        chunk_index: int = 0,
        session_id: str = "",
        session_time=None,
        candidate_checkpoint=None,
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
        check_deadline()
        accepted = candidate_checkpoint.load() if candidate_checkpoint is not None else None
        if accepted is not None:
            plans = self._write_checkpointed_candidates(accepted[0], ctx, candidate_checkpoint)
            if not plans:
                raise RuntimeError("raw chunk rejected before writing")
            return {"action": plans[0].action, "target_uri": plans[0].target_uri}
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

        if candidate_checkpoint is not None:
            check_deadline()
            accepted = candidate_checkpoint.accept([candidate], 1, 0)
            plans = self._write_checkpointed_candidates(accepted, ctx, candidate_checkpoint)
            if not plans:
                raise RuntimeError("raw chunk rejected before writing")
            plan = plans[0]
        else:
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
        thread_bound = getattr(type(self._fs), "requires_thread_bound_access", None)
        if callable(thread_bound) and thread_bound(self._fs):
            raise RuntimeError("an owning SQL session requires a separate service background handoff")
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
    source_occurrence_reconciliation_enabled: bool = False,
    dual_pass_enabled: bool = True,
    node_processing_enabled: bool = False,
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
        source_occurrence_reconciliation_enabled=source_occurrence_reconciliation_enabled,
        dual_pass_enabled=dual_pass_enabled,
        node_processing_enabled=node_processing_enabled,
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

from core.errors import AccessDeniedError
from core.errors import ValidationError as CoreValidationError
from core.models import (
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
)
from retrieval.context_reader import ContextReader
from retrieval.pipeline import RetrievalPipeline


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
        memory_directories: list[str] | None = None,
        use_scene: bool | None = None,
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
            memory_directories=memory_directories,
            **({"use_scene": use_scene} if use_scene is not None else {}),
        )

        if not include_debug:
            result.trace = None

        return result

    def read_memory(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        unit_key: str | None = None,
    ) -> RetrievedBlock:
        """Read L2 md file content by URI.

        Since search_memory already returns abstract in results,
        read_memory only reads the actual md file content.
        """
        return self._read_service.read(uri, ctx=ctx, **({"unit_key": unit_key} if unit_key is not None else {}))
