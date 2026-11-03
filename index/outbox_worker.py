"""OutboxWorker: Async index synchronization from OutboxEvents.

Processes OutboxEvents to update the vector index:
1. Read OutboxEvent from store
2. Handle event by type:
   - UPSERT_CONTEXT: Embed and upsert leaf node records
   - UPSERT_DIRECTORY: Generate directory summary and upsert
   - DELETE_CONTEXT: Remove from vector index
3. Delete event on success

Happy path: event → process → delete
Failure path: retry with exponential backoff → DLQ after MAX_RETRY
"""

import hashlib
import inspect
import json
import logging
import re
import uuid
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from core.errors import NodeNotFoundError
from core.interfaces import LLM, ContextFS, Embedder, VectorIndex
from core.models import IndexRecord, OutboxEvent, RequestContext
from providers.token_tracker import (
    distributed_usage_scope,
    embedding_scope_weights,
    pop_bucket,
    push_bucket,
    request_usage_scope,
    usage_scope,
)
from retrieval.bm25_index import Bm25Index, build_code_bm25_document

logger = logging.getLogger(__name__)

_MEMORY_OWNER_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/"
)
_RESOURCE_PATTERN = re.compile(r"^ctx://(?P<account>[^/]+)/resources(?:/|$)")

DEFAULT_MAX_RETRY: int = 5
DEFAULT_BASE_DELAY_MS: int = 1000
DEFAULT_MAX_DELAY_MS: int = 60000


class _DurableDeletePayloadFS(Protocol):
    def is_tombstoned(self, uri: str, ctx: RequestContext) -> bool: ...

    def delete_node_payload(self, uri: str, ctx: RequestContext) -> None: ...


@dataclass
class WorkerResult:
    """Result of processing a single OutboxEvent."""

    event_id: str
    success: bool
    records_upserted: int = 0
    error_message: str = ""
    moved_to_dlq: bool = False


@dataclass
class WorkerConfig:
    """Configuration for OutboxWorker."""

    max_retry: int = DEFAULT_MAX_RETRY
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS


class OutboxWorker:
    """Processes OutboxEvents to synchronize the vector index.

    Event types:
    - UPSERT_CONTEXT: Embed and upsert leaf node records
    - UPSERT_DIRECTORY: Generate directory summary and upsert
    - DELETE_CONTEXT: Remove from vector index
    - ARCHIVE_CONTEXT: Remove from vector index (soft-deleted node)

    Events that fail are retried with exponential backoff.
    Events exceeding MAX_RETRY are moved to the dead letter queue (DLQ).
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: WorkerConfig | None = None,
        fs: ContextFS | None = None,
        llm: LLM | None = None,
        directory_summary_enabled: bool = True,
        bm25_index: Bm25Index | None = None,
        agent_memory_handler=None,
        memory_projection_handler=None,
    ) -> None:
        """Initialize the OutboxWorker.

        Args:
            vector_index: Vector index for upsert
            embedder: Embedder for generating embeddings
            config: Worker configuration (optional)
            fs: ContextFS for reading directory children (required for UPSERT_DIRECTORY)
            llm: LLM for generating directory summaries (required for UPSERT_DIRECTORY)
            directory_summary_enabled: If False, skip UPSERT_DIRECTORY events
            bm25_index: Optional shared lexical index for code IndexRecords
        """
        self._vector_index = vector_index
        self._embedder = embedder
        self._config = config or WorkerConfig()
        self._fs = fs
        self._llm = llm
        self._directory_summary_enabled = directory_summary_enabled
        self._bm25_index = bm25_index
        self._agent_memory_handler = agent_memory_handler
        self._memory_projection_handler = memory_projection_handler

    def _upsert_code_bm25(self, records: list[IndexRecord]) -> None:
        if self._bm25_index is None:
            return
        documents = [
            document
            for record in records
            if (document := build_code_bm25_document(record)) is not None
        ]
        if documents:
            self._bm25_index.bulk_upsert(documents)

    def _delete_bm25(self, record_ids: list[str]) -> None:
        if self._bm25_index is not None:
            self._bm25_index.delete(record_ids)

    def process_event(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None = None,
    ) -> WorkerResult:
        """Process a single OutboxEvent.

        Routes to appropriate handler based on event_type.

        Args:
            event: OutboxEvent to process
            ctx: RequestContext (required for UPSERT_DIRECTORY)

        Returns:
            WorkerResult with success status
        """
        if (
            event.retry_count >= self._config.max_retry
            and not self._is_durable_delete_event(event)
        ):
            return self._handle_max_retry_exceeded(event)

        try:
            event_type = event.event_type

            if event_type == "EXTRACT_AGENT_MEMORY":
                if self._agent_memory_handler is None:
                    raise RuntimeError("agent memory handler is unavailable")
                with request_usage_scope(RequestContext(**event.payload["context"])):
                    self._agent_memory_handler(event)
                return WorkerResult(event_id=event.event_id, success=True, records_upserted=0)
            elif event_type == "UPSERT_CONTEXT":
                resolved_ctx = ctx or self._build_request_context(event)
                usage_ctx = (
                    self._usage_context_for_event(event, resolved_ctx)
                    if resolved_ctx
                    else None
                )
                scope = request_usage_scope(usage_ctx) if usage_ctx else nullcontext()
                with scope, ExitStack() as stack:
                    if (event.payload.get("agent_memory_version") == 1
                            and self._has_real_method(self._fs, "resource_mutation_lock")):
                        stack.enter_context(self._fs.resource_mutation_lock(event.uri, resolved_ctx))
                    return self._process_upsert_context(event)
            elif event_type == "UPSERT_DIRECTORY":
                resolved_ctx = ctx or self._build_request_context(event)
                usage_ctx = (
                    self._usage_context_for_event(event, resolved_ctx)
                    if resolved_ctx
                    else None
                )
                scope = request_usage_scope(usage_ctx) if usage_ctx else nullcontext()
                with scope:
                    return self._process_upsert_directory(event, resolved_ctx)
            elif event_type == "DELETE_CONTEXT":
                return self._process_delete_context(event, ctx)
            elif event_type == "ARCHIVE_CONTEXT":
                return self._process_archive_context(event)
            else:
                logger.warning("Unknown event type: %s", event_type)
                return WorkerResult(
                    event_id=event.event_id,
                    success=True,
                    records_upserted=0,
                )

        except Exception as e:
            logger.error(
                "Failed to process event %s: %s",
                event.event_id, e, exc_info=True,
            )
            return WorkerResult(
                event_id=event.event_id,
                success=False,
                records_upserted=0,
                error_message=str(e),
            )

    def _process_upsert_context(self, event: OutboxEvent) -> WorkerResult:
        """Process UPSERT_CONTEXT event for leaf node.

        Flow:
        1. Extract IndexRecords from payload
        2. Embed record texts
        3. Upsert to vector index

        Args:
            event: OutboxEvent with leaf node records

        Returns:
            WorkerResult with success status
        """
        records = self._extract_records(event)
        if event.payload.get("agent_memory_version") == 1:
            if self._fs is None:
                raise RuntimeError("agent memory indexing requires current node storage")
            from index.index_record_builder import build_index_records
            try:
                node = self._fs.read_node(event.uri, self._build_request_context(event))
                records = build_index_records(node)
            except NodeNotFoundError:
                records = []
            current_ids = {record.id for record in records}
            obsolete = [key for key in event.payload.get("ids_to_delete", []) if key not in current_ids]
            if obsolete:
                self._vector_index.delete(obsolete)
                self._delete_bm25(obsolete)
        records = self._current_projection_records(event, records)
        if not records:
            logger.warning("Event %s has no records to index", event.event_id)
            return WorkerResult(
                event_id=event.event_id,
                success=True,
                records_upserted=0,
            )

        texts = [record.text for record in records]
        inputs = self._embedding_scope_inputs(event, records)
        with distributed_usage_scope(embedding_scope_weights(inputs), inputs=inputs):
            embeddings = self._embedder.embed_texts(texts)
        augmented_records = self._augment_with_embeddings(records, embeddings)
        self._vector_index.upsert(augmented_records)
        self._upsert_code_bm25(records)

        logger.info(
            "Successfully processed UPSERT_CONTEXT %s, "
            "upserted %d records for URI %s",
            event.event_id, len(records), event.uri,
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=len(records),
        )

    @staticmethod
    def _projection_record_fingerprint(records: list[IndexRecord]) -> str:
        values = [(item.id, item.uri, item.level, item.text, item.filters, item.metadata)
                  for item in records]
        return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()

    def _is_memory_projection_event(self, event: OutboxEvent) -> bool:
        return bool(self._memory_projection_handler is not None
                    and _MEMORY_OWNER_PATTERN.match(event.uri)
                    and "/memories/" in event.uri
                    and event.event_type in {"UPSERT_CONTEXT", "ARCHIVE_CONTEXT", "DELETE_CONTEXT"})

    def _current_projection_records(self, event: OutboxEvent, records: list[IndexRecord]) -> list[IndexRecord]:
        if not self._is_memory_projection_event(event):
            return records
        if self._fs is None:
            raise RuntimeError("memory projection requires canonical node storage")
        from index.index_record_builder import build_index_records

        try:
            node = self._fs.read_node(event.uri, self._build_request_context(event))
        except NodeNotFoundError:
            return []
        if node.context_type != "MEMORY" or node.metadata.get("node_type") == "directory":
            return records
        current = build_index_records(node)
        event.payload["records"] = [asdict(record) for record in current]
        event.payload["_memory_projection_version"] = node.metadata.get("version")
        event.payload["_memory_projection_fingerprint"] = self._projection_record_fingerprint(current)
        return current

    def _notify_memory_projection(self, event: OutboxEvent, result: WorkerResult) -> None:
        if not self._is_memory_projection_event(event):
            return
        if event.event_type == "UPSERT_CONTEXT" and result.records_upserted == 0:
            return
        if self._fs is None:
            raise RuntimeError("memory projection requires canonical node storage")
        ctx = self._build_request_context(event)
        try:
            node = self._fs.read_node(event.uri, ctx)
        except NodeNotFoundError:
            node = None
        if event.event_type == "UPSERT_CONTEXT":
            if node is None or node.context_type != "MEMORY" or node.metadata.get("node_type") == "directory":
                return
            from index.index_record_builder import build_index_records

            fingerprint = self._projection_record_fingerprint(build_index_records(node))
            if (fingerprint != event.payload.get("_memory_projection_fingerprint")
                    or node.metadata.get("version") != event.payload.get("_memory_projection_version")):
                raise RuntimeError("canonical memory changed during indexing")
        elif node is not None:
            return
        self._memory_projection_handler(event, ctx)

    def _process_upsert_directory(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None,
    ) -> WorkerResult:
        """Process UPSERT_DIRECTORY event for parent directory.

        Flow:
        1. Check fs and llm are available
        2. Use DirectoryEventHandler for DAG-style bottom-up processing
        3. Handler waits for subdirectory summaries before generating parent
        4. Handler uses fallback when LLM fails
        5. Handler vectorizes and stores to database

        Args:
            event: OutboxEvent with directory URI
            ctx: RequestContext (required for FS access)

        Returns:
            WorkerResult with success status
        """
        if not self._directory_summary_enabled:
            logger.info("UPSERT_DIRECTORY skipped (directory summary disabled): %s", event.uri)
            return WorkerResult(
                event_id=event.event_id,
                success=True,
                records_upserted=0,
            )

        if not self._fs or not self._llm:
            raise ValueError(
                "UPSERT_DIRECTORY requires fs and llm. "
                "Initialize OutboxWorker with fs and llm parameters."
            )

        if not ctx:
            raise ValueError(
                "UPSERT_DIRECTORY requires RequestContext. "
                "Pass ctx parameter to process_event()."
            )

        from index.directory_event_handler import DirectoryEventHandler

        handler = DirectoryEventHandler(
            fs=self._fs,
            llm=self._llm,
            embedder=self._embedder,
            vector_index=self._vector_index,
        )

        result = handler.process_directory_event(event, ctx)

        if not result.success:
            logger.error(
                "Failed to process directory event %s: %s",
                event.uri, result.error_message
            )
            return WorkerResult(
                event_id=event.event_id,
                success=False,
                records_upserted=0,
                error_message=result.error_message,
            )

        logger.info(
            "Successfully processed UPSERT_DIRECTORY %s, "
            "completed %d directories, failed %d directories",
            event.event_id,
            result.stats.completed_dirs,
            result.stats.failed_dirs,
        )

        # Store actual record IDs and written directory URIs in payload
        # so that the post-upsert tombstone rollback can clean them up
        # without querying the (potentially already moved) filesystem.
        if result.record_ids:
            event.payload["records"] = [
                {"id": rid} for rid in result.record_ids
            ]
        if result.written_dir_uris:
            event.payload["written_dir_uris"] = result.written_dir_uris

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=result.stats.completed_dirs,
        )

    @staticmethod
    def _has_real_method(instance, name: str) -> bool:
        """Return whether ``name`` is defined by the instance's real class."""

        return instance is not None and callable(getattr(type(instance), name, None))

    def _cleanup_tombstoned_payload(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None,
    ) -> None:
        """Finish an AGFS durable delete before the event can be acknowledged."""

        if ctx is None or self._fs is None or not all(
            self._has_real_method(self._fs, method)
            for method in ("is_tombstoned", "delete_node_payload")
        ):
            return
        fs = cast(_DurableDeletePayloadFS, self._fs)
        if fs.is_tombstoned(event.uri, ctx):
            fs.delete_node_payload(event.uri, ctx)

    def _process_delete_context(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None,
    ) -> WorkerResult:
        """Process DELETE_CONTEXT event.

        If the payload contains ``ids_to_delete`` (emitted by move_node),
        use those directly — they correctly include the L2 /content.md
        suffix.  Otherwise, fall back to deriving ids from the event URI.

        Args:
            event: OutboxEvent with deleted URI
            ctx: RequestContext for tombstoned AGFS payload cleanup

        Returns:
            WorkerResult with success status
        """
        # Use pre-built ids when available (handles L2 /content.md suffix)
        ids_to_delete = event.payload.get("ids_to_delete")
        if not ids_to_delete:
            from index.index_record_builder import build_record_id
            l2_uri = event.uri.rstrip("/") + "/content.md"
            ids_to_delete = [
                build_record_id(event.uri, 0),
                build_record_id(event.uri, 1),
                build_record_id(l2_uri, 2),
            ]

        self._vector_index.delete(ids_to_delete)
        self._delete_bm25(ids_to_delete)
        self._cleanup_tombstoned_payload(event, ctx)

        logger.info(
            "Successfully processed DELETE_CONTEXT %s, "
            "deleted %d records for URI %s",
            event.event_id, len(ids_to_delete), event.uri,
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=0,
        )

    def _process_archive_context(self, event: OutboxEvent) -> WorkerResult:
        """Process ARCHIVE_CONTEXT event.

        Removes vector index records for the archived node so it no
        longer appears in search results.  Uses pre-built ids_to_delete
        from the payload (same approach as DELETE_CONTEXT).

        Args:
            event: OutboxEvent with archived URI

        Returns:
            WorkerResult with success status
        """
        ids_to_delete = event.payload.get("ids_to_delete")
        if not ids_to_delete:
            from index.index_record_builder import build_record_id
            l2_uri = event.uri.rstrip("/") + "/content.md"
            ids_to_delete = [
                build_record_id(event.uri, 0),
                build_record_id(event.uri, 1),
                build_record_id(l2_uri, 2),
            ]

        self._vector_index.delete(ids_to_delete)
        self._delete_bm25(ids_to_delete)

        logger.info(
            "Successfully processed ARCHIVE_CONTEXT %s, "
            "deleted %d records for URI %s",
            event.event_id, len(ids_to_delete), event.uri,
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=0,
        )

    def calculate_backoff(self, retry_count: int) -> timedelta:
        """Calculate exponential backoff delay.

        Args:
            retry_count: Current retry count

        Returns:
            Timedelta for delay before next retry
        """
        exponent = min(max(retry_count, 0), 30)
        delay_ms = self._config.base_delay_ms * (2 ** exponent)
        delay_ms = min(delay_ms, self._config.max_delay_ms)
        return timedelta(milliseconds=delay_ms)

    def should_retry(self, event: OutboxEvent, result: WorkerResult) -> bool:
        """Check if event should be retried.

        Args:
            event: The OutboxEvent that was processed
            result: Result from process_event

        Returns:
            True if event should be retried
        """
        if result.success:
            return False
        if self._is_durable_delete_event(event):
            return True
        if event.retry_count >= self._config.max_retry:
            return False
        return True

    @staticmethod
    def _is_durable_delete_event(event: OutboxEvent) -> bool:
        """Return whether DELETE must remain live until payload cleanup succeeds."""

        return bool(
            event.event_type == "DELETE_CONTEXT"
            and event.payload.get("durable_delete") is True
        )

    def _extract_records(self, event: OutboxEvent) -> list[IndexRecord]:
        """Extract IndexRecords from event payload.

        Args:
            event: OutboxEvent with payload

        Returns:
            List of IndexRecords to upsert

        Raises:
            ValueError: If payload format is invalid
        """
        payload = event.payload

        # Payload should contain "records" array
        if "records" not in payload:
            raise ValueError(f"Event payload missing 'records': {event.event_id}")

        records_data = payload["records"]
        if not isinstance(records_data, list):
            raise ValueError(f"Event 'records' is not a list: {event.event_id}")

        # Reconstruct IndexRecord objects
        records: list[IndexRecord] = []
        for record_dict in records_data:
            # Validate required fields
            required_fields = {"id", "uri", "level", "text", "filters"}
            missing = required_fields - record_dict.keys()
            if missing:
                raise ValueError(
                    f"Record missing fields {missing}: {event.event_id}"
                )

            # Validate mandatory filters (cross-tenant leakage prevention)
            if "account_id" not in record_dict["filters"]:
                raise ValueError(
                    f"Record missing account_id in filters: {event.event_id}. "
                    "This causes SILENT cross-tenant leakage!"
                )
            if "owner_space" not in record_dict["filters"]:
                raise ValueError(
                    f"Record missing owner_space in filters: {event.event_id}. "
                    "This causes SILENT cross-tenant leakage!"
                )

            record = IndexRecord(
                id=record_dict["id"],
                uri=record_dict["uri"],
                level=record_dict["level"],
                text=record_dict["text"],
                filters=record_dict["filters"],
                metadata=record_dict.get("metadata", {}),
            )
            records.append(record)

        # SECURITY: Validate all record account_ids match event URI
        # This prevents cross-tenant injection via malicious payloads
        try:
            expected_account = event.uri.split("://")[1].split("/")[0]
        except (IndexError, AttributeError):
            raise ValueError(
                f"Cannot extract account from event URI: {event.uri}"
            )

        for record in records:
            record_account = record.filters.get("account_id", "")
            if record_account != expected_account:
                raise ValueError(
                    f"Record account_id '{record_account}' does not match "
                    f"event URI account '{expected_account}': {event.event_id}. "
                    "Possible cross-tenant injection attempt!"
                )

        return records

    def _augment_with_embeddings(
        self,
        records: list[IndexRecord],
        embeddings: list[list[float]],
    ) -> list[IndexRecord]:
        """Augment IndexRecords with embedding vectors.

        Some VectorIndex implementations need embeddings in the record.
        This adds embeddings to metadata for those implementations.

        Args:
            records: IndexRecords without embeddings
            embeddings: Embedding vectors

        Returns:
            Augmented records with embeddings in metadata
        """
        augmented: list[IndexRecord] = []

        for record, embedding in zip(records, embeddings):
            # Create a copy with embedding in metadata
            new_metadata = dict(record.metadata)
            new_metadata["_embedding"] = embedding

            # Create new IndexRecord with augmented metadata
            # Note: IndexRecord is a dataclass, we can create a new instance
            augmented_record = IndexRecord(
                id=record.id,
                uri=record.uri,
                level=record.level,
                text=record.text,
                filters=record.filters,
                metadata=new_metadata,
            )
            augmented.append(augmented_record)

        return augmented

    def _handle_max_retry_exceeded(self, event: OutboxEvent) -> WorkerResult:
        """Handle event that has exceeded max retry attempts.

        Args:
            event: OutboxEvent with retry_count >= MAX_RETRY

        Returns:
            WorkerResult indicating DLQ transfer
        """
        logger.error(
            "Event %s exceeded max retry (%d), moving to DLQ",
            event.event_id, self._config.max_retry,
        )
        return WorkerResult(
            event_id=event.event_id,
            success=False,
            records_upserted=0,
            error_message="Max retry exceeded",
            moved_to_dlq=True,
        )

    def _handle_processing_result(
        self,
        event: OutboxEvent,
        node_uri: str,
        outbox_store,
        result: WorkerResult,
        stats: dict,
    ) -> None:
        """Handle the result of processing an event.

        Args:
            event: The OutboxEvent that was processed
            node_uri: URI of the associated node
            outbox_store: OutboxStore instance
            result: Processing result
            stats: Statistics dict to update
        """
        if result.success:
            self._notify_memory_projection(event, result)
            outbox_store.mark_done(event, node_uri)
            stats["succeeded"] += 1
            return

        reason = str(result.error_message or "").strip()
        previous_reason = str(event.payload.get("failure_reason") or "").strip()
        if reason and (reason != "Max retry exceeded" or not previous_reason):
            event.payload["failure_reason"] = reason

        if result.moved_to_dlq or not self.should_retry(event, result):
            # Move to DLQ if explicitly marked or retries exhausted
            outbox_store.move_to_dlq(event, node_uri)
            stats["moved_to_dlq"] += 1
            if not result.moved_to_dlq:
                logger.error(
                    "Event %s exceeded max retry, moved to DLQ",
                    event.event_id
                )
        else:
            # Schedule for retry
            backoff = self.calculate_backoff(event.retry_count)
            next_retry_at = datetime.now(UTC) + backoff
            outbox_store.increment_retry(event, node_uri, next_retry_at)
            outbox_store.release(event, node_uri)
            stats["failed"] += 1
            logger.info(
                "Event %s scheduled for retry #%d in %s",
                event.event_id, event.retry_count + 1, backoff
            )

    def _build_request_context(self, event: OutboxEvent) -> RequestContext | None:
        """Build a minimal RequestContext for event processing when possible."""
        match = _MEMORY_OWNER_PATTERN.match(event.uri)
        if not match:
            resource_match = _RESOURCE_PATTERN.match(event.uri)
            if not resource_match:
                return None
            return RequestContext(
                account_id=resource_match.group("account"),
                user_id="outbox-worker",
                agent_id="outbox-worker",
                session_id=f"outbox:{event.event_id}",
                trace_id=f"outbox:{event.event_id}",
            )

        account_id = match.group("account")
        owner_type = match.group("owner_type")
        owner_id = match.group("owner_id")

        return RequestContext(
            account_id=account_id,
            user_id=owner_id if owner_type == "users" else "outbox-worker",
            agent_id=owner_id if owner_type == "agents" else "outbox-worker",
            session_id=f"outbox:{event.event_id}",
            trace_id=f"outbox:{event.event_id}",
        )

    @staticmethod
    def _usage_context_for_event(
        event: OutboxEvent,
        ctx: RequestContext,
    ) -> RequestContext:
        """Map provider usage to the memory owner without changing worker identity."""
        match = _MEMORY_OWNER_PATTERN.match(event.uri)
        if not match:
            return ctx
        owner_id = match.group("owner_id")
        if match.group("owner_type") == "users":
            return replace(ctx, user_id=owner_id, agent_id="")
        return replace(ctx, user_id="outbox-worker", agent_id=owner_id)

    def _embedding_scope_inputs(self, event, records):
        ctx = self._build_request_context(event)
        ctx = self._usage_context_for_event(event, ctx) if ctx else None
        inputs = []
        for record in records:
            account = record.filters.get('account_id') or (ctx.account_id if ctx else '')
            owner = record.filters.get('owner_space') or (
                (ctx.agent_space_name() if ctx.agent_id else ctx.user_space_name()) if ctx else ''
            )
            scope = (account, owner)
            inputs.append((scope, max(1, len(record.text))))
        return inputs

    @staticmethod
    def _supports_batch_claim(outbox_store) -> bool:
        """Safely detect SQL-style batch claim support.

        Avoid ``hasattr``/``getattr(instance, ...)`` here because ``Mock()``
        fabricates arbitrary attributes, which would incorrectly route tests
        through the batch-claim path.
        """
        instance_values = getattr(outbox_store, "__dict__", {})
        if "supports_batch_claim" in instance_values:
            return instance_values["supports_batch_claim"] is True
        return getattr(type(outbox_store), "supports_batch_claim", False) is True

    @staticmethod
    def _supports_liveness_check(outbox_store) -> bool:
        """Safely detect is_event_current support.

        Uses the same class/instance flag pattern as _supports_batch_claim
        to avoid false positives with Mock().
        """
        return callable(getattr(type(outbox_store), "is_event_current", None))

    @staticmethod
    def _processing_lock(outbox_store, node_uri: str):
        lock = getattr(type(outbox_store), "processing_lock", None)
        if callable(lock):
            return outbox_store.processing_lock(node_uri)
        return nullcontext()

    @staticmethod
    def _is_upsert_event(event: OutboxEvent) -> bool:
        return event.event_type in {"UPSERT_CONTEXT", "UPSERT_DIRECTORY"}

    @staticmethod
    def _event_is_current(outbox_store, event: OutboxEvent, node_uri: str) -> bool:
        """Call one- or two-argument liveness implementations safely."""

        method = getattr(outbox_store, "is_event_current")
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            # Some extension-backed callables do not expose a signature.
            return bool(method(event.event_id))
        try:
            signature.bind(event.event_id, node_uri)
        except TypeError:
            return bool(method(event.event_id))
        return bool(method(event.event_id, node_uri))

    def _rollback_stale_upsert(
        self,
        event: OutboxEvent,
        outbox_store,
        ctx: RequestContext | None,
    ) -> None:
        """Delete vector records and directory nodes for a tombstoned event.

        Called after the post-upsert tombstone check detects that the event
        was deleted by move_node.  Both UPSERT_CONTEXT and UPSERT_DIRECTORY
        store their actual record IDs in event.payload["records"], so rollback
        is reliable even after move_node has relocated the source URI.

        For UPSERT_DIRECTORY, also deletes the directory summary context_nodes
        that the handler wrote back to ContextFS during processing.  If
        delete_node fails for a transient reason, a DELETE_CONTEXT event is
        emitted so the normal worker loop eventually cleans it up — because
        the original outbox row was already removed by move_node, a simple
        retry would silently match 0 rows.
        """
        records_data = event.payload.get("records", [])
        ids = [r["id"] for r in records_data if "id" in r]

        if ids:
            self._vector_index.delete(ids)
            self._delete_bm25(ids)
            logger.info(
                "Rolled back %d stale records for tombstoned event %s",
                len(ids), event.event_id,
            )

        # Delete stale directory summary nodes written during processing.
        written_dir_uris = event.payload.get("written_dir_uris", [])
        if written_dir_uris and self._fs:
            if ctx is None:
                logger.error(
                    "Cannot roll back stale directory nodes without request context: %s",
                    event.event_id,
                )
                return
            failed_uris: list[str] = []
            for dir_uri in written_dir_uris:
                try:
                    self._fs.delete_node(dir_uri, ctx)
                except NodeNotFoundError:
                    logger.debug(
                        "Stale dir node %s already gone", dir_uri,
                    )
                except Exception:
                    logger.warning(
                        "Failed to delete stale dir node %s, "
                        "will emit cleanup event",
                        dir_uri, exc_info=True,
                    )
                    failed_uris.append(dir_uri)

            # Emit DELETE_CONTEXT events for URIs we couldn't clean up.
            # The original outbox row is gone (tombstoned by move_node),
            # so increment_retry would match 0 rows and never retry.
            if failed_uris and hasattr(outbox_store, "register_delete"):
                for dir_uri in failed_uris:
                    try:
                        outbox_store.register_delete(dir_uri, ctx)
                        logger.info(
                            "Emitted cleanup DELETE for stale dir node %s",
                            dir_uri,
                        )
                    except Exception:
                        logger.error(
                            "Failed to emit cleanup event for %s, "
                            "stale node may persist",
                            dir_uri, exc_info=True,
                        )

            logger.info(
                "Rolled back %d/%d stale directory nodes for tombstoned event %s",
                len(written_dir_uris) - len(failed_uris),
                len(written_dir_uris),
                event.event_id,
            )

    def _process_claimed_events(
        self,
        claimed_events: list[tuple[str, OutboxEvent]],
        outbox_store,
        stats: dict,
        worker_id: str = "default",
    ) -> None:
        """Process already-claimed events and update acknowledgement state."""
        for node_uri, event in claimed_events:
            if (event.event_type == "EXTRACT_AGENT_MEMORY"
                    and self._has_real_method(outbox_store, "claim_agent_event")):
                self._process_agent_claim(event, outbox_store, stats, worker_id)
                continue
            stats["processed"] += 1
            lock_stack = ExitStack()
            try:
                lock_stack.enter_context(
                    self._processing_lock(outbox_store, node_uri)
                )
            except Exception as exc:
                logger.error(
                    "Failed to acquire processing lock for event %s: %s",
                    event.event_id,
                    exc,
                )
                failure_result = WorkerResult(
                    event_id=event.event_id,
                    success=False,
                    error_message=str(exc),
                )
                self._handle_processing_result(
                    event, node_uri, outbox_store, failure_result, stats
                )
                continue
            with lock_stack:
                self._process_claimed_event(node_uri, event, outbox_store, stats)

    def _process_agent_claim(self, event, outbox_store, stats, worker_id) -> None:
        try:
            with outbox_store.claim_agent_event(event, worker_id) as acquired:
                if acquired:
                    stats["processed"] += 1
                    self._process_claimed_event(event.uri, event, outbox_store, stats)
                else:
                    stats["skipped"] += 1
        except Exception:
            logger.exception("Agent extraction claim failed: %s", event.event_id)

    def _process_claimed_event(
        self,
        node_uri: str,
        event: OutboxEvent,
        outbox_store,
        stats: dict,
    ) -> None:
        """Process and acknowledge one event while its URI lock is held."""

        # Tombstone check: if move_node deleted this event between
        # claim_batch and here, skip processing to avoid writing stale
        # vectors at the old URI.  DB errors propagate so we don't
        # accidentally skip legitimate events.
        if (
            self._is_upsert_event(event)
            and self._supports_liveness_check(outbox_store)
        ):
            try:
                current = self._event_is_current(outbox_store, event, node_uri)
            except Exception:
                # DB unavailable — cannot determine liveness.
                # Proceed with processing; the event is still in
                # memory and the worst case is a duplicate upsert.
                logger.warning(
                    "Liveness check failed for %s, proceeding",
                    event.event_id, exc_info=True,
                )
                current = True
            if not current:
                logger.info(
                    "Event %s no longer in outbox (tombstoned), skipping",
                    event.event_id,
                )
                stats["skipped"] += 1
                return

        try:
            ctx = self._build_request_context(event)
            result = self.process_event(event, ctx)

            # Post-upsert tombstone check: close the race where
            # move_node deleted this event *after* the pre-check
            # returned True but *before* the upsert finished.
            # On DB error, treat the event as failed so it retries
            # rather than silently acking potentially stale vectors.
            if (
                result.success
                and self._is_upsert_event(event)
                and self._supports_liveness_check(outbox_store)
            ):
                try:
                    current = self._event_is_current(outbox_store, event, node_uri)
                except Exception:
                    logger.warning(
                        "Post-upsert liveness check failed for %s, "
                        "treating as failure to retry",
                        event.event_id, exc_info=True,
                    )
                    # Re-raise so the event goes through the failure
                    # path (retry or DLQ) instead of being acked.
                    raise
                if not current:
                    self._rollback_stale_upsert(event, outbox_store, ctx)
                    result = WorkerResult(
                        event_id=event.event_id,
                        success=True,
                        records_upserted=0,
                    )
                    logger.info(
                        "Event %s tombstoned after upsert, rolled back",
                        event.event_id,
                    )

            self._handle_processing_result(
                event, node_uri, outbox_store, result, stats
            )
        except Exception as e:
            logger.error(
                "Exception processing event %s: %s",
                event.event_id, e, exc_info=True
            )
            failure_result = WorkerResult(
                event_id=event.event_id,
                success=False,
                error_message=str(e),
            )
            self._handle_processing_result(
                event, node_uri, outbox_store, failure_result, stats
            )

    def _process_upsert_batch(self, batch, outbox_store, stats: dict) -> None:
        try:
            all_records = []
            boundaries = []
            inputs = []
            for event, _, records in batch:
                boundaries.append(len(all_records))
                all_records.extend(records)
                inputs.extend(self._embedding_scope_inputs(event, records))
            scope_weights = embedding_scope_weights(inputs)

            bucket, bucket_token = push_bucket()
            try:
                with usage_scope("", ""), distributed_usage_scope(scope_weights, inputs=inputs):
                    all_embeddings = self._embedder.embed_texts(
                        [record.text for record in all_records]
                    )
            finally:
                pop_bucket(bucket_token)
                tracker = getattr(self._embedder, "token_tracker", None)
                if tracker and hasattr(tracker, "record_scoped_embed_distribution"):
                    tracker.record_scoped_embed_distribution(
                        bucket.embed_tokens,
                        bucket.embed_calls,
                        scope_weights,
                    )
            for idx, (event, node_uri, records) in enumerate(batch):
                stats["processed"] += 1
                start = boundaries[idx]
                embeddings = all_embeddings[start : start + len(records)]
                augmented = self._augment_with_embeddings(records, embeddings)
                try:
                    self._vector_index.upsert(augmented)
                    self._upsert_code_bm25(records)
                    result = WorkerResult(
                        event_id=event.event_id,
                        success=True,
                        records_upserted=len(records),
                    )
                    if self._supports_liveness_check(outbox_store):
                        current = self._event_is_current(
                            outbox_store,
                            event,
                            node_uri,
                        )
                        if not current:
                            self._rollback_stale_upsert(
                                event,
                                outbox_store,
                                self._build_request_context(event),
                            )
                            result = WorkerResult(
                                event_id=event.event_id,
                                success=True,
                                records_upserted=0,
                            )
                    self._handle_processing_result(event, node_uri, outbox_store, result, stats)
                    logger.info(
                        "Batch-embedded UPSERT_CONTEXT %s: %d records for %s",
                        event.event_id, len(records), event.uri,
                    )
                except Exception as exc:
                    logger.error("Upsert failed for %s: %s", event.event_id, exc)
                    failure = WorkerResult(
                        event_id=event.event_id,
                        success=False,
                        error_message=str(exc),
                    )
                    self._handle_processing_result(
                        event, node_uri, outbox_store, failure, stats
                    )
        except Exception as exc:
            logger.error("Batch embedding failed: %s", exc, exc_info=True)
            for event, node_uri, _ in batch:
                stats["processed"] += 1
                failure = WorkerResult(
                    event_id=event.event_id,
                    success=False,
                    error_message=str(exc),
                )
                self._handle_processing_result(event, node_uri, outbox_store, failure, stats)

    def run_once(
        self,
        outbox_store,  # OutboxStore type, avoid import to prevent circular dependency
        account_ids: list[str],
        worker_id: str = "default",
    ) -> dict:
        """Single polling cycle: scan → process → ack/retry/dlq.

        Call this from a scheduler (e.g., APScheduler, cron, or asyncio loop).

        Args:
            outbox_store: OutboxStore instance with list_pending/mark_done/etc.
            account_ids: List of account IDs to scan
            worker_id: Unique identifier for this worker (for distributed locking)

        Returns:
            Dict with counts: processed / succeeded / failed / moved_to_dlq / skipped
        """
        stats = {"processed": 0, "succeeded": 0, "failed": 0, "moved_to_dlq": 0, "skipped": 0}
        excluded_types = ["EXTRACT_AGENT_MEMORY"] if self._agent_memory_handler is None else []

        # Fast path: SQL outbox with batch-claim support
        if self._supports_batch_claim(outbox_store):
            try:
                claimed_events = outbox_store.claim_batch(
                    worker_id=worker_id, exclude_event_types=excluded_types,
                )
            except Exception as e:
                logger.error("Failed to claim pending events: %s", e)
                return stats

            self._process_claimed_events(claimed_events, outbox_store, stats, worker_id)
            return stats

        # Collect all pending events across accounts, grouped by type.
        # UPSERT_CONTEXT events are batch-embedded for efficiency.
        all_pending: list[tuple[str, OutboxEvent]] = []
        for account_id in account_ids:
            try:
                pending = outbox_store.list_pending(account_id)
            except Exception as e:
                logger.error("Failed to list pending events for %s: %s", account_id, e)
                continue
            all_pending.extend(pending)

        # Phase 1: Batch-embed all UPSERT_CONTEXT events
        upsert_batch: list[tuple[OutboxEvent, str, list[IndexRecord]]] = []
        other_events: list[tuple[OutboxEvent, str]] = []

        for node_uri, event in all_pending:
            if event.event_type in excluded_types:
                stats["skipped"] += 1
                continue
            if (event.event_type == "EXTRACT_AGENT_MEMORY"
                    and self._has_real_method(outbox_store, "claim_agent_event")):
                self._process_agent_claim(event, outbox_store, stats, worker_id)
                continue
            if not outbox_store.try_acquire(event, node_uri, worker_id):
                stats["skipped"] += 1
                continue

            if (
                self._is_upsert_event(event)
                and self._supports_liveness_check(outbox_store)
            ):
                try:
                    current = self._event_is_current(outbox_store, event, node_uri)
                except Exception as exc:
                    logger.warning(
                        "Pre-upsert liveness check failed for %s",
                        event.event_id,
                        exc_info=True,
                    )
                    failure_result = WorkerResult(
                        event_id=event.event_id,
                        success=False,
                        error_message=str(exc),
                    )
                    stats["processed"] += 1
                    self._handle_processing_result(
                        event,
                        node_uri,
                        outbox_store,
                        failure_result,
                        stats,
                    )
                    continue
                if not current:
                    try:
                        outbox_store.release(event, node_uri)
                    except Exception:
                        logger.debug(
                            "Stale event lock was already removed for %s",
                            event.event_id,
                            exc_info=True,
                        )
                    stats["skipped"] += 1
                    continue

            if event.event_type == "UPSERT_CONTEXT" and event.payload.get("agent_memory_version") != 1:
                try:
                    records = self._current_projection_records(event, self._extract_records(event))
                    if records:
                        upsert_batch.append((event, node_uri, records))
                    else:
                        # Empty records — mark as succeeded immediately
                        outbox_store.mark_done(event, node_uri)
                        stats["processed"] += 1
                        stats["succeeded"] += 1
                except Exception as e:
                    logger.error("Failed to extract records for %s: %s", event.event_id, e)
                    failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                    self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)
            else:
                other_events.append((event, node_uri))

        # Preserve one provider batch while distributing its usage by owner.
        if upsert_batch:
            self._process_upsert_batch(upsert_batch, outbox_store, stats)

        # Phase 2: Process non-UPSERT_CONTEXT events individually
        for event, node_uri in other_events:
            stats["processed"] += 1
            try:
                ctx = self._build_request_context(event)
                result = self.process_event(event, ctx)
                if (
                    result.success
                    and self._is_upsert_event(event)
                    and self._supports_liveness_check(outbox_store)
                ):
                    current = self._event_is_current(outbox_store, event, node_uri)
                    if not current:
                        self._rollback_stale_upsert(
                            event,
                            outbox_store,
                            ctx,
                        )
                        result = WorkerResult(
                            event_id=event.event_id,
                            success=True,
                            records_upserted=0,
                        )
                self._handle_processing_result(event, node_uri, outbox_store, result, stats)
            except Exception as e:
                logger.error("Exception processing event %s: %s", event.event_id, e, exc_info=True)
                failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)

        return stats


def create_upsert_event(
    uri: str,
    records: list[IndexRecord],
) -> OutboxEvent:
    """Create an UPSERT_CONTEXT OutboxEvent.

    Utility function for write-path to create events.

    Args:
        uri: ContextNode URI
        records: IndexRecords to upsert

    Returns:
        OutboxEvent ready for serialization
    """
    # Serialize IndexRecords to dict for JSON storage
    records_data = [
        {
            "id": r.id,
            "uri": r.uri,
            "level": r.level,
            "text": r.text,
            "filters": r.filters,
            "metadata": r.metadata,
        }
        for r in records
    ]

    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type="UPSERT_CONTEXT",
        uri=uri,
        payload={
            "uri": uri,
            "records": records_data,
        },
        status="PENDING",
        retry_count=0,
    )
