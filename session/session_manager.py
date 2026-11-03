"""Server-side session lifecycle manager.

Manages per-session message buffers with optional AGFS persistence.
Accumulate → threshold commit → archive + extract session model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable, Optional

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from providers.token_tracker import UsageTracker
from session.models import ArchiveEntry, SessionMessage, SessionMeta, SessionWindowState
from session.session_state import Commitment, SessionState, TaskState
from session.topic_buffer import SessionTopicBuffer

logger = logging.getLogger("ogmem.session")

_SESSION_STATE_MISSING_TTL_SECONDS = 5.0


class SessionOperationInProgress(RuntimeError):
    """A managed operation conflicted with deletion or buffer replacement."""

    def __init__(
        self,
        *,
        conflict_type: str,
        retryable: bool,
    ) -> None:
        super().__init__("session_operation_in_progress")
        self.conflict_type = conflict_type
        self.retryable = retryable


def generate_archive_id() -> str:
    """Generate a unique archive_id.

    Format: {timestamp}_{uuid8} e.g., '20260515_100000_a1b2c3d4'
    """
    return (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )

# ---------------------------------------------------------------------------
# SessionBuffer — in-memory message accumulator
# ---------------------------------------------------------------------------


@dataclass
class SessionBuffer:
    """In-memory buffer for a single session's messages."""

    session_id: str
    durable_state_token: tuple[str, str | None] | None = field(default=None, repr=False, kw_only=True)
    ingest_handoff_ids: set[str] = field(default_factory=set, repr=False, kw_only=True)
    messages: list[SessionMessage] = field(default_factory=list)
    meta: SessionMeta = field(default_factory=SessionMeta)
    commit_in_progress: bool = False
    turn_count: int = 0
    window_state: SessionWindowState = field(default_factory=SessionWindowState)
    usage_stats: UsageTracker = field(default_factory=UsageTracker)
    # Unified tracker for provider/tool usage. ``tool_usage_stats`` remains
    # as a dict property for older callers and prompt rendering.

    # Incremental extraction tracking
    extraction_watermark: int = 0   # Index into self.messages after last extraction
    extraction_summary: str = ""    # Summary of already-extracted content (~500 tokens max)
    completed_request_ids: list[str] = field(default_factory=list)
    active_request_ids: set[str] = field(default_factory=set, repr=False)
    # Optional feature-owned state. It is serialized only when non-empty so
    # disabled extensions do not change the legacy SessionState payload.
    runtime_extensions: dict[str, dict] = field(default_factory=dict)
    compaction_prepare_token: str = ""
    compaction_prepare_archive_id: str = ""
    compaction_prepare_message_count: int = 0
    compaction_prepare_watermark: int = 0
    compaction_prepare_token_created_at: str = ""

    # Background extraction coordination: set when after_turn spawns a
    # background thread; cleared (and event set) when the thread finishes.
    # dispose() waits on this to avoid the race where the session is removed
    # before background commit completes.
    extraction_in_progress: bool = False
    extraction_done_event: object = None  # threading.Event — typed as object to avoid import
    extraction_active_count: int = 0
    extraction_lock: object = field(default_factory=threading.RLock, repr=False)
    # Serializes adapter-attested semantic-turn checkpoint governance from
    # high-water claim through persistence/stats completion or rollback.
    semantic_turn_lock: object = field(default_factory=threading.RLock, repr=False)
    # Cache-delay state performs durable load/save I/O and must serialize only
    # callers for this session, never unrelated sessions.
    tool_result_cache_delay_lock: object = field(
        default_factory=threading.RLock,
        repr=False,
    )
    session_state_save_lock: object = field(
        default_factory=threading.RLock,
        repr=False,
    )
    lifecycle_lock: object = field(default_factory=threading.RLock, repr=False)
    lifecycle_prepare_in_progress: bool = False
    lifecycle_source_ids: set[str] = field(default_factory=set, repr=False)
    ordinary_turns_in_progress: int = 0
    dispose_in_progress: bool = False
    active_operation_count: int = 0
    cleaning: bool = False
    cleanup_retry_pending: bool = False
    deleted_by_cleanup: bool = False

    @property
    def pending_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self.messages)

    @property
    def tool_usage_stats(self) -> dict[str, dict]:
        return self.usage_stats.tool_stats

    @tool_usage_stats.setter
    def tool_usage_stats(self, stats: dict[str, dict]) -> None:
        self.usage_stats.merge_tool_stats(stats)

    def _invalidate_compaction_prepare(self) -> None:
        # Extracted provenance keeps this archive identity until the source is archived.
        self.compaction_prepare_token = ""
        self.compaction_prepare_message_count = 0
        self.compaction_prepare_watermark = 0
        self.compaction_prepare_token_created_at = ""

    @contextmanager
    def lifecycle_source_scope(self, *, allow_active_extraction=False):
        """Reserve source capture through its durable lifecycle checkpoint."""
        with self.extraction_lock:
            conflict = ("ordinary_turn_in_progress" if self.ordinary_turns_in_progress
                        else "lifecycle_preparing" if self.lifecycle_prepare_in_progress
                        else "extraction_in_progress" if self.extraction_in_progress
                        and not allow_active_extraction else None)
            if conflict is None:
                self.lifecycle_prepare_in_progress = True
        try:
            yield conflict
        finally:
            if conflict is None:
                with self.extraction_lock:
                    self.lifecycle_source_ids.clear()
                    self.lifecycle_prepare_in_progress = False

    def capture_lifecycle_source(self, source=None):
        """Detach the original source; later appends stay outside this snapshot."""
        with self.extraction_lock:
            snapshot = deepcopy(self.messages if source is None else source)
            source_ids = {message.id for message in snapshot}
            if source is not None and [message for message in self.messages if message.id in source_ids] != snapshot:
                raise ValueError("lifecycle source snapshot changed before capture")
            if self.lifecycle_prepare_in_progress:
                self.lifecycle_source_ids = source_ids
            return snapshot

    def add(
        self,
        role: str,
        content: str,
        created_at: str | None = None,
        message_id: str | None = None,
    ) -> SessionMessage:
        with self.extraction_lock:
            if self.cleaning:
                raise SessionOperationInProgress(
                    conflict_type="session_deleting",
                    retryable=False,
                )
            if message_id:
                for index, existing in enumerate(self.messages):
                    if existing.id != message_id:
                        continue
                    if existing.role == role and existing.content == content:
                        return existing
                    if (index < self.extraction_watermark or existing.id in self.pending_extraction_ids()
                            or existing.id in self.lifecycle_source_ids):
                        raise ValueError("cannot rewrite a message in an extraction snapshot")
                    replacement = SessionMessage(
                        id=message_id,
                        role=role,
                        content=content,
                        created_at=created_at or existing.created_at,
                    )
                    self.messages[index] = replacement
                    self.turn_count = sum(1 for item in self.messages if item.role == "user")
                    self._invalidate_compaction_prepare()
                    self.meta.updated_at = datetime.now(timezone.utc).isoformat()
                    return replacement
            kwargs: dict = dict(
                id=message_id or f"msg_{uuid.uuid4().hex[:12]}",
                role=role,
                content=content,
            )
            if created_at:
                kwargs["created_at"] = created_at
            msg = SessionMessage(**kwargs)
            self.messages.append(msg)
            self._invalidate_compaction_prepare()
            if role == "user":
                self.turn_count += 1
                # Read-write closure: update active_task from user message
                # Simple heuristic: if message looks like a task statement, capture it
                if content and len(content.strip()) > 0:
                    # Take first 200 chars as potential task statement
                    task_candidate = content.strip()[:200]
                    # Basic heuristic: starts with action verb or contains "need to"
                    action_words = ["help", "need", "want", "please", "can you", "could you"]
                    if any(word in task_candidate.lower() for word in action_words):
                        self.window_state.active_task = task_candidate
            self.meta.message_count = len(self.messages)
            self.meta.updated_at = datetime.now(timezone.utc).isoformat()
            return msg

    def remove_unextracted(self, message_id: str) -> bool:
        """Remove a stable-id message only while it remains rewriteable."""
        if not message_id:
            return False
        with self.extraction_lock:
            for index, message in enumerate(self.messages):
                if message.id != message_id:
                    continue
                if (index < self.extraction_watermark or message.id in self.pending_extraction_ids()
                        or message.id in self.lifecycle_source_ids):
                    raise ValueError("cannot remove a message in an extraction snapshot")
                self.messages.pop(index)
                self.ingest_handoff_ids.discard(message_id)
                self.turn_count = sum(1 for item in self.messages if item.role == "user")
                self._invalidate_compaction_prepare()
                self.meta.message_count = len(self.messages)
                self.meta.updated_at = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def claim_incremental_extraction(self) -> dict:
        """Atomically claim the current unextracted message range.

        The snapshot, watermark advance, and active-extraction marker must move
        together. Otherwise concurrent ``after_turn`` requests can both observe
        the same watermark and extract/archive the same messages twice.
        """
        with self.extraction_lock:
            original_watermark = self.extraction_watermark
            extraction_in_progress_before = self.extraction_in_progress or self.lifecycle_prepare_in_progress
            pending = self.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
            # A lifecycle may reserve the source after after_turn's initial
            # check. Never re-claim its accepted source as an ordinary batch.
            if isinstance(pending, dict) and "lifecycle" in pending:
                extraction_in_progress_before = True
            if extraction_in_progress_before:
                return {
                    "incremental": [],
                    "original_watermark": original_watermark,
                    "extraction_in_progress_before": True,
                    "extraction_started": False,
                    "extraction_summary": self.extraction_summary,
                }

            pending_ids = self.pending_extraction_ids()
            incremental = ([message for message in self.messages if message.id in pending_ids]
                           if pending_ids else list(self.messages[original_watermark:]))
            if pending_ids and len(incremental) != len(pending_ids):
                raise ValueError("pending extraction source messages are missing")
            extraction_started = bool(
                incremental and any(message.role != "assistant" for message in incremental)
            )

            if incremental:
                last_id = incremental[-1].id
                self.extraction_watermark = next(index + 1 for index, message in enumerate(self.messages)
                                                if message.id == last_id)
            if extraction_started:
                self.begin_extraction()

            return {
                "incremental": incremental,
                "original_watermark": original_watermark,
                "extraction_in_progress_before": extraction_in_progress_before,
                "extraction_started": extraction_started,
                "extraction_summary": self.extraction_summary,
            }

    def pending_extraction_ids(self):
        extension = self.runtime_extensions.get("memory_extraction_recovery", {})
        batch = extension.get("batch", {}) if isinstance(extension, dict) else {}
        return set(batch.get("message_ids", [])) if isinstance(batch, dict) else set()

    def request_completed(self, request_id: str) -> bool:
        with self.extraction_lock:
            return request_id in self.completed_request_ids

    def claim_request(self, request_id: str) -> str:
        with self.extraction_lock:
            if request_id in self.active_request_ids:
                return "processing"
            if request_id in self.completed_request_ids:
                return "completed"
            self.active_request_ids.add(request_id)
            return "claimed"

    def release_request(self, request_id: str) -> None:
        with self.extraction_lock:
            self.active_request_ids.discard(request_id)

    def mark_request_completed(self, request_id: str, max_entries: int = 256) -> None:
        with self.extraction_lock:
            if request_id in self.completed_request_ids:
                return
            self.completed_request_ids.append(request_id)
            if len(self.completed_request_ids) > max_entries:
                del self.completed_request_ids[:-max_entries]

    def should_compress(
        self,
        turn_threshold: int = 10,
        token_threshold: int = 10_000,
    ) -> bool:
        """Check if window state needs recompression."""
        turns_since = self.turn_count - self.window_state.turn_count_at_last_compress
        tokens_since = self.pending_tokens - self.window_state.token_count_at_last_compress
        return turns_since >= turn_threshold or tokens_since >= token_threshold

    def _check_source_clear_allowed(self) -> None:
        """Keep recovery sources intact; caller holds extraction_lock."""
        extension = self.runtime_extensions.get("memory_extraction_recovery", {})
        pending = extension.get("batch", {})
        if (self.lifecycle_prepare_in_progress or self.lifecycle_source_ids
                or isinstance(pending, dict) and "lifecycle" in pending):
            raise SessionOperationInProgress(conflict_type="lifecycle_source_reserved", retryable=True)
        if "batch" in extension:
            raise SessionOperationInProgress(conflict_type="extraction_source_reserved", retryable=True)

    def snapshot_and_clear(self) -> list[SessionMessage]:
        """Return a copy of messages and clear the buffer.

        Note: extraction_watermark and extraction_summary are preserved across
        snapshots because they track which memories have been extracted, not
        which messages are in the buffer.
        """
        with self.extraction_lock:
            self._check_source_clear_allowed()
            snap = list(self.messages)
            summary = self.extraction_summary
            self.messages.clear()
            self.turn_count = 0
            self.window_state = SessionWindowState()
            self.meta.message_count = 0
            self.meta.updated_at = datetime.now(timezone.utc).isoformat()
            # Preserve extraction state — watermark is relative to messages,
            # so adjust for the cleared buffer
            self.extraction_watermark = 0
            self.extraction_summary = summary
            self.ingest_handoff_ids.clear()
            self.compaction_prepare_token = ""
            self.compaction_prepare_archive_id = ""
            self.compaction_prepare_message_count = 0
            self.compaction_prepare_watermark = 0
            self.compaction_prepare_token_created_at = ""
            return snap

    def capture_rollback_state(self) -> dict:
        """Capture buffer fields that snapshot_and_clear mutates."""
        return {
            "messages": list(self.messages),
            "turn_count": self.turn_count,
            "window_state": deepcopy(self.window_state),
            "meta_message_count": self.meta.message_count,
            "meta_updated_at": self.meta.updated_at,
            "extraction_watermark": self.extraction_watermark,
            "extraction_summary": self.extraction_summary,
            "ingest_handoff_ids": set(self.ingest_handoff_ids),
            "compaction_prepare_token": self.compaction_prepare_token,
            "compaction_prepare_archive_id": self.compaction_prepare_archive_id,
            "compaction_prepare_message_count": self.compaction_prepare_message_count,
            "compaction_prepare_watermark": self.compaction_prepare_watermark,
            "compaction_prepare_token_created_at": self.compaction_prepare_token_created_at,
        }

    def restore_rollback_state(self, state: dict) -> None:
        """Restore buffer fields after a failed archive commit."""
        original_messages = list(state.get("messages", []))
        original_ids = {msg.id for msg in original_messages}
        new_messages = [msg for msg in self.messages if msg.id not in original_ids]
        self.messages = original_messages + new_messages
        self.turn_count = sum(1 for msg in self.messages if msg.role == "user")
        self.window_state = deepcopy(state.get("window_state", SessionWindowState()))
        self.meta.message_count = len(self.messages)
        self.meta.updated_at = state.get("meta_updated_at", self.meta.updated_at)
        self.extraction_watermark = min(
            int(state.get("extraction_watermark", 0) or 0),
            len(self.messages),
        )
        self.extraction_summary = state.get("extraction_summary", "")
        self.ingest_handoff_ids.update(state.get("ingest_handoff_ids", ()))
        self.compaction_prepare_token = state.get("compaction_prepare_token", "")
        self.compaction_prepare_archive_id = state.get("compaction_prepare_archive_id", "")
        self.compaction_prepare_message_count = int(
            state.get("compaction_prepare_message_count", 0) or 0
        )
        self.compaction_prepare_watermark = int(
            state.get("compaction_prepare_watermark", 0) or 0
        )
        self.compaction_prepare_token_created_at = state.get(
            "compaction_prepare_token_created_at", ""
        )

    def begin_extraction(self):
        """Mark a background extraction as active and return the shared done event."""
        with self.extraction_lock:
            self.begin_operation()
            event = self.extraction_done_event
            if event is None or not self.extraction_in_progress or getattr(event, "is_set", lambda: False)():
                event = threading.Event()
                self.extraction_done_event = event
            self.extraction_active_count += 1
            self.extraction_in_progress = True
            return event

    def end_extraction(self) -> None:
        """Mark one background extraction as finished."""
        with self.extraction_lock:
            if self.extraction_active_count > 0:
                self.extraction_active_count -= 1
            if self.extraction_active_count <= 0:
                self.extraction_active_count = 0
                self.extraction_in_progress = False
                event = self.extraction_done_event
                if event is not None:
                    event.set()
            self.end_operation()

    def begin_operation(self) -> None:
        with self.extraction_lock:
            if self.cleaning:
                raise SessionOperationInProgress(
                    conflict_type="session_deleting",
                    retryable=False,
                )
            self.active_operation_count += 1

    def end_operation(self) -> None:
        with self.extraction_lock:
            if self.active_operation_count > 0:
                self.active_operation_count -= 1

    def remove_messages_by_id(self, message_ids: set[str]) -> int:
        """Remove archived snapshot messages without touching newer messages."""
        if not message_ids:
            return 0

        with self.extraction_lock:
            kept: list[SessionMessage] = []
            removed = 0
            removed_before_watermark = 0
            for idx, msg in enumerate(self.messages):
                if msg.id in message_ids:
                    removed += 1
                    if idx < self.extraction_watermark:
                        removed_before_watermark += 1
                else:
                    kept.append(msg)

            if removed:
                self.messages = kept
                self.ingest_handoff_ids.difference_update(message_ids)
                self._invalidate_compaction_prepare()
                if not kept:
                    self.compaction_prepare_archive_id = ""
                self.turn_count = sum(1 for msg in self.messages if msg.role == "user")
                self.window_state.session_state_sync_turn_count = min(
                    self.window_state.session_state_sync_turn_count,
                    self.turn_count,
                )
                self.extraction_watermark = max(
                    0,
                    min(len(self.messages), self.extraction_watermark - removed_before_watermark),
                )
                self.meta.message_count = len(self.messages)
                self.meta.updated_at = datetime.now(timezone.utc).isoformat()
            return removed

    def rewind_watermark_to_ids(self, message_ids: set[str]) -> bool:
        """Rewind extraction watermark to the first remaining failed message."""
        if not message_ids:
            return False
        with self.extraction_lock:
            for idx, msg in enumerate(self.messages):
                if msg.id in message_ids:
                    if idx < self.extraction_watermark:
                        self.extraction_watermark = idx
                        self.meta.updated_at = datetime.now(timezone.utc).isoformat()
                        return True
                    return False
            return False


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Server-side session lifecycle manager.

    In-memory buffer + optional AGFS persistence.
    Provides: get_or_create, add_message, get_session, commit, get_context.
    """

    def __init__(
        self,
        get_llm: Callable[[], Any] | None = None,
        get_write_api: Callable[[], Any] | None = None,
        get_agfs: Callable[[], Any] | None = None,
        get_archive_store: Callable[[], Any] | None = None,
        archive_store_required: bool = False,
        get_context_fs: Callable[[], Any] | None = None,
        session_state: SessionState | None = None,
        archive_max_count: int = 10,
        archive_merge_threshold: int = 10,
        compression_quality_enabled: bool = False,
        compression_quality_persist_metadata: bool = False,
        session_state_bridge_enabled: bool = True,
    ):
        self._sessions: dict[str, SessionBuffer] = {}
        self._get_llm = get_llm
        self._get_write_api = get_write_api
        self._get_agfs = get_agfs
        self._get_archive_store = get_archive_store
        self._archive_store_required = archive_store_required
        self._get_context_fs = get_context_fs
        self._session_state = session_state or SessionState()
        self._session_state_loaded: set[str] = set()
        self._extraction_recovery_loaded: set[str] = set()
        self._extraction_recovery_present: set[tuple[str, ...]] = set()
        self._extraction_recovery_required: set[str] = set()
        self._session_state_missing_until: dict[str, float] = {}
        self._session_load_events: dict[str, threading.Event] = {}
        self._topic_buffers: dict[str, SessionTopicBuffer] = {}
        self._archive_max_count = max(1, int(archive_max_count or 10))
        self._archive_merge_threshold = max(1, int(archive_merge_threshold or 10))
        self._archive_merge_max_messages = 200
        self._compression_quality_enabled = compression_quality_enabled
        self._compression_quality_persist_metadata = compression_quality_persist_metadata
        self._session_state_bridge_enabled = bool(session_state_bridge_enabled)

        # Background task tracking
        self._tasks: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._save_lock = threading.RLock()

    def get_session_state(self) -> SessionState:
        return self._session_state

    @staticmethod
    def _session_state_context(
        session_id: str,
        meta: SessionMeta | None,
        fallback_ctx: RequestContext,
    ) -> RequestContext:
        account_id = meta.account_id if meta and meta.account_id else fallback_ctx.account_id
        user_id = meta.user_id if meta and meta.user_id else fallback_ctx.user_id
        agent_id = meta.agent_id if meta and meta.agent_id else fallback_ctx.agent_id
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            trace_id=fallback_ctx.trace_id,
            role=fallback_ctx.role,
            visible_owner_spaces=(),
        )

    def load_session_state(self, session_id: str, ctx: RequestContext) -> bool:
        with self._lock:
            cached = self._cached_session_state_load(session_id)
            if cached is not None:
                return cached
        if self._get_context_fs is None:
            return False
        try:
            fs = self._get_context_fs()
            if fs is None:
                return False
            status, payload = self._session_state.load_with_payload_status(session_id, fs, ctx)
            if status == "loaded":
                self._restore_runtime_state_from_payload(session_id, payload or {})
                with self._lock:
                    self._session_state_loaded.add(session_id)
            elif status == "missing":
                with self._lock:
                    self._session_state_missing_until[session_id] = monotonic() + _SESSION_STATE_MISSING_TTL_SECONDS
            return status == "loaded"
        except Exception as exc:
            logger.warning("load_session_state failed session=%s: %s", session_id, exc)
            return False

    def extraction_recovery_loaded(self, session_id: str, ctx: RequestContext) -> bool:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is not None:
                self._ensure_session_visible(buf, ctx)
                if buf.cleaning:
                    raise SessionOperationInProgress(conflict_type="session_deleting", retryable=False)
            return session_id in self._extraction_recovery_loaded

    def require_extraction_recovery(self, session_id: str) -> None:
        """Prevent the legacy best-effort loader from preceding strict validation."""
        with self._lock:
            self._extraction_recovery_required.add(session_id)

    def enable_extraction_recovery(self, session_id: str, ctx: RequestContext, fs, *, force_reload=False) -> None:
        """Load the durable runtime journal even when ordinary bridge injection is off."""
        if ctx.session_id != session_id:
            raise ValueError("extraction recovery context must match its session")
        self.require_extraction_recovery(session_id)
        with self._lock:
            buf = self._sessions[session_id]
        with buf.session_state_save_lock:
            with self._lock:
                if (session_id in self._extraction_recovery_loaded or buf.messages
                        or buf.runtime_extensions or buf.active_operation_count):
                    self._ensure_session_visible(buf, ctx)
                if session_id in self._extraction_recovery_loaded and not force_reload:
                    self._get_context_fs = lambda: fs
                    return
            payload = None
            state_token = (self._session_state.state_uri(ctx.account_id, session_id), None)
            for uri in (self._session_state.state_uri(ctx.account_id, session_id),
                        self._session_state.legacy_state_uri(ctx.account_id, session_id)):
                try:
                    node = fs.read_node(uri, ctx)
                except NodeNotFoundError:
                    continue
                if (node.category != "state" or node.owner_space != f"session:{session_id}"
                        or node.metadata.get("owner_user_id") != ctx.user_id
                        or node.metadata.get("owner_agent_id") != ctx.agent_id):
                    raise ValueError("durable extraction state belongs to another owner")
                state_token = (uri, hashlib.sha256(node.content.encode()).hexdigest())
                payload = json.loads(node.content or "{}")
                if (not isinstance(payload, dict)
                        or self._session_state._has_malformed_commitments(payload)):
                    raise ValueError("durable extraction state is invalid")
                stored_messages = payload.get("buffer_messages", [])
                def valid_messages(values):
                    return (isinstance(values, list) and all(
                        isinstance(value, dict) and all(isinstance(value.get(field), str)
                                                       for field in ("id", "role", "content", "created_at"))
                        for value in values) and len({value["id"] for value in values}) == len(values))
                if not valid_messages(stored_messages):
                    raise ValueError("durable extraction buffer messages are invalid")
                prepared = payload.get("compaction_prepare", {})
                if (not isinstance(prepared, dict)
                        or any(not isinstance(prepared.get(key, ""), str)
                               for key in ("token", "archive_id", "created_at"))
                        or any(type(prepared.get(key, 0)) is not int
                               or not 0 <= prepared.get(key, 0) <= len(stored_messages)
                               for key in ("message_count", "watermark"))):
                    raise ValueError("durable compaction preparation is invalid")
                handoff_ids = payload.get("ingest_handoff_ids", [])
                if (not isinstance(handoff_ids, list)
                        or any(not isinstance(value, str) for value in handoff_ids)):
                    raise ValueError("durable ingest handoff IDs are invalid")
                extension = payload.get("runtime_extensions", {}).get("memory_extraction_recovery")
                if extension is not None:
                    from service.extraction_checkpoint import recovery_owner
                    batch = extension.get("batch")
                    if (extension.get("version") != 1 or not isinstance(batch, dict)
                            or batch.get("owner") != recovery_owner(ctx)):
                        raise ValueError("durable extraction batch belongs to another owner or format")
                    snapshot = batch.get("snapshot")
                    message_ids = batch.get("message_ids")
                    if (not valid_messages(snapshot) or not isinstance(message_ids, list)
                            or not all(isinstance(value, str) for value in message_ids)
                            or len(set(message_ids)) != len(message_ids)
                            or [value.get("id") for value in snapshot] != message_ids):
                        raise ValueError("durable extraction source snapshot is invalid")
                    by_id = {value.get("id"): value for value in stored_messages}
                    if any(by_id.get(value["id"]) != value for value in snapshot):
                        raise ValueError("durable extraction source snapshot changed")
                break
            if (force_reload and payload is None
                    and (self._recovery_owner_key(ctx) in self._extraction_recovery_present
                         or (buf.durable_state_token is not None and buf.durable_state_token[1] is not None)
                         or ((buf.messages or buf.runtime_extensions) and buf.durable_state_token != state_token))):
                from service.extraction_checkpoint import RecoveryNeedsConfirmationError
                raise RecoveryNeedsConfirmationError("previously durable session state is missing")
            if payload is not None:
                with self._lock, buf.extraction_lock:
                    current_messages = [] if force_reload else list(buf.messages)
                    stored_by_id = {value["id"]: value for value in payload.get("buffer_messages", [])}
                    for message in current_messages:
                        stored = stored_by_id.get(message.id)
                        if stored is not None and (stored["role"], stored["content"]) != (message.role, message.content):
                            raise ValueError("durable extraction source message changed")
                    self._session_state.load_dict(session_id, payload)
                    if force_reload:
                        buf.runtime_extensions = {}
                        buf.completed_request_ids = []
                    self._restore_runtime_state_from_payload(session_id, payload)
                    by_id = {message.id: message for message in buf.messages}
                    for message in current_messages:
                        stored = by_id.get(message.id)
                        if stored is not None:
                            if (stored.role, stored.content) != (message.role, message.content):
                                raise ValueError("durable extraction source message changed")
                        else:
                            buf.messages.append(message)
                    buf.turn_count = sum(message.role == "user" for message in buf.messages)
                    buf.meta.message_count = len(buf.messages)
            self._get_context_fs = lambda: fs
            with self._lock:
                buf.durable_state_token = state_token
                if payload is not None:
                    self._extraction_recovery_present.add(self._recovery_owner_key(ctx))
                buf.meta.account_id, buf.meta.user_id, buf.meta.agent_id = ctx.account_id, ctx.user_id, ctx.agent_id
                self._extraction_recovery_loaded.add(session_id)
                self._session_state_loaded.add(session_id)
                self._session_state_missing_until.pop(session_id, None)

    def _cached_session_state_load(self, session_id: str) -> bool | None:
        if session_id in self._session_state_loaded:
            return True
        if self._session_state_missing_until.get(session_id, 0.0) > monotonic():
            return False
        self._session_state_missing_until.pop(session_id, None)
        return None

    @staticmethod
    def _recovery_owner_key(ctx):
        return (ctx.account_id, ctx.user_id, ctx.agent_id, ctx.session_id)

    def _require_sql_mutation(self, ctx, fs=None):
        fs = fs if fs is not None else (self._get_context_fs() if self._get_context_fs else None)
        sql_fs = callable(getattr(type(fs), "session_mutation_lock", None))
        archive = None
        if not sql_fs and self._get_archive_store is not None:
            from session.sql_archive_store import SQLSessionArchiveStore
            try:
                store = self._get_archive_store()
            except Exception:
                if self._archive_store_required:
                    raise
                store = None  # Preserve optional generic-store fallback.
            if issubclass(type(store), SQLSessionArchiveStore):
                archive = store
        if archive is not None:
            return self._require_sql_archive_mutation(archive, ctx.session_id, ctx)
        if sql_fs:
            self._get_context_fs = lambda: fs
            from fs.sql_adapter.session_mutation import (
                SessionMutationLostError,
                active_session_guard,
            )
            guard = active_session_guard()
            if guard is None:
                raise SessionMutationLostError("SQL session state mutation requires its guard")
            guard.check_context(ctx)
            guard.ensure_owned()
            return True
        return False

    @staticmethod
    def _require_sql_archive_mutation(store, session_id, ctx):
        """Validate the resolved archive store before any mutation workflow."""
        from session.sql_archive_store import SQLSessionArchiveStore
        if not issubclass(type(store), SQLSessionArchiveStore):
            return False
        from fs.sql_adapter.session_mutation import (
            SessionMutationLostError,
            active_session_guard,
        )
        guard = active_session_guard()
        if guard is None:
            raise SessionMutationLostError("SQL archive mutation requires its guard")
        guard.check_context(ctx)
        if session_id != guard.ctx.session_id:
            guard.poison("SQL archive mutation changed its owning session")
            guard.check_active()
        guard.check_database(store._connection_string, store._sql_dialect)
        guard.ensure_owned()
        return True

    def save_session_state(self, session_id: str, ctx: RequestContext) -> bool:
        if self._get_context_fs is None:
            return False
        pinned_buffer = None
        try:
            pinned_buffer = self.begin_managed_operation(
                session_id,
                ctx,
                create=False,
            )
            if pinned_buffer is None:
                return False
            fs = self._get_context_fs()
            if fs is None:
                return False
            self._require_sql_mutation(ctx, fs)
            save_lock = pinned_buffer.session_state_save_lock
            with save_lock:
                with self._lock:
                    buf = self._sessions.get(session_id)
                    if buf is not pinned_buffer or buf.cleaning:
                        return False
                    with buf.extraction_lock:
                        if self._session_state_bridge_enabled:
                            self._sync_runtime_state_to_session_state(session_id, buf)
                            payload = self._session_state.to_dict(
                                session_id,
                                session_meta=deepcopy(buf.meta),
                                window_state=deepcopy(buf.window_state),
                                extraction_summary=buf.extraction_summary,
                            )
                        else:
                            # Tool-result governance needs durable runtime_extensions
                            # even when ordinary SessionState memory is disabled.
                            # Persist an intentionally empty bridge projection so
                            # task/window/commitment state cannot leak into an
                            # externalization-only session.
                            payload = {"task_state": {}, "commitments": []}
                        payload["completed_request_ids"] = list(
                            buf.completed_request_ids
                        )
                        payload["buffer_messages"] = [
                            {
                                "id": message.id,
                                "role": message.role,
                                "content": message.content,
                                "created_at": message.created_at,
                            }
                            for message in buf.messages
                        ]
                        payload["extraction_watermark"] = buf.extraction_watermark
                        payload["ingest_handoff_ids"] = sorted(buf.ingest_handoff_ids)
                        payload["compaction_prepare"] = {
                            "token": buf.compaction_prepare_token,
                            "archive_id": buf.compaction_prepare_archive_id,
                            "message_count": buf.compaction_prepare_message_count,
                            "watermark": buf.compaction_prepare_watermark,
                            "created_at": buf.compaction_prepare_token_created_at,
                        }
                        if buf.runtime_extensions:
                            payload["runtime_extensions"] = deepcopy(
                                buf.runtime_extensions
                            )
                from service.extraction_recovery_discovery import DISCOVERY_KEY, recovery_discovery_hint
                now = datetime.now(timezone.utc).isoformat()
                node = ContextNode(
                    uri=self._session_state.state_uri(ctx.account_id, session_id),
                    context_type="RESOURCE",
                    category="state",
                    level=0,
                    owner_space=f"session:{session_id}",
                    abstract="Session task state and commitments",
                    overview="Session task state and commitments",
                    content=json.dumps(payload, ensure_ascii=False, indent=2),
                    metadata={
                        "session_id": session_id,
                        "owner_user_id": ctx.user_id,
                        "owner_agent_id": ctx.agent_id,
                        DISCOVERY_KEY: recovery_discovery_hint(payload),
                        "updated_at": now,
                    },
                )
                fs.write_node(node, ctx)
                with self._lock:
                    buf.durable_state_token = (node.uri, hashlib.sha256(node.content.encode()).hexdigest())
                    self._extraction_recovery_present.add(self._recovery_owner_key(ctx))
                    self._session_state_loaded.add(session_id)
                    self._session_state_missing_until.pop(session_id, None)
            return True
        except Exception as exc:
            logger.warning("save_session_state failed session=%s: %s", session_id, exc)
            return False
        finally:
            if pinned_buffer is not None:
                pinned_buffer.end_operation()

    def _sync_runtime_state_to_session_state(
        self,
        session_id: str,
        buf: SessionBuffer,
    ) -> None:
        """Promote runtime window state into durable SessionState before save."""
        window_state = buf.window_state
        current = self._session_state.get_task_state(session_id)
        task_update = TaskState()
        if not current.objective and window_state.active_task:
            task_update.objective = window_state.active_task
        if not current.blockers and window_state.uncertainties:
            task_update.blockers = list(window_state.uncertainties)
        if task_update.objective is not None or task_update.blockers:
            self._session_state.update_task_state(session_id, task_update)

        existing = {
            (item.content, item.status, item.kind)
            for item in self._session_state.get_commitments(session_id, status=None)
        }

        def add_missing(content: str, status: str, kind: str) -> None:
            content = str(content or "").strip()
            if not content:
                return
            key = (content, status, kind)
            if key in existing:
                return
            self._session_state.add_commitment(
                session_id,
                Commitment(content=content, status=status, kind=kind),
            )
            existing.add(key)

        for content in window_state.open_loops:
            add_missing(content, "open", "loop")
        for content in window_state.confirmed_constraints:
            add_missing(content, "fulfilled", "constraint")
        for content in window_state.recent_decisions:
            add_missing(content, "fulfilled", "decision")

    @staticmethod
    def _restore_dataclass_fields(target: Any, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        for item in dataclass_fields(target):
            name = item.name
            if name not in payload:
                continue
            value = payload.get(name)
            current = getattr(target, name)
            if isinstance(current, list):
                setattr(target, name, list(value) if isinstance(value, list) else [])
            elif isinstance(current, int):
                try:
                    setattr(target, name, int(value or 0))
                except (TypeError, ValueError):
                    setattr(target, name, 0)
            elif isinstance(current, str):
                setattr(target, name, str(value or ""))
            else:
                setattr(target, name, value)

    def _restore_runtime_state_from_payload(self, session_id: str, payload: dict) -> None:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return
            if self._session_state_bridge_enabled:
                self._restore_dataclass_fields(
                    buf.meta, payload.get("session_meta") or {}
                )
                buf.meta.session_id = session_id
                self._restore_dataclass_fields(
                    buf.window_state, payload.get("window_state") or {}
                )
                extraction_summary = payload.get("extraction_summary")
                if isinstance(extraction_summary, str):
                    buf.extraction_summary = extraction_summary
            completed_request_ids = payload.get("completed_request_ids")
            if isinstance(completed_request_ids, list):
                buf.completed_request_ids = [
                    str(value)
                    for value in completed_request_ids[-256:]
                    if value and len(str(value)) <= 128
                ]
            buffer_messages = payload.get("buffer_messages")
            if isinstance(buffer_messages, list):
                restored_messages = []
                for value in buffer_messages:
                    if not isinstance(value, dict):
                        continue
                    message_id = value.get("id")
                    role = value.get("role")
                    content = value.get("content")
                    created_at = value.get("created_at")
                    if not all(
                        isinstance(item, str)
                        for item in (message_id, role, content, created_at)
                    ):
                        continue
                    restored_messages.append(SessionMessage(
                        id=message_id,
                        role=role,
                        content=content,
                        created_at=created_at,
                    ))
                buf.messages = restored_messages
                buf.turn_count = sum(1 for message in restored_messages if message.role == "user")
                buf.meta.message_count = len(restored_messages)
                try:
                    extraction_watermark = int(payload.get("extraction_watermark", 0) or 0)
                except (TypeError, ValueError):
                    extraction_watermark = 0
                buf.extraction_watermark = max(0, min(extraction_watermark, len(restored_messages)))
            handoff_ids = payload.get("ingest_handoff_ids", [])
            handoff_ids = set(handoff_ids) if isinstance(handoff_ids, list) else set()
            buf.ingest_handoff_ids = {message.id for message in buf.messages if message.id in handoff_ids}
            prepared = payload.get("compaction_prepare", {})
            buf.compaction_prepare_token = prepared.get("token", "")
            buf.compaction_prepare_archive_id = prepared.get("archive_id", "")
            buf.compaction_prepare_message_count = prepared.get("message_count", 0)
            buf.compaction_prepare_watermark = prepared.get("watermark", 0)
            buf.compaction_prepare_token_created_at = prepared.get("created_at", "")
            runtime_extensions = payload.get("runtime_extensions")
            if isinstance(runtime_extensions, dict):
                buf.runtime_extensions = deepcopy(runtime_extensions)
                pending_ids = buf.pending_extraction_ids()
                if pending_ids:
                    for index, message in enumerate(buf.messages):
                        if message.id in pending_ids:
                            buf.extraction_watermark = min(buf.extraction_watermark, index)
                            break

            if (
                self._session_state_bridge_enabled
                and self._session_state.has_state(session_id)
            ):
                try:
                    from session.session_state_bridge import apply_session_state_bridge

                    apply_session_state_bridge(
                        buf.window_state,
                        self._session_state,
                        session_id,
                        turn_count=buf.turn_count,
                        force=True,
                    )
                except Exception as exc:
                    logger.warning("restore session state bridge failed session=%s: %s", session_id, exc)

    def get_topic_buffer(self, session_id: str) -> SessionTopicBuffer:
        """Get or create the per-session topic buffer."""
        with self._lock:
            if session_id not in self._topic_buffers:
                self._topic_buffers[session_id] = SessionTopicBuffer(session_id)
            return self._topic_buffers[session_id]

    def _maybe_merge_archives(
        self,
        session_id: str,
        ctx: RequestContext,
        wait: bool = False,
    ) -> None:
        if self._get_archive_store is None:
            return
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            guard.ensure_owned()
            wait = True

        def _run() -> None:
            try:
                self._merge_archives_once(session_id, ctx)
            except Exception as exc:
                logger.warning("archive auto-merge failed session=%s: %s", session_id, exc)

        if wait:
            _run()
            return

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"archive-merge-{session_id[:8]}",
        )
        thread.start()

    def _merge_archives_once(self, session_id: str, ctx: RequestContext) -> bool:
        store = self._get_archive_store()
        if store is None:
            return False

        self._require_sql_archive_mutation(store, session_id, ctx)

        entries = store.list_archives(session_id, ctx)
        if len(entries) < 2 or len(entries) < self._archive_merge_threshold:
            return False

        sorted_entries = sorted(entries, key=lambda entry: entry.created_at or "")
        merge_count = max(0, len(sorted_entries) - self._archive_max_count + 1)
        merge_count = max(2, merge_count)
        to_merge_entries = sorted_entries[:merge_count]

        full_entries = []
        for entry in to_merge_entries:
            merged_entry = store.read_archive(session_id, entry.archive_id, ctx)
            full_entries.append(merged_entry if isinstance(merged_entry, ArchiveEntry) else entry)

        from session.archive_merger import ArchiveMerger

        merged = ArchiveMerger(max_messages=self._archive_merge_max_messages).merge(full_entries)
        if not merged.overview and not merged.messages:
            return False

        merged_archive_id = (
            f"merged_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        result = store.write_archive(
            session_id=session_id,
            overview=merged.overview,
            abstract=merged.abstract,
            messages=merged.messages,
            ctx=ctx,
            archive_id=merged_archive_id,
            metadata=merged.metadata,
        )
        if not getattr(result, "success", False):
            return False

        merged_archive_id = getattr(result, "archive_id", merged_archive_id) or merged_archive_id
        marked_archive_ids: list[str] = []
        if hasattr(store, "mark_archive_merged"):
            for entry in to_merge_entries:
                try:
                    ok = store.mark_archive_merged(
                        session_id,
                        entry.archive_id,
                        ctx,
                        merged_archive_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "archive merge mark failed session=%s archive=%s: %s",
                        session_id,
                        entry.archive_id,
                        exc,
                    )
                    ok = False
                if not ok:
                    if hasattr(store, "unmark_archive_merged"):
                        for marked_id in marked_archive_ids:
                            try:
                                store.unmark_archive_merged(
                                    session_id,
                                    marked_id,
                                    ctx,
                                    merged_archive_id,
                                )
                            except Exception:
                                logger.warning(
                                    "archive merge unmark failed session=%s archive=%s",
                                    session_id,
                                    marked_id,
                                )
                    if hasattr(store, "delete_archive"):
                        try:
                            store.delete_archive(session_id, merged_archive_id, ctx)
                        except Exception:
                            logger.warning(
                                "archive merge cleanup delete failed session=%s archive=%s",
                                session_id,
                                merged_archive_id,
                            )
                    return False
                marked_archive_ids.append(entry.archive_id)
        self.invalidate_topic_slot(session_id, "archive_history")
        return True

    @staticmethod
    def _commit_succeeded(task_info: dict) -> bool:
        return (
            task_info.get("status") == "completed"
            and task_info.get("archived", False) is True
        )

    def _update_task(self, task_info: dict, **fields) -> dict:
        with self._lock:
            task_info.update(fields)
            return dict(task_info)

    def _task_snapshot(self, task_info: dict) -> dict:
        with self._lock:
            return dict(task_info)

    # -- Buffer lifecycle ---------------------------------------------------

    def get_or_create(self, session_id: str, ctx: RequestContext | None = None) -> SessionBuffer:
        """Get existing buffer or create a new one.

        On first creation, binds identity (account_id, user_id, agent_id)
        from *ctx* into the SessionMeta so subsequent requests can inherit it.
        """
        load_event = None
        should_load = False
        wait_for_load = False
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                meta = SessionMeta(session_id=session_id)
                if ctx is not None:
                    meta.account_id = ctx.account_id
                    meta.user_id = ctx.user_id
                    meta.agent_id = ctx.agent_id
                buf = SessionBuffer(
                    session_id=session_id, meta=meta
                )
                self._sessions[session_id] = buf
                logger.debug("Created new session buffer: %s", session_id)
            elif ctx is not None:
                self._ensure_session_visible(buf, ctx)
            load_event = self._session_load_events.get(session_id)
            if load_event is not None and not load_event.is_set():
                wait_for_load = True
            elif (ctx is not None and session_id not in self._extraction_recovery_required
                  and self._cached_session_state_load(session_id) is None):
                should_load = True
                load_event = threading.Event()
                self._session_load_events[session_id] = load_event
        if should_load and load_event is not None:
            try:
                self.load_session_state(session_id, ctx)
            finally:
                with self._lock:
                    pending_event = self._session_load_events.pop(session_id, None)
                    if pending_event is not None:
                        pending_event.set()
        elif wait_for_load and load_event is not None:
            load_event.wait()
        if ctx is not None:
            self._ensure_session_visible(buf, ctx)
        return buf

    def _ensure_session_visible(self, buf: SessionBuffer, ctx: RequestContext) -> None:
        if not self._session_visible_to_ctx(buf, ctx):
            raise PermissionError("not_found_or_not_visible")

    @staticmethod
    def _session_visible_to_ctx(buf: SessionBuffer, ctx: RequestContext) -> bool:
        meta = buf.meta
        if meta.account_id and meta.account_id != ctx.account_id:
            return False
        if meta.user_id and meta.user_id != ctx.user_id:
            return False
        if meta.agent_id and meta.agent_id != ctx.agent_id:
            return False
        return True

    def get_window_state(self, session_id: str) -> SessionWindowState:
        """Get current window state for a session."""
        buf = self.get_or_create(session_id)
        with self._lock:
            return buf.window_state

    def update_window_state(
        self,
        session_id: str,
        window_state: SessionWindowState,
    ) -> None:
        """Update window state after compression."""
        buf = self.get_or_create(session_id)
        with self._lock:
            buf.window_state = window_state

    def invalidate_topic_slot(self, session_id: str, name: str) -> None:
        with self._lock:
            buffer = self._topic_buffers.get(session_id)
            if buffer is not None:
                buffer.invalidate(name)

    def list_session_working_set(self) -> list[dict]:
        with self._lock:
            rows: list[dict] = []
            for session_id, buf in self._sessions.items():
                rows.append({
                    "session_id": session_id,
                    "last_accessed_at": buf.window_state.last_accessed_at,
                    "message_count": len(buf.messages),
                    "pending_tokens": buf.pending_tokens,
                    "active_task": buf.window_state.active_task,
                })
        return sorted(
            rows,
            key=lambda row: row.get("last_accessed_at") or "",
            reverse=True,
        )

    def evict_idle_sessions(
        self,
        max_idle_seconds: int,
        now_iso: str | None = None,
        ctx: RequestContext | None = None,
    ) -> list[str]:
        if max_idle_seconds <= 0:
            return []
        now = (
            datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            if now_iso
            else datetime.now(timezone.utc)
        )
        evicted: list[str] = []
        with self._lock:
            candidates = list(self._sessions.items())
        for session_id, buf in candidates:
            if not self._is_idle_for_eviction(buf, now, max_idle_seconds):
                continue
            try:
                fs = self._get_context_fs() if self._get_context_fs else None
                session_ctx = self._session_state_context(
                    session_id, buf.meta, ctx or RequestContext(
                        buf.meta.account_id, buf.meta.user_id, buf.meta.agent_id,
                        session_id, "idle-eviction"))
                acquire = getattr(type(fs), "session_mutation_lock", None)
                scope = (fs.session_mutation_lock(self._session_state.state_uri(
                    session_ctx.account_id, session_id), session_ctx)
                    if callable(acquire) else nullcontext())
                with scope, buf.session_state_save_lock, self._lock, buf.extraction_lock:
                    if (self._sessions.get(session_id) is not buf
                            or not self._is_idle_for_eviction(buf, now, max_idle_seconds)):
                        continue
                    if callable(acquire) and not self._durable_state_matches(buf, fs, session_ctx):
                        continue
                    if fs is not None and not self.save_session_state(session_id, session_ctx):
                        continue
                    self.remove_session(session_id)
                    evicted.append(session_id)
            except Exception as exc:
                logger.warning("idle eviction deferred session=%s: %s", session_id, exc)
        return evicted

    @staticmethod
    def _durable_state_matches(buf, fs, ctx):
        if buf.durable_state_token is None:
            return False
        current_uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
        for uri in (current_uri, SessionState.legacy_state_uri(ctx.account_id, ctx.session_id)):
            try:
                node = fs.read_node(uri, ctx)
            except NodeNotFoundError:
                continue
            return (uri, hashlib.sha256(node.content.encode()).hexdigest()) == buf.durable_state_token
        return buf.durable_state_token == (current_uri, None)

    @staticmethod
    def _is_idle_for_eviction(buf, now, max_idle_seconds):
        with buf.extraction_lock:
            if (buf.active_operation_count or buf.cleaning or buf.dispose_in_progress
                    or buf.commit_in_progress or buf.extraction_in_progress
                    or len(buf.messages) > buf.extraction_watermark):
                return False
            try:
                accessed = datetime.fromisoformat(buf.window_state.last_accessed_at.replace("Z", "+00:00"))
                return (now - accessed).total_seconds() > max_idle_seconds
            except (AttributeError, TypeError, ValueError):
                return False

    def issue_compaction_prepare_token(self, session_id: str, archive_id: str = "") -> str:
        buf = self.get_or_create(session_id)
        with self._lock:
            token = uuid.uuid4().hex
            buf.compaction_prepare_token = token
            buf.compaction_prepare_archive_id = archive_id
            buf.compaction_prepare_message_count = len(buf.messages)
            buf.compaction_prepare_watermark = buf.extraction_watermark
            buf.compaction_prepare_token_created_at = datetime.now(timezone.utc).isoformat()
            return token

    def get_compaction_prepare_archive_id(self, session_id: str) -> str:
        buf = self.get_or_create(session_id)
        with self._lock:
            return buf.compaction_prepare_archive_id

    def get_compaction_prepare_state(
        self,
        session_id: str,
        ttl_seconds: int = 300,
    ) -> dict | None:
        buf = self.get_or_create(session_id)
        with self._lock:
            token = buf.compaction_prepare_token
            if not token:
                return None
            if not self._compaction_prepare_token_is_fresh(
                buf.compaction_prepare_token_created_at,
                ttl_seconds,
            ):
                self._clear_compaction_prepare_token_locked(buf)
                return None
            if (
                len(buf.messages) != buf.compaction_prepare_message_count
                or buf.extraction_watermark != buf.compaction_prepare_watermark
            ):
                self._clear_compaction_prepare_token_locked(buf)
                return None
            return {
                "prepareToken": token,
                "archive_id": buf.compaction_prepare_archive_id or "",
            }

    def consume_compaction_prepare_token(
        self,
        session_id: str,
        token: str,
        ttl_seconds: int = 300,
    ) -> bool:
        buf = self.get_or_create(session_id)
        with self._lock:
            token_matches = bool(token) and token == buf.compaction_prepare_token
            message_count_matches = len(buf.messages) == buf.compaction_prepare_message_count
            watermark_matches = buf.extraction_watermark == buf.compaction_prepare_watermark
            ttl_matches = self._compaction_prepare_token_is_fresh(
                buf.compaction_prepare_token_created_at,
                ttl_seconds,
            )
            if buf.compaction_prepare_token and not ttl_matches:
                self._clear_compaction_prepare_token_locked(buf)
                return False
            is_valid = token_matches and message_count_matches and watermark_matches and ttl_matches
            if is_valid:
                self._clear_compaction_prepare_token_locked(buf)
            return is_valid

    def clear_compaction_prepare_token(self, session_id: str) -> None:
        buf = self.get_or_create(session_id)
        with self._lock:
            self._clear_compaction_prepare_token_locked(buf)

    @staticmethod
    def _clear_compaction_prepare_token_locked(buf: SessionBuffer) -> None:
        buf._invalidate_compaction_prepare()

    @staticmethod
    def _compaction_prepare_token_is_fresh(created_at: str, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return False
        if not created_at:
            return False
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo or timezone.utc)
            return (now - created).total_seconds() < ttl_seconds
        except Exception:
            return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        ctx: RequestContext,
        created_at: str | None = None,
        message_id: str | None = None,
    ) -> dict:
        """Add a message to the session buffer."""
        while True:
            buf = self.get_or_create(session_id, ctx=ctx)
            with self._lock:
                # A concurrent dispose may remove the buffer after get_or_create()
                # returns. Never append to an object that is no longer managed.
                if self._sessions.get(session_id) is not buf:
                    continue
                existing = next(
                    (item for item in buf.messages if message_id and item.id == message_id),
                    None,
                )
                msg = buf.add(
                    role,
                    content,
                    created_at=created_at,
                    message_id=message_id,
                )
                pending_tokens = buf.pending_tokens
                result = {
                    "ok": True,
                    "message_id": msg.id,
                    "pending_tokens": pending_tokens,
                }
                if message_id:
                    result["action"] = (
                        "insert"
                        if existing is None
                        else "noop"
                        if existing.role == role and existing.content == content
                        else "replace"
                    )
                return result

    def remove_message(self, session_id: str, message_id: str, ctx: RequestContext) -> dict:
        """Remove a not-yet-extracted stable-id message."""
        buf = self.get_or_create(session_id, ctx=ctx)
        removed = buf.remove_unextracted(message_id)
        return {"ok": True, "removed": removed, "pending_tokens": buf.pending_tokens}

    def get_session(self, session_id: str, ctx: RequestContext) -> dict:
        """Return session meta + pending_tokens."""
        buf = self.get_or_create(session_id, ctx=ctx)
        with self._lock:
            return {
                "ok": True,
                "session_id": session_id,
                "pending_tokens": buf.pending_tokens,
                "message_count": buf.meta.message_count,
                "commit_count": buf.meta.commit_count,
                "last_commit_at": buf.meta.last_commit_at,
                "created_at": buf.meta.created_at,
                "updated_at": buf.meta.updated_at,
            }

    # -- Commit (archive + extract) -----------------------------------------

    def commit(
        self,
        session_id: str,
        ctx: RequestContext,
        wait: bool = False,
        skip_extraction: bool = False,
        session_time=None,
        archive_id: str | None = None,
    ) -> dict:
        """Two-phase commit.

        Phase 1 (sync): snapshot messages, clear buffer, write raw archive.
        Phase 2 (background thread): generate overview/abstract from archive.
        Returns { task_id, archived, archive_uri, status }.

        Args:
            skip_extraction: If True, skip memory extraction in phase 2
                (used when caller already performed extraction, e.g. after_turn).
            session_time: Optional datetime for temporal resolution in extraction.
        """
        sql_mutation = self._require_sql_mutation(ctx)
        if sql_mutation:
            wait = True  # A session owning connection is never handed to another thread.
        buf = self.get_or_create(session_id, ctx=ctx)

        with self._lock, buf.extraction_lock:
            if not buf.messages:
                return {"ok": True, "archived": False, "reason": "empty_buffer"}

            if buf.commit_in_progress:
                return {"ok": True, "archived": False, "reason": "commit_in_progress"}

            buf._check_source_clear_allowed()

            # Phase 1: snapshot + clear
            buf.begin_operation()
            rollback_state = buf.capture_rollback_state()
            try:
                buf.commit_in_progress = True
                self._sync_runtime_state_to_session_state(session_id, buf)
                snapshot = buf.snapshot_and_clear()

                if archive_id is None:
                    archive_id = generate_archive_id()

                task_id = f"task_{uuid.uuid4().hex[:12]}"
                task_info = {
                    "task_id": task_id,
                    "session_id": session_id,
                    "archive_id": archive_id,
                    "status": "processing",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self._tasks[task_id] = task_info
            except Exception:
                buf.commit_in_progress = False
                buf.restore_rollback_state(rollback_state)
                buf.end_operation()
                raise

        if wait:
            try:
                self._process_snapshot(
                    snapshot, session_id, archive_id, task_info, ctx
                )
            except Exception as exc:
                logger.error(
                    "commit phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            try:
                with self._lock:
                    buf.commit_in_progress = False
                    buf.meta.commit_count += 1
                    buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                    task_snapshot = dict(task_info)
                    should_save = self._commit_succeeded(task_snapshot)
                    if task_snapshot.get("status") == "failed":
                        buf.restore_rollback_state(rollback_state)
                if should_save and not self.save_session_state(session_id, ctx) and sql_mutation:
                    self._update_task(task_info, status="failed", error="session checkpoint was not acknowledged")
                    task_snapshot = dict(task_info)
            finally:
                buf.end_operation()
            return {
                "ok": True,
                "archived": task_snapshot.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_snapshot.get("status", "completed"),
                "error": task_snapshot.get("error"),
            }

        # Fire background thread
        def _commit_thread():
            try:
                self._process_snapshot(
                    snapshot, session_id, archive_id, task_info, ctx
                )
            except Exception as exc:
                logger.error(
                    "commit phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            finally:
                try:
                    with self._lock:
                        buf.commit_in_progress = False
                        buf.meta.commit_count += 1
                        buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                        task_snapshot = dict(task_info)
                        should_save = self._commit_succeeded(task_snapshot)
                        if task_snapshot.get("status") == "failed":
                            buf.restore_rollback_state(rollback_state)
                    if should_save and not self.save_session_state(session_id, ctx) and sql_mutation:
                        self._update_task(task_info, status="failed", error="session checkpoint was not acknowledged")
                finally:
                    buf.end_operation()

        try:
            t = threading.Thread(
                target=_commit_thread,
                daemon=True,
                name=f"commit-{session_id[:8]}",
            )
            t.start()
        except Exception as exc:
            with self._lock:
                buf.commit_in_progress = False
                buf.restore_rollback_state(rollback_state)
                task_info.update(status="failed", error=str(exc))
            buf.end_operation()
            raise

        return {
            "ok": True,
            "archived": False,
            "archive_id": archive_id,
            "task_id": task_id,
            "status": "processing",
        }

    def commit_snapshot(
        self,
        session_id: str,
        snapshot: list[SessionMessage],
        ctx: RequestContext,
        wait: bool = True,
        archive_id: str | None = None,
        archive_checkpoint=None,
    ) -> dict:
        """Archive a fixed message snapshot without reading/clearing live buffer."""
        sql_mutation = self._require_sql_mutation(ctx)
        if sql_mutation:
            wait = True
        if not snapshot:
            return {"ok": True, "archived": False, "reason": "empty_snapshot"}

        buf = self.get_or_create(session_id, ctx=ctx)
        with self._lock:
            buf.begin_operation()
            try:
                if archive_id is None:
                    archive_id = generate_archive_id()
                task_id = f"task_{uuid.uuid4().hex[:12]}"
                task_info = {
                    "task_id": task_id,
                    "session_id": session_id,
                    "archive_id": archive_id,
                    "status": "processing",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self._tasks[task_id] = task_info
                snapshot_copy = list(snapshot)
            except Exception:
                buf.end_operation()
                raise

        if wait:
            try:
                self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx,
                                       **({"archive_checkpoint": archive_checkpoint}
                                          if archive_checkpoint is not None else {}))
            except Exception as exc:
                logger.error(
                    "commit_snapshot phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                from service.extraction_checkpoint import RecoveryNeedsConfirmationError
                self._update_task(task_info, status="needs_confirmation" if isinstance(
                    exc, RecoveryNeedsConfirmationError) else "failed", error=str(exc))
            try:
                with self._lock:
                    buf.meta.commit_count += 1
                    buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                    task_snapshot = dict(task_info)
                    should_save = self._commit_succeeded(task_snapshot)
                if should_save and not self.save_session_state(session_id, ctx) and sql_mutation:
                    self._update_task(task_info, status="failed", error="session checkpoint was not acknowledged")
                    task_snapshot = dict(task_info)
            finally:
                buf.end_operation()
            return {
                "ok": True,
                "archived": task_snapshot.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_snapshot.get("status", "completed"),
                "error": task_snapshot.get("error"),
            }

        def _commit_snapshot_thread():
            try:
                self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx,
                                       **({"archive_checkpoint": archive_checkpoint}
                                          if archive_checkpoint is not None else {}))
            except Exception as exc:
                logger.error(
                    "commit_snapshot phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            finally:
                try:
                    with self._lock:
                        buf.meta.commit_count += 1
                        buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                        task_snapshot = dict(task_info)
                        should_save = self._commit_succeeded(task_snapshot)
                    if should_save and not self.save_session_state(session_id, ctx) and sql_mutation:
                        self._update_task(task_info, status="failed", error="session checkpoint was not acknowledged")
                finally:
                    buf.end_operation()

        try:
            t = threading.Thread(
                target=_commit_snapshot_thread,
                daemon=True,
                name=f"commit-snapshot-{session_id[:8]}",
            )
            t.start()
        except Exception as exc:
            with self._lock:
                task_info.update(status="failed", error=str(exc))
            buf.end_operation()
            raise

        return {
            "ok": True,
            "archived": False,
            "archive_id": archive_id,
            "task_id": task_id,
            "status": "processing",
        }

    def _process_snapshot(
        self,
        snapshot: list[SessionMessage],
        session_id: str,
        archive_id: str,
        task_info: dict,
        ctx: RequestContext,
        skip_extraction: bool = False,
        session_time=None,
        archive_checkpoint=None,
    ) -> None:
        """Background work: compress → write archive.

        Extraction is NOT done here — it is the caller's responsibility
        (e.g. after_turn or compact in MemoryService) to extract memories
        before calling commit().  This prevents double extraction.
        """
        messages_dicts = [
            {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at}
            for m in snapshot
        ]

        if archive_checkpoint is not None and archive_checkpoint.archive_record() is not None:
            archive_result = archive_checkpoint.archive_recover(self)
            self._update_task(task_info, archived=True, archive_uri=archive_result["uri"],
                              status="completed")
            self.invalidate_topic_slot(session_id, "archive_history")
            return

        from service.extraction_retry import source_only_reason
        recovery_reason = source_only_reason()
        if recovery_reason:
            from session.compressor import SessionCompressor
            overview, abstract = SessionCompressor(llm=None).compress(messages_dicts)
        else:
            prev_overview, prev_abstract = self._get_latest_archive_context(session_id, ctx)
            overview, abstract = self._compress(
                messages_dicts, prev_overview=prev_overview, prev_abstract=prev_abstract)
        archive_metadata: dict = {}
        if recovery_reason:
            archive_metadata.update(memory_extraction_degraded=True,
                                    memory_extraction_recovery_reason=recovery_reason[:1000])
        if self._compression_quality_enabled:
            try:
                from session.compression_quality import CompressionQualityEvaluator

                metrics = CompressionQualityEvaluator().evaluate(
                    messages_dicts,
                    overview,
                    abstract,
                )
                self._update_task(task_info, compression_quality=metrics)
                if self._compression_quality_persist_metadata:
                    archive_metadata["compression_quality"] = metrics
            except Exception as exc:
                logger.warning(
                    "compression quality evaluation failed session=%s: %s",
                    session_id,
                    exc,
                )
        if archive_checkpoint is not None:
            archive_metadata.update({
                "memory_lifecycle_source_sha256": archive_checkpoint.batch["lifecycle"]["source_digest"],
                "memory_lifecycle_archive_id": archive_id,
                "owner_user_id": ctx.user_id, "owner_agent_id": ctx.agent_id,
                "session_id": session_id,
            })
            archive_checkpoint.archive_begin({"messages": messages_dicts, "overview": overview,
                                               "abstract": abstract, "metadata": archive_metadata})
        archive_result = self._write_archive(
            session_id,
            archive_id,
            overview,
            abstract,
            messages_dicts,
            ctx,
            metadata=archive_metadata,
        )

        if archive_checkpoint is not None and archive_result.get("success", False):
            archive_checkpoint.archive_complete(archive_result.get("uri", ""))

        self._update_task(
            task_info,
            archived=archive_result.get("success", False),
            archive_uri=archive_result.get("uri", ""),
        )
        if not archive_result.get("success", False):
            self._update_task(
                task_info,
                status="failed",
                error=archive_result.get("error", "archive write failed"),
            )
            return

        self._update_task(task_info, status="completed")
        self.invalidate_topic_slot(session_id, "archive_history")
        # A lifecycle archive must stay readable until its durable source claim
        # is released. Its journal triggers auto-merge after that checkpoint.
        if archive_checkpoint is None and not recovery_reason:
            self._maybe_merge_archives(session_id, ctx)

    def read_archive_for_recovery(self, session_id, archive_id, ctx):
        """Read the exact archive through the same backend used for writes."""
        if self._get_archive_store is not None:
            store = self._get_archive_store()
        elif self._get_agfs is not None:
            fs = self._get_agfs()
            from session.archive_store import SessionArchiveStore
            store = SessionArchiveStore(fs=fs) if fs is not None else None
        else:
            store = None
        return store.read_archive(session_id, archive_id, ctx) if store is not None else None

    def _compress(
        self,
        messages: list[dict],
        prev_overview: str = "",
        prev_abstract: str = "",
    ) -> tuple[str, str]:
        """Compress messages into overview + abstract, fusing with previous archive."""
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            guard.ensure_owned()
        try:
            if self._get_llm is not None:
                llm = self._get_llm()
                if llm is not None:
                    from session.compressor import SessionCompressor
                    compressor = SessionCompressor(llm=llm)
                    return compressor.compress(
                        messages,
                        prev_overview=prev_overview,
                        prev_abstract=prev_abstract,
                    )
        except Exception as exc:
            logger.warning("LLM compress failed, using fallback: %s", exc)

        from session.compressor import SessionCompressor
        return SessionCompressor(llm=None).compress(
            messages,
            prev_overview=prev_overview,
            prev_abstract=prev_abstract,
        )

    def _write_archive(
        self,
        session_id: str,
        archive_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
        metadata: dict | None = None,
    ) -> dict:
        """Write archive via archive store if available.

        When ``archive_store_required`` is enabled, archive persistence
        failures are surfaced instead of silently degrading to an in-memory
        URI. A resolved actual SQL store is always durable: an uncertain write
        cannot be acknowledged through an in-memory fallback.
        """
        sql_archive = False
        try:
            store = None
            if self._get_archive_store is not None:
                store = self._get_archive_store()
                if store is None and self._archive_store_required:
                    return {
                        "success": False,
                        "uri": "",
                        "archive_id": archive_id,
                        "error": "archive store unavailable",
                    }
            elif self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)

            if store is not None:
                sql_archive = self._require_sql_archive_mutation(store, session_id, ctx)
                result = store.write_archive(
                    session_id=session_id,
                    overview=overview,
                    abstract=abstract,
                    messages=messages,
                    ctx=ctx,
                    archive_id=archive_id,
                    metadata=metadata or {},
                )
                if result.success:
                    return {
                        "success": True,
                        "uri": result.uri,
                        "archive_id": result.archive_id,
                    }
                logger.warning("Archive write failed: %s", result.error)
                if self._archive_store_required or sql_archive:
                    return {
                        "success": False,
                        "uri": result.uri,
                        "archive_id": archive_id,
                        "error": result.error or "archive write failed",
                    }
        except Exception as exc:
            logger.warning("Archive write failed: %s", exc, exc_info=True)
            from fs.sql_adapter.session_mutation import SessionMutationLostError
            if self._archive_store_required or sql_archive or isinstance(exc, SessionMutationLostError):
                return {
                    "success": False,
                    "uri": "",
                    "archive_id": archive_id,
                    "error": str(exc),
                }

        # In-memory fallback — archive still exists in memory
        return {
            "success": True,
            "uri": f"memory://{session_id}/{archive_id}",
            "archive_id": archive_id,
        }

    def _get_latest_archive_context(
        self, session_id: str, ctx: RequestContext,
    ) -> tuple[str, str]:
        """Fetch overview and abstract from the most recent archive for fusion.

        Returns:
            (prev_overview, prev_abstract) — empty strings if none found.
        """
        def _latest_from_entries(entries) -> tuple[str, str]:
            if not entries:
                return "", ""
            entries.sort(key=lambda e: e.created_at or "", reverse=True)
            latest = entries[0]
            return latest.overview or "", latest.abstract or ""

        if self._get_archive_store is not None:
            try:
                store = self._get_archive_store()
                if store is not None:
                    overview, abstract = _latest_from_entries(
                        store.list_archives(session_id, ctx),
                    )
                    if overview or abstract:
                        return overview, abstract
            except Exception as exc:
                logger.warning(
                    "Failed to fetch latest archive for fusion via archive store: %s",
                    exc,
                )

        if self._get_agfs is not None:
            try:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore

                    store = SessionArchiveStore(fs=agfs)
                    return _latest_from_entries(store.list_archives(session_id, ctx))
            except Exception as exc:
                logger.warning(
                    "Failed to fetch latest archive for fusion via AGFS fallback: %s",
                    exc,
                )
        return "", ""

    def _extract_memories(
        self, messages: list[dict], ctx: RequestContext,
        buf: SessionBuffer | None = None,
    ) -> dict:
        """Extract candidate memories from committed messages."""
        try:
            if self._get_write_api is not None:
                write_api = self._get_write_api()
                if write_api is not None:
                    with self._lock:
                        session_summary = buf.extraction_summary if buf else ""
                    result = write_api.commit_session(
                        messages, ctx, confidence_threshold=0.5, wait=True,
                        session_summary=session_summary,
                    )
                    # Update buffer's extraction_summary after successful extraction
                    if buf and result.get("candidates_extracted", 0) > 0:
                        new_items = [
                            f"- [{p['action']}] {p['target_uri']}"
                            for p in result.get("plans", [])
                            if p["action"] != "skip"
                        ]
                        if new_items:
                            with self._lock:
                                appended = (
                                    buf.extraction_summary + "\n" + "\n".join(new_items)
                                ).strip()
                                buf.extraction_summary = appended[-2000:]
                    return {
                        "candidates_extracted": result.get("candidates_extracted", 0),
                        "writes_completed": result.get("writes_completed", 0),
                    }
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc, exc_info=True)
        return {"candidates_extracted": 0, "writes_completed": 0}

    # -- Context assembly ---------------------------------------------------

    def get_context(
        self,
        session_id: str,
        token_budget: int,
        ctx: RequestContext,
    ) -> dict:
        """Assemble context: profile + archives + active messages."""
        buf = self.get_or_create(session_id)
        with self._lock:
            active_tokens = buf.pending_tokens
            active_message_count = len(buf.messages)

        # Collect archives
        latest_archive_overview = ""
        archive_refs: list[dict] = []

        try:
            store = None
            if self._get_archive_store is not None:
                store = self._get_archive_store()
            elif self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)

            if store is not None:
                entries = store.list_archives(session_id, ctx)
                if entries:
                    latest = entries[0]
                    latest_archive_overview = latest.overview or ""
                    for entry in entries:
                        archive_refs.append({
                            "archive_id": entry.archive_id,
                            "abstract": entry.abstract,
                        })
        except Exception as exc:
            logger.warning("get_context archive collection failed: %s", exc)

        return {
            "ok": True,
            "session_id": session_id,
            "pending_tokens": active_tokens,
            "estimatedTokens": active_tokens,
            "active_message_count": active_message_count,
            "archive_count": len(archive_refs),
            "latest_archive_overview": latest_archive_overview,
            "archives": archive_refs,
        }

    # -- Task polling -------------------------------------------------------

    def get_task(self, task_id: str) -> dict | None:
        """Poll background task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task is not None else None

    # -- Cleanup ------------------------------------------------------------

    def has_session(self, session_id: str, ctx: RequestContext | None = None) -> bool:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return False
            if ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return False
            return True

    def begin_managed_operation(
        self,
        session_id: str,
        ctx: RequestContext | None,
        *,
        create: bool = True,
    ) -> SessionBuffer | None:
        """Atomically pin the currently managed buffer against cleanup/removal."""
        if create:
            buf = self.get_or_create(session_id, ctx=ctx)
        else:
            with self._lock:
                buf = self._sessions.get(session_id)
                if buf is None:
                    return None
                if ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                    return None
        with self._lock, buf.extraction_lock:
            if self._sessions.get(session_id) is not buf:
                if buf.deleted_by_cleanup:
                    raise SessionOperationInProgress(
                        conflict_type="session_deleting",
                        retryable=False,
                    )
                raise SessionOperationInProgress(
                    conflict_type="session_replaced",
                    retryable=True,
                )
            if buf.cleaning:
                raise SessionOperationInProgress(
                    conflict_type="session_deleting",
                    retryable=False,
                )
            buf.active_operation_count += 1
            return buf

    def remove_session(self, session_id: str, ctx: RequestContext | None = None) -> None:
        """Remove session buffer (for dispose/cleanup)."""
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is not None and ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return
            removed = self._sessions.pop(session_id, None)
            self._topic_buffers.pop(session_id, None)
            # The loaded marker is valid only for the exact in-memory buffer.
            # A later recreation must reload durable runtime_extensions (Turn
            # Ledger, TTL state, and other bridges) before it can save again.
            self._session_state_loaded.discard(session_id)
            self._extraction_recovery_loaded.discard(session_id)
            self._extraction_recovery_required.discard(session_id)
            self._session_state_missing_until.pop(session_id, None)
        if removed:
            logger.debug("Removed session buffer: %s", session_id)

    def remove_session_if_empty(
        self,
        session_id: str,
        expected_buffer: SessionBuffer,
        ctx: RequestContext | None = None,
        *,
        require_quiescent: bool = False,
    ) -> bool:
        """Remove only the expected buffer when no unextracted work remains.

        A dispose background task must not remove a replacement buffer or a
        buffer that received newer messages after its snapshot was claimed.
        """
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is not expected_buffer:
                return False
            if ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return False
            with buf.extraction_lock:
                if require_quiescent and (
                    buf.messages or buf.runtime_extensions.get("memory_extraction_recovery")
                    or buf.active_operation_count or buf.dispose_in_progress
                ):
                    return False
                if (
                    buf.extraction_in_progress
                    or len(buf.messages) > buf.extraction_watermark
                ):
                    return False
                self._sessions.pop(session_id, None)
                self._topic_buffers.pop(session_id, None)
                # Do not let the replacement buffer inherit a stale "loaded"
                # claim and overwrite durable extensions with an empty state.
                self._session_state_loaded.discard(session_id)
                self._extraction_recovery_loaded.discard(session_id)
                self._extraction_recovery_required.discard(session_id)
                self._session_state_missing_until.pop(session_id, None)
        logger.debug("Removed empty session buffer: %s", session_id)
        return True

    def try_begin_session_cleanup(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> SessionBuffer | None:
        """Enter the non-waiting exclusive cleanup region or return None."""
        buf = self.get_or_create(session_id, ctx=ctx)
        with self._lock, buf.extraction_lock:
            if buf.cleaning:
                if not buf.cleanup_retry_pending:
                    return None
                if (
                    buf.active_operation_count > 0
                    or buf.extraction_in_progress
                    or buf.commit_in_progress
                    or buf.dispose_in_progress
                ):
                    return None
                # Claim the durable cleanup tombstone for one idempotent retry.
                buf.cleanup_retry_pending = False
                return buf
            if (
                buf.active_operation_count > 0
                or buf.extraction_in_progress
                or buf.commit_in_progress
                or buf.dispose_in_progress
            ):
                return None
            buf.cleaning = True
            buf.cleanup_retry_pending = False
            return buf

    def finish_session_cleanup(
        self,
        session_id: str,
        expected_buffer: SessionBuffer,
        *,
        remove: bool,
        retry_pending: bool = False,
    ) -> None:
        with self._lock, expected_buffer.extraction_lock:
            if remove and self._sessions.get(session_id) is expected_buffer:
                self._sessions.pop(session_id, None)
                self._topic_buffers.pop(session_id, None)
                self._session_state_loaded.discard(session_id)
                self._extraction_recovery_loaded.discard(session_id)
                self._extraction_recovery_required.discard(session_id)
                self._session_state.clear_session(session_id)
                meta = expected_buffer.meta
                self._extraction_recovery_present.discard(
                    (meta.account_id, meta.user_id, meta.agent_id, session_id))
                self._session_state_missing_until.pop(session_id, None)
                expected_buffer.deleted_by_cleanup = True
            expected_buffer.cleanup_retry_pending = bool(retry_pending and not remove)
            expected_buffer.cleaning = bool(retry_pending and not remove)

    def durable_session_owner(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> tuple[str, str] | None:
        """Read durable owner metadata from state first, then archives."""
        verification_errors: list[Exception] = []
        verified_owners: set[tuple[str, str]] = set()
        if self._get_context_fs is not None:
            try:
                fs = self._get_context_fs()
                if fs is not None:
                    for uri in (
                        self._session_state.state_uri(ctx.account_id, session_id),
                        self._session_state.legacy_state_uri(ctx.account_id, session_id),
                    ):
                        if hasattr(fs, "exists") and not fs.exists(uri, ctx):
                            continue
                        node = fs.read_node(uri, ctx)
                        metadata = node.metadata or {}
                        owner = (
                            str(metadata.get("owner_user_id", "")),
                            str(metadata.get("owner_agent_id", "")),
                        )
                        if all(owner):
                            verified_owners.add(owner)
                            continue
                        if any(owner):
                            raise PermissionError("invalid durable session owner metadata")
            except Exception as exc:
                from core.errors import NodeNotFoundError

                if isinstance(exc, PermissionError):
                    raise
                if not isinstance(exc, (NodeNotFoundError, FileNotFoundError)):
                    verification_errors.append(exc)
                    logger.warning(
                        "durable session owner state read failed session=%s",
                        session_id,
                    )
        if self._get_archive_store is not None:
            try:
                store = self._get_archive_store()
                if store is not None and hasattr(store, "get_session_owner"):
                    owner = store.get_session_owner(session_id, ctx)
                    if owner is not None:
                        verified_owners.add(owner)
            except Exception as exc:
                if isinstance(exc, PermissionError):
                    raise
                verification_errors.append(exc)
                logger.warning(
                    "durable session owner archive read failed session=%s",
                    session_id,
                )
        if len(verified_owners) > 1:
            raise PermissionError("conflicting durable session owners")
        if verified_owners:
            return next(iter(verified_owners))
        if verification_errors:
            raise PermissionError("durable_session_owner_unavailable") from verification_errors[0]
        return None

    def delete_session_archives(self, session_id: str, ctx: RequestContext) -> int:
        if self._get_archive_store is None:
            return 0
        store = self._get_archive_store()
        if store is None:
            return 0
        self._require_sql_archive_mutation(store, session_id, ctx)
        if hasattr(store, "delete_session_archives"):
            return int(store.delete_session_archives(session_id, ctx) or 0)
        deleted = 0
        for entry in store.list_archives(session_id, ctx):
            if hasattr(store, "delete_archive") and store.delete_archive(
                session_id, entry.archive_id, ctx,
            ):
                deleted += 1
        return deleted

    def delete_persisted_session_state(self, session_id: str, ctx: RequestContext) -> int:
        if self._get_context_fs is None:
            return 0
        fs = self._get_context_fs()
        if fs is None:
            return 0
        self._require_sql_mutation(ctx, fs)
        deleted = 0
        for uri in (
            self._session_state.state_uri(ctx.account_id, session_id),
            self._session_state.legacy_state_uri(ctx.account_id, session_id),
        ):
            try:
                if hasattr(fs, "exists") and not fs.exists(uri, ctx):
                    continue
                fs.delete_node(uri, ctx)
                deleted += 1
            except Exception as exc:
                from core.errors import NodeNotFoundError

                if isinstance(exc, (NodeNotFoundError, FileNotFoundError)):
                    continue
                raise
        return deleted

    def get_pending_session_ids(self) -> list[str]:
        """Return IDs of sessions with unextracted messages."""
        with self._lock:
            return [
                sid for sid, buf in self._sessions.items()
                if buf.extraction_watermark < len(buf.messages)
            ]
