"""MemoryService — Core oG-Memory business logic, decoupled from transport.

Used by both:
  - server/app.py        (HTTP / Flask mode)
  - bridge/memory_api.py (CLI / subprocess mode)
"""

from __future__ import annotations
from typing import Any

import importlib.metadata as _metadata
import logging
import os
import re as _re
import threading
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import time
from copy import deepcopy
from core.write_contract import (
    OUTCOMES_KEY, PUBLIC_FIELDS, error_fields, needs_projection_refresh, normalize_write_result,
    projection_outcome, store_outcome, write_request_contract,
)
from dataclasses import asdict, replace
from datetime import UTC
from datetime import datetime as _dt
from functools import wraps
from uuid import uuid4

from commit.write_receipts import WriteOutcomeUnknownError
from commit.outbox_scope import capture_outbox_events, current_outbox_events
from core.errors import (
    AccessDeniedError,
    ExtractionIncompleteError,
    LLMTransientError,
    MemoryCardFormatError,
    OperationDeadlineExceededError,
    ToolCallProtocolError,
    ValidationError,
)
from core.models import (
    ArchiveRef,
    CandidateMemory,
    ComposedContext,
    ContextNode,
    RequestContext,
    RetrievalConfig,
    Role,
    SeedHit,
    TokenBudget,
)
from core.operation_deadline import check_deadline, operation_deadline

# Chunking support
from extraction.chunking import ConversationChunker
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from providers.config import ProviderConfig
from providers.llm import get_openai_llm
from providers.token_tracker import UsagePersistenceError, request_usage_scope
from providers.unified_config import OgMemConfig, get_config
from retrieval.context_reader import ContextReader
from retrieval.evidence_selector import canonical_uri, hydrate_evidence, select_evidence
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.identity import memory_identity, memory_unit_key
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner, sanitize_query
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthorizationError, AuthService, ResolvedIdentity
from server.control_plane_store import ControlPlaneStore
from server.internal_tool_usage import InternalToolUsageStore, InternalToolUsageTracker
from server.tenant_admin import TenantAdminService
from service.extraction_checkpoint import (
    EXTENSION_KEY,
    ExtractionBatchJournal,
    RecoveryNeedsConfirmationError,
    RecoveryPersistenceError,
    recovery_owner,
    request_message_id,
)
from service.extraction_retry import (
    MAX_ATTEMPTS,
    ExtractionRetryWorker,
    is_frozen_source_retry,
    retry_failed_source,
    source_only_recovery,
)
from service.lifecycle_checkpoint import LifecycleBatchJournal, lifecycle_kind
from session.models import ArchiveEntry, SessionMessage, SessionWindowState
from session.rolling_compressor import RollingCompressor
from session.session_manager import SessionManager, SessionOperationInProgress
from session.topic_buffer import SlotContent

try:
    from pyagfs import AGFSClient

    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter import AGFSContextFS, AGFSTraceEventStore
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI, ReadAPI
    _HAS_AGFS = True
except ImportError:
    _HAS_AGFS = False

try:
    from commit.sql_outbox_store import SQLOutboxStore
    from fs.sql_adapter import SQLContextFS
    from providers.relation_store.sql_relation_store import SQLRelationStore
    _HAS_SQL = True
except ImportError:
    _HAS_SQL = False

logger = logging.getLogger("ogmem.service")


def _ambiguous_pending_replay(pending, messages, client_request_id):
    if not pending or client_request_id:
        return False
    source = {(value["role"], value["content"]) for value in pending["snapshot"]}
    return any(
        isinstance(message, dict) and not message.get("id")
        and (message.get("role", "user"),
             extract_content_and_tool_calls(message.get("content", "")).get("text", "")) in source
        for message in messages
    )


def _after_turn_deadline(function):
    """An explicit budget belongs to a synchronous caller, never a background job."""
    @wraps(function)
    def bounded(self, params):
        params = params or {}
        if "timeoutSeconds" not in params:
            return function(self, params)
        timeout = params["timeoutSeconds"]
        if timeout is None:
            raise ValueError("timeoutSeconds must be a positive finite number")
        synchronous = bool(str(params.get("clientRequestId") or params.get("client_request_id") or "").strip())
        if not synchronous and not self._param_bool(params, "wait", "wait", default=False):
            raise ValueError("timeoutSeconds requires wait=true or clientRequestId")
        try:
            with operation_deadline(timeout):
                return function(self, params)
        except OperationDeadlineExceededError as exc:
            return {"ok": False, "status": "failed", "retryable": True, "error": str(exc),
                    **error_fields(exc, str(params.get("clientRequestId") or params.get("client_request_id") or ""))}
    return bounded


def _guard_session_mutation(function):
    """Acquire ownership before reading or changing the cached session source."""
    @wraps(function)
    def guarded(self, params):
        params = params or {}
        if not params.get("sessionId") and params.get("session_id"):
            params = {**params, "sessionId": params["session_id"]}
        if function.__name__ in ("after_turn", "ingest") and not params.get("messages") and not params.get("message"):
            return function(self, params)
        if function.__name__ == "dispose" and not (params.get("sessionId") or params.get("session_id")):
            return function(self, params)
        from service.api import MemoryWriteAPI
        fs = self._write_api._fs if isinstance(self._write_api, MemoryWriteAPI) else None
        if not self._use_sql and self._write_api is None:
            manager = self.get_session_manager()
            fs = manager._get_context_fs() if callable(manager._get_context_fs) else self._get_context_fs()
            if fs is None:
                return function(self, params)
            manager._get_context_fs = lambda: fs
        if (not self._use_sql and not callable(getattr(type(fs), "session_mutation_lock", None))
                and not callable(getattr(type(fs), "session_recovery_lock", None))):
            return function(self, params)
        from fs.sql_adapter.session_mutation import SessionMutationLostError
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            with self._session_mutation_scope(ctx, refresh=function.__name__ != "delete_session") as guard:
                if (guard is not None and function.__name__ == "after_turn"
                        and callable(getattr(guard, "identity_lock", None))):
                    from service.extraction_retry import background_retry_running
                    if (not background_retry_running() and (
                            str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
                            or self._param_bool(params, "wait", "wait", default=False))):
                        guard.wait_for_identity_lock = True
                if guard is not None and function.__name__ == "commit_session_buffer":
                    params = {**params, "wait": True}
                result = function(self, {**params, "_ctx": ctx})
                if guard is not None:
                    guard.ensure_owned()
                    if (function.__name__ not in ("dispose", "delete_session")
                            and result.get("ok", True)
                            and not self.get_session_manager().save_session_state(ctx.session_id, ctx)):
                        raise RecoveryNeedsConfirmationError("session source checkpoint was not acknowledged")
            if guard is not None and guard.post_commit_drain:
                drain = self.drain_outbox_sync(account_id=ctx.account_id)
                if "drain" in result:
                    result["drain"] = drain
            return result
        except SessionMutationBusyError as exc:
            return {"ok": False, "status": "processing", "retryable": True,
                    "accepted": False, **error_fields(exc)}
        except (SessionMutationLostError, RecoveryNeedsConfirmationError) as exc:
            return {"ok": False, "status": "needs_confirmation", "retryable": False, "error": str(exc)}
    return guarded

# Fallback archive-trim safety margin used only when incoming messages cannot
# be aligned with the session buffer.  It preserves a likely in-flight
# user/assistant turn that has not reached after_turn yet.
_ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN = 2


# ---------------------------------------------------------------------------
# Helpers (stateless, reusable)
# ---------------------------------------------------------------------------

def _serialize_lazy_initialization(attribute: str):
    """Protect only the first initialization of a process-wide resource."""
    def _decorate(method):
        @wraps(method)
        def _wrapped(self, *args, **kwargs):
            if getattr(self, attribute) is not None:
                return method(self, *args, **kwargs)
            with self._init_lock:
                return method(self, *args, **kwargs)

        return _wrapped

    return _decorate


def _attribute_request_usage(method):
    """Attribute synchronous provider calls to the request tenant owner."""
    @wraps(method)
    def _wrapped(self, params=None, *args, **kwargs):
        ctx = params.get("_ctx") if isinstance(params, dict) else None
        if not isinstance(ctx, RequestContext):
            return method(self, params, *args, **kwargs)
        with request_usage_scope(ctx):
            return method(self, params, *args, **kwargs)

    return _wrapped


def _run_with_usage_scope(ctx: RequestContext, target, *args):
    with request_usage_scope(ctx):
        return target(*args)


class _DisposeOperationLease:
    """Release only this dispose call's operation and flag, exactly once."""

    def __init__(self, buffer):
        self.buffer = buffer
        self.worker_owned = False
        self._owns_dispose = False
        self._released = False

    def claim_dispose(self):
        with self.buffer.extraction_lock:
            self.buffer.dispose_in_progress = True
            self._owns_dispose = True

    def release(self, *, local=False):
        with self.buffer.extraction_lock:
            if self._released or (local and self.worker_owned):
                return
            self._released = True
            if self._owns_dispose:
                self.buffer.dispose_in_progress = False
            self.buffer.end_operation()


def extract_content_text(content) -> str:
    """Extract text from message content (string or structured blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content)


def _message_diag(messages) -> list[dict]:
    """Return compact message fingerprints for extraction diagnostics."""

    diag = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = extract_content_text(msg.get("content", ""))
            msg_id = msg.get("id", "")
            created_at = msg.get("created_at", "")
        else:
            role = getattr(msg, "role", "")
            content = extract_content_text(getattr(msg, "content", ""))
            msg_id = getattr(msg, "id", "")
            created_at = getattr(msg, "created_at", "")

        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        diag.append({
            "i": idx,
            "role": role,
            "chars": len(content),
            "sha": digest,
            "id": msg_id,
            "created_at": str(created_at) if created_at else "",
        })
    return diag


def extract_content_and_tool_calls(content) -> dict:
    """Extract text AND tool call metadata from message content.

    Returns:
        {"text": str, "tool_calls": list[dict]}
    """
    from extraction.tool_collector import parse_message_parts
    return parse_message_parts(content)


def extract_query(messages: list[dict]) -> str:
    """Build a search query from the last 3 user messages."""
    user_texts = []
    for msg in messages:
        if msg.get("role") == "user":
            text = extract_content_text(msg.get("content", ""))
            if text:
                user_texts.append(text)
    return "\n".join(user_texts[-3:]).strip()


def _message_role_content(msg) -> tuple[str, str]:
    if isinstance(msg, dict):
        role = str(msg.get("role", ""))
        content = extract_content_text(msg.get("content", ""))
    else:
        role = str(getattr(msg, "role", ""))
        content = extract_content_text(getattr(msg, "content", ""))
    return role, content


def _seed_hit_to_working_set_item(hit: SeedHit) -> dict:
    return {
        "uri": hit.uri,
        "category": hit.category or "memory",
        "abstract": hit.abstract or "",
        "overview": "",
        "content": "",
        "metadata": hit.metadata or {},
        "score": hit.score,
    }


def _retrieved_block_to_seed_hit(hit) -> SeedHit:
    metadata = {"source": "prefetch"}
    if unit_key := memory_unit_key(getattr(hit, "metadata", None)):
        metadata["unit_key"] = unit_key
    return SeedHit(
        uri=getattr(hit, "uri", "") or "",
        score=float(getattr(hit, "score", 0.0) or 0.0),
        level=2,
        category=getattr(hit, "category", "") or "memory",
        abstract=(
            getattr(hit, "abstract", None)
            or getattr(hit, "overview", None)
            or getattr(hit, "content_excerpt", None)
            or ""
        ),
        metadata=metadata,
    )


def _find_last_message_sequence_start(messages: list, sequence: list) -> int | None:
    """Find the last contiguous occurrence of sequence inside messages."""
    if not messages or not sequence or len(sequence) > len(messages):
        return None

    haystack = [_message_role_content(msg) for msg in messages]
    needle = [_message_role_content(msg) for msg in sequence]
    last_start = len(haystack) - len(needle)
    for start in range(last_start, -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def _bounded_message_count(value, max_count: int) -> int:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        count = 0
    return max(0, min(count, max_count))


def _update_extraction_summary(existing_summary: str, write_result: dict, max_chars: int = 2000) -> str:
    """Append new candidate abstracts to the extraction summary.

    Simple concatenation strategy: keep [{category}] {abstract} lines,
    truncate to max_chars (oldest entries dropped when exceeded).

    Args:
        existing_summary: Current summary text from previous extractions
        write_result: Dict from commit_session() with "plans" list
        max_chars: Maximum summary length in chars (~500 tokens)

    Returns:
        Updated summary string
    """
    plans = write_result.get("plans", [])
    new_lines = []
    for plan in plans:
        if plan.get("action") == "skip":
            continue
        uri = plan.get("target_uri", "")
        # Extract category and routing_key from URI for context
        parts = uri.rstrip("/").split("/")
        category = parts[-2] if len(parts) >= 2 else ""
        slug = parts[-1] if parts else ""
        new_lines.append(f"[{category}] {slug}")

    if not new_lines:
        return existing_summary

    combined = existing_summary + "\n" + "\n".join(new_lines) if existing_summary else "\n".join(new_lines)

    # Truncate to max_chars, keeping newest entries
    if len(combined) > max_chars:
        combined = combined[-max_chars:]

    return combined


def _positive_int(value: object, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _summary_max_chars(config: OgMemConfig, params: dict | None = None, default: int = 2000) -> int:
    """Resolve request-level summary trim overrides before falling back to config."""
    config_value = _positive_int(getattr(config, "summary_max_chars", default), default)
    request_value = None if params is None else params.get("summaryMaxChars")
    if request_value is not None:
        return _positive_int(request_value, config_value)
    return config_value


def _trim_summary(summary: str, max_chars: int) -> str:
    """Trim summary text explicitly, keeping the newest tail when oversized."""
    if not summary:
        return ""
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:]


def _short_term_index_mode(params: dict | None = None) -> str:
    """Normalize compact short-term index mode for Phase 1 backend handling."""
    raw_value = None if params is None else params.get("shortTermIndexMode")
    if raw_value is None:
        return "sync"
    normalized = str(raw_value).strip().lower()
    if normalized in {"async", "off"}:
        return normalized
    return "sync"


def _parse_session_time(created_at: object) -> object | None:
    if not created_at:
        return None
    try:
        if isinstance(created_at, str):
            return _dt.fromisoformat(created_at.replace("Z", "+00:00"))
        if isinstance(created_at, (int, float)):
            return _dt.fromtimestamp(created_at)
        return created_at
    except Exception as exc:
        logger.warning("failed to parse session_time created_at=%s: %s", created_at, exc)
        return None


_DATE_FROM_CONTENT = _re.compile(
    r"\[group chat conversation:\s*(.+?)\]", _re.IGNORECASE
)


def _extract_date_from_content(content: str):
    """Parse date from message content like '[group chat conversation: 1:56 pm on 8 May, 2023]'."""
    m = _DATE_FROM_CONTENT.search(content or "")
    if not m:
        return None
    date_str = m.group(1).strip()
    if " on " in date_str:
        date_part = date_str.rsplit(" on ", 1)[-1]
    else:
        date_part = date_str
    try:
        return _dt.strptime(date_part.strip(), "%d %B, %Y")
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Archive helpers (stateless, testable)
# ---------------------------------------------------------------------------

def build_archive_refs(
    entries: list[ArchiveEntry],
    budget: TokenBudget,
) -> tuple[list[ArchiveRef], list[ArchiveRef]]:
    """Build distance-tiered archive references within token budget.

    Sorting: newest first (by created_at descending).
    Tier 1: Latest archive gets full overview.
    Tier 2: Remaining archives get abstract only, budget-truncated
            (oldest dropped first when budget exceeded).

    Args:
        entries: List of ArchiveEntry objects from SessionArchiveStore
        budget: Token budget for assembly

    Returns:
        Tuple of (latest_archives, pre_archives)
    """
    if not entries:
        return [], []

    # Sort by created_at descending (newest first)
    sorted_entries = sorted(
        entries,
        key=lambda e: e.created_at or "",
        reverse=True,
    )

    archive_budget = budget.archive_limit

    # Tier 1: Latest archive gets full overview
    latest = sorted_entries[0]
    overview_text = latest.overview or ""
    overview_tokens = len(overview_text) // 4

    latest_ref = ArchiveRef(
        archive_id=latest.archive_id,
        archive_uri=f"archive://{latest.session_id}/{latest.archive_id}",
        abstract=latest.abstract or "",
        overview=overview_text,
        tokens=overview_tokens,
    )
    latest_refs = [latest_ref]
    used_tokens = overview_tokens

    # Tier 2: Remaining archives get abstract only
    # Budget-truncated: iterate newest→oldest, stop when budget exceeded
    # (oldest entries are naturally dropped first)
    pre_refs: list[ArchiveRef] = []
    for entry in sorted_entries[1:]:
        abstract_text = entry.abstract or ""
        abstract_tokens = len(abstract_text) // 4

        if used_tokens + abstract_tokens > archive_budget:
            break

        pre_refs.append(ArchiveRef(
            archive_id=entry.archive_id,
            archive_uri=f"archive://{entry.session_id}/{entry.archive_id}",
            abstract=abstract_text,
            overview=None,
            tokens=abstract_tokens,
        ))
        used_tokens += abstract_tokens

    return latest_refs, pre_refs


# ---------------------------------------------------------------------------
# MemoryService — warm singleton holding initialized resources
# ---------------------------------------------------------------------------

class MemoryService:
    """Transport-agnostic oG-Memory service.

    Holds lazy-initialized LLM, ReadAPI, WriteAPI instances.
    Each handler method accepts a plain dict and returns a plain dict.
    """

    def __init__(
        self, config: OgMemConfig | None = None,
        *, scene_manager=None, scene_ranker=None, scene_source=None,
    ):
        cfg = config or get_config()
        self._cfg = cfg
        self._provider_cfg = ProviderConfig.from_ogmem_config(cfg)

        self._agfs_base_url = cfg.agfs_base_url
        self._mount_prefix = cfg.agfs_mount_prefix
        self._default_account_id = cfg.account_id
        self._default_user_id = cfg.user_id
        self._default_agent_id = cfg.agent_id

        self._llm = None
        self._decision_client = None
        self._decision_client_lock = threading.Lock()
        self._write_api = None
        self._read_api = None
        self._session_mgr: SessionManager | None = None
        self._archive_store = None
        self._archive_store_failed = False
        self._sql_pool = None  # Shared pool for all SQL stores
        self._trace_event_store = None
        self._skill_sedimentation_service = None
        self._skill_sedimentation_worker = None
        self._skill_sedimentation_thread: threading.Thread | None = None

        # Shared vector index + embedder (used by both read API and outbox worker)
        self._vector_index = None
        self._projection_index = None
        self._lexical_index = None
        self._lexical_backfill = None
        self._embedder = None
        self._graph_component_cache = None
        self._model_usage_store = None
        self._outbox_thread: threading.Thread | None = None
        self._outbox_stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._extraction_retry = ExtractionRetryWorker(self)
        self._extraction_discovery = None
        self._outbox_listener = None
        self._control_store = None
        self._key_manager = None
        self._auth = None
        self._audit = None
        self._tenant_admin = None
        self._internal_tool_usage = InternalToolUsageTracker(
            max_rounds_per_session=_positive_int(
                os.environ.get("OGMEM_TOOL_USAGE_MAX_ROUNDS"),
                1000,
            )
        )
        self._internal_tool_usage_store = None
        self._pending_tool_usage_snapshots: dict[str, dict] = {}
        self._pending_tool_usage_lock = threading.Lock()
        self._retrieval_monitoring: dict[tuple[str, str], dict[str, Any]] = {}
        self._retrieval_monitoring_lock = threading.Lock()
        self._init_lock = threading.RLock()
        self._scene_manager = scene_manager
        self._scene_ranker = scene_ranker
        self._scene_source = scene_source
        self._scene_service = None
        if self._cfg.skill_sedimentation_enabled:
            self._start_skill_sedimentation_worker()


    @property
    def _use_sql(self) -> bool:
        return self._cfg.storage_backend == "sql" and _HAS_SQL and self._cfg.sql_connection_string

    def _get_shared_sql_pool(self, *, deadline=None):
        """Create the shared pool within the caller's optional acquisition budget."""
        if self._sql_pool is not None:
            return self._sql_pool
        from fs.sql_adapter.pool import SharedConnectionPool, remaining_seconds
        acquired = (self._init_lock.acquire() if deadline is None
                    else self._init_lock.acquire(timeout=remaining_seconds(deadline)))
        if not acquired:
            raise TimeoutError("SQL pool initialization deadline elapsed")
        try:
            if self._sql_pool is None and _HAS_SQL and self._cfg.sql_connection_string:
                self._sql_pool = SharedConnectionPool(
                    connection_string=self._cfg.sql_connection_string,
                    pool_size=self._cfg.sql_pool_size,
                    dialect=self._cfg.sql_dialect,
                    deadline=deadline,
                )
            return self._sql_pool
        finally:
            self._init_lock.release()

    def _get_context_fs(self):
        """Create the appropriate ContextFS implementation based on config."""
        if self._use_sql:
            return SQLContextFS(pool=self._get_shared_sql_pool())
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
        return None

    @_serialize_lazy_initialization("_model_usage_store")
    def _get_model_usage_store(self):
        if self._model_usage_store is None and self._cfg.sql_connection_string:
            from fs.sql_adapter.model_usage_store import SQLModelUsageStore
            self._model_usage_store = SQLModelUsageStore(self._get_shared_sql_pool())
        return self._model_usage_store

    def _bind_model_usage(self, provider):
        tracker = getattr(provider, 'token_tracker', None)
        if tracker is not None:
            tracker.set_usage_store(self._get_model_usage_store())
        return provider

    def _get_outbox_store(self, fs):
        """Create the appropriate OutboxStore implementation based on config."""
        if self._use_sql:
            return SQLOutboxStore(
                pool=self._get_shared_sql_pool(),
                fs=fs,
            )
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return OutboxStore(client=client, fs=fs, mount_prefix=self._mount_prefix)
        return None

    def _get_relation_store(self):
        """Create the appropriate RelationStore implementation based on config."""
        if self._use_sql:
            return SQLRelationStore(pool=self._get_shared_sql_pool(),
                edge_projection_profile_id=(self._cfg.graph_edge_embedding_profile if self._graph_enabled() else None),
                edge_projection_representation=getattr(
                    self._cfg, "graph_projection_representation", "relation-target-v2",
                ))
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return AGFSRelationStore(client=client, mount_prefix=self._mount_prefix)
        return None

    def _graph_enabled(self):
        return getattr(self._cfg,"graph_backend","none") == "sql"

    def _graph_profile(self):
        from providers.relation_store.projection_identity import (
            FACTORIZED_REPRESENTATION_VERSION, REPRESENTATION_VERSION,
        )
        embedder = self._get_shared_embedder()
        profile = dict(profile_id=self._cfg.graph_edge_embedding_profile,
                    model=getattr(embedder,"model",type(embedder).__name__),
                    dimension=int(getattr(embedder,"dimension",None) or getattr(embedder,"dimensions",None)
                                  or self._cfg.opengauss_dimension),
                    representation_version=getattr(
                        self._cfg, "graph_projection_representation", REPRESENTATION_VERSION,
                    ))
        if profile["representation_version"] == FACTORIZED_REPRESENTATION_VERSION:
            profile.update(max_input_chars=self._cfg.embedding_max_input_chars,
                           multimodal=False)
        return profile

    @_serialize_lazy_initialization("_graph_component_cache")
    def _get_graph_component_cache(self):
        """Share bounded target and relation components across projection drains."""
        if self._graph_component_cache is None:
            from index.embedding_components import EmbeddingComponentCache
            self._graph_component_cache = EmbeddingComponentCache(max_entries=1024)
        return self._graph_component_cache

    def _graph_projection_components(self):
        if not self._graph_enabled():
            return None,None
        from providers.relation_store.sql_projection import SQLEdgeProjectionStore
        from providers.relation_store.projection_identity import FACTORIZED_REPRESENTATION_VERSION
        profile = self._graph_profile()
        options = {}
        if profile["representation_version"] == FACTORIZED_REPRESENTATION_VERSION:
            options["component_cache"] = self._get_graph_component_cache()
        # Workers own their claim bookkeeping; canonical jobs are shared SQL rows.
        return SQLEdgeProjectionStore(pool=self._get_shared_sql_pool(), **options),profile

    @_serialize_lazy_initialization("_trace_event_store")
    def _get_trace_event_store(self):
        """Return the lifecycle-trace store for the configured backend."""
        if self._trace_event_store is None and self._use_sql:
            try:
                from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

                self._trace_event_store = SQLTraceEventStore(
                    pool=self._get_shared_sql_pool(),
                    max_turn_events=self._cfg.trace_max_turn_events,
                )
            except Exception as exc:
                logger.warning("SQLTraceEventStore initialization failed: %s", exc)
                return None
        elif self._trace_event_store is None and self._cfg.storage_backend == "agfs" and _HAS_AGFS:
            try:
                self._trace_event_store = AGFSTraceEventStore(
                    client=AGFSClient(api_base_url=self._agfs_base_url),
                    mount_prefix=self._mount_prefix,
                    max_turn_events=self._cfg.trace_max_turn_events,
                )
            except Exception as exc:
                logger.warning("AGFSTraceEventStore initialization failed: %s", exc)
                return None
        return self._trace_event_store

    def _get_skill_sedimentation_service(self):
        if not self._cfg.skill_sedimentation_enabled:
            return None
        if self._skill_sedimentation_service is None:
            from skills.sedimentation import SedimentationSettings, SkillSedimentationService

            trace_store = self._get_trace_event_store()
            context_fs = self._get_context_fs()
            if trace_store is None or context_fs is None:
                raise RuntimeError("skill sedimentation stores are unavailable")
            assessor_cfg = replace(self._provider_cfg, llm_temperature=0.0)
            reviewer_cfg = replace(
                self._provider_cfg,
                provider=self._cfg.skill_reviewer_provider or self._provider_cfg.provider,
                openai_llm_model=(
                    self._cfg.skill_reviewer_model or self._provider_cfg.openai_llm_model
                ),
                llm_temperature=0.0,
            )
            try:
                embedder = self._get_shared_embedder()
            except Exception as exc:
                raise RuntimeError(
                    "skill sedimentation requires semantic embeddings for safe deduplication"
                ) from exc
            self._skill_sedimentation_service = SkillSedimentationService(
                trace_store=trace_store,
                context_fs=context_fs,
                assessor_llm=self._bind_model_usage(assessor_cfg.create_llm()),
                reviewer_llm=self._bind_model_usage(reviewer_cfg.create_llm()),
                settings=SedimentationSettings.from_config(self._cfg),
                embedder=embedder,
            )
        return self._skill_sedimentation_service

    def _start_skill_sedimentation_worker(self) -> None:
        if self._skill_sedimentation_thread is not None:
            if self._skill_sedimentation_thread.is_alive():
                return
            self._skill_sedimentation_thread = None
            self._skill_sedimentation_worker = None
        from skills.sedimentation import SkillSedimentationWorker

        skill_service = self._get_skill_sedimentation_service()
        if skill_service is None:
            return
        self._skill_sedimentation_worker = SkillSedimentationWorker(
            skill_service,
            self._skill_account_ids,
            interval_seconds=self._cfg.skill_scan_interval_seconds,
            worker_id=f"skill-{uuid4().hex[:16]}",
            batch_size=self._cfg.skill_batch_size,
            account_workers=self._cfg.skill_account_workers,
        )
        self._skill_sedimentation_thread = threading.Thread(
            target=self._skill_sedimentation_worker.run,
            daemon=True,
            name="ogmemory-skill-sedimentation",
        )
        self._skill_sedimentation_thread.start()

    def _skill_account_ids(self) -> list[str]:
        try:
            trace_store = self._get_trace_event_store()
            if trace_store is None:
                return []
            return trace_store.list_pending_skill_accounts(
                limit=max(1, self._cfg.skill_account_workers * 2)
            )
        except Exception:
            logger.warning("Unable to discover accounts with pending skill work", exc_info=True)
            raise

    def wake_skill_sedimentation(self) -> None:
        """Wake the asynchronous worker; never run generation on the request thread."""

        if self._skill_sedimentation_worker is not None:
            self._skill_sedimentation_worker.wake()

    def stop_skill_sedimentation(self, timeout: float = 5.0) -> bool:
        if self._skill_sedimentation_worker is not None:
            self._skill_sedimentation_worker.stop()
        if self._skill_sedimentation_thread is not None:
            self._skill_sedimentation_thread.join(timeout=timeout)
            if self._skill_sedimentation_thread.is_alive():
                logger.warning("Skill sedimentation worker did not stop within %.1fs", timeout)
                return False
        self._skill_sedimentation_worker = None
        self._skill_sedimentation_thread = None
        return True

    def list_skill_candidates(
        self,
        ctx: RequestContext,
        *,
        state: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        if not ctx.agent_id:
            raise ValueError("agent_id is required to list skill candidates")
        store = self._get_trace_event_store()
        if store is None:
            raise RuntimeError("trace store unavailable")
        return store.list_skill_candidates(
            account_id=ctx.account_id,
            agent_id=ctx.agent_id,
            state=state,
            limit=limit,
        )

    def list_published_skills(
        self,
        ctx: RequestContext,
        *,
        limit: int = 50,
        cursor: str | None = None,
        source: str | None = None,
    ) -> dict:
        """Return one stable page from generated and managed Skill stores."""

        from skills.artifact import is_governed_skill, is_managed_skill, validate_skill_name
        from skills.management import validate_managed_skill_name

        if not ctx.agent_id:
            raise ValueError("agent_id is required to list skills")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if source not in {None, "generated", "imported"}:
            raise ValueError("source must be generated or imported")
        cursor_name = None
        cursor_source = None
        if cursor:
            if "|" in cursor:
                cursor_name, cursor_source = cursor.rsplit("|", 1)
                if cursor_source == "imported":
                    validate_managed_skill_name(cursor_name)
                elif cursor_source == "generated":
                    validate_skill_name(cursor_name)
                else:
                    raise ValueError("invalid skill cursor")
            else:
                cursor_name, cursor_source = validate_skill_name(cursor), "generated"
        context_fs = self._get_context_fs()
        rows: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()
        roots = (
            (
                "generated",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/generated",
                is_governed_skill,
            ),
            (
                "imported",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/imported",
                is_managed_skill,
            ),
            (
                "generated",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills",
                is_governed_skill,
            ),
        )
        for row_source, base_uri, published in roots:
            if source and row_source != source:
                continue
            root_rows: list[dict] = []
            target = limit + 1

            def append_node(node) -> None:
                if not published(node) or len(root_rows) >= target:
                    return
                name = node.uri.rsplit("/", 1)[-1]
                row_key = (name, row_source)
                if row_key in seen_keys or (
                    cursor_name and row_key <= (cursor_name, cursor_source)
                ):
                    return
                seen_keys.add(row_key)
                root_rows.append({
                    "name": name,
                    "description": node.metadata.get(
                        "package_description", node.abstract
                    ),
                    "source": row_source,
                    "updated_at": node.metadata.get("updated_at", ""),
                })

            after_uri = f"{base_uri}/{cursor_name}" if cursor_name else None
            if cursor_name and row_source == "imported" and cursor_source == "generated":
                same_name_uri = f"{base_uri}/{cursor_name}"
                if context_fs.exists(same_name_uri, ctx):
                    append_node(context_fs.read_node(same_name_uri, ctx))

            if hasattr(context_fs, "list_node_summaries_page"):
                while len(root_rows) < target:
                    batch_size = target - len(root_rows)
                    nodes = context_fs.list_node_summaries_page(
                        base_uri, ctx, limit=batch_size, after_uri=after_uri,
                    )
                    if not nodes:
                        break
                    for node in nodes:
                        append_node(node)
                    after_uri = nodes[-1].uri
                    if len(nodes) < batch_size:
                        break
            else:
                uris = sorted(context_fs.list_children(base_uri, ctx))
                for uri in uris:
                    if after_uri and uri <= after_uri:
                        continue
                    append_node(context_fs.read_node(uri, ctx))
                    if len(root_rows) >= target:
                        break
            rows.extend(root_rows)
        rows.sort(key=lambda row: (row["name"], row["source"]))
        page = rows[:limit]
        truncated = len(rows) > limit
        payload = {"skills": page, "truncated": truncated}
        if truncated and page:
            payload["next_cursor"] = f"{page[-1]['name']}|{page[-1]['source']}"
        return payload

    @property
    def default_agent_id(self) -> str:
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self) -> int:
        return int(self._cfg.skill_search_scan_limit)

    def read_published_skill(
        self, ctx: RequestContext, name: str, source: str | None = None
    ):
        """Read one published Skill selected by name and optional source."""

        from core.errors import NodeNotFoundError
        from skills.artifact import (
            internal_managed_skill_uri,
            internal_skill_uri,
            is_governed_skill,
            is_managed_skill,
            legacy_internal_skill_uri,
        )

        if source not in {None, "generated", "imported"}:
            raise ValueError("source must be generated or imported")
        candidates = []
        if source in {None, "generated"}:
            try:
                generated_uris = (
                    internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                    legacy_internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                )
                candidates.extend(
                    ("generated", uri, is_governed_skill) for uri in generated_uris
                )
            except ValueError:
                if source == "generated":
                    raise
        if source in {None, "imported"}:
            candidates.append((
                "imported",
                internal_managed_skill_uri(ctx.account_id, ctx.agent_id, name),
                is_managed_skill,
            ))
        found = []
        found_sources = set()
        context_fs = self._get_context_fs()
        for candidate_source, uri, published in candidates:
            try:
                node = context_fs.read_node(uri, ctx)
            except (FileNotFoundError, NodeNotFoundError):
                continue
            if published(node) and candidate_source not in found_sources:
                if candidate_source == "imported":
                    node.content = node.metadata.get("package_skill_md", node.content)
                    node.abstract = node.metadata.get("package_description", node.abstract)
                found.append(node)
                found_sources.add(candidate_source)
        if not found:
            raise FileNotFoundError("published skill not found")
        if len(found) > 1:
            raise ValueError("skill name is ambiguous; source is required")
        return found[0]

    def validate_managed_skill(
        self,
        data,
        *,
        expected_name: str | None = None,
        strict: bool = False,
        skill_dir_name: str | None = None,
        source_path: str | None = None,
    ) -> dict:
        """Validate supported skill input without writing it."""

        from skills.management import (
            package_file_manifest,
            parse_managed_skill,
            validate_skill_format,
        )

        result = validate_skill_format(
            data,
            strict=strict,
            skill_dir_name=skill_dir_name,
            source_path=source_path,
        )
        if expected_name is not None and result.get("name") != expected_name:
            raise ValueError("skill content name must match the requested name")
        if not result["valid"]:
            return result
        try:
            skill = parse_managed_skill(data)
        except ValueError as exc:
            result["valid"] = False
            result["errors"].append({
                "rule": "installable",
                "message": str(exc),
                "field": "data",
            })
            return result
        result.update({
            "content_bytes": len(skill.content.encode("utf-8")),
            "package_bytes": skill.total_bytes,
            "auxiliary_files": skill.auxiliary_files,
            "files": package_file_manifest(skill, source="imported"),
        })
        return result

    def add_managed_skill(
        self,
        ctx: RequestContext,
        data=None,
        *,
        archive_base64: str | None = None,
        path: str | None = None,
    ) -> dict:
        """Install one explicit complete Skill package."""

        from core.errors import ConcurrentModificationError
        from skills.artifact import artifact_body_hash, internal_managed_skill_uri
        from skills.management import parse_managed_skill

        self._require_skill_management_role(ctx)
        data = self._managed_skill_input(data, archive_base64, path)
        skill = parse_managed_skill(data)
        uri = internal_managed_skill_uri(ctx.account_id, ctx.agent_id, skill.name)
        fs = self._require_skill_fs()
        now = _dt.now(UTC).isoformat()
        node = ContextNode(
            uri=uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space=ctx.agent_space_name(),
            abstract=skill.description[:200],
            overview=f"Agent skill: {skill.name}",
            content=skill.content,
            metadata={
                "ogmemory_managed": True,
                "publication_status": "published",
                "content_hash": artifact_body_hash(skill.content),
                "package_skill_md": skill.content,
                "package_description": skill.description,
                "package_files": skill.serialized_files(),
                "package_bytes": skill.total_bytes,
                "created_at": now,
                "updated_at": now,
                "_expected_version": 0,
            },
        )
        try:
            fs.write_node(node, ctx)
        except ConcurrentModificationError as exc:
            raise FileExistsError(f"skill already exists: {skill.name}") from exc
        return self._managed_skill_result(node, action="create", version=1)

    def update_managed_skill(
        self,
        ctx: RequestContext,
        name: str,
        data=None,
        *,
        archive_base64: str | None = None,
        path: str | None = None,
    ) -> dict:
        """Replace one imported Skill package without touching generated skills."""

        from skills.artifact import artifact_body_hash
        from skills.management import parse_managed_skill, validate_managed_skill_name

        self._require_skill_management_role(ctx)
        data = self._managed_skill_input(data, archive_base64, path)
        name = validate_managed_skill_name(name)
        skill = parse_managed_skill(data)
        if skill.name != name:
            raise ValueError("skill content name must match the requested name")

        existing = self.read_published_skill(ctx, name, source="imported")
        version = existing.metadata.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError("imported skill has an invalid version")

        now = _dt.now(UTC).isoformat()
        node = ContextNode(
            uri=existing.uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space=ctx.agent_space_name(),
            abstract=skill.description[:200],
            overview=f"Agent skill: {skill.name}",
            content=skill.content,
            metadata={
                "ogmemory_managed": True,
                "publication_status": "published",
                "content_hash": artifact_body_hash(skill.content),
                "package_skill_md": skill.content,
                "package_description": skill.description,
                "package_files": skill.serialized_files(),
                "package_bytes": skill.total_bytes,
                "created_at": existing.metadata.get("created_at", now),
                "updated_at": now,
                "_expected_version": version,
            },
        )
        self._require_skill_fs().write_node(node, ctx)
        return self._managed_skill_result(
            node, action="update", version=version + 1
        )

    def delete_published_skill(
        self, ctx: RequestContext, name: str, source: str | None
    ) -> dict:
        """Delete one published Skill from the selected source."""

        from core.errors import NodeNotFoundError
        from skills.artifact import (
            canonical_skill_uri,
            internal_skill_uri,
            is_governed_skill,
            legacy_internal_skill_uri,
        )

        if source not in {"generated", "imported"}:
            raise ValueError("source is required and must be generated or imported")
        self._require_skill_management_role(ctx)
        node = self.read_published_skill(ctx, name, source=source)
        fs = self._require_skill_fs()
        nodes = [node]
        if source == "generated":
            for uri in (
                internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                legacy_internal_skill_uri(ctx.account_id, ctx.agent_id, name),
            ):
                if uri == node.uri:
                    continue
                try:
                    candidate = fs.read_node(uri, ctx)
                except (FileNotFoundError, NodeNotFoundError):
                    continue
                if is_governed_skill(candidate):
                    nodes.append(candidate)

        deleted = False
        for candidate in nodes:
            try:
                fs.delete_node(candidate.uri, ctx)
                deleted = True
            except (FileNotFoundError, NodeNotFoundError):
                continue
        if not deleted:
            raise FileNotFoundError("published skill not found")
        return {
            "name": name,
            "uri": canonical_skill_uri(name, source),
            "source": source,
            "action": "delete",
        }

    def read_managed_skill_file(
        self, ctx: RequestContext, name: str, path: str
    ) -> dict:
        """Read one sanitized file from an installed managed Skill package."""

        import base64

        from skills.management import (
            PackageFile,
            deserialize_package_file,
            package_file_uri,
        )

        node = self.read_published_skill(ctx, name, source="imported")
        if path == "SKILL.md":
            item = PackageFile("SKILL.md", node.content.encode("utf-8"), "text/markdown")
        else:
            item = None
            for record in node.metadata.get("package_files") or []:
                candidate = deserialize_package_file(record)
                if candidate.path == path:
                    item = candidate
                    break
            if item is None:
                raise FileNotFoundError(f"skill package file not found: {path}")
        result = {
            "name": item.path.rsplit("/", 1)[-1],
            "path": item.path,
            "uri": package_file_uri(name, item.path),
            "source": "imported",
            "media_type": item.media_type,
            "size": len(item.data),
            "sha256": item.sha256,
        }
        if b"\x00" not in item.data:
            try:
                result["content"] = item.data.decode("utf-8")
                result["encoding"] = "utf-8"
                return result
            except UnicodeDecodeError:
                pass
        result["content_base64"] = base64.b64encode(item.data).decode("ascii")
        result["encoding"] = "base64"
        return result

    def _require_skill_fs(self):
        fs = self._get_context_fs()
        if fs is None:
            raise RuntimeError("ContextFS unavailable for skill management")
        return fs

    def _require_skill_management_role(self, ctx: RequestContext) -> None:
        if not ctx.agent_id:
            raise ValueError("agent_id is required to manage skills")
        self.get_auth_service().require_role(ctx, Role.ADMIN, Role.ROOT)

    @staticmethod
    def _managed_skill_input(data, archive_base64: str | None, path: str | None):
        sources = sum(value is not None for value in (data, archive_base64, path))
        if sources != 1:
            raise ValueError("exactly one of data, archive_base64, or path is required")
        if archive_base64 is not None:
            from skills.management import decode_skill_archive_base64

            return decode_skill_archive_base64(archive_base64)
        if path is not None:
            from skills.management import load_managed_skill_path

            return load_managed_skill_path(path)
        return data

    @staticmethod
    def _managed_skill_result(node: ContextNode, *, action: str, version: int) -> dict:
        name = node.uri.rsplit("/", 1)[-1]
        uri = f"ogmemory://skills/imported/{name}"
        return {
            "status": "success",
            "name": name,
            "description": node.metadata.get("package_description", node.abstract),
            "root_uri": uri,
            "uri": uri,
            "source": "imported",
            "action": action,
            "version": version,
            "auxiliary_files": len(node.metadata.get("package_files") or []),
            "package_bytes": node.metadata.get(
                "package_bytes", len(node.content.encode("utf-8"))
            ),
        }

    def _get_chunker(self) -> ConversationChunker | None:
        """Create a ConversationChunker if chunking is enabled.

        Returns None if chunking is disabled or not configured.
        """
        if not getattr(self._cfg, "chunking_enabled", False):
            return None

        return ConversationChunker(
            llm=self.get_llm(),
            max_tokens=getattr(self._cfg, "chunk_max_segment_tokens", 8000),
            max_messages=getattr(self._cfg, "chunk_max_messages", 500),
            min_chunk_tokens=getattr(self._cfg, "chunk_min_tokens", 300),
            use_llm_boundary=getattr(self._cfg, "chunk_llm_boundary", True),
        )

    @staticmethod
    def _generate_summary_candidate(messages, session_time, session_id, llm):
        """Generate the summary candidate from a frozen source snapshot."""
        from extraction.summary_generator import SessionSummaryGenerator

        check_deadline()
        lines, participants = [], set()
        for msg in messages:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("text"))
            if content:
                lines.append(f"{role}: {content}")
                if role not in ("system", "assistant"):
                    participants.add(role)
        messages_text = "\n".join(lines)
        if not messages_text.strip():
            return None
        participants_str = ", ".join(sorted(participants))
        generator = SessionSummaryGenerator(llm=llm)
        summary = generator.generate(messages_text, session_time, participants_str)
        check_deadline()
        if summary:
            formatted = generator.format_summary(summary)
            if not formatted.strip():
                logger.info("Skipping blank L0 summary: session=%s", session_id[:8])
                return None
            return generator.build_candidate(
                summary, formatted, session_time, participants_str, session_id,
            )
        return None

    @staticmethod
    def _summary_stage_inputs(messages, session_time, session_id, llm):
        from extraction.summary_generator import SUMMARY_PROMPT, SUMMARY_SCHEMA
        from core.model_stage import model_signature

        return {"messages": deepcopy(messages), "session_id": session_id,
                "session_time": session_time.isoformat() if session_time else None,
                "prompt": SUMMARY_PROMPT, "schema": SUMMARY_SCHEMA,
                "model": model_signature(llm), "version": 1}

    @contextmanager
    def _parallel_session_summary(self, messages, session_time, ctx, session_id, write_api,
                                  candidate_checkpoint=None):
        """Join summary generation at the owner-thread write boundary."""
        from concurrent.futures import Future, ThreadPoolExecutor
        from contextvars import copy_context
        from core.model_stage import stage_collector_scope, wait_model_result

        restored = candidate_checkpoint.load() if candidate_checkpoint is not None else None
        if restored is not None:
            self._generate_and_write_summary(messages, session_time, ctx, session_id, write_api,
                                             candidate_checkpoint=candidate_checkpoint,
                                             accepted_candidates=restored[0])
            yield lambda: None
            return
        llm = self.get_llm()
        inputs = (self._summary_stage_inputs(messages, session_time, session_id, llm)
                  if candidate_checkpoint is not None else None)
        cached = (candidate_checkpoint.load_stage("session_summary", inputs)
                  if candidate_checkpoint is not None else None)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-summary") as pool:
            if cached is not None:
                future = Future()
                future.set_result(CandidateMemory(**cached["candidate"]) if cached["candidate"] else None)
            else:
                future = pool.submit(copy_context().run, self._generate_summary_candidate,
                                     deepcopy(messages), session_time, session_id, llm)
            written, saved = False, cached is not None

            def collect():
                nonlocal saved
                if (candidate_checkpoint is not None and not saved and future.done()
                        and future.exception() is None):
                    candidate = future.result()
                    candidate_checkpoint.save_stage("session_summary", inputs,
                        {"candidate": asdict(candidate) if candidate is not None else None})
                    saved = True

            def before_write():
                nonlocal written
                if not written:
                    self._generate_and_write_summary(
                        messages, session_time, ctx, session_id, write_api,
                        candidate_checkpoint=candidate_checkpoint, candidate_future=future,
                    )
                    written = True

            with stage_collector_scope(collect):
                try:
                    yield before_write
                    before_write()
                finally:
                    try:
                        wait_model_result(future)
                    except Exception:
                        if not future.done() or future.exception() is None:
                            raise
                    else:
                        collect()

    def _generate_and_write_summary(
        self,
        messages: list[dict],
        session_time,
        ctx,
        session_id: str,
        write_api,
        candidate_checkpoint=None,
        candidate_future=None,
        accepted_candidates=None,
    ) -> None:
        """Generate L0 structured summary for a session and write to storage."""
        from core.model_stage import wait_model_result

        check_deadline()
        try:
            if accepted_candidates is None and candidate_checkpoint is not None:
                restored = candidate_checkpoint.load()
                accepted_candidates = restored[0] if restored is not None else None
            if accepted_candidates is not None:
                plans = write_api._write_checkpointed_candidates(accepted_candidates, ctx, candidate_checkpoint)
                if accepted_candidates and not plans:
                    raise RuntimeError("summary rejected before writing")
                return
            candidate = (wait_model_result(candidate_future) if candidate_future is not None else
                         self._generate_summary_candidate(messages, session_time, session_id, self.get_llm()))
            check_deadline()
            if candidate is None:
                if candidate_checkpoint is not None:
                    candidate_checkpoint.accept([], 0, 0)
                return
            if candidate_checkpoint is not None:
                check_deadline()
                accepted = candidate_checkpoint.accept([candidate], 1, 0)
                plans = write_api._write_checkpointed_candidates(accepted, ctx, candidate_checkpoint)
                if not plans:
                    raise RuntimeError("summary rejected before writing")
            else:
                write_api.write_memory(candidate, ctx)
            logger.info(
                "L0 summary written: session=%s len=%d",
                session_id[:8], len(candidate.content),
            )
        except OperationDeadlineExceededError:
            raise
        except Exception as exc:
            if candidate_checkpoint is not None:
                raise
            logger.warning("L0 summary generation failed: %s", exc, exc_info=True)

    def _chunk_and_extract(
        self,
        messages: list[dict],
        write_api,
        ctx,
        session_time,
        session_summary: str,
        tool_stats_text: str,
        extraction_run_id: str,
        session_id: str,
        archive_id: str | None = None,
        extraction_journal=None,
    ) -> dict:
        """Extract memories from messages, optionally chunking first.

        When chunking is enabled, splits messages into chunks via
        ConversationChunker and calls write_api.commit_session() for each
        chunk independently.  Aggregates results from all chunks.

        When chunking is disabled (or produces a single chunk), falls back
        to a single commit_session() call — identical to the original path.
        """
        from fs.sql_adapter.session_mutation import active_session_guard
        check_deadline()
        guard = active_session_guard()
        if guard is not None:
            guard.ensure_owned()
        if isinstance(extraction_journal, LifecycleBatchJournal):
            return self._extract_lifecycle_source(write_api, ctx, session_id, extraction_journal)
        if extraction_journal is not None:
            return self._chunk_and_extract_recoverable(write_api, ctx, session_id, extraction_journal)
        with self._parallel_session_summary(messages, session_time, ctx, session_id, write_api) as before_write:
            return self._commit_session_chunks(messages, write_api, ctx, session_time, session_summary,
                                               tool_stats_text, extraction_run_id, session_id, archive_id,
                                               before_write)

    def _commit_session_chunks(self, messages, write_api, ctx, session_time, session_summary,
                               tool_stats_text, extraction_run_id, session_id, archive_id, before_write):
        chunker = self._get_chunker()

        if chunker is None:
            result = write_api.commit_session(
                messages=messages,
                ctx=ctx,
                confidence_threshold=0.5,
                wait=True,
                session_time=session_time,
                session_summary=session_summary,
                tool_stats_text=tool_stats_text,
                archive_id=archive_id,
                before_write=before_write,
            )
            # Store raw chunk text for retrieval fallback
            try:
                write_api.write_raw_chunk(
                    messages=messages, ctx=ctx, chunk_index=0,
                    session_id=session_id, session_time=session_time,
                )
            except Exception as exc:
                logger.debug("raw chunk write failed (no-chunker): %s", exc)
            return result

        chunking_result = chunker.chunk_messages(messages, flush=True)

        if not chunking_result.chunks:
            logger.info(
                "chunk_and_extract: chunker returned no chunks, run=%s session=%s",
                extraction_run_id, session_id[:8],
            )
            return {
                "candidates_extracted": 0,
                "candidates_filtered": 0,
                "writes_completed": 0,
                "writes_skipped": 0,
                "writes_failed": 0,
                "plans": [],
            }

        if len(chunking_result.chunks) == 1:
            # Single chunk — no need for multi-extract overhead
            logger.info(
                "chunk_and_extract: single chunk, run=%s session=%s msgs=%d",
                extraction_run_id, session_id[:8], len(chunking_result.chunks[0]),
            )
            result = write_api.commit_session(
                messages=chunking_result.chunks[0],
                ctx=ctx,
                confidence_threshold=0.5,
                wait=True,
                session_time=session_time,
                session_summary=session_summary,
                tool_stats_text=tool_stats_text,
                archive_id=archive_id,
                before_write=before_write,
            )
            # Store raw chunk text for retrieval fallback
            try:
                write_api.write_raw_chunk(
                    messages=chunking_result.chunks[0], ctx=ctx, chunk_index=0,
                    session_id=session_id, session_time=session_time,
                )
            except Exception as exc:
                logger.debug("raw chunk write failed (single chunk): %s", exc)
            return result

        # Multi-chunk extraction: each chunk gets its own commit_session()
        logger.info(
            "chunk_and_extract: multi-chunk extraction, run=%s session=%s "
            "chunks=%d total_msgs=%d",
            extraction_run_id, session_id[:8],
            len(chunking_result.chunks), len(messages),
        )

        aggregated = {
            "candidates_extracted": 0,
            "candidates_filtered": 0,
            "writes_completed": 0,
            "writes_skipped": 0,
            "writes_failed": 0,
            "plans": [],
        }

        for idx, chunk in enumerate(chunking_result.chunks):
            check_deadline()
            logger.info(
                "chunk_and_extract: processing chunk %d/%d, run=%s session=%s msgs=%d",
                idx + 1, len(chunking_result.chunks),
                extraction_run_id, session_id[:8], len(chunk),
            )
            try:
                chunk_result = write_api.commit_session(
                    messages=chunk,
                    ctx=ctx,
                    confidence_threshold=0.5,
                    wait=True,
                    session_time=session_time,
                    session_summary=session_summary,
                    tool_stats_text=tool_stats_text,
                    archive_id=archive_id,
                    before_write=before_write,
                )
                for key in (
                    "candidates_extracted",
                    "candidates_filtered",
                    "writes_completed",
                    "writes_skipped",
                    "writes_failed",
                ):
                    aggregated[key] += chunk_result.get(key, 0)
                aggregated["plans"].extend(chunk_result.get("plans", []))
                # Store raw chunk text for retrieval fallback
                try:
                    write_api.write_raw_chunk(
                        messages=chunk, ctx=ctx, chunk_index=idx,
                        session_id=session_id, session_time=session_time,
                    )
                except Exception as raw_exc:
                    logger.debug("raw chunk write failed (chunk %d): %s", idx, raw_exc)
            except OperationDeadlineExceededError:
                raise
            except Exception as exc:
                logger.warning(
                    "chunk_and_extract: chunk %d/%d failed, run=%s session=%s: %s",
                    idx + 1, len(chunking_result.chunks),
                    extraction_run_id, session_id[:8], exc,
                )
                aggregated["writes_failed"] += 1

        logger.info(
            "chunk_and_extract: all chunks done, run=%s session=%s "
            "chunks=%d extracted=%d writes=%d failed=%d",
            extraction_run_id, session_id[:8],
            len(chunking_result.chunks),
            aggregated["candidates_extracted"],
            aggregated["writes_completed"],
            aggregated["writes_failed"],
        )
        return aggregated

    def _extract_lifecycle_source(self, write_api, ctx, session_id, journal):
        while True:
            lifecycle = journal.batch["lifecycle"]
            reason = lifecycle.get("source_only_reason")
            if reason:
                with source_only_recovery(ctx, journal.batch["archive_id"], reason):
                    return self._chunk_and_extract_recoverable(write_api, ctx, session_id, journal)
            try:
                return self._chunk_and_extract_recoverable(write_api, ctx, session_id, journal)
            except (LLMTransientError, OperationDeadlineExceededError):
                raise
            except ExtractionIncompleteError as exc:
                failures = lifecycle.get("extraction_failures", 0) + 1
                detail = error_fields(exc, journal.batch["client_request_id"])["error_detail"]
                update = {"extraction_failures": failures, "extraction_error_detail": detail}
                if failures > MAX_ATTEMPTS:
                    update["source_only_reason"] = str(exc)[:1000] or type(exc).__name__
                journal.mutate(lambda batch: batch["lifecycle"].update(update))
                logger.warning("Lifecycle extraction recovery session=%s archive=%s attempt=%d error=%s",
                               session_id, journal.batch["archive_id"], failures, exc)

    def _chunk_and_extract_recoverable(self, write_api, ctx, session_id, journal):
        batch = journal.batch
        messages = batch["messages"]
        session_time = _parse_session_time(batch.get("session_time"))

        def make_chunks():
            chunker = self._get_chunker()
            return chunker.chunk_messages(messages, flush=True).chunks if chunker else [messages]

        chunks = journal.chunks(make_chunks)
        for index, chunk in enumerate(chunks):
            write_api.write_raw_chunk(messages=chunk, ctx=ctx, chunk_index=index,
                                     session_id=session_id, session_time=session_time,
                                     candidate_checkpoint=journal.checkpoint(f"raw:{index}"))
        aggregated = {"candidates_extracted": 0, "candidates_filtered": 0,
                      "writes_completed": 0, "writes_skipped": 0, "writes_failed": 0, "plans": []}
        from service.extraction_retry import source_only_reason
        reason = source_only_reason()
        if reason:
            for slot in list(journal.batch["slots"]):
                if slot.startswith("raw:"):
                    continue
                checkpoint = journal.checkpoint(slot)
                candidates, extracted, filtered = checkpoint.load()
                plans = write_api._write_checkpointed_candidates(candidates, ctx, checkpoint, source_only=True)
                aggregated["candidates_extracted"] += extracted
                aggregated["candidates_filtered"] += filtered
                aggregated["writes_completed"] += sum(plan.action != "skip" for plan in plans)
                aggregated["writes_skipped"] += sum(plan.action == "skip" for plan in plans)
                aggregated["plans"].extend({"action": plan.action, "target_uri": plan.target_uri,
                                             "merged_fields": plan.merged_fields} for plan in plans)
            logger.warning("Extraction completed from preserved source session=%s archive=%s reason=%s",
                           session_id, batch["archive_id"], reason)
            return {**aggregated, "degraded": True, "recovery_mode": "source_only", "reason": reason}
        with self._parallel_session_summary(messages, session_time, ctx, session_id, write_api,
                                            journal.checkpoint("summary")) as before_write:
            for index, chunk in enumerate(chunks):
                check_deadline()
                result = write_api.commit_session(
                    messages=chunk, ctx=ctx, confidence_threshold=.5, wait=True,
                    session_time=session_time, session_summary=batch["session_summary"],
                    tool_stats_text=batch["tool_stats_text"], archive_id=batch["archive_id"],
                    candidate_checkpoint=journal.checkpoint(f"chunk:{index}"), before_write=before_write)
                for key in ("candidates_extracted", "candidates_filtered", "writes_completed",
                            "writes_skipped", "writes_failed"):
                    aggregated[key] += result.get(key, 0)
                aggregated["plans"].extend(result.get("plans", []))
        return aggregated

    def _complete_extraction_source(self, ctx, archive_id, reason):
        journal = self._extraction_retry._journal(ctx, archive_id)
        if journal is None:
            return {"ok": True, "status": "completed"}
        batch = journal.batch
        with source_only_recovery(ctx, archive_id, reason):
            result = self.after_turn({
                "_ctx": ctx, "sessionId": ctx.session_id, "wait": True,
                "forceExtract": True, "timeoutSeconds": 1200,
                "clientRequestId": batch["client_request_id"],
                "messages": deepcopy(batch["request_messages"]),
            })
        if result.get("status") == "failed":
            result.setdefault("retryable", True)
        return result

    def start_extraction_recovery(self):
        from service.extraction_recovery_discovery import ExtractionRecoveryDiscovery
        with self._init_lock:
            if self._extraction_discovery is None:
                self._extraction_discovery = ExtractionRecoveryDiscovery(
                    self._get_context_fs, self._extraction_recovery_account_ids,
                    projection_handler=self._refresh_write_outcomes)
                self._extraction_retry.start_discovery(self._extraction_discovery)

    def _extraction_recovery_account_ids(self):
        accounts = [self._default_account_id]
        if self._cfg.role_control_enabled:
            accounts.extend(item["account_id"] for item in self.get_key_manager().get_accounts()
                            if item.get("account_id"))
        else:
            fs = self._get_context_fs()
            enumerate_accounts = getattr(type(fs), "recovery_account_ids", None)
            if enumerate_accounts is not None:
                accounts.extend(enumerate_accounts(fs))
        return accounts

    @staticmethod
    def _supports_batch_claim(outbox_store) -> bool:
        """Safely detect SQL outbox implementations with batch-claim support."""
        instance_values = getattr(outbox_store, "__dict__", {})
        if "supports_batch_claim" in instance_values:
            return instance_values["supports_batch_claim"] is True
        return getattr(type(outbox_store), "supports_batch_claim", False) is True

    # -- RequestContext factory ------------------------------------------------

    def build_context(
        self,
        params: dict,
        identity: ResolvedIdentity | None = None,
    ) -> RequestContext:
        """Build RequestContext, allowing per-request overrides.

        Empty strings from the plugin are treated as absent, so defaults apply.
        """
        session_id = params.get("sessionId") or params.get("session_id") or "unknown"
        explicit_agent_id = params.get("agentId")
        user_id = params.get("userId") or params.get("user_id")
        if not user_id:
            # Inherit from session meta when caller doesn't supply userId
            mgr = self.get_session_manager()
            if mgr.has_session(session_id):
                user_id = mgr.get_or_create(session_id).meta.user_id
            if not user_id:
                user_id = self._default_user_id
        account_id = params.get("accountId") or params.get("account_id") or self._default_account_id
        logger.debug("build_context userId=%s (from params: %s)", user_id, params.get("userId"))

        if identity is not None:
            requested_agent_id = explicit_agent_id
            ctx = self.get_auth_service().build_request_context(
                identity,
                account_id=account_id,
                user_id=user_id,
                agent_id=requested_agent_id or "",
                session_id=session_id,
            )
            if ctx.role == Role.ROOT:
                visible = [ctx.user_space_name()]
                if requested_agent_id:
                    visible.append(f"agent:{requested_agent_id}")
                return RequestContext(
                    account_id=ctx.account_id,
                    user_id=ctx.user_id,
                    agent_id=ctx.agent_id,
                    session_id=ctx.session_id,
                    trace_id=ctx.trace_id,
                    role=ctx.role,
                    visible_owner_spaces=tuple(dict.fromkeys(v for v in visible if v)),
                )
            visible = [ctx.user_space_name()]
            visible_agent_ids = self.get_tenant_admin_service().list_visible_agent_ids(ctx.account_id, ctx.user_id, self._cfg)
            visible.extend(f"agent:{agent_id}" for agent_id in visible_agent_ids)
            if requested_agent_id and requested_agent_id not in set(visible_agent_ids):
                raise PermissionError(f"agent access denied: {requested_agent_id}")
            return RequestContext(
                account_id=ctx.account_id,
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                session_id=ctx.session_id,
                trace_id=ctx.trace_id,
                role=ctx.role,
                visible_owner_spaces=tuple(dict.fromkeys(v for v in visible if v)),
            )

        requested_agent_id = explicit_agent_id or self._default_agent_id
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=requested_agent_id,
            session_id=session_id,
            trace_id=str(uuid4()),
            role=Role.ROOT,
            visible_owner_spaces=(),
        )

    @_serialize_lazy_initialization("_control_store")
    def get_control_plane_store(self) -> ControlPlaneStore:
        if self._control_store is None:
            if self._use_sql:
                from server.sql_control_plane_store import SQLControlPlaneStore
                self._control_store = SQLControlPlaneStore(
                    pool=self._get_shared_sql_pool(),
                )
            elif _HAS_AGFS:
                client = AGFSClient(api_base_url=self._agfs_base_url)
                self._control_store = ControlPlaneStore(
                    mount_prefix=self._mount_prefix,
                    client=client,
                )
            else:
                self._control_store = ControlPlaneStore(
                    mount_prefix=self._mount_prefix,
                    local_root=os.path.join(os.getcwd(), ".ogmem_control"),
                )
        return self._control_store

    @_serialize_lazy_initialization("_internal_tool_usage_store")
    def get_internal_tool_usage_store(self) -> InternalToolUsageStore:
        if self._internal_tool_usage_store is None:
            self._internal_tool_usage_store = InternalToolUsageStore(self.get_control_plane_store())
        return self._internal_tool_usage_store

    @_serialize_lazy_initialization("_key_manager")
    def get_key_manager(self) -> APIKeyManager:
        if self._key_manager is None:
            self._key_manager = APIKeyManager(self.get_control_plane_store())
        return self._key_manager

    @_serialize_lazy_initialization("_auth")
    def get_auth_service(self) -> AuthService:
        if self._auth is None:
            self._auth = AuthService(self._cfg, self.get_key_manager())
        return self._auth

    @_serialize_lazy_initialization("_audit")
    def get_audit_service(self) -> AuditService:
        if self._audit is None:
            self._audit = AuditService(self.get_control_plane_store())
        return self._audit

    @_serialize_lazy_initialization("_tenant_admin")
    def get_tenant_admin_service(self) -> TenantAdminService:
        if self._tenant_admin is None:
            self._tenant_admin = TenantAdminService(
                self.get_key_manager(),
                self.get_control_plane_store(),
                self.get_audit_service(),
            )
        return self._tenant_admin

    def _read_profile(self, ctx: RequestContext) -> str:
        """Read user profile L2 content directly from storage.

        Profile may be stored as a single node or as field-level child nodes
        (e.g. profile/name, profile/location, profile/identity). This method
        handles both: if the profile node has content, use it; otherwise
        enumerate children and concatenate their content.
        """
        try:
            fs = self._get_context_fs()
            if fs is None:
                return ""
            profile_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            # Try reading as a single node first
            if fs.exists(profile_uri, ctx):
                node = fs.read_node(profile_uri, ctx)
                if node.content:
                    return node.content

            # Field-level split: enumerate children and read each
            try:
                children = fs.list_children(profile_uri, ctx)
            except Exception as exc:
                logger.error("list_children failed in _read_profile: %s", exc)
                raise

            if not children:
                return ""

            parts = []
            for child_uri in children:
                if fs.exists(child_uri, ctx):
                    child_node = fs.read_node(child_uri, ctx)
                    if child_node.content:
                        # Extract field name from URI for labeling
                        field_name = child_uri.rstrip("/").split("/")[-1]
                        parts.append(f"- {field_name}: {child_node.content.strip()}")

            return "\n".join(parts) if parts else ""
        except Exception as exc:
            logger.error("_read_profile failed for %s: %s", ctx.user_id, exc, exc_info=True)
            raise

    def _collect_archives(self, ctx: RequestContext, token_budget: TokenBudget) -> tuple[list[ArchiveRef], list[ArchiveRef]]:
        """Collect session archives with distance-based graduated compression.

        Tier 1 (latest): full overview.  Tier 2 (older): abstract only.
        """
        try:
            store = self._get_archive_store()
            if store is None:
                return [], []

            entries = store.list_archives(session_id=ctx.session_id, ctx=ctx)
            return build_archive_refs(entries, token_budget)
        except Exception as exc:
            logger.warning("_collect_archives failed: %s", exc, exc_info=True)
            return [], []


    def _search_working_set(self, query: str, ctx: RequestContext) -> tuple[list[dict], list[dict], dict]:
        """Two-round vector search for memory nodes.

        Round 1: All categories → structured items for retrievedEvidence.
        Round 2: session_summary only, top_k=1 → historical retrieval evidence.

        Returns (structured_items, summary_items, stats).
        """
        items = []
        summary_items = []
        stats: dict = {}
        try:
            read_api = self.get_read_api()
            graph_kwargs = dict(include_debug=True,collect_graph_supplement=True) if self._graph_enabled() else {}

            # Round 1: structured nodes (all categories)
            if self._cfg.skill_sedimentation_enabled:
                result = self._search_without_generated_skills(
                    read_api, query=query, ctx=ctx, top_k=15,
                    fill_content_for_top_k=0,
                )
            else:
                result = read_api.search_memory(
                    query=query, ctx=ctx, top_k=15,
                    fill_content_for_top_k=5 if self._graph_enabled() else 0, mode="QUICK",
                    **graph_kwargs,
                    hints={"include_all_memories": True},
                )
            if self._graph_enabled() and result is not None and result.trace is not None:
                stats["graph_expansion"] = dict(result.trace.graph_expansion)
            if result and result.hits:
                ordinary_hits = list(result.hits)
                stats["hit_count"] = len(ordinary_hits)
                stats["hit_categories"] = list(
                    {h.category for h in ordinary_hits if h.category}
                )

                items = [
                    {
                        "uri": h.uri,
                        "abstract": h.abstract or "",
                        "overview": h.overview or "",
                        "content": h.content_excerpt or "",
                        "score": h.score,
                        "category": h.category,
                        "metadata": dict(h.metadata or {}),
                    }
                    for h in ordinary_hits
                ]
                if self._graph_enabled():
                    vector_items = [i for i in items if i["metadata"].get("retrieval_origin") != "graph_neighbor"]
                    graph_items = [i for i in items if i["metadata"].get("retrieval_origin") == "graph_neighbor"]
                    items = self._truncate_by_score_gap(vector_items) + graph_items
                else:
                    items = self._truncate_by_score_gap(items)

            # Round 2: L0 session summary for sessionContext injection
            try:
                summary_result = read_api.search_memory(
                    query=query, ctx=ctx, top_k=1,
                    categories=["session_summary"],
                    fill_content_for_top_k=0, mode="QUICK",
                )
                if summary_result and summary_result.hits:
                    summary_items = [
                        {
                            "uri": h.uri,
                            "abstract": h.abstract or "",
                            "overview": h.overview or "",
                            "content": h.content_excerpt or "",
                            "score": h.score,
                            "category": h.category,
                            "metadata": dict(h.metadata or {}),
                        }
                        for h in summary_result.hits
                    ]
                    logger.info(
                        "L0 summary search: found=%d score=%.3f",
                        len(summary_items),
                        summary_items[0]["score"] if summary_items else 0,
                    )
            except Exception as exc:
                logger.debug("L0 summary search failed: %s", exc)
        except Exception as exc:
            logger.warning("Working set search failed: %s", exc)
            return [], [], {}
        return items, summary_items, stats

    def _search_without_generated_skills(
        self,
        read_api,
        *,
        query: str,
        ctx: RequestContext,
        top_k: int,
        fill_content_for_top_k: int,
    ):
        """Search ordinary memories and legacy skills before applying top-k.

        Generated skills are consumed through the explicit skill catalog. Legacy
        or user-authored skill memories retain their pre-feature compose behavior.
        """

        search_top_k = min(
            int(getattr(self._cfg, "retrieval_max_top_k", top_k)),
            max(
                top_k,
                top_k * int(getattr(self._cfg, "retrieval_candidate_pool_multiplier", 3)),
                int(getattr(self._cfg, "retrieval_min_candidate_pool_size", top_k)),
            ),
        )
        result = read_api.search_memory(
            query=query,
            ctx=ctx,
            top_k=search_top_k,
            fill_content_for_top_k=fill_content_for_top_k,
            mode="QUICK",
            **(dict(include_debug=True,collect_graph_supplement=True) if self._graph_enabled() else {}),
            hints={"include_all_memories": True},
        )
        if result is None:
            return None
        candidates = [
            hit
            for hit in (getattr(result, "hits", []) or [])
            if (getattr(hit, "metadata", {}) or {}).get("ogmemory_generated") is not True
        ]
        graph_hits = []
        if self._graph_enabled():
            graph_hits = [h for h in candidates if h.metadata.get("retrieval_origin") == "graph_neighbor"]
            candidates = [h for h in candidates if h.metadata.get("retrieval_origin") != "graph_neighbor"]
        candidates.sort(
            key=lambda hit: float(
                getattr(hit, "ranking_score", None)
                if getattr(hit, "ranking_score", None) is not None
                else getattr(hit, "score", 0.0)
            ),
            reverse=True,
        )
        deduped = {}
        for hit in candidates:
            deduped.setdefault(memory_identity(hit.uri, getattr(hit, "metadata", None)), hit)
        result.hits = list(deduped.values())[:top_k]
        seen = set(deduped)
        for hit in graph_hits:
            identity = memory_identity(hit.uri, getattr(hit, "metadata", None))
            if identity not in seen:
                result.hits.append(hit)
                seen.add(identity)
        return result

    @staticmethod
    def _truncate_by_score_gap(items: list[dict], min_keep: int = 3, gap_ratio: float = 0.10) -> list[dict]:
        """Truncate items at the largest score gap.

        Finds the biggest relative score drop between consecutive items.
        Keeps at least min_keep items. If no significant gap, returns all.
        """
        if len(items) <= min_keep:
            return items

        scores = [it["score"] for it in items]
        max_gap_idx = min_keep - 1
        max_gap = 0.0

        for i in range(min_keep - 1, len(scores) - 1):
            if scores[i] > 0:
                gap = (scores[i] - scores[i + 1]) / scores[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i

        if max_gap >= gap_ratio:
            return items[: max_gap_idx + 1]

        return items


    # -- System prompt builder (KV-cache friendly) ----------------------------

    def _format_profile(
        self,
        profile: str,
        budget: dict[str, int],
    ) -> str:
        """Build stable identity context (Profile only) → systemPromptAddition.

        This is the MOST stable layer — changes only when user profile updates.
        Fully KV-cacheable across all turns and sessions.
        """
        if not profile:
            return ""

        max_tokens = budget.get("identity", 5000)
        text = f"## Profile\n{profile}"
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _format_archives(
        self,
        latest_archives: list[ArchiveRef],
        pre_archives: list[ArchiveRef],
        budget: dict[str, int],
    ) -> str:
        """Build episodic history context (Archives only) → systemPromptSuffix prefix.

        Goes into systemPromptSuffix BEFORE session state. Semi-stable:
        only changes when sessions are archived (not every turn).
        """
        max_tokens = budget.get("archive", 40000)

        archive_parts: list[str] = []
        if latest_archives:
            latest = latest_archives[0]
            if latest.overview:
                archive_parts.append(f"### Latest Session\n{latest.overview}")
        if pre_archives:
            lines = ["### Previous Sessions"]
            for ref in pre_archives:
                lines.append(f"- {ref.archive_id}: {ref.abstract}")
            archive_parts.append("\n".join(lines))

        if not archive_parts:
            return ""

        text = "## Archive History\n" + "\n".join(archive_parts)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _format_working_set(
        self, working_set: list[dict], budget: dict[str, int], *,
        query: str = "", delivery_stats: dict | None = None,
    ) -> str:
        """Build DYNAMIC working set as user message content.

        This goes into memoryUserMessage and changes every turn.
        Placed in user message so KV cache for stable system prompt
        prefix is preserved across turns.

        Args:
            working_set: List of retrieved memory items
            budget: dict allocation from allocate()
        """
        if not working_set:
            return ""
        ws_lines = [
            "## Retrieved Memories",
            "The following memories were retrieved from the user's long-term memory. "
            "Use them to answer the user's question. If the answer is found in these memories, respond based on them.",
            "",
        ]
        graph_items = [item for item in working_set
                       if (item.get("metadata") or {}).get("retrieval_origin") == "graph_neighbor"]
        ordinary_items = [item for item in working_set
                          if (item.get("metadata") or {}).get("retrieval_origin") != "graph_neighbor"]
        working_set_budget = budget.get("working_set", 20000)
        max_chars = max(0, working_set_budget * 4)
        remaining = max(0, max_chars - len("\n".join(ws_lines)) - 1)
        evidence, selection_stats = select_evidence(
            ordinary_items, query, remaining,
            embedder_factory=self._get_shared_embedder,
        )
        if delivery_stats is not None:
            delivery_stats.update(selection_stats)
        if evidence:
            ws_lines.append(evidence)
        elif not graph_items:
            return ""
        ws_text = "\n".join(ws_lines)
        if len(ws_text) > max_chars and working_set_budget > 100:
            ws_text = ws_text[:max_chars]
        # Reuse dev's source-preserving renderer, but graph additions must not
        # consume its original 15-candidate quota or rewrite its evidence prefix.
        for item in graph_items:
            body_budget = sum(len(str(item.get(k) or "")) for k in
                              ("uri", "abstract", "overview", "content", "category")) + 4096
            evidence, graph_stats = select_evidence([item], "", body_budget)
            addition = "\n\n" + evidence if evidence else ""
            if addition and len(ws_text) + len(addition) <= max_chars:
                ws_text += addition
                if delivery_stats is not None:
                    delivery_stats.setdefault("delivered_uris", []).extend(graph_stats["delivered_uris"])
        return ws_text

    def _get_session_state(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> SessionWindowState:
        """Get or compress session window state (Layer 2)."""
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)

        mgr.load_session_state(session_id, ctx)
        window_state = buf.window_state
        window_state.last_accessed_at = _dt.now(UTC).isoformat()
        self._sync_session_state_bridge_if_due(session_id, buf, window_state)
        mgr.update_window_state(session_id, window_state)

        if not getattr(self._cfg, "rolling_compress_enabled", False):
            return window_state

        if not buf.should_compress():
            return window_state

        # Need to compress — use RollingCompressor
        try:
            compressor = RollingCompressor(
                llm=self.get_llm(),
                fallback_enabled=getattr(
                    self._cfg,
                    "rolling_compress_fallback_enabled",
                    False,
                ),
            )
            window_state = compressor.compress(
                buf.messages,
                window_state,
                session_state=(
                    mgr.get_session_state()
                    if getattr(self._cfg, "session_state_bridge_enabled", True)
                    else None
                ),
                session_id=session_id,
            )
            mgr.update_window_state(session_id, window_state)
            logger.info(
                "Session window compressed: session=%s turns=%d tokens=%d",
                session_id,
                window_state.turn_count_at_last_compress,
                window_state.token_count_at_last_compress,
            )
        except Exception as exc:
            logger.warning("Session window compression failed: %s", exc)

        return window_state

    def _sync_session_state_bridge_if_due(
        self,
        session_id: str,
        buf,
        window_state: SessionWindowState,
    ) -> bool:
        if not getattr(self._cfg, "session_state_bridge_enabled", True):
            return False
        try:
            session_state = self.get_session_manager().get_session_state()
            version = session_state.get_version(session_id)
            if version == window_state.session_state_version:
                return False

            interval = _positive_int(
                getattr(self._cfg, "session_state_sync_interval_turns", 1),
                1,
            )
            synced_turn_count = getattr(window_state, "session_state_sync_turn_count", 0)
            if synced_turn_count > 0 and (buf.turn_count - synced_turn_count) < interval:
                return False

            from session.session_state_bridge import apply_session_state_bridge

            return apply_session_state_bridge(
                window_state,
                session_state,
                session_id,
                turn_count=buf.turn_count,
            )
        except Exception as exc:
            logger.warning("SessionState bridge sync failed: %s", exc)
            return False

    def _format_session_state(
        self,
        window_state: SessionWindowState,
        budget: dict[str, int],
    ) -> str:
        """Build Layer 2 session state suffix for system prompt.

        Goes into systemPromptSuffix — updated atomically every N turns.
        Placed at end of system prompt so KV cache for stable prefix
        (Layer 1) is preserved; only this suffix chunk changes.

        Args:
            window_state: Current session window state
            budget: dict allocation from allocate()
        """
        sections: list[str] = []

        if window_state.active_task:
            sections.append(f"## Active Task\n{window_state.active_task}")

        if window_state.skills_text:
            sections.append(f"## Skills Summary\n{window_state.skills_text}")

        if window_state.confirmed_constraints:
            items = "\n".join(f"- {c}" for c in window_state.confirmed_constraints[:5])
            sections.append(f"## Confirmed Constraints\n{items}")

        if window_state.recent_decisions:
            items = "\n".join(f"- {d}" for d in window_state.recent_decisions[:5])
            sections.append(f"## Recent Decisions\n{items}")

        if window_state.open_loops:
            items = "\n".join(f"- {loop}" for loop in window_state.open_loops[:5])
            sections.append(f"## Open Loops\n{items}")

        if window_state.compressed_text:
            sections.append(f"## Recent Session Summary\n{window_state.compressed_text}")

        if not sections:
            return ""

        combined = "\n\n".join(sections)

        session_state_budget = budget.get("session_state", 10000)

        # Truncate to budget
        max_chars = session_state_budget * 4
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        return combined

    # -- Token estimation -----------------------------------------------------

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a list of messages.

        Uses CJK-aware estimation: ~1.5 chars/token for CJK, ~4 chars/token for English.
        """
        total_chars = 0
        cjk_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
                cjk_chars += sum(1 for c in content if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
            elif isinstance(content, list):
                for block in content:
                    text = block.get("text", "") if isinstance(block, dict) else (block if isinstance(block, str) else "")
                    total_chars += len(text)
                    cjk_chars += sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        # CJK: ~1.5 chars/token, English: ~4 chars/token
        cjk_tokens = int(cjk_chars / 1.5)
        en_tokens = (total_chars - cjk_chars) // 4
        return cjk_tokens + en_tokens

    # -- Lazy resource getters ------------------------------------------------

    @_serialize_lazy_initialization("_vector_index")
    def _get_shared_vector_index(self):
        """Lazy-init shared vector index (same instance for read + outbox worker)."""
        if self._vector_index is None:
            # Use the same conservative identity as the session mutation guard.
            from fs.sql_adapter.pool import sql_connection_identity
            pool = None
            vector_dsn = getattr(self._provider_cfg, "opengauss_connection_string", None)
            if (self._use_sql
                    and str(self._provider_cfg.vector_db_type).strip().casefold() == "opengauss"
                    and self._cfg.sql_dialect in {"postgresql", "opengauss"}
                    and vector_dsn
                    and sql_connection_identity(vector_dsn) == sql_connection_identity(self._cfg.sql_connection_string)):
                pool = self._get_shared_sql_pool()
            vector_index = (self._provider_cfg.create_vector_index(pool=pool) if pool is not None
                            else self._provider_cfg.create_vector_index())
            projection_index = vector_index
            lexical_index = None
            if getattr(self._cfg, "retrieval_hybrid_enabled", False):
                try:
                    from core.lexical import HybridLexicalIndex
                    from providers.composite_index import CompositeProjectionWriter
                    from providers.lexical_index import create_lexical_index

                    lexical_index = create_lexical_index(self._cfg)
                    if not isinstance(lexical_index, HybridLexicalIndex):
                        raise TypeError(
                            "hybrid lexical provider lacks durable checkpoint capabilities"
                        )
                    projection_index = CompositeProjectionWriter(
                        vector_index,
                        lexical_index,
                    )
                    from core.lexical import lexical_scopes_from_index_records

                    projection_index.set_failure_callbacks(
                        records_failed=lambda records: lexical_index.mark_scopes_dirty(
                            lexical_scopes_from_index_records(records)
                        ),
                        scope_unknown_failed=lexical_index.mark_all_scopes_dirty,
                    )
                except Exception:
                    for resource in (lexical_index, vector_index):
                        close = getattr(resource, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                logger.warning(
                                    "Resource cleanup failed after lexical initialization error",
                                    exc_info=True,
                                )
                    raise
            # Publish the resource tree only after required lexical
            # initialization succeeds, so transient failures remain retryable.
            self._vector_index = vector_index
            self._projection_index = projection_index
            self._lexical_index = lexical_index
            logger.info("Shared vector_index ready: %s", type(self._vector_index).__name__)
        return self._vector_index

    def _get_shared_projection_index(self):
        """Return the outbox sink that keeps vector and lexical projections aligned."""
        self._get_shared_vector_index()
        return self._projection_index

    @_serialize_lazy_initialization("_lexical_backfill")
    def _get_lexical_backfill(self):
        """Return the lifecycle-owned lexical backfill coordinator."""
        if not getattr(self._cfg, "retrieval_hybrid_enabled", False):
            return None
        if self._lexical_backfill is None:
            self._get_shared_vector_index()
            if self._lexical_index is None:
                raise RuntimeError("hybrid retrieval requires a lexical index")
            from fs.sql_adapter.node_scanner import SQLCanonicalNodeScanner
            from providers.lexical_backfill import LexicalBackfill

            self._lexical_backfill = LexicalBackfill(
                SQLCanonicalNodeScanner(self._get_shared_sql_pool()),
                self._lexical_index,
                page_size=getattr(self._cfg, "retrieval_backfill_page_size", 500),
            )
        return self._lexical_backfill

    @_serialize_lazy_initialization("_embedder")
    def _get_shared_embedder(self):
        """Lazy-init shared embedder (same instance for read + outbox worker)."""
        if self._embedder is None:
            self._embedder = self._bind_model_usage(self._provider_cfg.create_embedder())
            logger.info("Shared embedder ready: %s", type(self._embedder).__name__)
        return self._embedder

    @_serialize_lazy_initialization("_outbox_thread")
    def _start_outbox_worker(self):
        """Start OutboxWorker background thread (once, idempotent)."""
        if self._outbox_thread is not None or self._outbox_stop.is_set():
            return
        if not _HAS_AGFS and not _HAS_SQL:
            logger.info("OutboxWorker skipped: no storage backend available")
            return
        try:
            from index.outbox_worker import OutboxWorker

            vector_index = self._get_shared_projection_index()
            embedder = self._get_shared_embedder()

            fs = self._get_context_fs()
            if fs is None:
                return
            outbox_store = self._get_outbox_store(fs)
            if outbox_store is None:
                return

            directory_summary_enabled = bool(
                getattr(self._cfg, "directory_summary_enabled", False)
            )
            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=fs,
                llm=self.get_llm() if directory_summary_enabled else None,
                directory_summary_enabled=directory_summary_enabled,
                agent_memory_handler=self._process_agent_memory,
                edge_projection_components=self._graph_projection_components(),
                memory_projection_handler=self._scene_projection_changed
                    if self._cfg.scene_enabled and self._cfg.scene_generation_enabled else None,
            )

            # SQL backend: use notify-based listener for real-time processing
            if self._supports_batch_claim(outbox_store):
                from index.sql_notify_listener import SQLNotifyListener

                worker_id = f"sql-listener-{uuid4().hex[:8]}"
                listener = SQLNotifyListener(
                    outbox_store=outbox_store,
                    worker=worker,
                    worker_id=worker_id,
                )
                self._outbox_listener = listener
                self._outbox_thread = threading.Thread(
                    target=listener.run_forever,
                    daemon=True,
                    name="outbox-listener",
                )
                self._outbox_thread.start()
                logger.info("OutboxWorker SQL listener thread launched")
                return

            # AGFS backend: polling loop with multi-account support
            def _get_account_ids_to_scan():
                if self._cfg.role_control_enabled:
                    try:
                        accounts = self.get_key_manager().get_accounts()
                        return [a["account_id"] for a in accounts if a.get("account_id")]
                    except Exception as exc:
                        logger.warning("Failed to list accounts for outbox scan: %s", exc)
                return [self._default_account_id]

            def _outbox_loop():
                logger.info("OutboxWorker thread started (idle polling every 5s)")
                while not self._outbox_stop.is_set():
                    wait_seconds = 5
                    try:
                        account_ids = _get_account_ids_to_scan()
                        stats = worker.run_once(
                            outbox_store=outbox_store,
                            account_ids=account_ids,
                        )
                        progressed = any(stats.get(key, 0) > 0 for key in (
                            "succeeded", "edge_projections",
                        ))
                        failed = any(stats.get(key, 0) > 0 for key in (
                            "failed", "edge_projection_failures", "moved_to_dlq",
                        ))
                        # Drain further work after progress; wait when idle or backing off.
                        if progressed and not failed:
                            continue
                        future_delay = getattr(outbox_store, "future_pending_delay", None)
                        if not failed and callable(future_delay):
                            delay = future_delay(account_ids)
                            if delay is not None and delay > 0:
                                wait_seconds = min(wait_seconds, delay)
                    except Exception as exc:
                        logger.warning("OutboxWorker poll error: %s", exc)
                    self._outbox_stop.wait(wait_seconds)

            self._outbox_thread = threading.Thread(
                target=_outbox_loop, daemon=True, name="outbox-worker"
            )
            self._outbox_thread.start()
            logger.info("OutboxWorker background thread launched")
        except Exception as exc:
            logger.warning("Failed to start OutboxWorker: %s", exc, exc_info=True)

    def _async_drain(self, account_id: str | None = None):
        """Background thread wrapper for outbox drain with logging."""
        try:
            drain_stats = self.drain_outbox_sync(account_id=account_id)
            logger.info(
                "outbox_drain processed=%d succeeded=%d failed=%d",
                drain_stats.get("processed", 0),
                drain_stats.get("succeeded", 0),
                drain_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.error("outbox_drain failed in background: %s", exc, exc_info=True)

    def drain_outbox_sync(self, account_id: str | None = None) -> dict:
        """Synchronously process pending OutboxEvents: embed → upsert to vector index.

        In subprocess mode the background OutboxWorker thread dies with the process,
        so we must drain inline before returning to ensure data is indexed.
        ChromaDB persists to disk, so the next subprocess call will find the data.
        """
        if not _HAS_AGFS and not _HAS_SQL:
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        try:
            from fs.sql_adapter.session_mutation import active_session_guard
            guard = active_session_guard()
            if guard is not None:
                guard.ensure_owned()
                if account_id is not None and account_id != guard.ctx.account_id:
                    guard.poison("projection request changed its mutation account")
                    guard.check_active()
                guard.post_commit_drain = True
                return {"processed": 0, "succeeded": 0, "failed": 0, "deferred": True}
            from index.outbox_worker import OutboxWorker

            vector_index = self._get_shared_projection_index()
            embedder = self._get_shared_embedder()

            fs = self._get_context_fs()
            if fs is None:
                return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
            outbox_store = self._get_outbox_store(fs)
            if outbox_store is None:
                return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

            directory_summary_enabled = bool(
                getattr(self._cfg, "directory_summary_enabled", False)
            )
            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=fs,
                llm=self.get_llm() if directory_summary_enabled else None,
                directory_summary_enabled=directory_summary_enabled,
                agent_memory_handler=self._process_agent_memory,
                edge_projection_components=self._graph_projection_components(),
                memory_projection_handler=self._scene_projection_changed
                    if self._cfg.scene_enabled and self._cfg.scene_generation_enabled else None,
            )

            account_ids = []
            if self._graph_enabled():
                account_ids = [account_id or self._default_account_id]
            if not self._supports_batch_claim(outbox_store):
                effective_account_id = account_id or self._default_account_id
                account_ids = [effective_account_id]

            targeted = {}
            references = current_outbox_events(account_id)
            if references is not None and callable(getattr(type(outbox_store), "list_pending_events", None)):
                targeted["pending_events"] = outbox_store.list_pending_events(references)
            stats = worker.run_once(
                outbox_store=outbox_store,
                account_ids=account_ids,
                **targeted,
            )
            logger.info("drain_outbox_sync: %s", stats)
            return stats
        except Exception as exc:
            logger.warning("drain_outbox_sync failed: %s", exc)
            return {"processed": 0, "succeeded": 0, "failed": 1, "error": str(exc)}

    def shutdown(self, join_timeout: float = 1.0) -> None:
        """Stop the outbox worker before closing its shared projections."""
        with self._shutdown_lock:
            with self._init_lock:
                self._outbox_stop.set()
            self._shutdown_resources(join_timeout)

    def _shutdown_resources(self, join_timeout: float) -> None:
        if self._scene_manager is not None and not self._scene_manager.stop(timeout=join_timeout):
            logger.warning("Scene worker is completing its current request")
            return
        self._extraction_retry.close(join_timeout)
        scene_index = getattr(self._scene_source, "candidate_index", None)
        if scene_index is not None and not scene_index.close(timeout=join_timeout):
            logger.warning("Scene lexical index is still closing")
            return
        self.stop_skill_sedimentation(timeout=join_timeout)
        if self._outbox_listener is not None:
            try:
                self._outbox_listener.stop()
            except Exception as exc:
                logger.warning("Failed to stop outbox listener: %s", exc)

        thread = self._outbox_thread
        if (
            thread is not None
            and hasattr(thread, "is_alive")
            and thread.is_alive()
        ):
            if thread is not threading.current_thread():
                try:
                    thread.join(timeout=join_timeout)
                except Exception as exc:
                    logger.warning("Failed to join outbox thread: %s", exc)
            if thread.is_alive():
                logger.warning("Outbox worker did not stop before shutdown deadline")
                return

        if self._lexical_backfill is not None:
            if not self._lexical_backfill.close(timeout=join_timeout):
                logger.warning("Lexical backfill did not stop before shutdown deadline")
                return
        if self._lexical_index is not None and self._projection_index is not None:
            try:
                if not self._projection_index.close(timeout=join_timeout):
                    logger.warning("Hybrid retrieval indexes did not close before deadline")
                    return
            except Exception as exc:
                logger.warning("Failed to close hybrid retrieval indexes: %s", exc)
                return
            self._projection_index = None
            self._lexical_index = None
            self._lexical_backfill = None
            self._vector_index = None
        with self._init_lock:
            self._outbox_listener = None
            self._outbox_thread = None
            self._outbox_stop.clear()


    def get_decision_client(self):
        """One optional protocol client shared by retrieval and lazy extraction."""
        config = getattr(self._cfg, "decision", None)
        if config is None:
            return None
        with self._decision_client_lock:
            if self._decision_client is None:
                from providers.decision import JevDecisionClient
                self._decision_client = self._bind_model_usage(JevDecisionClient(config))
            return self._decision_client

    def _rerank_decision_evidence(self, query, working_set):
        if not self._cfg.retrieval_decision_rerank_enabled:
            return working_set, None
        client = self.get_decision_client()
        if client is None:
            return working_set, None
        from retrieval.decision_reranker import rerank_evidence
        return rerank_evidence(query, working_set, client)

    @_serialize_lazy_initialization("_llm")
    def get_llm(self):
        if self._llm is None:
            cfg = self._provider_cfg
            if cfg.provider == "mock" and cfg.openai_api_key:
                cfg = ProviderConfig(
                    provider="openai",
                    openai_api_key=cfg.openai_api_key,
                    openai_base_url=cfg.openai_base_url,
                    openai_llm_model=cfg.openai_llm_model,
                    llm_temperature=cfg.llm_temperature,
                    llm_max_tokens=cfg.llm_max_tokens,
                    llm_request_timeout_seconds=cfg.llm_request_timeout_seconds,
                    llm_json_mode=cfg.llm_json_mode,
                    llm_reasoning_effort=cfg.llm_reasoning_effort,
                    llm_is_reasoning_model=cfg.llm_is_reasoning_model,
                    llm_extra_body=dict(cfg.llm_extra_body),
                )
            if cfg.provider == "mock":
                from providers.llm import MockLLM
                llm = MockLLM()
            else:
                openai_llm_cls, _ = get_openai_llm()
                llm = openai_llm_cls(
                    api_key=cfg.effective_openai_api_key(),
                    base_url=cfg.openai_base_url,
                    model=cfg.openai_llm_model,
                    temperature=cfg.llm_temperature,
                    max_tokens=cfg.llm_max_tokens,
                    request_timeout_seconds=cfg.llm_request_timeout_seconds,
                    json_mode=cfg.llm_json_mode,
                    reasoning_effort=cfg.llm_reasoning_effort,
                    is_reasoning_model=cfg.llm_is_reasoning_model,
                    extra_body=cfg.llm_extra_body,
                )
            self._llm = self._bind_model_usage(llm)
            logger.info("LLM ready: %s", type(self._llm).__name__)
        return self._llm

    def _ensure_scene_runtime(self):
        if not self._cfg.scene_enabled or self._scene_manager is not None:
            return
        with self._init_lock:
            if self._scene_manager is not None:
                return
            from scene.adapters import CanonicalSceneAdapter
            from scene.config import SceneConfig
            from scene.decision import SceneDecisionEngine
            from scene.manager import SceneMemoryManager
            from scene.retrieval import SceneRanker
            from scene.service import SceneService
            from scene.sql_store import SQLSceneConfigProvider, SQLSceneJobStore, SQLSceneRepository

            if not self._use_sql:
                raise RuntimeError("scene memory requires a configured durable scene-store adapter")
            pool = self._get_shared_sql_pool()
            config = SceneConfig().merged(self._cfg.scene_options)
            repository = SQLSceneRepository(pool)
            provider = SQLSceneConfigProvider(pool, default=config)
            from copy import copy

            from providers.lexical_index.factory import create_lexical_index
            from scene.candidate_index import LazySceneIndex, SceneIndexRefresh

            lexical_config = copy(self._cfg)
            lexical_config.retrieval_lexical_options = {
                **self._cfg.retrieval_lexical_options, "table_name": "scene_lexical_v1",
            }
            candidate_index = LazySceneIndex(lambda: create_lexical_index(lexical_config),
                                             projection_queue=repository)
            refresh_projection = SceneIndexRefresh(repository, candidate_index)

            def projection_refresh():
                if candidate_index.initialized:
                    refresh_projection()
            try:
                adapter = CanonicalSceneAdapter(
                    repository, self._get_context_fs(), self._get_shared_vector_index(),
                    self._get_shared_embedder(), config=config, config_provider=provider,
                    archive_store=self._get_archive_store(), candidate_index=candidate_index,
                )
                manager = SceneMemoryManager(
                    SQLSceneJobStore(pool), adapter,
                    SceneDecisionEngine(self.get_llm(), config, evidence_reader=adapter),
                    config=config, config_provider=provider, projection_refresh=projection_refresh,
                )
                scene_service = SceneService(
                    repository, adapter, manager, provider, self._get_shared_embedder(),
                )
                if self._cfg.scene_generation_enabled:
                    manager.start()
            except Exception:
                candidate_index.close()
                raise
            self._scene_source = adapter
            self._scene_ranker = SceneRanker(config)
            self._scene_service = scene_service
            self._scene_manager = manager

    def get_scene_service(self):
        self._ensure_scene_runtime()
        if self._scene_service is None:
            raise RuntimeError("scene memory is disabled")
        return self._scene_service

    def _scene_projection_changed(self, event, ctx):
        if not self._cfg.scene_enabled or not self._cfg.scene_generation_enabled:
            return
        from urllib.parse import urlsplit

        from core.errors import NodeNotFoundError

        parsed = urlsplit(event.uri)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 4 or parts[0] not in {"users", "agents"} or parts[2] != "memories":
            return
        self._ensure_scene_runtime()
        owner = f"{parts[0][:-1]}:{parts[1]}"
        tokens = 0
        try:
            node = self._get_context_fs().read_node(event.uri, ctx)
            if node.context_type != "MEMORY" or node.metadata.get("node_type") == "directory":
                return
            tokens = max(1, (len(node.content or node.abstract) + 3) // 4)
        except NodeNotFoundError:
            if event.event_type == "UPSERT_CONTEXT":
                return
        self._scene_manager.after_memory_write(
            parsed.netloc, owner, [event.uri], token_count=tokens,
            force=event.event_type in {"ARCHIVE_CONTEXT", "DELETE_CONTEXT"},
            notice_id=event.event_id,
        )

    @_serialize_lazy_initialization("_write_api")
    def get_write_api(self):
        if self._write_api is None:
            fs = self._get_context_fs()
            if fs is None:
                return None
            outbox = self._get_outbox_store(fs)
            if outbox is None:
                return None
            # Create SchemaRegistry for dynamic tool generation
            schema_registry = None
            try:
                from extraction.schemas.registry import SchemaRegistry
                schema_registry = SchemaRegistry()
                if self._cfg.agent_memory_enabled:
                    for category in ("case", "tool", "skill"):
                        schema_registry.register(replace(schema_registry.get(category), enabled=False))
                if self._cfg.skill_sedimentation_enabled:
                    schema_registry.unregister("skill")
                logger.info("SchemaRegistry initialized for dynamic tool generation")
            except Exception as exc:
                logger.warning("SchemaRegistry initialization failed; write API unavailable: %s", exc)
                return None

            # Create URIResolver for prefetch
            uri_resolver = None
            try:
                from core.uri_resolver import URIResolver
                if schema_registry:
                    uri_resolver = URIResolver(schema_registry)
                    logger.info("URIResolver initialized for prefetch")
            except Exception as exc:
                logger.warning("URIResolver initialization failed, prefetch disabled: %s", exc)

            # Get shared vector_index and embedder for prefetch
            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()
            self._ensure_scene_runtime()

            graph_kwargs = {}
            if self._graph_enabled():
                from retrieval.authorized_vector_search import AuthorizedVectorSearch
                from service.relation_candidate_service import RelationCandidateService
                graph_kwargs = dict(graph_enabled=True,relation_store=self._get_relation_store(),
                    relation_candidate_service=RelationCandidateService(AuthorizedVectorSearch(
                        vector_index,embedder,ContextReader(fs=fs,storage_backend=self._cfg.storage_backend))))

            self._write_api = MemoryWriteAPI(
                fs=fs,
                llm=self.get_llm(),
                outbox_store=outbox,
                schema_registry=schema_registry,
                vector_index=vector_index,
                embedder=embedder,
                uri_resolver=uri_resolver,
internal_tool_usage_tracker=self._internal_tool_usage,
                extraction_mode=self._cfg.extraction_mode,
                source_occurrence_reconciliation_enabled=self._cfg.source_occurrence_reconciliation_enabled,
                card_format_enabled=self._cfg.card_format_enabled,
                dual_pass_enabled=self._cfg.dual_pass_enabled,
                node_processing_enabled=self._cfg.node_processing_enabled,
                directory_summary_enabled=self._cfg.directory_summary_enabled,
                **graph_kwargs,
            )
            logger.info("WriteAPI ready (backend=%s, extraction_mode=%s)", self._cfg.storage_backend, self._cfg.extraction_mode)
            # Start outbox worker so written memories get indexed
            self._start_outbox_worker()
        return self._write_api

    def forget_memory(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        external_uri: str | None = None,
        reason: str | None = None,
        client_request_id: str | None = None,
    ) -> dict:
        """Forget a single memory via soft archive and index outbox sync."""

        fs = self._get_context_fs()
        if fs is None:
            raise RuntimeError("ContextFS unavailable for forget_memory")

        node = fs.read_node(uri, ctx)
        stored_uri = node.uri
        if node.category == "profile" or stored_uri.rstrip("/").endswith("/memories/profile"):
            raise AuthorizationError("profile memory cannot be forgotten via MCP")

        role = ctx.role if isinstance(ctx.role, Role) else Role(ctx.role)
        agent_scoped = node.owner_space == ctx.agent_space_name() or "/agents/" in stored_uri
        if agent_scoped and role not in (Role.ADMIN, Role.ROOT):
            raise AuthorizationError("agent-scoped memory forget requires ADMIN or ROOT")

        original_node = deepcopy(node)
        fs.archive_node(stored_uri, ctx)

        outbox_event_id = None
        outbox = self._get_outbox_store(fs)
        if outbox is not None:
            try:
                if original_node.metadata.get("agent_memory_version") == 1:
                    from index.index_record_builder import agent_memory_index_ids
                    event = outbox.register_archive(
                        stored_uri, ctx, record_ids=agent_memory_index_ids(original_node),
                    )
                else:
                    event = outbox.register_archive(stored_uri, ctx)
                outbox_event_id = getattr(event, "event_id", None)
            except Exception:
                try:
                    fs.write_node(original_node, ctx)
                except Exception as rollback_exc:
                    logger.error(
                        "forget_memory rollback failed after outbox registration error uri=%s: %s",
                        stored_uri,
                        rollback_exc,
                        exc_info=True,
                    )
                raise

        self.get_audit_service().record(
            ctx.account_id,
            actor=ctx.user_id,
            target=external_uri or stored_uri,
            action="mcp.delete_memory",
            result="success",
            trace_id=ctx.trace_id,
            details={
                "mode": "archive",
                "internal_uri": stored_uri,
                "external_uri": external_uri,
                "category": node.category,
                "owner_space": node.owner_space,
                "role": str(ctx.role),
                "agent_id": ctx.agent_id,
                "session_id": ctx.session_id,
                "reason": reason,
                "client_request_id": client_request_id,
                "outbox_event_id": outbox_event_id,
            },
        )

        return {
            "ok": True,
            "deleted": True,
            "mode": "archive",
            "uri": external_uri or uri,
            "category": node.category,
            "outbox_event_id": outbox_event_id,
        }

    @_serialize_lazy_initialization("_read_api")
    def get_read_api(self, *, start_outbox_worker: bool = True):
        """Return the shared read API and optionally start index synchronization."""
        if self._read_api is None:
            embedder = self._get_shared_embedder()
            vector_index = self._get_shared_vector_index()
            self._ensure_scene_runtime()

            cfg = RetrievalConfig(
                include_agent_memories=self._cfg.agent_memory_enabled,
                expand_hierarchy=getattr(self._cfg, "retrieval_expand_hierarchy", True),
                max_top_k=getattr(self._cfg, "retrieval_max_top_k", 50),
                default_top_k=getattr(self._cfg, "retrieval_default_top_k", 10),
                global_search_topk=getattr(self._cfg, "retrieval_global_search_topk", 30),
                candidate_pool_enabled=getattr(self._cfg, "retrieval_candidate_pool_enabled", False),
                candidate_pool_multiplier=getattr(self._cfg, "retrieval_candidate_pool_multiplier", 3),
                min_candidate_pool_size=getattr(self._cfg, "retrieval_min_candidate_pool_size", 15),
                result_ranker_strategy=getattr(self._cfg, "retrieval_result_ranker_strategy", "semantic"),
                global_result_rerank_enabled=getattr(
                    self._cfg, "retrieval_global_result_rerank_enabled", False,
                ),
                owner_scope_filter_enabled=getattr(self._cfg, "retrieval_owner_scope_filter_enabled", False),
                hybrid_enabled=getattr(self._cfg, "retrieval_hybrid_enabled", False),
                vector_top_k=getattr(self._cfg, "retrieval_vector_top_k", 100),
                lexical_top_k=getattr(self._cfg, "retrieval_lexical_top_k", 50),
                rrf_k=getattr(self._cfg, "retrieval_rrf_k", 60),
                semantic_weight=getattr(self._cfg, "retrieval_semantic_weight", 1.0),
                lexical_weight=getattr(self._cfg, "retrieval_lexical_weight", 1.0),
                enable_graph_expansion=self._graph_enabled() and self._cfg.enable_graph_expansion,
                graph_expand_top_k=getattr(self._cfg,"graph_expand_top_k",5),
                graph_expand_limit=getattr(self._cfg,"graph_expand_limit",3),
                graph_min_weight=getattr(self._cfg,"graph_min_weight",0.0),
                graph_edge_similarity_threshold=getattr(self._cfg,"graph_edge_similarity_threshold",0.35),
                graph_supplement_enabled=getattr(self._cfg,"graph_supplement_enabled",True),
            )

            # Create relation_store
            relation_store = None
            try:
                relation_store = self._get_relation_store()
            except Exception:
                pass

            # Create context_reader for content filling (shared by pipeline and read_service)
            context_reader = None
            fs = None
            try:
                fs = self._get_context_fs()
                if fs is not None:
                    context_reader = ContextReader(fs=fs, storage_backend=self._cfg.storage_backend)
            except Exception as exc:
                logger.warning("ContextFS unavailable for read_memory: %s", exc)

            lexical_backfill = None
            if cfg.hybrid_enabled:
                if fs is None or self._lexical_index is None:
                    raise RuntimeError(
                        "hybrid retrieval requires canonical storage and lexical index"
                    )
                lexical_backfill = self._get_lexical_backfill()

            graph_kwargs = {}
            if cfg.enable_graph_expansion:
                if relation_store is None or context_reader is None:
                    raise RuntimeError("Graph retrieval requires canonical relation storage and reader")
                from service.graph_service import GraphService
                from retrieval.graph_retriever import OneHopRetriever
                from retrieval.graph_supplement import GraphSupplement
                graph = GraphService(None,relation_reader=relation_store)
                profile = self._graph_profile()
                graph_kwargs = dict(graph_retriever=OneHopRetriever(graph,
                    dict(edges_per_node=cfg.graph_expand_limit,expansion_limit=cfg.graph_expand_top_k,min_weight=cfg.graph_min_weight,
                         edge_similarity_threshold=cfg.graph_edge_similarity_threshold),
                    embedder=embedder,embedding_profile=profile),
                    graph_supplement=GraphSupplement(graph,context_reader,vector_index,embedder,profile))

            pipeline = RetrievalPipeline(
                planner=QueryPlanner(cfg),
                seed_retriever=SeedRetriever(
                    vector_index,
                    embedder,
                    cfg,
                    self._lexical_index,
                    lexical_backfill,
                ),
                hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
                assembly=ResultRanker(cfg, relation_store=relation_store),
                config=cfg,
                context_reader=context_reader,
                **graph_kwargs,
                scene_ranker=self._scene_ranker,
                scene_source=self._scene_source,
                scene_enabled=self._cfg.scene_retrieval_enabled,
            )

            self._read_api = ReadAPI(
                pipeline=pipeline,
                read_service=context_reader,
                config=cfg,
            )
            logger.info(
                "ReadAPI ready: embedder=%s, index=%s, backend=%s",
                type(embedder).__name__,
                type(vector_index).__name__,
                self._cfg.storage_backend,
            )
        if start_outbox_worker:
            self._start_outbox_worker()
        return self._read_api


    # Alias for downstream adapters that use the service-level verb.


    # -- Lifecycle handlers ---------------------------------------------------

    @_serialize_lazy_initialization("_session_mgr")
    def get_session_manager(self) -> SessionManager:
        """Lazy-init SessionManager with current service dependencies."""
        if self._session_mgr is None:
            from service.api import MemoryWriteAPI as _CheckpointWriteAPI
            checkpoint_fs = (self._write_api._fs
                             if isinstance(self._write_api, _CheckpointWriteAPI) else None)
            durable_runtime_state_enabled = bool(
                getattr(self._cfg, "session_state_bridge_enabled", True)
                or checkpoint_fs is not None
            )
            session_state_context_fs = (
                (lambda: checkpoint_fs) if checkpoint_fs is not None else self._get_context_fs
                if durable_runtime_state_enabled
                else None
            )
            self._session_mgr = SessionManager(
                get_llm=self.get_llm,
                get_write_api=self.get_write_api,
                get_agfs=self._get_context_fs,
                get_archive_store=self._get_archive_store,
                archive_store_required=self._use_sql,
                get_context_fs=session_state_context_fs,
                archive_max_count=self._cfg.archive_max_count,
                archive_merge_threshold=self._cfg.archive_merge_threshold,
                compression_quality_enabled=self._cfg.compression_quality_enabled,
                compression_quality_persist_metadata=self._cfg.compression_quality_persist_metadata,
                session_state_bridge_enabled=getattr(
                    self._cfg,
                    "session_state_bridge_enabled",
                    True,
                ),
            )
            logger.info("SessionManager ready")
        return self._session_mgr


    def _get_archive_store(self):
        """Return an archive store instance.

        SQL mode: cached for service lifetime (shared connection pool).
        AGFS mode: fresh instance each call (self-healing on transient failures).
        """
        if self._use_sql:
            if self._archive_store is not None:
                return self._archive_store
            if self._archive_store_failed:
                return None
            try:
                from session.sql_archive_store import SQLSessionArchiveStore
                self._archive_store = SQLSessionArchiveStore(
                    pool=self._get_shared_sql_pool(),
                )
                return self._archive_store
            except Exception as exc:
                logger.error("Failed to create SQL archive store: %s", exc)
                self._archive_store_failed = True
                return None
        if _HAS_AGFS:
            try:
                from session import SessionArchiveStore
                client = AGFSClient(api_base_url=self._agfs_base_url)
                agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
                return SessionArchiveStore(fs=agfs)
            except Exception as exc:
                logger.error("Failed to create AGFS archive store: %s", exc)
                return None
        return None

    def _get_agfs_fs(self):
        """Create a fresh ContextFS instance for session operations."""
        try:
            return self._get_context_fs()
        except Exception as exc:
            logger.warning("_get_agfs_fs failed: %s", exc)
            return None

    def _require_global_session_admin(self, params: dict | None) -> None:
        ctx = (params or {}).get("_ctx")
        auth = self.get_auth_service()
        if not isinstance(ctx, RequestContext):
            if auth.role_control_active():
                raise PermissionError("authenticated context required")
            return
        auth.require_role(ctx, Role.ROOT, Role.ADMIN)

    @_attribute_request_usage
    def prefetch(self, params: dict) -> dict:
        """Best-effort pre-compose retrieval staged into the session topic buffer."""
        if not getattr(self._cfg, "prefetch_enabled", False):
            return {"ok": True, "prefetched": 0, "reason": "disabled"}
        try:
            ctx = params.get("_ctx") or self.build_context(params)
            session_id = params.get("sessionId") or ctx.session_id
            messages = params.get("messages", [])
            raw = params.get("prompt", "") or params.get("query", "") or extract_query(messages)
            query = sanitize_query(raw)
            if not query:
                return {"ok": True, "prefetched": 0, "reason": "empty_query"}

            top_k = _positive_int(
                params.get("prefetchTopK", getattr(self._cfg, "prefetch_top_k", 5)),
                5,
            )
            read_api = self.get_read_api()
            if self._cfg.skill_sedimentation_enabled:
                result = self._search_without_generated_skills(
                    read_api, query=query, ctx=ctx, top_k=top_k,
                    fill_content_for_top_k=0,
                )
            else:
                result = read_api.search_memory(
                    query=query,
                    ctx=ctx,
                    top_k=top_k,
                    fill_content_for_top_k=0,
                    mode="QUICK",
                    hints={"include_all_memories": True},
                )
            ordinary_hits = list(getattr(result, "hits", []) or [])
            hits = [
                _retrieved_block_to_seed_hit(hit)
                for hit in ordinary_hits[:top_k]
                if getattr(hit, "uri", "")
            ]
            topic_buffer = self.get_session_manager().get_topic_buffer(session_id)
            topic_buffer.set_pending_injection(hits)
            return {"ok": True, "prefetched": len(hits)}
        except Exception as exc:
            logger.warning("prefetch failed: %s", exc, exc_info=True)
            return {"ok": False, "prefetched": 0, "error": str(exc)}

    def session_working_set(self, params: dict | None = None) -> dict:
        params = params or {}
        self._require_global_session_admin(params)
        return {
            "ok": True,
            "sessions": self.get_session_manager().list_session_working_set(),
        }

    def evict_idle_sessions(self, params: dict) -> dict:
        self._require_global_session_admin(params)
        max_idle_seconds = _positive_int(params.get("maxIdleSeconds"), 0)
        evicted = self.get_session_manager().evict_idle_sessions(
            max_idle_seconds=max_idle_seconds,
            now_iso=params.get("nowIso"),
            ctx=params.get("_ctx"),
        )
        return {"ok": True, "evicted": evicted}

    @_guard_session_mutation
    def add_session_message(self, params: dict) -> dict:
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        return self.get_session_manager().add_message(
            session_id,
            params.get("role", "user"),
            params.get("content", ""),
            ctx,
            created_at=params.get("created_at"),
        )

    @_guard_session_mutation
    def commit_session_buffer(self, params: dict) -> dict:
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        return self.get_session_manager().commit(
            session_id,
            ctx,
            wait=params.get("wait", False),
        )

    @_attribute_request_usage
    @_guard_session_mutation
    def compose(self, params: dict) -> dict:
        """Assemble memory context for the current turn.

        Pipeline with 3-layer architecture:
          1. Extract query + budget
          2. Read profile          (Layer 1 — stable identity, rarely changes)
          3. Collect archives       (Layer 1b — episodic history, semi-stable)
          4. Search working set     (Layer 3 — dynamic, changes every turn)
          4b. Get/compress session state (Layer 2 — updates every N turns)
          5. Build stable identity  (Layer 1 output → system message)
          5b. Build episodic + session state (→ system messages)
          6. Return (all context injected into messages as system messages)

        Memory context is injected into messages as system messages,
        ordered by stability (less dynamic first, more dynamic last):

          Before original messages:
            ## Profile            ← changes only on profile update
            ## Archive History    ← changes when sessions end
            ## Session State      ← updates every N turns
          After original messages:
            ## Working Set        ← changes every turn

        systemPromptAddition is empty for KV-cache efficiency.

        Args:
            params: Dict with optional keys:
                - messages: list[dict] - Current conversation messages
                - prompt: str - Direct query prompt (optional)
                - tokenBudget: int - Token budget (default: 128_000)

        Returns:
            Dict with:
                - messages: list[dict] - Messages with injected system messages
                - systemPromptAddition: str - Empty (content in messages)
                - systemPromptSuffix: str - Empty (content in messages)
                - memoryUserMessage: str - Empty (content in messages)
                - estimatedTokens: int - Total estimated tokens
                - archiveCount: int - Number of archives found
                - archiveIncluded: bool - Whether archives were injected
                - identityContext: str - Semantic slot: profile only
                - episodicContext: str - Semantic slot: archive history
                - sessionContext: str - Semantic slot: structured session state
        """
        messages = params.get("messages", [])
        prompt = params.get("prompt", "") or params.get("query", "")

        # Strip previous synthetic memory messages to prevent cumulative duplication.
        # Synthetic messages are tagged with _ogmem=True by _to_response().
        messages = [
            m for m in messages
            if not (isinstance(m, dict) and m.get("_ogmem"))
        ]
        logger.info(
            "assemble entry: msgs=%d prompt_len=%d keys=%s",
            len(messages), len(prompt), sorted(params.keys()),
        )
        raw = prompt.strip() if prompt else extract_query(messages)
        query = sanitize_query(raw)

        # Step 1: Budget
        token_budget_value = params.get("tokenBudget", 128_000)
        token_budget = TokenBudget(total=token_budget_value)
        budget_allocation = token_budget.allocate()

        # Empty query: return messages unchanged
        if not query:
            result = ComposedContext(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

        try:
            ctx = params.get("_ctx") or self.build_context(params)
        except Exception as exc:
            logger.warning("assemble build_context failed: %s", exc, exc_info=True)
            result = ComposedContext(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

        # Each step degrades independently — a failure in one doesn't kill the rest
        session_id = params.get("sessionId") or ctx.session_id
        mgr = self.get_session_manager()
        topic_buffer = None
        try:
            if session_id:
                topic_buffer = mgr.get_topic_buffer(session_id)
        except Exception as exc:
            logger.warning("assemble topic buffer unavailable: %s", exc)

        try:
            if (
                topic_buffer is not None
                and session_id
                and getattr(self._cfg, "topic_detection_enabled", False)
            ):
                from session.topic_detector import TopicDetector

                buf = mgr.get_or_create(session_id, ctx=ctx)
                topic_messages = list(buf.messages)
                for idx, msg in enumerate(messages[-8:]):
                    role, content = _message_role_content(msg)
                    if role and content:
                        topic_messages.append(
                            SessionMessage(
                                id=f"compose_topic_{idx}",
                                role=role,
                                content=content,
                            )
                        )
                embedder = None
                try:
                    embedder = self._get_shared_embedder()
                except Exception as exc:
                    logger.debug("topic detection embedder unavailable: %s", exc)
                topic_changed = topic_buffer.update_topic(
                    TopicDetector(embedder=embedder).detect(
                        topic_messages,
                        previous=topic_buffer.get_current_topic(),
                    )
                )
                if topic_changed:
                    topic_buffer.clear_injected_tracking()
        except Exception as exc:
            logger.warning("assemble topic detection failed: %s", exc)

        profile = ""
        identity_context = ""
        try:
            identity_slot = topic_buffer.get_cached_slot("identity") if topic_buffer else None
            if identity_slot is not None:
                identity_context = identity_slot.content
            else:
                profile = self._read_profile(ctx)
        except Exception as exc:
            logger.warning("assemble profile read failed: %s", exc)

        latest_archives, pre_archives = [], []
        episodic_context = ""
        archive_count_from_cache = None
        try:
            archive_slot = topic_buffer.get_cached_slot("archive_history") if topic_buffer else None
            if archive_slot is not None:
                episodic_context = archive_slot.content
                archive_count_from_cache = len(archive_slot.uris) or (
                    1 if episodic_context else 0
                )
            else:
                latest_archives, pre_archives = self._collect_archives(ctx, token_budget)
        except Exception as exc:
            logger.warning("assemble archive collection failed: %s", exc)

        working_set = []
        summary_items = []
        search_stats: dict = {}
        try:
            working_set, summary_items, search_stats = self._search_working_set(query, ctx)
            working_set.extend(summary_items)
        except Exception as exc:
            logger.warning("assemble working set search failed: %s", exc)

        injection_snapshot = None
        try:
            if topic_buffer is not None:
                injection_snapshot = topic_buffer.injection_snapshot()
                pending_hits = injection_snapshot[0] or []
                if pending_hits:
                    existing_memories = {
                        memory_identity(item["uri"], item.get("metadata")) for item in working_set
                        if isinstance(item, dict) and item.get("uri")
                    }
                    for hit in pending_hits:
                        uri = canonical_uri(hit.uri)
                        identity = memory_identity(uri, hit.metadata)
                        if not uri or identity in existing_memories or topic_buffer.was_injected(uri, hit.metadata):
                            continue
                        working_set.append(_seed_hit_to_working_set_item(hit))
                        existing_memories.add(identity)
        except Exception as exc:
            logger.warning("assemble pending injection merge failed: %s", exc)

        delivery_stats = {}
        if any(item.get("category") != "code" for item in working_set):
            graph_items = [item for item in working_set
                           if (item.get("metadata") or {}).get("retrieval_origin") == "graph_neighbor"]
            ordinary_items = [item for item in working_set
                              if (item.get("metadata") or {}).get("retrieval_origin") != "graph_neighbor"]
            read_many = None
            if self._cfg.storage_backend == "sql" and not self._graph_enabled():
                def read_many(requests, read_ctx):
                    return self.get_read_api().read_memories(requests, read_ctx)
            working_set, delivery_stats = hydrate_evidence(
                ordinary_items, lambda uri, read_ctx, **kwargs: self.get_read_api().read_memory(uri, read_ctx, **kwargs),
                ctx, read_many=read_many,
            )
            if graph_items:
                graph_items, _ = hydrate_evidence(
                    graph_items, lambda uri, read_ctx, **kwargs: self.get_read_api().read_memory(uri, read_ctx, **kwargs), ctx)
                working_set.extend(graph_items)
            search_stats["evidence_delivery"] = delivery_stats
            working_set, decision_stats = self._rerank_decision_evidence(query, working_set)
            if decision_stats is not None:
                search_stats["decision_rerank"] = decision_stats

        # Each step degrades independently — a failure in one doesn't kill the rest

        # Step 5a: Build stable identity (Profile only) → systemPromptAddition
        try:
            if not identity_context:
                identity_context = self._format_profile(
                    profile=profile,
                    budget=budget_allocation,
                )
                if topic_buffer is not None and identity_context:
                    topic_buffer.set_cached_slot(
                        "identity",
                        SlotContent(
                            content=identity_context,
                            uris=[f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"],
                            tokens=len(identity_context) // 4,
                        ),
                    )
        except Exception as exc:
            logger.warning("assemble identity_context build failed: %s", exc)

        # Step 5b: Build episodic history (Archives only) → systemPromptSuffix prefix
        try:
            if not episodic_context:
                episodic_context = self._format_archives(
                    latest_archives=latest_archives,
                    pre_archives=pre_archives,
                    budget=budget_allocation,
                )
                if topic_buffer is not None and episodic_context:
                    topic_buffer.set_cached_slot(
                        "archive_history",
                        SlotContent(
                            content=episodic_context,
                            uris=[
                                ref.archive_uri
                                for ref in [*latest_archives, *pre_archives]
                                if ref.archive_uri
                            ],
                            tokens=len(episodic_context) // 4,
                        ),
                    )
        except Exception as exc:
            logger.warning("assemble episodic_context build failed: %s", exc)

        # Step 5c: Build dynamic working set → user message (preserves KV cache)
        if self._graph_enabled():
            working_set = self.get_read_api().append_graph_supplement(
                working_set,query,search_stats,ctx,budget_allocation,
                lambda items, allocation: self._format_working_set(items, allocation, query=query))
        working_set_msg = ""
        try:
            working_set_msg = self._format_working_set(
                working_set, budget_allocation, query=query, delivery_stats=delivery_stats,
            )
        except Exception as exc:
            logger.warning("assemble working_set_msg build failed: %s", exc)

        # Step 4b: Get/compress session state → Layer 2 (updates every N turns)
        session_state_suffix = ""
        window_state = SessionWindowState()  # Default empty state
        try:
            window_state = self._get_session_state(session_id, ctx)
            skills_slot = topic_buffer.get_cached_slot("skills_summary") if topic_buffer else None
            if skills_slot is not None and skills_slot.content and not window_state.skills_text:
                window_state.skills_text = skills_slot.content
            elif topic_buffer is not None and window_state.skills_text and skills_slot is None:
                topic_buffer.set_cached_slot(
                    "skills_summary",
                    SlotContent(
                        content=window_state.skills_text,
                        tokens=len(window_state.skills_text) // 4,
                    ),
                )
            session_state_suffix = self._format_session_state(
                window_state, budget_allocation,
            )
        except Exception as exc:
            logger.warning("assemble session_state_suffix build failed: %s", exc)

        session_context = session_state_suffix

        # Step 5d: Trim archived messages from the message list.
        # If archives exist, the archived content is already represented in
        # episodic_context (position 2). Only keep unarchived messages
        # (session buffer) plus the current turn's new messages.
        trimmed_messages = messages
        try:
            session_id_for_buf = params.get("sessionId") or ctx.session_id
            if episodic_context and session_id_for_buf:
                mgr = self.get_session_manager()
                buf = mgr.get_or_create(session_id_for_buf)
                buf_count = len(buf.messages)

                # buf.messages are the unarchived messages recorded by
                # after_turn.  Prefer finding that exact suffix in the caller's
                # message list so current-turn messages after the buffer are
                # also preserved.  If matching fails, fall back to the named
                # tail margin above so this heuristic is explicit.
                pre_prompt_count = _bounded_message_count(
                    params.get("prePromptMessageCount", 0),
                    len(messages),
                )
                pre_prompt_messages = messages[:pre_prompt_count]
                conversation_messages = messages[pre_prompt_count:]

                if buf_count == 0:
                    keep_from = 0
                    trim_reason = "empty_buffer"
                    if conversation_messages:
                        logger.warning(
                            "compose archive trim skipped because session buffer "
                            "is empty; preserving messages: session=%s trace=%s "
                            "total=%d pre_prompt=%d",
                            getattr(ctx, "session_id", ""),
                            getattr(ctx, "trace_id", ""),
                            len(messages),
                            pre_prompt_count,
                        )
                else:
                    keep_from = _find_last_message_sequence_start(
                        conversation_messages,
                        buf.messages,
                    )
                    trim_reason = "matched_buffer"
                    if keep_from is None:
                        if "prePromptMessageCount" not in params:
                            logger.warning(
                                "compose archive trim using tail margin without "
                                "prePromptMessageCount: session=%s trace=%s "
                                "total=%d buf=%d margin=%d",
                                getattr(ctx, "session_id", ""),
                                getattr(ctx, "trace_id", ""),
                                len(messages),
                                buf_count,
                                _ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN,
                            )
                        keep_count = min(
                            len(conversation_messages),
                            buf_count + _ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN,
                        )
                        keep_from = len(conversation_messages) - keep_count
                        trim_reason = "tail_margin"

                if keep_from > 0:
                    kept_messages = conversation_messages[keep_from:]
                    trimmed_messages = pre_prompt_messages + kept_messages
                    logger.info(
                        "compose trimmed archived messages: total=%d buf=%d "
                        "pre_prompt=%d dropped=%d kept=%d reason=%s",
                        len(messages), buf_count, pre_prompt_count, keep_from,
                        len(trimmed_messages), trim_reason,
                    )
        except Exception as exc:
            logger.warning("compose archive trim failed, using full messages: %s", exc)

        # Also trim messages compressed by RollingCompressor if session_state exists.
        # turn_count_at_last_compress is a user-turn count, so derive the exact
        # compressed message span from the runtime buffer and token watermark.
        if session_state_suffix and window_state.turn_count_at_last_compress > 0:
            pre_prompt_count = _bounded_message_count(
                params.get("prePromptMessageCount", 0),
                len(trimmed_messages),
            )
            compressed_turns = window_state.turn_count_at_last_compress
            compressed_tokens = window_state.token_count_at_last_compress
            conversation_count = len(trimmed_messages) - pre_prompt_count

            if pre_prompt_count < len(trimmed_messages) and compressed_tokens > 0:
                conversation_messages = trimmed_messages[pre_prompt_count:]
                compressed_message_count = 0
                try:
                    session_id_for_buf = params.get("sessionId") or ctx.session_id
                    if session_id_for_buf:
                        buf = self.get_session_manager().get_or_create(
                            session_id_for_buf,
                        )
                        user_turns = 0
                        estimated_tokens = 0
                        for idx, msg in enumerate(buf.messages):
                            if msg.role == "user":
                                user_turns += 1
                            estimated_tokens += msg.estimated_tokens
                            if (
                                user_turns == compressed_turns
                                and estimated_tokens == compressed_tokens
                            ):
                                compressed_message_count = idx + 1
                                break
                            if (
                                user_turns > compressed_turns
                                or estimated_tokens > compressed_tokens
                            ):
                                break
                except Exception as exc:
                    logger.debug(
                        "compose compressed trim buffer span check failed; "
                        "preserving caller messages: %s",
                        exc,
                    )

                compression_covers_messages = False
                if (
                    compressed_message_count > 0
                    and len(conversation_messages) >= compressed_message_count
                ):
                    buffer_prefix = buf.messages[:compressed_message_count]
                    caller_prefix = conversation_messages[:compressed_message_count]
                    compression_covers_messages = [
                        _message_role_content(msg)
                        for msg in buffer_prefix
                    ] == [
                        _message_role_content(msg)
                        for msg in caller_prefix
                    ]

                if compression_covers_messages:
                    trimmed_messages = (
                        trimmed_messages[:pre_prompt_count]
                        + conversation_messages[compressed_message_count:]
                    )
                    logger.info(
                        "compose trimmed verified compressed messages "
                        "(compressed_turns=%d, compressed_messages=%d, "
                        "pre_prompt=%d, kept=%d)",
                        compressed_turns,
                        compressed_message_count,
                        pre_prompt_count,
                        len(trimmed_messages),
                    )
                else:
                    logger.debug(
                        "compose skipped compression trim because caller messages "
                        "are not confirmed in verified compressed buffer span "
                        "(compressed_turns=%d, compressed_tokens=%d, "
                        "compressed_messages=%d, pre_prompt=%d, conversation=%d)",
                        compressed_turns,
                        compressed_tokens,
                        compressed_message_count,
                        pre_prompt_count,
                        conversation_count,
                    )

        # Step 6: Assemble final output
        try:
            system_prompt_suffix = "\n\n".join(
                p for p in [episodic_context, session_state_suffix] if p
            )

            archive_count = (
                archive_count_from_cache
                if archive_count_from_cache is not None
                else len(latest_archives) + len(pre_archives)
            )
            prompt_tokens = (
                len(identity_context) // 4
                + len(system_prompt_suffix) // 4
                + len(working_set_msg) // 4
            )
            msg_tokens = self._estimate_tokens(trimmed_messages)

            # Extract open_loops and uncertainties from window_state
            open_loops = window_state.open_loops if window_state else []
            uncertainties = window_state.uncertainties if window_state else []

            # Build budget metadata by slot
            budget_used_by_slot = {
                "identity": len(identity_context) // 4,
                "episodic": len(episodic_context) // 4,
                "session": len(session_context) // 4,
                "retrieved": len(working_set_msg) // 4,
            }

            # Create ComposedContext with all semantic slots populated
            result = ComposedContext(
                # Semantic slots
                identity_context=identity_context,
                episodic_context=episodic_context,
                session_context=session_context,
                task_context="",
                retrieved_evidence=working_set_msg,
                open_loops=open_loops,
                uncertainties=uncertainties,
                # Budget metadata
                estimated_tokens=msg_tokens + prompt_tokens,
                budget_used_by_slot=budget_used_by_slot,
                # Stats for analysis
                stats=search_stats,
                # Legacy compat
                messages=trimmed_messages,
                archive_count=archive_count,
                archive_included=archive_count > 0,
                system_prompt_suffix=system_prompt_suffix,
            )

            response = self._to_response(result)
            if topic_buffer is not None and injection_snapshot is not None:
                topic_buffer.consume_injection(
                    injection_snapshot, delivery_stats.get("delivered_uris", []) if working_set_msg else [],
                    delivered_units=delivery_stats.get("delivered_units", []) if working_set_msg else [],
                )
            return response
        except Exception as exc:
            logger.warning("assemble final result construction failed: %s", exc, exc_info=True)
            result = ComposedContext(
                messages=trimmed_messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

    def _to_response(self, result: ComposedContext) -> dict:
        """Convert ComposedContext to dict for the OpenClaw plugin.

        Message injection order (all go into messages as user role for KV-cache):
          1. Profile (identity_context) — stable prefix
          2. Episodic (archive history) — rarely changes
          3. Working Set (retrieved_evidence) — before question to avoid confirmation bias
          4. Original messages — grows each turn
          5. Session state — dynamic suffix

        systemPromptAddition/systemPromptSuffix/memoryUserMessage are empty —
        all context is in messages.  The plugin only passes messages through.
        """
        injected_messages = []

        if result.identity_context:
            injected_messages.append({"role": "user", "content": result.identity_context, "_ogmem": True})

        if result.episodic_context:
            injected_messages.append({"role": "user", "content": result.episodic_context, "_ogmem": True})

        injected_messages.extend(result.messages)

        if result.session_context:
            injected_messages.append({"role": "user", "content": result.session_context, "_ogmem": True})

        if result.retrieved_evidence:
            injected_messages.append({"role": "user", "content": result.retrieved_evidence, "_ogmem": True})

        return {
            "messages": injected_messages,
            "systemPromptAddition": "",
            "systemPromptSuffix": "",
            "memoryUserMessage": "",
            "estimatedTokens": result.estimated_tokens,
            "archiveCount": result.archive_count,
            "archiveIncluded": result.archive_included,
            "stats": result.stats,

            "identityContext": result.identity_context,
            "episodicContext": result.episodic_context or "",
            "sessionContext": result.session_context or "",
            "taskContext": result.task_context,
            "retrievedEvidence": result.retrieved_evidence,
            "openLoops": result.open_loops,
            "uncertainties": result.uncertainties,
            "budgetUsedBySlot": result.budget_used_by_slot,
        }

    def add_session_trace(self, params: dict) -> dict:
        """Persist one bounded agent event directly to SQL."""
        ctx = params.get("_ctx")
        if ctx is None:
            return {"ok": False, "error": "missing context", "retryable": False}

        event = params.get("event")
        if not isinstance(event, dict) or not event:
            return {
                "ok": False,
                "error": "missing event payload",
                "retryable": False,
            }

        event_id = str(params.get("eventId") or "").strip()
        session_id = str(params.get("sessionId") or "").strip()
        hook_type = str(params.get("hookType") or event.get("hook_type") or "unknown")
        if not event_id:
            return {"ok": False, "error": "missing event_id", "retryable": False}
        if len(event_id) > 64:
            return {
                "ok": False,
                "error": "event_id exceeds 64 characters",
                "retryable": False,
            }
        if not session_id or session_id == "unknown":
            return {"ok": False, "error": "missing session_id", "retryable": False}
        if len(session_id) > 128:
            return {
                "ok": False,
                "error": "session_id exceeds 128 characters",
                "retryable": False,
            }
        if len(hook_type) > 64:
            return {
                "ok": False,
                "error": "hook_type exceeds 64 characters",
                "retryable": False,
            }

        event_time = params.get("eventTime")
        if isinstance(event_time, str):
            try:
                event_time = _dt.fromisoformat(event_time.replace("Z", "+00:00"))
            except ValueError:
                return {
                    "ok": False,
                    "error": "invalid event_time",
                    "retryable": False,
                }
        if event_time is not None and not isinstance(event_time, _dt):
            return {
                "ok": False,
                "error": "invalid event_time",
                "retryable": False,
            }
        if isinstance(event_time, _dt) and event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)

        store = self._get_trace_event_store()
        if store is None:
            return {
                "ok": False,
                "error": "trace store unavailable",
                "retryable": True,
            }

        try:
            return store.insert(
                event_id=event_id,
                account_id=ctx.account_id,
                owner_space=ctx.user_space_name(),
                agent_id=ctx.agent_id,
                session_id=session_id,
                hook_type=hook_type,
                content=event,
                event_time=event_time,
            )
        except Exception as exc:
            logger.warning(
                "add_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "trace persistence unavailable",
                "error_code": "trace_persistence_unavailable",
                "retryable": True,
            }

    def _process_agent_memory(self, event):
        from server.agent_memory import process_trace_memory

        fs = self._get_context_fs()
        with request_usage_scope(RequestContext(**event.payload["context"])):
            process_trace_memory(
                event, trace_store=self._get_trace_event_store(),
                outbox=self._get_outbox_store(fs), fs=fs, llm=self.get_llm(),
                directory_summary_enabled=self._cfg.directory_summary_enabled,
                vector_index=self._get_shared_vector_index(), embedder=self._get_shared_embedder(),
            )

    def finalize_session_trace(self, params: dict) -> dict:
        """Finalize the turn-end-bounded turn containing ``stopEventIndex``."""
        ctx = params.get("_ctx")
        if ctx is None:
            return {"ok": False, "error": "missing context", "retryable": False}
        session_id = str(params.get("sessionId") or "").strip()
        try:
            stop_event_index = int(params.get("stopEventIndex") or 0)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "invalid stop_event_index",
                "retryable": False,
            }
        if not session_id or session_id == "unknown":
            return {"ok": False, "error": "missing session_id", "retryable": False}
        if stop_event_index <= 0:
            return {
                "ok": False,
                "error": "invalid stop_event_index",
                "retryable": False,
            }

        store = self._get_trace_event_store()
        if store is None:
            return {
                "ok": False,
                "error": "trace store unavailable",
                "retryable": True,
            }
        try:
            finalized = store.finalize_session(
                account_id=ctx.account_id,
                owner_space=ctx.user_space_name(),
                agent_id=ctx.agent_id,
                session_id=session_id,
                stop_event_index=stop_event_index,
            )
            if self._cfg.agent_memory_enabled and ctx.agent_id and finalized.get("uri"):
                from server.agent_memory import trace_memory_event

                outbox = self._get_outbox_store(self._get_context_fs())
                if outbox is None:
                    raise RuntimeError("agent memory outbox unavailable")
                event = trace_memory_event(finalized["uri"], replace(ctx, session_id=session_id))
                outbox.register_once(event, ctx)
                finalized["agent_memory_event_id"] = event.event_id
                self._start_outbox_worker()
            if self._cfg.skill_sedimentation_enabled:
                queued = store.enqueue_skill_candidate_from_trace(
                    account_id=ctx.account_id,
                    agent_id=ctx.agent_id,
                    session_id=session_id,
                    turn_id=str(finalized["turn_id"]),
                    trace_uri=str(finalized["uri"]),
                )
                finalized["skill_candidate_id"] = queued["candidate_id"]
            return finalized
        except ValueError as exc:
            logger.warning(
                "finalize_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "invalid trace boundary",
                "error_code": "invalid_trace_boundary",
                "retryable": False,
            }
        except Exception as exc:
            logger.warning(
                "finalize_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "trace finalization unavailable",
                "error_code": "trace_finalization_unavailable",
                "retryable": True,
            }

    @contextmanager
    def _session_mutation_scope(self, ctx, *, refresh=True):
        import time
        from fs.sql_adapter.session_mutation import active_session_guard
        from service.api import MemoryWriteAPI
        nested = active_session_guard() is not None
        if self._write_api is None and self._use_sql:
            from psycopg2 import InterfaceError, OperationalError
            deadline = time.monotonic() + 1.0
            try:
                pool = self._get_shared_sql_pool(deadline=deadline)
            except (TimeoutError, InterfaceError, OperationalError) as exc:
                raise SessionMutationBusyError("session mutation dialect discovery unavailable") from exc
            with SQLContextFS.configured_session_mutation_lock(
                self._cfg.sql_connection_string, ctx, dialect=pool.sql_dialect.name,
                slots=pool.session_mutation_slots, timeout_seconds=max(0, deadline - time.monotonic()),
            ) as guard:
                try:
                    if refresh and not nested:
                        self._load_extraction_recovery_before_accumulation(ctx.session_id, ctx, force_reload=True)
                except SessionOperationInProgress:
                    raise
                except Exception as exc:
                    raise RecoveryNeedsConfirmationError("durable session refresh failed: " + str(exc)) from exc
                yield guard
            return
        if isinstance(self._write_api, MemoryWriteAPI):
            fs = self._write_api._fs
        elif self._write_api is not None:
            yield None
            return
        else:
            manager = self.get_session_manager()
            fs = manager._get_context_fs() if callable(manager._get_context_fs) else self._get_context_fs()
            if fs is not None:
                manager._get_context_fs = lambda: fs
        acquire = (getattr(type(fs), "session_mutation_lock", None)
                   or getattr(type(fs), "session_recovery_lock", None))
        if not callable(acquire):
            yield None
            return
        if callable(getattr(type(fs), "active_session_recovery", None)):
            nested = fs.active_session_recovery() is not None
        from session.session_state import SessionState
        with acquire(fs, SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
            try:
                if refresh and not nested:
                    self._load_extraction_recovery_before_accumulation(ctx.session_id, ctx, force_reload=True)
            except SessionOperationInProgress:
                raise
            except Exception as exc:
                raise RecoveryNeedsConfirmationError("durable session refresh failed: " + str(exc)) from exc
            yield guard

    def _load_extraction_recovery_before_accumulation(self, session_id, ctx, *, force_reload=False):
        """Load once without creating any model, embedder or extraction API."""
        from service.api import MemoryWriteAPI
        manager = self.get_session_manager()
        if manager.extraction_recovery_loaded(session_id, ctx) and not force_reload:
            return
        if isinstance(self._write_api, MemoryWriteAPI):
            fs = self._write_api._fs
        elif self._write_api is not None:
            # Legacy integrations with mocked/in-process writers do not have
            # the persistent accepted-candidate and node-receipt contract.
            return
        elif callable(manager._get_context_fs):
            fs = manager._get_context_fs()
        else:
            fs = self._get_context_fs()
        if fs is None:
            return
        if not (callable(getattr(type(fs), "read_node", None))
                and callable(getattr(type(fs), "write_node", None))):
            return
        manager.require_extraction_recovery(session_id)
        # Bind a new, empty buffer only after its durable owner is validated.
        manager.get_or_create(session_id)
        manager.enable_extraction_recovery(session_id, ctx, fs, force_reload=force_reload)

    def _queue_extraction_recovery(self, ctx, error, *, archive_id, client_request_id):
        from service.extraction_retry import background_retry_running
        if (not isinstance(error, ExtractionIncompleteError) or isinstance(error, OperationDeadlineExceededError)
                or background_retry_running()):
            return None
        kind = ("memory_card_format" if isinstance(error, MemoryCardFormatError) else
                "llm_transient" if isinstance(error, LLMTransientError) else "")
        fields = error_fields(error, client_request_id)
        task = self._extraction_retry.register(
            ctx, {"error": str(error), "failure_kind": kind, **fields,
                  "extraction_archive_id": archive_id, "client_request_id": client_request_id})
        if task is None or task.get("status") not in {"queued", "running"}:
            return None
        return {"ok": True, "status": "queued", "reason": kind or "extraction_incomplete",
                "request_id": client_request_id, "source_status": "CHECKPOINTED", "semantic_status": "PENDING",
                "recovery_task": task, "last_error": str(error), **fields}

    def _refresh_write_outcomes(self, ctx):
        """Persist completed projection receipts under the existing session fence."""
        try:
            with self._session_mutation_scope(ctx):
                manager = self.get_session_manager()
                buf = manager.get_or_create(ctx.session_id, ctx)
                with buf.session_state_save_lock:
                    with buf.extraction_lock:
                        entries = buf.runtime_extensions.get(OUTCOMES_KEY, {}).get("entries", {})
                        updates = {}
                        for request_id, outcome in entries.items():
                            if needs_projection_refresh(outcome):
                                resolved = self._resolve_write_outcome(outcome)
                                if resolved != outcome:
                                    updates[request_id] = resolved
                        if not updates:
                            return
                        previous = {key: entries[key] for key in updates}
                        entries.update(updates)
                    if not manager.save_session_state(ctx.session_id, ctx):
                        with buf.extraction_lock:
                            entries.update(previous)
                        raise RecoveryNeedsConfirmationError("projection receipt checkpoint was not acknowledged")
        except (SessionMutationBusyError, SessionOperationInProgress):
            return

    def _resolve_write_outcome(self, outcome):
        if not outcome.get("outbox_tracking_complete") or not outcome.get("outbox_events"):
            return projection_outcome(outcome, None)
        try:
            manager = self.get_session_manager()
            fs = manager._get_context_fs() if callable(manager._get_context_fs) else self._get_context_fs()
            store = self._get_outbox_store(fs) if fs is not None else None
            paused = ("PROJECT_GRAPH",) if (_HAS_SQL and isinstance(store, SQLOutboxStore)
                                             and not self._graph_enabled()) else ()
            return projection_outcome(outcome, store, paused_event_types=paused)
        except Exception as exc:
            return normalize_write_result({**outcome, "projection_status": "UNKNOWN",
                                           "request_status": "DEGRADED" if outcome.get("semantic_status") == "DEGRADED"
                                           else "PROJECTING", **error_fields(exc, outcome.get("request_id", ""))})

    def get_write_request_status(self, params: dict) -> dict:
        """Read the owner-scoped durable receipt and its exact projection events."""
        import json
        from core.errors import NodeNotFoundError

        session_id = str(params.get("sessionId") or params.get("session_id") or "").strip()
        request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
        if not session_id or not request_id or len(request_id) > 128:
            raise ValueError("session_id and client_request_id are required")
        ctx = params.get("_ctx") or self.build_context(params)
        if ctx.session_id != session_id:
            raise ValueError("request context must match its session")
        manager = self.get_session_manager()
        fs = manager._get_context_fs() if callable(manager._get_context_fs) else self._get_context_fs()
        if fs is None:
            raise RecoveryNeedsConfirmationError("durable request state unavailable")
        payload = None
        for uri in (manager._session_state.state_uri(ctx.account_id, session_id),
                    manager._session_state.legacy_state_uri(ctx.account_id, session_id)):
            try:
                node = fs.read_node(uri, ctx)
            except (NodeNotFoundError, FileNotFoundError):
                continue
            if (node.category != "state" or node.owner_space != f"session:{session_id}"
                    or node.metadata.get("owner_user_id") != ctx.user_id
                    or node.metadata.get("owner_agent_id") != ctx.agent_id):
                raise AccessDeniedError(uri, ctx.account_id, "request receipt owner differs")
            payload = json.loads(node.content)
            break
        extensions = (payload or {}).get("runtime_extensions", {})
        receipt = extensions.get(OUTCOMES_KEY, {}).get("entries", {}).get(request_id)
        batch = extensions.get("memory_extraction_recovery", {}).get("batch", {})
        owns_batch = batch.get("client_request_id") == request_id
        buffered_batch = (receipt is not None and receipt.get("source_status") == "BUFFERED"
                          and (owns_batch or set(receipt.get("source_message_ids", [])).difference(
                              receipt.get("archived_source_ids", [])).intersection(batch.get("message_ids", []))))
        if (receipt is None and owns_batch) or buffered_batch:
            task = batch.get("retry_task", {})
            lifecycle = batch.get("lifecycle", {})
            failure = lifecycle.get("last_error", {})
            if failure.get("status") not in {"needs_confirmation", "failed"}:
                failure = {}
            if failure:
                state = "NEEDS_CONFIRMATION" if failure["status"] == "needs_confirmation" else "FAILED"
            elif task.get("status") == "needs_confirmation":
                state = "NEEDS_CONFIRMATION"
            else:
                state = "RECOVERING" if task else "RUNNING"
            detail = (failure or task).get("error_detail")
            if (state == "FAILED" and lifecycle.get("kind") == "dispose" and failure.get("retryable")
                    and detail and detail.get("retry_action") == "retry_same_request"):
                detail = {**detail, "retry_action": "retry_dispose"}
            return normalize_write_result({**(receipt or {}), **failure,
                "request_status": state, "request_id": request_id, "retryable": False,
                "source_status": "CHECKPOINTED", "semantic_status": "PENDING",
                "archive_id": batch.get("archive_id"), "accepted": True,
                **({"error_detail": detail} if detail else {})})
        if receipt is not None:
            return self._resolve_write_outcome(receipt)
        return normalize_write_result({"request_status": "UNKNOWN", "request_id": request_id,
                                       "accepted": request_id in (payload or {}).get("completed_request_ids", [])})

    @write_request_contract
    @capture_outbox_events
    @retry_failed_source
    @_after_turn_deadline
    @_attribute_request_usage
    @_guard_session_mutation
    def after_turn(self, params: dict) -> dict:
        messages = params.get("messages", [])
        if not messages:
            return {"ok": True}

        ctx = params.get("_ctx") or self.build_context(params)
        try:
            self._load_extraction_recovery_before_accumulation(params.get("sessionId", "unknown"), ctx)
        except SessionOperationInProgress:
            raise
        except Exception as exc:
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "error": "durable extraction state unavailable: " + str(exc)}

        client_request_id = str(
            params.get("clientRequestId") or params.get("client_request_id") or ""
        ).strip()
        if not client_request_id:
            return self._after_turn(params)
        if len(client_request_id) > 128:
            raise ValueError("client_request_id must be at most 128 characters")

        buf = self.get_session_manager().get_or_create(
            params.get("sessionId", "unknown"),
            ctx=ctx,
        )
        claim_status = buf.claim_request(client_request_id)
        if claim_status == "completed":
            return self._after_turn(params)
        if claim_status == "processing":
            batch = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
            if batch.get("client_request_id") == client_request_id:
                return {"ok": True, "status": "processing", "client_request_id": client_request_id,
                        "request_status": "RUNNING", "source_status": "CHECKPOINTED", "accepted": True}
            return {
                "ok": False,
                "status": "request_in_progress",
                "request_status": "FAILED",
                "accepted": False,
                "http_status": 409,
                "error_detail": {"code": "WRITE_REQUEST_IN_PROGRESS", "component": "memory",
                                 "request_id": client_request_id, "retry_action": "retry_same_request"},
                "retryable": True,
                "client_request_id": client_request_id,
            }

        try:
            result = self._after_turn(params)
        except Exception:
            buf.release_request(client_request_id)
            raise
        if result.get("ok", True) is False or result.get("status") == "queued":
            buf.release_request(client_request_id)
        return result

    def _after_turn(self, params: dict) -> dict:
        return self._after_turn_impl(params)

    def _after_turn_impl(self, params: dict) -> dict:
        """Reserve ordinary source preparation without serializing ordinary turns."""
        if not params.get("messages"):
            return self._after_turn_source_impl(params)
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        buf = self.get_session_manager().get_or_create(session_id, ctx=ctx)
        with buf.extraction_lock:
            if buf.lifecycle_prepare_in_progress:
                return {"ok": False, "status": "processing", "retryable": True,
                        "reason": "lifecycle_preparing",
                        **error_fields(SessionOperationInProgress(conflict_type="lifecycle_preparing", retryable=True),
                                       str(params.get("clientRequestId") or params.get("client_request_id") or ""))}
            if lifecycle_kind(buf) is not None:
                return {"ok": False, "status": "needs_confirmation", "retryable": False,
                        "error": "resume the pending lifecycle batch before after_turn"}
            buf.ordinary_turns_in_progress += 1
        try:
            return self._after_turn_source_impl(params)
        finally:
            with buf.extraction_lock:
                buf.ordinary_turns_in_progress -= 1


    def _after_turn_source_impl(self, params: dict) -> dict:
        """Unified write entry point: accumulate session buffer + extract memories.

        Step 1 (every turn): add new messages to session buffer (lightweight).
        Step 2 (threshold reached): extract memories via LLM → write to AGFS →
              drain outbox (embed + upsert vector index) → commit session archive.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if not messages:
            return {"ok": True}

        ctx = params.get("_ctx") or self.build_context(params)
        client_request_id = str(
            params.get("clientRequestId") or params.get("client_request_id") or ""
        ).strip()
        automatic_card_identity = (self._cfg.card_format_enabled or self._cfg.node_processing_enabled) and not client_request_id
        if automatic_card_identity:
            client_request_id = "card-" + uuid4().hex
            params = {**params, "clientRequestId": client_request_id}
        if len(client_request_id) > 128:
            raise ValueError("client_request_id must be at most 128 characters")
        pre_prompt_count = _bounded_message_count(
            params.get("prePromptMessageCount", 0),
            len(messages),
        )
        if (
            not client_request_id and (not messages[pre_prompt_count:])
        ):
            return {"ok": True, "status": "no_new_messages"}

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)

        if lifecycle_kind(buf) is not None:
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "error": "resume the pending lifecycle batch before after_turn"}

        from service.api import MemoryWriteAPI as _CheckpointWriteAPI

        def _save_idempotency_state() -> bool:
            if (
                not getattr(self._cfg, 'session_state_bridge_enabled', True) and (not isinstance(self._write_api, _CheckpointWriteAPI))
            ):
                return True
            return mgr.save_session_state(session_id, ctx)

        request_source_ids = set()
        completed_outcome = {}

        def _complete_request(result: dict) -> dict:
            if client_request_id and result.get("ok", True):
                previous_completed = list(buf.completed_request_ids)
                previous_outcomes = deepcopy(buf.runtime_extensions.get(OUTCOMES_KEY))
                receipt = buf.runtime_extensions.get(OUTCOMES_KEY, {}).get("entries", {}).get(client_request_id)
                if receipt is None:
                    receipt = {"request_id": client_request_id, "request_status": "ACCEPTED",
                               "source_status": "BUFFERED", "semantic_status": "PENDING",
                               "projection_status": "UNKNOWN", "accepted": True,
                               "source_message_ids": sorted(request_source_ids)}
                buf.mark_request_completed(client_request_id)
                receipt = store_outcome(buf, client_request_id, receipt,
                                        resolve_outcome=self._resolve_write_outcome)
                result["client_request_id"] = client_request_id
                if not _save_idempotency_state():
                    buf.completed_request_ids = previous_completed
                    if previous_outcomes is None:
                        buf.runtime_extensions.pop(OUTCOMES_KEY, None)
                    else:
                        buf.runtime_extensions[OUTCOMES_KEY] = previous_outcomes
                    return {
                        "ok": False,
                        "status": "failed",
                        "error": "idempotency state unavailable",
                        "retryable": True,
                        "client_request_id": client_request_id,
                    }
                buf.release_request(client_request_id)
                result.update(deepcopy(receipt))
            elif completed_outcome and request_source_ids <= archive_snapshot_ids:
                result.update(completed_outcome)
            elif completed_outcome:
                result.update(status="accumulating", request_status="ACCEPTED",
                              source_status="BUFFERED", semantic_status="PENDING")
            return result
        pending = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
        pending_request_retry = bool(client_request_id and pending.get("client_request_id") == client_request_id)
        if client_request_id and buf.request_completed(client_request_id) and not pending_request_retry:
            if not _save_idempotency_state():
                return {
                    "ok": False,
                    "status": "failed",
                    "error": "idempotency state unavailable",
                    "retryable": True,
                    "client_request_id": client_request_id,
                }
            buf.release_request(client_request_id)
            return {
                "ok": True,
                "status": "duplicate",
                "client_request_id": client_request_id,
                **deepcopy(buf.runtime_extensions.get(OUTCOMES_KEY, {}).get("entries", {}).get(client_request_id, {})),
            }

        if pending_request_retry and is_frozen_source_retry(ctx, pending):
            new_messages = deepcopy(pending["request_messages"])
        else:
            new_messages = messages[pre_prompt_count:]
        if pending_request_retry and pending.get("request_messages") != new_messages:
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "error": "the pending request source messages changed"}

        if _ambiguous_pending_replay(pending, new_messages, client_request_id):
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "error": "a stable source ID is required to distinguish a pending replay from a new message"}
        if not new_messages:
            return _complete_request({"ok": True, "status": "no_new_messages"})

        request_session_time = None
        for msg in new_messages:
            created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
            request_session_time = _parse_session_time(created_at)
            if request_session_time is not None:
                break

        # Threshold comes from unified config first; env fallback is handled by OgMemConfig.
        threshold = _positive_int(getattr(self._cfg, "after_turn_threshold", 400), 400)

        # Step 1: Accumulate in session buffer (lightweight, every turn)
        # Dedup: skip messages already written by ingest()
        from extraction.tool_collector import update_buffer_stats
        if not pending_request_retry:
            # Snapshot buffer tail once to avoid TOCTOU race with concurrent ingest().
            buf_snapshot = list(buf.messages[-len(new_messages):]) if buf.messages else []
            for index, msg in enumerate(new_messages):
                role = msg.get("role", "user")
                content_raw = msg.get("content", "")
                parsed = extract_content_and_tool_calls(content_raw)
                content = parsed["text"]
                message_id = msg.get("id") or (request_message_id(ctx, client_request_id, index)
                                              if client_request_id else None)
                if content and buf_snapshot and not msg.get("id"):
                    for existing in buf_snapshot:
                        if (existing.id not in buf.pending_extraction_ids()
                                and (existing.id in buf.ingest_handoff_ids
                                     or not existing.id.startswith("msg_request_"))
                                and existing.role == role and existing.content == content):
                            buf.ingest_handoff_ids.discard(existing.id)
                            request_source_ids.add(existing.id)
                            buf_snapshot.remove(existing)
                            content = None
                            break
                if content:
                    created_at = (
                        msg.get("created_at")
                        if isinstance(msg, dict)
                        else getattr(msg, "created_at", None)
                    )
                    added = mgr.add_message(
                        session_id,
                        role,
                        content,
                        ctx,
                        created_at=created_at,
                        message_id=message_id,
                    )
                    buf.ingest_handoff_ids.discard(added["message_id"])
                    request_source_ids.add(added["message_id"])
                if parsed["tool_calls"]:
                    update_buffer_stats(
                        buf.usage_stats,
                        parsed["tool_calls"],
                        session_id=session_id,
                    )


        # Check pending_tokens
        session = mgr.get_session(session_id, ctx)
        pending_tokens = session.get("pending_tokens", 0)
        force_extract = self._param_bool(
            params, "forceExtract", "force_extract", default=False
        )

        if pending_tokens < threshold and not force_extract and not buf.pending_extraction_ids():
            return _complete_request(
                {"ok": True, "status": "accumulating", "pending_tokens": pending_tokens,
                 "source_status": "BUFFERED", "semantic_status": "PENDING"}
            )

        # Step 2: Threshold reached → extract memories + write + index + archive
        check_deadline()
        write_api = self.get_write_api()
        if write_api is None:
            logger.warning("after_turn: write API unavailable, session commit only")
            result = mgr.commit(session_id, ctx, wait=bool(client_request_id))
            return _complete_request({"ok": True, **result})

        checkpoint_enabled = isinstance(write_api, _CheckpointWriteAPI)
        if checkpoint_enabled:
            mgr.enable_extraction_recovery(session_id, ctx, write_api._fs)
            if client_request_id and buf.request_completed(client_request_id):
                return _complete_request({"ok": True, "status": "duplicate"})

        # Extract candidates → filter → write to AGFS → create OutboxEvents
        # Use INCREMENTAL messages (after watermark) + extraction summary
        buf = mgr.get_or_create(session_id, ctx=ctx)
        extraction_run_id = uuid4().hex[:12]
        extraction_claim = buf.claim_incremental_extraction()
        extraction_in_progress_before = extraction_claim["extraction_in_progress_before"]
        incremental = extraction_claim["incremental"]
        if not incremental:
            if extraction_in_progress_before:
                return _complete_request({
                    "ok": False,
                    "status": "processing",
                    "retryable": True,
                    "pending_tokens": pending_tokens,
                    "client_request_id": client_request_id,
                })
            return _complete_request(
                {
                    "ok": True,
                    "status": "no_new_messages",
                    "pending_tokens": pending_tokens,
                }
            )

        try:
            extraction_state = self._build_incremental_extraction_state(
                buf,
                incremental=incremental,
            )
            check_deadline()
        except Exception:
            buf.rewind_watermark_to_ids({message.id for message in incremental})
            if extraction_claim["extraction_started"]:
                buf.end_extraction()
            raise
        archive_snapshot = list(incremental)
        archive_snapshot_ids = {m.id for m in archive_snapshot}
        extraction_messages = extraction_state["messages"]
        original_extraction_watermark = extraction_claim["original_watermark"]

        # Pre-generate archive_id so extraction and commit_snapshot share the
        # same provenance link. commit_session auto-generates when archive_id
        # is None, but after_turn must pass it explicitly across both calls.
        from session.session_manager import generate_archive_id
        archive_id = generate_archive_id()
        extraction_journal = None
        if checkpoint_enabled:
            try:
                extraction_journal = ExtractionBatchJournal(
                    mgr, buf, ctx, archive_snapshot, archive_id=archive_id,
                    messages=extraction_messages,
                    session_time=extraction_state["session_time"] or request_session_time,
                    session_summary=extraction_claim["extraction_summary"],
                    tool_stats_text=extraction_state["tool_stats_text"],
                    client_request_id=client_request_id or "extraction-retry-" + archive_id,
                    request_messages=new_messages,
                    request_source_ids=sorted(request_source_ids))
                archive_id = extraction_journal.batch["archive_id"]
                if not extraction_journal.batch.get("client_request_id"):
                    extraction_journal.mutate(lambda batch: batch.update(
                        client_request_id="extraction-retry-" + archive_id))
            except Exception as exc:
                buf.rewind_watermark_to_ids(archive_snapshot_ids)
                if extraction_claim["extraction_started"]:
                    buf.end_extraction()
                return {"ok": False, "status": "needs_confirmation" if isinstance(exc, RecoveryNeedsConfirmationError) else "failed",
                        "error": str(exc), "retryable": not isinstance(exc, RecoveryNeedsConfirmationError)}

        def _finish_archived_batch(write_result=None):
            write_result = write_result or {}
            if extraction_journal is not None:
                degraded = bool(write_result.get("degraded") or write_result.get("recovery_mode") == "source_only")
                outcome = {"semantic_status": "DEGRADED" if degraded else "COMPLETE",
                           "request_status": "DEGRADED" if degraded else "PROJECTING",
                           "accepted": True,
                           "writes_completed": write_result.get("writes_completed", 0),
                           "candidates_extracted": write_result.get("candidates_extracted", 0)}
                if degraded:
                    outcome["recovery_mode"] = "source_only"
                    outcome["error_detail"] = {"code": "EXTRACTION_SOURCE_ONLY", "component": "extraction",
                        "request_id": extraction_journal_request_id, "retry_action": "manual_review",
                        "cause": {"message": str(write_result.get("reason", "source preserved after recovery"))[:1000]}}
                completed_outcome.update(extraction_journal.finish(
                    outcome=outcome, resolve_outcome=self._resolve_write_outcome,
                    current_request_id=client_request_id, current_source_ids=request_source_ids))
                return len(archive_snapshot)
            removed = buf.remove_messages_by_id(archive_snapshot_ids)
            if removed:
                mgr.save_session_state(session_id, ctx)
            return removed

        def _completion_status():
            if extraction_journal is not None and request_source_ids - archive_snapshot_ids:
                return "accumulating"
            return "completed"

        extraction_journal_request_id = (extraction_journal.batch.get("client_request_id", "")
                                         if extraction_journal is not None else client_request_id)

        def _rewind_failed_archive_watermark() -> bool:
            rewound = buf.rewind_watermark_to_ids(archive_snapshot_ids)
            if not rewound:
                buf.extraction_watermark = min(original_extraction_watermark, len(buf.messages))
            return rewound

        logger.info(
            "after_turn extraction scheduled: run=%s session=%s "
            "pending_tokens=%d buffer_len=%d watermark=%d incremental=%d "
            "extractable=%d active_extractions=%d",
            extraction_run_id,
            session_id[:8],
            pending_tokens,
            len(buf.messages),
            original_extraction_watermark,
            len(incremental),
            len(extraction_messages),
            getattr(buf, "extraction_active_count", 0),
        )
        if extraction_in_progress_before:
            logger.warning(
                "after_turn overlapping extraction detected: run=%s session=%s "
                "buffer_len=%d watermark=%d active_extractions=%d",
                extraction_run_id,
                session_id[:8],
                len(buf.messages),
                buf.extraction_watermark,
                getattr(buf, "extraction_active_count", 0),
            )
        if not extraction_messages:
            try:
                check_deadline()
                with mgr.parallel_archive_compression(
                    archive_snapshot, session_id, ctx, extraction_journal, archive_id=archive_id,
                ) as compression_work:
                    drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
                    check_deadline()
                commit_result = mgr.commit_snapshot(
                    session_id, archive_snapshot, ctx, wait=True, archive_id=archive_id,
                    compression_work=compression_work, compression_checkpoint=extraction_journal)
            except OperationDeadlineExceededError:
                _rewind_failed_archive_watermark()
                if extraction_claim["extraction_started"]:
                    buf.end_extraction()
                raise
            removed = 0
            if commit_result.get("archived"):
                removed = _finish_archived_batch()
            logger.info(
                "after_turn no-extractable snapshot commit: run=%s session=%s "
                "archived=%s removed=%d reason=%s",
                extraction_run_id,
                session_id[:8],
                commit_result.get("archived"),
                removed,
                commit_result.get("reason", ""),
            )
            if commit_result.get("status") in {"failed", "needs_confirmation"}:
                rewound = _rewind_failed_archive_watermark()
                logger.error(
                    "after_turn no-extractable snapshot commit failed: "
                    "run=%s session=%s rewound=%s error=%s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    commit_result.get("error", "archive commit failed"),
                )
                return {
                    "ok": False,
                    "status": commit_result.get("status", "failed"),
                    "error": commit_result.get("error", "archive commit failed"),
                    "drain": drain_stats,
                    "commit": commit_result,
                    "extraction_run_id": extraction_run_id,
                }
            return _complete_request(
                {
                    "ok": True,
                    "status": "no_extractable_messages",
                    "drain": drain_stats,
                    "commit": commit_result,
                    "extraction_run_id": extraction_run_id,
                }
            )

        # Keep the upstream async after_turn behavior, but source extraction
        # inputs from the shared incremental-state helper so after_turn and
        # prepare_compaction operate on the same delta semantics.
        tool_stats_text = extraction_state["tool_stats_text"]
        session_time = extraction_state["session_time"] or request_session_time
        extraction_watermark = original_extraction_watermark
        extraction_summary = extraction_claim["extraction_summary"]

        import time as _time

        def _background_extract_write():
            start_time = _time.monotonic()
            try:
                with mgr.parallel_archive_compression(
                    archive_snapshot, session_id, ctx, extraction_journal, archive_id=archive_id,
                ) as compression_work:
                    write_result = self._chunk_and_extract(
                        messages=extraction_messages,
                        write_api=write_api,
                        ctx=ctx,
                        session_time=session_time,
                        session_summary=extraction_summary,
                        tool_stats_text=tool_stats_text,
                        extraction_run_id=extraction_run_id,
                        session_id=session_id,
                        archive_id=archive_id,
                        extraction_journal=extraction_journal,
                    )
                if self._write_result_has_failures(write_result):
                    raise RuntimeError("memory extraction write failed")
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                elapsed_ms = int((_time.monotonic() - start_time) * 1000)
                logger.info(
                    "after_turn background extraction writes ready: run=%s session=%s "
                    "extracted=%d filtered=%d writes=%d skipped=%d failed=%d "
                    "watermark=%d->%d buffer_len_now=%d elapsed_ms=%d",
                    extraction_run_id,
                    session_id[:8],
                    write_result.get("candidates_extracted", 0),
                    write_result.get("candidates_filtered", 0),
                    write_result.get("writes_completed", 0),
                    write_result.get("writes_skipped", 0),
                    write_result.get("writes_failed", 0),
                    extraction_watermark,
                    buf.extraction_watermark,
                    len(buf.messages),
                    elapsed_ms,
                )
                if write_result.get("candidates_extracted", 0) == 0:
                    logger.warning(
                        "after_turn zero-candidate diagnostic: run=%s session=%s "
                        "extractable=%d extraction_diag=%s write_result_keys=%s",
                        extraction_run_id,
                        session_id[:8],
                        len(extraction_messages),
                        _message_diag(extraction_messages),
                        sorted(write_result.keys()),
                    )

                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(
                        buf.extraction_summary, write_result
                    )

                try:
                    self._async_drain(account_id=ctx.account_id)
                except Exception as exc:
                    logger.warning("outbox_drain failed after background extract: %s", exc)

                commit_result = mgr.commit_snapshot(
                    session_id, archive_snapshot, ctx, wait=True, archive_id=archive_id,
                    compression_work=compression_work, compression_checkpoint=extraction_journal)
                removed = 0
                if commit_result.get("status") in {"failed", "needs_confirmation"}:
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn background snapshot commit failed: "
                        "run=%s session=%s rewound=%s error=%s",
                        extraction_run_id,
                        session_id[:8],
                        rewound,
                        commit_result.get("error", "archive commit failed"),
                    )
                elif commit_result.get("archived"):
                    removed = _finish_archived_batch(write_result)
                logger.info(
                    "after_turn background snapshot commit done: run=%s session=%s "
                    "archived=%s reason=%s removed=%d buffer_len=%d watermark=%d",
                    extraction_run_id,
                    session_id[:8],
                    commit_result.get("archived"),
                    commit_result.get("reason", ""),
                    removed,
                    len(buf.messages),
                    buf.extraction_watermark,
                )
                if commit_result.get("status") == "completed" and commit_result.get("archived"):
                    _complete_request({"ok": True, "status": "completed"})
                    # Bench clients use this marker as a completion barrier.
                    # Emitting it before commit_snapshot could announce
                    # success even when durable session archival later failed.
                    logger.info(
                        "after_turn background extract done: run=%s session=%s "
                        "extracted=%d writes=%d archived=%s elapsed_ms=%d",
                        extraction_run_id,
                        session_id[:8],
                        write_result.get("candidates_extracted", 0),
                        write_result.get("writes_completed", 0),
                        commit_result.get("archived"),
                        int((_time.monotonic() - start_time) * 1000),
                    )
            except Exception as exc:
                rewound = buf.rewind_watermark_to_ids(archive_snapshot_ids)
                logger.error(
                    "after_turn background extract failed for run=%s session=%s "
                    "rewound=%s: %s",
                    extraction_run_id, session_id[:8], rewound, exc, exc_info=True,
                )
                if (isinstance(exc, (ExtractionIncompleteError, ToolCallProtocolError, SessionMutationBusyError))
                        and not isinstance(exc, OperationDeadlineExceededError)):
                    self._extraction_retry.register_background_failure(ctx, extraction_journal, exc)
            finally:
                buf.end_extraction()

        @capture_outbox_events
        def _background_owned():
            """The child acquires its own connection after the HTTP handoff."""
            nonlocal extraction_journal
            ran = False
            try:
                with self._session_mutation_scope(ctx) as guard:
                    if guard is not None and extraction_journal is not None:
                        latest = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
                        if latest.get("archive_id") != archive_id:
                            if archive_snapshot_ids.intersection(message.id for message in buf.messages):
                                raise RecoveryNeedsConfirmationError("the scheduled source batch changed")
                            # Another fenced worker durably finished the original source.
                            return
                    if extraction_journal is not None:
                        extraction_journal = ExtractionBatchJournal.resume(mgr, buf, ctx)
                    ran = True
                    _background_extract_write()
                if guard is not None and guard.post_commit_drain:
                    self._async_drain(account_id=ctx.account_id)
            except Exception as exc:
                logger.error("fenced background extraction stopped session=%s: %s", session_id[:8], exc)
                self._extraction_retry.wake(ctx, pending_task or {})
            finally:
                if not ran:
                    buf.end_extraction()

        def _extract_synchronously():
            try:
                recovering_source = (extraction_journal is not None
                                     and is_frozen_source_retry(ctx, extraction_journal.batch))
                check_deadline()
                with mgr.parallel_archive_compression(
                    archive_snapshot, session_id, ctx, extraction_journal, archive_id=archive_id,
                ) as compression_work:
                    write_result = self._chunk_and_extract(
                        messages=extraction_messages,
                        write_api=write_api,
                        ctx=ctx,
                        session_time=session_time,
                        session_summary=extraction_summary,
                        tool_stats_text=tool_stats_text,
                        extraction_run_id=extraction_run_id,
                        session_id=session_id,
                        archive_id=archive_id,
                        extraction_journal=extraction_journal,
                    )
                if self._write_result_has_failures(write_result):
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn wait extraction write failed: "
                        "run=%s session=%s rewound=%s write_result=%s",
                        extraction_run_id,
                        session_id[:8],
                        rewound,
                        write_result,
                    )
                    return {
                        **write_result,
                        "ok": False,
                        "status": "failed",
                        "error": "memory extraction write failed",
                        "retryable": (extraction_journal is not None
                                      and extraction_journal.has_retryable_write_failure()),
                        "commit": None,
                        "extraction_run_id": extraction_run_id,
                    }
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(
                        buf.extraction_summary, write_result
                    )
                check_deadline()
                drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
                check_deadline()
                commit_result = mgr.commit_snapshot(
                    session_id,
                    archive_snapshot,
                    ctx,
                    wait=True,
                    archive_id=archive_id,
                    compression_work=compression_work,
                    compression_checkpoint=extraction_journal,
                )
                if commit_result.get("status") in {"failed", "needs_confirmation"}:
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn wait snapshot commit failed: "
                        "run=%s session=%s rewound=%s error=%s",
                        extraction_run_id,
                        session_id[:8],
                        rewound,
                        commit_result.get("error", "archive commit failed"),
                    )
                    return {
                        **write_result,
                        "ok": False,
                        "status": commit_result.get("status", "failed"),
                        "error": commit_result.get("error", "archive commit failed"),
                        "drain": drain_stats,
                        "commit": commit_result,
                        "extraction_run_id": extraction_run_id,
                    }
                if commit_result.get("archived"):
                    _finish_archived_batch(write_result)
                result = _complete_request(
                    {
                        "ok": True,
                        **write_result,
                        "status": _completion_status(),
                        "extraction_run_id": extraction_run_id,
                        "drain": drain_stats,
                        "commit": commit_result,
                    }
                )
                if result.get("status") == "completed" and recovering_source:
                    logger.info(
                        "after_turn background extract done: run=%s session=%s "
                        "archive=%s request=%s extracted=%d writes=%d archived=%s recovery=true",
                        extraction_run_id, session_id[:8], archive_id,
                        extraction_journal_request_id,
                        write_result.get("candidates_extracted", 0),
                        write_result.get("writes_completed", 0), commit_result.get("archived"),
                    )
                return result
            except Exception as exc:
                rewound = _rewind_failed_archive_watermark()
                if (isinstance(exc, SessionMutationBusyError) and extraction_journal is not None
                        and is_frozen_source_retry(ctx, extraction_journal.batch)):
                    return {"ok": False, "status": "processing", "retryable": True,
                            "error": str(exc), "extraction_run_id": extraction_run_id}
                queued = self._queue_extraction_recovery(
                    ctx, exc, archive_id=archive_id, client_request_id=extraction_journal_request_id)
                if queued is not None:
                    return {**queued, "extraction_run_id": extraction_run_id}
                logger.error(
                    "after_turn wait extraction failed for run=%s session=%s "
                    "rewound=%s: %s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    exc,
                    exc_info=True,
                )
                return {
                    "ok": False,
                    "status": "needs_confirmation" if isinstance(exc, (RecoveryNeedsConfirmationError, WriteOutcomeUnknownError)) else "failed",
                    "error": str(exc),
                    "retryable": not isinstance(exc, (RecoveryNeedsConfirmationError, WriteOutcomeUnknownError)),
                    "extraction_run_id": extraction_run_id,
                    "failure_kind": ("memory_card_format" if isinstance(exc, MemoryCardFormatError) else
                                     "llm_transient" if isinstance(exc, LLMTransientError) else ""),
                    **error_fields(exc, extraction_journal_request_id),
                }
            finally:
                buf.end_extraction()

        if (client_request_id and not automatic_card_identity) or self._param_bool(params, "wait", "wait", default=False):
            result = _extract_synchronously()
            if result.get("ok") is False and extraction_journal is not None:
                result.update(extraction_archive_id=archive_id,
                              client_request_id=extraction_journal_request_id)
            return result

        try:
            pending_task = self._extraction_retry.register_pending_source(ctx, extraction_journal)
            if extraction_journal is not None:
                extraction_journal_request_id = extraction_journal.batch.get("client_request_id", "")
        except Exception as exc:
            _rewind_failed_archive_watermark()
            buf.end_extraction()
            logger.exception("Cannot register background extraction session=%s", session_id[:8])
            return {"ok": False, "status": "failed", "retryable": True, "error": str(exc),
                    "extraction_archive_id": archive_id, "client_request_id": extraction_journal_request_id}
        try:
            t = threading.Thread(
                target=lambda: _run_with_usage_scope(ctx, _background_owned),
                daemon=True,
                name=f"extract-{session_id[:8]}",
            )
            t.start()
        except Exception as exc:
            logger.error(
                "Failed to spawn background extract thread: run=%s session=%s %s",
                extraction_run_id,
                session_id[:8],
                exc,
                exc_info=True,
            )
            return _extract_synchronously()

        return {"ok": True, "status": "processing", "extraction_run_id": extraction_run_id,
                "request_id": extraction_journal_request_id}


    @_attribute_request_usage
    @_guard_session_mutation
    def ingest(self, params: dict) -> dict:
        """Ingest messages into the session buffer.

        Writes messages to the session buffer so that compact() can find data
        even when called before after_turn().  after_turn() will deduplicate
        messages already written here.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if not messages:
            return {"ingested": True}
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            self._load_extraction_recovery_before_accumulation(session_id, ctx)
        except SessionOperationInProgress:
            raise
        except Exception as exc:
            return {"ok": False, "ingested": False, "status": "needs_confirmation", "retryable": False,
                    "error": "durable extraction state unavailable: " + str(exc)}
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.extraction_lock:
            if buf.lifecycle_prepare_in_progress:
                return {"ok": False, "ingested": False, "status": "processing", "retryable": True,
                        "reason": "lifecycle_preparing"}
            if lifecycle_kind(buf) is not None:
                return {"ok": False, "ingested": False, "status": "needs_confirmation", "retryable": False,
                        "error": "resume the pending lifecycle batch before ingest"}
            buf.begin_operation()
            buf.ordinary_turns_in_progress += 1
        try:
            return self._ingest_source_impl(params, messages, ctx, session_id, mgr)
        finally:
            with buf.extraction_lock:
                buf.ordinary_turns_in_progress -= 1
                buf.end_operation()


    def _ingest_source_impl(self, params, messages, ctx, session_id, mgr):
        client_request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
        if len(client_request_id) > 128:
            raise ValueError("client_request_id must be at most 128 characters")
        buf = mgr.get_or_create(session_id, ctx=ctx)
        pending = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
        if client_request_id and pending.get("client_request_id") == client_request_id:
            if pending.get("request_messages") == messages:
                return {"ingested": True, "count": 0}
            return {"ok": False, "ingested": False, "status": "needs_confirmation", "retryable": False,
                    "error": "the pending request source messages changed"}
        if _ambiguous_pending_replay(pending, messages, client_request_id):
            return {"ok": False, "ingested": False, "status": "needs_confirmation", "retryable": False,
                    "error": "ingest requires a stable source ID to distinguish a pending replay from a new message"}
        count = 0
        for index, msg in enumerate(messages):
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content_raw = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            content = extract_content_and_tool_calls(content_raw).get("text", "")
            created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
            if content:
                message_id = (msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None))
                message_id = message_id or (request_message_id(ctx, client_request_id, index)
                                            if client_request_id else None)
                added = mgr.add_message(session_id, role, content, ctx, created_at=created_at, message_id=message_id)
                with buf.extraction_lock:
                    buf.ingest_handoff_ids.add(added["message_id"])
                count += 1
        return {"ingested": True, "count": count}


    @_attribute_request_usage
    def ingest_batch(self, params: dict) -> dict:
        """Batch ingest — same as ingest() for multiple messages."""
        return self.ingest(params)

    @capture_outbox_events
    @_guard_session_mutation
    def compact(self, params: dict) -> dict:
        """Run the complete compaction flow under the shared session operation guard."""
        if params.get("prepared") and not params.get("prepareToken"):
            raise ValueError("prepare token required when prepared=true")
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            self._load_extraction_recovery_before_accumulation(session_id, ctx)
        except SessionOperationInProgress:
            raise
        except Exception as exc:
            return {"ok": False, "compacted": False, "status": "needs_confirmation",
                    "retryable": False, "error": "durable extraction state unavailable: " + str(exc)}
        buf = self.get_session_manager().begin_managed_operation(session_id, ctx)
        try:
            with request_usage_scope(ctx), buf.lifecycle_lock:
                return self._compact_impl(params)
        except UsagePersistenceError as exc:
            return {"compacted": False, **self._lifecycle_usage_failure(exc, params, buf, ctx)}
        except (RecoveryNeedsConfirmationError, WriteOutcomeUnknownError) as exc:
            return {"ok": False, "compacted": False, "status": "needs_confirmation",
                    "retryable": False, "error": str(exc)}
        except RecoveryPersistenceError as exc:
            return {"ok": False, "compacted": False, "status": "failed", "retryable": True, "error": str(exc)}
        finally:
            buf.end_operation()

    def _lifecycle_usage_failure(self, exc, params, buf, ctx):
        request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "")
        batch = buf.runtime_extensions.get(EXTENSION_KEY, {}).get("batch", {})
        checkpointed = isinstance(batch, dict) and batch.get("owner") == recovery_owner(ctx)
        lifecycle = batch.get("lifecycle", {}) if checkpointed else {}
        if lifecycle.get("kind") == "compaction":
            journal = LifecycleBatchJournal.resume(self.get_session_manager(), buf, ctx)
            self._update_lifecycle_error(journal, exc, retryable=False)
        logger.exception("Lifecycle model usage persistence failed: session=%s", ctx.session_id)
        return normalize_write_result({
            "ok": False, "status": "failed", "error": str(exc), "retryable": False,
            "request_status": "FAILED", "accepted": checkpointed,
            "source_status": "CHECKPOINTED" if checkpointed else "UNKNOWN",
            "semantic_status": lifecycle.get("semantic_status", "UNKNOWN"), "projection_status": "UNKNOWN",
            "archive_id": batch.get("archive_id") if checkpointed else None,
            **error_fields(exc, request_id),
        }, request_id)

    def _compact_impl(self, params: dict) -> dict:
        """Compact session: commit synchronously, return compressed context."""
        prepare_token = params.get("prepareToken")
        prepared_flag = bool(params.get("prepared"))
        if prepare_token:
            prepared = self._consume_prepared_compaction_state(params, str(prepare_token))
        elif prepared_flag:
            raise ValueError("prepare token required when prepared=true")
        else:
            prepared = self.prepare_compaction(params)
        session_id = prepared["session_id"]
        ctx = prepared["ctx"]
        token_budget = params.get("tokenBudget", 128_000)
        mgr = prepared["mgr"]

        # Estimate tokensBefore
        session = mgr.get_session(session_id, ctx)
        tokens_before = session.get("pending_tokens", 0)

        # Commit synchronously (wait=true)
        journal = prepared.get("_lifecycle_journal")
        degraded_receipt = {}
        if journal is not None:
            lifecycle = journal.batch["lifecycle"]
            reason = lifecycle.get("source_only_reason")
            scope = source_only_recovery(ctx, journal.batch["archive_id"], reason) if reason else nullcontext()
            with scope:
                commit_result = mgr.commit_snapshot(session_id, journal.snapshot, ctx, wait=True,
                                                     archive_id=journal.batch["archive_id"],
                                                     archive_checkpoint=journal)
            if commit_result.get("archived") and commit_result.get("status") == "completed":
                try:
                    outcome = self._lifecycle_extraction_outcome(lifecycle)
                    receipt = journal.finish(reset_window_state=True, outcome=outcome,
                                             resolve_outcome=self._resolve_write_outcome)
                    if receipt.get("semantic_status") == "DEGRADED":
                        receipt = normalize_write_result(receipt)
                        fields = (*PUBLIC_FIELDS, "ok", "status", "recovery_mode",
                                  "source_message_ids", "archived_source_ids")
                        degraded_receipt = {key: receipt[key] for key in fields if key in receipt}
                except Exception as exc:
                    return {"ok": False, "compacted": False, "status": "failed",
                            "retryable": True, "error": str(exc), "commit": commit_result}
        else:
            commit_result = mgr.commit(
                session_id, ctx, wait=True, archive_id=prepared.get("archive_id"))

        if not commit_result.get("archived") or commit_result.get("status") in {"failed", "needs_confirmation"}:
            if commit_result.get("status") in {"failed", "needs_confirmation"}:
                return {
                    "ok": False,
                    "compacted": False,
                    "status": commit_result["status"],
                    "retryable": commit_result["status"] != "needs_confirmation",
                    "error": commit_result.get("error", "archive commit failed"),
                    "commit": commit_result,
                    "result": {
                        "summary": "",
                        "tokensBefore": tokens_before,
                    },
                }
            return {
                "ok": True,
                "compacted": False,
                "reason": commit_result.get("reason", "no_archive"),
                "result": {
                    "summary": "",
                    "tokensBefore": tokens_before,
                },
            }

        # Get compressed context (archive overview as summary)
        context = mgr.get_context(session_id, token_budget, ctx)
        summary = _trim_summary(
            context.get("latest_archive_overview", ""),
            _summary_max_chars(self._cfg, params),
        )
        tokens_after = context.get("estimatedTokens", 0)
        self._apply_compact_short_term_index_mode(params)

        result = {
            "ok": True,
            **degraded_receipt,
            "compacted": True,
            "result": {
                "summary": summary,
                "tokensBefore": tokens_before,
                "tokensAfter": tokens_after,
            },
        }
        first_kept_entry_id = (
            commit_result.get("firstKeptEntryId")
            or commit_result.get("first_kept_entry_id")
        )
        if first_kept_entry_id:
            result["result"]["firstKeptEntryId"] = first_kept_entry_id
        return result

    def _apply_compact_short_term_index_mode(self, params: dict) -> None:
        """Apply Phase 1 short-term index semantics without blocking compact success."""
        mode = _short_term_index_mode(params)
        if mode == "off":
            logger.info("compact short-term index update skipped: mode=off")
            return
        if mode == "async":
            logger.info("compact short-term index update deferred: mode=async")
            return

        try:
            ctx = params.get("_ctx") or self.build_context(params)
            drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
            logger.info(
                "compact short-term index sync processed=%d succeeded=%d failed=%d",
                drain_stats.get("processed", 0),
                drain_stats.get("succeeded", 0),
                drain_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.warning("compact short-term index sync skipped: %s", exc)

    def _build_compaction_state(self, params: dict) -> dict:
        """Build transport-agnostic compact state without mutating extraction progress."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        kind = lifecycle_kind(buf)
        if buf.pending_extraction_ids() and kind != "compaction":
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "session_id": session_id,
                    "error": "resume the pending memory batch before compaction"}
        with buf.extraction_lock:
            if kind == "compaction":
                batch = buf.runtime_extensions[EXTENSION_KEY]["batch"]
                source_snapshot = buf.capture_lifecycle_source(
                    [SessionMessage(**value) for value in batch["snapshot"]])
                extraction_state = {"incremental": source_snapshot,
                                    "messages": deepcopy(batch["messages"]),
                                    "session_time": _parse_session_time(batch.get("session_time")),
                                    "tool_stats_text": batch["tool_stats_text"]}
            else:
                source_snapshot = buf.capture_lifecycle_source()
                extraction_state = self._build_incremental_extraction_state(
                    buf, incremental=source_snapshot[buf.extraction_watermark:])

        result = {
            "session_id": session_id,
            "ctx": ctx,
            "mgr": mgr,
            "buffer": buf,
            "source_snapshot": source_snapshot,
            **extraction_state,
        }

        return result

    def _build_incremental_extraction_state(self, buf, incremental=None) -> dict:
        if incremental is None:
            incremental = buf.messages[buf.extraction_watermark:]
        extraction_messages = [
            {"role": message.role, "content": message.content, "id": message.id}
            for message in incremental
            if message.role != "assistant"
        ]
        from extraction.tool_collector import format_tool_stats_text

        session_time = None
        time_source_messages = [
            message for message in incremental
            if message.role != "assistant"
        ] or incremental or buf.messages
        for message in time_source_messages:
            session_time = _parse_session_time(getattr(message, "created_at", None))
            if session_time is not None:
                break

        # Fallback: if created_at is missing or very recent (<24h, likely auto-generated),
        # try to extract date from message content like '[group chat conversation: ...]'.
        now = _dt.now(session_time.tzinfo) if session_time and session_time.tzinfo else _dt.now()
        if session_time is None or abs((now - session_time).total_seconds()) < 86400:
            for message in time_source_messages:
                content = getattr(message, "content", "")
                extracted = _extract_date_from_content(content)
                if extracted is not None:
                    session_time = extracted
                    break

        return {
            "incremental": incremental,
            "messages": extraction_messages,
            "session_time": session_time,
            "tool_stats_text": format_tool_stats_text(buf.usage_stats),
        }

    def _consume_prepared_compaction_state(self, params: dict, prepare_token: str) -> dict:
        result = self._build_compaction_state(params)
        if result.get("ok") is False:
            return result
        ttl_seconds = _positive_int(getattr(self._cfg, "compact_prepare_token_ttl", 300), 300)
        if lifecycle_kind(result["buffer"]) == "compaction":
            journal = self._compaction_journal(result)
            prepared = journal.prepare_state(ttl_seconds)
            if prepared is None or prepared["prepareToken"] != prepare_token:
                raise ValueError("invalid prepare token")
            self._update_lifecycle_error(journal)
            result.update(prepared)
            result["_lifecycle_journal"] = journal
            return result
        archive_id = result["mgr"].get_compaction_prepare_archive_id(result["session_id"])
        if not result["mgr"].consume_compaction_prepare_token(
            result["session_id"],
            prepare_token,
            ttl_seconds=ttl_seconds,
        ):
            raise ValueError("invalid prepare token")
        result["prepareToken"] = prepare_token
        result["archive_id"] = archive_id or None
        return result

    @staticmethod
    def _write_result_has_failures(write_result: dict | None) -> bool:
        if not isinstance(write_result, dict):
            return True
        if write_result.get("ok") is False or write_result.get("status") == "failed":
            return True
        try:
            return int(write_result.get("writes_failed", 0) or 0) > 0
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _plan_field(plan, name: str, default=None):
        if isinstance(plan, dict):
            return plan.get(name, default)
        return getattr(plan, name, default)

    @classmethod
    def _write_result_updates_profile(cls, write_result: dict | None) -> bool:
        if not isinstance(write_result, dict):
            return False

        for plan in write_result.get("plans") or []:
            if cls._plan_field(plan, "action") == "skip":
                continue
            category = str(cls._plan_field(plan, "category", "") or "").lower()
            if category == "profile":
                return True

            uri = str(
                cls._plan_field(plan, "target_uri", None)
                or cls._plan_field(plan, "uri", "")
                or ""
            ).rstrip("/")
            if uri.endswith("/memories/profile") or "/memories/profile/" in uri:
                return True

        return False

    def _invalidate_identity_cache_if_profile_written(
        self,
        session_id: str,
        write_result: dict | None,
    ) -> None:
        if self._write_result_updates_profile(write_result):
            self.get_session_manager().invalidate_topic_slot(session_id, "identity")

    @capture_outbox_events
    @_guard_session_mutation
    def prepare_compaction(self, params: dict) -> dict:
        """Prepare extraction while preventing lifecycle cleanup from interleaving."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            self._load_extraction_recovery_before_accumulation(session_id, ctx)
        except SessionOperationInProgress:
            raise
        except Exception as exc:
            return {"ok": False, "status": "needs_confirmation", "retryable": False,
                    "error": "durable extraction state unavailable: " + str(exc)}
        buf = self.get_session_manager().begin_managed_operation(session_id, ctx)
        try:
            with request_usage_scope(ctx), buf.lifecycle_lock:
                with buf.lifecycle_source_scope() as conflict:
                    if conflict is not None:
                        return {"ok": False, "status": "processing", "retryable": True,
                                "session_id": session_id, "reason": conflict}
                    buf.begin_extraction()
                    try:
                        return self._prepare_compaction_impl(params)
                    finally:
                        buf.end_extraction()
        except UsagePersistenceError as exc:
            return self._lifecycle_usage_failure(exc, params, buf, ctx)
        except (RecoveryNeedsConfirmationError, WriteOutcomeUnknownError) as exc:
            return {"ok": False, "status": "needs_confirmation", "retryable": False, "error": str(exc)}
        except RecoveryPersistenceError as exc:
            return {"ok": False, "status": "failed", "retryable": True, "error": str(exc)}
        finally:
            buf.end_operation()

    def _prepare_compaction_impl(self, params: dict) -> dict:
        """Prepare incremental extraction state before compact commit."""
        result = self._build_compaction_state(params)
        if result.get("ok") is False:
            return result
        result["mgr"].load_session_state(result["session_id"], result["ctx"])
        buf = result["buffer"]
        incremental = result["incremental"]
        ttl_seconds = _positive_int(getattr(self._cfg, "compact_prepare_token_ttl", 300), 300)

        from service.api import MemoryWriteAPI as _CheckpointWriteAPI
        write_api = self.get_write_api() if result["messages"] else self._write_api
        if (lifecycle_kind(buf) == "compaction"
                or isinstance(write_api, _CheckpointWriteAPI) and result["source_snapshot"]):
            journal = self._compaction_journal(result)
            result["archive_id"] = journal.batch["archive_id"]
            result["_lifecycle_journal"] = journal
            prepared = journal.prepare_state(ttl_seconds)
            self._update_lifecycle_error(journal)
            if prepared is not None:
                result.update(prepared)
                return result
            if journal.batch["messages"]:
                write_result = self._chunk_and_extract(
                    messages=journal.batch["messages"], write_api=write_api, ctx=result["ctx"],
                    session_time=result["session_time"], session_summary=journal.batch["session_summary"],
                    tool_stats_text=journal.batch["tool_stats_text"], extraction_run_id="prepare_compaction",
                    session_id=result["session_id"], archive_id=journal.batch["archive_id"],
                    extraction_journal=journal)
                if self._write_result_has_failures(write_result):
                    raise RuntimeError("compaction extraction write failed")
                self._invalidate_identity_cache_if_profile_written(result["session_id"], write_result)
                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(buf.extraction_summary, write_result)
                result["write_result"] = write_result
                journal.mutate(lambda batch: batch["lifecycle"].update(
                    semantic_status="DEGRADED" if write_result.get("degraded") else "COMPLETE"))
            result["prepareToken"] = journal.issue_prepare_token()
            source_ids = set(journal.batch["message_ids"])
            buf.extraction_watermark = max((index + 1 for index, message in enumerate(buf.messages)
                                           if message.id in source_ids), default=buf.extraction_watermark)
            return result

        existing_prepare = result["mgr"].get_compaction_prepare_state(
            result["session_id"],
            ttl_seconds=ttl_seconds,
        )
        if existing_prepare is not None:
            result["prepareToken"] = existing_prepare["prepareToken"]
            result["archive_id"] = existing_prepare.get("archive_id") or None
            return result

        from session.session_manager import generate_archive_id
        archive_id = result["mgr"].get_compaction_prepare_archive_id(result["session_id"]) or generate_archive_id()
        result["archive_id"] = archive_id

        if not incremental:
            result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
                result["session_id"],
                archive_id=archive_id,
            )
            return result

        if not result["messages"]:
            buf.extraction_watermark = len(buf.messages)
            result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
                result["session_id"],
                archive_id=archive_id,
            )
            return result

        write_api = self.get_write_api()
        if write_api is None:
            raise RuntimeError("write API unavailable for compaction")

        # Apply chunking if enabled
        write_result = self._chunk_and_extract(
            messages=result["messages"],
            write_api=write_api,
            ctx=result["ctx"],
            session_time=result["session_time"],
            session_summary=buf.extraction_summary,
            tool_stats_text=result["tool_stats_text"],
            extraction_run_id="prepare_compaction",
            session_id=result["session_id"],
            archive_id=archive_id,
        )
        if self._write_result_has_failures(write_result):
            raise RuntimeError("compaction extraction write failed")
        self._invalidate_identity_cache_if_profile_written(result["session_id"], write_result)
        buf.extraction_watermark = len(buf.messages)
        if write_result.get("candidates_extracted", 0) > 0:
            buf.extraction_summary = _update_extraction_summary(
                buf.extraction_summary,
                write_result,
            )
        result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
            result["session_id"],
            archive_id=archive_id,
        )
        result["write_result"] = write_result
        return result

    def _compaction_journal(self, result):
        from session.session_manager import generate_archive_id
        buf = result["buffer"]
        pending = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch")
        snapshot = result["source_snapshot"]
        return LifecycleBatchJournal(result["mgr"], buf, result["ctx"], snapshot,
            kind="compaction", archive_id=(pending["archive_id"] if pending
                else buf.compaction_prepare_archive_id or generate_archive_id()),
            messages=result["messages"],
            session_time=result["session_time"], session_summary=buf.extraction_summary,
            tool_stats_text=result["tool_stats_text"])

    def bootstrap(self, params: dict) -> dict:
        return {"bootstrapped": True}

    def prepare_subagent_spawn(self, params: dict) -> dict:
        return {"prepared": True}

    def on_subagent_ended(self, params: dict) -> dict:
        return {"cleaned": True}

    @staticmethod
    def _lifecycle_extraction_outcome(lifecycle):
        if lifecycle.get("source_only_reason"):
            return {"semantic_status": "DEGRADED", "recovery_mode": "source_only",
                    "error_detail": lifecycle["extraction_error_detail"]}
        return {"semantic_status": lifecycle.get("semantic_status", "COMPLETE")}

    @staticmethod
    def _update_lifecycle_error(journal, exc=None, *, retryable=True):
        if exc is None:
            if journal.batch["lifecycle"].get("last_error"):
                journal.mutate(lambda batch: batch["lifecycle"].pop("last_error", None))
            return
        try:
            needs_confirmation = isinstance(exc, (RecoveryNeedsConfirmationError, WriteOutcomeUnknownError))
            failure = {"status": "needs_confirmation" if needs_confirmation else "failed",
                       "retryable": retryable and not needs_confirmation,
                       "error": str(exc),
                       **error_fields(exc, journal.batch.get("client_request_id", ""))}
            journal.mutate(lambda batch: batch["lifecycle"].update(last_error=failure))
        except Exception:
            logger.exception("lifecycle failure checkpoint unavailable: session=%s", journal.buffer.session_id[:8])

    def _save_dispose_snapshot(self, mgr, params, buf, snapshot_ids):
        with buf.session_state_save_lock, mgr._lock, buf.extraction_lock:
            snapshot = [message for message in buf.messages if message.id in snapshot_ids]
        if not snapshot:
            return
        ctx = params.get("_ctx") or self.build_context(params)
        journal = self._dispose_journal(mgr, buf, ctx, snapshot)
        try:
            self._update_lifecycle_error(journal)
            archived = self.get_write_api().write_raw_chunk(
                messages=[{"role": message.role, "content": message.content, "id": message.id} for message in snapshot],
                ctx=ctx, session_id=journal.batch["archive_id"], chunk_index=0,
                session_time=_parse_session_time(journal.batch.get("session_time")),
                candidate_checkpoint=journal.checkpoint("raw:source"))
            if not archived.get("target_uri"):
                raise RecoveryPersistenceError("dispose source archive requires confirmation")
            journal.finish(outcome={"semantic_status": "DEGRADED", "request_status": "DEGRADED",
                "source_status": "RAW_ARCHIVED", "raw_source_uris": [archived["target_uri"]],
                "error_detail": {"code": "SEMANTIC_EXTRACTION_SKIPPED", "component": "extraction",
                    "request_id": "", "retry_action": "manual_review",
                    "cause": {"message": "dispose extraction threshold or extractable source requirement was not met"}}},
                resolve_outcome=self._resolve_write_outcome)
        except Exception as exc:
            self._update_lifecycle_error(journal, exc)
            raise
        if (_HAS_AGFS and not self._cfg.role_control_enabled
                and ctx.account_id != self._default_account_id):
            recovery = AGFSContextFS.active_session_recovery()
            if recovery is not None:
                recovery[0].post_commit_drain = True

    @capture_outbox_events
    def _run_dispose_background(self, mgr, session_id: str, params: dict, ctx, buf, source_snapshot=None, operation=None) -> None:
        try:
            from service.api import MemoryWriteAPI
            fs = self._write_api._fs if isinstance(self._write_api, MemoryWriteAPI) else None
            if (not self._use_sql and not callable(getattr(type(fs), "session_mutation_lock", None))
                    and not callable(getattr(type(fs), "session_recovery_lock", None))):
                with buf.lifecycle_lock:
                    _run_with_usage_scope(ctx, self._run_dispose_background_impl,
                                          mgr, session_id, params, ctx, buf, source_snapshot)
                return
            try:
                effective_ctx = ctx or self.build_context(params)
                while True:
                    # The ordinary extraction worker needs its own SQL connection.
                    # Never wait for its event while holding the session advisory lock.
                    with buf.extraction_lock:
                        event = buf.extraction_done_event if buf.extraction_in_progress else None
                    if event is not None and not event.wait(timeout=600):
                        return
                    with self._session_mutation_scope(effective_ctx) as guard:
                        # An ordinary worker may have started before acquisition.
                        # Release ownership and wait outside it before taking source.
                        with buf.extraction_lock:
                            extraction_active = buf.extraction_in_progress
                        if extraction_active:
                            continue
                        with buf.lifecycle_lock:
                            _run_with_usage_scope(effective_ctx, self._run_dispose_background_impl,
                                                  mgr, session_id, params, effective_ctx, buf, source_snapshot)
                        if guard is not None:
                            guard.ensure_owned()
                    if guard is not None and guard.post_commit_drain:
                        self._async_drain(account_id=effective_ctx.account_id)
                    break
            except Exception as exc:
                logger.error("fenced dispose stopped session=%s: %s", session_id[:8], exc)
        finally:
            if operation is not None:
                operation.release()

    def _dispose_journal(self, mgr, buf, ctx, snapshot):
        from session.session_manager import generate_archive_id
        state = self._build_incremental_extraction_state(buf, incremental=snapshot)
        pending = buf.runtime_extensions.get("memory_extraction_recovery", {}).get("batch")
        return LifecycleBatchJournal(mgr, buf, ctx, snapshot, kind="dispose",
            archive_id=pending["archive_id"] if pending else generate_archive_id(), messages=state["messages"],
            session_time=state["session_time"], session_summary=buf.extraction_summary,
            tool_stats_text=state["tool_stats_text"])

    def _prepare_dispose_source(self, mgr, buf, ctx, params):
        from service.api import MemoryWriteAPI as _CheckpointWriteAPI

        with buf.lifecycle_lock:
            with buf.extraction_lock:
                if buf.extraction_in_progress:
                    return {"source_snapshot": None}
                pending_ids = buf.pending_extraction_ids()
                snapshot = buf.capture_lifecycle_source(
                    [message for message in buf.messages if message.id in pending_ids]
                    if pending_ids else buf.messages[buf.extraction_watermark:])
                durable_flush = (snapshot and sum(len(message.content) // 4 for message in snapshot) >= 200
                                 and any(message.role != "assistant" for message in snapshot))
            if durable_flush and (isinstance(self._write_api, _CheckpointWriteAPI)
                    or getattr(self.get_write_api, "__func__", None) is MemoryService.get_write_api):
                write_api = self.get_write_api()
                if isinstance(write_api, _CheckpointWriteAPI):
                    ctx = ctx or self.build_context(params)
                    self._load_extraction_recovery_before_accumulation(buf.session_id, ctx)
                    journal = self._dispose_journal(mgr, buf, ctx, snapshot)
                    return {"source_snapshot": snapshot, "request_id": journal.batch["client_request_id"]}
            return {"source_snapshot": snapshot}

    @capture_outbox_events
    def _run_dispose_background_impl(self, mgr, session_id: str, params: dict, ctx, buf, source_snapshot=None) -> None:
        snapshot_ids: set[str] = set()
        extraction_started = False
        succeeded = False
        effective_ctx = ctx
        journal = None
        source_stack = ExitStack()
        try:
            with buf.extraction_lock:
                event = buf.extraction_done_event if buf.extraction_in_progress else None
            if event is not None:
                from fs.sql_adapter.session_mutation import active_session_guard
                if active_session_guard() is not None:
                    logger.warning("dispose source became active; preserving pending session=%s", session_id[:8])
                    return
                event.wait(timeout=600)
                with buf.extraction_lock:
                    if buf.extraction_in_progress:
                        logger.warning(
                            "dispose: extraction still active after timeout; keeping session=%s",
                            session_id[:8],
                        )
                        return

            with buf.extraction_lock:
                if buf.pending_extraction_ids() and lifecycle_kind(buf) != "dispose":
                    return
            conflict = source_stack.enter_context(buf.lifecycle_source_scope())
            if conflict is not None:
                logger.warning("dispose source preparation busy session=%s: %s", session_id[:8], conflict)
                return

            with buf.extraction_lock:
                original_watermark = buf.extraction_watermark
                pending_ids = buf.pending_extraction_ids()
                snapshot = buf.capture_lifecycle_source(
                    [SessionMessage(**value) for value in buf.runtime_extensions[EXTENSION_KEY]["batch"]["snapshot"]]
                    if pending_ids else source_snapshot if source_snapshot is not None
                    else buf.messages[original_watermark:])
                snapshot_ids = {message.id for message in snapshot}
                pending_tokens = sum(len(message.content) // 4 for message in snapshot)
                extraction_messages = [
                    {"role": message.role, "content": message.content}
                    for message in snapshot
                    if message.role != "assistant"
                ]
                if snapshot:
                    buf.extraction_watermark = max(index + 1 for index, message in enumerate(buf.messages)
                                                   if message.id in snapshot_ids)
                if pending_tokens >= 200 and extraction_messages:
                    buf.begin_extraction()
                    extraction_started = True
                    extraction_summary = buf.extraction_summary

            if pending_tokens >= 200 and extraction_messages:
                write_api = self.get_write_api()
                if write_api is None:
                    raise RuntimeError("dispose write API unavailable")
                effective_ctx = ctx or self.build_context(params)
                from service.api import MemoryWriteAPI as _CheckpointWriteAPI
                if isinstance(write_api, _CheckpointWriteAPI):
                    journal = self._dispose_journal(mgr, buf, effective_ctx, snapshot)
                    extraction_messages = journal.batch["messages"]
                    self._update_lifecycle_error(journal)
                tool_stats_text = ""
                try:
                    from extraction.tool_collector import format_tool_stats_text

                    tool_stats_text = format_tool_stats_text(buf.usage_stats)
                except Exception:
                    pass
                run_id = f"dispose-{uuid4().hex[:12]}"
                write_result = self._chunk_and_extract(
                    messages=extraction_messages,
                    write_api=write_api,
                    ctx=effective_ctx,
                    session_time=None,
                    session_summary=extraction_summary,
                    tool_stats_text=tool_stats_text,
                    extraction_run_id=run_id,
                    session_id=session_id,
                    archive_id=journal.batch["archive_id"] if journal is not None else None,
                    extraction_journal=journal,
                )
                if self._write_result_has_failures(write_result):
                    raise RuntimeError("dispose extraction write failed")
                if journal is not None:
                    assistant_source = [{"role": message.role, "content": message.content, "id": message.id}
                                        for message in snapshot if message.role == "assistant"]
                    if assistant_source:
                        write_api.write_raw_chunk(messages=assistant_source, ctx=effective_ctx,
                            session_id=journal.batch["archive_id"], chunk_index=0,
                            session_time=_parse_session_time(journal.batch.get("session_time")),
                            candidate_checkpoint=journal.checkpoint("raw:assistant"))
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                self.drain_outbox_sync(account_id=effective_ctx.account_id)
                if journal is not None:
                    raw_uris = [operation["plan"]["target_uri"]
                        for slot, record in journal.batch["slots"].items() if slot.startswith("raw:")
                        for operation in record["operations"].values() if operation.get("status") == "complete"]
                    outcome = self._lifecycle_extraction_outcome(journal.batch["lifecycle"])
                    journal.finish(outcome={**outcome, "source_status": "RAW_ARCHIVED", "raw_source_uris": raw_uris},
                                   resolve_outcome=self._resolve_write_outcome)
                else:
                    self._save_dispose_snapshot(mgr, params, buf, snapshot_ids)
                succeeded = True
                logger.info(
                    "dispose background flush done: run=%s session=%s extracted=%d writes=%d",
                    run_id,
                    session_id[:8],
                    write_result.get("candidates_extracted", 0),
                    write_result.get("writes_completed", 0),
                )
            else:
                self._save_dispose_snapshot(mgr, params, buf, snapshot_ids)
                succeeded = True
        except Exception as exc:
            if snapshot_ids:
                buf.rewind_watermark_to_ids(snapshot_ids)
            if journal is not None:
                self._update_lifecycle_error(journal, exc)
            logger.exception("dispose background flush failed: session=%s", session_id[:8])
        finally:
            source_stack.close()
            if extraction_started:
                buf.end_extraction()
            if succeeded:
                removed_session = mgr.remove_session_if_empty(
                    session_id,
                    buf,
                    ctx=effective_ctx,
                )
                if removed_session:
                    self._flush_session_tool_usage_async(session_id, effective_ctx)

    @_attribute_request_usage
    @capture_outbox_events
    @_guard_session_mutation
    def dispose(self, params: dict | None = None) -> dict:
        """Session end: safely dispatch a non-blocking pending-memory flush."""
        params = params or {}
        self.wake_skill_sedimentation()
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"disposed": True, "flushed": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        ctx = params.get("_ctx")
        if ctx is not None and mgr.has_session(session_id) and not mgr.has_session(session_id, ctx=ctx):
            return {"disposed": True, "flushed": False, "reason": "session_not_found"}
        from service.api import MemoryWriteAPI as _CheckpointWriteAPI
        if (isinstance(self._write_api, _CheckpointWriteAPI) or callable(mgr._get_context_fs)
                or not mgr.has_session(session_id)):
            ctx = ctx or self.build_context(params)
            try:
                self._load_extraction_recovery_before_accumulation(session_id, ctx)
            except SessionOperationInProgress:
                raise
            except Exception as exc:
                return {"disposed": False, "flushed": False, "status": "needs_confirmation",
                        "retryable": False, "error": "durable extraction state unavailable: " + str(exc)}
        try:
            buf = mgr.begin_managed_operation(
                session_id,
                ctx,
                create=False,
            )
        except SessionOperationInProgress as exc:
            if exc.retryable:
                raise
            return {
                "disposed": False,
                "flushed": False,
                "status": 409,
                "error_code": "session_operation_in_progress",
                "conflict_type": exc.conflict_type,
                "retryable": exc.retryable,
            }
        if buf is None:
            return {"disposed": True, "flushed": False, "reason": "session_not_found"}
        operation = _DisposeOperationLease(buf)
        source_stack = ExitStack()
        try:
            with buf.extraction_lock:
                if buf.ordinary_turns_in_progress:
                    return {"disposed": False, "flushed": False, "status": "processing",
                            "retryable": True, "reason": "ordinary_turn_in_progress"}
                pending_kind = lifecycle_kind(buf)
                ordinary_worker_active = pending_kind is None and buf.extraction_in_progress
                if buf.pending_extraction_ids() and pending_kind != "dispose" and not ordinary_worker_active:
                    return {"disposed": False, "flushed": False, "status": "needs_confirmation",
                            "retryable": False, "reason": "pending_memory_batch"}
                if buf.dispose_in_progress:
                    return {
                        "disposed": True,
                        "flushed": False,
                        "status": "background_flush_dispatched",
                        "reason": "dispose_in_progress",
                    }
                operation.claim_dispose()
            source_stack.enter_context(buf.lifecycle_lock)
            conflict = source_stack.enter_context(buf.lifecycle_source_scope(allow_active_extraction=True))
            if conflict is not None:
                return {"disposed": False, "flushed": False, "status": "processing",
                        "retryable": True, "reason": conflict}
            return self._dispose_source_impl(mgr, buf, ctx, params, source_stack, operation)
        finally:
            try:
                source_stack.close()
            finally:
                operation.release(local=True)

    def _dispose_source_impl(self, mgr, buf, ctx, params, source_stack, operation):
        session_id = buf.session_id
        try:
            source_state = self._prepare_dispose_source(mgr, buf, ctx, params)
            if source_state.get("ok") is False:
                return {"disposed": False, "flushed": False, **source_state}
            source_snapshot = source_state["source_snapshot"]
        except Exception as exc:
            if isinstance(exc, RecoveryPersistenceError):
                return {"disposed": False, "flushed": False, "status": "failed",
                        "retryable": True, "error": str(exc)}
            if isinstance(exc, RecoveryNeedsConfirmationError):
                return {"disposed": False, "flushed": False, "status": "needs_confirmation",
                        "retryable": False, "error": str(exc)}
            raise

        fast_path_reason = None
        snapshot_ids = set()
        with buf.extraction_lock:
            extraction_active = buf.extraction_in_progress
            reason = "extraction_in_progress" if extraction_active else "background_flush_dispatched"
            pending_tokens = None
            if not extraction_active and source_snapshot is not None:
                unextracted = source_snapshot
                pending_tokens = sum(len(message.content) // 4 for message in unextracted)
                if not unextracted:
                    reason = "no_pending_messages"
                elif pending_tokens < 200:
                    reason = "below_token_threshold"
                elif not any(message.role != "assistant" for message in unextracted):
                    reason = "no_user_messages"

                if reason in {
                    "no_pending_messages",
                    "below_token_threshold",
                    "no_user_messages",
                }:
                    fast_path_reason = reason
                    if unextracted:
                        snapshot_ids = {message.id for message in unextracted}

        if fast_path_reason is not None:
            self._save_dispose_snapshot(mgr, params, buf, snapshot_ids)
            removed_session = mgr.remove_session_if_empty(
                session_id,
                buf,
                ctx=ctx,
            )
            if removed_session:
                self._flush_session_tool_usage_async(session_id, ctx)

            result = {
                "disposed": True,
                "flushed": False,
                "reason": fast_path_reason,
            }
            if pending_tokens is not None:
                result["pending_tokens"] = pending_tokens
            return result

        # The fixed durable batch now owns eligible real-writer source. The
        # worker acquires its own preparation scope, including inline callers.
        source_stack.close()
        try:
            operation.worker_owned = True
            threading.Thread(
                target=self._run_dispose_background,
                args=(mgr, session_id, params, ctx, buf, source_snapshot, operation),
                daemon=True,
                name=f"dispose-{session_id[:8]}",
            ).start()
        except Exception as exc:
            operation.release()
            logger.error(
                "dispose background thread unavailable: session=%s: %s",
                session_id[:8],
                exc,
                exc_info=True,
            )
            return {"disposed": True, "flushed": False, "reason": "thread_unavailable"}

        result = {
            "disposed": True,
            "flushed": False,
            "status": "dispose_deferred" if extraction_active else "background_flush_dispatched",
            "reason": reason,
        }
        if source_state.get("request_id"):
            result["request_id"] = source_state["request_id"]
        if pending_tokens is not None:
            result["pending_tokens"] = pending_tokens
        return result

    @_attribute_request_usage
    def wait_until_idle(self, params: dict | None = None) -> dict:
        import time

        params = params or {}
        started = time.monotonic()
        result = self._wait_until_session_idle(params)
        if (not result.get("idle") or not self._cfg.scene_enabled or not self._cfg.scene_generation_enabled
                or not self._param_bool(params, "waitScenes", "wait_scenes", default=False)):
            return result
        self._ensure_scene_runtime()
        ctx = params.get("_ctx") or self.build_context(params)
        owners = (ctx.user_space_name(), ctx.agent_space_name())
        for owner in owners:
            self._scene_manager.flush(ctx.account_id, owner)
        deadline = started + self._idle_timeout_seconds(params)
        while True:
            states = {owner: self._scene_manager.jobs.summary(ctx.account_id, owner) for owner in owners}
            busy_owners = {owner for owner, state in states.items()
                           if state["pending_nodes"] or (state.get("active_job") or {}).get("status")
                           in {"PENDING", "RUNNING"}}
            failed = self._scene_manager.halted or any(
                owner not in busy_owners and (state.get("last_job") or {}).get("status") == "FAILED"
                for owner, state in states.items()
            )
            busy = bool(busy_owners)
            if failed or not busy or time.monotonic() >= deadline:
                return {**result, "idle": not failed and not busy,
                        "reason": "scene_failed" if failed else "scene_in_progress_timeout" if busy else result["reason"],
                        "scenes": states}
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    def _wait_until_session_idle(self, params: dict | None = None) -> dict:
        """Wait for process-local dispose/extraction/indexing to settle."""
        params = params or {}
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"idle": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        ctx = params.get("_ctx") or self.build_context(params)
        drain_outbox = self._param_bool(params, "drainOutbox", "drain_outbox", default=True)
        wait_outbox = self._param_bool(params, "waitOutbox", "wait_outbox", default=not drain_outbox)
        had_local_session = mgr.has_session(session_id, ctx=ctx)
        try:
            # Preserve the public not-found contract for a known foreign local
            # buffer. An absent local buffer still requires the strict read.
            if had_local_session or not mgr.has_session(session_id):
                local_busy = had_local_session and mgr.get_or_create(
                    session_id, ctx=ctx).active_operation_count > 0
                scope = nullcontext() if local_busy else self._session_mutation_scope(ctx, refresh=False)
                with scope as guard:
                    local_active = False
                    if had_local_session:
                        local = mgr.get_or_create(session_id, ctx=ctx)
                        with local.extraction_lock:
                            local_active = bool(local.active_operation_count or local.dispose_in_progress
                                                or local.extraction_in_progress or local.commit_in_progress)
                    if guard is None or not local_active:
                        self._load_extraction_recovery_before_accumulation(
                            session_id, ctx, force_reload=guard is not None)
        except SessionOperationInProgress:
            raise
        except Exception as exc:
            if isinstance(exc, SessionMutationBusyError):
                return {"idle": False, "reason": "session_mutation_in_progress", "retryable": True}
            return {"idle": False, "reason": "recovery_state_unavailable",
                    "status": "needs_confirmation", "retryable": False,
                    "error": "durable extraction state unavailable: " + str(exc)}
        if not had_local_session and mgr.has_session(session_id, ctx=ctx):
            # Strict observation can materialize an empty finished state. Keep
            # idle from resurrecting a disposed buffer, without removing any
            # source or operation that arrived during that read.
            mgr.remove_session_if_empty(session_id, mgr.get_or_create(session_id, ctx=ctx),
                                        ctx=ctx, require_quiescent=True)
        if not mgr.has_session(session_id, ctx=ctx):
            if drain_outbox:
                drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
                return {
                    "idle": True,
                    "reason": "session_not_found",
                    "extraction_in_progress": False,
                    "drain": drain_stats,
                }
            outbox_wait = (
                self._wait_for_outbox_idle(ctx, self._idle_timeout_seconds(params))
                if wait_outbox
                else {"idle": True, "outbox": {"skipped": True}}
            )
            if not outbox_wait.get("idle", True):
                return {
                    "idle": False,
                    "reason": "outbox_in_progress_timeout",
                    "extraction_in_progress": False,
                    "outbox": outbox_wait.get("outbox", {}),
                }
            return {
                "idle": True,
                "reason": "session_not_found",
                "extraction_in_progress": False,
                "outbox": outbox_wait.get("outbox", {}),
            }

        buf = mgr.get_or_create(session_id, ctx=ctx)
        timeout_seconds = self._idle_timeout_seconds(params)
        import time as _time

        deadline = _time.monotonic() + timeout_seconds
        while True:
            with buf.extraction_lock:
                extraction_in_progress = bool(buf.extraction_in_progress)
                dispose_in_progress = bool(buf.dispose_in_progress)
                extraction_event = buf.extraction_done_event
                active_extractions = getattr(buf, "extraction_active_count", 0)

            if not extraction_in_progress and not dispose_in_progress:
                break

            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return {
                    "idle": False,
                    "reason": (
                        "dispose_in_progress_timeout"
                        if dispose_in_progress
                        else "extraction_in_progress_timeout"
                    ),
                    "extraction_in_progress": extraction_in_progress,
                    "dispose_in_progress": dispose_in_progress,
                    "active_extractions": active_extractions,
                }

            # Extraction has an event; dispose itself does not, so poll briefly
            # while sharing the same overall timeout budget for both states.
            if (
                extraction_in_progress
                and extraction_event is not None
                and not extraction_event.is_set()
            ):
                extraction_event.wait(timeout=remaining)
            else:
                _time.sleep(min(0.01, remaining))

        with buf.extraction_lock:
            extension = deepcopy(buf.runtime_extensions.get(EXTENSION_KEY))
        if extension is not None:
            batch = extension.get("batch") if isinstance(extension, dict) else None
            if (not isinstance(extension, dict) or extension.get("version") != 1
                    or not isinstance(batch, dict) or batch.get("owner") != recovery_owner(ctx)):
                return {"idle": False, "reason": "extraction_recovery_pending",
                        "status": "needs_confirmation", "retryable": False,
                        "error": "pending extraction batch has a different owner or format"}
            lifecycle = batch.get("lifecycle")
            if "lifecycle" in batch and (not isinstance(lifecycle, dict)
                    or lifecycle.get("kind") not in {"compaction", "dispose"}):
                return {"idle": False, "reason": "lifecycle_recovery_pending",
                        "status": "needs_confirmation", "retryable": False,
                        "error": "pending lifecycle checkpoint is invalid"}
            failure = (lifecycle or {}).get("last_error", {})
            task = batch.get("retry_task")
            if task is not None and not lifecycle:
                self._extraction_retry.wake(ctx, task)
                return {"idle": False, "reason": "extraction_retry_task",
                        "status": task["status"], "recovery_task": deepcopy(task),
                        "retryable": task["status"] in {"queued", "running", "failed"},
                        "extraction_in_progress": False, "dispose_in_progress": False}
            return {"idle": False, "reason": "lifecycle_recovery_pending" if lifecycle
                    else "extraction_recovery_pending",
                    "status": failure.get("status", "pending"),
                    "retryable": failure.get("retryable", True),
                    "extraction_in_progress": False, "dispose_in_progress": False}

        if drain_outbox:
            drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
            return {
                "idle": True,
                "reason": "idle",
                "extraction_in_progress": False,
                "drain": drain_stats,
            }

        outbox_wait = (
            self._wait_for_outbox_idle(ctx, timeout_seconds)
            if wait_outbox
            else {"idle": True, "outbox": {"skipped": True}}
        )
        if not outbox_wait.get("idle", True):
            return {
                "idle": False,
                "reason": "outbox_in_progress_timeout",
                "extraction_in_progress": False,
                "outbox": outbox_wait.get("outbox", {}),
            }
        return {
            "idle": True,
            "reason": "idle",
            "extraction_in_progress": False,
            "outbox": outbox_wait.get("outbox", {}),
        }

    @staticmethod
    def _param_bool(params: dict, camel_name: str, snake_name: str, *, default: bool) -> bool:
        value = params.get(camel_name, params.get(snake_name, default))
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _idle_timeout_seconds(params: dict) -> float:
        raw_timeout = params.get("timeoutSeconds")
        if raw_timeout is None:
            raw_timeout = params.get("timeout_seconds")
        if raw_timeout is None:
            raw_timeout = 30
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_seconds = 30.0
        return max(0.0, timeout_seconds)

    def _wait_for_outbox_idle(self, ctx: RequestContext, timeout_seconds: float) -> dict:
        """Wait for active indexing work; paused graph jobs remain in task/status APIs."""
        import time as _time

        deadline = _time.monotonic() + timeout_seconds
        last_counts = self._count_outbox_unfinished(ctx, exclude_paused_graph=True)
        while True:
            # Idle does not mean successfully indexed. In particular, dead-lettered
            # events and an unavailable status store must never certify completion.
            if (last_counts.get("supported") is False or last_counts.get("error")
                    or int(last_counts.get("failed") or 0) > 0):
                return {"idle": False, "outbox": last_counts}
            total = int(last_counts.get("total") or 0)
            if total <= 0:
                return {"idle": True, "outbox": last_counts}
            if _time.monotonic() >= deadline:
                return {"idle": False, "outbox": last_counts}
            _time.sleep(min(1.0, max(0.0, deadline - _time.monotonic())))
            last_counts = self._count_outbox_unfinished(ctx, exclude_paused_graph=True)

    def _count_outbox_unfinished(self, ctx: RequestContext, *, exclude_paused_graph: bool = False) -> dict:
        try:
            fs = self._get_context_fs()
            outbox_store = self._get_outbox_store(fs) if fs is not None else None
            if outbox_store is not None and hasattr(outbox_store, "count_unfinished"):
                if (exclude_paused_graph and _HAS_SQL and isinstance(outbox_store, SQLOutboxStore)
                        and not self._graph_enabled()):
                    return outbox_store.count_unfinished(
                        account_id=ctx.account_id, exclude_event_types=["PROJECT_GRAPH"])
                return outbox_store.count_unfinished(account_id=ctx.account_id)
            return {"pending": 0, "processing": 0, "total": 0, "supported": False}
        except Exception as exc:
            logger.warning("outbox idle status unavailable: %s", exc)
            return {"pending": 0, "processing": 0, "total": 0, "supported": False, "error": str(exc)}

    def _save_session_state_best_effort(
        self,
        mgr: SessionManager,
        session_id: str,
        params: dict,
    ) -> None:
        try:
            ctx = params.get("_ctx") or self.build_context(params)
            saved = mgr.save_session_state(session_id, ctx)
            from fs.sql_adapter.session_mutation import active_session_guard
            if active_session_guard() is not None and not saved:
                raise RecoveryNeedsConfirmationError("dispose checkpoint was not acknowledged")
        except Exception as exc:
            from fs.sql_adapter.session_mutation import active_session_guard
            if active_session_guard() is not None:
                raise
            logger.warning("dispose session state save skipped session=%s: %s", session_id, exc)

    def get_cumulative_token_usage(self, reset: bool = False) -> dict:
        result: dict = {}
        internal_tool_counts = self._internal_tool_usage.count_snapshot()
        llm = self._llm
        if llm and hasattr(llm, "token_tracker"):
            tracker = llm.token_tracker
            snap = tracker.snapshot_and_reset() if reset else tracker.snapshot()
            result["llm"] = {
                "input_tokens": snap.input_tokens,
                "output_tokens": snap.output_tokens,
                "cache_read": snap.cache_read,
                "cache_write": snap.cache_write,
                "total_tokens": snap.input_tokens + snap.output_tokens,
                "calls": snap.llm_calls,
            }
        if self._embedder and hasattr(self._embedder, "token_tracker"):
            tracker = self._embedder.token_tracker
            snap = tracker.snapshot_and_reset() if reset else tracker.snapshot()
            result["embedding"] = {"total_tokens": snap.embed_tokens, "calls": snap.embed_calls}
        if internal_tool_counts["rounds"] or internal_tool_counts["tool_calls"]:
            result["internal_tool_rounds"] = internal_tool_counts
        if reset:
            self._internal_tool_usage.reset()
        if result:
            result["total_tokens"] = (
                result.get("llm", {}).get("total_tokens", 0)
                + result.get("embedding", {}).get("total_tokens", 0)
            )
        return result

    def get_model_usage_metrics(
        self,
        days: int = 14,
        *,
        account_id: str = "",
        owner_spaces: tuple[str, ...] = (),
    ) -> dict:
        store = self._get_model_usage_store()
        if store is not None:
            metrics = store.metrics(days, account_id, owner_spaces)
            if not metrics['models']['llm']:
                metrics['models']['llm'] = self._provider_cfg.openai_llm_model
            if not metrics['models']['embedding']:
                metrics['models']['embedding'] = (
                    self._provider_cfg.st_model
                    if self._provider_cfg.embedding_provider == 'st'
                    else self._provider_cfg.openai_embedding_model
                )
            if not metrics["models"].get("decision"):
                metrics["models"]["decision"] = self._cfg.decision.model if self._cfg.decision else ""
            return metrics
        decision_tracker = getattr(self._decision_client, "token_tracker", None)
        trackers = {
            "decision": decision_tracker,
            "llm": getattr(self._llm, "token_tracker", None) if self._llm else None,
            "embedding": (
                getattr(self._embedder, "token_tracker", None) if self._embedder else None
            ),
        }
        series_by_date: dict[str, dict] = {}
        for source, tracker in trackers.items():
            if tracker is None or not hasattr(tracker, "daily_snapshot"):
                continue
            for row in tracker.daily_snapshot(
                days, account_id=account_id, owner_spaces=owner_spaces
            ):
                item = series_by_date.setdefault(
                    str(row["date"]),
                    {
                        "date": str(row["date"]),
                        "llm_input": 0,
                        "llm_output": 0,
                        "embedding_input": 0,
                        "decision_input": 0,
                        "decision_output": 0,
                        "total": 0,
                    },
                )
                if source == "decision":
                    item["decision_input"] = int(row["input_tokens"])
                    item["decision_output"] = int(row["output_tokens"])
                elif source == "llm":
                    item["llm_input"] = int(row["input_tokens"])
                    item["llm_output"] = int(row["output_tokens"])
                else:
                    item["embedding_input"] = int(row["embed_tokens"])

        series = [series_by_date[key] for key in sorted(series_by_date)]
        for item in series:
            item["total"] = (item["llm_input"] + item["llm_output"] + item["embedding_input"]
                             + item["decision_input"] + item["decision_output"])

        llm_snapshot = (
            trackers["llm"].scoped_snapshot(account_id, owner_spaces)
            if trackers["llm"] and account_id
            else trackers["llm"].snapshot() if trackers["llm"] else None
        )
        embedding_snapshot = (
            trackers["embedding"].scoped_snapshot(account_id, owner_spaces)
            if trackers["embedding"] and account_id
            else trackers["embedding"].snapshot() if trackers["embedding"] else None
        )
        decision_snapshot = (
            decision_tracker.scoped_snapshot(account_id, owner_spaces)
            if decision_tracker and account_id
            else decision_tracker.snapshot() if decision_tracker else None
        )
        decision = {
            "input_tokens": decision_snapshot.input_tokens if decision_snapshot else 0,
            "output_tokens": decision_snapshot.output_tokens if decision_snapshot else 0,
            "total_tokens": (decision_snapshot.input_tokens + decision_snapshot.output_tokens
                             if decision_snapshot else 0),
            "calls": decision_snapshot.llm_calls if decision_snapshot else 0,
            **(decision_tracker.decision_status(account_id, owner_spaces) if decision_tracker else {
                "failed_calls": 0, "missing_usage_calls": 0,
            }),
        }
        llm = {
            "input_tokens": llm_snapshot.input_tokens if llm_snapshot else 0,
            "output_tokens": llm_snapshot.output_tokens if llm_snapshot else 0,
            "cache_read": llm_snapshot.cache_read if llm_snapshot else 0,
            "cache_write": llm_snapshot.cache_write if llm_snapshot else 0,
            "total_tokens": (
                llm_snapshot.input_tokens + llm_snapshot.output_tokens if llm_snapshot else 0
            ),
            "calls": llm_snapshot.llm_calls if llm_snapshot else 0,
        }
        embedding = {
            "total_tokens": embedding_snapshot.embed_tokens if embedding_snapshot else 0,
            "calls": embedding_snapshot.embed_calls if embedding_snapshot else 0,
        }
        today = (
            series[-1]
            if series
            else {
                "date": "",
                "llm_input": 0,
                "llm_output": 0,
                "embedding_input": 0,
                "decision_input": 0,
                "decision_output": 0,
                "total": 0,
            }
        )
        llm_model = getattr(trackers["llm"], "model", None)
        if not llm_model:
            llm_model = (
                "mock"
                if self._provider_cfg.provider == "mock"
                else self._provider_cfg.openai_llm_model
            )
        embedding_provider = self._provider_cfg.embedding_provider or self._provider_cfg.provider
        embedding_model = getattr(trackers["embedding"], "model", None)
        if not embedding_model:
            if embedding_provider == "st":
                embedding_model = self._provider_cfg.st_model
            elif embedding_provider == "mock":
                embedding_model = "mock"
            else:
                embedding_model = self._provider_cfg.openai_embedding_model
        return {
            "enabled": True,
            "scope": {"account_id": account_id, "owner_spaces": list(owner_spaces)},
            "models": {"llm": llm_model, "embedding": embedding_model,
                       "decision": (getattr(decision_tracker, "model", None)
                                    or (self._cfg.decision.model if self._cfg.decision else ""))},
            "today": today,
            "cumulative": {
                "total_tokens": llm["total_tokens"] + embedding["total_tokens"] + decision["total_tokens"],
                "llm": llm,
                "embedding": embedding,
                "decision": decision,
                "missing_usage_calls": decision["missing_usage_calls"],
                "failed_calls": decision["failed_calls"],
            },
            "series": series,
        }

    def record_retrieval_metric(
        self,
        ctx: RequestContext,
        *,
        request_id: str,
        duration_ms: float,
        scores: list[float],
        categories: list[str],
        rerank_used: bool,
        failed: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> None:
        owner_space = ctx.agent_space_name() if ctx.agent_id else ctx.user_space_name()
        key = (ctx.account_id, owner_space)
        with self._retrieval_monitoring_lock:
            metric = self._retrieval_monitoring.setdefault(key, {
                "queries": 0, "failures": 0, "results": 0, "zero_results": 0,
                "score_total": 0.0, "score_count": 0, "score_min": None,
                "score_max": None, "duration_total_ms": 0.0, "duration_max_ms": 0.0,
                "rerank_used": 0, "by_type": {}, "stages": {},
            })
            metric["queries"] += 1
            metric["failures"] += int(failed)
            metric["results"] += len(scores)
            metric["zero_results"] += int(not scores and not failed)
            metric["duration_total_ms"] += max(0.0, duration_ms)
            metric["duration_max_ms"] = max(metric["duration_max_ms"], duration_ms)
            metric["rerank_used"] += int(rerank_used)
            metric["request_id"] = request_id
            metric["updated_at"] = _dt.now(UTC).isoformat()
            if scores:
                metric["score_total"] += sum(scores)
                metric["score_count"] += len(scores)
                metric["score_min"] = min(scores) if metric["score_min"] is None else min(metric["score_min"], *scores)
                metric["score_max"] = max(scores) if metric["score_max"] is None else max(metric["score_max"], *scores)
            for category in categories:
                metric["by_type"][category] = metric["by_type"].get(category, 0) + 1
            for stage in (trace or {}).get("stages", []):
                name = str(stage.get("name") or stage.get("stage") or "unknown")
                elapsed = float(stage.get("duration_ms") or stage.get("elapsed_ms") or 0.0)
                row = metric["stages"].setdefault(name, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
                row["calls"] += 1
                row["total_ms"] += elapsed
                row["max_ms"] = max(row["max_ms"], elapsed)

    def get_retrieval_monitoring_metrics(
        self, account_id: str, owner_spaces: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        visible = set(owner_spaces)
        lock = getattr(self, "_retrieval_monitoring_lock", None)
        if lock is None:
            rows = []
        else:
            with lock:
                rows = [deepcopy(value) for (account, owner), value in self._retrieval_monitoring.items()
                        if account == account_id and (not visible or owner in visible)]
        result = {
            "queries": 0, "failures": 0, "results": 0, "zero_results": 0,
            "score_total": 0.0, "score_count": 0, "score_min": None,
            "score_max": None, "duration_total_ms": 0.0, "duration_max_ms": 0.0,
            "rerank_used": 0, "by_type": {}, "stages": {}, "request_id": "",
        }
        for row in rows:
            for key in ("queries", "failures", "results", "zero_results", "score_total",
                        "score_count", "duration_total_ms", "rerank_used"):
                result[key] += row.get(key, 0)
            result["duration_max_ms"] = max(result["duration_max_ms"], row.get("duration_max_ms", 0.0))
            for key in ("score_min", "score_max"):
                value = row.get(key)
                if value is not None:
                    result[key] = value if result[key] is None else (min if key == "score_min" else max)(result[key], value)
            if row.get("updated_at", "") >= result.get("updated_at", ""):
                result["updated_at"] = row.get("updated_at", "")
                result["request_id"] = row.get("request_id", "")
            for name, count in row.get("by_type", {}).items():
                result["by_type"][name] = result["by_type"].get(name, 0) + count
            for name, stage in row.get("stages", {}).items():
                target = result["stages"].setdefault(name, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
                target["calls"] += stage["calls"]
                target["total_ms"] += stage["total_ms"]
                target["max_ms"] = max(target["max_ms"], stage["max_ms"])
        return result

    def get_monitoring_overview(self, ctx: RequestContext) -> dict:
        health = self.health()
        outbox = self._count_outbox_unfinished(ctx)
        worker_alive = bool(self._outbox_thread and self._outbox_thread.is_alive())
        storage_backend = str(health.get("storage_backend") or self._cfg.storage_backend)
        storage_healthy = (
            health.get("sql") == "connected"
            if storage_backend == "sql"
            else health.get("agfs") is True and bool(health.get("agfs_url"))
        )
        lexical = health.get("lexical") if isinstance(health.get("lexical"), dict) else {}
        lexical_backfill = (
            health.get("lexical_backfill")
            if isinstance(health.get("lexical_backfill"), dict)
            else {}
        )
        hybrid_enabled = bool(self._cfg.retrieval_hybrid_enabled)
        lexical_healthy = not hybrid_enabled or lexical.get("available") is True
        backfill_busy = any(
            lexical_backfill.get(key)
            for key in (
                "queued",
                "running",
                "at_capacity",
                "cooldown_failures",
                "last_errors",
                "stopping",
            )
        )
        index_status = "error" if not lexical_healthy else "degraded" if backfill_busy else "ok"
        queue_status = (
            "error"
            if outbox.get("error")
            else "degraded"
            if int(outbox.get("total") or 0) > 0 and not worker_alive
            else "ok"
        )
        service_status = str(health.get("status") or "error")
        if service_status not in {"ok", "degraded", "error"}:
            service_status = "error"
        embedding_provider = self._provider_cfg.embedding_provider or self._provider_cfg.provider
        embedding_model = (
            self._provider_cfg.st_model
            if embedding_provider == "st"
            else "mock"
            if embedding_provider == "mock"
            else self._provider_cfg.openai_embedding_model
        )
        owner_spaces = tuple(ctx.visible_owner_spaces)
        try:
            model_usage = self.get_model_usage_metrics(
                days=1, account_id=ctx.account_id, owner_spaces=owner_spaces
            )
        except Exception:
            model_usage = {"cumulative": {"llm": {}, "embedding": {}}}
        llm_usage = model_usage.get("cumulative", {}).get("llm", {})
        embedding_usage = model_usage.get("cumulative", {}).get("embedding", {})
        retrieval = self.get_retrieval_monitoring_metrics(ctx.account_id, owner_spaces)
        try:
            vector_count = self._get_shared_vector_index().count({
                "account_id": ctx.account_id,
                **({"owner_space": list(owner_spaces)} if owner_spaces else {}),
            })
        except Exception:
            vector_count = None
        try:
            version = _metadata.version("context-engine")
        except _metadata.PackageNotFoundError:
            version = "dev"
        queries = int(retrieval["queries"])
        score_count = int(retrieval["score_count"])
        average_duration = retrieval["duration_total_ms"] / queries if queries else 0.0
        average_score = retrieval["score_total"] / score_count if score_count else 0.0
        zero_rate = retrieval["zero_results"] / queries * 100 if queries else 0.0
        score_range = (
            f"{retrieval['score_min']:.4f} - {retrieval['score_max']:.4f}"
            if retrieval["score_min"] is not None else "暂无结果"
        )
        components = [
            {
                "id": "service",
                "name": "服务",
                "description": "oGMemory HTTP 服务与核心运行状态。",
                "status": service_status,
                "details": {
                    "后端": health.get("backend", "og-memory"),
                    "版本": version,
                    "状态": health.get("status", "unknown"),
                    "错误": health.get("error", ""),
                },
            },
            {
                "id": "storage",
                "name": "存储",
                "description": "ContextFS 持久化与连接状态。",
                "status": "ok" if storage_healthy else "error",
                "details": {
                    "类型": storage_backend,
                    "连接": health.get("sql") or ("connected" if storage_healthy else "unavailable"),
                    "地址": health.get("agfs_url", ""),
                },
            },
            {
                "id": "models",
                "name": "模型",
                "description": "LLM 与 Embedding 模型配置。",
                "status": "ok" if health.get("llm") else "error",
                "details": {
                    "LLM": health.get("llm", "unavailable"),
                    "Embedding": embedding_model or "unavailable",
                    "Embedding Provider": embedding_provider or "unavailable",
                    "统计范围": "当前服务进程",
                },
                "tables": [{
                    "title": "模型用量",
                    "columns": ["类型", "模型", "Provider", "调用次数", "输入 Token", "输出 Token", "总 Token"],
                    "rows": [
                        ["LLM", health.get("llm", "unknown"), getattr(self._provider_cfg, "provider", "unknown"),
                         int(llm_usage.get("calls") or 0), int(llm_usage.get("input_tokens") or 0),
                         int(llm_usage.get("output_tokens") or 0), int(llm_usage.get("total_tokens") or 0)],
                        ["Embedding", embedding_model or "unknown", embedding_provider or "unknown",
                         int(embedding_usage.get("calls") or 0), int(embedding_usage.get("input_tokens") or 0),
                         "—", int(embedding_usage.get("total_tokens") or 0)],
                    ],
                }],
            },
            {
                "id": "index",
                "name": "索引",
                "description": "向量索引与混合检索投影。",
                "status": index_status,
                "details": {
                    "VectorDB": self._cfg.vector_db_type,
                    "混合检索": hybrid_enabled,
                    "BM25": lexical.get("available") if hybrid_enabled else "未启用",
                    "历史回填": "进行中" if backfill_busy else "就绪",
                    "向量数量": vector_count if vector_count is not None else "不可用",
                },
                "tables": [{
                    "title": "索引规模",
                    "columns": ["范围", "向量数量", "VectorDB", "BM25", "历史回填"],
                    "rows": [[ctx.account_id, vector_count if vector_count is not None else "不可用",
                              self._cfg.vector_db_type, lexical.get("available") if hybrid_enabled else "未启用",
                              "进行中" if backfill_busy else "就绪"]],
                }],
            },
            {
                "id": "queue",
                "name": "任务队列",
                "description": "当前账户的记忆索引异步任务。",
                "status": queue_status,
                "details": {
                    "账户": ctx.account_id,
                    "待处理": int(outbox.get("pending") or 0),
                    "处理中": int(outbox.get("processing") or 0),
                    "总计": int(outbox.get("total") or 0),
                    "后台 Worker": "运行中" if worker_alive else "按需启动",
                    "错误": outbox.get("error", ""),
                },
                "tables": [{
                    "title": "实时队列",
                    "columns": ["队列", "待处理", "处理中", "未完成", "Worker", "错误"],
                    "rows": [["Memory Index Outbox", int(outbox.get("pending") or 0),
                              int(outbox.get("processing") or 0), int(outbox.get("total") or 0),
                              "运行中" if worker_alive else "按需启动", outbox.get("error") or "—"]],
                }],
            },
            {
                "id": "retrieval",
                "name": "检索",
                "description": "记忆语义检索与混合召回能力。",
                "status": "error" if not storage_healthy else index_status,
                "details": {
                    "语义检索": True,
                    "混合检索": hybrid_enabled,
                    "Embedding": embedding_model or "unavailable",
                },
                "tables": [
                    {
                        "title": "检索质量与性能",
                        "columns": ["指标", "数值"],
                        "rows": [
                            ["查询总数", queries], ["失败查询", retrieval["failures"]],
                            ["返回结果", retrieval["results"]],
                            ["平均结果/查询", f"{retrieval['results'] / queries if queries else 0:.1f}"],
                            ["零结果查询", retrieval["zero_results"]], ["零结果率", f"{zero_rate:.1f}%"],
                            ["平均分数", f"{average_score:.4f}"], ["分数区间", score_range],
                            ["Rerank 使用", retrieval["rerank_used"]],
                            ["平均端到端延迟", f"{average_duration:.1f} ms"],
                            ["最大端到端延迟", f"{retrieval['duration_max_ms']:.1f} ms"],
                            ["最近请求 ID", retrieval["request_id"] or "暂无"],
                        ],
                    },
                    {
                        "title": "结果类型分布", "columns": ["记忆类型", "结果数"],
                        "rows": [[name, count] for name, count in sorted(retrieval["by_type"].items())],
                    },
                    {
                        "title": "检索阶段", "columns": ["阶段", "次数", "平均耗时", "最大耗时"],
                        "rows": [[name, stage["calls"], f"{stage['total_ms'] / stage['calls']:.1f} ms",
                                  f"{stage['max_ms']:.1f} ms"]
                                 for name, stage in sorted(retrieval["stages"].items())],
                    },
                ],
            },
        ]
        severity = {"ok": 0, "degraded": 1, "error": 2}
        overall_status = max(
            (component["status"] for component in components),
            key=severity.__getitem__,
        )
        return {
            "account_id": ctx.account_id,
            "version": version,
            "status": overall_status,
            "healthy": overall_status == "ok",
            "healthy_count": sum(component["status"] == "ok" for component in components),
            "total_count": len(components),
            "updated_at": _dt.now(UTC).isoformat(),
            "errors": [
                {"component": component["id"], "message": component["details"].get("错误")}
                for component in components if component["details"].get("错误")
            ],
            "components": components,
        }

    def _flush_session_tool_usage_async(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> None:
        snapshot = self._internal_tool_usage.session_snapshot(session_id)
        if not snapshot:
            self._internal_tool_usage.clear_session(session_id)
            return
        if not snapshot.get("account_id"):
            snapshot["account_id"] = ctx.account_id if ctx is not None else self._default_account_id
        with self._pending_tool_usage_lock:
            self._pending_tool_usage_snapshots[session_id] = snapshot

        def _write_snapshot():
            try:
                self.get_internal_tool_usage_store().write_session(snapshot)
                with self._pending_tool_usage_lock:
                    self._pending_tool_usage_snapshots.pop(session_id, None)
            except Exception as exc:
                logger.warning("tool usage AGFS write failed: session=%s error=%s", session_id[:8], exc)
                with self._pending_tool_usage_lock:
                    self._pending_tool_usage_snapshots[session_id] = snapshot
            finally:
                self._internal_tool_usage.clear_session(session_id)

        threading.Thread(target=_write_snapshot, daemon=True, name=f"tool-usage-{session_id[:8]}").start()

    def get_tool_usage_stats(self, params: dict | None = None) -> dict:
        params = params or {}
        session_id = params.get("session_id") or params.get("sessionId")
        user_id = params.get("user_id") or params.get("userId")
        start_time = params.get("start_time") or params.get("startTime")
        end_time = params.get("end_time") or params.get("endTime")
        pipeline_name = params.get("pipeline")
        include_rounds = str(params.get("include_rounds") or params.get("includeRounds") or "").lower() == "true"
        stats = self._internal_tool_usage.get_stats(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            pipeline_name=pipeline_name,
            include_rounds=include_rounds,
        )
        if session_id and not stats["summary"]["llm_tool_rounds"] and not stats["summary"]["tool_calls"]:
            with self._pending_tool_usage_lock:
                pending = deepcopy(self._pending_tool_usage_snapshots.get(session_id))
            if pending:
                if not include_rounds:
                    pending.pop("rounds", None)
                    pending.pop("rounds_limit", None)
                    pending.pop("rounds_truncated", None)
                return pending
            try:
                account_id = params.get("accountId") or params.get("account_id") or self._default_account_id
                return self.get_internal_tool_usage_store().read_session(
                    account_id,
                    session_id,
                    include_rounds=include_rounds,
                )
            except Exception:
                logger.debug("tool_usage: AGFS session stats unavailable", exc_info=True)
        return stats

    def health(self) -> dict:
        agfs_required = self._cfg.storage_backend == "agfs"
        sql_required = self._cfg.storage_backend == "sql"
        info: dict = {
            "backend": "og-memory",
            "storage_backend": self._cfg.storage_backend,
            "agfs": agfs_required and _HAS_AGFS,
            "sql": None,
            "status": "ok",
        }
        try:
            if agfs_required:
                if not _HAS_AGFS:
                    raise RuntimeError(
                        "AGFS backend configured but AGFS client is unavailable"
                    )
                AGFSClient(api_base_url=self._agfs_base_url).ls("/")
                info["agfs_url"] = self._agfs_base_url
            elif sql_required:
                dsn = self._cfg.sql_connection_string
                if not dsn:
                    raise RuntimeError(
                        "SQL backend configured but sql_connection_string is empty"
                    )
                pool = self._get_shared_sql_pool()
                conn = pool.get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    conn.rollback()
                finally:
                    pool.return_connection(conn)
                info["sql"] = "connected"
                info["sql_dialect"] = self._cfg.sql_dialect
                if self._cfg.retrieval_hybrid_enabled:
                    self._get_shared_vector_index()
                    lexical_health = self._lexical_index.health()
                    info["lexical"] = lexical_health
                    if not lexical_health.get("available", False):
                        raise RuntimeError("required openGauss BM25 projection is unavailable")
                    backfill_health = self._get_lexical_backfill().health()
                    info["lexical_backfill"] = backfill_health
                    if (
                        backfill_health.get("queued")
                        or backfill_health.get("running")
                        or backfill_health.get("at_capacity")
                        or backfill_health.get("cooldown_failures")
                        or backfill_health.get("last_errors")
                        or backfill_health.get("stopping")
                    ):
                        info["status"] = "degraded"
                        info["error"] = "lexical historical projection is not ready"
            if self._cfg.vector_db_type.strip().casefold() == "milvus":
                info["vector_db"] = {"type": "milvus", "available": False}
                # A cached client can outlive a failed server or revoked grant.
                try:
                    self._get_shared_vector_index().probe(self._cfg.account_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"Milvus vector backend unavailable ({type(exc).__name__})"
                    ) from None
                info["vector_db"]["available"] = True
            llm = self.get_llm()
            info["llm"] = getattr(llm, "model", "unknown")
            if self._cfg.skill_sedimentation_enabled:
                worker_status = (
                    self._skill_sedimentation_worker.status()
                    if self._skill_sedimentation_worker is not None
                    else {"running": False, "last_error": "worker not initialized"}
                )
                worker_status["thread_alive"] = bool(
                    self._skill_sedimentation_thread
                    and self._skill_sedimentation_thread.is_alive()
                )
                trace_store = self._get_trace_event_store()
                worker_status["scheduler_backlog"] = (
                    trace_store.count_skill_work_accounts() if trace_store else None
                )
                info["skill_sedimentation"] = worker_status
                if not worker_status["thread_alive"]:
                    info["status"] = "error"
                    info["error"] = "skill sedimentation worker is not running"
                elif worker_status.get("last_error"):
                    info["status"] = "error"
                    info["error"] = str(worker_status["last_error"])
        except Exception as exc:
            info["status"] = "error"
            info["error"] = str(exc)
        return info
