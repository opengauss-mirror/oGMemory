"""Durable, server-owned synchronization for registered code workspaces.

The service is intentionally below HTTP/MCP adapters.  It owns registry state,
snapshot comparison, trigger coalescing, and leases, but delegates every
resource mutation to :class:`CodeResourceLifecycleService`.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import RequestContext
from core.resource_identity import (
    ResourceIdentityError,
    compute_root_fingerprint,
    normalize_resource_relative_path,
)
from extraction.code_processor import MAX_CODE_FILE_SIZE
from extraction.code_workspace import iter_code_files, normalize_language_hints
from server.control_plane_store import CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID
from service.code_resource_lifecycle import (
    DEFAULT_CODE_RESOURCE_MAX_FILES,
    normalize_resource_root_slug,
)

logger = logging.getLogger(__name__)

_TRIGGER_MODES = frozenset({"hybrid", "manual", "external_event"})
_WAIT_MODES = frozenset({"accepted", "filesystem_complete", "index_visible"})
_CODE_WORKSPACE_ALLOWED_ROOTS_CONFIG = "code_workspace_allowed_roots"
_SYNC_DIAGNOSTIC_ERROR_KINDS = frozenset(
    {"timeout", "lifecycle_failure", "workspace_unavailable", "workspace_invalid"}
)
_SYNC_PARTIAL_FAILURE_KINDS = frozenset({"timeout", "lifecycle_failure"})
_SNAPSHOT_HASH_CHUNK_BYTES = 1024 * 1024


def _streaming_file_sha256(path: Path) -> str | None:
    """Hash at most the code-reader limit; return ``None`` when it is exceeded."""

    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_SNAPSHOT_HASH_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_CODE_FILE_SIZE:
                return None
            digest.update(chunk)
    return digest.hexdigest()


class _WorkspaceUnavailableError(RuntimeError):
    """Raised internally when a registered server-local root cannot be read."""


@dataclass(frozen=True)
class _CodeWorkspaceRootPolicy:
    """Canonical server-owned boundary for local workspace authority."""

    allowed_roots: tuple[Path, ...]

    @classmethod
    def from_config(cls, config: object) -> _CodeWorkspaceRootPolicy:
        raw_roots = getattr(config, _CODE_WORKSPACE_ALLOWED_ROOTS_CONFIG, ())
        if raw_roots is None or not isinstance(raw_roots, (list, tuple)):
            raise ValueError(
                "code_workspace_allowed_roots must be a list of configured directories"
            )
        canonical_roots: list[Path] = []
        for raw_root in raw_roots:
            if not isinstance(raw_root, str) or not raw_root.strip():
                raise ValueError(
                    "code_workspace_allowed_roots must contain non-empty directory paths"
                )
            try:
                candidate = Path(raw_root).expanduser()
            except RuntimeError as exc:
                raise ValueError(
                    "code_workspace_allowed_roots must contain existing absolute directories"
                ) from exc
            if not candidate.is_absolute():
                raise ValueError(
                    "code_workspace_allowed_roots must contain existing absolute directories"
                )
            try:
                canonical_root = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    "code_workspace_allowed_roots must contain existing absolute directories"
                ) from exc
            if not canonical_root.is_dir():
                raise ValueError(
                    "code_workspace_allowed_roots must contain existing absolute directories"
                )
            if canonical_root == canonical_root.parent:
                raise ValueError("code_workspace_allowed_roots must not include filesystem root")
            if canonical_root not in canonical_roots:
                canonical_roots.append(canonical_root)
        return cls(tuple(canonical_roots))

    def require_contains(self, root: Path) -> None:
        """Reject an empty policy or a root outside its canonical directories."""

        if not self.allowed_roots:
            raise ValueError(
                "code_workspace_allowed_roots must contain at least one directory"
            )
        try:
            canonical_root = root.expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError("workspace root cannot be resolved") from exc
        for allowed_root in self.allowed_roots:
            try:
                canonical_root.relative_to(allowed_root)
            except ValueError:
                continue
            return
        raise ValueError("workspace root is outside configured allowed roots")


_WORKSPACE_RESOLUTION_REASONS = frozenset(
    {
        "workspace_not_registered",
        "workspace_scope_ambiguous",
        "workspace_unavailable",
        "workspace_invalid",
    }
)


class WorkspaceResolutionError(RuntimeError):
    """A path-free candidate-local root selection outcome."""

    def __init__(self, reason: str, *, available_root_count: int = 0) -> None:
        if reason not in _WORKSPACE_RESOLUTION_REASONS:
            reason = "workspace_invalid"
        if (
            isinstance(available_root_count, bool)
            or not isinstance(available_root_count, int)
            or available_root_count < 0
        ):
            available_root_count = 0
        self.reason = reason
        # This is a path-free cardinality only.  It lets public diagnostics
        # explain the sole-root decision without disclosing registered slugs,
        # fingerprints, or physical paths.
        self.available_root_count = available_root_count
        super().__init__(reason)


@dataclass(frozen=True)
class ResolvedCodeWorkspace:
    """Protected descriptor usable only by internal candidate retrieval.

    ``physical_root`` must never cross a public adapter, evidence, diagnostics,
    or normal log boundary.  It exists here solely so the retrieval service can
    revalidate and read the already-registered workspace without mutating it.
    """

    root_id: str
    root_slug: str
    root_fingerprint: str
    physical_root: Path
    max_files: int
    language_hints: Mapping[str, str]
    available_root_count: int = 1


def _strict_registered_workspace(
    state: Mapping[str, Any],
    *,
    root_id: str,
    require_manual_registration: bool = False,
    root_policy: _CodeWorkspaceRootPolicy | None = None,
) -> ResolvedCodeWorkspace:
    """Revalidate one protected registry row without changing its state."""

    if not isinstance(state, Mapping):
        raise WorkspaceResolutionError("workspace_not_registered")
    if state.get("root_id") != root_id:
        raise WorkspaceResolutionError("workspace_invalid")
    if require_manual_registration:
        # Candidate-local is intentionally limited to an explicit manual
        # registration.  A hybrid/external-event root is a different sync
        # lifecycle and must not become query-time filesystem authority just
        # because it happens to share the protected registry.
        if state.get("trigger_mode") != "manual":
            raise WorkspaceResolutionError("workspace_not_registered")
        if state.get("watch_enabled") is not False:
            raise WorkspaceResolutionError("workspace_invalid")
    root_slug = state.get("resource_root_slug")
    root_fingerprint = state.get("root_fingerprint")
    raw_path = state.get("workspace_path")
    if (
        not isinstance(root_slug, str)
        or not root_slug
        or not isinstance(root_fingerprint, str)
        or not root_fingerprint
        or not isinstance(raw_path, str)
        or not raw_path
    ):
        raise WorkspaceResolutionError("workspace_invalid")
    try:
        if normalize_resource_root_slug(root_slug) != root_slug:
            raise WorkspaceResolutionError("workspace_invalid")
    except ResourceIdentityError:
        raise WorkspaceResolutionError("workspace_invalid") from None
    try:
        root = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceResolutionError("workspace_unavailable") from None
    if not root.is_dir():
        raise WorkspaceResolutionError("workspace_unavailable")
    if root_policy is not None:
        try:
            root_policy.require_contains(root)
        except ValueError:
            raise WorkspaceResolutionError("workspace_invalid") from None
    try:
        current_fingerprint = compute_root_fingerprint(root)
    except (OSError, RuntimeError):
        raise WorkspaceResolutionError("workspace_unavailable") from None
    if current_fingerprint != root_fingerprint:
        raise WorkspaceResolutionError("workspace_invalid")
    try:
        max_files = int(state.get("max_files", DEFAULT_CODE_RESOURCE_MAX_FILES))
    except (TypeError, ValueError):
        raise WorkspaceResolutionError("workspace_invalid") from None
    if isinstance(state.get("max_files", max_files), bool) or max_files < 1:
        raise WorkspaceResolutionError("workspace_invalid")
    try:
        hints = normalize_language_hints(state.get("language_hints") or {})
    except (TypeError, ValueError, ResourceIdentityError):
        raise WorkspaceResolutionError("workspace_invalid") from None
    return ResolvedCodeWorkspace(
        root_id=root_id,
        root_slug=root_slug,
        root_fingerprint=root_fingerprint,
        physical_root=root,
        max_files=max_files,
        language_hints=hints,
    )


class CodeWorkspaceRegistryResolver:
    """Read-only selector for an account's registered workspaces.

    Unlike :class:`CodeWorkspaceSyncService`, this class has no lifecycle,
    scheduler, mutation, watcher, or Outbox dependencies. Candidate-local
    retrieval uses :meth:`resolve` to select exactly one current manual root;
    registered-root operations such as exact source editing use
    :meth:`resolve_registered` without weakening the same account, binding,
    fingerprint, and availability checks.
    """

    def __init__(self, control_store, *, config: object | None = None) -> None:
        if control_store is None:
            raise RuntimeError("control plane store is required for workspace resolution")
        self._store = control_store
        # Earlier internal callers constructed the resolver with only a store.
        # Keep that read-only API intact while allowing public adapters to pass
        # the process config and enforce the registered-root filesystem policy.
        self._root_policy = (
            _CodeWorkspaceRootPolicy.from_config(config) if config is not None else None
        )

    def resolve(self, *, ctx: RequestContext, root_slugs: tuple[str, ...] = ()) -> ResolvedCodeWorkspace:
        """Resolve one strict manual root for candidate-local retrieval."""

        return self._resolve(
            ctx=ctx,
            root_slugs=root_slugs,
            require_manual_registration=True,
        )

    def resolve_registered(
        self,
        *,
        ctx: RequestContext,
        root_slugs: tuple[str, ...] = (),
    ) -> ResolvedCodeWorkspace:
        """Resolve one strict registered root without selecting a sync mode."""

        return self._resolve(
            ctx=ctx,
            root_slugs=root_slugs,
            require_manual_registration=False,
        )

    def _resolve(
        self,
        *,
        ctx: RequestContext,
        root_slugs: tuple[str, ...],
        require_manual_registration: bool,
    ) -> ResolvedCodeWorkspace:
        if not isinstance(ctx, RequestContext) or not ctx.account_id:
            raise WorkspaceResolutionError("workspace_not_registered")
        requested = tuple(root_slugs)
        if len(requested) > 1:
            raise WorkspaceResolutionError("workspace_scope_ambiguous")
        bindings = self._store.read_code_workspace_sync_state(
            ctx.account_id,
            CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,
        )
        by_slug = dict(bindings.get("by_slug") or {}) if isinstance(bindings, Mapping) else {}
        if requested:
            return self._resolve_explicit(
                ctx.account_id,
                requested[0],
                by_slug,
                require_manual_registration=require_manual_registration,
            )
        return self._resolve_sole_valid(
            ctx.account_id,
            by_slug,
            require_manual_registration=require_manual_registration,
        )

    def _resolve_explicit(
        self,
        account_id: str,
        root_slug: str,
        by_slug: Mapping[str, Any],
        *,
        require_manual_registration: bool,
    ) -> ResolvedCodeWorkspace:
        binding = by_slug.get(root_slug)
        if not isinstance(binding, Mapping):
            raise WorkspaceResolutionError("workspace_not_registered")
        root_id = binding.get("root_id")
        expected_fingerprint = binding.get("root_fingerprint")
        if not isinstance(root_id, str) or not root_id or not isinstance(expected_fingerprint, str):
            raise WorkspaceResolutionError("workspace_invalid")
        state = self._store.read_code_workspace_sync_state(account_id, root_id)
        if state is None:
            raise WorkspaceResolutionError("workspace_not_registered")
        workspace = _strict_registered_workspace(
            state,
            root_id=root_id,
            require_manual_registration=require_manual_registration,
            root_policy=self._root_policy,
        )
        if (
            workspace.root_slug != root_slug
            or workspace.root_fingerprint != expected_fingerprint
        ):
            raise WorkspaceResolutionError("workspace_invalid")
        return workspace

    def _resolve_sole_valid(
        self,
        account_id: str,
        by_slug: Mapping[str, Any],
        *,
        require_manual_registration: bool,
    ) -> ResolvedCodeWorkspace:
        states = self._store.list_code_workspace_sync_states(account_id)
        if not states:
            raise WorkspaceResolutionError("workspace_not_registered")
        valid: list[ResolvedCodeWorkspace] = []
        failures: list[str] = []
        for root_id, state in states:
            # Production stores omit the protected binding row from list
            # results, but tolerate simpler protocol doubles as well.  That
            # row is an account-wide index, never a resolvable workspace.
            if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            try:
                workspace = _strict_registered_workspace(
                    state,
                    root_id=root_id,
                    require_manual_registration=require_manual_registration,
                    root_policy=self._root_policy,
                )
                binding = by_slug.get(workspace.root_slug)
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("root_id") != root_id
                    or binding.get("root_fingerprint") != workspace.root_fingerprint
                ):
                    raise WorkspaceResolutionError("workspace_invalid")
            except WorkspaceResolutionError as exc:
                failures.append(exc.reason)
                continue
            valid.append(workspace)
        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            raise WorkspaceResolutionError(
                "workspace_scope_ambiguous",
                available_root_count=len(valid),
            )
        if "workspace_invalid" in failures:
            raise WorkspaceResolutionError("workspace_invalid")
        if "workspace_unavailable" in failures:
            raise WorkspaceResolutionError("workspace_unavailable")
        raise WorkspaceResolutionError("workspace_not_registered")


class CodeWorkspaceSyncService:
    """Run canonical workspace syncs through the existing lifecycle service."""

    def __init__(
        self,
        *,
        control_store,
        lifecycle_service,
        config,
        invalidate_cache: Callable[[str], None] | None = None,
        wake_outbox: Callable[[], None] | None = None,
        is_index_visible: Callable[[RequestContext], bool] | None = None,
        context_resolver: Callable[[str], RequestContext | None] | None = None,
        worker_id: str | None = None,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if control_store is None:
            raise RuntimeError("control plane store is required for workspace sync")
        if lifecycle_service is None:
            raise RuntimeError("code lifecycle service is required for workspace sync")
        self._store = control_store
        self._lifecycle = lifecycle_service
        self._cfg = config
        self._root_policy = _CodeWorkspaceRootPolicy.from_config(config)
        self._invalidate_cache = invalidate_cache
        self._wake_outbox = wake_outbox
        self._is_index_visible = is_index_visible
        self._context_resolver = context_resolver
        self._worker_id = worker_id or f"code-sync-{uuid.uuid4().hex}"
        self._clock = clock
        self._monotonic = monotonic_clock
        self._lock = threading.RLock()
        self._scheduler = None
        self._known_contexts: dict[tuple[str, str], RequestContext] = {}

    def register(
        self,
        *,
        path: str,
        ctx: RequestContext,
        name: str | None = None,
        trigger_mode: str | None = None,
        watch_enabled: bool | None = None,
        reconcile_interval_seconds: int | None = None,
        max_files: int | None = None,
        wait_mode: str | None = None,
        language_hints: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one authorized workspace without publishing its local path."""

        if not isinstance(path, str) or not path.strip():
            raise ValueError("workspace path is required")
        root, server_accessible = self._resolve_registration_root(path)
        self._root_policy.require_contains(root)
        fingerprint = compute_root_fingerprint(root)
        root_id = self._root_id(ctx.account_id, fingerprint)
        mode, effective_watch = self._validate_trigger_settings(
            trigger_mode,
            watch_enabled,
            server_accessible,
        )
        interval = self._positive_setting(
            reconcile_interval_seconds,
            "reconcile_interval_seconds",
            int(getattr(self._cfg, "code_sync_reconcile_interval_seconds", 300)),
        )
        cap = self._max_files_setting(max_files)
        effective_wait_mode = self._wait_mode(wait_mode)
        hints = normalize_language_hints(language_hints)
        now = self._clock()
        canonical_path = str(root)

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            state = dict(current or {})
            generation = int(state.get("dirty_generation", 0)) + 1
            state.update(
                {
                    "version": 1,
                    "root_id": root_id,
                    "resource_root_slug": root_slug,
                    "root_fingerprint": fingerprint,
                    "workspace_path": canonical_path,
                    "server_accessible": server_accessible,
                    "trigger_mode": mode,
                    "watch_enabled": effective_watch,
                    "reconcile_interval_seconds": interval,
                    "max_files": cap,
                    "wait_mode": effective_wait_mode,
                    "language_hints": hints,
                    "snapshot": dict(state.get("snapshot") or {}),
                    "dirty": True,
                    "dirty_generation": generation,
                    "index_visible": False,
                    "last_dirty_at": now,
                    "registered_at": state.get("registered_at", now),
                    "updated_at": now,
                }
            )
            return state

        with self._store.code_workspace_registration_lock():
            root_slug, fingerprint = self._resolve_registration_binding(
                root,
                server_accessible=server_accessible,
                ctx=ctx,
                name=name,
                root_id=root_id,
                expected_fingerprint=fingerprint,
            )
            self._reserve_root_binding(
                ctx=ctx,
                root_id=root_id,
                root_slug=root_slug,
                fingerprint=fingerprint,
            )
            state = self._store.mutate_code_workspace_sync_state(
                ctx.account_id, root_id, transition
            )
        self._remember_context(ctx, root_id)
        if bool(getattr(self._cfg, "code_sync_enabled", False)):
            self.start()
        return self._render_state(state, accepted=True, filesystem_complete=False)

    def _check_directory_conflict(self, root: Path) -> None:
        for _, registered_id, state in self._store.list_code_workspace_sync_registrations():
            if registered_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            path = state.get("workspace_path")
            if not isinstance(path, str) or "\x00" in path:
                raise RuntimeError("workspace registry record is unreadable")
            registered_root = Path(path)
            if not registered_root.is_absolute() or ".." in registered_root.parts:
                raise RuntimeError("workspace registry record is unreadable")
            if root.is_relative_to(registered_root) or registered_root.is_relative_to(root):
                raise FileExistsError("workspace directory conflicts with an existing registration")

    def get_state(self, *, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        """Return a non-sensitive freshness projection for one registered root."""

        state = self._require_state(root_id, ctx)
        self._remember_context(ctx, root_id)
        return self._render_state(
            state,
            accepted=False,
            filesystem_complete=not bool(state.get("syncing", False))
            and not bool(state.get("dirty", False)),
        )

    def list_states(self, *, ctx: RequestContext) -> list[dict[str, Any]]:
        """List this account's safe registry projections for reconciliation callers."""

        result: list[dict[str, Any]] = []
        for root_id, state in self._store.list_code_workspace_sync_states(ctx.account_id):
            if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            self._remember_context(ctx, root_id)
            result.append(
                self._render_state(
                    state,
                    accepted=False,
                    filesystem_complete=not bool(state.get("syncing", False))
                    and not bool(state.get("dirty", False)),
                )
            )
        return result

    def mark_dirty(self, *, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        """Accept a manual, watcher, or external-event dirty signal only."""

        self._require_enabled()
        state = self._mark_dirty_state(root_id, ctx)
        return self._render_state(state, accepted=True, filesystem_complete=False)

    def external_event(
        self,
        *,
        root_id: str,
        ctx: RequestContext,
        wait_mode: str | None = None,
    ) -> dict[str, Any]:
        """Handle a path-free external event through the canonical queue contract."""

        self._require_enabled()
        state = self._require_state(root_id, ctx)
        if state.get("trigger_mode") == "manual":
            raise ValueError("workspace sync external events require external_event or hybrid mode")
        state = self._mark_dirty_state(root_id, ctx)
        effective_wait_mode = self._wait_mode(wait_mode or state.get("wait_mode"))
        if effective_wait_mode == "accepted":
            return self._render_state(state, accepted=True, filesystem_complete=False)
        return self._sync_root(
            root_id,
            ctx,
            trigger="external_event",
            wait_mode=effective_wait_mode,
        )

    def sync(
        self,
        *,
        root_id: str,
        ctx: RequestContext,
        wait_mode: str | None = None,
    ) -> dict[str, Any]:
        """Run an explicit manual sync through one bounded lifecycle path."""

        self._require_enabled()
        state = self._mark_dirty_state(root_id, ctx)
        return self._sync_root(
            root_id,
            ctx,
            trigger="manual",
            wait_mode=self._wait_mode(wait_mode or state.get("wait_mode")),
        )

    def poll(self, *, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        """Polling watcher fast path: mark dirty only on metadata changes."""

        self._require_enabled()
        state = self._require_state(root_id, ctx)
        if not bool(state.get("server_accessible")) or not bool(state.get("watch_enabled")):
            return self._render_state(
                state,
                accepted=False,
                filesystem_complete=not bool(state.get("syncing", False))
                and not bool(state.get("dirty", False)),
            )
        _, current_snapshot = self._discover_snapshot(
            state,
            include_content_hash=False,
        )
        previous_snapshot = dict(state.get("snapshot") or {})
        if self._snapshot_metadata_changed(previous_snapshot, current_snapshot):
            state = self._mark_dirty_state(root_id, ctx)
        now = self._clock()
        state = self._store.mutate_code_workspace_sync_state(
            ctx.account_id,
            root_id,
            lambda current: self._with_timestamp(current, "last_poll_at", now),
        )
        return self._render_state(
            state,
            accepted=False,
            filesystem_complete=not bool(state.get("syncing", False))
            and not bool(state.get("dirty", False)),
        )

    def reconcile(self, *, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        """Scheduled fallback: mark the root dirty and use the same sync run."""

        self._require_enabled()
        state = self._require_state(root_id, ctx)
        if not bool(state.get("server_accessible")) or state.get("trigger_mode") != "hybrid":
            return self._render_state(
                state,
                accepted=False,
                filesystem_complete=not bool(state.get("syncing", False))
                and not bool(state.get("dirty", False)),
            )
        state = self._mark_dirty_state(root_id, ctx)
        now = self._clock()
        self._store.mutate_code_workspace_sync_state(
            ctx.account_id,
            root_id,
            lambda current: self._with_timestamp(current, "last_reconcile_at", now),
        )
        return self._sync_root(
            root_id,
            ctx,
            trigger="scheduled",
            wait_mode=self._wait_mode(state.get("wait_mode")),
        )

    def run_pending(self, *, ctx: RequestContext) -> list[dict[str, Any]]:
        """Run due dirty roots for one account; scheduler code owns no lifecycle logic."""

        self._require_enabled()
        results: list[dict[str, Any]] = []
        now = self._clock()
        for root_id, state in self._store.list_code_workspace_sync_states(ctx.account_id):
            if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            if len(results) >= self._max_pending_roots():
                break
            self._remember_context(ctx, root_id)
            if not self._queue_sync_enabled(state) or not bool(state.get("dirty")):
                continue
            retry_at = self._float_or_none(state.get("next_retry_at"))
            if (
                (retry_at is not None and retry_at > now)
                or not self._debounce_elapsed(state, now)
            ):
                continue
            results.append(
                self._sync_root(
                    root_id,
                    ctx,
                    trigger="scheduled",
                    wait_mode="accepted",
                )
            )
        return results

    def start(self) -> None:
        """Start lightweight polling/reconciliation scheduling when the gate is enabled."""

        if not bool(getattr(self._cfg, "code_sync_enabled", False)):
            return
        self._recover_registered_contexts()
        with self._lock:
            if self._scheduler is not None:
                return
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
            except ImportError:
                logger.warning("Code workspace sync scheduler is unavailable")
                return
            interval = min(
                int(getattr(self._cfg, "code_sync_watch_interval_seconds", 5)),
                int(getattr(self._cfg, "code_sync_reconcile_interval_seconds", 300)),
            )
            scheduler = BackgroundScheduler()
            scheduler.add_job(
                self._scheduled_tick,
                "interval",
                seconds=max(1, interval),
                id=f"code-workspace-sync-{self._worker_id}",
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
            scheduler.start()
            self._scheduler = scheduler

    def shutdown(self) -> None:
        """Stop this process's scheduler without changing durable registry state."""

        with self._lock:
            scheduler = self._scheduler
            self._scheduler = None
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    def _sync_root(
        self,
        root_id: str,
        ctx: RequestContext,
        *,
        trigger: str,
        wait_mode: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] | None = None
        for run_index in range(2):
            state, completed, rerun = self._sync_once(root_id, ctx, trigger=trigger)
            result = self._render_state(state, accepted=True, filesystem_complete=completed)
            if not rerun or run_index == 1:
                break
        if result is None:
            raise RuntimeError("workspace sync did not produce a state")
        if wait_mode == "index_visible" and result["filesystem_complete"]:
            state = self._wait_for_index_visibility(root_id, ctx)
            result = self._render_state(state, accepted=True, filesystem_complete=True)
        return result

    def _sync_once(
        self,
        root_id: str,
        ctx: RequestContext,
        *,
        trigger: str,
    ) -> tuple[dict[str, Any], bool, bool]:
        claimed, state, generation = self._claim_run(root_id, ctx, trigger=trigger)
        if not claimed:
            return state, False, False
        try:
            root, current_snapshot = self._discover_snapshot(state)
        except _WorkspaceUnavailableError:
            return self._finish_run(
                root_id,
                ctx,
                generation=generation,
                snapshot=dict(state.get("snapshot") or {}),
                changed=0,
                deleted=0,
                failed=0,
                timed_out=False,
                error_kind="workspace_unavailable",
                index_visible=False,
            )
        except (OSError, ResourceIdentityError, ValueError):
            return self._finish_run(
                root_id,
                ctx,
                generation=generation,
                snapshot=dict(state.get("snapshot") or {}),
                changed=0,
                deleted=0,
                failed=0,
                timed_out=False,
                error_kind="workspace_invalid",
                index_visible=False,
            )

        previous_snapshot = dict(state.get("snapshot") or {})
        changed_paths = sorted(
            path
            for path, file_state in current_snapshot.items()
            if previous_snapshot.get(path) != file_state
        )
        deleted_paths = sorted(set(previous_snapshot) - set(current_snapshot))
        next_snapshot = dict(previous_snapshot)
        completed_changed = 0
        completed_deleted = 0
        failed = 0
        timed_out = False
        deadline = self._monotonic() + float(
            int(getattr(self._cfg, "code_sync_run_timeout_seconds", 60))
        )
        hints = dict(state.get("language_hints") or {})
        actions = [("delete", path) for path in deleted_paths] + [
            ("refresh", path) for path in changed_paths
        ]

        for action_index, (kind, relative_path) in enumerate(actions):
            if self._monotonic() >= deadline:
                timed_out = True
                failed += len(actions) - action_index
                break
            if not self._renew_run_lease(root_id, ctx):
                failed += len(actions) - action_index
                break
            try:
                if kind == "delete":
                    self._lifecycle.delete_path(
                        path=relative_path,
                        workspace_root=str(root),
                        ctx=ctx,
                        name=state["resource_root_slug"],
                    )
                    next_snapshot.pop(relative_path, None)
                    completed_deleted += 1
                else:
                    self._lifecycle.refresh(
                        path=relative_path,
                        workspace_root=str(root),
                        ctx=ctx,
                        name=state["resource_root_slug"],
                        language_hint=hints.get(relative_path),
                    )
                    next_snapshot[relative_path] = current_snapshot[relative_path]
                    completed_changed += 1
            except Exception:
                failed += 1

        if completed_changed or completed_deleted:
            self._after_lifecycle_mutation(ctx.account_id)
        index_visible = self._probe_index_visibility(ctx)
        return self._finish_run(
            root_id,
            ctx,
            generation=generation,
            snapshot=next_snapshot,
            changed=completed_changed,
            deleted=completed_deleted,
            failed=failed,
            timed_out=timed_out,
            error_kind="timeout" if timed_out else ("lifecycle_failure" if failed else None),
            index_visible=index_visible,
        )

    def _claim_run(
        self,
        root_id: str,
        ctx: RequestContext,
        *,
        trigger: str,
    ) -> tuple[bool, dict[str, Any], int]:
        now = self._clock()
        claimed = False
        generation = 0
        lease_seconds = max(
            60,
            int(getattr(self._cfg, "code_sync_run_timeout_seconds", 60)) + 30,
        )

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal claimed, generation
            state = self._require_current_state(current)
            lease_owner = state.get("lease_owner")
            lease_expires_at = self._float_or_none(state.get("lease_expires_at"))
            retry_at = self._float_or_none(state.get("next_retry_at"))
            if lease_owner and lease_owner != self._worker_id and lease_expires_at and lease_expires_at > now:
                return state
            if trigger != "manual" and retry_at is not None and retry_at > now:
                return state
            generation = int(state.get("dirty_generation", 0))
            state.update(
                {
                    "lease_owner": self._worker_id,
                    "lease_expires_at": now + lease_seconds,
                    "syncing": True,
                    "dirty": False,
                    "index_visible": False,
                    "last_attempt_at": now,
                    "run_generation": generation,
                }
            )
            claimed = True
            return state

        state = self._store.mutate_code_workspace_sync_state(ctx.account_id, root_id, transition)
        return claimed, state, generation

    def _renew_run_lease(self, root_id: str, ctx: RequestContext) -> bool:
        now = self._clock()
        renewed = False
        lease_seconds = max(
            60,
            int(getattr(self._cfg, "code_sync_run_timeout_seconds", 60)) + 30,
        )

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal renewed
            state = self._require_current_state(current)
            if state.get("lease_owner") != self._worker_id or not bool(state.get("syncing")):
                return state
            state["lease_expires_at"] = now + lease_seconds
            renewed = True
            return state

        self._store.mutate_code_workspace_sync_state(ctx.account_id, root_id, transition)
        return renewed

    def _finish_run(
        self,
        root_id: str,
        ctx: RequestContext,
        *,
        generation: int,
        snapshot: Mapping[str, Mapping[str, int | str]],
        changed: int,
        deleted: int,
        failed: int,
        timed_out: bool,
        error_kind: str | None,
        index_visible: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        now = self._clock()
        completed = False
        rerun = False

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal completed, rerun
            state = self._require_current_state(current)
            if state.get("lease_owner") != self._worker_id:
                return state
            newest_generation = int(state.get("dirty_generation", 0))
            rerun = newest_generation > generation
            retry_failed = failed > 0 or timed_out or error_kind is not None
            retry_needed = retry_failed or rerun
            state.update(
                {
                    "snapshot": {key: dict(value) for key, value in snapshot.items()},
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "syncing": False,
                    "dirty": retry_needed,
                    "index_visible": bool(index_visible and not retry_needed),
                    "last_changed_files": changed,
                    "last_deleted_files": deleted,
                    "last_failed_files": failed,
                    "last_error_kind": error_kind,
                    "updated_at": now,
                }
            )
            if error_kind == "workspace_unavailable":
                state.update(
                    {
                        "server_accessible": False,
                        "trigger_mode": "external_event",
                        "watch_enabled": False,
                    }
                )
            if retry_failed:
                state["next_retry_at"] = now + int(
                    getattr(self._cfg, "code_sync_failure_backoff_seconds", 30)
                )
            elif rerun:
                # A concurrent dirty mark must get one coalesced run without
                # being delayed by the failure backoff reserved for real errors.
                state["next_retry_at"] = None
            else:
                state["last_success_at"] = now
                state["next_retry_at"] = None
            if state["index_visible"]:
                state["last_index_visible_at"] = now
            completed = not retry_needed
            return state

        state = self._store.mutate_code_workspace_sync_state(ctx.account_id, root_id, transition)
        return state, completed, rerun and not (failed > 0 or timed_out or error_kind is not None)

    def _discover_snapshot(
        self,
        state: Mapping[str, Any],
        *,
        include_content_hash: bool = True,
    ) -> tuple[Path, dict[str, dict[str, int | str]]]:
        try:
            workspace = _strict_registered_workspace(
                state,
                root_id=str(state.get("root_id") or ""),
                root_policy=self._root_policy,
            )
        except WorkspaceResolutionError as exc:
            if exc.reason == "workspace_invalid":
                raise ResourceIdentityError("workspace identity changed") from None
            raise _WorkspaceUnavailableError from None
        root = workspace.physical_root
        max_files = workspace.max_files
        hints = workspace.language_hints
        snapshot: dict[str, dict[str, int | str]] = {}
        for source_path in iter_code_files(root, max_files=max_files, language_hints=hints):
            relative_path = normalize_resource_relative_path(root, source_path)
            stat = source_path.stat()
            if stat.st_size > MAX_CODE_FILE_SIZE:
                continue
            file_state: dict[str, int | str] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
            if include_content_hash:
                content_hash = _streaming_file_sha256(source_path)
                if content_hash is None:
                    continue
                file_state["sha256"] = content_hash
            language_hint = hints.get(relative_path)
            if language_hint is not None:
                file_state["language_hint"] = language_hint
            snapshot[relative_path] = file_state
        return root, snapshot

    @staticmethod
    def _snapshot_metadata_changed(
        previous: Mapping[str, Any],
        current: Mapping[str, Mapping[str, int | str]],
    ) -> bool:
        """Compare watcher-safe metadata without reading file contents."""

        if set(previous) != set(current):
            return True
        fields = ("mtime_ns", "size", "language_hint")
        return any(
            not isinstance(previous.get(path), Mapping)
            or any(previous[path].get(field) != state.get(field) for field in fields)
            for path, state in current.items()
        )

    def _mark_dirty_state(self, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        now = self._clock()

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            state = self._require_current_state(current)
            if bool(state.get("dirty")) and not bool(state.get("syncing")):
                return state
            state.update(
                {
                    "dirty": True,
                    "dirty_generation": int(state.get("dirty_generation", 0)) + 1,
                    "index_visible": False,
                    "last_dirty_at": now,
                    "updated_at": now,
                }
            )
            return state

        state = self._state_mapping(
            self._store.mutate_code_workspace_sync_state(ctx.account_id, root_id, transition)
        )
        self._remember_context(ctx, root_id)
        return state

    def _wait_for_index_visibility(self, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        timeout = float(int(getattr(self._cfg, "code_sync_run_timeout_seconds", 60)))
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            if self._probe_index_visibility(ctx):
                now = self._clock()
                return self._state_mapping(
                    self._store.mutate_code_workspace_sync_state(
                        ctx.account_id,
                        root_id,
                        lambda current: self._mark_index_visible(current, now),
                    )
                )
            time.sleep(min(0.05, max(0.0, deadline - self._monotonic())))
        return self._require_state(root_id, ctx)

    def _scheduled_tick(self) -> None:
        if not bool(getattr(self._cfg, "code_sync_enabled", False)):
            return
        self._recover_registered_contexts()
        processed = 0
        for (account_id, root_id), ctx in self._known_context_items():
            if processed >= self._max_pending_roots():
                break
            if ctx.account_id != account_id:
                continue
            try:
                state = self._require_state(root_id, ctx)
                if not self._queue_sync_enabled(state):
                    continue
                now = self._clock()
                if (
                    state.get("trigger_mode") == "hybrid"
                    and state.get("watch_enabled")
                    and now - float(state.get("last_poll_at") or 0) >= int(
                        getattr(self._cfg, "code_sync_watch_interval_seconds", 5)
                    )
                ):
                    self.poll(root_id=root_id, ctx=ctx)
                    state = self._require_state(root_id, ctx)
                if (
                    state.get("trigger_mode") == "hybrid"
                    and now - float(state.get("last_reconcile_at") or 0) >= int(
                        state.get(
                            "reconcile_interval_seconds",
                            getattr(self._cfg, "code_sync_reconcile_interval_seconds", 300),
                        )
                    )
                ):
                    self.reconcile(root_id=root_id, ctx=ctx)
                    processed += 1
                elif bool(state.get("dirty")) and self._debounce_elapsed(state, now):
                    self._sync_root(
                        root_id,
                        ctx,
                        trigger="scheduled",
                        wait_mode="accepted",
                    )
                    processed += 1
            except Exception:
                logger.warning("Code workspace sync scheduler tick failed root_id=%s", root_id)

    def _resolve_registration_root(self, path: str) -> tuple[Path, bool]:
        candidate = Path(path).expanduser()
        try:
            root = candidate.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return candidate.resolve(strict=False), False
        if not root.is_dir():
            return root, False
        return root, True

    def _resolve_registration_binding(
        self,
        root: Path,
        *,
        server_accessible: bool,
        ctx: RequestContext,
        name: str | None,
        root_id: str,
        expected_fingerprint: str,
    ) -> tuple[str, str]:
        existing = self._store.read_code_workspace_sync_state(ctx.account_id, root_id)
        requested_name = name
        if existing is not None:
            existing_fingerprint = existing.get("root_fingerprint")
            existing_slug = existing.get("resource_root_slug")
            if existing_fingerprint != expected_fingerprint:
                raise ResourceIdentityError("workspace registry root identity collision")
            if not isinstance(existing_slug, str) or not existing_slug:
                raise ResourceIdentityError("workspace registry root binding is invalid")
            if name is not None and normalize_resource_root_slug(name) != existing_slug:
                raise ResourceIdentityError(
                    "workspace root fingerprint is already registered with a different root slug"
                )
            requested_name = existing_slug
        else:
            self._check_directory_conflict(root)
        if server_accessible:
            binding = self._lifecycle.resolve_workspace_binding(
                workspace_root=root,
                ctx=ctx,
                name=requested_name,
            )
            if (
                not isinstance(binding, tuple)
                or len(binding) != 2
                or not all(isinstance(value, str) and value for value in binding)
            ):
                raise RuntimeError("code lifecycle returned an invalid workspace binding")
            root_slug, fingerprint = binding
        else:
            root_slug = normalize_resource_root_slug(requested_name or root.name or "workspace")
            fingerprint = compute_root_fingerprint(root)
        if fingerprint != expected_fingerprint:
            raise ResourceIdentityError("workspace registry fingerprint changed during registration")
        return root_slug, fingerprint

    def _reserve_root_binding(
        self,
        *,
        ctx: RequestContext,
        root_id: str,
        root_slug: str,
        fingerprint: str,
    ) -> None:
        """Atomically preserve the account-wide root slug/fingerprint bijection."""

        def transition(current: dict[str, Any] | None) -> dict[str, Any]:
            state = dict(current or {})
            by_fingerprint = dict(state.get("by_fingerprint") or {})
            by_slug = dict(state.get("by_slug") or {})
            existing_by_fingerprint = by_fingerprint.get(fingerprint)
            existing_by_slug = by_slug.get(root_slug)
            expected = {"root_id": root_id, "resource_root_slug": root_slug}
            if existing_by_fingerprint is not None and existing_by_fingerprint != expected:
                raise ResourceIdentityError(
                    "workspace root fingerprint is already registered with a different root slug"
                )
            expected_slug = {"root_id": root_id, "root_fingerprint": fingerprint}
            if existing_by_slug is not None and existing_by_slug != expected_slug:
                raise ResourceIdentityError(
                    "workspace root slug is already registered with a different fingerprint"
                )
            by_fingerprint[fingerprint] = expected
            by_slug[root_slug] = expected_slug
            return {
                "version": 1,
                "by_fingerprint": by_fingerprint,
                "by_slug": by_slug,
            }

        self._store.mutate_code_workspace_sync_state(
            ctx.account_id,
            CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,
            transition,
        )

    def _validate_trigger_settings(
        self,
        trigger_mode: str | None,
        watch_enabled: bool | None,
        server_accessible: bool,
    ) -> tuple[str, bool]:
        mode = str(
            trigger_mode
            if trigger_mode is not None
            else getattr(self._cfg, "code_sync_default_trigger_mode", "hybrid")
        ).strip().lower()
        if mode not in _TRIGGER_MODES:
            raise ValueError("workspace sync trigger_mode is invalid")
        if watch_enabled is not None and not isinstance(watch_enabled, bool):
            raise ValueError("workspace sync watch_enabled must be a boolean")
        requested_watch = mode == "hybrid" if watch_enabled is None else watch_enabled
        if requested_watch and mode != "hybrid":
            raise ValueError("workspace sync watcher requires hybrid trigger_mode")
        if not server_accessible:
            if watch_enabled is True:
                raise ValueError("workspace sync watcher requires a server-accessible workspace")
            return ("external_event" if mode == "hybrid" else mode), False
        return mode, bool(requested_watch)

    def _max_files_setting(self, value: int | None) -> int:
        cap = self._default_max_files()
        if value is None:
            return cap
        if isinstance(value, bool):
            raise ValueError("workspace sync max_files must be a positive integer")
        try:
            requested = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("workspace sync max_files must be a positive integer") from exc
        if requested < 1 or requested > cap:
            raise ValueError("workspace sync max_files exceeds configured cap")
        return requested

    def _default_max_files(self) -> int:
        value = int(getattr(self._cfg, "code_resources_max_files", DEFAULT_CODE_RESOURCE_MAX_FILES))
        if value < 1:
            raise ValueError("code_resources_max_files must be positive")
        return value

    @staticmethod
    def _positive_setting(value: int | None, label: str, default: int) -> int:
        candidate = default if value is None else value
        if isinstance(candidate, bool):
            raise ValueError(f"workspace sync {label} must be positive")
        try:
            normalized = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"workspace sync {label} must be positive") from exc
        if normalized < 1:
            raise ValueError(f"workspace sync {label} must be positive")
        return normalized

    @staticmethod
    def _root_id(account_id: str, fingerprint: str) -> str:
        payload = "\0".join((account_id, fingerprint)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]

    def _wait_mode(self, value: object) -> str:
        mode = str(value or getattr(self._cfg, "code_sync_default_wait_mode", "accepted")).strip().lower()
        if mode not in _WAIT_MODES:
            raise ValueError("workspace sync wait_mode is invalid")
        return mode

    def _require_enabled(self) -> None:
        if not bool(getattr(self._cfg, "code_sync_enabled", False)):
            raise ValueError("code workspace sync is disabled")

    def _require_state(self, root_id: str, ctx: RequestContext) -> dict[str, Any]:
        if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
            raise FileNotFoundError("code workspace registry entry was not found")
        state = self._store.read_code_workspace_sync_state(ctx.account_id, root_id)
        if state is None:
            raise FileNotFoundError("code workspace registry entry was not found")
        return self._state_mapping(state)

    @staticmethod
    def _state_mapping(value: object) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError("workspace sync store returned an invalid state")
        return dict(value)

    @staticmethod
    def _require_current_state(current: dict[str, Any] | None) -> dict[str, Any]:
        if current is None:
            raise FileNotFoundError("code workspace registry entry was not found")
        return dict(current)

    def _remember_context(self, ctx: RequestContext, root_id: str) -> None:
        with self._lock:
            self._known_contexts[(ctx.account_id, root_id)] = ctx

    def _known_context_items(self) -> list[tuple[tuple[str, str], RequestContext]]:
        with self._lock:
            return sorted(self._known_contexts.items())

    def _recover_registered_contexts(self) -> None:
        """Rebuild scheduler contexts from protected durable registrations."""

        if self._context_resolver is None:
            return
        list_registrations = getattr(
            self._store,
            "list_code_workspace_sync_registrations",
            None,
        )
        if not callable(list_registrations):
            return
        try:
            registrations = list_registrations()
        except Exception:
            logger.warning("Code workspace sync registry recovery failed")
            return
        for account_id, root_id, _state in registrations:
            if root_id == CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID:
                continue
            try:
                ctx = self._context_resolver(account_id)
            except Exception:
                logger.warning("Code workspace sync context recovery failed account_id=%s", account_id)
                continue
            if not isinstance(ctx, RequestContext) or ctx.account_id != account_id:
                logger.warning("Code workspace sync context recovery rejected account_id=%s", account_id)
                continue
            self._remember_context(ctx, root_id)

    @staticmethod
    def _queue_sync_enabled(state: Mapping[str, Any]) -> bool:
        """Return whether the server may drain accepted dirty work for this root."""

        return bool(state.get("server_accessible")) and state.get("trigger_mode") in {
            "hybrid",
            "external_event",
        }

    def _debounce_elapsed(self, state: Mapping[str, Any], now: float) -> bool:
        dirty_at = self._float_or_none(state.get("last_dirty_at"))
        if dirty_at is None:
            return True
        debounce_seconds = int(getattr(self._cfg, "code_sync_debounce_seconds", 2))
        return now - dirty_at >= debounce_seconds

    def _max_pending_roots(self) -> int:
        return max(1, int(getattr(self._cfg, "code_sync_max_pending_roots", 100)))

    def _after_lifecycle_mutation(self, account_id: str) -> None:
        if self._invalidate_cache is not None:
            try:
                self._invalidate_cache(account_id)
            except Exception:
                logger.warning("Code workspace sync cache invalidation failed")
        if self._wake_outbox is not None:
            try:
                self._wake_outbox()
            except Exception:
                logger.warning("Code workspace sync Outbox wake-up failed")

    def _probe_index_visibility(self, ctx: RequestContext) -> bool:
        if self._is_index_visible is None:
            return False
        try:
            return bool(self._is_index_visible(ctx))
        except Exception:
            return False

    def _render_state(
        self,
        state: Mapping[str, Any],
        *,
        accepted: bool,
        filesystem_complete: bool,
    ) -> dict[str, Any]:
        dirty = bool(state.get("dirty"))
        syncing = bool(state.get("syncing"))
        index_visible = bool(state.get("index_visible")) and not dirty and not syncing
        rendered = {
            "root_id": state.get("root_id"),
            "resource_root_slug": state.get("resource_root_slug"),
            "trigger_mode": state.get("trigger_mode"),
            "watch_enabled": bool(state.get("watch_enabled")),
            "server_accessible": bool(state.get("server_accessible")),
            "accepted": accepted,
            "filesystem_complete": filesystem_complete,
            "index_visible": index_visible,
            "syncing": syncing,
            "dirty": dirty,
            "last_attempt_at": state.get("last_attempt_at"),
            "last_success_at": state.get("last_success_at"),
            "last_index_visible_at": state.get("last_index_visible_at"),
            "last_error_kind": state.get("last_error_kind"),
            "changed_files": int(state.get("last_changed_files", 0)),
            "deleted_files": int(state.get("last_deleted_files", 0)),
            "failed_files": int(state.get("last_failed_files", 0)),
        }
        rendered["diagnostics"] = self._diagnostic_projection(rendered)
        return rendered

    @staticmethod
    def _diagnostic_projection(rendered: Mapping[str, Any]) -> dict[str, Any]:
        """Project rendered workspace state without reading protected registry data."""

        def count(value: object) -> int:
            if isinstance(value, bool):
                return 0
            if isinstance(value, int):
                return max(0, value)
            return 0

        def timestamp(value: object) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        raw_error_kind = rendered.get("last_error_kind")
        error_kind = (
            raw_error_kind
            if isinstance(raw_error_kind, str) and raw_error_kind in _SYNC_DIAGNOSTIC_ERROR_KINDS
            else ("unknown" if raw_error_kind else None)
        )
        if bool(rendered.get("syncing")):
            state = "syncing"
        elif error_kind:
            state = "partial_failure" if error_kind in _SYNC_PARTIAL_FAILURE_KINDS else "error"
        elif bool(rendered.get("index_visible")):
            state = "index_visible"
        elif bool(rendered.get("filesystem_complete")):
            state = "filesystem_complete"
        elif bool(rendered.get("accepted")):
            state = "accepted"
        else:
            state = "stale"
        return {
            "state": state,
            "freshness": {
                "accepted": bool(rendered.get("accepted")),
                "filesystem_complete": bool(rendered.get("filesystem_complete")),
                "index_visible": bool(rendered.get("index_visible")),
                "dirty": bool(rendered.get("dirty")),
                "syncing": bool(rendered.get("syncing")),
                "last_attempt_at": timestamp(rendered.get("last_attempt_at")),
                "last_success_at": timestamp(rendered.get("last_success_at")),
                "last_index_visible_at": timestamp(rendered.get("last_index_visible_at")),
            },
            "work": {
                "changed_files": count(rendered.get("changed_files")),
                "deleted_files": count(rendered.get("deleted_files")),
                "failed_files": count(rendered.get("failed_files")),
            },
            "error_kind": error_kind,
        }

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _with_timestamp(
        current: dict[str, Any] | None,
        key: str,
        value: float,
    ) -> dict[str, Any]:
        state = CodeWorkspaceSyncService._require_current_state(current)
        state[key] = value
        state["updated_at"] = value
        return state

    @staticmethod
    def _mark_index_visible(
        current: dict[str, Any] | None,
        now: float,
    ) -> dict[str, Any]:
        state = CodeWorkspaceSyncService._require_current_state(current)
        if not bool(state.get("dirty")) and not bool(state.get("syncing")):
            state["index_visible"] = True
            state["last_index_visible_at"] = now
            state["updated_at"] = now
        return state


__all__ = [
    "CodeWorkspaceRegistryResolver",
    "CodeWorkspaceSyncService",
    "ResolvedCodeWorkspace",
    "WorkspaceResolutionError",
]
