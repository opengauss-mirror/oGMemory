"""Unit tests for durable code-workspace sync orchestration.

The tests use ``FakeControlStore`` in place of the control-plane backends and
``FakeLifecycle`` in place of ContextFS/Outbox-backed lifecycle writes.  They
prove local state transitions and delegation only; they are not backend
integration or end-to-end evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import service.code_workspace_sync as code_workspace_sync_module
from core.models import RequestContext
from core.resource_identity import ResourceIdentityError, compute_root_fingerprint
from server.control_plane_store import CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID
from service.code_workspace_sync import (
    CodeWorkspaceRegistryResolver,
    CodeWorkspaceSyncService,
    WorkspaceResolutionError,
)


class FakeControlStore:
    """In-memory copy-on-transition control-plane double shared by workers."""

    def __init__(self) -> None:
        self.states: dict[tuple[str, str], dict[str, Any]] = {}
        self.registration_lock = threading.Lock()

    def code_workspace_registration_lock(self):
        return self.registration_lock

    def read_code_workspace_sync_state(self, account_id: str, root_id: str):
        state = self.states.get((account_id, root_id))
        return deepcopy(state) if state is not None else None

    def mutate_code_workspace_sync_state(self, account_id: str, root_id: str, mutator):
        current = self.states.get((account_id, root_id))
        updated = mutator(deepcopy(current) if current is not None else None)
        self.states[(account_id, root_id)] = deepcopy(dict(updated))
        return deepcopy(dict(updated))

    def list_code_workspace_sync_states(self, account_id: str):
        return [
            (root_id, deepcopy(state))
            for (stored_account_id, root_id), state in sorted(self.states.items())
            if stored_account_id == account_id
        ]

    def list_code_workspace_sync_registrations(self):
        return [
            (account_id, root_id, deepcopy(state))
            for (account_id, root_id), state in sorted(self.states.items())
        ]


class FakeLifecycle:
    """Lifecycle double that records canonical refresh/delete delegation."""

    def __init__(self) -> None:
        self.refresh_attempts: list[str] = []
        self.refreshes: list[str] = []
        self.refresh_hints: list[str | None] = []
        self.deletes: list[str] = []
        self.fail_refreshes: dict[str, int] = {}
        self.on_refresh = None

    def resolve_workspace_binding(self, *, workspace_root, ctx, name=None):
        root = Path(workspace_root).resolve()
        return name or root.name, compute_root_fingerprint(root)

    def refresh(self, *, path, workspace_root, ctx, name=None, language_hint=None):
        self.refresh_attempts.append(path)
        remaining = self.fail_refreshes.get(path, 0)
        if remaining:
            self.fail_refreshes[path] = remaining - 1
            raise RuntimeError(f"refresh failed for {path}")
        self.refreshes.append(path)
        self.refresh_hints.append(language_hint)
        if self.on_refresh is not None:
            self.on_refresh(path)
        return {"status": "PENDING"}

    def delete_path(self, *, path, workspace_root, ctx, name=None):
        self.deletes.append(path)
        return {"status": "PENDING"}


class SyncClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.monotonic_now = 0.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.monotonic_now


def _ctx(account_id: str = "acct") -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _config(**overrides):
    values = {
        "code_sync_enabled": True,
        "code_resources_enabled": True,
        "code_resources_max_files": 10,
        "code_workspace_allowed_roots": ["/tmp"],
        "code_sync_default_trigger_mode": "hybrid",
        "code_sync_default_wait_mode": "accepted",
        "code_sync_reconcile_interval_seconds": 300,
        "code_sync_watch_interval_seconds": 5,
        "code_sync_run_timeout_seconds": 60,
        "code_sync_failure_backoff_seconds": 30,
        "code_sync_max_pending_roots": 100,
        "code_sync_debounce_seconds": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _service(
    store: FakeControlStore,
    lifecycle: FakeLifecycle,
    *,
    config=None,
    clock: SyncClock | None = None,
    worker_id: str = "worker-a",
    invalidate_cache=None,
    wake_outbox=None,
    is_index_visible=None,
    context_resolver=None,
    monotonic_clock=None,
) -> CodeWorkspaceSyncService:
    clock = clock or SyncClock()
    service = CodeWorkspaceSyncService(
        control_store=store,
        lifecycle_service=lifecycle,
        config=config or _config(),
        invalidate_cache=invalidate_cache,
        wake_outbox=wake_outbox,
        is_index_visible=is_index_visible,
        context_resolver=context_resolver,
        worker_id=worker_id,
        clock=clock.time,
        monotonic_clock=monotonic_clock or clock.monotonic,
    )
    # Registration starts the optional APScheduler. Unit tests drive ticks
    # explicitly so the fake state is deterministic.
    setattr(service, "start", lambda: None)
    return service


def _register(service: CodeWorkspaceSyncService, root: Path, **kwargs) -> str:
    name = kwargs.pop("name", "repo")
    result = service.register(path=str(root), ctx=_ctx(), name=name, **kwargs)
    root_id = result["root_id"]
    assert isinstance(root_id, str)
    return root_id


def _state(store: FakeControlStore, root_id: str) -> dict[str, Any]:
    state = store.read_code_workspace_sync_state("acct", root_id)
    assert state is not None
    return dict(state)


def test_registration_enforces_one_durable_identity_for_one_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    first_service = _service(store, lifecycle, worker_id="worker-a")
    root_id = _register(first_service, root)
    second_service = _service(store, lifecycle, worker_id="worker-b")

    def reject_global_scan():
        pytest.fail("existing registrations must not scan other tenants")

    monkeypatch.setattr(store, "list_code_workspace_sync_registrations", reject_global_scan)

    with pytest.raises(ResourceIdentityError, match="different root slug"):
        second_service.register(path=str(root), ctx=_ctx(), name="renamed-repo")

    recovered = second_service.register(path=str(root), ctx=_ctx())
    workspace_root_ids = {
        stored_root_id
        for account_id, stored_root_id in store.states
        if account_id == "acct" and stored_root_id != CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID
    }

    assert recovered["root_id"] == root_id
    assert workspace_root_ids == {root_id}

    store.states[("acct", root_id)]["root_fingerprint"] = "invalid"
    with pytest.raises(ResourceIdentityError, match="identity collision"):
        second_service.register(path=str(root), ctx=_ctx())


def test_registration_rejects_a_slug_bound_to_another_workspace(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)

    service.register(path=str(first_root), ctx=_ctx(), name="shared-root")

    with pytest.raises(ResourceIdentityError, match="root slug"):
        service.register(path=str(second_root), ctx=_ctx(), name="shared-root")


@pytest.mark.parametrize("account", ["acct", "other-tenant"])
@pytest.mark.parametrize("relationship", ["same", "parent", "child", "symlink", "dotdot"])
def test_registration_rejects_overlapping_directories_without_side_effects(tmp_path, account, relationship):
    from unittest.mock import Mock

    root = tmp_path / "repo"
    (root / "child").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    candidate = {"same": root, "parent": tmp_path, "child": root / "child",
                 "symlink": alias, "dotdot": root / "child" / ".."}[relationship]
    store, lifecycle = FakeControlStore(), FakeLifecycle()
    service = _service(store, lifecycle)
    _register(service, root)
    before = deepcopy(store.states)
    lifecycle.resolve_workspace_binding = Mock(wraps=lifecycle.resolve_workspace_binding)
    if account == "acct" and relationship in {"same", "symlink", "dotdot"}:
        service.register(path=str(candidate), ctx=_ctx(account), name="repo")
        assert len(store.states) == len(before)
    else:
        with pytest.raises(FileExistsError) as exc:
            service.register(path=str(candidate), ctx=_ctx(account), name="another")
        assert str(exc.value) == "workspace directory conflicts with an existing registration"
        assert store.states == before
        lifecycle.resolve_workspace_binding.assert_not_called()


def test_registration_allows_siblings_and_preserves_existing_legacy_update(tmp_path):
    store, lifecycle = FakeControlStore(), FakeLifecycle()
    service = _service(store, lifecycle)
    root = tmp_path / "repo"
    root.mkdir()
    root_id = _register(service, root)
    sibling = tmp_path / "repo-other"
    sibling.mkdir()
    service.register(path=str(sibling), ctx=_ctx("other"), name="repo")
    # Existing conflicting rows are not silently deleted or claimed by a new root.
    store.states[("legacy", "legacy-root")] = {"workspace_path": str(root / "child")}
    assert service.register(path=str(root), ctx=_ctx())["root_id"] == root_id
    with pytest.raises(FileExistsError):
        service.register(path=str(root / "child" / "nested"), ctx=_ctx("new"))


@pytest.mark.parametrize("unavailable", ["removed", "symlink_loop"])
def test_unavailable_registered_directory_does_not_block_unrelated_tenant(tmp_path, unavailable):
    store = FakeControlStore()
    service = _service(store, FakeLifecycle())
    root, sibling = tmp_path / "private", tmp_path / "sibling"
    root.mkdir()
    sibling.mkdir()
    root_id = _register(service, root)
    before = _state(store, root_id)
    root.rmdir()
    if unavailable == "symlink_loop":
        root.symlink_to(root, target_is_directory=True)

    with pytest.raises(FileExistsError) as exc:
        service.register(path=str(tmp_path), ctx=_ctx("other"), name="parent")
    assert str(exc.value) == "workspace directory conflicts with an existing registration"
    result = service.register(path=str(sibling), ctx=_ctx("other"), name="sibling")
    assert result["accepted"] is True
    assert str(root) not in json.dumps(result)
    assert _state(store, root_id) == before


def test_concurrent_directory_claims_have_one_winner(tmp_path):
    from server.control_plane_store import ControlPlaneStore

    root = tmp_path / "repo"
    root.mkdir()
    barrier = threading.Barrier(2)

    def register(account):
        store = ControlPlaneStore("", local_root=str(tmp_path / "state"))
        service = _service(store, FakeLifecycle())
        barrier.wait(timeout=5)
        try:
            service.register(path=str(root), ctx=_ctx(account), name="repo")
            return "accepted"
        except FileExistsError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(register, ["a", "b"])) == ["accepted", "conflict"]


@pytest.mark.parametrize("invalid", [
    b"{", b"{}", b'{"workspace_path":null}', b'{"workspace_path":""}', b'{"workspace_path":"  "}',
    b'{"workspace_path":"private/relative"}', b'{"workspace_path":"/private/../root"}',
    br'{"workspace_path":"/private/\u0000"}',
])
def test_unreadable_registry_does_not_allow_another_tenant_to_claim_directory(tmp_path, invalid):
    from server.control_plane_store import ControlPlaneStore

    store = ControlPlaneStore("", local_root=str(tmp_path / "state"))
    service = _service(store, FakeLifecycle())
    root = tmp_path / "repo"
    root.mkdir()
    root_id = _register(service, root)
    path = Path(store.code_workspace_sync_path("acct", root_id))
    original = path.read_bytes()
    path.write_bytes(invalid)
    with pytest.raises(RuntimeError, match="^workspace registry record is unreadable$"):
        service.register(path=str(root), ctx=_ctx("other"), name="repo")
    assert store.list_code_workspace_sync_states("other") == []
    assert path.read_bytes() == invalid
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        service.register(path=str(root), ctx=_ctx("other"), name="repo")


@pytest.mark.parametrize("register_child", [False, True])
def test_registration_accepts_configured_root_or_its_child(tmp_path, register_child):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    workspace = allowed_root / "repo" if register_child else allowed_root
    if register_child:
        workspace.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(
        store,
        lifecycle,
        config=_config(code_workspace_allowed_roots=[str(allowed_root)]),
    )

    root_id = _register(service, workspace)

    assert _state(store, root_id)["workspace_path"] == str(workspace.resolve())


def test_registration_rejects_roots_outside_the_configured_canonical_boundary(tmp_path):
    allowed_root = tmp_path / "allowed"
    sibling_prefix = tmp_path / "allowed-sibling"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    sibling_prefix.mkdir()
    outside_root.mkdir()
    service = _service(
        FakeControlStore(),
        FakeLifecycle(),
        config=_config(code_workspace_allowed_roots=[str(allowed_root)]),
    )

    for candidate in (sibling_prefix, outside_root, Path("/")):
        with pytest.raises(ValueError, match="outside configured allowed roots"):
            service.register(path=str(candidate), ctx=_ctx(), name="repo")


def test_registration_rejects_empty_or_invalid_allowed_root_policy(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    empty_policy_service = _service(
        FakeControlStore(),
        FakeLifecycle(),
        config=_config(code_workspace_allowed_roots=[]),
    )

    with pytest.raises(ValueError, match="must contain at least one directory"):
        empty_policy_service.register(path=str(workspace), ctx=_ctx(), name="repo")

    with pytest.raises(ValueError, match="must not include filesystem root"):
        _service(
            FakeControlStore(),
            FakeLifecycle(),
            config=_config(code_workspace_allowed_roots=["/"]),
        )

    for invalid_roots in (["relative"], [str(tmp_path / "missing")]):
        with pytest.raises(ValueError, match="must contain existing absolute directories"):
            _service(
                FakeControlStore(),
                FakeLifecycle(),
                config=_config(code_workspace_allowed_roots=invalid_roots),
            )


def test_registration_rejects_a_symlink_that_escapes_the_configured_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    escaped_workspace = allowed_root / "escaped"
    escaped_workspace.symlink_to(outside_root, target_is_directory=True)
    service = _service(
        FakeControlStore(),
        FakeLifecycle(),
        config=_config(code_workspace_allowed_roots=[str(allowed_root)]),
    )

    with pytest.raises(ValueError, match="outside configured allowed roots"):
        service.register(path=str(escaped_workspace), ctx=_ctx(), name="repo")


def test_registered_root_revalidation_rejects_stale_outside_policy_row(tmp_path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    config = _config(code_workspace_allowed_roots=[str(allowed_root)])
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle, config=config)
    root_id = _register(service, allowed_root, trigger_mode="manual", watch_enabled=False)
    outside_fingerprint = compute_root_fingerprint(outside_root)

    store.mutate_code_workspace_sync_state(
        "acct",
        root_id,
        lambda current: {
            **current,
            "workspace_path": str(outside_root),
            "root_fingerprint": outside_fingerprint,
        },
    )
    store.mutate_code_workspace_sync_state(
        "acct",
        CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,
        lambda current: {
            **current,
            "by_fingerprint": {
                outside_fingerprint: {
                    "root_id": root_id,
                    "resource_root_slug": "repo",
                }
            },
            "by_slug": {
                "repo": {
                    "root_id": root_id,
                    "root_fingerprint": outside_fingerprint,
                }
            },
        },
    )
    resolver = CodeWorkspaceRegistryResolver(store, config=config)

    with pytest.raises(WorkspaceResolutionError) as resolved:
        resolver.resolve_registered(ctx=_ctx(), root_slugs=("repo",))
    assert resolved.value.reason == "workspace_invalid"

    blocked_sync = service.sync(root_id=root_id, ctx=_ctx())
    assert blocked_sync["last_error_kind"] == "workspace_invalid"
    assert lifecycle.refreshes == []


def test_candidate_registry_resolver_selects_only_one_strict_registered_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    sync_service = _service(store, lifecycle)
    root_id = _register(sync_service, root, trigger_mode="manual", watch_enabled=False)
    resolver = CodeWorkspaceRegistryResolver(store)

    # Query-time selection may read protected state but must not use lifecycle
    # or mutate registry state after registration.
    store.mutate_code_workspace_sync_state = lambda *args, **kwargs: pytest.fail(
        "candidate registry resolver must be read-only"
    )

    explicit = resolver.resolve(ctx=_ctx(), root_slugs=("repo",))
    implicit = resolver.resolve(ctx=_ctx(), root_slugs=())

    assert explicit.root_id == root_id
    assert explicit.root_slug == "repo"
    assert explicit.physical_root == root.resolve()
    assert explicit.available_root_count == 1
    assert implicit.root_id == root_id
    assert implicit.available_root_count == 1
    assert lifecycle.refreshes == []
    assert lifecycle.deletes == []


def test_candidate_registry_resolver_requires_manual_watcher_disabled_registration(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    sync_service = _service(store, lifecycle)
    root_id = _register(
        sync_service,
        root,
        trigger_mode="hybrid",
        watch_enabled=False,
    )
    resolver = CodeWorkspaceRegistryResolver(store)

    with pytest.raises(WorkspaceResolutionError) as explicit:
        resolver.resolve(ctx=_ctx(), root_slugs=("repo",))
    assert explicit.value.reason == "workspace_not_registered"

    with pytest.raises(WorkspaceResolutionError) as implicit:
        resolver.resolve(ctx=_ctx(), root_slugs=())
    assert implicit.value.reason == "workspace_not_registered"

    store.mutate_code_workspace_sync_state(
        "acct",
        root_id,
        lambda current: {**current, "trigger_mode": "manual", "watch_enabled": True},
    )
    with pytest.raises(WorkspaceResolutionError) as malformed_manual:
        resolver.resolve(ctx=_ctx(), root_slugs=("repo",))
    assert malformed_manual.value.reason == "workspace_invalid"


def test_registered_workspace_resolver_reuses_strict_checks_without_manual_only_filter(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    sync_service = _service(store, lifecycle)
    root_id = _register(
        sync_service,
        root,
        trigger_mode="hybrid",
        watch_enabled=False,
    )
    resolver = CodeWorkspaceRegistryResolver(store)

    workspace = resolver.resolve_registered(ctx=_ctx(), root_slugs=("repo",))

    assert workspace.root_id == root_id
    assert workspace.root_slug == "repo"
    assert workspace.physical_root == root.resolve()
    with pytest.raises(WorkspaceResolutionError) as candidate_only:
        resolver.resolve(ctx=_ctx(), root_slugs=("repo",))
    assert candidate_only.value.reason == "workspace_not_registered"


def test_candidate_registry_resolver_freezes_scope_and_workspace_failure_reasons(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    sync_service = _service(store, lifecycle)
    first_id = _register(
        sync_service,
        first,
        name="first",
        trigger_mode="manual",
        watch_enabled=False,
    )
    _register(
        sync_service,
        second,
        name="second",
        trigger_mode="manual",
        watch_enabled=False,
    )
    resolver = CodeWorkspaceRegistryResolver(store)

    with pytest.raises(WorkspaceResolutionError) as ambiguous:
        resolver.resolve(ctx=_ctx(), root_slugs=())
    assert ambiguous.value.reason == "workspace_scope_ambiguous"
    assert ambiguous.value.available_root_count == 2
    assert str(ambiguous.value) == "workspace_scope_ambiguous"

    with pytest.raises(WorkspaceResolutionError) as explicit_many:
        resolver.resolve(ctx=_ctx(), root_slugs=("first", "second"))
    assert explicit_many.value.reason == "workspace_scope_ambiguous"
    assert explicit_many.value.available_root_count == 0

    with pytest.raises(WorkspaceResolutionError) as unknown:
        resolver.resolve(ctx=_ctx(), root_slugs=("missing",))
    assert unknown.value.reason == "workspace_not_registered"
    assert unknown.value.available_root_count == 0

    with pytest.raises(WorkspaceResolutionError) as foreign:
        resolver.resolve(ctx=_ctx("other"), root_slugs=("first",))
    assert foreign.value.reason == "workspace_not_registered"

    first_state = _state(store, first_id)
    store.mutate_code_workspace_sync_state(
        "acct",
        first_id,
        lambda current: {**current, "root_fingerprint": "0" * 64},
    )
    with pytest.raises(WorkspaceResolutionError) as invalid:
        resolver.resolve(ctx=_ctx(), root_slugs=("first",))
    assert invalid.value.reason == "workspace_invalid"

    # Restore a correct registration, then remove its protected root.  The
    # outcome stays path-free and must not guess another account root.
    store.mutate_code_workspace_sync_state(
        "acct",
        first_id,
        lambda _current: dict(first_state),
    )
    first.rmdir()
    with pytest.raises(WorkspaceResolutionError) as unavailable:
        resolver.resolve(ctx=_ctx(), root_slugs=("first",))
    assert unavailable.value.reason == "workspace_unavailable"


def test_dirty_idle_state_is_not_rendered_as_filesystem_complete(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)

    state = service.get_state(root_id=root_id, ctx=_ctx())
    listed = service.list_states(ctx=_ctx())
    store.mutate_code_workspace_sync_state(
        "acct",
        root_id,
        lambda current: {
            **current,
            "server_accessible": False,
            "watch_enabled": False,
            "trigger_mode": "external_event",
        },
    )
    polled = service.poll(root_id=root_id, ctx=_ctx())
    reconciled = service.reconcile(root_id=root_id, ctx=_ctx())

    assert state["dirty"] is True
    assert state["filesystem_complete"] is False
    assert state["diagnostics"]["state"] == "stale"
    assert state["diagnostics"]["freshness"]["dirty"] is True
    assert state["diagnostics"]["freshness"]["index_visible"] is False
    assert listed == [state]
    assert polled["filesystem_complete"] is False
    assert reconciled["filesystem_complete"] is False


def test_sync_converges_new_changed_deleted_and_renamed_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "a.py"
    source.write_text("value = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)

    assert service.sync(root_id=root_id, ctx=_ctx())["filesystem_complete"] is True
    assert lifecycle.refreshes == ["a.py"]

    source.write_text("value = 100\n", encoding="utf-8")
    service.sync(root_id=root_id, ctx=_ctx())
    assert lifecycle.refreshes == ["a.py", "a.py"]

    source.unlink()
    (root / "b.py").write_text("value = 2\n", encoding="utf-8")
    service.sync(root_id=root_id, ctx=_ctx())
    assert lifecycle.deletes == ["a.py"]
    assert lifecycle.refreshes[-1] == "b.py"

    (root / "b.py").rename(root / "c.py")
    service.sync(root_id=root_id, ctx=_ctx())
    assert lifecycle.deletes == ["a.py", "b.py"]
    assert lifecycle.refreshes[-1] == "c.py"
    assert sorted(_state(store, root_id)["snapshot"]) == ["c.py"]


def test_sync_detects_same_size_content_change_with_preserved_mtime(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "same.py"
    source.write_text("value = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)

    service.sync(root_id=root_id, ctx=_ctx())
    original_state = dict(_state(store, root_id)["snapshot"]["same.py"])
    original_stat = source.stat()
    source.write_text("value = 2\n", encoding="utf-8")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    polled = service.poll(root_id=root_id, ctx=_ctx())

    assert polled["dirty"] is False
    assert lifecycle.refreshes == ["same.py"]

    result = service.reconcile(root_id=root_id, ctx=_ctx())
    updated_state = _state(store, root_id)["snapshot"]["same.py"]

    assert result["changed_files"] == 1
    assert lifecycle.refreshes == ["same.py", "same.py"]
    assert updated_state["size"] == original_state["size"]
    assert updated_state["mtime_ns"] == original_state["mtime_ns"]
    assert updated_state["sha256"] != original_state["sha256"]


def test_sync_deletes_previously_indexed_file_when_git_now_ignores_it(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    source = root / "still-present.py"
    source.write_text("value = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)
    service.sync(root_id=root_id, ctx=_ctx())

    (root / ".gitignore").write_text("still-present.py\n", encoding="utf-8")
    result = service.sync(root_id=root_id, ctx=_ctx())

    assert source.exists()
    assert lifecycle.deletes == ["still-present.py"]
    assert result["deleted_files"] == 1
    assert _state(store, root_id)["snapshot"] == {}


def test_partial_failure_keeps_only_unsucceeded_paths_dirty_for_retry(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_text("bad = True\n", encoding="utf-8")
    (root / "good.py").write_text("good = True\n", encoding="utf-8")
    clock = SyncClock()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    lifecycle.fail_refreshes["bad.py"] = 1
    service = _service(store, lifecycle, clock=clock)
    root_id = _register(service, root)

    first = service.sync(root_id=root_id, ctx=_ctx())
    failed_state = _state(store, root_id)

    assert first["filesystem_complete"] is False
    assert first["last_error_kind"] == "lifecycle_failure"
    assert first["diagnostics"]["state"] == "partial_failure"
    assert first["diagnostics"]["error_kind"] == "lifecycle_failure"
    assert first["diagnostics"]["work"] == {
        "changed_files": 1,
        "deleted_files": 0,
        "failed_files": 1,
    }
    assert failed_state["dirty"] is True
    assert failed_state["next_retry_at"] == clock.now + 30
    assert sorted(failed_state["snapshot"]) == ["good.py"]
    assert lifecycle.refresh_attempts == ["bad.py", "good.py"]

    second = service.sync(root_id=root_id, ctx=_ctx())

    assert second["filesystem_complete"] is True
    assert second["last_error_kind"] is None
    assert sorted(_state(store, root_id)["snapshot"]) == ["bad.py", "good.py"]
    assert lifecycle.refresh_attempts == ["bad.py", "good.py", "bad.py"]


def test_timeout_leaves_work_dirty_without_claiming_filesystem_completion(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    monotonic_values = iter((0.0, 2.0))
    service = _service(
        store,
        lifecycle,
        config=_config(code_sync_run_timeout_seconds=1),
        monotonic_clock=lambda: next(monotonic_values),
    )
    root_id = _register(service, root)

    result = service.sync(root_id=root_id, ctx=_ctx())
    state = _state(store, root_id)

    assert result["filesystem_complete"] is False
    assert result["last_error_kind"] == "timeout"
    assert result["failed_files"] == 2
    assert result["diagnostics"]["state"] == "partial_failure"
    assert result["diagnostics"]["error_kind"] == "timeout"
    assert result["diagnostics"]["work"]["failed_files"] == 2
    assert state["dirty"] is True
    assert lifecycle.refreshes == []


def test_expired_lease_is_reclaimed_but_active_lease_blocks_second_worker(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    clock = SyncClock()
    store = FakeControlStore()
    lifecycle_a = FakeLifecycle()
    lifecycle_b = FakeLifecycle()
    service_a = _service(store, lifecycle_a, clock=clock, worker_id="worker-a")
    root_id = _register(service_a, root)
    service_b = _service(store, lifecycle_b, clock=clock, worker_id="worker-b")

    store.mutate_code_workspace_sync_state(
        "acct",
        root_id,
        lambda state: {
            **state,
            "lease_owner": "worker-a",
            "lease_expires_at": clock.now + 10,
            "syncing": True,
        },
    )
    blocked = service_b.sync(root_id=root_id, ctx=_ctx())

    assert blocked["filesystem_complete"] is False
    assert lifecycle_b.refreshes == []

    clock.now += 11
    recovered = service_b.sync(root_id=root_id, ctx=_ctx())

    assert recovered["filesystem_complete"] is True
    assert lifecycle_b.refreshes == ["a.py"]
    assert _state(store, root_id)["lease_owner"] is None


def test_dirty_mark_during_run_gets_one_coalesced_rerun(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)
    sync_runs = 0
    original_sync_once = service._sync_once

    def count_sync_runs(*args, **kwargs):
        nonlocal sync_runs
        sync_runs += 1
        return original_sync_once(*args, **kwargs)

    service._sync_once = count_sync_runs
    lifecycle.on_refresh = lambda _path: service.mark_dirty(root_id=root_id, ctx=_ctx())

    result = service.sync(root_id=root_id, ctx=_ctx())

    assert result["filesystem_complete"] is True
    assert sync_runs == 2
    assert lifecycle.refreshes == ["a.py"]
    state = _state(store, root_id)
    assert state["dirty"] is False
    assert state["next_retry_at"] is None


def test_workspace_unavailable_stays_dirty_and_downgrades_to_external_event(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    clock = SyncClock()
    service = _service(store, lifecycle, clock=clock)
    root_id = _register(service, root)
    root.rmdir()

    result = service.sync(root_id=root_id, ctx=_ctx())
    state = _state(store, root_id)

    assert result["filesystem_complete"] is False
    assert result["last_error_kind"] == "workspace_unavailable"
    assert result["server_accessible"] is False
    assert result["trigger_mode"] == "external_event"
    assert state["dirty"] is True
    assert state["next_retry_at"] == clock.now + 30
    assert state.get("last_success_at") is None


def test_sync_refreshes_unchanged_misclassified_file_when_hint_changes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "template.txt").write_text("def render():\n    return 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root, language_hints={"template.txt": "python"})

    service.sync(root_id=root_id, ctx=_ctx())
    service.register(
        path=str(root),
        ctx=_ctx(),
        name="repo",
        language_hints={"template.txt": "javascript"},
    )
    result = service.sync(root_id=root_id, ctx=_ctx())

    assert lifecycle.refreshes == ["template.txt", "template.txt"]
    assert lifecycle.refresh_hints == ["python", "javascript"]
    assert result["changed_files"] == 1
    assert _state(store, root_id)["snapshot"]["template.txt"]["language_hint"] == "javascript"


def test_poll_uses_metadata_only_and_does_not_hash_unchanged_files(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)
    service.sync(root_id=root_id, ctx=_ctx())
    monkeypatch.setattr(
        code_workspace_sync_module,
        "_streaming_file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("poll must not hash file contents")
        ),
    )

    result = service.poll(root_id=root_id, ctx=_ctx())

    assert result["accepted"] is False
    assert result["dirty"] is False
    assert result["filesystem_complete"] is True
    assert _state(store, root_id)["last_poll_at"] is not None


def test_reconcile_drops_newly_oversized_file_without_hashing(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "large.py"
    source.write_text("value = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root)
    service.sync(root_id=root_id, ctx=_ctx())

    monkeypatch.setattr(code_workspace_sync_module, "MAX_CODE_FILE_SIZE", 4)
    monkeypatch.setattr(
        code_workspace_sync_module,
        "_streaming_file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("oversized files must not be hashed")
        ),
    )

    result = service.reconcile(root_id=root_id, ctx=_ctx())

    assert result["deleted_files"] == 1
    assert lifecycle.deletes == ["large.py"]
    assert _state(store, root_id)["snapshot"] == {}


def test_polling_coalesces_dirty_state_until_debounce_allows_sync(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    clock = SyncClock()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(
        store,
        lifecycle,
        clock=clock,
        config=_config(
            code_sync_watch_interval_seconds=5,
            code_sync_debounce_seconds=1,
            code_sync_reconcile_interval_seconds=300,
        ),
    )
    root_id = _register(service, root)
    store.mutate_code_workspace_sync_state(
        "acct",
        root_id,
        lambda state: {**state, "last_reconcile_at": clock.now},
    )

    service._scheduled_tick()
    first_state = _state(store, root_id)

    assert first_state["last_dirty_at"] == 100.0
    assert first_state["dirty"] is True
    assert lifecycle.refreshes == []

    clock.now += 5
    service._scheduled_tick()

    assert lifecycle.refreshes == ["a.py"]
    assert _state(store, root_id)["dirty"] is False


def test_restart_recovers_durable_registry_with_account_context_resolver(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    clock = SyncClock()
    store = FakeControlStore()
    initial_lifecycle = FakeLifecycle()
    initial_service = _service(store, initial_lifecycle, clock=clock)
    root_id = _register(initial_service, root)

    restarted_lifecycle = FakeLifecycle()
    restarted_service = _service(
        store,
        restarted_lifecycle,
        clock=clock,
        worker_id="worker-restarted",
        context_resolver=lambda account_id: _ctx(account_id),
    )
    clock.now += 2
    restarted_service._scheduled_tick()
    clock.now += 2
    restarted_service._scheduled_tick()

    assert restarted_lifecycle.refreshes == ["a.py"]
    assert ("acct", root_id) in restarted_service._known_contexts
    assert _state(store, root_id)["dirty"] is False


def test_rendered_state_hides_workspace_path_and_callbacks_only_follow_mutations(tmp_path):
    root = tmp_path / "private-repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    invalidated: list[str] = []
    wakes: list[bool] = []
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(
        store,
        lifecycle,
        invalidate_cache=invalidated.append,
        wake_outbox=lambda: wakes.append(True),
        is_index_visible=lambda _ctx: True,
    )
    registration = service.register(path=str(root), ctx=_ctx(), name="private-repo")
    root_id = registration["root_id"]

    first = service.sync(root_id=root_id, ctx=_ctx(), wait_mode="index_visible")
    second = service.sync(root_id=root_id, ctx=_ctx())
    rendered = service.get_state(root_id=root_id, ctx=_ctx())

    serialized = json.dumps(
        {"registration": registration, "first": first, "second": second, "state": rendered}
    )
    assert str(root) not in serialized
    assert "workspace_path" not in serialized
    assert registration["diagnostics"]["state"] == "accepted"
    assert first["diagnostics"]["state"] == "index_visible"
    assert first["diagnostics"]["freshness"]["filesystem_complete"] is True
    assert first["diagnostics"]["freshness"]["index_visible"] is True
    assert first["index_visible"] is True
    assert invalidated == ["acct"]
    assert wakes == [True]


def test_registration_enforces_file_cap_and_external_wait_modes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle, config=_config(code_resources_max_files=1))

    with pytest.raises(ValueError, match="exceeds configured cap"):
        _register(service, root, max_files=2)

    root_id = _register(service, root, max_files=1, wait_mode="accepted")
    accepted = service.external_event(root_id=root_id, ctx=_ctx())

    assert accepted["accepted"] is True
    assert accepted["filesystem_complete"] is False
    assert lifecycle.refreshes == []
    with pytest.raises(ValueError, match="wait_mode is invalid"):
        service.external_event(root_id=root_id, ctx=_ctx(), wait_mode="forever")


def test_external_event_root_drains_accepted_work_without_hybrid_reconciliation(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "service.py").write_text("value = 1\n", encoding="utf-8")
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    clock = SyncClock()
    service = _service(
        store,
        lifecycle,
        clock=clock,
        config=_config(code_sync_debounce_seconds=1),
    )
    root_id = _register(
        service,
        root,
        trigger_mode="external_event",
        wait_mode="accepted",
    )

    accepted = service.external_event(root_id=root_id, ctx=_ctx())
    assert accepted["accepted"] is True
    assert lifecycle.refreshes == []

    clock.now += 1
    results = service.run_pending(ctx=_ctx())

    assert len(results) == 1
    assert results[0]["filesystem_complete"] is True
    assert lifecycle.refreshes == ["service.py"]


def test_external_event_rejects_manual_trigger_mode(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = FakeControlStore()
    lifecycle = FakeLifecycle()
    service = _service(store, lifecycle)
    root_id = _register(service, root, trigger_mode="manual")

    with pytest.raises(ValueError, match="external events require"):
        service.external_event(root_id=root_id, ctx=_ctx())
