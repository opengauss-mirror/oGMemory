"""Unit coverage for code RESOURCE lifecycle orchestration and HTTP wiring.

The storage and Outbox objects in this file are test doubles.  These tests
prove local lifecycle ordering and transport delegation; they are not backend
integration or end-to-end evidence.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, RLock
from types import SimpleNamespace

import pytest

from tests.code_search_helpers import deterministic_code_tokenizer  # noqa: F401

from commit.code_resource_events import (
    build_code_resource_delete_event,
    build_code_resource_write_event,
)
from core.enums import EventType
from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from core.resource_identity import (
    RootBindingError,
    compute_file_key,
    compute_root_fingerprint,
    digest_slug,
    reserve_workspace_root_binding_state,
    resolve_workspace_root_binding_state,
    resource_owner_space,
)
from service.code_resource_lifecycle import (
    CodeResourceLifecycleService,
    build_code_resource_nodes,
)


def _ctx(account_id: str = "acct") -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _config(*, enabled: bool = True, max_files: int = 5000):
    return SimpleNamespace(
        code_resources_enabled=enabled,
        code_resources_target_chars=2500,
        code_resources_overlap_chars=300,
        code_resources_chunk_max_chars=4000,
        code_resources_max_files=max_files,
    )


class FakeContextFS:
    def __init__(self, nodes=()):
        self.nodes = {node.uri: node for node in nodes}
        self.written: list[str] = []
        self.deleted: list[str] = []

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        prefix = uri.rstrip("/") + "/"
        return sorted(
            node_uri
            for node_uri in self.nodes
            if node_uri.startswith(prefix) and "/" not in node_uri[len(prefix) :]
        )

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        try:
            return self.nodes[uri]
        except KeyError as exc:
            raise NodeNotFoundError(uri) from exc

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        return uri in self.nodes

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        self.nodes[node.uri] = node
        self.written.append(node.uri)

    def write_node_with_outbox(self, node: ContextNode, ctx: RequestContext, event) -> None:
        """Explicit atomic fake: publish node and event as one operation."""
        if getattr(event, "event_type", None) != EventType.UPSERT_CONTEXT.value:
            raise ValueError("expected UPSERT_CONTEXT event")
        self.nodes[node.uri] = node
        self.written.append(node.uri)

    def delete_nodes_with_outbox(self, uris, ctx: RequestContext, events) -> int:
        if len(events) != len(uris):
            raise ValueError("events must match URIs")
        deleted = 0
        for uri in uris:
            if uri in self.nodes:
                del self.nodes[uri]
                self.deleted.append(uri)
                deleted += 1
        return deleted

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        del self.nodes[uri]
        self.deleted.append(uri)


class FakeOutbox:
    def __init__(self, *, fail_write: bool = False):
        self.fail_write = fail_write
        self.writes: list[str] = []
        self.deletes: list[str] = []

    def register_write(self, node: ContextNode, ctx: RequestContext):
        if self.fail_write:
            raise RuntimeError("outbox write failed")
        self.writes.append(node.uri)

    def build_write_event(self, node: ContextNode):
        if self.fail_write:
            raise RuntimeError("outbox write failed")
        return build_code_resource_write_event(node)

    def build_delete_event(self, uri: str):
        return build_code_resource_delete_event(uri)

    def register_delete(self, uri: str, ctx: RequestContext):
        self.deletes.append(uri)


class AtomicBindingFakeContextFS(FakeContextFS):
    """Unit double for the backend's atomic binding transition only."""

    def __init__(self, shared, lock, ready=None):
        super().__init__()
        self._shared = shared
        self._binding_lock = lock
        self._ready = ready

    def read_code_workspace_root_binding(
        self,
        *,
        ctx,
        root_slug=None,
        root_fingerprint=None,
    ):
        with self._binding_lock:
            return resolve_workspace_root_binding_state(
                self._shared.get("state"),
                account_id=ctx.account_id,
                root_slug=root_slug,
                root_fingerprint=root_fingerprint,
            )

    def reserve_code_workspace_root_binding(
        self,
        *,
        ctx,
        root_slug,
        root_fingerprint,
    ):
        if self._ready is not None:
            self._ready.wait(timeout=2)
        with self._binding_lock:
            updated = reserve_workspace_root_binding_state(
                self._shared.get("state"),
                account_id=ctx.account_id,
                root_slug=root_slug,
                root_fingerprint=root_fingerprint,
            )
            self._shared["state"] = updated
            return resolve_workspace_root_binding_state(
                updated,
                account_id=ctx.account_id,
                root_fingerprint=root_fingerprint,
            )


def _existing_code_node(
    root,
    relative_path: str,
    *,
    root_slug: str = "repo",
    uri_suffix: str = "a" * 32,
) -> ContextNode:
    fingerprint = compute_root_fingerprint(root)
    file_key = compute_file_key(fingerprint, root_slug, relative_path)
    return ContextNode(
        uri=f"ctx://acct/resources/{uri_suffix}",
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space=resource_owner_space("acct"),
        abstract="old chunk",
        overview=f"path: {relative_path}",
        content="old content",
        metadata={
            "resource_type": "code",
            "resource_root_slug": root_slug,
            "root_fingerprint": fingerprint,
            "file_key": file_key,
            "file_slug": digest_slug(file_key),
            "relative_path": relative_path,
            "chunk_key": "old-" + uri_suffix,
            "version": 1,
        },
    )


def test_build_code_resource_nodes_preserves_identity_and_omits_absolute_path(tmp_path):
    root = tmp_path / "repo"
    source = root / "pkg" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("def load_config(path):\n    return path\n", encoding="utf-8")

    file_key, file_slug, nodes = build_code_resource_nodes(
        account_id="acct",
        workspace_root=root,
        file_path=source,
        resource_root_slug="repo",
    )

    assert len(file_key) == 64
    assert len(file_slug) == 32
    assert nodes
    for node in nodes:
        assert re.fullmatch(r"ctx://acct/resources/[0-9a-f]{32}", node.uri)
        assert node.context_type == "RESOURCE"
        assert node.category == "code"
        assert node.level == 4
        assert node.owner_space == "resource:acct"
        assert node.metadata["file_key"] == file_key
        assert node.metadata["relative_path"] == "pkg/service.py"
        assert len(node.metadata["chunk_key"]) == 64
        serialized = json.dumps(
            {
                "abstract": node.abstract,
                "overview": node.overview,
                "metadata": node.metadata,
            }
        )
        assert str(root) not in serialized


def test_build_code_resource_nodes_accepts_valid_hint_for_nonstandard_file(tmp_path):
    root = tmp_path / "repo"
    source = root / "Dockerfile"
    root.mkdir()
    source.write_text("FROM python:3.11\nRUN echo ready\n", encoding="utf-8")

    _, _, nodes = build_code_resource_nodes(
        account_id="acct",
        workspace_root=root,
        file_path=source,
        resource_root_slug="repo",
        language_hint="shell",
    )

    assert nodes
    assert {node.metadata["language"] for node in nodes} == {"shell"}
    assert {node.metadata["source_extension"] for node in nodes} == {""}


def test_refresh_rejects_unknown_language_hint_before_mutation(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "template.txt"
    source.write_text("def render():\n    return 1\n", encoding="utf-8")
    fs = FakeContextFS()
    outbox = FakeOutbox()
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(),
    )

    with pytest.raises(ValueError, match="unsupported code language hint"):
        service.refresh(
            path=str(source),
            workspace_root=str(root),
            name="repo",
            language_hint="unknown",
            ctx=_ctx(),
        )

    assert fs.nodes == {}
    assert outbox.writes == []


def test_bootstrap_rejects_max_files_above_configured_cap(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    fs = FakeContextFS()
    outbox = FakeOutbox()
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(max_files=1),
    )

    with pytest.raises(ValueError, match="max_files exceeds configured cap"):
        service.bootstrap(
            path=str(root),
            ctx=_ctx(),
            name="repo",
            max_files=2,
        )

    assert fs.nodes == {}
    assert outbox.writes == []


@pytest.mark.parametrize("collision", ["slug", "fingerprint"])
def test_concurrent_bootstrap_reserves_binding_before_chunk_or_outbox_write(
    tmp_path,
    collision,
):
    first_root = tmp_path / "first"
    second_root = first_root if collision == "fingerprint" else tmp_path / "second"
    first_root.mkdir()
    if second_root != first_root:
        second_root.mkdir()
    (first_root / "module.py").write_text("value = 1\n", encoding="utf-8")
    if second_root != first_root:
        (second_root / "module.py").write_text("value = 2\n", encoding="utf-8")

    shared = {"state": None}
    binding_lock = RLock()
    ready = Barrier(2)
    filesystems = [
        AtomicBindingFakeContextFS(shared, binding_lock, ready),
        AtomicBindingFakeContextFS(shared, binding_lock, ready),
    ]
    outboxes = [FakeOutbox(), FakeOutbox()]
    services = [
        CodeResourceLifecycleService(
            context_fs=filesystem,
            outbox_store=outbox,
            config=_config(),
        )
        for filesystem, outbox in zip(filesystems, outboxes, strict=True)
    ]
    names = ("first", "second") if collision == "fingerprint" else ("shared", "shared")
    roots = (first_root, second_root)

    def bootstrap(index):
        try:
            return services[index].bootstrap(
                path=str(roots[index]),
                name=names[index],
                ctx=_ctx(),
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(bootstrap, range(2)))

    successful_indexes = [
        index for index, result in enumerate(results) if isinstance(result, dict)
    ]
    failed_indexes = [
        index for index, result in enumerate(results) if isinstance(result, Exception)
    ]
    assert len(successful_indexes) == 1
    assert len(failed_indexes) == 1
    assert isinstance(results[failed_indexes[0]], RootBindingError)
    winner = successful_indexes[0]
    loser = failed_indexes[0]
    assert filesystems[winner].written
    assert filesystems[loser].written == []


def test_unnamed_resolution_reuses_reserved_custom_slug(tmp_path):
    root = tmp_path / "physical-root"
    root.mkdir()
    shared = {"state": None}
    filesystem = AtomicBindingFakeContextFS(shared, RLock())
    service = CodeResourceLifecycleService(
        context_fs=filesystem,
        outbox_store=FakeOutbox(),
        config=_config(),
    )
    fingerprint = compute_root_fingerprint(root)
    service.reserve_workspace_binding(
        root_slug="custom-name",
        root_fingerprint=fingerprint,
        ctx=_ctx(),
    )

    assert service.resolve_workspace_binding(
        workspace_root=root,
        name=None,
        ctx=_ctx(),
    ) == ("custom-name", fingerprint)


def test_code_node_versions_increase_monotonically_across_repeated_writes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    existing = _existing_code_node(root, "module.py")
    expected_versions = []

    class VersionedContextFS(FakeContextFS):
        def write_node_with_outbox(self, node, ctx, event):
            expected_version = node.metadata["_expected_version"]
            current = self.nodes.get(node.uri)
            actual_version = current.metadata["version"] if current else 0
            assert expected_version == actual_version
            expected_versions.append(expected_version)
            node.metadata["version"] = expected_version + 1
            super().write_node_with_outbox(node, ctx, event)

    fs = VersionedContextFS([existing])
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=FakeOutbox(),
        config=_config(),
    )

    def pending_update() -> ContextNode:
        return ContextNode(
            uri=existing.uri,
            context_type=existing.context_type,
            category=existing.category,
            level=existing.level,
            owner_space=existing.owner_space,
            abstract=existing.abstract,
            overview=existing.overview,
            content=existing.content,
            metadata={
                key: value
                for key, value in existing.metadata.items()
                if key != "version"
            },
        )

    created = _existing_code_node(
        root,
        "created.py",
        uri_suffix="b" * 32,
    )
    created.metadata.pop("version")

    assert service._write_nodes([pending_update()], _ctx()) == (0, 1)
    assert fs.nodes[existing.uri].metadata["version"] == 2
    assert service._write_nodes([pending_update()], _ctx()) == (0, 1)
    assert fs.nodes[existing.uri].metadata["version"] == 3
    assert service._write_nodes([created], _ctx()) == (1, 0)
    assert fs.nodes[created.uri].metadata["version"] == 1
    assert expected_versions == [1, 2, 0]


def test_delete_path_removes_existing_identity_without_reading_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "now-ignored.py"
    source.write_text("still present = True\n", encoding="utf-8")
    old = _existing_code_node(root, "now-ignored.py")
    fs = FakeContextFS([old])
    outbox = FakeOutbox()
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(),
    )

    result = service.delete_path(
        path="now-ignored.py",
        workspace_root=str(root),
        name="repo",
        ctx=_ctx(),
    )

    assert result["status"] == "PENDING"
    assert result["nodes_deleted"] == 1
    assert fs.nodes == {}
    assert fs.deleted == [old.uri]


def test_missing_refresh_is_pending_and_idempotent(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    old = _existing_code_node(root, "gone.py")
    fs = FakeContextFS([old])
    outbox = FakeOutbox()
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(),
    )

    first = service.refresh(
        path=str(root / "gone.py"),
        workspace_root=str(root),
        name="repo",
        ctx=_ctx(),
    )
    second = service.refresh(
        path=str(root / "gone.py"),
        workspace_root=str(root),
        name="repo",
        ctx=_ctx(),
    )

    assert first["status"] == second["status"] == "PENDING"
    assert first["nodes_deleted"] == 1
    assert second["nodes_deleted"] == 0
    assert fs.deleted == [old.uri]


def test_refresh_updates_source_snapshot_when_only_unindexed_comment_changes(tmp_path):
    path = tmp_path / "sample.py"
    source = "# before\ndef actual(): pass\n"
    path.write_text(source)
    fs = FakeContextFS()
    service = CodeResourceLifecycleService(context_fs=fs, outbox_store=FakeOutbox(), config=_config())
    service.refresh(path=path.name, workspace_root=str(tmp_path), name="repo", ctx=_ctx())
    original_uris = set(fs.nodes)
    # The storage fake does not assign versions like the atomic backend does.
    for node in fs.nodes.values():
        node.metadata["version"] = 1
    source = source.replace("before", "after")
    path.write_text(source)
    service.refresh(path=path.name, workspace_root=str(tmp_path), name="repo", ctx=_ctx())
    assert set(fs.nodes) == original_uris  # Chunk content and identity did not change.
    assert [n.metadata["file_source"] for n in fs.nodes.values() if "file_source" in n.metadata] == [source]
    service.delete_path(path=path.name, workspace_root=str(tmp_path), name="repo", ctx=_ctx())
    assert not fs.nodes


def test_per_file_refresh_repairs_reimport_snapshot_conflict_and_delete_cleans_up(tmp_path):
    from core.errors import ValidationError
    from retrieval.code_retrieval import ActiveCodeSearchRequest, CodeRetrievalService, CodeSearchScope
    from service.code_workspace_sync import ResolvedCodeWorkspace

    path = tmp_path / "sample.py"
    fs = FakeContextFS()
    lifecycle = CodeResourceLifecycleService(context_fs=fs, outbox_store=FakeOutbox(), config=_config())
    workspace = ResolvedCodeWorkspace("root", "repo", compute_root_fingerprint(tmp_path), tmp_path, 300, {})

    def search(mode, marker):
        # Reload persisted state, as after the facade invalidates its retrieval cache.
        retrieval = CodeRetrievalService(
            context_fs=fs, workspace_resolver=SimpleNamespace(resolve=lambda **kwargs: workspace),
            config=SimpleNamespace(code_retrieval_enabled=True, code_retrieval_mode=mode,
                                   code_retrieval_route_enabled=lambda route: route == "symbolic"),
        )
        return retrieval.search_active(ActiveCodeSearchRequest("parse_payload", scope=CodeSearchScope(
            paths=(path.name,), grep_terms=(marker, "parse_payload"))), _ctx())

    path.write_text("# before_marker\ndef parse_payload(): return 1\n")
    lifecycle.bootstrap(path=str(tmp_path), name="repo", ctx=_ctx())
    old_uris = set(fs.nodes)
    source = "# after_marker\ndef parse_payload(): return 2\n"
    path.write_text(source)
    lifecycle.bootstrap(path=str(tmp_path), name="repo", ctx=_ctx())
    assert sum("file_source" in n.metadata for n in fs.nodes.values()) == 2
    for marker in ("before_marker", "after_marker"):
        with pytest.raises(ValidationError) as error:
            search("full", marker)
        assert error.value.field == "scope.source_snapshot_conflict"

    for node in fs.nodes.values():
        node.metadata["version"] = 1  # The fake does not assign backend versions.
    lifecycle.refresh(path=path.name, workspace_root=str(tmp_path), name="repo", ctx=_ctx())
    assert old_uris.isdisjoint(fs.nodes)
    assert [n.metadata["file_source"] for n in fs.nodes.values() if "file_source" in n.metadata] == [source]
    outcomes = [search(mode, "after_marker") for mode in ("full", "candidate_local")]
    assert all(o.status == "ok" for o in outcomes)
    assert [e.excerpt for e in outcomes[0].evidence] == [e.excerpt for e in outcomes[1].evidence]
    assert all(search(mode, "before_marker").status == "zero_result" for mode in ("full", "candidate_local"))
    lifecycle.delete_path(path=path.name, workspace_root=str(tmp_path), name="repo", ctx=_ctx())
    assert not fs.nodes
    assert search("full", "after_marker").status == "zero_result"


def test_refresh_outbox_failure_keeps_the_complete_old_collection(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "module.py"
    source.write_text("def current():\n    return 2\n", encoding="utf-8")
    old = _existing_code_node(root, "module.py")
    fs = FakeContextFS([old])
    outbox = FakeOutbox(fail_write=True)
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="outbox write failed"):
        service.refresh(
            path=str(source),
            workspace_root=str(root),
            name="repo",
            ctx=_ctx(),
        )

    assert old.uri in fs.nodes
    assert fs.deleted == []
    assert fs.deleted == []


def test_atomic_outbox_failure_leaves_new_node_invisible(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text(
        "def current():\n    return 2\n",
        encoding="utf-8",
    )

    class AtomicFailingContextFS(FakeContextFS):
        def write_node_with_outbox(self, node, ctx, event):
            assert event.uri == node.uri
            assert event.payload["records"]
            raise RuntimeError("atomic Outbox staging failed")

    fs = AtomicFailingContextFS()
    outbox = FakeOutbox()
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="atomic Outbox staging failed"):
        service.bootstrap(path=str(root), name="repo", ctx=_ctx())

    assert fs.nodes == {}
    assert fs.written == []
    assert fs.written == []


def test_code_delete_rejects_non_code_resource():
    non_code = ContextNode(
        uri="ctx://acct/resources/dataset",
        context_type="RESOURCE",
        category="dataset",
        level=4,
        owner_space="resource:acct",
        abstract="dataset",
        overview="",
        content="",
        metadata={"resource_type": "dataset"},
    )
    service = CodeResourceLifecycleService(
        context_fs=FakeContextFS([non_code]),
        outbox_store=FakeOutbox(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="not a code resource"):
        service.delete(slug="dataset", ctx=_ctx())


def test_delete_remains_available_when_code_resource_writes_are_disabled(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    old = _existing_code_node(root, "module.py")
    fs = FakeContextFS([old])
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=FakeOutbox(),
        config=_config(enabled=False),
    )

    result = service.delete(slug="repo", ctx=_ctx())

    assert result["status"] == "PENDING"
    assert result["nodes_deleted"] == 1
    assert old.uri not in fs.nodes
