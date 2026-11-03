from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from pyagfs.exceptions import AGFSClientError

from commit.outbox_store import OutboxStore
from core.enums import EventType
from core.errors import AccessDeniedError
from core.models import ContextNode, RequestContext
from core.resource_identity import RootBindingError, compute_root_fingerprint
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.code_resource_lifecycle import CodeResourceLifecycleService


def _ctx(account_id: str) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="any-user",
        agent_id="any-agent",
        session_id="session",
        trace_id="trace",
    )


def _node(account_id: str, slug: str) -> ContextNode:
    return ContextNode(
        uri=f"ctx://{account_id}/resources/{slug}",
        context_type="RESOURCE",
        category="code",
        level=0,
        owner_space=f"resource:{account_id}",
        abstract="parse config",
        overview="parse_config(path)",
        content="def parse_config(path):\n    return path\n",
        metadata={
            "resource_type": "code",
            "resource_root_slug": "project",
            "root_fingerprint": "f" * 64,
            "file_key": "a" * 64,
            "file_slug": "a" * 32,
            "relative_path": "src/service.py",
            "source_extension": ".py",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "symbol": "parse_config",
            "symbol_kind": "function",
            "signature": "parse_config(path)",
            "chunk_hash": "b" * 64,
            "chunk_key": "c" * 64,
            "graph": {
                "calls": ["open"],
                "relations": [{"type": "calls", "target_symbol": "open"}],
            },
        },
    )


def _assert_resource_round_trip(fs, account_id: str, slug: str) -> None:
    ctx = _ctx(account_id)
    node = _node(account_id, slug)
    fs.write_node(node, ctx)

    loaded = fs.read_node(node.uri, ctx)
    fields = fs.read_node_fields(node.uri, ctx, "abstract")
    children = fs.list_children(f"ctx://{account_id}/resources/", ctx)

    assert loaded.context_type == "RESOURCE"
    assert loaded.owner_space == f"resource:{account_id}"
    assert loaded.metadata["relative_path"] == "src/service.py"
    assert loaded.metadata["chunk_key"] == "c" * 64
    assert loaded.metadata["graph"]["calls"] == ["open"]
    assert fields.metadata["graph"] == loaded.metadata["graph"]
    assert node.uri in children

    with pytest.raises(AccessDeniedError):
        fs.read_node(node.uri, _ctx(f"other-{account_id}"))


def _lifecycle(fs, outbox) -> CodeResourceLifecycleService:
    return CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=SimpleNamespace(
            code_resources_enabled=True,
            code_resources_target_chars=2500,
            code_resources_overlap_chars=300,
            code_resources_chunk_max_chars=4000,
        ),
    )


def _worker(fs) -> tuple[OutboxWorker, InMemoryVectorIndex]:
    index = InMemoryVectorIndex(dimension=8)
    return (
        OutboxWorker(
            vector_index=index,
            embedder=MockEmbedder(dimension=8),
            fs=fs,
            llm=None,
            directory_summary_enabled=False,
        ),
        index,
    )


def _claimed_stats() -> dict[str, int]:
    return {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "moved_to_dlq": 0,
        "skipped": 0,
    }


def _binding_race_inputs(tmp_path, collision: str):
    first_root = tmp_path / "first"
    second_root = first_root if collision == "fingerprint" else tmp_path / "second"
    first_root.mkdir()
    if second_root != first_root:
        second_root.mkdir()
    (first_root / "module.py").write_text("value = 1\n", encoding="utf-8")
    if second_root != first_root:
        (second_root / "module.py").write_text("value = 2\n", encoding="utf-8")
    names = ("first", "second") if collision == "fingerprint" else ("shared", "shared")
    return (first_root, second_root), names


def _run_lifecycle_binding_race(services, roots, names, ctx):
    ready = Barrier(2)
    for service in services:
        real_reserve = service._fs.reserve_code_workspace_root_binding

        def coordinated_reserve(*, _reserve=real_reserve, **kwargs):
            ready.wait(timeout=5)
            return _reserve(**kwargs)

        setattr(service._fs, "reserve_code_workspace_root_binding", coordinated_reserve)

    def bootstrap(index):
        try:
            return services[index].bootstrap(
                path=str(roots[index]),
                name=names[index],
                ctx=ctx,
            )
        except Exception as exc:  # noqa: BLE001 - asserted by the integration test
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(bootstrap, range(2)))


def _assert_one_binding_winner(results) -> None:
    successes = [result for result in results if isinstance(result, dict)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RootBindingError)


@pytest.mark.integration
def test_agfs_code_resource_metadata_round_trip(agfs):
    account_id = f"ogmemory-code-agfs-{uuid.uuid4().hex[:12]}"
    slug = uuid.uuid4().hex
    _assert_resource_round_trip(agfs, account_id, slug)


@pytest.mark.integration
@pytest.mark.parametrize("collision", ["slug", "fingerprint"])
def test_agfs_concurrent_bootstrap_reserves_one_account_root_binding(
    agfs,
    agfs_client,
    tmp_path,
    collision,
):
    account_id = f"ogmemory-code-agfs-binding-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    roots, names = _binding_race_inputs(tmp_path, collision)
    filesystems = [
        AGFSContextFS(agfs_client, mount_prefix=agfs._mount_prefix),
        AGFSContextFS(agfs_client, mount_prefix=agfs._mount_prefix),
    ]
    outboxes = [
        OutboxStore(
            client=agfs_client,
            fs=filesystem,
            mount_prefix=agfs._mount_prefix,
        )
        for filesystem in filesystems
    ]
    services = [
        _lifecycle(filesystem, outbox)
        for filesystem, outbox in zip(filesystems, outboxes, strict=True)
    ]
    account_path = agfs._mount_prefix + f"/accounts/{account_id}"

    try:
        results = _run_lifecycle_binding_race(services, roots, names, ctx)
        _assert_one_binding_winner(results)

        child_uris = filesystems[0].list_children(
            f"ctx://{account_id}/resources/",
            ctx,
        )
        nodes = [filesystems[0].read_node(uri, ctx) for uri in child_uris]
        assert nodes
        persisted_pairs = {
            (
                str(node.metadata["resource_root_slug"]),
                str(node.metadata["root_fingerprint"]),
            )
            for node in nodes
        }
        assert len(persisted_pairs) == 1
        winner_slug, winner_fingerprint = next(iter(persisted_pairs))
        binding = filesystems[1].read_code_workspace_root_binding(
            ctx=ctx,
            root_slug=winner_slug,
        )
        assert binding is not None
        assert binding[1:] == (winner_slug, winner_fingerprint)
        pending = outboxes[0].list_pending(account_id)
        assert {event.uri for _, event in pending} == set(child_uris)
        losing_pairs = {
            (names[index], compute_root_fingerprint(roots[index]))
            for index, result in enumerate(results)
            if isinstance(result, Exception)
        }
        assert persisted_pairs.isdisjoint(losing_pairs)
    finally:
        try:
            agfs_client.rm(account_path, recursive=True)
        except Exception:
            pass


@pytest.mark.integration
def test_agfs_claimed_upsert_is_suppressed_by_durable_delete(agfs, agfs_client):
    account_id = f"ogmemory-code-agfs-race-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    node = _node(account_id, uuid.uuid4().hex)
    outbox = OutboxStore(
        client=agfs_client,
        fs=agfs,
        mount_prefix=agfs._mount_prefix,
    )
    lifecycle = _lifecycle(agfs, outbox)
    worker, index = _worker(agfs)

    agfs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    pending = outbox.list_pending(account_id)
    assert len(pending) == 1
    claimed_uri, claimed_event = pending[0]
    assert claimed_event.event_type == EventType.UPSERT_CONTEXT.value
    assert outbox.try_acquire(claimed_event, claimed_uri, "integration-claimed")
    outbox.mark_processing(claimed_event, claimed_uri)

    result = lifecycle.delete(slug=str(node.metadata["file_slug"]), ctx=ctx)

    assert result["status"] == "PENDING"
    assert result["nodes_deleted"] == 1
    assert agfs.exists(node.uri, ctx) is False
    assert agfs.is_tombstoned(node.uri, ctx) is True

    claimed_stats = _claimed_stats()
    worker._process_claimed_events(
        [(claimed_uri, claimed_event)],
        outbox,
        claimed_stats,
    )
    assert claimed_stats["skipped"] == 1
    assert index.count() == 0

    delete_pending = outbox.list_pending(account_id)
    assert len(delete_pending) == 1
    delete_uri, delete_event = delete_pending[0]
    assert delete_event.event_type == EventType.DELETE_CONTEXT.value
    assert delete_event.payload["durable_delete"] is True
    assert outbox.try_acquire(delete_event, delete_uri, "integration-delete")
    delete_stats = _claimed_stats()
    worker._process_claimed_events(
        [(delete_uri, delete_event)],
        outbox,
        delete_stats,
    )

    assert delete_stats["succeeded"] == 1
    assert outbox.list_pending(account_id) == []
    assert agfs.is_tombstoned(node.uri, ctx) is False
    assert index.count() == 0


@pytest.mark.integration
def test_agfs_delete_payload_failure_eventually_converges_and_allows_rewrite(
    agfs,
    agfs_client,
):
    account_id = f"ogmemory-code-agfs-delete-retry-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    node = _node(account_id, uuid.uuid4().hex)
    outbox = OutboxStore(
        client=agfs_client,
        fs=agfs,
        mount_prefix=agfs._mount_prefix,
    )
    lifecycle = _lifecycle(agfs, outbox)
    worker, index = _worker(agfs)

    agfs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    upsert_stats = worker.run_once(outbox, [account_id], worker_id="agfs-upsert")
    assert upsert_stats["succeeded"] == 1
    assert index.count() > 0

    real_delete_node_payload = agfs.delete_node_payload
    payload_cleanup_attempts = 0

    def fail_first_payload_cleanup(uri, cleanup_ctx, *, mutation_lease=None):
        nonlocal payload_cleanup_attempts
        payload_cleanup_attempts += 1
        if payload_cleanup_attempts == 1:
            raise RuntimeError("transient AGFS payload delete failure")
        real_delete_node_payload(uri, cleanup_ctx, mutation_lease=mutation_lease)

    setattr(agfs, "delete_node_payload", fail_first_payload_cleanup)

    with pytest.raises(RuntimeError, match="transient AGFS payload delete failure"):
        lifecycle.delete(slug=str(node.metadata["file_slug"]), ctx=ctx)

    node_path = agfs._uri_to_agfs_path(node.uri)
    assert agfs_client.read(node_path + agfs.FILE_META)
    assert agfs.is_tombstoned(node.uri, ctx) is True
    delete_pending = outbox.list_pending(account_id)
    assert len(delete_pending) == 1
    assert delete_pending[0][1].event_type == EventType.DELETE_CONTEXT.value

    delete_stats = worker.run_once(outbox, [account_id], worker_id="agfs-delete-retry")

    assert delete_stats["succeeded"] == 1
    assert payload_cleanup_attempts == 2
    assert index.count() == 0
    assert outbox.list_pending(account_id) == []
    assert agfs.is_tombstoned(node.uri, ctx) is False
    assert agfs.exists(node.uri, ctx) is False
    for filename in (
        agfs.FILE_CONTENT,
        agfs.FILE_RELATIONS,
        agfs.FILE_ABSTRACT,
        agfs.FILE_OVERVIEW,
        agfs.FILE_META,
    ):
        with pytest.raises(AGFSClientError):
            agfs_client.read(node_path + filename)

    node.metadata["_expected_version"] = 0
    recreate_event = outbox.build_write_event(node)
    agfs.write_node_with_outbox(node, ctx, recreate_event)
    assert agfs.exists(node.uri, ctx) is True
    assert any(
        pending_event.event_id == recreate_event.event_id
        for _, pending_event in outbox.list_pending(account_id)
    )


@pytest.mark.integration
@pytest.mark.parametrize("first", ["write", "delete"])
def test_agfs_atomic_write_delete_race_never_resurrects(
    agfs,
    agfs_client,
    first,
):
    account_id = f"ogmemory-code-agfs-mutation-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    node = _node(account_id, uuid.uuid4().hex)
    writer_fs = AGFSContextFS(agfs_client, mount_prefix=agfs._mount_prefix)
    delete_fs = AGFSContextFS(agfs_client, mount_prefix=agfs._mount_prefix)
    writer_outbox = OutboxStore(
        client=agfs_client,
        fs=writer_fs,
        mount_prefix=agfs._mount_prefix,
    )
    delete_outbox = OutboxStore(
        client=agfs_client,
        fs=delete_fs,
        mount_prefix=agfs._mount_prefix,
    )
    lifecycle = _lifecycle(delete_fs, delete_outbox)
    node.metadata["_expected_version"] = 0
    writer_fs.write_node_with_outbox(
        node,
        ctx,
        writer_outbox.build_write_event(node),
    )
    updated = ContextNode(
        **{
            **node.__dict__,
            "metadata": {
                **node.metadata,
                "_expected_version": 1,
                "description": "metadata-only update",
            },
        }
    )
    updated_event = writer_outbox.build_write_event(updated)
    first_acquired = Event()
    release_first = Event()
    second_attempted = Event()

    if first == "write":
        first_fs = writer_fs
        second_fs = delete_fs
    else:
        first_fs = delete_fs
        second_fs = writer_fs
    real_first_lock = first_fs.resource_mutation_lock
    real_second_lock = second_fs.resource_mutation_lock

    @contextmanager
    def coordinated_first_lock(uri, lock_ctx):
        with real_first_lock(uri, lock_ctx) as lease:
            first_acquired.set()
            if not release_first.wait(5):
                raise TimeoutError("timed out while holding AGFS mutation lock")
            yield lease

    @contextmanager
    def coordinated_second_lock(uri, lock_ctx):
        second_attempted.set()
        with real_second_lock(uri, lock_ctx) as lease:
            yield lease

    setattr(first_fs, "resource_mutation_lock", coordinated_first_lock)
    setattr(second_fs, "resource_mutation_lock", coordinated_second_lock)

    with ThreadPoolExecutor(max_workers=2) as executor:
        if first == "write":
            first_future = executor.submit(
                writer_fs.write_node_with_outbox,
                updated,
                ctx,
                updated_event,
            )
            assert first_acquired.wait(5)
            second_future = executor.submit(
                lifecycle.delete,
                slug=str(node.metadata["file_slug"]),
                ctx=ctx,
            )
        else:
            first_future = executor.submit(
                lifecycle.delete,
                slug=str(node.metadata["file_slug"]),
                ctx=ctx,
            )
            assert first_acquired.wait(5)
            second_future = executor.submit(
                writer_fs.write_node_with_outbox,
                updated,
                ctx,
                updated_event,
            )

        assert second_attempted.wait(5)
        time.sleep(0.1)
        assert second_future.done() is False
        release_first.set()
        first_result = first_future.result(timeout=10)
        if first == "write":
            assert first_result is None
            assert second_future.result(timeout=10)["nodes_deleted"] == 1
        else:
            assert first_result["nodes_deleted"] == 1
            with pytest.raises(RuntimeError, match="durable deletion is pending"):
                second_future.result(timeout=10)

    assert delete_fs.is_tombstoned(node.uri, ctx) is True
    assert delete_fs.exists(node.uri, ctx) is False
    worker, _index = _worker(delete_fs)
    stats = worker.run_once(
        delete_outbox,
        [account_id],
        worker_id="agfs-mutation-cleanup",
    )
    assert stats["succeeded"] == 1
    assert delete_fs.is_tombstoned(node.uri, ctx) is False
    assert delete_fs.exists(node.uri, ctx) is False
