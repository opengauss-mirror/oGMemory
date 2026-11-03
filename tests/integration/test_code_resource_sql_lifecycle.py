from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.enums import EventType
from core.errors import AccessDeniedError
from core.models import ContextNode, RequestContext
from core.resource_identity import RootBindingError, compute_root_fingerprint
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
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


def _assert_resource_round_trip(fs, account_id: str, slug: str, outbox) -> None:
    ctx = _ctx(account_id)
    node = _node(account_id, slug)
    fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))

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


def _cleanup_sql_account(pool: SharedConnectionPool, account_id: str) -> None:
    conn = pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM outbox_events WHERE account_id = %s", (account_id,))
            cur.execute("DELETE FROM context_nodes WHERE account_id = %s", (account_id,))
            cur.execute(
                "DELETE FROM code_workspace_sync_roots WHERE account_id = %s",
                (account_id,),
            )
        conn.commit()
    finally:
        pool.return_connection(conn)


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


def _run_sql_delete_writer_race(
    dsn: str,
    *,
    existing_node: bool,
) -> tuple[int, Exception | None, bool, list[tuple[str, str]]]:
    """Run one real two-connection delete/write schedule with bounded coordination."""

    account_id = f"ogmemory-code-sql-mutation-race-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    node = _node(account_id, uuid.uuid4().hex)
    delete_pool = SharedConnectionPool(dsn, pool_size=1)
    writer_pool = SharedConnectionPool(dsn, pool_size=1)
    delete_fs = SQLContextFS(pool=delete_pool)
    writer_fs = SQLContextFS(pool=writer_pool)
    outbox = SQLOutboxStore(pool=delete_pool, fs=delete_fs)
    delete_has_lock = Event()
    writer_backend_ready = Event()
    writer_is_waiting_on_lock = Event()
    allow_delete_to_commit = Event()
    delete_backend_pid: list[int] = []
    writer_backend_pid: list[int] = []
    timeout_seconds = 10.0

    try:
        if existing_node:
            writer_fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))

        real_delete_lock = delete_fs._acquire_resource_mutation_lock
        real_writer_lock = writer_fs._acquire_resource_mutation_lock

        def coordinated_delete_lock(cur, locked_account_id: str, locked_uri: str) -> None:
            real_delete_lock(cur, locked_account_id, locked_uri)
            cur.execute("SELECT pg_backend_pid()")
            delete_backend_pid.append(int(cur.fetchone()[0]))
            delete_has_lock.set()
            if not writer_backend_ready.wait(timeout=timeout_seconds):
                raise TimeoutError("writer did not expose its SQL backend connection")
            if writer_backend_pid[0] == delete_backend_pid[0]:
                raise AssertionError("delete and writer must use independent SQL connections")

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                cur.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_locks "
                    "WHERE pid = %s AND locktype = 'advisory' AND NOT granted"
                    ")",
                    (writer_backend_pid[0],),
                )
                if cur.fetchone() == (True,):
                    writer_is_waiting_on_lock.set()
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("writer did not block on the SQL resource mutation lock")

            if not allow_delete_to_commit.wait(timeout=timeout_seconds):
                raise TimeoutError("timed out while holding the SQL resource mutation lock")

        def coordinated_writer_lock(cur, locked_account_id: str, locked_uri: str) -> None:
            cur.execute("SET LOCAL lock_timeout = '10s'")
            cur.execute("SELECT pg_backend_pid()")
            writer_backend_pid.append(int(cur.fetchone()[0]))
            writer_backend_ready.set()
            real_writer_lock(cur, locked_account_id, locked_uri)

        # These wrappers only coordinate thread timing around the real production
        # advisory-lock call; both operations still use real SQL transactions.
        setattr(delete_fs, "_acquire_resource_mutation_lock", coordinated_delete_lock)
        setattr(writer_fs, "_acquire_resource_mutation_lock", coordinated_writer_lock)

        delete_event = outbox.build_delete_event(node.uri)
        writer_event = outbox.build_write_event(node)
        writer_error: Exception | None = None
        writer_future = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            delete_future = executor.submit(
                delete_fs.delete_nodes_with_outbox,
                [node.uri],
                ctx,
                [delete_event],
            )
            try:
                if not delete_has_lock.wait(timeout=timeout_seconds):
                    raise TimeoutError("delete did not acquire the SQL resource mutation lock")
                writer_future = executor.submit(
                    writer_fs.write_node_with_outbox,
                    node,
                    ctx,
                    writer_event,
                )
                if not writer_is_waiting_on_lock.wait(timeout=timeout_seconds):
                    raise TimeoutError("writer did not block on the SQL resource mutation lock")
            finally:
                allow_delete_to_commit.set()

            deleted = delete_future.result(timeout=timeout_seconds)
            if writer_future is None:
                raise AssertionError("competing writer was not submitted")
            try:
                writer_future.result(timeout=timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - returned for explicit assertions
                writer_error = exc

        node_exists = writer_fs.exists(node.uri, ctx)
        conn = delete_pool.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_type, status FROM outbox_events "
                    "WHERE account_id = %s AND uri = %s ORDER BY seq",
                    (account_id, node.uri),
                )
                event_rows = [(str(row[0]), str(row[1])) for row in cur.fetchall()]
            conn.rollback()
        finally:
            delete_pool.return_connection(conn)
        return deleted, writer_error, node_exists, event_rows
    finally:
        allow_delete_to_commit.set()
        try:
            _cleanup_sql_account(delete_pool, account_id)
        finally:
            delete_pool.close_all()
            writer_pool.close_all()


@pytest.mark.integration
def test_sql_code_resource_metadata_round_trip():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for real SQL backend parity")

    account_id = f"ogmemory-code-sql-{uuid.uuid4().hex[:12]}"
    slug = uuid.uuid4().hex
    pool = SharedConnectionPool(dsn, pool_size=1)
    fs = SQLContextFS(pool=pool)
    try:
        _assert_resource_round_trip(fs, account_id, slug, SQLOutboxStore(pool=pool, fs=fs))
    finally:
        _cleanup_sql_account(pool, account_id)
        pool.close_all()

@pytest.mark.integration
@pytest.mark.parametrize("collision", ["slug", "fingerprint"])
def test_sql_concurrent_bootstrap_reserves_one_account_root_binding(tmp_path, collision):
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get(
        "TEST_SQL_CONNECTION_STRING"
    )
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for the SQL root binding race")

    account_id = f"ogmemory-code-sql-binding-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    roots, names = _binding_race_inputs(tmp_path, collision)
    pools = [
        SharedConnectionPool(dsn, pool_size=2),
        SharedConnectionPool(dsn, pool_size=2),
    ]
    filesystems = [SQLContextFS(pool=pool) for pool in pools]
    outboxes = [
        SQLOutboxStore(pool=pool, fs=filesystem)
        for pool, filesystem in zip(pools, filesystems, strict=True)
    ]
    services = [
        _lifecycle(filesystem, outbox)
        for filesystem, outbox in zip(filesystems, outboxes, strict=True)
    ]

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
            root_fingerprint=winner_fingerprint,
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
            _cleanup_sql_account(pools[0], account_id)
        finally:
            for pool in pools:
                pool.close_all()

@pytest.mark.integration
def test_sql_claimed_upsert_is_suppressed_by_atomic_delete():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for real SQL lifecycle parity")

    account_id = f"ogmemory-code-sql-race-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    node = _node(account_id, uuid.uuid4().hex)
    pool = SharedConnectionPool(dsn, pool_size=3)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    lifecycle = _lifecycle(fs, outbox)
    worker, index = _worker(fs)
    try:
        fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))
        pending = outbox.list_pending(account_id)
        assert len(pending) == 1
        claimed_uri, claimed_event = pending[0]
        assert outbox.try_acquire(
            claimed_event,
            claimed_uri,
            "integration-claimed",
        )

        result = lifecycle.delete(slug=str(node.metadata["file_slug"]), ctx=ctx)

        assert result["status"] == "PENDING"
        assert result["nodes_deleted"] == 1
        assert fs.exists(node.uri, ctx) is False

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
        assert outbox.try_acquire(delete_event, delete_uri, "integration-delete")
        delete_stats = _claimed_stats()
        worker._process_claimed_events(
            [(delete_uri, delete_event)],
            outbox,
            delete_stats,
        )

        assert delete_stats["succeeded"] == 1
        assert outbox.list_pending(account_id) == []
        assert index.count() == 0

        # A successful DELETE acknowledgement removes the durable marker, so
        # a later, intentional recreation of the same code-resource URI works.
        fs.write_node_with_outbox(node, ctx, outbox.build_write_event(node))
        assert fs.exists(node.uri, ctx) is True
        recreated = outbox.list_pending(account_id)
        assert len(recreated) == 1
        assert recreated[0][1].event_type == EventType.UPSERT_CONTEXT.value
    finally:
        _cleanup_sql_account(pool, account_id)
        pool.close_all()


@pytest.mark.integration
def test_sql_delete_transaction_rolls_back_nodes_upserts_and_deletes_together():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for real SQL rollback parity")

    account_id = f"ogmemory-code-sql-rollback-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id)
    first = _node(account_id, uuid.uuid4().hex)
    second = _node(account_id, uuid.uuid4().hex)
    pool = SharedConnectionPool(dsn, pool_size=3)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    try:
        fs.write_node_with_outbox(first, ctx, outbox.build_write_event(first))
        fs.write_node_with_outbox(second, ctx, outbox.build_write_event(second))
        assert len(outbox.list_pending(account_id)) == 2

        first_delete = outbox.build_delete_event(first.uri)
        second_delete = outbox.build_delete_event(second.uri)
        second_delete.event_id = first_delete.event_id

        with pytest.raises(Exception):
            fs.delete_nodes_with_outbox(
                [first.uri, second.uri],
                ctx,
                [first_delete, second_delete],
            )

        assert fs.exists(first.uri, ctx) is True
        assert fs.exists(second.uri, ctx) is True
        remaining = outbox.list_pending(account_id)
        assert len(remaining) == 2
        assert {event.event_type for _, event in remaining} == {EventType.UPSERT_CONTEXT.value}
    finally:
        _cleanup_sql_account(pool, account_id)
        pool.close_all()


@pytest.mark.integration
def test_sql_delete_rejects_competing_atomic_write_for_existing_resource():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for the SQL mutation race regression")

    deleted, writer_error, node_exists, event_rows = _run_sql_delete_writer_race(
        dsn,
        existing_node=True,
    )

    assert deleted == 1
    assert isinstance(writer_error, RuntimeError)
    assert str(writer_error) == "code resource URI is pending durable deletion"
    assert node_exists is False
    assert event_rows == [(EventType.DELETE_CONTEXT.value, "PENDING")]


@pytest.mark.integration
def test_sql_delete_rejects_competing_atomic_write_for_absent_resource():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for the SQL mutation race regression")

    deleted, writer_error, node_exists, event_rows = _run_sql_delete_writer_race(
        dsn,
        existing_node=False,
    )

    assert deleted == 0
    assert isinstance(writer_error, RuntimeError)
    assert str(writer_error) == "code resource URI is pending durable deletion"
    assert node_exists is False
    assert event_rows == [(EventType.DELETE_CONTEXT.value, "PENDING")]
