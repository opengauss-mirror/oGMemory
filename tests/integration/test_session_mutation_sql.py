"""Disposable PostgreSQL, real public entries, deterministic extraction only."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg2
import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.errors import ToolCallProtocolError
from core.models import RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.session_mutation import (
    SessionMutationBusyError,
    SessionMutationLostError,
    active_session_guard,
)
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.extraction_checkpoint import EXTENSION_KEY
from session.models import ArchiveWriteResult
from session.session_state import SessionState
from session.sql_archive_store import SQLSessionArchiveStore
from tests.unit.server.test_extraction_batch_recovery import candidate, params, setup_service
from tests.unit.server.test_lifecycle_batch_recovery import reject_entity_once


@pytest.fixture
def database():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    created = []

    def make():
        pool = SharedConnectionPool(
            dsn, pool_size=1,
            dialect=os.environ.get("OGMEM_IDENTITY_TEST_DIALECT", "postgresql"),
        )
        fs = SQLContextFS(pool=pool)
        outbox = SQLOutboxStore(pool=pool, fs=fs)
        archive = SQLSessionArchiveStore(pool=pool)
        created.append(pool)
        return fs, outbox, archive

    yield dsn, make
    for pool in created:
        pool.close_all()


@pytest.fixture
def ctx():
    return RequestContext("session-fence-" + uuid4().hex, "reader", "agent", "session", "trace")


def service_pair(make, ctx, extract=None):
    fs, outbox, archive = make()
    service, writer, manager = setup_service(fs, ctx, outbox=outbox, extract=extract)
    del manager.commit_snapshot  # The real public archive path, with deterministic compression.
    manager._get_archive_store = lambda: archive
    manager._compress = Mock(return_value=("Receiver repair", "Receiver repair"))
    return service, writer, manager, archive, outbox


def event_nodes(writer, ctx):
    directory = writer._writer._identity_resolver.uris.get_directory_uri("event", ctx)
    return [writer._fs.read_node(uri, ctx) for uri in writer._fs.list_children(directory, ctx)]


def event_outbox(outbox, ctx):
    return [event for uri, event in outbox.list_pending(ctx.account_id) if "/memories/events/" in uri]


@pytest.mark.parametrize("task_status, attempts", [("queued", 0), ("running", 1)])
def test_background_retry_resumes_after_sql_lock_contention(database, ctx, task_status, attempts):
    _, make = database
    service, writer, manager, archive, _ = service_pair(make, ctx)
    worker = service._extraction_retry
    extract = writer._pipeline.extract.side_effect
    writer._pipeline.extract.side_effect = ToolCallProtocolError("temporary extraction failure")
    with patch.object(worker, "wake"):
        result = service.after_turn({**params(ctx), "retryInBackground": True})
    task = result["recovery_task"]
    with service._session_mutation_scope(ctx):
        buf = manager.get_or_create(ctx.session_id)
        saved_task = buf.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        saved_task.update(status=task_status, attempts=attempts, next_attempt_at=0)
        assert manager.save_session_state(ctx.session_id, ctx)
    writer._pipeline.extract.side_effect = extract
    busy, completed = threading.Event(), threading.Event()
    run_fenced = worker._run_fenced

    def observe(request_ctx, archive_id):
        try:
            outcome = run_fenced(request_ctx, archive_id)
        except SessionMutationBusyError:
            busy.set()
            raise
        if manager.get_or_create(ctx.session_id).request_completed("original"):
            completed.set()
        return outcome

    worker._run_fenced = observe
    try:
        state_uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
        with writer._fs.session_mutation_lock(state_uri, ctx):
            worker.wake(ctx, task)
            assert busy.wait(5)
            durable = json.loads(writer._fs.read_node(state_uri, ctx).content)
            checkpoint = durable["runtime_extensions"][EXTENSION_KEY]["batch"]["retry_task"]
            assert checkpoint["status"] == task_status
            assert checkpoint["attempts"] == attempts
        assert completed.wait(10)
    finally:
        worker.close(timeout=5)
    assert len(event_nodes(writer, ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_background_topic_directory_contention_recovers_after_repeated_waits(database, ctx, mode):
    from service.api import MemoryWriteAPI
    from tests.unit.extraction.test_node_processing import node_payload, operation

    _, make = database
    started, release, repeated = threading.Event(), threading.Event(), threading.Event()
    calls, observations, services = [], [], []

    class Model:
        def __init__(self, key):
            self.key = key

        def extraction(self):
            calls.append((self.key, "extract"))
            return {"tool": "extract_entity", "input": {
                "owner_scope": "user", "routing_key": self.key, "object_name": self.key,
                "abstract": f"Receiver {self.key}.", "overview": "- Operates at 37 MHz.",
                "content": f"# {self.key}\n\n## Features\n- Operates at 37 MHz.", "confidence": .99}}

        def complete_json(self, prompt, schema=None, **kwargs):
            return {"spans": [{"start": 0, "end": 0, "categories": ["entity"]}]}

        def complete_with_tools_messages(self, messages, tools, tool_choice):
            item = self.extraction()
            return [], json.dumps([{"name": item["tool"], "input": item["input"]}])

        def complete_with_tools(self, **kwargs):
            if not kwargs["prompt"].startswith("NODE_PROCESSING:"):
                return [self.extraction()]
            calls.append((self.key, "node_processing"))
            if self.key == "receiver_red":
                active_session_guard().ensure_owned()
                started.set()
                assert release.wait(55)
            payload = node_payload(kwargs)
            item = operation([draft["id"] for draft in payload["drafts"]], fact="Operates at 37 MHz.")
            item.update(routing_key=self.key, object_name=self.key, abstract=f"Receiver {self.key}.")
            return [{"tool": "process_memory_nodes", "input": {"operations": [item]}}]

    def build(request_ctx, key):
        service, old_writer, manager, archive, outbox = service_pair(make, request_ctx)
        writer = MemoryWriteAPI(old_writer._fs, Model(key), outbox, extraction_mode=mode,
                                node_processing_enabled=True, card_format_enabled=True,
                                dual_pass_enabled=False, directory_summary_enabled=False)
        service._write_api = writer
        service._cfg.node_processing_enabled = service._cfg.card_format_enabled = True
        services.append(service)
        return service, writer, manager, archive

    first, fw, fm, fa = build(ctx, "receiver_red")
    other_ctx = replace(ctx, session_id=ctx.session_id + "-other", trace_id="directory-busy-other")
    second, sw, sm, sa = build(other_ctx, "receiver_blue")
    worker = second._extraction_retry
    run_fenced = worker._run_fenced

    def observe(request_ctx, archive_id):
        began = time.monotonic()
        result = run_fenced(request_ctx, archive_id)
        if time.monotonic() - began >= .9:
            state = json.loads(sw._fs.read_node(SessionState.state_uri(
                request_ctx.account_id, request_ctx.session_id), request_ctx).content)
            batch = state.get("runtime_extensions", {}).get(EXTENSION_KEY, {}).get("batch", {})
            if batch.get("retry_task"):
                observations.append(batch["retry_task"])
                if len(observations) >= 3:
                    repeated.set()
        return result

    worker._run_fenced = observe
    began = time.monotonic()
    try:
        assert first.after_turn({**params(ctx, request="", content="My receiver_red operates at 37 MHz."),
                                 "wait": False})["status"] == "processing"
        assert started.wait(10)
        assert second.after_turn({**params(other_ctx, request="", content="My receiver_blue operates at 37 MHz."),
                                  "wait": False})["status"] == "processing"
        assert sm.get_or_create(other_ctx.session_id).extraction_done_event.wait(8)
        durable = json.loads(sw._fs.read_node(SessionState.state_uri(
            other_ctx.account_id, other_ctx.session_id), other_ctx).content)
        batch = durable["runtime_extensions"][EXTENSION_KEY]["batch"]
        print("directory_busy_initial=" + json.dumps({"mode": mode, "account_id": ctx.account_id,
              "archive_id": batch["archive_id"], "retry_task": batch.get("retry_task"),
              "elapsed_seconds": time.monotonic() - began, "model_calls": calls}))
        assert batch["retry_task"]["status"] == "queued"
        assert repeated.wait(35)
        assert all(item["status"] == "queued" and item["attempts"] == 0 for item in observations)
        release.set()
        assert fm.get_or_create(ctx.session_id).extraction_done_event.wait(10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = json.loads(sw._fs.read_node(SessionState.state_uri(
                other_ctx.account_id, other_ctx.session_id), other_ctx).content)
            if EXTENSION_KEY not in state.get("runtime_extensions", {}):
                break
            time.sleep(.05)
        assert EXTENSION_KEY not in state.get("runtime_extensions", {})
        directory = sw._writer._identity_resolver.uris.get_directory_uri("entity", other_ctx)
        nodes = [sw._fs.read_node(uri, other_ctx) for uri in sw._fs.list_children(directory, other_ctx)]
        assert len(nodes) == 2
        assert {node.metadata["version"] for node in nodes} == {1}
        for service, manager, archive, request_ctx in ((first, fm, fa, ctx), (second, sm, sa, other_ctx)):
            assert len(archive.list_archives(request_ctx.session_id, request_ctx)) == 1
            assert manager.get_or_create(request_ctx.session_id).messages == []
        assert sorted(calls) == sorted([(key, stage) for key in ("receiver_red", "receiver_blue")
                                       for stage in ("extract", "node_processing")])
        print("directory_busy_completed=" + json.dumps({"mode": mode, "observations": observations,
              "entity_nodes": len(nodes), "archives_per_session": [1, 1],
              "elapsed_seconds": time.monotonic() - began, "model_calls": calls}))
    finally:
        release.set()
        fm.get_or_create(ctx.session_id).extraction_done_event.wait(10)
        for service in services:
            service._extraction_retry.close(timeout=5)


def test_cached_workers_replay_one_archive_and_node(database, ctx, monkeypatch):
    _, make = database
    if os.environ.get("OGMEM_FENCE_TEST_LEGACY"):
        monkeypatch.setattr(SQLContextFS, "session_mutation_lock", None)
    b, bw, _, archive, outbox = service_pair(make, ctx)
    b._cfg.after_turn_threshold = 10000
    assert b.after_turn(params(ctx, request="warmup", content="A harmless prior message."))["status"] == "accumulating"
    started, release = threading.Event(), threading.Event()

    def slow(**kwargs):
        started.set()
        assert release.wait(10)
        return [replace(candidate(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]

    a, aw, _, _, _ = service_pair(make, ctx, slow)
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(a.after_turn, params(ctx))
        assert started.wait(10)
        b._cfg.after_turn_threshold = 1
        blocked = b.after_turn(params(ctx))
        release.set()
        assert result.result(timeout=10)["status"] == "completed"
    replay = b.after_turn(params(ctx))
    assert len(event_nodes(bw, ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert len(event_outbox(outbox, ctx)) == 1
    assert len(outbox.list_pending(ctx.account_id)) == 2  # memory and its archive projection
    assert blocked["status"] == "processing" and replay["status"] == "duplicate"
    assert aw._pipeline.extract.call_count == 1 and bw._pipeline.extract.call_count == 0


def test_guard_routes_shared_pool_and_nested_locks_without_extra_borrow(database, ctx):
    _, make = database
    fs, outbox, archive = make()
    state_uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with fs.session_mutation_lock(state_uri, ctx) as guard:
        for store in (fs, outbox, archive):
            conn = store._get_connection()
            assert conn is guard.connection
            store.bind_tenant(conn, ctx.account_id)
            conn.rollback()
            store._return_connection(conn)
        directory = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/events"
        with fs.memory_identity_lock(directory, ctx), fs.memory_identity_lock(directory, ctx):
            guard.ensure_owned()
        assert len(guard.depths) == 1
        guard.ensure_owned()


def test_lost_owner_connection_cannot_write_or_replace_after_handoff(database, ctx):
    dsn, make = database
    started, release = threading.Event(), threading.Event()
    owner_pid = []

    def slow(**kwargs):
        guard = active_session_guard()
        with guard.raw.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            owner_pid.append(cursor.fetchone()[0])
        guard.raw.rollback()
        started.set()
        assert release.wait(10)
        return [candidate()]

    old, old_writer, old_manager, archive, outbox = service_pair(make, ctx, slow)
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(old.after_turn, params(ctx))
        assert started.wait(10)
        control = psycopg2.connect(dsn)
        try:
            with control.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(%s)", (owner_pid[0],))
                assert cursor.fetchone()[0]
            control.commit()
        finally:
            control.close()
        new, writer, manager, _, _ = service_pair(make, ctx)
        assert new.after_turn(params(ctx))["status"] == "completed"
        release.set()
        assert result.result(timeout=10)["status"] == "needs_confirmation"
    assert len(event_nodes(writer, ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert len(event_outbox(outbox, ctx)) == 1
    assert len(outbox.list_pending(ctx.account_id)) == 2
    assert not manager.get_or_create(ctx.session_id).messages
    assert EXTENSION_KEY in old_manager.get_or_create(ctx.session_id).runtime_extensions
    # The old in-memory poisoned snapshot is refreshed, never written back.
    assert old.after_turn(params(ctx))["status"] == "duplicate"
    assert old_writer._pipeline.extract.call_count == 1


def test_default_async_child_owns_connection_and_blocks_another_worker(database, ctx):
    _, make = database
    started, release = threading.Event(), threading.Event()
    guards = []

    def slow(**kwargs):
        guard = active_session_guard()
        assert guard.thread_id == threading.get_ident()
        guard.ensure_owned()
        guards.append(guard)
        started.set()
        assert release.wait(10)
        return [replace(candidate(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]

    a, writer, manager, archive, outbox = service_pair(make, ctx, slow)
    b, _, _, _, _ = service_pair(make, ctx)
    request = {"sessionId": ctx.session_id, "_ctx": ctx,
               "messages": [{"id": "source-one", "role": "user", "content": "I repaired the receiver."}]}
    assert a.after_turn(request)["status"] == "processing"
    assert started.wait(10)
    assert guards[0].thread_id != threading.get_ident()
    assert b.after_turn(params(ctx, request="later", content="An independent tail."))["status"] == "processing"
    release.set()
    assert manager.get_or_create(ctx.session_id).extraction_done_event.wait(10)
    assert len(event_nodes(writer, ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert len(event_outbox(outbox, ctx)) == 1
    assert len(outbox.list_pending(ctx.account_id)) == 2
    assert not manager.get_or_create(ctx.session_id).messages


def test_busy_acquisition_is_bounded_and_context_mismatch_poisons(database, ctx):
    _, make = database
    first, _, _ = make()
    second, _, _ = make()
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with first.session_mutation_lock(uri, ctx):
        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(lambda: second.session_mutation_lock(uri, ctx, timeout_seconds=.1).__enter__())
            with pytest.raises(SessionMutationBusyError):
                future.result(timeout=2)
    with pytest.raises(SessionMutationLostError), first.session_mutation_lock(uri, ctx) as guard:
        try:
            first.read_node(uri, replace(ctx, user_id="another-owner"))
        except SessionMutationLostError:
            with pytest.raises(SessionMutationLostError):
                guard.ensure_owned()


def test_unguarded_cached_save_cannot_overwrite_active_durable_batch(database, ctx):
    _, make = database
    stale, writer, manager, _, _ = service_pair(make, ctx)
    if os.environ.get("OGMEM_FENCE_TEST_LEGACY_SAVE"):
        manager._require_sql_mutation = lambda *args, **kwargs: False
    stale._cfg.after_turn_threshold = 10000
    assert stale.after_turn(params(ctx, request="warmup"))["status"] == "accumulating"
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    fs, _, _ = make()
    with fs.session_mutation_lock(uri, ctx):
        latest, _, current, _, _ = service_pair(make, ctx)
        # The nested public request uses its existing owning connection.
        latest._cfg.after_turn_threshold = 10000
        assert latest.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
            {"id": "new-tail", "role": "user", "content": "An independent tail."}]})["ingested"]
        saved = fs.read_node(uri, ctx).content
        # Deliberately remove ownership to reproduce a different cached worker.
        from fs.sql_adapter.session_mutation import _ACTIVE
        token = _ACTIVE.set(None)
        try:
            assert manager.save_session_state(ctx.session_id, ctx) is False
        finally:
            _ACTIVE.reset(token)
        assert fs.read_node(uri, ctx).content == saved
        assert len(current.get_or_create(ctx.session_id).messages) == 2
    assert len(json.loads(writer._fs.read_node(uri, ctx).content)["buffer_messages"]) == 2


def test_cached_hard_deleted_session_state_requires_confirmation(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    service._cfg.after_turn_threshold = 10000
    assert service.after_turn(params(ctx))["status"] == "accumulating"
    before = list(manager.get_or_create(ctx.session_id).messages)
    writer._fs.delete_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx)
    response = service.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": "later", "role": "user", "content": "A new independent source."}]})
    assert response["status"] == "needs_confirmation"
    assert manager.get_or_create(ctx.session_id).messages == before
    writer._pipeline.extract.assert_not_called()


def test_outbox_foreign_uri_is_rejected_even_with_original_context(database, ctx):
    _, make = database
    fs, outbox, _ = make()
    from core.models import ContextNode
    foreign = replace(ctx, account_id="other-" + uuid4().hex)
    node = ContextNode(uri=f"ctx://{foreign.account_id}/users/reader/memories/events/item",
                       category="event", owner_space="user:reader", context_type="MEMORY",
                       level=2, abstract="fact", overview="fact", content="fact")
    with pytest.raises(SessionMutationLostError), fs.session_mutation_lock(
        SessionState.state_uri(ctx.account_id, ctx.session_id), ctx,
    ):
        outbox.register_write(node, ctx)
    assert outbox.list_pending(foreign.account_id) == []


def test_sql_bootstrap_uses_owning_connection_without_model_or_pool_wait(database, ctx):
    dsn, make = database
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    service = MemoryService(config=OgMemConfig(storage_backend="sql", sql_connection_string=dsn,
                                              session_state_bridge_enabled=False, after_turn_threshold=10000))
    service.get_write_api = Mock(side_effect=AssertionError("bootstrap initialized a writer"))
    service.get_llm = Mock(side_effect=AssertionError("bootstrap initialized a model"))
    service._get_shared_embedder = Mock(side_effect=AssertionError("bootstrap initialized embeddings"))
    pool = service._get_shared_sql_pool()
    # A regular pool borrow is unavailable; mutation bootstrap must bypass it.
    pool.get_connection = Mock(side_effect=AssertionError("bootstrap borrowed a regular pooled connection"))
    try:
        assert service.after_turn(params(ctx))["status"] == "accumulating"
        assert service.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
            {"id": "tail", "role": "user", "content": "An independent source."}]})["ingested"]
        assert len(service.get_session_manager().get_or_create(ctx.session_id).messages) == 2
        service.get_write_api.assert_not_called()
        service.get_llm.assert_not_called()
        service._get_shared_embedder.assert_not_called()
    finally:
        pool.close_all()


def test_sql_compact_keeps_archive_and_state_on_its_owning_session(database, ctx):
    _, make = database
    service, writer, manager, archive, outbox = service_pair(make, ctx)
    service._cfg.after_turn_threshold = 10000
    assert service.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": "compact-source", "role": "user", "content": "I repaired the receiver." * 30}]})["ingested"]
    result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"})
    assert result["compacted"] is True
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert len(event_nodes(writer, ctx)) == 1
    assert len(event_outbox(outbox, ctx)) == 1
    assert not manager.get_or_create(ctx.session_id).messages


def test_sql_default_dispose_child_fences_ingest_and_idle_wait_does_not_hold_lock(database, ctx):
    _, make = database
    started, release = threading.Event(), threading.Event()

    def slow(**kwargs):
        guard = active_session_guard()
        assert guard.thread_id == threading.get_ident()
        guard.ensure_owned()
        started.set()
        assert release.wait(10)
        return [candidate()]

    service, writer, manager, _, outbox = service_pair(make, ctx, slow)
    service.wake_skill_sedimentation = Mock()
    service._flush_session_tool_usage_async = Mock()
    service._cfg.after_turn_threshold = 10000
    assert service.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": "dispose-source", "role": "user", "content": "I repaired the receiver." * 60}]})["ingested"]
    result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "background_flush_dispatched"
    assert started.wait(10)
    other, _, _, _, _ = service_pair(make, ctx)
    assert other.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": "tail", "role": "user", "content": "An independent source."}]})["status"] == "processing"
    release.set()
    idle = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                   "drainOutbox": False, "waitOutbox": False, "timeoutSeconds": 10})
    assert idle["idle"] is True
    assert len(event_nodes(writer, ctx)) == 1
    assert len(event_outbox(outbox, ctx)) == 1
    assert not manager.has_session(ctx.session_id, ctx)


def test_public_parallel_writes_keep_every_transaction_on_the_owning_thread(database, ctx):
    _, make = database
    service, writer, _, _, outbox = service_pair(make, ctx)
    original = writer._writer.write_candidate
    observed = []
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with writer._fs.session_mutation_lock(uri, ctx) as guard:
        def checked_write(entry, request):
            assert active_session_guard() is guard
            assert guard.thread_id == threading.get_ident()
            connection = writer._fs._get_connection()
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid()")
                    observed.append(cursor.fetchone()[0])
                connection.rollback()
            finally:
                writer._fs._return_connection(connection)
            return original(entry, request)

        writer._writer.write_candidate = checked_write
        results = writer.write_memories([
            replace(candidate(), routing_key="receiver_alpha", title="Receiver Alpha repair",
                    provenance_ids=["prov:1:archive:distinct-source:alpha"]),
            replace(candidate(), routing_key="receiver_beta", title="Receiver Beta repair",
                    provenance_ids=["prov:1:archive:distinct-source:beta"]),
        ], ctx, parallel=True)
        guard.ensure_owned()
    assert len(observed) == 2 and len(set(observed)) == 1
    assert len({result["target_uri"] for result in results}) == 2
    assert len(event_nodes(writer, ctx)) == len(event_outbox(outbox, ctx)) == 2
    service.get_write_api()._pipeline.extract.assert_not_called()


def test_direct_async_api_rejects_owning_scope_before_dispatch_without_poisoning(database, ctx, monkeypatch):
    _, make = database
    _, writer, _, _, outbox = service_pair(make, ctx)
    import service.api as api_module

    def forbidden_thread(*args, **kwargs):
        pytest.fail("direct async API dispatched a foreign-thread SQL write")

    with writer._fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        monkeypatch.setattr(api_module.threading, "Thread", forbidden_thread)
        with pytest.raises(RuntimeError, match="separate service background handoff"):
            writer.commit_session_async([{"role": "user", "content": "An original source."}], ctx)
        assert not writer._tasks
        writer._pipeline.extract.assert_not_called()
        guard.ensure_owned()
        assert writer.write_memory(candidate(), ctx)["action"] == "create"
        guard.ensure_owned()
    assert len(event_nodes(writer, ctx)) == len(event_outbox(outbox, ctx)) == 1


def seed_lifecycle_public(service, ctx, *, message_id="fixed-source", content=None):
    service._cfg.after_turn_threshold = 10000
    service.wake_skill_sedimentation = Mock()
    service._flush_session_tool_usage_async = Mock()
    assert service.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": message_id, "role": "user", "content": content or "I repaired the receiver. " * 60}]})["ingested"]


def idle_lifecycle(service, ctx):
    return service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                   "timeoutSeconds": 10, "drainOutbox": False, "waitOutbox": False})


def trace_lifecycle_connections(writer, archive, outbox, observed):
    for name, store in {"state_node": writer._fs, "archive": archive, "outbox": outbox}.items():
        original = store._get_connection

        def get_owned(original=original, name=name):
            connection = original()
            guard = active_session_guard()
            if guard is not None:
                assert connection is guard.connection
                assert guard.thread_id == threading.get_ident()
                observed.append((guard, name, connection.get_backend_pid()))
            return connection

        store._get_connection = get_owned


def test_dispose_worker_persists_under_its_own_sql_guard(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    caller_thread = threading.get_ident()
    observed = []
    extract = writer._pipeline.extract.side_effect

    def owned_extract(*args, **kwargs):
        guard = active_session_guard()
        observed.append((threading.get_ident(), guard))
        assert guard is not None and guard.thread_id == threading.get_ident()
        return extract(*args, **kwargs)

    writer._pipeline.extract.side_effect = owned_extract
    result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "background_flush_dispatched"
    assert idle_lifecycle(service, ctx)["idle"]
    assert observed and observed[0][0] != caller_thread
    payload = json.loads(writer._fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert payload["buffer_messages"] == []
    restarted, _, manager2, _, _ = service_pair(make, ctx)
    with restarted._session_mutation_scope(ctx):
        assert manager2.get_or_create(ctx.session_id, ctx).messages == []


def test_delete_session_completes_retry_and_repeated_cleanup(database, ctx):
    from core.models import Role
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    cleanup = Mock()
    cleanup.delete_session_artifacts.side_effect = [OSError("temporary cleanup outage"), 0, 0]
    service._get_tool_result_cleanup_store = Mock(return_value=cleanup)
    request = {"sessionId": ctx.session_id, "_ctx": replace(ctx, role=Role.ADMIN)}
    assert service.delete_session(request)["retryable"]
    assert service.delete_session(request)["cleaned"]
    assert service.delete_session(request)["cleaned"]
    assert manager._recovery_owner_key(ctx) not in manager._extraction_recovery_present
    seed_lifecycle_public(service, ctx, message_id="new-source")
    assert manager.get_or_create(ctx.session_id, ctx).messages[0].id == "new-source"


def test_compose_window_survives_next_sql_request(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    manager._session_state_bridge_enabled = True
    seed_lifecycle_public(service, ctx, content="Remember the repair plan")

    def compose_window(params):
        state = service._get_session_state(ctx.session_id, ctx)
        state.compressed_text = "Receiver repair plan"
        state.turn_count_at_last_compress = 1
        manager.update_window_state(ctx.session_id, state)
        return {"ok": True}

    service._compose_impl = compose_window
    assert service.compose({"sessionId": ctx.session_id, "_ctx": ctx})["ok"]
    with service._session_mutation_scope(ctx):
        state = manager.get_or_create(ctx.session_id, ctx).window_state
        assert state.compressed_text == "Receiver repair plan"
        assert state.turn_count_at_last_compress == 1


def test_idle_eviction_saves_window_before_reload(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    manager._session_state_bridge_enabled = True
    with service._session_mutation_scope(ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
    buf = manager.get_or_create(ctx.session_id, ctx)
    buf.window_state.last_accessed_at = "2026-01-01T00:00:00+00:00"
    buf.window_state.compressed_text = "Persist this idle summary"
    assert manager.evict_idle_sessions(60, "2026-01-01T00:02:00+00:00", ctx) == [ctx.session_id]
    with service._session_mutation_scope(ctx):
        assert manager.get_or_create(ctx.session_id, ctx).window_state.compressed_text == "Persist this idle summary"


@pytest.mark.parametrize("content", ["Brief source", "Preserve this pending source. " * 60])
def test_dispose_checkpoint_recovers_after_storage_returns(database, ctx, content):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx, content=content)
    fs = writer._fs
    write_node = fs.write_node
    failed = []

    def recoverable_write(node, request_ctx):
        if node.category == "state" and json.loads(node.content)["buffer_messages"] == [] and not failed:
            failed.append(node.uri)
            raise OSError("checkpoint temporarily unavailable")
        return write_node(node, request_ctx)

    fs.write_node = recoverable_write
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    settled = idle_lifecycle(service, ctx)
    if len(content) // 4 >= 200:
        assert settled["reason"] == "lifecycle_recovery_pending"
    else:
        assert settled["idle"]
    assert failed
    assert [message.id for message in manager.get_or_create(ctx.session_id, ctx).messages] == ["fixed-source"]
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert idle_lifecycle(service, ctx)["idle"]
    payload = json.loads(fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert payload["buffer_messages"] == []


def test_idle_eviction_retries_acknowledged_save(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    manager._session_state_bridge_enabled = True
    with service._session_mutation_scope(ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
    buf = manager.get_or_create(ctx.session_id, ctx)
    buf.window_state.last_accessed_at = "2026-01-01T00:00:00+00:00"
    buf.window_state.compressed_text = "Retain until acknowledged"
    save = manager.save_session_state
    attempts = []

    def retry_save(*args):
        attempts.append(active_session_guard())
        return False if len(attempts) == 1 else save(*args)

    manager.save_session_state = retry_save
    assert manager.evict_idle_sessions(60, "2026-01-01T00:02:00+00:00", ctx) == []
    assert manager.get_or_create(ctx.session_id, ctx) is buf
    assert manager.evict_idle_sessions(60, "2026-01-01T00:02:00+00:00", ctx) == [ctx.session_id]
    assert all(attempts)
    with service._session_mutation_scope(ctx):
        assert manager.get_or_create(ctx.session_id, ctx).window_state.compressed_text == "Retain until acknowledged"


@pytest.mark.parametrize("pending_batch", [False, True])
def test_idle_eviction_preserves_another_workers_new_source(database, ctx, pending_batch):
    _, make = database
    first, writer, manager, _, _ = service_pair(make, ctx)
    with first._session_mutation_scope(ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
    manager.get_or_create(ctx.session_id, ctx).window_state.last_accessed_at = "2026-01-01T00:00:00+00:00"
    second, _, _, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(second, ctx, message_id="remote-source")
    if pending_batch:
        second._cfg.after_turn_threshold = 1
        second._write_api._pipeline.extract.side_effect = OSError("temporary model outage")
        assert second.after_turn(params(ctx))["ok"] is False
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    persisted = writer._fs.read_node(uri, ctx).content
    manager.evict_idle_sessions(60, "2026-01-01T00:02:00+00:00", ctx)
    assert writer._fs.read_node(uri, ctx).content == persisted
    with first._session_mutation_scope(ctx):
        restored = manager.get_or_create(ctx.session_id, ctx)
        assert restored.messages[0].id == "remote-source"
        assert bool(restored.pending_extraction_ids()) == pending_batch


def test_idle_observer_sees_another_worker_complete_recovery(database, ctx):
    _, make = database
    attempts = []

    def extract(**kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary model outage")
        return [replace(candidate(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]

    worker, _, manager, _, _ = service_pair(make, ctx, extract=extract)
    request = params(ctx)
    assert worker.after_turn(request)["ok"] is False
    observer, _, _, _, _ = service_pair(make, ctx)
    assert idle_lifecycle(observer, ctx)["reason"] == "extraction_recovery_pending"
    assert worker.after_turn(request)["ok"]
    assert idle_lifecycle(observer, ctx)["idle"]


@pytest.mark.parametrize("restart", [False, True])
def test_governed_request_retry_completes_pending_extraction(database, ctx, tmp_path, restart):
    from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.unit.server.test_memory_service_tool_results import _user_message_turn

    _, make = database
    attempts = []

    def extract(**kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary model outage")
        return [replace(candidate(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]

    def make_service():
        base, writer, manager, _, _ = service_pair(make, ctx, extract=extract)
        service = MemoryService(OgMemConfig(tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message", tool_result_single_chars=10,
            session_state_bridge_enabled=False, after_turn_threshold=1),
            tool_result_store=LocalToolResultArtifactStore(tmp_path))
        service._write_api, service._session_mgr = writer, manager
        service._generate_and_write_summary = Mock()
        service.drain_outbox_sync = Mock(return_value={})
        return service, manager

    service, manager = make_service()
    request = {**params(ctx), "messages": _user_message_turn("Receiver repair details. " * 20),
               "semanticTurnMode": "user_message"}
    assert service.after_turn(request)["ok"] is False
    assert manager.get_or_create(ctx.session_id, ctx).pending_extraction_ids()
    if restart:
        service, manager = make_service()
    assert service.after_turn(request)["ok"]
    assert manager.get_or_create(ctx.session_id, ctx).pending_extraction_ids() == set()
    assert len(attempts) == 2
    assert service.after_turn(request)["status"] == "duplicate"


def test_idle_wait_observes_local_worker_before_refresh(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    entered = threading.Event()
    release = threading.Event()
    extract = writer._pipeline.extract.side_effect

    def slow_extract(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return extract(*args, **kwargs)

    writer._pipeline.extract.side_effect = slow_extract
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert entered.wait(3)
    timer = threading.Timer(1.5, release.set)
    timer.start()
    try:
        assert idle_lifecycle(service, ctx)["idle"]
    finally:
        release.set()
        timer.join()


def test_lifecycle_session_alias_uses_durable_owner(database, ctx):
    _, make = database
    service, _, _, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    request = {"session_id": ctx.session_id, "accountId": ctx.account_id,
        "userId": ctx.user_id, "agentId": ctx.agent_id,
        "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False}
    assert service.wait_until_idle(request)["idle"]
    assert service.dispose(request)["disposed"]
    assert service.wait_until_idle(request)["idle"]


@pytest.mark.parametrize("resume_mode", ["prepare", "token", "expired", "new_message"])
def test_compaction_restart_preserves_extraction_archive_identity(database, ctx, resume_mode):
    _, make = database
    service, writer, manager, archive, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    prepared = service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    archive_id = prepared["archive_id"]
    assert archive_id in event_nodes(writer, ctx)[0].metadata["provenance_ids"][0]
    if resume_mode == "expired":
        with service._session_mutation_scope(ctx):
            batch = manager.get_or_create(ctx.session_id, ctx).runtime_extensions[EXTENSION_KEY]["batch"]
            batch["lifecycle"]["prepared"]["created_at"] = "2000-01-01T00:00:00+00:00"
            assert manager.save_session_state(ctx.session_id, ctx)
    elif resume_mode == "new_message":
        assert service.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx,
            "role": "user", "content": "I tested the repaired receiver."})["ok"]
    resumed, _, _, new_archive, _ = service_pair(make, ctx)
    request = {"sessionId": ctx.session_id, "_ctx": ctx}
    if resume_mode == "token":
        request.update(prepared=True, prepareToken=prepared["prepareToken"])
    result = resumed.compact(request)
    assert result["compacted"]
    assert new_archive.read_archive(ctx.session_id, archive_id, ctx).archive_id == archive_id


@pytest.mark.parametrize("explicit_source_id", [False, True])
def test_ingest_handoff_survives_worker_change(database, ctx, explicit_source_id):
    _, make = database
    service, _, _, _, _ = service_pair(make, ctx)
    service._cfg.after_turn_threshold = 10000
    request = params(ctx)
    assert service.ingest(request)["ingested"]
    resumed, _, manager, _, _ = service_pair(make, ctx)
    resumed._cfg.after_turn_threshold = 10000
    handoff = {key: value for key, value in request.items() if key != "clientRequestId"}
    if explicit_source_id:
        handoff["messages"][0]["id"] = service.get_session_manager().get_or_create(ctx.session_id, ctx).messages[0].id
    assert resumed.after_turn(handoff)["ok"]
    buf = manager.get_or_create(ctx.session_id, ctx)
    assert len(buf.messages) == 1
    assert not buf.ingest_handoff_ids
    request["messages"][0].pop("id", None)
    assert resumed.after_turn({**request, "clientRequestId": "next-turn"})["ok"]
    assert len(buf.messages) == 2


def test_first_checkpoint_retry_completes_after_storage_recovers(database, ctx, monkeypatch):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    fs = writer._fs
    write_node = fs.write_node
    failed = []

    def fail_first_checkpoint(node, request):
        if node.category == "state" and not failed:
            failed.append(True)
            raise OSError("temporary checkpoint outage")
        return write_node(node, request)

    monkeypatch.setattr(fs, "write_node", fail_first_checkpoint)
    request = params(ctx)
    first = service.after_turn(request)
    assert first["status"] == "failed"
    assert first["retryable"]
    assert manager.get_or_create(ctx.session_id, ctx).messages
    retried = service.after_turn(request)
    assert retried["status"] == "completed"
    writer._pipeline.extract.assert_called_once()
    assert len(event_nodes(writer, ctx)) == 1
    resumed, _, new_manager, _, _ = service_pair(make, ctx)
    assert resumed.after_turn(request)["status"] == "duplicate"
    assert not new_manager.get_or_create(ctx.session_id, ctx).messages
@pytest.mark.parametrize("entry", ["compact", "prepare_compaction", "dispose"])
def test_union_typed_partial_retry_fixed_source_new_tail_and_owning_connection(database, ctx, entry):
    _, make = database
    entries = [candidate(), candidate("entity")]
    initial, writer, manager, archive, outbox = service_pair(make, ctx, lambda **kwargs: entries)
    observed = []
    trace_lifecycle_connections(writer, archive, outbox, observed)
    seed_lifecycle_public(initial, ctx)
    reject_entity_once(writer)
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    if entry == "dispose":
        assert initial.dispose(request)["status"] == "background_flush_dispatched"
        assert idle_lifecycle(initial, ctx)["reason"] == "lifecycle_recovery_pending"
    else:
        with pytest.raises(RuntimeError, match="compaction extraction write failed"):
            getattr(initial, entry)(request)
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch["lifecycle"]["kind"] == ("dispose" if entry == "dispose" else "compaction")
    assert batch["message_ids"] == ["fixed-source"]
    accepted = batch["slots"]["chunk:0"]["candidates"]
    original_event, = event_nodes(writer, ctx)
    original_event_outbox = len(event_outbox(outbox, ctx))
    source_archive_id = batch["archive_id"]
    assert manager.get_or_create(ctx.session_id).active_operation_count == 0
    assert not manager.get_or_create(ctx.session_id).dispose_in_progress

    # A different service appends a genuinely independent tail. Its authoritative
    # refresh retains the fixed lifecycle source and accepted candidates.
    restarted, writer2, manager2, archive2, outbox2 = service_pair(
        make, ctx, lambda **kwargs: pytest.fail("accepted typed lifecycle was re-extracted"))
    trace_lifecycle_connections(writer2, archive2, outbox2, observed)
    tail = "An independent amplifier follow-up is pending."
    assert restarted.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx,
        "role": "user", "content": tail})["ok"]
    persisted = json.loads(writer2._fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    persisted_batch = persisted["runtime_extensions"][EXTENSION_KEY]["batch"]
    assert persisted_batch["archive_id"] == source_archive_id
    assert persisted_batch["slots"]["chunk:0"]["candidates"] == accepted
    assert persisted_batch["message_ids"] == ["fixed-source"]
    if entry == "dispose":
        restarted.dispose(request)
        assert idle_lifecycle(restarted, ctx)["idle"] is True
    else:
        prepared = restarted.prepare_compaction(request)
        assert prepared["prepareToken"]
        assert restarted.compact({**request, "prepareToken": prepared["prepareToken"]})["compacted"] is True
        archived = archive2.read_archive(ctx.session_id, source_archive_id, ctx)
        assert [message["id"] for message in archived.messages] == ["fixed-source"]
    assert event_nodes(writer2, ctx) == [original_event]
    assert len(event_outbox(outbox2, ctx)) == original_event_outbox
    entity_directory = writer2._writer._identity_resolver.uris.get_directory_uri("entity", ctx)
    assert len(writer2._fs.list_children(entity_directory, ctx)) == 1
    buffer = manager2.get_or_create(ctx.session_id)
    assert [message.content for message in buffer.messages] == [tail]
    assert buffer.extraction_watermark == 0 and not buffer.pending_extraction_ids()
    assert not buffer.dispose_in_progress and buffer.active_operation_count == 0
    writer2._pipeline.extract.assert_not_called()
    for guard in {entry[0] for entry in observed}:
        assert len({pid for owner, _, pid in observed if owner is guard}) == 1
    if entry != "dispose":
        assert any(name == "archive" for _, name, _ in observed)


def test_union_compaction_lost_connection_stops_old_worker_after_new_owner_finishes(database, ctx):
    dsn, make = database
    reached, release = threading.Event(), threading.Event()
    owner_pids = []

    def paused(**kwargs):
        guard = active_session_guard()
        owner_pids.append(guard.raw.get_backend_pid())
        reached.set()
        assert release.wait(10)
        return [candidate()]

    old, old_writer, old_manager, _, _ = service_pair(make, ctx, paused)
    seed_lifecycle_public(old, ctx)
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(old.compact, request)
        try:
            assert reached.wait(10)
            control = psycopg2.connect(dsn)
            try:
                with control.cursor() as cursor:
                    cursor.execute("SELECT pg_terminate_backend(%s)", (owner_pids[0],))
                    assert cursor.fetchone()[0]
                control.commit()
            finally:
                control.close()
            new, writer, manager, archive, outbox = service_pair(make, ctx)
            assert new.compact(request)["compacted"] is True
        finally:
            release.set()
        assert pending.result(timeout=10)["status"] == "needs_confirmation"
    assert len(event_nodes(writer, ctx)) == len(event_outbox(outbox, ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert not manager.get_or_create(ctx.session_id).messages
    assert old_manager.get_or_create(ctx.session_id).pending_extraction_ids()
    assert old_writer._pipeline.extract.call_count == 1


def test_union_dispose_real_busy_worker_scope_releases_lease_and_preserves_pending(database, ctx):
    _, make = database
    service, writer, manager, _, outbox = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    reached, release, done = threading.Event(), threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    original_scope = service._session_mutation_scope
    original_worker = service._run_dispose_background

    @contextmanager
    def pause_worker(request, **kwargs):
        if threading.get_ident() != main_thread:
            reached.set()
            assert release.wait(10)
        with original_scope(request, **kwargs) as guard:
            yield guard

    def observed_worker(*args, **kwargs):
        try:
            return original_worker(*args, **kwargs)
        finally:
            done.set()

    service._session_mutation_scope = pause_worker
    service._run_dispose_background = observed_worker
    assert service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "background_flush_dispatched"
    assert reached.wait(10)
    control, _, _ = make()
    try:
        with control.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            release.set()
            assert done.wait(5)  # The real one-second SQL acquisition expires.
    finally:
        release.set()
    buffer = manager.get_or_create(ctx.session_id)
    assert buffer.active_operation_count == 0 and not buffer.dispose_in_progress
    assert buffer.pending_extraction_ids() == {"fixed-source"}
    assert event_nodes(writer, ctx) == [] and event_outbox(outbox, ctx) == []
    writer._pipeline.extract.assert_not_called()
    service._session_mutation_scope = original_scope
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert idle_lifecycle(service, ctx)["idle"] is True
    assert len(event_nodes(writer, ctx)) == len(event_outbox(outbox, ctx)) == 1


def test_union_dispose_legacy_missing_context_failure_releases_lease_before_scope(database, ctx):
    _, make = database
    service, writer, manager, _, outbox = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    from server.memory_service import _DisposeOperationLease
    buffer = manager.begin_managed_operation(ctx.session_id, ctx)
    operation = _DisposeOperationLease(buffer)
    operation.claim_dispose()
    operation.worker_owned = True
    service.build_context = Mock(side_effect=ValueError("controlled legacy context failure"))
    service._run_dispose_background(manager, ctx.session_id, {}, None, buffer,
                                    list(buffer.messages), operation)
    operation.release()
    assert buffer.active_operation_count == 0 and not buffer.dispose_in_progress
    assert [message.id for message in buffer.messages] == ["fixed-source"]
    writer._pipeline.extract.assert_not_called()
    assert event_nodes(writer, ctx) == [] and event_outbox(outbox, ctx) == []


def test_union_deferred_dispose_waits_outside_guard_and_owns_remaining_new_tail(database, ctx):
    _, make = database
    main_thread = threading.get_ident()
    reached, release = threading.Event(), threading.Event()
    source_rounds = []

    def extract(messages, request, **kwargs):
        guard = active_session_guard()
        guard.ensure_owned()
        source_rounds.append(([message["id"] for message in messages], guard.raw.get_backend_pid()))
        category = "entity" if messages[0]["id"] == "independent-tail" else "event"
        return [replace(candidate(category), provenance_ids=[
            f"prov:1:archive:{kwargs['archive_id']}:{messages[0]['id']}"])]

    service, writer, manager, _, outbox = service_pair(make, ctx)
    writer._pipeline.extract.side_effect = extract
    service.wake_skill_sedimentation = Mock()
    service._flush_session_tool_usage_async = Mock()
    original_scope = service._session_mutation_scope

    @contextmanager
    def pause_first_child(request, **kwargs):
        if threading.get_ident() != main_thread and not reached.is_set():
            reached.set()
            assert release.wait(10)
        with original_scope(request, **kwargs) as guard:
            yield guard

    service._session_mutation_scope = pause_first_child
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
        {"id": "ordinary-source", "role": "user", "content": "I repaired the receiver. " * 60}]}
    assert service.after_turn(request)["status"] == "processing"
    assert reached.wait(10)
    try:
        result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert result["status"] == "dispose_deferred", result
        other, _, _, _, _ = service_pair(make, ctx)
        # Success proves that the waiting dispose thread has no session SQL lock.
        assert other.ingest({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [
            {"id": "independent-tail", "role": "user", "content": "A separate amplifier follow-up. " * 60}]})["ingested"]
    finally:
        release.set()
    assert idle_lifecycle(service, ctx)["idle"] is True
    assert not manager.has_session(ctx.session_id)
    assert [ids for ids, _ in source_rounds] == [["ordinary-source"], ["independent-tail"]]
    assert len({pid for _, pid in source_rounds}) == 2
    assert len(event_nodes(writer, ctx)) == len(event_outbox(outbox, ctx)) == 1


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("remove", [False, True])
def test_union_uncertain_typed_archive_recovery_requires_exact_original_under_guard(database, ctx, remove, required):
    _, make = database
    service, writer, manager, archive, _ = service_pair(make, ctx)
    # Both the configured required SQL path and actual optional SQL stores must
    # preserve uncertainty instead of acknowledging an in-memory fallback.
    manager._archive_store_required = required
    seed_lifecycle_public(service, ctx)
    observed = []
    trace_lifecycle_connections(writer, archive, writer._writer._outbox_store, observed)
    original_write = archive.write_archive

    def commit_then_lose_ack(*args, **kwargs):
        result = original_write(*args, **kwargs)
        assert result.success
        return ArchiveWriteResult(archive_id=result.archive_id, session_id=result.session_id,
                                  uri=result.uri, success=False, error="lost acknowledgement after real SQL commit")

    archive.write_archive = commit_then_lose_ack
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    assert service.compact(request)["status"] == "failed"
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch["lifecycle"]["archive"]["status"] == "writing"
    original_archive = archive.read_archive(ctx.session_id, batch["archive_id"], ctx)
    assert original_archive is not None
    if remove:
        with writer._fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            assert archive.delete_archive(ctx.session_id, batch["archive_id"], ctx)
    restarted, writer2, manager2, archive2, outbox2 = service_pair(
        make, ctx, lambda **kwargs: pytest.fail("accepted archive recovery re-extracted"))
    trace_lifecycle_connections(writer2, archive2, outbox2, observed)
    manager2._write_archive = Mock(side_effect=AssertionError("uncertain archive cannot be recreated"))
    result = restarted.compact(request)
    if remove:
        assert result["status"] == "needs_confirmation", result
        assert archive2.read_archive(ctx.session_id, batch["archive_id"], ctx) is None
        assert manager2.get_or_create(ctx.session_id).pending_extraction_ids()
    else:
        assert result["compacted"] is True
        assert archive2.read_archive(ctx.session_id, batch["archive_id"], ctx) == original_archive
        assert not manager2.get_or_create(ctx.session_id).pending_extraction_ids()
    assert len(event_nodes(writer2, ctx)) == len(event_outbox(outbox2, ctx)) == 1
    manager2._write_archive.assert_not_called()
    writer2._pipeline.extract.assert_not_called()
    for guard in {entry[0] for entry in observed}:
        assert len({pid for owner, _, pid in observed if owner is guard}) == 1


@pytest.mark.parametrize("entry", ["prepare_compaction", "dispose"])
def test_union_uncertain_typed_memory_recovery_reuses_receipt_with_no_extraction(database, ctx, entry, monkeypatch):
    _, make = database
    service, writer, manager, _, outbox = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    original_write = type(writer._fs).write_node_with_outbox
    original_read = writer._fs.read_node
    unread_receipts = set()

    def recover_receipt_after_reconnect(uri, request):
        if uri in unread_receipts:
            unread_receipts.remove(uri)
            raise ConnectionError("receipt read unavailable after connection loss")
        return original_read(uri, request)

    monkeypatch.setattr(writer._fs, "read_node", recover_receipt_after_reconnect)

    def commit_then_lose_ack(fs, node, request, outbox_event=None):
        original_write(fs, node, request, outbox_event)
        if fs is writer._fs and node.category == "event":
            unread_receipts.add(node.uri)
            raise RuntimeError("lost acknowledgement after real SQL node/receipt/outbox commit")

    monkeypatch.setattr(type(writer._fs), "write_node_with_outbox", commit_then_lose_ack)
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    result = getattr(service, entry)(request)
    if entry == "dispose":
        assert result["status"] == "background_flush_dispatched"
        assert idle_lifecycle(service, ctx)["status"] == "needs_confirmation"
    else:
        assert result["status"] == "needs_confirmation"
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch["slots"]["chunk:0"]["operations"]["0"]["status"] == "writing"
    original_event, = event_nodes(writer, ctx)
    assert len(event_outbox(outbox, ctx)) == 1
    restarted, writer2, manager2, _, outbox2 = service_pair(
        make, ctx, lambda **kwargs: pytest.fail("uncertain accepted memory was re-extracted"))
    if entry == "dispose":
        restarted.dispose(request)
        assert idle_lifecycle(restarted, ctx)["idle"] is True
    else:
        prepared = restarted.prepare_compaction(request)
        assert restarted.compact({**request, "prepareToken": prepared["prepareToken"]})["compacted"] is True
    assert event_nodes(writer2, ctx) == [original_event]
    assert len(event_outbox(outbox2, ctx)) == 1
    assert not manager2.has_session(ctx.session_id) if entry == "dispose" else not manager2.get_or_create(ctx.session_id).messages
    writer2._pipeline.extract.assert_not_called()


@pytest.mark.parametrize("use_token", [False, True])
def test_compaction_resumes_pre_lifecycle_preparation(database, ctx, use_token):
    from session.session_manager import generate_archive_id

    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    archive_id = generate_archive_id()
    with service._session_mutation_scope(ctx):
        buf = manager.get_or_create(ctx.session_id, ctx)
        source = [{"id": message.id, "role": message.role, "content": message.content} for message in buf.messages]
        writer.commit_session(source, ctx, archive_id=archive_id)
        buf.extraction_watermark = len(buf.messages)
        token = manager.issue_compaction_prepare_token(ctx.session_id, archive_id)
        assert manager.save_session_state(ctx.session_id, ctx)
    original_event, = event_nodes(writer, ctx)
    assert archive_id in original_event.metadata["provenance_ids"][0]
    resumed, writer2, _, archive, _ = service_pair(
        make, ctx, lambda **kwargs: pytest.fail("prepared source was extracted again"))
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    if use_token:
        request.update(prepared=True, prepareToken=token)
    assert resumed.compact(request)["compacted"]
    assert archive.read_archive(ctx.session_id, archive_id, ctx).archive_id == archive_id
    assert event_nodes(writer2, ctx) == [original_event]


@pytest.mark.parametrize("retry_finish", [False, True])
def test_migrated_compaction_archives_assistant_tail_separately(database, ctx, retry_finish):
    from session.session_manager import generate_archive_id

    _, make = database
    service, _, manager, archive, _ = service_pair(make, ctx)
    seed_lifecycle_public(service, ctx)
    archive_id = generate_archive_id()
    with service._session_mutation_scope(ctx):
        buf = manager.get_or_create(ctx.session_id, ctx)
        buf.extraction_watermark = len(buf.messages)
        manager.issue_compaction_prepare_token(ctx.session_id, archive_id)
        assert manager.save_session_state(ctx.session_id, ctx)
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    prepared = service.prepare_compaction(request)
    assert service.add_session_message({**request, "role": "assistant", "content": "Repair confirmed."})["ok"]
    if retry_finish:
        save = manager.save_session_state

        def save_pending(session_id, request_ctx):
            if EXTENSION_KEY not in manager.get_or_create(session_id, request_ctx).runtime_extensions:
                return False
            return save(session_id, request_ctx)

        manager.save_session_state = save_pending
        assert service.compact({**request, "prepared": True, "prepareToken": prepared["prepareToken"]})["retryable"]
        assert buf.compaction_prepare_archive_id == archive_id
        manager.save_session_state = save
    assert service.compact({**request, "prepared": True, "prepareToken": prepared["prepareToken"]})["compacted"]

    resumed, _, manager2, archive2, _ = service_pair(make, ctx)
    assert resumed.compact(request)["compacted"]
    entries = archive2.list_archives(ctx.session_id, ctx)
    assert len(entries) == 2
    assert {message["content"] for entry in entries
            for message in archive2.read_archive(ctx.session_id, entry.archive_id, ctx).messages} == {
        "I repaired the receiver. " * 60, "Repair confirmed."}
    assert manager2.get_or_create(ctx.session_id, ctx).messages == []
    assert archive.read_archive(ctx.session_id, archive_id, ctx).messages[0]["id"] == "fixed-source"


@pytest.mark.parametrize("use_token", [False, True])
def test_assistant_compaction_recovers_with_cold_writer(database, ctx, use_token):
    dsn, make = database
    service, _, manager, _, _ = service_pair(make, ctx)
    request = {"sessionId": ctx.session_id, "_ctx": ctx, "shortTermIndexMode": "off"}
    assert service.ingest({**request, "messages": [
        {"id": "assistant-source", "role": "assistant", "content": "Repair confirmed."}]})["ingested"]
    prepared = service.prepare_compaction(request)
    archive_id = prepared["archive_id"]

    resumed, writer, manager2, archive, _ = service_pair(make, ctx)
    manager2._get_context_fs = lambda: writer._fs
    resumed._write_api = None
    resumed._cfg.storage_backend = "sql"
    resumed._cfg.sql_connection_string = dsn
    resumed._cfg.sql_dialect = "postgresql"
    if use_token:
        renewed = resumed.prepare_compaction(request)
        request.update(prepared=True, prepareToken=renewed["prepareToken"])
    assert resumed.compact(request)["compacted"]
    assert archive.read_archive(ctx.session_id, archive_id, ctx).messages == (
        manager.get_or_create(ctx.session_id, ctx).runtime_extensions[EXTENSION_KEY]["batch"]["snapshot"])
    assert manager2.get_or_create(ctx.session_id, ctx).messages == []
