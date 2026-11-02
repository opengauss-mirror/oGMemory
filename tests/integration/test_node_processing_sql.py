"""Node processing commits a read-bound update under the SQL session fence."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import psycopg2
import pytest

from fs.sql_adapter import sql_context_fs as sql_module
from fs.sql_adapter.session_mutation import active_session_guard
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from tests.integration.test_session_mutation_sql import database, ctx, service_pair
from tests.integration.test_extraction_batch_recovery_sql import category_nodes
from tests.unit.server.test_extraction_batch_recovery import params
from tests.unit.server.test_memory_card_recovery import enable
from tests.unit.server.test_extraction_retry_task import ready
from tests.unit.extraction.test_node_processing import NodeLLM, card, operation


def test_synchronous_node_processing_waits_for_busy_user_directory(database, ctx):
    """A different session's topic commit must not fail after one second."""
    _, make = database
    holder_fs, _, _ = make()
    service, writer, manager, archive, _ = service_pair(
        make, ctx, extract=lambda **kwargs: [replace(
            card(), provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"]
        )],
    )
    writer._node_processing_enabled = True
    writer._llm = NodeLLM(lambda _: [operation([0])])
    service._cfg.node_processing_enabled = True
    directory = writer._writer._identity_resolver.uris.get_directory_uri("entity", ctx).rstrip("/")
    holder_ctx = replace(ctx, session_id=ctx.session_id + "-holder")
    entered, release = threading.Event(), threading.Event()

    def hold_directory():
        with holder_fs.memory_identity_lock(directory, holder_ctx):
            entered.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(hold_directory)
        assert entered.wait(5)
        write = executor.submit(service.after_turn, params(ctx))
        try:
            time.sleep(1.2)
            assert not write.done()
        finally:
            release.set()
        holder.result(timeout=5)
        result = write.result(timeout=10)

    assert result["ok"] is True and result["status"] == "completed"
    assert len(category_nodes(writer, "entity", ctx)) == 1
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1


def test_node_processing_sql_reconnect_reuses_drafts(database, ctx):
    _, make = database
    service, writer, manager, _, _ = service_pair(make, ctx, extract=lambda **kwargs: [card()])
    enable(service, writer)
    writer._node_processing_enabled = True
    writer._llm = NodeLLM(lambda _: [operation([0], action="defer")])
    queued = service.after_turn(params(ctx))
    assert queued["status"] == "queued"
    restarted, writer, manager, archive, outbox = service_pair(make, ctx, extract=lambda **kwargs: [card()])
    enable(restarted, writer)
    writer._node_processing_enabled = True
    writer._llm = NodeLLM(lambda _: [operation([0])])
    restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
    with restarted._session_mutation_scope(ctx):
        ready(manager, ctx)
    restarted._extraction_retry.run_once(ctx, queued["recovery_task"]["id"])
    assert len(category_nodes(writer, "entity", ctx)) == 1
    assert writer._pipeline.extract.call_count == 0
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("with_event", [False, True])
def test_concurrent_public_commits_share_identity_connection(database, ctx, monkeypatch, background, with_event):
    _, make = database
    fs, outbox, _ = make()
    slots = threading.BoundedSemaphore(8)
    monkeypatch.setattr(sql_module, "_TOOL_OUTPUT_LOCK_CONNECTION_SLOTS", slots)
    monkeypatch.setattr(sql_module, "_MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS", 3.0)
    first_locks = threading.Barrier(8)
    thread_state = threading.local()
    original_lock = SQLContextFS.memory_identity_lock

    @contextmanager
    def coordinated_lock(self, directory, request_ctx):
        with original_lock(self, directory, request_ctx):
            if not getattr(thread_state, "entered", False):
                thread_state.entered = True
                first_locks.wait(10)
            yield

    monkeypatch.setattr(SQLContextFS, "memory_identity_lock", coordinated_lock)
    writer = MemoryWriteAPI(fs, NodeLLM(lambda _: [operation([0])]), outbox,
                            node_processing_enabled=True, card_format_enabled=True,
                            directory_summary_enabled=False)
    drafts = [card(), card("preference", category="preference", content="# Necklace\n\n- Prefers a heart charm.")]
    if with_event:
        drafts.append(card("inspection", category="event", content="The technician inspected the necklace."))
    writer._pipeline.extract = lambda *args, **kwargs: [
        replace(draft, provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:source"]) for draft in drafts]
    contexts = [replace(ctx, user_id=f"reader-{index}", session_id=f"session-{index}") for index in range(8)]

    def commit(request_ctx):
        assert active_session_guard() is None
        if not background:
            return writer.commit_session([], request_ctx)
        task = writer.commit_session_async([], request_ctx)
        until = time.monotonic() + 15
        while time.monotonic() < until:
            state = writer.get_task_status(task)
            if state["status"] in {"completed", "failed"}:
                assert state["status"] == "completed", state
                return state["result"]
            time.sleep(.01)
        pytest.fail(f"commit task did not complete: {writer.get_task_status(task)}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(commit, contexts))
    assert all(result["writes_completed"] == len(drafts) for result in results)
    for request_ctx in contexts:
        for category in {draft.category for draft in drafts}:
            assert len(category_nodes(writer, category, request_ctx)) == 1
    with ExitStack() as stack:
        for _ in contexts:
            assert slots.acquire(timeout=.1)
            stack.callback(slots.release)


def test_identity_connection_reentry_and_exception_release(database, ctx, monkeypatch):
    dsn, make = database
    first, _, _ = make()
    second, _, _ = make()
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(sql_module, "_TOOL_OUTPUT_LOCK_CONNECTION_SLOTS", slots)
    monkeypatch.setattr(sql_module, "_MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS", 3.0)
    root = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/"
    entity, preference, event = [root + category for category in ("entities", "preferences", "events")]
    keys = [sql_module._resource_mutation_lock_key(ctx.account_id, directory)
            for directory in (entity, preference, event)]
    with first.memory_identity_lock(entity, ctx):
        with second.memory_identity_lock(entity, ctx):
            with pytest.raises(ValueError, match="interrupted selection"):
                with second.memory_identity_lock(preference, ctx):
                    raise ValueError("interrupted selection")
        with second.memory_identity_lock(event, ctx):
            assert active_session_guard() is None
            with psycopg2.connect(dsn) as observer:
                with observer.cursor() as cursor:
                    cursor.execute("SELECT pg_try_advisory_lock(%s)", (keys[1],))
                    assert cursor.fetchone() == (True,)
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (keys[1],))
    with second.memory_identity_lock(preference, ctx):
        with first.memory_identity_lock(event, ctx):
            assert active_session_guard() is None
    assert slots.acquire(timeout=.1)
    slots.release()
    with psycopg2.connect(dsn) as observer:
        with observer.cursor() as cursor:
            for key in keys:
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (key,))
                assert cursor.fetchone() == (True,)
                cursor.execute("SELECT pg_advisory_unlock(%s)", (key,))
