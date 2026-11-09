"""Database deadlines wake the legacy worker for durable graph completion."""

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg2.extensions import TRANSACTION_STATUS_IDLE

from core.models import OutboxEvent, RelationEdge
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def test_future_pending_delay_follows_selected_accounts_and_returns_idle_connection(graph_db, monkeypatch):
    ctx, fs, _, _, _, outbox, _, nodes, _ = graph_db
    other_account = ctx.account_id + "-other"
    for account, seconds in [(ctx.account_id, 30), (ctx.account_id, 20), (other_account, 10)]:
        event = OutboxEvent(uuid4().hex, "PROJECT_GRAPH",
                            f"ctx://{account}/users/alice/memories/entity/future", {}, "PENDING",
                            next_retry_at=(datetime.now(UTC) + timedelta(seconds=seconds)).isoformat())
        outbox._insert_event(event, account)

    returned_states = []
    return_connection = outbox._return_connection

    def observe_return(connection):
        returned_states.append(connection.get_transaction_status())
        return_connection(connection)

    monkeypatch.setattr(outbox, "_return_connection", observe_return)
    own_delay = outbox.future_pending_delay([ctx.account_id])
    other_delay = outbox.future_pending_delay([other_account])
    combined_delay = outbox.future_pending_delay([ctx.account_id, other_account])
    assert 18 < own_delay <= 20
    assert 8 < combined_delay <= other_delay <= 10
    assert returned_states == [TRANSACTION_STATUS_IDLE] * 3
    _graph_sql(fs, ctx, "UPDATE outbox_events SET status='DONE' WHERE account_id=%s", (ctx.account_id,))
    assert outbox.future_pending_delay([ctx.account_id]) is None
    assert outbox.future_pending_delay([]) is None
    _graph_sql(fs, ctx, "UPDATE outbox_events SET status='DONE' WHERE account_id=%s", (other_account,))


def test_one_second_graph_defer_finishes_in_native_legacy_loop(graph_db, monkeypatch, record_property):
    ctx, fs, index, relations, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    relations.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    _graph_sql(fs, ctx, "UPDATE outbox_events SET retry_count=2 WHERE account_id=%s", (ctx.account_id,))
    uri, event = outbox.list_pending(ctx.account_id)[0]
    assert outbox.try_acquire(event, uri, "deadline-owner")
    outbox.defer(event, uri, datetime.now(UTC) + timedelta(seconds=1))
    assert projections.drain_sync(ctx.account_id, worker._embedder, profile) == 1
    assert relations.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)

    outbox.supports_batch_claim = False
    service = MemoryService(config=OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        sql_dialect=outbox.sql_dialect.name, sql_connection_string=os.environ["GRAPH_TEST_DSN"],
        directory_summary_enabled=False,
    ))
    service._default_account_id = ctx.account_id
    waits = []

    class ObservedStop(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return super().wait(timeout)

    service._outbox_stop = ObservedStop()
    finished = threading.Event()
    run_once = worker.run_once

    def observe_run(**kwargs):
        stats = run_once(**kwargs)
        if outbox.get_event(ctx.account_id, event.event_id).status == "DONE":
            finished.set()
            service._outbox_stop.set()
        return stats

    monkeypatch.setattr(worker, "run_once", observe_run)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda context_fs: outbox)
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: index)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: worker._embedder)
    monkeypatch.setattr(service, "_graph_projection_components", lambda: (projections, profile))
    monkeypatch.setattr("index.outbox_worker.OutboxWorker", lambda **kwargs: worker)
    started = time.perf_counter()
    service._start_outbox_worker()
    try:
        assert finished.wait(3)
    finally:
        service._outbox_stop.set()
        service._outbox_thread.join(timeout=1)
    elapsed = time.perf_counter() - started
    completed = outbox.get_event(ctx.account_id, event.event_id)
    assert (completed.status, completed.retry_count) == ("DONE", 2)
    assert len(waits) == 1
    assert 0 < waits[0] <= 1
    assert elapsed < 3
    assert outbox.count_unfinished(ctx.account_id)["total"] == 0
    record_property("scheduled_wait_seconds", waits[0])
    record_property("completion_elapsed_seconds", elapsed)
