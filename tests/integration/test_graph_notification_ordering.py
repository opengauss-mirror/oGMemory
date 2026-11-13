"""SQL graph wakeups and ordinary projections advance in their ordered lanes."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from core.models import OutboxEvent
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from tests.integration.test_sql_outbox_claim_ownership import (
    publish,
    sql_runtime as sql_runtime,
    stats,
)


def graph_wakeup(store, ctx, uri):
    event = OutboxEvent(uuid4().hex, "PROJECT_GRAPH", uri,
                        {"edge_id": uuid4().hex}, "PENDING")
    store._insert_event(event, ctx.account_id)
    return event


def complete_wakeup(store, claimed):
    uri, event = claimed
    with store.processing_lock(uri):
        store.ensure_processing_owned(event, uri)
        store.mark_done(event, uri)


def test_node_versions_complete_during_graph_wait_then_graph_resumes(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    uri = f"ctx://{ctx.account_id}/users/reader/memories/entities/shared"
    first_graph = graph_wakeup(store, ctx, uri)
    second_graph = graph_wakeup(store, ctx, uri)
    owned = store.claim_batch("graph-owner")
    assert [event.event_id for _, event in owned] == [first_graph.event_id]
    with store.processing_lock(uri):
        store.defer(owned[0][1], uri, datetime.now(UTC) + timedelta(minutes=1))

    first, _ = publish(store, ctx, "shared", "First node revision")
    second, latest = publish(store, ctx, "shared", "Latest node revision")
    index = InMemoryVectorIndex(dimension=16)
    worker = OutboxWorker(index, MockEmbedder(16))
    for expected in (first, second):
        claimed = store.claim_batch("node-owner")
        assert [event.event_id for _, event in claimed] == [expected.event_id]
        result = stats()
        worker._process_claimed_events(claimed, store, result, "node-owner")
        assert result["succeeded"] == 1
        assert store.get_event(ctx.account_id, expected.event_id).status == "DONE"
    assert index._records[latest.id].text == "Latest node revision"

    conn = store._get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE outbox_events SET next_retry_at=NOW() "
                        "WHERE account_id=%s AND event_id=%s",
                        (ctx.account_id, first_graph.event_id))
        conn.commit()
    finally:
        store._return_connection(conn)
    for expected in (first_graph, second_graph):
        claimed = store.claim_batch("graph-resumed")
        assert [event.event_id for _, event in claimed] == [expected.event_id]
        complete_wakeup(store, claimed[0])
        assert store.get_event(ctx.account_id, expected.event_id).status == "DONE"
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_graph_wakeups_follow_ordinary_and_graph_predecessors(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    ordinary, _ = publish(store, ctx, "ordered", "Node before graph publication")
    first_graph = graph_wakeup(store, ctx, ordinary.uri)
    second_graph = graph_wakeup(store, ctx, ordinary.uri)
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16))
    claimed = store.claim_batch("node-owner")
    assert [event.event_id for _, event in claimed] == [ordinary.event_id]
    result = stats()
    worker._process_claimed_events(claimed, store, result, "node-owner")
    assert result["succeeded"] == 1
    for expected in (first_graph, second_graph):
        claimed = store.claim_batch("graph-owner")
        assert [event.event_id for _, event in claimed] == [expected.event_id]
        complete_wakeup(store, claimed[0])
    assert [store.get_event(ctx.account_id, event.event_id).status
            for event in (ordinary, first_graph, second_graph)] == ["DONE"] * 3
