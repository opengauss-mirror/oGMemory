"""A durable graph wakeup waits through repeated projection polling."""

from datetime import UTC, datetime, timedelta

from core.models import RelationEdge
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def test_owned_graph_wakeup_defers_and_completes_with_original_retry_budget(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    _graph_sql(fs, ctx,
        "UPDATE outbox_events SET retry_count=2 WHERE account_id=%s", (ctx.account_id,))
    uri, original = outbox.list_pending(ctx.account_id)[0]
    for step in range(worker._config.max_retry + 2):
        uri, event = outbox.list_pending(ctx.account_id)[0]
        assert outbox.try_acquire(event, uri, f"worker-{step}")
        due = datetime.now(UTC) + timedelta(seconds=1)
        outbox.defer(event, uri, due)
        persisted = outbox.get_event(ctx.account_id, original.event_id)
        assert (persisted.status, persisted.retry_count) == ("PENDING", 2)
        assert persisted.payload == original.payload
        assert datetime.fromisoformat(persisted.next_retry_at) == due
        _graph_sql(fs, ctx,
            "UPDATE outbox_events SET next_retry_at=NOW() WHERE account_id=%s", (ctx.account_id,))
    worker.run_once(outbox, [ctx.account_id], worker_id="final-worker")
    completed = outbox.get_event(ctx.account_id, original.event_id)
    assert (completed.status, completed.retry_count) == ("DONE", 2)
    assert store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)
