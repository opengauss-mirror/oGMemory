"""Legacy SQL polling discovers and completes an expired write claim."""

from core.models import IndexRecord
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def test_expired_write_claim_is_recovered_by_legacy_polling(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    node = fs.read_node(nodes[0].uri, ctx)
    node.content = "Content published after the original worker stopped"
    node.metadata["_expected_version"] = node.metadata["version"]
    fs.write_node(node, ctx)
    event = outbox.register_write(node, ctx)
    assert outbox.try_acquire(event, event.uri, "previous-worker")
    _graph_sql(fs, ctx,
        "UPDATE outbox_events SET locked_at=NOW()-INTERVAL '6 minutes' "
        "WHERE account_id=%s AND event_id=%s", (ctx.account_id, event.event_id))
    outbox.supports_batch_claim = False

    discovered = outbox.list_pending(ctx.account_id)
    assert [(uri, item.event_id, item.status) for uri, item in discovered] == [
        (node.uri, event.event_id, "PROCESSING")]
    stats = worker.run_once(outbox, [ctx.account_id], worker_id="restarted-worker")

    assert stats["succeeded"] == 1
    persisted = outbox.get_event(ctx.account_id, event.event_id)
    assert (persisted.status, persisted.retry_count) == ("DONE", 0)
    record = index.get_record(IndexRecord.generate_id(node.uri + "/content.md", 2))
    assert record.text == node.content
