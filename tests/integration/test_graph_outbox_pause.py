"""Graph on -> off -> on with real SQL wakeups, ordering and node triggers."""
from types import SimpleNamespace

import pytest

from core.models import IndexRecord, RelationEdge
from index.outbox_worker import OutboxWorker
from server.memory_service import MemoryService
from tests.integration.test_semantic_context_graph import graph_db as graph_db, _graph_sql


@pytest.mark.parametrize("initial_state", ["PENDING", "PROCESSING"])
def test_disabled_graph_preserves_jobs_and_unblocks_indexing_then_resumes(graph_db, initial_state):
    ctx, fs, index, store, projections, outbox, enabled, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "related", 1., "evidence")], ctx)
    source_record_id = IndexRecord.generate_id(source.uri + "/content.md", 2)
    index.delete([source_record_id])
    # A graph wakeup precedes the normal update for exactly the same URI.
    ordinary = outbox.register_write(source, ctx)
    _graph_sql(fs, ctx,
        "UPDATE outbox_events SET status=%s, retry_count=4, worker_id='old-worker', locked_at=NOW() "
        "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (initial_state, ctx.account_id))
    before = _graph_sql(fs, ctx,
        "SELECT event_id,status,retry_count,payload FROM outbox_events "
        "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
    assert len(before) == 1
    disabled = OutboxWorker(index, enabled._embedder, fs=fs, directory_summary_enabled=False)
    for _ in range(disabled._config.max_retry + 2):
        disabled.run_once(outbox, [], worker_id="graph-off")
    assert outbox.get_event(ctx.account_id, ordinary.event_id) is None
    assert index.get_record(source_record_id) is not None
    assert _graph_sql(fs, ctx,
        "SELECT event_id,status,retry_count,payload FROM outbox_events "
        "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,)) == before
    assert _graph_sql(fs, ctx,
        "SELECT status FROM edge_projection_jobs WHERE account_id=%s", (ctx.account_id,)) == [("PENDING",)]
    # General status reporting still exposes paused work. Only explicit
    # ordinary/code readiness excludes the disabled graph feature.
    assert outbox.count_unfinished(ctx.account_id)["total"] == 1
    assert outbox.count_unfinished(ctx.account_id, exclude_event_types=["PROJECT_GRAPH"])["total"] == 0
    memory = MemoryService.__new__(MemoryService)
    memory._cfg = SimpleNamespace(graph_backend="none")
    memory._get_context_fs = lambda: fs
    memory._get_outbox_store = lambda _: outbox
    assert memory._code_workspace_index_visible(ctx)
    assert memory._wait_for_outbox_idle(ctx, timeout_seconds=0)["idle"]
    assert memory._count_outbox_unfinished(ctx)["total"] == 1
    memory._cfg.graph_backend = "sql"
    assert not memory._code_workspace_index_visible(ctx)
    assert not memory._wait_for_outbox_idle(ctx, timeout_seconds=0)["idle"]

    # The node-update trigger keeps producing durable graph work even while
    # the runtime is off. The latest target revision must be projected later.
    changed = fs.read_node(target.uri, ctx)
    changed.content = "target updated while graph disabled"
    changed.metadata["_expected_version"] = changed.metadata["version"]
    fs.write_node(changed, ctx)
    target_write = outbox.register_write(changed, ctx)
    disabled.run_once(outbox, [], worker_id="graph-off")
    assert outbox.get_event(ctx.account_id, target_write.event_id) is None
    assert index.get_record(IndexRecord.generate_id(target.uri + "/content.md", 2)).text == changed.content
    # Simulate expiry of a prior enabled worker's processing lease; the off
    # worker never changes that claim or spends its remaining retry budget.
    _graph_sql(fs, ctx,
        "UPDATE outbox_events SET locked_at=NOW()-INTERVAL '6 minutes' "
        "WHERE account_id=%s AND event_type='PROJECT_GRAPH'", (ctx.account_id,))
    for _ in range(4):
        enabled.run_once(outbox, [], worker_id="graph-on-again")
    assert _graph_sql(fs, ctx,
        "SELECT count(*) FROM outbox_events WHERE account_id=%s AND event_type='PROJECT_GRAPH'",
        (ctx.account_id,)) == [(0,)]
    assert _graph_sql(fs, ctx,
        "SELECT status FROM edge_projection_jobs WHERE account_id=%s", (ctx.account_id,)) == [("COMPLETE",)]
    assert store.get_relation_neighbors(source.uri, ctx)[0].edge_projection["target_revision"] == 2
