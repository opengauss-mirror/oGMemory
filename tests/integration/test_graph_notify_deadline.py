"""PostgreSQL listener regression for graph projections split by the byte budget."""

import json
import time
from contextlib import closing
from itertools import count

import pytest

from core.models import RelationEdge
from index.sql_notify_listener import SQLNotifyListener
from providers.relation_store.projection_identity import canonical_edge_id
from tests.integration.test_graph_projection_source_refresh import _update_content
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def test_notify_listener_finishes_split_graph_batches_at_retry_deadline(graph_db):
    ctx, fs, _, relations, _, outbox, worker, nodes, _ = graph_db
    if not outbox.supports_batch_claim:
        pytest.skip("PostgreSQL LISTEN/NOTIFY integration")
    source, first, second = nodes
    for target, word in ((first, "a"), (second, "b")):
        _update_content(fs, ctx, target, word * 5000)

    def edge_id(edge):
        return canonical_edge_id(ctx.account_id, edge.from_uri, edge.to_uri, edge.relation_type)

    first_edge = RelationEdge(source.uri, first.uri, "supports", 1., "evidence")
    # Same transaction timestamps tie on edge ID. Outbox order follows target URI.
    second_edge = next(edge for edge in (
        RelationEdge(source.uri, second.uri, f"supports_{i}", 1., "evidence") for i in count()
    ) if edge_id(edge) < edge_id(first_edge))
    edges = [first_edge, second_edge]

    def state():
        jobs = _graph_sql(fs, ctx,
            "SELECT status, attempt_count FROM edge_projection_jobs "
            "WHERE account_id=%s ORDER BY created_at, edge_id", (ctx.account_id,))
        events = _graph_sql(fs, ctx,
            "SELECT status, retry_count FROM outbox_events "
            "WHERE account_id=%s AND event_type='PROJECT_GRAPH' ORDER BY seq", (ctx.account_id,))
        return {"jobs": jobs, "events": events}

    listener = SQLNotifyListener(outbox, worker)
    with closing(outbox.open_listener_connection()) as conn:
        print(json.dumps({"request": "SQLRelationStore.upsert_edges", "account": ctx.account_id,
                          "source": source.uri, "edges": [(e.to_uri, e.relation_type) for e in edges],
                          "target_content_bytes_each": 5000}))
        started = time.monotonic()
        relations.upsert_edges(edges, ctx)
        listener.run_cycle(conn)
        after_notify = state()
        print(json.dumps({"after_notify": after_notify}))
        assert after_notify == {"jobs": [("COMPLETE", 1), ("PENDING", 0)],
                                "events": [("PENDING", 0), ("PENDING", 0)]}
        listener.run_cycle(conn)
        elapsed = time.monotonic() - started
        completed = state()
        print(json.dumps({"response": completed, "elapsed_seconds": elapsed}))
        assert completed == {"jobs": [("COMPLETE", 1)] * 2, "events": [("DONE", 0)] * 2}
        assert elapsed < 4
