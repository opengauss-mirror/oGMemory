"""Deletion regressions; GRAPH_TEST_DSN must be an isolated database."""
from dataclasses import replace
import pytest
from core.models import RelationEdge
from index.edge_projection import EdgeProjectionWorker, InMemoryEdgeProjectionStore
from tests.integration.test_semantic_context_graph import graph_db

@pytest.mark.parametrize('endpoint', [0, 1])
@pytest.mark.parametrize('phase', ['pending', 'claimed', 'complete'])
@pytest.mark.parametrize('batch', [False, True])
def test_deleted_endpoint_cannot_retain_or_revive_projection(graph_db, endpoint, phase, batch):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    a, b, _ = nodes
    store.upsert_edges([RelationEdge(a.uri,b.uri,'supports',1.,'source evidence')],ctx)
    late = []
    if phase != 'pending':
        items = projections.claim_pending(ctx.account_id)
        late = EdgeProjectionWorker(worker._embedder, InMemoryEdgeProjectionStore(),profile).project_accepted_sync(items)
        if phase == 'complete': projections.put_many(late)
    target = nodes[endpoint]
    if batch:
        fs.delete_nodes_with_outbox([target.uri],ctx,[outbox.build_delete_event(target.uri)])
    else:
        fs.delete_node(target.uri,ctx)
    conn = fs._get_connection()
    try:
        fs.bind_tenant(conn,ctx.account_id)
        with conn.cursor() as cur:
            for table in ('edge_projection_jobs','relation_edges'):
                cur.execute(f'SELECT count(*) FROM {table} WHERE account_id=%s',(ctx.account_id,))
                assert cur.fetchone()[0] == 0
    finally:
        conn.rollback()
        fs._return_connection(conn)
    # Same URI, same initial version, different content: old rows must not reappear.
    fs.write_node(replace(target,content='Entirely different fact.',
        metadata={'_expected_version':0}),ctx)
    assert fs.read_node(target.uri,ctx).metadata['version'] == 1
    projections.put_many(late)  # stale in-flight results cannot restore the projection
    assert projections.claim_pending(ctx.account_id) == []
    assert store.get_relation_neighbors(a.uri,ctx) == []
