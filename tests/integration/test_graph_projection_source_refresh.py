"""Completed relation-target projections follow source-only revisions atomically."""

from core.models import RelationEdge
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def _update_content(fs, ctx, node, content):
    current = fs.read_node(node.uri, ctx)
    current.content = content
    current.metadata["_expected_version"] = current.metadata["version"]
    fs.write_node(current, ctx)


def test_source_refresh_reuses_complete_projection_and_keeps_traversal_live(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, third = nodes
    store.upsert_edges([
        RelationEdge(source.uri, target.uri, "owns", 1., "evidence"),
        RelationEdge(source.uri, third.uri, "approves", 1., "evidence"),
    ], ctx)
    worker.run_once(outbox, [ctx.account_id])
    before = {row.uri: row.edge_projection for row in store.get_relation_neighbors(source.uri, ctx)}
    before_events = _graph_sql(fs, ctx,
        "SELECT event_id,status FROM outbox_events WHERE account_id=%s ORDER BY event_id",
        (ctx.account_id,))
    initial_calls = worker._embedder.embed_texts.call_count

    for revision in (2, 3):
        _update_content(fs, ctx, source, f"Source delivery date changed, revision {revision}")
        jobs = _graph_sql(fs, ctx,
            "SELECT status,source_revision,target_revision,claim_token,attempt_count "
            "FROM edge_projection_jobs WHERE account_id=%s ORDER BY edge_id", (ctx.account_id,))
        assert jobs == [("COMPLETE", revision, 1, None, 1)] * 2
        for row in store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile):
            expected = dict(before[row["target_uri"]], source_revision=revision)
            assert row["edge_projection"] == expected
        assert len(store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)) == 2

    assert worker._embedder.embed_texts.call_count == initial_calls
    assert _graph_sql(fs, ctx,
        "SELECT event_id,status FROM outbox_events WHERE account_id=%s ORDER BY event_id",
        (ctx.account_id,)) == before_events


def test_target_content_update_projects_new_text_after_source_refresh(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    worker.run_once(outbox, [ctx.account_id])
    _update_content(fs, ctx, source, "Source changed")
    _update_content(fs, ctx, target, "Target changed")
    assert _graph_sql(fs, ctx,
        "SELECT status,source_revision,target_revision FROM edge_projection_jobs WHERE account_id=%s",
        (ctx.account_id,)) == [("PENDING", 2, 2)]
    worker.run_once(outbox, [ctx.account_id])
    row = store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)[0]
    assert row["edge_projection"]["source_revision"] == 2
    assert row["edge_projection"]["target_revision"] == 2
    assert "Target changed" in row["edge_projection"]["text"]


def test_source_refresh_during_projection_claim_finishes_at_latest_revision(graph_db):
    from index.edge_projection import EdgeProjectionWorker

    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    original_claim = projections.claim_pending(ctx.account_id)
    _update_content(fs, ctx, source, "Source updated while projection runs")
    current_claim = projections.claim_pending(ctx.account_id)
    assert current_claim[0].source_revision == 2
    assert current_claim[0].claim_token != original_claim[0].claim_token
    projector = EdgeProjectionWorker(worker._embedder, projections, profile)
    projector.project_accepted_sync(current_claim)
    projector.project_accepted_sync(original_claim)
    row = store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)[0]
    assert row["edge_projection"]["source_revision"] == 2
    assert _graph_sql(fs, ctx,
        "SELECT status,source_revision,claim_token FROM edge_projection_jobs WHERE account_id=%s",
        (ctx.account_id,)) == [("COMPLETE", 2, None)]
