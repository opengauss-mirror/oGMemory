"""Shared source revisions stay aligned with graph jobs after moving memories."""

from unittest.mock import Mock

import pytest

from core.models import RelationEdge
from fs.sql_adapter.sql_context_fs import SQLContextFS
from index.edge_projection import EdgeProjectionWorker
from providers.relation_store.sql_projection import SQLEdgeProjectionStore
from providers.relation_store.sql_relation_store import SQLRelationStore
from tests.integration.test_space_job_review import pool as pool
from tests.integration.test_space_job_review import source_node
from tests.integration.test_space_job_review import workspace as workspace


@pytest.mark.parametrize("endpoint", ["source", "target", "subtree", "self"])
@pytest.mark.parametrize("state", ["claimed", "complete"])
def test_move_shared_sources_refreshes_projection_revisions(workspace, pool, endpoint, state):
    jobs, ctx = workspace
    fs = SQLContextFS(pool=pool)
    relations = SQLRelationStore(pool=pool, edge_projection_profile_id="move-test")
    projections = SQLEdgeProjectionStore(pool=pool)
    profile = {"profile_id": "move-test", "model": "test", "dimension": 2,
               "representation_version": "relation-target-v2"}
    embedder = Mock()
    embedder.embed_texts.side_effect = lambda texts: [[1., 0.] for _ in texts]
    worker = EdgeProjectionWorker(embedder, projections, profile)
    parent = source_node(ctx, "", "Folder")
    parent.uri = parent.uri.rstrip("/")
    source = source_node(ctx, "source", "Pool source")
    target = source_node(ctx, "target", "Pool target")
    for node in (parent, source, target):
        fs.write_node(node, ctx)
    if endpoint == "self":
        target = source
    relations.upsert_edges([RelationEdge(source.uri, target.uri, "related", 1., "evidence")], ctx)
    original = projections.claim_pending(ctx.account_id)
    if state == "complete":
        worker.project_accepted_sync(original)

    moving = {"source": source, "target": target, "subtree": parent, "self": source}[endpoint]
    destination = moving.uri + "-moved"
    print("request", {"method": "SQLContextFS.move_node", "from_uri": moving.uri,
                      "to_uri": destination, "projection_state": state})
    fs.move_node(moving.uri, destination, ctx)
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("""SELECT j.source_uri,j.target_uri,j.source_revision,j.target_revision,
            s.version AS source_version,t.version AS target_version,j.status
            FROM edge_projection_jobs j
            JOIN context_nodes s ON s.account_id=j.account_id AND s.uri=j.source_uri
            JOIN context_nodes t ON t.account_id=j.account_id AND t.uri=j.target_uri
            WHERE j.account_id=%s""", (ctx.account_id,))
        row = dict(cur.fetchone())
    current = projections.claim_pending(ctx.account_id)
    print("response", row, "claim_pending_count", len(current))
    assert (row["source_revision"], row["target_revision"]) == (row["source_version"], row["target_version"])
    assert len(current) == 1
    worker.project_accepted_sync(current)
    [neighbor] = relations.get_traversal_neighbors(row["source_uri"], ctx, embedding_profile=profile)
    assert neighbor["target_uri"] == row["target_uri"]
    assert neighbor["edge_projection"]["source_revision"] == row["source_version"]
    assert neighbor["edge_projection"]["target_revision"] == row["target_version"]