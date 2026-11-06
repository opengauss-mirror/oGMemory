"""Exercise projection numeric identity checks through the SQL traversal query."""

import os

import psycopg2
import pytest
from psycopg2.extras import Json

from core.models import RequestContext
from providers.relation_store.sql_relation_store import SQLRelationStore


@pytest.mark.parametrize("source_revision,target_revision", [(1, 2), (2147483646, 2147483647)])
def test_current_numeric_projection_is_traversable(source_revision, target_revision):
    dsn = os.environ.get("GRAPH_TEST_DSN")
    if not dsn:
        pytest.skip("GRAPH_TEST_DSN must point to a disposable database")
    ctx = RequestContext("numeric-graph", "alice", "agent", "session", "trace")
    source = "ctx://numeric-graph/users/alice/memories/entity/source"
    target = "ctx://numeric-graph/users/alice/memories/entity/target"
    profile = dict(profile_id="numeric", model="test", dimension=2,
                   representation_version="relation-target-v2")
    projection = dict(embedding_profile="numeric", model="test", dimension=2,
                      representation_version="relation-target-v2", vector=[1.0, 0.0],
                      source_revision=source_revision, target_revision=target_revision)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TEMP TABLE context_nodes (
                account_id text, uri text, version integer)""")
            cur.execute("""CREATE TEMP TABLE relation_edges (
                account_id text, from_uri text, to_uri text, relation_type text,
                weight double precision, edge_projection jsonb)""")
            cur.executemany("INSERT INTO context_nodes VALUES (%s, %s, %s)", [
                (ctx.account_id, source, source_revision),
                (ctx.account_id, target, target_revision),
            ])
            cur.execute("INSERT INTO relation_edges VALUES (%s, %s, %s, %s, %s, %s)",
                        (ctx.account_id, source, target, "supports", 1.0, Json(projection)))
        conn.commit()
        store = SQLRelationStore.__new__(SQLRelationStore)
        store._pool = [conn]
        store._pool_size = 1
        for anchor in (source, target):
            result = store.get_traversal_neighbors(anchor, ctx, embedding_profile=profile)
            assert len(result) == 1
            assert result[0]["source_revision"] == source_revision
            assert result[0]["target_revision"] == target_revision
            assert result[0]["edge_projection"] == projection
            assert result[0]["target_uri"] == target
    finally:
        conn.close()
