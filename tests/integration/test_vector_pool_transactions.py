"""Vector reads release transactions while session-owned work retains its boundary."""

import os
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS, make_dsn

from core.models import IndexRecord, RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.session_mutation import SQLSessionGuard, _ACTIVE
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


@pytest.fixture
def vector_pool():
    dsn = os.environ.get("OGMEM_SCENE_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_SCENE_SQL_DSN is required")
    dialect = os.environ.get("OGMEM_SCENE_SQL_DIALECT", "postgresql")
    namespace = "vector_tx_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    pool = SharedConnectionPool(make_dsn(dsn, options=f"-c search_path={namespace},public"),
                                pool_size=1, dialect=dialect)
    try:
        index = OpenGaussVectorIndex(pool=pool, dimension=3, table_name="vectors")
        uri = "ctx://acct/users/alice/memories/events/item"
        index.upsert([IndexRecord(id="item", uri=uri, level=0, text="release evidence",
                      filters={"account_id": "acct", "owner_space": "user:alice"},
                      metadata={"_embedding": [1.0, 0.0, 0.0], "parent_uri": "parent"})])
        yield pool, index, control, namespace, uri
    finally:
        pool.close_all()
        with control.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))
        control.close()


@pytest.mark.parametrize("operation", ["search", "embeddings", "children", "count"])
def test_vector_read_returns_idle_connection_and_releases_table_lock(vector_pool, operation):
    pool, index, control, namespace, uri = vector_pool
    filters = {"account_id": "acct", "owner_space": "user:alice"}
    if operation == "search":
        assert [hit.uri for hit in index.search_by_vector([1, 0, 0], filters, 2)] == [uri]
    elif operation == "embeddings":
        assert index.get_embeddings([uri], filters) == {uri: [1.0, 0.0, 0.0]}
    elif operation == "children":
        assert [hit.uri for hit in index.search_children("parent", [1, 0, 0], filters, 2)] == [uri]
    else:
        assert index.count(filters) == 1
    assert pool._pool[0].info.transaction_status == TRANSACTION_STATUS_IDLE
    with control.cursor() as cursor:
        cursor.execute("SET statement_timeout = 1000")
        cursor.execute(sql.SQL("ALTER TABLE {}.vectors ADD COLUMN marker integer").format(sql.Identifier(namespace)))
    assert index.count(filters) == 1


def test_vector_query_recovers_on_same_pool_after_statement_error(vector_pool):
    pool, index, _, _, _ = vector_pool
    with pytest.raises(psycopg2.Error):
        index.search_by_vector([1, 0], {"account_id": "acct"}, 2)
    assert pool._pool[0].info.transaction_status == TRANSACTION_STATUS_IDLE
    assert index.count({"account_id": "acct"}) == 1


def test_session_owned_vector_read_preserves_pending_write_until_commit(vector_pool):
    pool, index, _, _, _ = vector_pool
    raw = pool.get_connection()
    ctx = RequestContext("acct", "alice", "agent", "session", "trace")
    guard = SQLSessionGuard(raw, pool.connection_string, ctx, "test", dialect=pool.dialect)
    token = _ACTIVE.set(guard)
    try:
        with raw.cursor() as cursor:
            cursor.execute("UPDATE vectors SET text = 'pending evidence' WHERE id = 'item'")
        assert index.count({"account_id": "acct"}) == 1
        assert raw.info.transaction_status == TRANSACTION_STATUS_INTRANS
        with raw.cursor() as cursor:
            cursor.execute("SELECT text FROM vectors WHERE id = 'item'")
            assert cursor.fetchone()[0] == "pending evidence"
        assert guard.borrowed == 0
        raw.commit()
    finally:
        _ACTIVE.reset(token)
        raw.rollback()
        pool.return_connection(raw)
    assert index.count({"account_id": "acct"}) == 1
