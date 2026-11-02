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


@pytest.mark.parametrize("configured_dialect", ["auto", "explicit"])
def test_cold_service_guard_bootstraps_storage(vector_pool, configured_dialect):
    from threading import RLock
    from fs.sql_adapter import SQLContextFS
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    pool, _, _, _, _ = vector_pool
    service = MemoryService.__new__(MemoryService)
    service._cfg = OgMemConfig(
        storage_backend="sql", sql_connection_string=pool.connection_string,
        sql_dialect="auto" if configured_dialect == "auto" else pool.sql_dialect.name,
    )
    service._sql_pool = None
    service._init_lock = RLock()
    service._write_api = None
    ctx = RequestContext(account_id="acct", user_id="alice", agent_id="agent",
                         session_id="cold-" + uuid4().hex, trace_id="cold-guard")
    try:
        with service._session_mutation_scope(ctx, refresh=False) as guard:
            fs = SQLContextFS(pool=service._get_shared_sql_pool())
            conn = fs._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    assert cursor.fetchone()[0] == 1
                conn.commit()
            finally:
                fs._return_connection(conn)
            guard.ensure_owned()
            assert guard.dialect == pool.sql_dialect.name
    finally:
        if service._sql_pool is not None:
            service._sql_pool.close_all()


@pytest.mark.parametrize("method", ["after_turn", "ingest"])
def test_cold_service_connectivity_recovery(vector_pool, method):
    import socket
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    pool, _, _, _, _ = vector_pool
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    service = MemoryService(config=OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        sql_connection_string=f"host=127.0.0.1 port={port} dbname=postgres",
        sql_dialect="auto",
    ))
    ctx = RequestContext("acct", "alice", "agent", "recovery-" + uuid4().hex, "recovery")
    try:
        result = getattr(service, method)({
            "_ctx": ctx, "session_id": ctx.session_id,
            "messages": [{"role": "user", "content": "Remember my preference for tea."}],
        })
        assert result["status"] == "processing"
        assert result["retryable"] is True
        service._cfg.sql_connection_string = pool.connection_string
        with service._session_mutation_scope(ctx, refresh=False) as guard:
            guard.ensure_owned()
            assert guard.dialect == pool.sql_dialect.name
    finally:
        if service._sql_pool is not None:
            service._sql_pool.close_all()


@pytest.mark.parametrize("schema_kind", ["current", "legacy_node_id"])
def test_legacy_vector_backfill_completes_with_canonical_schema(vector_pool, schema_kind):
    from fs.sql_adapter import SQLContextFS

    pool, index, _, _, uri = vector_pool
    SQLContextFS(pool=pool)
    node_id = str(uuid4())
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, "acct")
        with conn.cursor() as cur:
            cur.execute("INSERT INTO context_nodes "
                        "(uri, account_id, owner_space, category, context_type) "
                        "VALUES (%s, 'acct', 'user:alice', 'events', 'MEMORY')", (uri,))
            if schema_kind == "legacy_node_id":
                cur.execute("ALTER TABLE context_nodes ADD COLUMN node_id UUID")
                cur.execute("UPDATE context_nodes SET node_id=%s WHERE uri=%s", (node_id, uri))
            cur.execute("UPDATE vectors SET account_id=NULL, owner_space=NULL, node_uri=NULL")
        conn.commit()
    finally:
        pool.return_connection(conn)
    assert index.migrate_legacy_rows() is True
    assert index.count({"account_id": "acct", "owner_space": "user:alice"}) == 1
    assert index.search_by_vector([1, 0, 0], {"account_id": "acct"}, 1)[0].uri == uri
    if schema_kind == "legacy_node_id":
        assert index.search_by_vector([1, 0, 0], {"account_id": "acct"}, 1)[0].metadata["node_id"] == node_id
    assert index.migrate_legacy_rows() is False
