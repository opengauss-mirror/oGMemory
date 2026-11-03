"""No-clear-cache SQL initialization, resolved targets and pending transactions."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY
from core.models import CandidateMemory, RequestContext
from fs.sql_adapter import schema
from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from session.session_state import SessionState
from session.sql_archive_store import SQLSessionArchiveStore
from tests.integration.test_sql_schema_owner_upgrade import NoModel


@pytest.fixture(params=["postgresql", "opengauss"])
def targets(request):
    dialect = request.param
    dsn = os.environ.get(f"OGMEM_SCOPE_{dialect.upper()}_TEST_DSN")
    if not dsn:
        pytest.skip(f"configure OGMEM_SCOPE_{dialect.upper()}_TEST_DSN for a disposable database")
    names = ["scope18_" + uuid4().hex for _ in range(3)]
    control = psycopg2.connect(dsn)
    control.autocommit = True
    pools = []

    def pool_for(name):
        isolated = psycopg2.extensions.make_dsn(dsn, options=f"-c search_path={name} -c statement_timeout=10000")
        pool = SharedConnectionPool(isolated, pool_size=1, dialect=dialect)
        pools.append(pool)
        return pool

    try:
        with control.cursor() as cur:
            for name in names:
                cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        yield dialect, names, control, pool_for
    finally:
        for pool in pools:
            pool.close_all()
        with control.cursor() as cur:
            for name in reversed(names):
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
        control.close()


def table_names(control, name):
    with control.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s", (name,))
        return {row[0] for row in cur.fetchall()}


def test_transaction_reinitializes_while_another_initializer_waits_for_its_ddl(targets, monkeypatch):
    dialect, names, control, _ = targets
    dsn = psycopg2.extensions.make_dsn(os.environ[f"OGMEM_SCOPE_{dialect.upper()}_TEST_DSN"], options=f"-c search_path={names[0]} -c statement_timeout=8000")
    first, second = psycopg2.connect(dsn), psycopg2.connect(dsn)
    held = threading.Event()
    original_lock = schema._target_lock

    @contextmanager
    def observed_lock(target):
        with original_lock(target):
            held.set()
            yield

    try:
        with first.cursor() as cur:
            cur.execute("SELECT 1")
        schema.ensure_schema(first, dialect)
        monkeypatch.setattr(schema, "_target_lock", observed_lock)
        with ThreadPoolExecutor(max_workers=2) as workers:
            waiting = workers.submit(schema.ensure_schema, second, dialect)
            assert held.wait(2)
            continued = workers.submit(schema.ensure_schema, first, dialect)
            try:
                continued.result(timeout=2)
                first.commit()
                waiting.result(timeout=3)
            finally:
                second.cancel()
                continued.result(timeout=3)
                first.rollback()
    finally:
        first.close()
        second.close()


def public_write(fs, pool, *, operation="scope-operation"):
    ctx = RequestContext("SchemaScopePrincipal", "Nora研发", "Agent.Lab+2", "session", "trace")
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    candidate = CandidateMemory("event", "user", "scope-repair", "I repaired the receiver.",
                                "I repaired the receiver.", "I repaired the receiver.", .9,
                                when="2026-10-05", who="Nora", title="Repair", write_operation_id=operation)
    plan = writer.write_memory(candidate, ctx)
    stored = fs.read_node(plan["target_uri"], ctx)
    assert candidate.write_operation_id in stored.metadata[RECEIPTS_KEY]
    assert stored.owner_space == ctx.user_space_name()
    assert writer.write_memory(candidate, ctx)["action"] == "skip"
    assert len(outbox.list_pending(ctx.account_id)) == 1
    return ctx, plan["target_uri"], stored.content


def test_two_targets_public_write_read_without_clear_and_same_target_new_pool_cache_hit(targets, monkeypatch):
    _, names, control, pool_for = targets
    a, b = pool_for(names[0]), pool_for(names[1])
    assert not table_names(control, names[0]) and not table_names(control, names[1])
    fs_a = SQLContextFS(pool=a)
    ctx, uri, content = public_write(fs_a, a)
    fs_b = SQLContextFS(pool=b)
    public_write(fs_b, b)
    assert "context_nodes" in table_names(control, names[0])
    assert "context_nodes" in table_names(control, names[1])
    assert fs_a.read_node(uri, ctx).content == content
    same_a = pool_for(names[0])
    with monkeypatch.context() as patch:
        patch.setattr(schema, "_SCHEMA_SQL", "SELECT 1/0")
        same_fs = SQLContextFS(pool=same_a)
    assert same_fs.read_node(uri, ctx).content == content


def test_runtime_search_path_and_shadow_schema_are_resolved_on_current_connection(targets):
    dialect, names, control, pool_for = targets
    pool = pool_for(names[0])
    SQLContextFS(pool=pool)
    conn = pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path={},{}").format(sql.Identifier(names[1]), sql.Identifier(names[0])))
        conn.commit()
        schema.ensure_schema(conn, dialect)
        assert conn.get_transaction_status() == TRANSACTION_STATUS_IDLE
    finally:
        pool.return_connection(conn)
    fs = SQLContextFS(pool=pool)
    public_write(fs, pool)
    assert "context_nodes" in table_names(control, names[0])
    assert "context_nodes" in table_names(control, names[1])
    conn = pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path={},{}").format(sql.Identifier(names[0]), sql.Identifier(names[1])))
        conn.commit()
        before = len(schema._schema_ensured)
        schema.ensure_schema(conn, dialect)
        assert len(schema._schema_ensured) == before + 1
    finally:
        pool.return_connection(conn)


def test_recreated_schema_with_same_name_does_not_reuse_old_target_cache(targets):
    _, names, control, pool_for = targets
    pool = pool_for(names[0])
    public_write(SQLContextFS(pool=pool), pool)
    conn = pool.get_connection()
    try:
        conn.rollback()  # End the caller's final read transaction before dropping its schema.
    finally:
        pool.return_connection(conn)
    with control.cursor() as cur:
        cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(names[0])))
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(names[0])))
    assert not table_names(control, names[0])
    public_write(SQLContextFS(pool=pool), pool)
    assert "context_nodes" in table_names(control, names[0])


def test_cached_constructors_preserve_pending_transaction_tenant_and_guard(targets):
    _, names, control, pool_for = targets
    pool = pool_for(names[0])
    fs = SQLContextFS(pool=pool)
    conn = pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE pending_probe (value TEXT)")
        conn.commit()
    finally:
        pool.return_connection(conn)
    ctx = RequestContext("PendingPrincipal", "Nora", "Agent", "session", "trace")
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        conn = fs._get_connection()
        fs.bind_tenant(conn, ctx.account_id)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO pending_probe VALUES ('caller-pending')")
        fs._return_connection(conn)
        assert guard.raw.get_transaction_status() == TRANSACTION_STATUS_INTRANS
        cached_fs = SQLContextFS(pool=pool)
        SQLOutboxStore(pool=pool, fs=cached_fs)
        SQLSessionArchiveStore(pool=pool)
        assert guard.raw.get_transaction_status() == TRANSACTION_STATUS_INTRANS
        assert guard.poison_reason is None
        with control.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}.pending_probe").format(sql.Identifier(names[0])))
            assert cur.fetchone()[0] == 0
        conn = cached_fs._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), current_setting('app.account_id') FROM pending_probe")
            assert cur.fetchone() == (1, ctx.account_id)
        conn.commit()
        cached_fs._return_connection(conn)
        guard.ensure_owned()
        archive = SQLSessionArchiveStore(pool=pool)
        assert archive.write_archive(ctx.session_id, "overview", "abstract", [], ctx, archive_id="pending-archive").success
        guard.ensure_owned()
    with control.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}.pending_probe").format(sql.Identifier(names[0])))
        assert cur.fetchone()[0] == 1


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_catalog_and_function_shadows_preserve_public_guarded_constructor_transaction(
    targets, monkeypatch, cached,
):
    dialect, names, control, pool_for = targets
    pool = pool_for(names[0])
    fs = SQLContextFS(pool=pool)
    with control.cursor() as cur:
        cur.execute(sql.SQL("CREATE TABLE {}.pending_probe (value TEXT)").format(sql.Identifier(names[1])))
        for name, arguments, returns, body in (
            ("inet_server_addr", "", "inet", "SELECT NULL::inet"),
            ("inet_server_port", "", "integer", "SELECT 0"),
            ("current_database", "", "name", "SELECT 'schema18-shadow-db'::name"),
            ("current_schema", "", "name", "SELECT 'schema18-shadow-schema'::name"),
            ("current_schemas", "boolean", "name[]", "SELECT ARRAY[]::name[]"),
        ):
            cur.execute(sql.SQL("CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE SQL AS {}")
                        .format(sql.Identifier(names[1]), sql.Identifier(name), sql.SQL(arguments),
                                sql.SQL(returns), sql.Literal(body)))
        cur.execute("SELECT pg_catalog.current_database()")
        actual_database = cur.fetchone()[0]
    ctx = RequestContext("ShadowPrincipal", "Nora", "Agent", "shadow-session", "trace")
    adapter_class = type(resolve_sql_dialect(dialect))
    original = adapter_class.prepare_schema
    calls = []

    def counted(self, text):
        calls.append(text)
        return original(self, text)

    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        conn = fs._get_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE pg_database (bad_column INTEGER)")
            cur.execute("CREATE TEMP TABLE pg_namespace (bad_column INTEGER)")
            cur.execute(sql.SQL("SET search_path={},pg_catalog").format(sql.Identifier(names[1])))
        conn.commit()
        fs._return_connection(conn)
        if cached:
            SQLContextFS(pool=pool)
        before_cache = set(schema._schema_ensured)
        monkeypatch.setattr(adapter_class, "prepare_schema", counted)
        conn = fs._get_connection()
        fs.bind_tenant(conn, ctx.account_id)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT {}.current_database(), {}.current_schema(), {}.current_schemas(true), "
                                "{}.inet_server_port(), {}.inet_server_addr()::text")
                        .format(*[sql.Identifier(names[1])] * 5))
            shadow_values = ("schema18-shadow-db", "schema18-shadow-schema", [], 0, None)
            assert cur.fetchone() == shadow_values
            cur.execute("SELECT pg_catalog.current_database(), pg_catalog.current_schema(), "
                        "pg_catalog.current_schemas(true), pg_catalog.inet_server_port(), "
                        "pg_catalog.inet_server_addr()::pg_catalog.text")
            actual_values = cur.fetchone()
            cur.execute("SELECT current_database(), current_schema(), current_schemas(true), "
                        "inet_server_port(), inet_server_addr()::text")
            # PG resolves these functions through the path; OG prioritizes its builtins.
            assert cur.fetchone() == (shadow_values if dialect == "postgresql" else actual_values)
            cur.execute("INSERT INTO pending_probe VALUES ('caller-pending')")
        fs._return_connection(conn)
        with control.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}.pending_probe").format(sql.Identifier(names[1])))
            assert cur.fetchone()[0] == 0
        SQLContextFS(pool=pool)
        assert len(calls) == (0 if cached else 1)
        assert set(schema._schema_ensured) == before_cache
        if cached:
            targets_for_database = [target for target in before_cache if target[2] == actual_database
                                    and target[5] == names[1]]
            assert targets_for_database and all(target[6] for target in targets_for_database)
        assert guard.raw.get_transaction_status() == TRANSACTION_STATUS_INTRANS
        assert guard.poison_reason is None
        conn = fs._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), pg_catalog.current_setting('app.account_id') FROM pending_probe")
            assert cur.fetchone() == (1, ctx.account_id)
            unsigned = guard.key & ((1 << 64) - 1)
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_locks WHERE locktype='advisory' "
                        "AND pid=pg_catalog.pg_backend_pid() AND classid=%s AND objid=%s AND objsubid=1)",
                        (unsigned >> 32, unsigned & ((1 << 32) - 1)))
            assert cur.fetchone()[0] is True
        conn.commit()  # Only the caller commits its pending row and any uncached DDL.
        fs._return_connection(conn)
        guard.ensure_owned()
    with control.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}.pending_probe").format(sql.Identifier(names[1])))
        assert cur.fetchone()[0] == 1


def test_failed_target_initialization_retries_without_cache_reset(targets):
    _, names, control, pool_for = targets
    pool = pool_for(names[0])
    with control.cursor() as cur:
        cur.execute(sql.SQL("CREATE TABLE {}.session_archives (archive_id TEXT)").format(sql.Identifier(names[0])))
    with pytest.raises(RuntimeError) as error:
        SQLContextFS(pool=pool)
    assert isinstance(error.value.__cause__, psycopg2.Error) and error.value.__cause__.pgcode == "42703"
    with control.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE {}.session_archives").format(sql.Identifier(names[0])))
    public_write(SQLContextFS(pool=pool), pool)
    assert "context_nodes" in table_names(control, names[0])


def test_uncached_initialization_keeps_caller_transaction_and_rollback_is_retryable(targets):
    dialect, names, control, pool_for = targets
    pool = pool_for(names[0])
    with control.cursor() as cur:
        cur.execute(sql.SQL("CREATE TABLE {}.pending_probe (value TEXT)").format(sql.Identifier(names[0])))
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, "PendingPrincipal")
        with conn.cursor() as cur:
            cur.execute("INSERT INTO pending_probe VALUES ('caller-pending')")
        before = set(schema._schema_ensured)
        schema.ensure_schema(conn, dialect)
        assert conn.get_transaction_status() == TRANSACTION_STATUS_INTRANS and conn.autocommit is False
        assert set(schema._schema_ensured) == before
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), current_setting('app.account_id') FROM pending_probe")
            assert cur.fetchone() == (1, "PendingPrincipal")
        assert "context_nodes" not in table_names(control, names[0])
        conn.rollback()  # The caller owns the rollback, including its uncommitted DDL.
    finally:
        pool.return_connection(conn)
    assert "context_nodes" not in table_names(control, names[0])
    public_write(SQLContextFS(pool=pool), pool)
    assert "context_nodes" in table_names(control, names[0])


def test_two_workers_same_target_apply_canonical_schema_once(targets, monkeypatch):
    dialect, names, control, pool_for = targets
    adapter_class = type(resolve_sql_dialect(dialect))
    original = adapter_class.prepare_schema
    calls = []
    calls_lock = threading.Lock()

    def counted(self, text):
        with calls_lock:
            calls.append(text)
        return original(self, text)

    monkeypatch.setattr(adapter_class, "prepare_schema", counted)
    barrier = threading.Barrier(2)

    def construct(pool):
        barrier.wait(timeout=3)
        return SQLContextFS(pool=pool)

    a, same_a = pool_for(names[0]), pool_for(names[0])
    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.submit(construct, a), workers.submit(construct, same_a)
        first.result(timeout=15)
        second.result(timeout=15)
    assert len(calls) == 1
    assert "context_nodes" in table_names(control, names[0])


@pytest.mark.parametrize("dialect", ["postgresql", "opengauss"])
def test_same_principal_different_database_target_is_not_cached_as_first(dialect):
    dsn = os.environ.get(f"OGMEM_SCOPE_{dialect.upper()}_TEST_DSN")
    other = os.environ.get(f"OGMEM_SCOPE_{dialect.upper()}_OTHER_DATABASE_DSN")
    if not dsn or not other:
        pytest.skip("configure a second disposable database for actual database-target isolation")
    pools = [SharedConnectionPool(value, pool_size=1, dialect=dialect) for value in (dsn, other)]
    try:
        for pool in pools:
            public_write(SQLContextFS(pool=pool), pool, operation="database-scope-operation")
    finally:
        for pool in pools:
            pool.close_all()
