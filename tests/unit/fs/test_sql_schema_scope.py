"""Resolved schema-cache identity and transaction/concurrency contracts."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from psycopg2.extensions import TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS

from fs.sql_adapter import schema


class Connection:
    def __init__(self, name="a", *, endpoint="127.0.0.1", database="db", oid=1, role="reader", path=None):
        self.target = (endpoint, 5432, database, oid, role, name, path or ["pg_catalog", name])
        self.namespaces = {value: 10 + index for index, value in enumerate(self.target[6])}
        self.info = SimpleNamespace(host="/var/run/postgresql", port=5432)
        self.state = TRANSACTION_STATUS_IDLE
        self._autocommit = False
        self.pending = []
        self.ddl_count = 0
        self.fail_ddl = False
        self.fail_lookup = False
        self.ddl_started = None
        self.ddl_release = None
        self.metadata_ready = threading.Event()
        self.statements = []

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        if self.state != TRANSACTION_STATUS_IDLE:
            raise AssertionError("autocommit must not change inside caller transaction")
        self._autocommit = value

    def get_transaction_status(self):
        return self.state

    def cursor(self):
        return Cursor(self)

    def commit(self):
        raise AssertionError("ensure_schema must not commit caller work")

    def rollback(self):
        raise AssertionError("ensure_schema must not roll back caller work")


class Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement):
        self.conn.statements.append(statement)
        if not self.conn.autocommit:
            self.conn.state = TRANSACTION_STATUS_INTRANS
        if statement.startswith("SELECT pg_catalog.inet_server_addr"):
            self.conn.metadata_ready.set()
            if self.conn.fail_lookup:
                raise ValueError("lookup unavailable")
        if statement.startswith("CREATE TABLE"):
            self.conn.ddl_count += 1
            if self.conn.ddl_started:
                self.conn.ddl_started.set()
                assert self.conn.ddl_release.wait(2)
            if self.conn.fail_ddl:
                raise ValueError("DDL rejected")

    def fetchone(self):
        return self.conn.target

    def fetchall(self):
        return list(self.conn.namespaces.items())


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    previous = set(schema._schema_ensured)
    schema._schema_ensured.clear()
    monkeypatch.setattr(schema, "_SCHEMA_SQL", "CREATE TABLE canonical_probe (id INTEGER)")
    yield
    schema._schema_ensured.clear()
    schema._schema_ensured.update(previous)


def test_two_schema_targets_initialize_without_clear_and_same_target_pool_reuses_cache():
    a, b, same_a = Connection("a"), Connection("b"), Connection("a")
    for conn in (a, b, same_a):
        schema.ensure_schema(conn)
        assert conn.state == TRANSACTION_STATUS_IDLE and conn.autocommit is False
    assert [c.ddl_count for c in (a, b, same_a)] == [1, 1, 0]
    assert len(schema._schema_ensured) == 2


@pytest.mark.parametrize("change", [
    {"endpoint": "127.0.0.2"}, {"database": "other"}, {"oid": 2}, {"role": "other"},
    {"path": ["pg_catalog", "a", "shadow"]},
])
def test_server_database_role_and_complete_effective_path_do_not_share_cache(change):
    first, second = Connection(), Connection(**change)
    schema.ensure_schema(first)
    schema.ensure_schema(second)
    assert first.ddl_count == second.ddl_count == 1
    assert len(schema._schema_ensured) == 2


def test_unidentified_target_is_initialized_without_false_cache():
    conn = Connection()
    conn.target = None
    schema.ensure_schema(conn)
    schema.ensure_schema(conn)
    assert conn.ddl_count == 2 and not schema._schema_ensured


def test_unix_socket_endpoint_uses_actual_connection_address_and_port():
    first, second = Connection(endpoint=None), Connection(endpoint=None)
    second.info = SimpleNamespace(host="/other/postgresql", port=5432)
    schema.ensure_schema(first)
    schema.ensure_schema(second)
    assert first.ddl_count == second.ddl_count == 1 and len(schema._schema_ensured) == 2


def test_recreated_namespace_oid_with_same_name_does_not_hit_old_cache():
    first, second = Connection(), Connection()
    second.namespaces["a"] = 99
    schema.ensure_schema(first)
    schema.ensure_schema(second)
    assert first.ddl_count == second.ddl_count == 1 and len(schema._schema_ensured) == 2


def test_failed_ddl_is_retryable_and_only_complete_success_is_cached():
    conn = Connection()
    conn.fail_ddl = True
    with pytest.raises(ValueError, match="DDL rejected"):
        schema.ensure_schema(conn)
    assert not schema._schema_ensured and conn.autocommit is False
    conn.fail_ddl = False
    schema.ensure_schema(conn)
    schema.ensure_schema(conn)
    assert conn.ddl_count == 2 and len(schema._schema_ensured) == 1


def test_cached_lookup_keeps_existing_transaction_and_pending_work():
    schema.ensure_schema(Connection())
    conn = Connection()
    conn.state = TRANSACTION_STATUS_INTRANS
    conn.pending = ["uncommitted caller write"]
    schema.ensure_schema(conn)
    assert conn.pending == ["uncommitted caller write"] and conn.state == TRANSACTION_STATUS_INTRANS
    assert conn.ddl_count == 0 and conn.autocommit is False


def test_uncached_ddl_in_caller_transaction_does_not_claim_durable_cache():
    conn = Connection()
    conn.state = TRANSACTION_STATUS_INTRANS
    conn.pending = ["caller write"]
    schema.ensure_schema(conn)
    assert conn.ddl_count == 1 and not schema._schema_ensured and conn.pending == ["caller write"]
    conn.state = TRANSACTION_STATUS_IDLE  # Caller later ends its own transaction.
    schema.ensure_schema(conn)
    assert conn.ddl_count == 2 and len(schema._schema_ensured) == 1


def test_lookup_failure_rolls_back_only_its_savepoint_before_propagating():
    conn = Connection()
    conn.state = TRANSACTION_STATUS_INTRANS
    conn.pending = ["caller write"]
    conn.fail_lookup = True
    with pytest.raises(ValueError, match="lookup unavailable"):
        schema.ensure_schema(conn)
    assert conn.pending == ["caller write"] and conn.state == TRANSACTION_STATUS_INTRANS
    assert any(s.startswith("ROLLBACK TO SAVEPOINT schema_lookup_") for s in conn.statements)
    assert conn.ddl_count == 0 and not schema._schema_ensured


@pytest.mark.parametrize("second_target", ["a", "b"])
def test_concurrent_initialization_is_serial_only_for_same_target(second_target):
    first, second = Connection("a"), Connection(second_target)
    first.ddl_started, first.ddl_release = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(schema.ensure_schema, first)
        assert first.ddl_started.wait(1)
        b = workers.submit(schema.ensure_schema, second)
        assert second.metadata_ready.wait(1)
        if second_target == "b":
            b.result(timeout=1)
            assert second.ddl_count == 1
        else:
            assert second.ddl_count == 0
        first.ddl_release.set()
        a.result(timeout=2)
        b.result(timeout=2)
    assert second.ddl_count == (0 if second_target == "a" else 1)
