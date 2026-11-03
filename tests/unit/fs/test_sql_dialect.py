import pytest

from fs.sql_adapter import pool as pool_module
from fs.sql_adapter import schema as schema_module
from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.opengauss_dialect import OpenGaussDialect
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.postgresql_dialect import PostgreSQLDialect


class _Cursor:
    def __init__(self):
        self.statements = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchone(self):
        if self.statements[-1][0].startswith("SELECT pg_catalog.inet_server_addr"):
            return ("127.0.0.1", 5432, "test", 1, "tester", "public", ["pg_catalog", "public"])
        return ("PG",)

    def fetchall(self):
        return [("pg_catalog", 11), ("public", 2200)]


class _Connection:
    def __init__(self):
        self.autocommit = False
        self.cursor_instance = _Cursor()

    def cursor(self):
        return self.cursor_instance

    def get_transaction_status(self):
        return 0


def test_postgresql_schema_is_applied_through_dialect():
    conn = _Connection()
    schema_module._schema_ensured.clear()

    schema_module.ensure_schema(conn, "postgresql")

    schema_sql = conn.cursor_instance.statements[-1][0]
    assert "current_setting('app.account_id', true)" in schema_sql
    assert conn.autocommit is False
    schema_module._schema_ensured.clear()


def test_opengauss_schema_preserves_tenant_write_isolation():
    conn = _Connection()
    schema_module._schema_ensured.clear()

    schema_module.ensure_schema(conn, "opengauss")

    assert any(statement[0] == "SHOW sql_compatibility" for statement in conn.cursor_instance.statements)
    schema_sql = conn.cursor_instance.statements[-1][0]
    assert "current_setting('app.account_id')" in schema_sql
    assert "current_setting('app.account_id', true)" not in schema_sql
    assert "WITH CHECK (account_id" not in schema_sql
    assert "CHECK (account_id = current_setting('app.account_id')) NOT VALID" in schema_sql
    assert schema_sql.count("_tenant_write_guard") == 11
    assert conn.autocommit is False
    schema_module._schema_ensured.clear()


def test_shared_pool_exposes_postgresql_dialect(monkeypatch):
    monkeypatch.setattr(pool_module, "_HAS_PSYCOPG2", True)

    pool = SharedConnectionPool("dbname=test")

    assert pool.dialect == "postgresql"
    assert isinstance(pool.sql_dialect, PostgreSQLDialect)


def test_dialect_factory_returns_postgresql_implementation():
    dialect = resolve_sql_dialect("postgres")

    assert isinstance(dialect, PostgreSQLDialect)


def test_dialect_factory_returns_opengauss_implementation():
    assert isinstance(resolve_sql_dialect("opengauss"), OpenGaussDialect)


def test_dialect_factory_rejects_unknown_implementation():
    with pytest.raises(ValueError, match="postgresql or opengauss"):
        resolve_sql_dialect("mysql")


def test_workspace_root_sql_is_owned_by_postgresql_dialect():
    cursor = _Cursor()

    PostgreSQLDialect().ensure_workspace_root(cursor, "acct", "root")

    assert "ON CONFLICT" in cursor.statements[0][0]


def test_opengauss_uses_merge_and_polling():
    cursor = _Cursor()
    dialect = OpenGaussDialect()

    dialect.ensure_workspace_root(cursor, "acct", "root")
    dialect.notify_outbox(cursor, "ogmem_outbox", "{}")

    assert "MERGE INTO" in cursor.statements[0][0]
    assert dialect.supports_batch_claim is False
    assert len(cursor.statements) == 1


def test_opengauss_requires_pg_compatibility():
    cursor = _Cursor()
    cursor.fetchone = lambda: ("A",)

    with pytest.raises(RuntimeError, match="DBCOMPATIBILITY 'PG'"):
        OpenGaussDialect().validate_connection(cursor)


def test_postgresql_declares_notify_capability():
    cursor = _Cursor()
    dialect = PostgreSQLDialect()

    dialect.notify_outbox(cursor, "ogmem_outbox", "{}")

    assert dialect.supports_batch_claim is True
    assert "pg_notify" in cursor.statements[0][0]
