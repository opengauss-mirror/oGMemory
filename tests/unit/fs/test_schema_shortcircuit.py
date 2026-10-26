"""Unit tests for ensure_schema DDL short-circuit and advisory locking.

Covers the parallel cold-start contract:
* DDL is skipped when ``graph_nodes`` already exists (the keystone table).
* DDL runs when the table is absent (fresh database).
* Non-postgres backends (fake/mock connections) skip the advisory lock.
"""

from fs.sql_adapter import schema as schema_module


class _FakeCursor:
    """Captures executed SQL and serves queued fetchone results."""

    def __init__(self, results=None):
        self.statements: list[tuple[str, object]] = []
        self._results = list(results or [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchone(self):
        if self._results:
            return self._results.pop(0)
        return None


class _FakeConnection:
    """Connection fake; ``server_version`` makes it look like real psycopg."""

    def __init__(self, *, server_version=None, fetch_results=None):
        self._cursor = _FakeCursor(fetch_results)
        self.autocommit = False
        if server_version is not None:
            self.server_version = server_version

    @property
    def cursor_instance(self):
        return self._cursor

    def cursor(self):
        return self._cursor

    def executed_sql(self):
        return [sql for sql, _ in self._cursor.statements]


def _clear_flag():
    schema_module._schema_ensured.clear()


def test_skips_ddl_when_graph_nodes_exists():
    conn = _FakeConnection(server_version=160001, fetch_results=[(True,)])
    _clear_flag()

    schema_module.ensure_schema(conn, "postgresql")

    sql = conn.executed_sql()
    assert any("pg_advisory_lock" in s for s in sql)
    assert any("to_regclass" in s for s in sql)
    assert any("pg_advisory_unlock" in s for s in sql)
    # No DDL executed: the schema script never runs.
    assert not any("CREATE TABLE" in s for s in sql)
    assert "postgresql" in schema_module._schema_ensured
    assert conn.autocommit is False
    _clear_flag()


def test_runs_ddl_when_graph_nodes_absent():
    conn = _FakeConnection(server_version=160001, fetch_results=[(False,)])
    _clear_flag()

    schema_module.ensure_schema(conn, "postgresql")

    sql = conn.executed_sql()
    assert any("pg_advisory_lock" in s for s in sql)
    assert any("to_regclass" in s for s in sql)
    # DDL executed after the short-circuit check misses.
    assert any("CREATE TABLE" in s for s in sql)
    assert any("pg_advisory_unlock" in s for s in sql)
    assert conn.autocommit is False
    _clear_flag()


def test_non_postgres_backend_skips_advisory_lock():
    # No server_version -> not a real psycopg connection.
    conn = _FakeConnection(fetch_results=[(True,)])
    _clear_flag()

    schema_module.ensure_schema(conn, "postgresql")

    sql = conn.executed_sql()
    assert not any("pg_advisory_lock" in s for s in sql)
    assert not any("pg_advisory_unlock" in s for s in sql)
    assert not any("to_regclass" in s for s in sql)
    # DDL still applied for mock backends (tests drive it directly).
    assert any("CREATE TABLE" in s for s in sql)
    assert conn.autocommit is False
    _clear_flag()


def test_process_flag_shortcircuits_everything():
    conn = _FakeConnection(server_version=160001, fetch_results=[(True,)])
    schema_module._schema_ensured.add("postgresql")

    schema_module.ensure_schema(conn, "postgresql")

    assert conn.executed_sql() == []
    assert conn.autocommit is False
    _clear_flag()


def test_lock_released_when_ddl_fails():
    class _FailingCursor(_FakeCursor):
        def execute(self, sql, params=None):
            if "CREATE TABLE" in sql:
                raise RuntimeError("DDL failed")
            super().execute(sql, params)

    cursor = _FailingCursor([(False,)])
    conn = _FakeConnection(server_version=160001)
    conn._cursor = cursor
    _clear_flag()

    try:
        schema_module.ensure_schema(conn, "postgresql")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected DDL failure to propagate")

    sql = conn.executed_sql()
    assert any("pg_advisory_unlock" in s for s in sql)
    assert conn.autocommit is False
    _clear_flag()
