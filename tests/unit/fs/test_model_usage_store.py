"""Usage persistence must work with an ordinary openGauss service account."""

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import psycopg2
import pytest

from fs.sql_adapter import model_usage_store as store_module
from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.model_usage_store import SQLModelUsageStore


@pytest.mark.parametrize("dialect", ["opengauss", "postgresql"])
@pytest.mark.parametrize("use_deadline", [False, True])
def test_record_usage_without_admin_lock_timeout(dialect, use_deadline, monkeypatch):
    cursor = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.connection = connection
    cursor.mogrify.return_value = b"('event-1')"

    def execute(sql, params=None):
        if dialect == "opengauss" and isinstance(sql, str) and "lockwait_timeout" in sql:
            raise psycopg2.errors.InsufficientPrivilege(
                "permission denied to set parameter lockwait_timeout"
            )

    cursor.execute.side_effect = execute
    bounded_connection = MagicMock()
    bounded_connection.__enter__.return_value = connection
    pool = SimpleNamespace(
        sql_dialect=resolve_sql_dialect(dialect),
        get_connection=Mock(return_value=connection),
        return_connection=Mock(),
        deadline_connection=Mock(return_value=bounded_connection),
    )
    # Schema initialization is independent of the event-writing regression.
    store = SQLModelUsageStore.__new__(SQLModelUsageStore)
    store.pool = pool
    wait = Mock()
    monkeypatch.setattr(store_module, "wait_for_postgres", wait)
    event = {
        "id": "event-1",
        "model": "test-model",
        "kind": "llm",
        "at": datetime.now(UTC),
        "allocations": {("acct-1", "user:alice"): {"llm_calls": 1, "input_tokens": 10}},
    }
    deadline = time.monotonic() + 10 if use_deadline else None

    store.record(event, deadline=deadline)

    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert "SET LOCAL statement_timeout = '5s'" in statements
    assert not any("lockwait_timeout" in sql for sql in statements if isinstance(sql, str))
    assert ("SET LOCAL lock_timeout = '5s'" in statements) is (dialect == "postgresql")
    inserts = [sql for sql in statements if isinstance(sql, bytes)]
    assert len(inserts) == 1
    assert inserts[0].startswith(b"INSERT INTO model_usage_events ")
    assert cursor.mogrify.call_args.args[1][:5] == (
        "event-1", "acct-1", "user:alice", "test-model", "llm",
    )
    if use_deadline:
        pool.deadline_connection.assert_called_once_with(deadline)
        assert statements[0] == "BEGIN"
        assert statements[-1] == "COMMIT"
        assert wait.call_count == len(statements)
        assert all(call.args == (connection, deadline) for call in wait.call_args_list)
        pool.get_connection.assert_not_called()
    else:
        connection.commit.assert_called_once()
        pool.return_connection.assert_called_once_with(connection)
        wait.assert_not_called()
