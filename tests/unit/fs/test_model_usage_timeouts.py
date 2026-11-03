from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.model_usage_store import SQLModelUsageStore


@pytest.mark.parametrize("dialect", ["postgresql", "opengauss"])
def test_usage_insert_keeps_statement_bound_for_ordinary_roles(dialect):
    store = SQLModelUsageStore.__new__(SQLModelUsageStore)
    store.pool = SimpleNamespace(sql_dialect=resolve_sql_dialect(dialect))
    cursor = MagicMock()
    cursor.mogrify.return_value = b"(fixture)"
    store._insert_events(
        cursor,
        [
            dict(
                id="event",
                model="model",
                kind="llm",
                at="2026-10-06",
                allocations={("account", "user:alice"): {}},
            )
        ],
    )
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert statements[0] == "SET LOCAL statement_timeout = '5s'"
    if dialect == "postgresql":
        assert statements[1] == "SET LOCAL lock_timeout = '5s'"
    else:
        assert len(statements) == 2
        assert not any(isinstance(s, str) and "lockwait_timeout" in s for s in statements)
    assert statements[-1].startswith(b"INSERT INTO model_usage_events")
