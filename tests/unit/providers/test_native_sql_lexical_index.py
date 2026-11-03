from __future__ import annotations

import threading

import pytest

from core.default_lexical_analyzer import DefaultLexicalAnalyzer
from core.lexical import LexicalRecord
from providers.lexical_index.native_sql_index import (
    DatabaseNativeLexicalIndex,
    OpenGaussLexicalIndex,
    _validate_native_table_prefix,
)
from providers.lexical_index.sql_index import _schema_lock_key


class FakeCursor:
    def __init__(self, *, one=None, rows=()):
        self.one = one
        self.rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.one

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    closed = 0

    def __init__(self, cursor):
        self.cursor_value = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1
        self.closed = 1


def _index(cursor):
    index = OpenGaussLexicalIndex.__new__(OpenGaussLexicalIndex)
    index._table = "lex"
    index._provider_name = "opengauss_native"
    index._query_timeout_ms = 5000
    index._max_query_chars = 32_768
    index._connect_timeout_seconds = 10
    index._operation_timeout_ms = 300000
    index._analyzer = DefaultLexicalAnalyzer()
    index._max_query_terms = 64
    index._limit_metrics_lock = threading.Lock()
    index._query_truncations = 0
    index._engine = "opengauss_bm25"
    index._query_count = 0
    index._query_seconds_total = 0.0
    index._last_query_seconds = 0.0
    index._max_query_seconds = 0.0
    index._metrics_lock = threading.Lock()
    index._last_error = None
    index._bm25_topk = None
    index._bm25_k1 = None
    index._bm25_b = None
    index._bm25_taat = None
    connection = FakeConnection(cursor)
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    return index, connection


def test_native_identifiers_reject_sql_fragments_and_oversized_prefixes():
    assert _validate_native_table_prefix("x" * 39) == "x" * 39
    with pytest.raises(ValueError, match="39 ASCII characters"):
        _validate_native_table_prefix("x" * 40)


def test_native_provider_rejects_shared_pool_without_matching_timeouts():
    class UnsafePool:
        connect_timeout_seconds = None
        session_statement_timeout_ms = None

    with pytest.raises(ValueError, match="shared native lexical pools"):
        OpenGaussLexicalIndex(
            connection_string="postgres://unused",
            pool=UnsafePool(),
            connect_timeout_seconds=10,
            operation_timeout_ms=300000,
        )


def test_native_provider_rejects_invalid_query_character_bound():
    with pytest.raises(ValueError, match="max_query_chars"):
        OpenGaussLexicalIndex(
            connection_string="postgres://unused",
            max_query_chars=0,
        )


def test_native_record_projection_bounds_content():
    index = DatabaseNativeLexicalIndex.__new__(DatabaseNativeLexicalIndex)
    index._max_document_chars = 5
    index._limit_metrics_lock = threading.Lock()
    index._document_truncations = 0
    record = LexicalRecord(
        id="m1",
        uri="ctx://acct/m1",
        level=0,
        text="database retrieval",
        account_id="acct",
        owner_space="user:u1",
        metadata={},
    )

    prepared = index._prepare_record(record)

    assert prepared.content == "datab"
    assert index._document_truncations == 1


def test_native_query_bound_only_rewrites_queries_over_limit():
    index = DatabaseNativeLexicalIndex.__new__(DatabaseNativeLexicalIndex)
    index._analyzer = DefaultLexicalAnalyzer()
    index._max_query_terms = 2
    index._max_query_chars = 100
    index._limit_metrics_lock = threading.Lock()
    index._query_truncations = 0

    assert index._bounded_query("Coffee, please!") == "Coffee, please!"
    assert index._bounded_query("one two three") == "one two"
    assert index._query_truncations == 1


def test_native_query_bound_limits_input_before_analysis():
    class RecordingAnalyzer:
        def __init__(self):
            self.input_length = 0

        def analyze_query(self, query):
            self.input_length = len(query)
            return [query]

    index = DatabaseNativeLexicalIndex.__new__(DatabaseNativeLexicalIndex)
    index._analyzer = RecordingAnalyzer()
    index._max_query_terms = 64
    index._max_query_chars = 8
    index._limit_metrics_lock = threading.Lock()
    index._query_truncations = 0

    assert index._bounded_query("x" * 100) == "x" * 8
    assert index._analyzer.input_length == 8
    assert index._query_truncations == 1


def test_native_query_bound_limits_unicode_expansion_after_analysis():
    index = DatabaseNativeLexicalIndex.__new__(DatabaseNativeLexicalIndex)
    index._analyzer = DefaultLexicalAnalyzer()
    index._max_query_terms = 1
    index._max_query_chars = 4
    index._limit_metrics_lock = threading.Lock()
    index._query_truncations = 0

    assert index._bounded_query("\ufb03\ufb03 x") == "ffif"
    assert index._query_truncations == 1


@pytest.mark.parametrize(
    ("catalog_row", "expected"),
    [((True, True), True), ((True, False), False), ((False, True), False)],
)
def test_bm25_capability_requires_access_method_and_operator(catalog_row, expected):
    cursor = FakeCursor(one=catalog_row)
    index, _ = _index(cursor)

    assert index._database_supports_bm25() is expected
    sql, _ = cursor.executed[-1]
    assert "am.amname='bm25'" in sql
    assert "opr.oprname='<&>'" in sql
    assert "pg_operator_is_visible(opr.oid)" in sql


def test_provider_fails_when_native_bm25_is_unavailable():
    index = OpenGaussLexicalIndex.__new__(OpenGaussLexicalIndex)
    index._database_supports_bm25 = lambda: False

    with pytest.raises(RuntimeError, match="native BM25"):
        index._ensure_schema()


def test_provider_fails_when_created_bm25_index_is_not_ready(monkeypatch):
    index = OpenGaussLexicalIndex.__new__(OpenGaussLexicalIndex)
    index._database_supports_bm25 = lambda: True
    index._create_bm25_index = lambda: None
    index._native_projection_health = lambda: {
        "schema_compatible": True,
        "index_ready": False,
    }
    monkeypatch.setattr(
        "providers.lexical_index.native_sql_index.SQLNativeProjectionStore._ensure_schema",
        lambda _self: None,
    )

    with pytest.raises(RuntimeError, match="post-create validation"):
        index._ensure_schema()


def test_native_schema_identity_includes_document_storage_limit():
    index = OpenGaussLexicalIndex.__new__(OpenGaussLexicalIndex)
    index._engine = "opengauss_bm25"
    index._max_document_chars = 5
    first = index._schema_identity
    index._max_document_chars = 500

    assert index._schema_identity != first


def test_bm25_index_creation_uses_native_access_method():
    cursor = FakeCursor()
    index, connection = _index(cursor)

    index._create_bm25_index()

    assert cursor.executed[0] == (
        "SELECT pg_advisory_xact_lock(%s)",
        (_schema_lock_key("lex"),),
    )
    assert any("USING bm25(content)" in sql for sql, _ in cursor.executed)
    assert not any("USING GIN" in sql for sql, _ in cursor.executed)
    assert connection.commits == 1


@pytest.mark.parametrize("table_name", ["lex", "scene_lexical_v1"])
def test_bm25_search_applies_scope_ranking_limits_and_session_settings(table_name):
    cursor = FakeCursor(
        rows=[("m1", '{"uri":"ctx://acct/m1","level":0}', 1.25)]
    )
    index, connection = _index(cursor)
    index._table = table_name
    index._bm25_topk = 25
    index._bm25_k1 = 1.4
    index._bm25_b = 0.7
    index._bm25_taat = True

    hits = index.search(
        "database",
        10,
        {"account_id": "acct", "owner_space": "user:u1"},
    )

    setting_calls = [params for sql, params in cursor.executed if "set_config" in sql]
    assert len(setting_calls) == 1
    settings = list(zip(setting_calls[0][::2], setting_calls[0][1::2]))
    assert ("bm25_topk", "25") in settings
    assert ("bm25_k1", "1.4") in settings
    assert ("bm25_b", "0.7") in settings
    assert ("enable_bm25_taat", "on") in settings
    assert ("statement_timeout", "5000") in settings
    sql, params = cursor.executed[-1]
    assert f"indexscan (d idx_{table_name}_native_bm25)" in sql
    assert f"FROM {table_name}_docs d" in sql
    assert "ORDER BY d.content <&> %s DESC" in sql
    assert params == ("database", "acct", "user:u1", "database", 10)
    assert hits[0].score == 1.25
    assert hits[0].id == "m1"
    assert hits[0].metadata == {"uri": "ctx://acct/m1", "level": 0}
    assert connection.rollbacks == 1


def test_bm25_search_preserves_query_error_when_rollback_also_fails():
    class FailingCursor(FakeCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            if "ORDER BY" in sql:
                raise RuntimeError("original query error")

    index, connection = _index(FailingCursor())

    def fail_rollback():
        connection.rollbacks += 1
        raise RuntimeError("rollback error")

    connection.rollback = fail_rollback
    with pytest.raises(RuntimeError, match="original query error"):
        index.search("coffee", 5, {"account_id": "acct"})

    assert index._last_error == "RuntimeError"
    assert connection.rollbacks == 1
    assert connection.closes == 1
