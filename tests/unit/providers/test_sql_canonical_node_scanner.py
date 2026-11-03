from fs.sql_adapter.node_scanner import SQLCanonicalNodeScanner


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Connection:
    closed = 0

    def __init__(self, rows):
        self.cursor_value = _Cursor(rows)
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_value

    def rollback(self):
        self.rollbacks += 1


class _Pool:
    def __init__(self, rows):
        self.connection = _Connection(rows)
        self.returns = 0

    def get_connection(self, *, validate=True):
        assert validate is False
        return self.connection

    def return_connection(self, connection):
        assert connection is self.connection
        self.returns += 1


def test_scanner_pages_active_nodes_by_owner_space_without_uri_shape_assumptions():
    pool = _Pool(
        [
            (
                "ctx://acct/resources/guide",
                "RESOURCE",
                "resource",
                2,
                "agent_main",
                "guide",
                "",
                "BM25 setup",
                '{"source":"docs"}',
            )
        ]
    )
    scanner = SQLCanonicalNodeScanner(pool)

    nodes = list(
        scanner.scan_active_nodes(
            account_id="acct",
            owner_spaces=["agent_main"],
            page_size=100,
        )
    )

    assert [node.uri for node in nodes] == ["ctx://acct/resources/guide"]
    assert nodes[0].metadata == {"source": "docs"}
    select = next(
        (sql, params)
        for sql, params in pool.connection.cursor_value.executed
        if "FROM context_nodes" in sql
    )
    assert "status='ACTIVE'" in select[0]
    assert select[1][1] == ["agent_main"]
    assert any(
        params == ("statement_timeout", "30000")
        for _sql, params in pool.connection.cursor_value.executed
    )
    assert pool.returns == 1


def test_projection_quiescence_blocks_only_runnable_events():
    pool = _Pool([(1,)])
    scanner = SQLCanonicalNodeScanner(pool)

    assert scanner.projection_is_quiescent(account_id="acct") is False

    query, params = next(
        (sql, params)
        for sql, params in pool.connection.cursor_value.executed
        if "FROM outbox_events" in sql
    )
    assert "'PENDING','PROCESSING'" in query
    assert "FAILED" not in query
    assert params == ("acct", "EXTRACT_AGENT_MEMORY")
    assert any(
        params == ("statement_timeout", "30000")
        for _sql, params in pool.connection.cursor_value.executed
    )
    assert pool.connection.rollbacks == 1
    assert pool.returns == 1


def test_projection_quiescence_allows_account_without_unfinished_events():
    pool = _Pool([])

    assert SQLCanonicalNodeScanner(pool).projection_is_quiescent(account_id="acct")


def test_projection_quiescence_preserves_query_error_when_rollback_fails():
    pool = _Pool([])
    pool.connection.cursor_value.execute = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("query-original")
    )
    pool.connection.rollback = lambda: (_ for _ in ()).throw(
        RuntimeError("rollback-secondary")
    )

    try:
        SQLCanonicalNodeScanner(pool).projection_is_quiescent(account_id="acct")
    except RuntimeError as exc:
        assert str(exc) == "query-original"
    else:
        raise AssertionError("projection check should propagate query failure")


def test_scanner_preserves_query_error_when_rollback_also_fails():
    pool = _Pool([])

    def fail_execute(_sql, _params=None):
        raise RuntimeError("query-original")

    pool.connection.cursor_value.execute = fail_execute
    pool.connection.rollback = lambda: (_ for _ in ()).throw(
        RuntimeError("rollback-masked")
    )

    scanner = SQLCanonicalNodeScanner(pool)
    try:
        list(
            scanner.scan_active_nodes(
                account_id="acct",
                owner_spaces=["user_u1"],
                page_size=10,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "query-original"
    else:
        raise AssertionError("scanner query should fail")


def test_scanner_streams_scopes_larger_than_the_old_total_node_limit():
    rows = [
        (
            f"ctx://acct/resources/{index:05d}",
            "RESOURCE",
            "resource",
            2,
            "agent_main",
            "",
            "",
            "content",
            {},
        )
        for index in range(10_001)
    ]
    scanner = SQLCanonicalNodeScanner(_Pool(rows))

    nodes = list(
        scanner.scan_active_nodes(
            account_id="acct",
            owner_spaces=["agent_main"],
            page_size=500,
        )
    )

    assert len(nodes) == 10_001
