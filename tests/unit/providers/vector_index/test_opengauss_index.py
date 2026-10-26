"""Unit tests for OpenGaussVectorIndex.

Uses mock database connection for isolated testing.
For integration tests with real database, see scripts/test_opengauss_index.py
"""

from unittest.mock import patch

import pytest

from core.models import IndexRecord, SeedHit, TypedQuery


# Mock Json to just pass through the input value
class MockJson:
    """Mock for psycopg2.extras.Json that just returns its input."""
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        if isinstance(other, MockJson):
            return self.obj == other.obj
        return self.obj == other


@pytest.fixture(autouse=True)
def mock_json_adapter():
    with patch("providers.vector_index.opengauss_index.Json", MockJson):
        yield


class MockCursor:
    """Mock database cursor for testing."""

    def __init__(self):
        self.executed_sql: list[str] = []
        self.executed_params: list[tuple] = []
        self.results: list[tuple | dict] = []  # Can be tuples or dicts (for RealDictCursor)
        self.rowcount = 0

    def execute(self, sql: str, params: tuple = None):
        self.executed_sql.append(sql)
        if params:
            self.executed_params.append(params)

    def fetchone(self):
        if self.results:
            return self.results.pop(0)
        return None

    def fetchall(self):
        results = self.results
        self.results = []
        # Filter out single-value tuples (table existence checks)
        # Return only dicts (actual query results for RealDictCursor)
        return [r for r in results if isinstance(r, dict)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, cursor: MockCursor = None):
        self._cursor = cursor or MockCursor()
        self.committed = False
        self._closed = False
        self.rolled_back = False

    def cursor(self, cursor_factory=None, **kwargs):
        # Accept cursor_factory and other kwargs for compatibility
        # but don't use them in the mock
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, value):
        self._closed = value


class TestVectorToString:
    """Tests for _vec_literal utility function."""

    def test_simple_vector(self):
        """Test conversion of simple vector."""
        from providers.vector_index.opengauss_index import _vec_literal

        result = _vec_literal([1.0, 2.0, 3.0])
        assert result == "[1.00000000,2.00000000,3.00000000]"

    def test_negative_values(self):
        """Test conversion with negative values."""
        from providers.vector_index.opengauss_index import _vec_literal

        result = _vec_literal([-1.5, 0.0, 2.5])
        assert result == "[-1.50000000,0.00000000,2.50000000]"

    def test_empty_vector(self):
        """Test conversion of empty vector."""
        from providers.vector_index.opengauss_index import _vec_literal

        result = _vec_literal([])
        assert result == "[]"

    def test_float_precision(self):
        """Test that float values are formatted to 8 decimal places."""
        from providers.vector_index.opengauss_index import _vec_literal

        result = _vec_literal([0.123456789, 1.0])
        assert "0.12345679" in result  # Rounded to 8 decimal places


class TestUpsert:
    """Tests for upsert operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_single_record(self, mock_psycopg2):
        """Test upserting a single record."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists check
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        record = IndexRecord(
            id="test_id_1",
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="Test abstract",
            filters={"account_id": "test", "owner_space": "users/u1"},
            metadata={"_embedding": [0.1] * 1536},
        )

        index.upsert([record])

        assert len(mock_cursor.executed_sql) == 1
        # Implementation changed to MERGE INTO (openGauss compatible)
        assert "MERGE INTO" in mock_cursor.executed_sql[0]
        assert "t.account_id = s.account_id" in mock_cursor.executed_sql[0]
        assert mock_cursor.executed_params[0][:2] == (record.id, "test")
        update_clause = mock_cursor.executed_sql[0].split("WHEN MATCHED")[1].split("WHEN NOT MATCHED")[0]
        assert "account_id =" not in update_clause
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_multiple_records(self, mock_psycopg2):
        """Test upserting multiple records."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        records = [
            IndexRecord(
                id=f"test_id_{i}",
                uri="ctx://test/users/u1/memories/profile",
                level=i,
                text=f"Text {i}",
                filters={"account_id": "test", "owner_space": "users/u1"},
                metadata={"_embedding": [0.1] * 1536},
            )
            for i in range(3)
        ]

        index.upsert(records)

        assert len(mock_cursor.executed_sql) == 3
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_empty_list(self, mock_psycopg2):
        """Test upserting empty list does nothing."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.upsert([])

        assert not mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_missing_embedding_logs_warning(self, mock_psycopg2):
        """Test that missing embedding logs warning and skips record."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        record = IndexRecord(
            id="test_id_1",
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="Test abstract",
            filters={"account_id": "test", "owner_space": "users/u1"},
            metadata={},  # Missing _embedding
        )

        # Missing embedding now logs warning instead of raising error
        index.upsert([record])

        # No SQL should be executed since record is skipped
        assert len(mock_cursor.executed_sql) == 0


class TestDelete:
    """Tests for delete operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_single_id(self, mock_psycopg2):
        """Test deleting a single record by ID."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 1
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete(["id1"])

        assert len(mock_cursor.executed_sql) == 1
        assert "DELETE FROM" in mock_cursor.executed_sql[0]
        assert "ANY(%s)" in mock_cursor.executed_sql[0]

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_multiple_ids(self, mock_psycopg2):
        """Test deleting multiple records."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 3
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete(["id1", "id2", "id3"])

        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_empty_list(self, mock_psycopg2):
        """Test deleting empty list does nothing."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete([])

        assert not mock_conn.committed


class TestSearch:
    """Tests for search operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_returns_results(self, mock_psycopg2):
        """Test search returns SeedHit results."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [
            (True,),  # Table exists check
            {
                "uri": "ctx://test/users/u1/memories/profile",
                "score": 0.85,
                "level": 0,
                "text": "Test abstract",
                "metadata": {"context_type": "MEMORY", "category": "profile"},
                "filters": {"owner_space": "users/u1", "account_id": "test"},
                "parent_uri": None,
                "has_overview": True,
                "has_content": True,
                "active_count": 0,
                "updated_at": None,
            },
            {
                "uri": "ctx://test/users/u1/memories/pref",
                "score": 0.72,
                "level": 0,
                "text": "Pref abstract",
                "metadata": {"context_type": "MEMORY", "category": "preference"},
                "filters": {"owner_space": "users/u1", "account_id": "test"},
                "parent_uri": None,
                "has_overview": False,
                "has_content": True,
                "active_count": 0,
                "updated_at": None,
            },
        ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=["profile"],
            account_id="test",
            owner_space="users/u1",
            top_k=10,
        )

        results = index.search(query)

        assert len(results) == 2
        assert isinstance(results[0], SeedHit)
        assert results[0].uri == "ctx://test/users/u1/memories/profile"
        assert results[0].score == 0.85
        assert results[0].metadata["category"] == "profile"

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_generates_mock_vector_when_no_embedder(self, mock_psycopg2):
        """Test that search generates mock vector when no embedder is provided."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=[],
            account_id="test",
            top_k=10,
        )

        # Should not raise an error - mock vector is generated internally
        index.search(query)

        # Verify SQL was executed
        assert len(mock_cursor.executed_sql) == 1

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_with_category_filter(self, mock_psycopg2):
        """Test search applies category filter."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=["profile", "preference"],
            account_id="test",
            top_k=10,
        )

        index.search(query)

        sql = mock_cursor.executed_sql[0]
        # PostgreSQL uses = ANY() for array comparison, not IN
        assert "metadata->>'category' = ANY(%(f_category)s)" in sql

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_with_owner_space_filter(self, mock_psycopg2):
        """Test search applies owner_space filter."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=[],
            account_id="test",
            owner_space="users/alice",
            top_k=10,
        )

        index.search(query)

        sql = mock_cursor.executed_sql[0]
        assert "COALESCE(owner_space, filters->>'owner_space') = %(f_owner_space)s" in sql


class TestCount:
    """Tests for count operation.

    Note: count() is a test-only method on InMemoryVectorIndex, not part of
    the VectorIndex protocol. These tests are skipped for OpenGaussVectorIndex.
    """

    @pytest.mark.skip(reason="count() method not part of VectorIndex protocol")
    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_count_all_records(self, mock_psycopg2):
        """Test counting all records."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), (42,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        count = index.count()

        assert count == 42
        assert "SELECT COUNT(*)" in mock_cursor.executed_sql[0]

    @pytest.mark.skip(reason="count() method not part of VectorIndex protocol")
    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_count_by_account(self, mock_psycopg2):
        """Test counting records for specific account."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), (10,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        count = index.count(account_id="test_account")

        assert count == 10
        assert "filters->>'account_id'" in mock_cursor.executed_sql[0]


class TestClear:
    """Tests for clear operation.

    Note: clear() is a test-only method on InMemoryVectorIndex, not part of
    the VectorIndex protocol. These tests are skipped for OpenGaussVectorIndex.
    """

    @pytest.mark.skip(reason="clear() method not part of VectorIndex protocol")
    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_clear_all_records(self, mock_psycopg2):
        """Test clearing all records."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 100
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        deleted = index.clear()

        assert deleted == 100
        assert mock_conn.committed

    @pytest.mark.skip(reason="clear() method not part of VectorIndex protocol")
    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_clear_by_account(self, mock_psycopg2):
        """Test clearing records for specific account."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 25
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        deleted = index.clear(account_id="test_account")

        assert deleted == 25
        assert "filters->>'account_id'" in mock_cursor.executed_sql[0]


class TestConnectionPool:
    """Tests for connection pool behavior."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_connection_reuse(self, mock_psycopg2):
        """Test that connections are reused from pool."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_conn1 = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn1

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 2
        index._pool = []
        index._initialized = True

        conn1 = index._get_connection()
        index._return_connection(conn1)
        conn2 = index._get_connection()

        assert conn1 is conn2
        assert len(index._pool) == 0

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_close_all_connections(self, mock_psycopg2):
        """Test that close() closes all pooled connections."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        conn = index._get_connection()
        index._return_connection(conn)

        index.close()

        assert len(index._pool) == 0


class TestContextManager:
    """Tests for context manager protocol."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_context_manager_closes_on_exit(self, mock_psycopg2):
        """Test that context manager closes connections on exit."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        with index:
            conn = index._get_connection()
            index._return_connection(conn)

        assert len(index._pool) == 0


@pytest.mark.parametrize(
    ("dialect", "first_statement"),
    [
        ("opengauss", "CREATE TABLE IF NOT EXISTS vector_index"),
        ("postgresql", "CREATE EXTENSION IF NOT EXISTS vector"),
    ],
)
def test_shared_pool_dialect_selects_vector_bootstrap(monkeypatch, dialect, first_statement):
    from fs.sql_adapter.pool import SharedConnectionPool
    from providers.vector_index.opengauss_index import OpenGaussVectorIndex

    cursor = MockCursor()
    connection = MockConnection(cursor)
    pool = SharedConnectionPool("host=storage dbname=memory", dialect=dialect)
    monkeypatch.setattr(pool, "get_connection", lambda: connection)
    returned = []
    monkeypatch.setattr(pool, "return_connection", returned.append)

    index = OpenGaussVectorIndex(pool=pool, dimension=3)

    assert index.sql_dialect.name == dialect
    assert index._shared_pool is pool
    assert cursor.executed_sql[0].strip().startswith(first_statement)
    assert any("embedding     vector(3)" in sql for sql in cursor.executed_sql)
    assert any("USING hnsw (embedding vector_cosine_ops)" in sql for sql in cursor.executed_sql)
    assert connection.committed
    assert returned == [connection]


@pytest.mark.skip(reason="Auto-initialization feature not implemented yet")
class TestAutoInit:
    """Tests for auto-initialization.

    Note: These tests are skipped because the current OpenGaussVectorIndex
    implementation does not include auto-initialization of tables and indexes.
    The table must be created manually before using the index.
    """

    def setUp(self):
        # Set the flag before importing to avoid ImportError
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_extension(self, mock_psycopg2):
        """Test that auto_init creates pgvector extension.

        Note: The implementation doesn't actually create the extension yet,
        this test documents the expected behavior based on the docstring.
        """
        # Ensure the flag is set before import
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test")

        # The implementation creates the table and indexes
        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "CREATE TABLE vector_index" in sql_commands
        assert "CREATE INDEX IF NOT EXISTS idx_vector_index_embedding" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_table_with_dimension(self, mock_psycopg2):
        """Test that auto_init creates table with correct dimension."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test", dimension=768)

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "vector(768)" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_skips_if_table_exists(self, mock_psycopg2):
        """Test that auto_init skips table creation if table already exists."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        # When table exists, only the check query is run
        assert "CREATE TABLE" not in sql_commands
        assert "CREATE INDEX" not in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_hnsw_index(self, mock_psycopg2):
        """Test that auto_init creates HNSW index."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "USING hnsw" in sql_commands
        assert "vector_cosine_ops" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_filter_indexes(self, mock_psycopg2):
        """Test that auto_init creates filter indexes."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "filters->>'account_id'" in sql_commands
        assert "filters->>'owner_space'" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_custom_table_name(self, mock_psycopg2):
        """Test that auto_init uses custom table name."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test", table_name="custom_vectors")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "custom_vectors" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_trigger(self, mock_psycopg2):
        """Test that auto_init creates auto-update trigger."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussVectorIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "CREATE TRIGGER" in sql_commands
        assert "updated_at" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_ensure_table_exists_is_idempotent(self, mock_psycopg2):
        """Test that _ensure_table_exists is idempotent."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        first_count = len(mock_cursor.executed_sql)

        index._ensure_table_exists()
        second_count = len(mock_cursor.executed_sql)

        assert first_count == second_count


class TestSQLInjectionProtection:
    """Tests for SQL injection protection via table name validation."""

    def setup_method(self):
        """Ensure the module flag is set before each test."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_reject_table_name_with_semicolon(self, mock_psycopg2):
        """Test that table names with SQL injection patterns are rejected."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        with pytest.raises(ValueError, match="Invalid table name"):
            OpenGaussVectorIndex("postgres://test", table_name="vector_index; DROP TABLE--")

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_reject_table_name_with_union(self, mock_psycopg2):
        """Test that table names with UNION keyword are rejected."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        with pytest.raises(ValueError, match="reserved SQL keyword"):
            OpenGaussVectorIndex("postgres://test", table_name="UNION")

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_reject_table_name_with_dash(self, mock_psycopg2):
        """Test that table names with dashes are rejected (not valid identifiers)."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        with pytest.raises(ValueError, match="Invalid table name"):
            OpenGaussVectorIndex("postgres://test", table_name="vector-index")

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_reject_table_name_with_space(self, mock_psycopg2):
        """Test that table names with spaces are rejected."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        with pytest.raises(ValueError, match="Invalid table name"):
            OpenGaussVectorIndex("postgres://test", table_name="vector index")

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_reject_table_name_starting_with_digit(self, mock_psycopg2):
        """Test that table names starting with digit are rejected."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        with pytest.raises(ValueError, match="Invalid table name"):
            OpenGaussVectorIndex("postgres://test", table_name="123_vector")

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_accept_valid_table_names(self, mock_psycopg2):
        """Test that valid table names are accepted."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_psycopg2.connect.return_value = MockConnection()

        # These should all be valid
        valid_names = [
            "vector_index",
            "VectorIndex",
            "_vector_index",
            "vector123index",
            "vec_idx",
        ]

        for name in valid_names:
            index = OpenGaussVectorIndex("postgres://test", table_name=name)
            assert index._table_name == name
