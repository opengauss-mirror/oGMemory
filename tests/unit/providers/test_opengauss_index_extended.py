"""Extended tests for OpenGaussVectorIndex delete operations.

Uses mock database connection for isolated testing.
"""

from unittest.mock import patch

import pytest


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


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, cursor=None):
        self.cursor_obj = cursor or MockCursor()
        self.committed = False
        self.closed = 0

    def cursor(self, cursor_factory=None):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = 1


class MockCursor:
    """Mock database cursor for testing."""

    def __init__(self):
        self.executed_sql: list[str] = []
        self.executed_params: list[tuple] = []
        self.results: list[tuple | dict] = []
        self.rowcount = 0

    def execute(self, sql: str, params: dict = None):
        self.executed_sql.append(sql)
        if params:
            self.executed_params.append(params)

    def fetchone(self):
        if self.results:
            return self.results.pop(0)
        return None

    def fetchall(self):
        result = self.results
        self.results = []
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class TestOpenGaussDeleteOperations:
    """Tests for delete_account_data and delete_by_owner_space methods."""

    def setup_method(self):
        """Ensure the module flag is set before each test."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_account_data_executes_correct_sql(self, mock_psycopg2):
        """Test delete_account_data executes correct SQL."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 5
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        deleted = index.delete_account_data("acme")

        assert deleted == 5
        # Filter out CREATE TABLE and CREATE INDEX statements from auto-init
        delete_sqls = [sql for sql in mock_cursor.executed_sql if "DELETE FROM" in sql]
        assert len(delete_sqls) == 1
        assert "DELETE FROM vector_index" in delete_sqls[0]
        assert "filters->>'account_id' = %(account_id)s" in delete_sqls[0]
        assert mock_cursor.executed_params[0]["account_id"] == "acme"
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_account_data_with_custom_table(self, mock_psycopg2):
        """Test delete_account_data uses custom table name."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 3
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test", table_name="custom_vectors")
        deleted = index.delete_account_data("test_account")

        assert deleted == 3
        # Filter out CREATE TABLE statements from auto-init
        delete_sqls = [sql for sql in mock_cursor.executed_sql if "DELETE FROM" in sql]
        assert "DELETE FROM custom_vectors" in delete_sqls[0]

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_by_owner_space_executes_correct_sql(self, mock_psycopg2):
        """Test delete_by_owner_space executes correct SQL."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 2
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        deleted = index.delete_by_owner_space("acme", "user:alice")

        assert deleted == 2
        # Filter out CREATE TABLE statements from auto-init
        delete_sqls = [sql for sql in mock_cursor.executed_sql if "DELETE FROM" in sql]
        assert len(delete_sqls) == 1
        assert "DELETE FROM vector_index" in delete_sqls[0]
        assert "filters->>'account_id' = %(account_id)s" in delete_sqls[0]
        assert "filters->>'owner_space' = %(owner_space)s" in delete_sqls[0]
        assert mock_cursor.executed_params[0]["account_id"] == "acme"
        assert mock_cursor.executed_params[0]["owner_space"] == "user:alice"
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_by_owner_space_scoped_to_account(self, mock_psycopg2):
        """Test delete_by_owner_space requires both account_id and owner_space."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 1
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        index.delete_by_owner_space("acme", "user:alice")

        # Filter out CREATE TABLE statements from auto-init
        delete_sqls = [sql for sql in mock_cursor.executed_sql if "DELETE FROM" in sql]
        sql = delete_sqls[0]
        # Verify both conditions are in WHERE clause
        assert "WHERE" in sql
        assert "AND" in sql
        assert sql.index("filters->>'account_id'") < sql.index("AND")
        assert sql.index("AND") < sql.index("filters->>'owner_space'")


class TestOpenGaussSearchByVector:
    """Tests for search_by_vector method."""

    def setup_method(self):
        """Ensure the module flag is set before each test."""
        import providers.vector_index.opengauss_index as ogi_module
        ogi_module.OPENGAUSS_AVAILABLE = True

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_by_vector_builds_correct_sql(self, mock_psycopg2):
        """Test search_by_vector builds correct SQL query."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [
            {
                "id": "1",
                "uri": "ctx://acme/users/u1/memories/profile",
                "level": 0,
                "text": "profile abstract",
                "filters": {"account_id": "acme", "owner_space": "user:u1"},
                "metadata": {"category": "profile", "has_overview": True, "has_content": True},
                "score": 0.95,
            }
        ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        results = index.search_by_vector(
            query_vector=[0.1] * 1536,
            filters={"account_id": "acme"},
            top_k=10
        )

        assert len(results) == 1
        # Filter out CREATE TABLE statements from auto-init
        select_sqls = [sql for sql in mock_cursor.executed_sql if "SELECT" in sql]
        sql = select_sqls[0]
        # Remove extra whitespace for robust checking
        sql_normalized = " ".join(sql.split())
        assert "SELECT id, uri, level, text, filters, metadata" in sql_normalized
        assert "1 - (embedding <=> %(qvec)s::vector) AS score" in sql_normalized
        assert "ORDER BY embedding <=> %(qvec)s::vector" in sql_normalized
        assert "LIMIT %(topk)s" in sql_normalized

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_children_adds_parent_uri_filter(self, mock_psycopg2):
        """Test search_children adds parent_uri filter to WHERE clause."""
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        mock_cursor = MockCursor()
        mock_cursor.results = []
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussVectorIndex("postgres://test")
        index.search_children(
            parent_uri="ctx://acme/users/u1/memories/preferences/",
            query_vector=[0.1] * 1536,
            filters={"account_id": "acme"},
            top_k=10
        )

        # Filter out CREATE TABLE statements from auto-init
        select_sqls = [sql for sql in mock_cursor.executed_sql if "SELECT" in sql]
        sql = select_sqls[0]
        assert "metadata->>'parent_uri' = %(parent_uri)s" in sql
        assert mock_cursor.executed_params[0]["parent_uri"] == "ctx://acme/users/u1/memories/preferences/"
