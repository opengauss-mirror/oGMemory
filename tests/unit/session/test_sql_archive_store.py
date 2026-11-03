"""Unit tests for SQLSessionArchiveStore.

Tests use mocked psycopg2 connections to verify behavior without
requiring a running PostgreSQL instance.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import RequestContext, Role

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(account_id="test_account", user_id="test_user", role=Role.MEMBER):
    return RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
        role=role,
    )


SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Doing well!"},
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create SQLSessionArchiveStore with mocked psycopg2 connection pool."""
    def mock_json(value):
        return value

    with patch("session.sql_archive_store.psycopg2") as mock_pg, \
         patch("session.sql_archive_store.Json", mock_json), \
         patch("session.sql_archive_store._HAS_PSYCOPG2", True):
        mock_conn = MagicMock()
        mock_conn.autocommit = False
        mock_conn.get_transaction_status.return_value = 0
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value = mock_conn

        from session.sql_archive_store import SQLSessionArchiveStore
        store = SQLSessionArchiveStore.__new__(SQLSessionArchiveStore)
        store._connection_string = "host=localhost dbname=test"
        store._pool_size = 5
        store._pool = []

        # Override pool to use mock
        store._get_connection = lambda: mock_conn
        store._return_connection = lambda c: None

        yield store, mock_conn, mock_cursor


# ===========================================================================
# _ensure_table
# ===========================================================================

class TestEnsureTable:

    def test_creates_table_with_composite_pk(self, mock_store):
        """_ensure_table issues CREATE TABLE with composite primary key."""
        store, conn, cursor = mock_store
        store._ensure_table()

        sqls = [call.args[0] for call in cursor.execute.call_args_list]
        assert any("PRIMARY KEY (account_id, session_id, archive_id)" in s for s in sqls)

    def test_raises_on_failure(self, mock_store):
        """_ensure_table raises RuntimeError when DDL fails."""
        import fs.sql_adapter.schema as schema_mod
        schema_mod._schema_ensured.clear()

        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("connection lost")

        with pytest.raises(RuntimeError, match="Failed to ensure schema: connection lost"):
            store._ensure_table()
        cursor.execute.assert_called_once()
        conn.rollback.assert_called()


# ===========================================================================
# write_archive
# ===========================================================================

class TestWriteArchive:

    def test_success(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="Test overview",
            abstract="Test abstract",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
            archive_id="arc001",
        )

        assert result.success is True
        assert result.archive_id == "arc001"
        assert result.session_id == "sess1"
        assert "ctx://test_account/sessions/sess1/history/arc001" in result.uri
        assert result.error is None
        conn.commit.assert_called_once()

    def test_auto_generates_archive_id(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="Overview",
            abstract="Abstract",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
        )

        assert result.success is True
        assert result.archive_id is not None
        assert len(result.archive_id) > 0
        # Format: YYYYMMDD_HHMMSS_{8hex}
        parts = result.archive_id.split("_")
        assert len(parts) == 3

    def test_passes_correct_params(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        store.write_archive(
            session_id="sess1",
            overview="Overview text",
            abstract="Abstract text",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
            archive_id="arc001",
        )

        # First call is SET LOCAL (bind_tenant for RLS), second is the INSERT.
        insert_call = cursor.execute.call_args_list[-1]
        sql = insert_call.args[0]
        params = insert_call.args[1]

        assert "INSERT INTO session_archives" in sql
        assert "ON CONFLICT (account_id, session_id, archive_id)" in sql
        # params order: archive_id, session_id, account_id, abstract, overview, messages, metadata, created_at
        assert params[0] == "arc001"
        assert params[1] == "sess1"
        assert params[2] == "test_account"
        assert params[3] == "Abstract text"
        assert params[4] == "Overview text"
        assert params[5] == SAMPLE_MESSAGES  # Json wrapper is identity in mock
        assert params[6]["message_count"] == 4

    def test_returns_failure_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("disk full")
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="O",
            abstract="A",
            messages=[],
            ctx=ctx,
            archive_id="arc001",
        )

        assert result.success is False
        assert "disk full" in result.error
        conn.rollback.assert_called()

    def test_empty_strings_for_none_abstract_overview(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        store.write_archive(
            session_id="sess1",
            overview=None,
            abstract=None,
            messages=[],
            ctx=ctx,
            archive_id="arc001",
        )

        params = cursor.execute.call_args.args[1]
        assert params[3] == ""  # abstract
        assert params[4] == ""  # overview

    def test_merges_extra_metadata(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="overview",
            abstract="abstract",
            messages=[],
            ctx=ctx,
            archive_id="arc-quality",
            metadata={"compression_quality": {"entity_retention_ratio": 1.0}},
        )

        assert result.success is True
        params = cursor.execute.call_args.args[1]
        metadata = params[6]
        assert metadata["archive_id"] == "arc-quality"
        assert metadata["compression_quality"] == {"entity_retention_ratio": 1.0}


# ===========================================================================
# list_archives
# ===========================================================================

class TestListArchives:

    def test_empty(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = []
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)

        assert entries == []
        # Verify query filters by account_id + session_id
        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "WHERE account_id = %s AND session_id = %s" in sql
        assert params == ("test_account", "sess1")

    def test_returns_entries_newest_first(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = [
            ("arc2", "sess1", "Abstract 2", "Overview 2",
             [], {"archive_id": "arc2"}, "2025-04-15T12:00:00"),
            ("arc1", "sess1", "Abstract 1", "Overview 1",
             [{"role": "user", "content": "hi"}], {"archive_id": "arc1"}, "2025-04-14T12:00:00"),
        ]
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)

        assert len(entries) == 2
        assert entries[0].archive_id == "arc2"
        assert entries[0].overview == "Overview 2"
        assert entries[0].messages == []  # list never includes messages
        assert entries[1].archive_id == "arc1"
        sql = cursor.execute.call_args.args[0]
        assert "ORDER BY created_at DESC" in sql

    def test_returns_empty_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)
        assert entries == []

    def test_isolates_by_account(self, mock_store):
        """list_archives filters by account_id — cross-account data invisible."""
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = []
        ctx_other = _ctx(account_id="other_account")

        store.list_archives("sess1", ctx_other)

        params = cursor.execute.call_args.args[1]
        assert params[0] == "other_account"

    def test_excludes_merged_archives(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = [
            ("arc2", "sess1", "Abstract 2", "Overview 2",
             [], {"archive_id": "arc2", "status": "MERGED"}, "2025-04-15T12:00:00"),
            ("arc1", "sess1", "Abstract 1", "Overview 1",
             [], {"archive_id": "arc1"}, "2025-04-14T12:00:00"),
        ]
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)

        assert [entry.archive_id for entry in entries] == ["arc1"]


class TestListArchivesSince:
    def test_excludes_merged_archives(self, mock_store):
        from datetime import datetime

        store, conn, cursor = mock_store
        cursor.fetchall.return_value = [
            ("arc2", "sess1", "Abstract 2", "Overview 2",
             [], {"archive_id": "arc2", "status": "MERGED"}, "2025-04-15T12:00:00"),
            ("arc1", "sess1", "Abstract 1", "Overview 1",
             [], {"archive_id": "arc1"}, "2025-04-14T12:00:00"),
        ]
        ctx = _ctx()

        entries = store.list_archives_since(datetime(2025, 4, 1), ctx)

        sql = cursor.execute.call_args.args[0]
        assert "metadata->>'archive_status'" in sql
        assert "metadata->>'status'" in sql
        assert sql.index("metadata->>'archive_status'") < sql.index("LIMIT")
        assert [entry.archive_id for entry in entries] == ["arc1"]


# ===========================================================================
# read_archive
# ===========================================================================

class TestReadArchive:

    def test_not_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        entry = store.read_archive("sess1", "missing", ctx)

        assert entry is None

    def test_found(self, mock_store):
        store, conn, cursor = mock_store
        test_messages = [{"role": "user", "content": "Hello"}]
        cursor.fetchone.return_value = (
            "arc001", "sess1", "Test abstract", "Test overview",
            json.dumps(test_messages), {"key": "val"}, "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)

        assert entry is not None
        assert entry.archive_id == "arc001"
        assert entry.session_id == "sess1"
        assert entry.abstract == "Test abstract"
        assert entry.overview == "Test overview"
        assert entry.messages == test_messages
        assert entry.metadata == {"key": "val"}

    def test_filters_by_session_id(self, mock_store):
        """read_archive includes session_id in WHERE — prevents cross-session read."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        store.read_archive("sess1", "arc001", ctx)

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "session_id = %s" in sql
        assert params == ("arc001", "sess1", "test_account")

    def test_returns_none_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry is None

    def test_handles_string_metadata(self, mock_store):
        """Metadata returned as string from psycopg2 is parsed correctly."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "arc001", "sess1", "abs", "ovr",
            '[]', '{"archive_id": "arc001"}', "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry.metadata == {"archive_id": "arc001"}
        assert entry.messages == []

    def test_handles_null_messages(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "arc001", "sess1", "abs", "ovr",
            None, {}, "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry.messages == []

    def test_cross_session_read_returns_none(self, mock_store):
        """Reading with wrong session_id returns None even if archive_id exists for another session."""
        store, conn, cursor = mock_store
        # Simulate: archive arc001 belongs to sess1, not sess2
        cursor.fetchone.return_value = None
        ctx = _ctx()

        result = store.read_archive("sess2", "arc001", ctx)
        assert result is None

        # Verify the query includes session_id filter
        params = cursor.execute.call_args.args[1]
        assert params[1] == "sess2"

    def test_returns_merged_archive_for_direct_lookup(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "arc001", "sess1", "abs", "ovr",
            "[]", {"archive_id": "arc001", "status": "MERGED"}, "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)

        assert entry is not None
        assert entry.archive_id == "arc001"
        assert entry.metadata["status"] == "MERGED"


# ===========================================================================
# read_archive_abstract
# ===========================================================================

class TestReadArchiveAbstract:

    def test_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = ("Test abstract",)
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)

        assert abstract == "Test abstract"

    def test_not_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "missing", ctx)
        assert abstract is None

    def test_empty_string_returns_none(self, mock_store):
        """Empty abstract string should return None, not empty string."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = ("",)
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)
        assert abstract is None

    def test_filters_by_session_id(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        store.read_archive_abstract("sess1", "arc001", ctx)

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "session_id = %s" in sql
        assert params == ("arc001", "sess1", "test_account")

    def test_returns_none_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)
        assert abstract is None

    def test_returns_merged_abstract_for_direct_lookup(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "Test abstract",
            {"archive_id": "arc001", "status": "MERGED"},
        )
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)

        assert abstract == "Test abstract"


# ===========================================================================
# Constructor guard
# ===========================================================================

class TestConstructor:

    def test_raises_without_psycopg2(self):
        with patch("session.sql_archive_store._HAS_PSYCOPG2", False):
            from session.sql_archive_store import SQLSessionArchiveStore
            with pytest.raises(ImportError, match="psycopg2 is required"):
                SQLSessionArchiveStore("host=localhost")


def test_delete_archive_removes_single_row(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test")
    cur.rowcount = 1

    ok = store.delete_archive("sess1", "arc001", ctx)

    assert ok is True
    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert "DELETE FROM session_archives" in sql
    assert "archive_id = %s" in sql
    assert params == ("arc001", "sess1", "acct-test")
    conn.commit.assert_called_once()


def test_delete_archive_returns_false_when_no_row_deleted(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test")
    cur.rowcount = 0

    ok = store.delete_archive("sess1", "missing", ctx)

    assert ok is False
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_admin_session_cleanup_includes_ownerless_legacy_rows(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test", role=Role.ADMIN)
    cur.rowcount = 2

    assert store.delete_session_archives("sess1", ctx) == 2

    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert "COALESCE(owner_user_id, '') = ''" in sql
    assert "COALESCE(owner_agent_id, '') = ''" in sql
    assert params == ("acct-test", "sess1", "test_user", "test_agent")
    conn.commit.assert_called_once()


def test_member_session_cleanup_excludes_ownerless_legacy_rows(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test", role=Role.MEMBER)
    cur.rowcount = 1

    assert store.delete_session_archives("sess1", ctx) == 1

    sql = cur.execute.call_args.args[0]
    assert "COALESCE(owner_user_id" not in sql
    assert cur.execute.call_args.args[1] == (
        "acct-test", "sess1", "test_user", "test_agent",
    )


def test_session_owner_lookup_allows_legacy_rows_but_rejects_conflicts(mock_store):
    store, _conn, cur = mock_store
    ctx = _ctx(account_id="acct-test", role=Role.ADMIN)
    cur.fetchall.return_value = [("", ""), ("test_user", "test_agent")]

    assert store.get_session_owner("sess1", ctx) == ("test_user", "test_agent")

    cur.fetchall.return_value = [("user-a", "agent-a"), ("user-b", "agent-b")]
    with pytest.raises(PermissionError, match="conflicting durable session owners"):
        store.get_session_owner("sess1", ctx)


@pytest.mark.parametrize("failure_point", ["bind_tenant", "execute", "fetchall"])
def test_session_owner_lookup_rolls_back_before_return_on_db_error(
    mock_store, failure_point
):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test", role=Role.ADMIN)
    error = RuntimeError(f"{failure_point} failed")
    events = []

    conn.rollback.side_effect = lambda: events.append("rollback")
    store._return_connection = lambda returned: events.append("return")
    if failure_point == "bind_tenant":
        store.bind_tenant = MagicMock(side_effect=error)
    elif failure_point == "execute":
        cur.execute.side_effect = [None, error]
    else:
        cur.fetchall.side_effect = error

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        store.get_session_owner("sess1", ctx)

    assert events == ["rollback", "return"]


def test_mark_archive_merged_updates_metadata_atomically(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test")
    cur.rowcount = 1

    ok = store.mark_archive_merged("sess1", "arc001", ctx, merged_into="merged_001")

    assert ok is True
    executed = [call.args[0] for call in cur.execute.call_args_list]
    assert not any("SELECT metadata FROM session_archives" in sql for sql in executed)
    assert any("COALESCE(metadata" in sql and "|| %s::jsonb" in sql for sql in executed)
    update_params = cur.execute.call_args.args[1]
    assert update_params[0]["archive_status"] == "MERGED"
    assert update_params[0]["merged_into"] == "merged_001"
    assert update_params[1:] == ("arc001", "sess1", "acct-test")
    conn.commit.assert_called_once()


def test_mark_archive_merged_returns_false_when_no_sql_row_updated(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test")
    cur.rowcount = 0

    ok = store.mark_archive_merged("sess1", "missing", ctx, merged_into="merged_001")

    assert ok is False
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_unmark_archive_merged_updates_metadata(mock_store):
    store, conn, cur = mock_store
    ctx = _ctx(account_id="acct-test")
    cur.fetchone.return_value = (
        {"archive_id": "arc001", "status": "MERGED", "merged_into": "merged_001"},
    )

    ok = store.unmark_archive_merged("sess1", "arc001", ctx, merged_into="merged_001")

    assert ok is True
    executed = [call.args[0] for call in cur.execute.call_args_list]
    assert any("SELECT metadata FROM session_archives" in sql for sql in executed)
    assert any("UPDATE session_archives" in sql for sql in executed)
    update_params = cur.execute.call_args_list[-1].args[1]
    assert "status" not in update_params[0]
    assert "merged_into" not in update_params[0]
    assert "merged_at" not in update_params[0]
    assert update_params[1:] == ("arc001", "sess1", "acct-test")
    conn.commit.assert_called_once()
