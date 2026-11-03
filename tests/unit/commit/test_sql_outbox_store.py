"""Unit tests for SQLOutboxStore."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from core.models import OutboxEvent


@pytest.fixture
def mock_sql_outbox_store():
    """Create SQLOutboxStore with mocked psycopg2."""
    def mock_json(value):
        return value

    with patch("commit.sql_outbox_store.psycopg2") as mock_pg, \
         patch("commit.sql_outbox_store.Json", mock_json), \
         patch("commit.sql_outbox_store._HAS_PSYCOPG2", True):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value = mock_conn

        from commit.sql_outbox_store import SQLOutboxStore

        store = SQLOutboxStore.__new__(SQLOutboxStore)
        store._connection_string = "host=localhost dbname=test"
        store._fs = MagicMock()
        store._pool_size = 5
        store._pool = []
        store._get_connection = lambda: mock_conn
        store._return_connection = lambda c: None

        yield store, mock_pg, mock_conn, mock_cursor


def test_insert_event_sends_notify(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    event = OutboxEvent(
        event_id="evt-1",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acme/users/alice/memories/profile",
        payload={"records": []},
        status="PENDING",
        retry_count=0,
        created_at="2026-04-17T10:00:00+00:00",
    )

    store._insert_event(event, "acme")

    assert cursor.execute.call_count == 2
    notify_sql, notify_params = cursor.execute.call_args_list[1].args
    assert "SELECT pg_notify(%s, %s)" in notify_sql
    assert notify_params[0] == store.notify_channel
    payload = json.loads(notify_params[1])
    assert payload == {
        "event_id": "evt-1",
        "account_id": "acme",
        "event_type": "UPSERT_CONTEXT",
    }
    conn.commit.assert_called_once()


def test_agent_memory_checkpoint_and_completion_receipt(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    event = OutboxEvent("job-1", "EXTRACT_AGENT_MEMORY", "ctx://acme/agents/a/memories/agent_extraction/job-1",
                        {"cursor": 32, "value_reviews": [{"memory_type": "tool", "decisions": []}]}, "PROCESSING")
    store.save_payload(event)
    sql, values = cursor.execute.call_args.args
    assert "SET payload" in sql
    assert values == (event.payload, event.event_id)
    store.mark_done(event, event.uri)
    sql, values = cursor.execute.call_args.args
    assert "status = 'DONE'" in sql
    assert values == ({"value_reviews": event.payload["value_reviews"]}, event.event_id)
    assert conn.commit.call_count == 2


def test_claim_batch_returns_claimed_events(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    created_at = datetime(2026, 4, 17, 10, 0, tzinfo=UTC)
    cursor.fetchall.return_value = [
        (
            "evt-2",
            "UPSERT_CONTEXT",
            "ctx://acme/users/alice/memories/profile",
            {"records": []},
            "PROCESSING",
            1,
            created_at,
            None,
        )
    ]

    claimed = store.claim_batch(worker_id="worker-1", limit=50, timeout_seconds=42)

    assert len(claimed) == 1
    node_uri, event = claimed[0]
    assert node_uri == "ctx://acme/users/alice/memories/profile"
    assert event.event_id == "evt-2"
    assert event.status == "PROCESSING"
    sql, params = cursor.execute.call_args.args
    assert "earlier.account_id = candidate.account_id" in sql
    assert "earlier.uri = candidate.uri" in sql
    assert "earlier.seq < candidate.seq" in sql
    assert "earlier.status IN ('PENDING', 'PROCESSING')" in sql
    assert "FOR UPDATE OF candidate SKIP LOCKED" in sql
    assert "LIMIT %s" in sql
    assert params == (42, [], 50, "worker-1")
    conn.commit.assert_called_once()


def test_list_recent_returns_live_and_failed_tasks(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.fetchall.return_value = [
        (
            "evt-3",
            "UPSERT_DIRECTORY",
            "ctx://acme/users/alice/memories/events/",
            {"filters": {"owner_space": "user:alice"}},
            "FAILED",
            3,
            datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
            None,
        )
    ]

    tasks = store.list_recent("acme", limit=25)

    assert tasks[0][1].event_id == "evt-3"
    assert tasks[0][1].status == "FAILED"
    sql, params = cursor.execute.call_args.args
    assert "status IN ('PENDING', 'PROCESSING', 'FAILED')" in sql
    assert "ORDER BY created_at DESC" in sql
    assert params == ("acme", 25)
    conn.rollback.assert_called_once()


def test_list_recent_rolls_back_failed_query(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.execute.side_effect = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        store.list_recent("acme")

    conn.rollback.assert_called_once()


def test_get_event_scopes_lookup_to_account(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (
        "evt-failed",
        "UPSERT_CONTEXT",
        "ctx://acme/users/alice/memories/profile",
        {"failure_reason": "embedding unavailable"},
        "FAILED",
        5,
        datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
        None,
    )

    event = store.get_event("acme", "evt-failed")

    assert event is not None
    assert event.event_id == "evt-failed"
    assert cursor.execute.call_args.args[1] == ("acme", "evt-failed")
    conn.rollback.assert_called_once()


def test_retry_failed_requeues_and_notifies_worker(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    event = OutboxEvent(
        event_id="evt-failed",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acme/users/alice/memories/profile",
        payload={"records": [], "failure_reason": "embedding unavailable"},
        status="FAILED",
        retry_count=5,
    )
    cursor.fetchone.return_value = (
        event.event_id,
        event.event_type,
        event.uri,
        {"records": []},
        "PENDING",
        0,
        datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
        None,
    )

    retried = store.retry_failed("acme", event)

    assert retried is not None
    assert retried.status == "PENDING"
    assert retried.retry_count == 0
    update_sql, update_params = cursor.execute.call_args_list[0].args
    assert "status = 'PENDING'" in update_sql
    assert "status = 'FAILED'" in update_sql
    assert update_params == ({"records": []}, "acme", "evt-failed")
    assert cursor.execute.call_count == 2
    conn.commit.assert_called_once()


def test_move_to_dlq_persists_failure_reason(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    event = OutboxEvent(
        event_id="evt-failed",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acme/users/alice/memories/profile",
        payload={"failure_reason": "embedding unavailable"},
        status="PENDING",
    )

    store.move_to_dlq(event, event.uri)

    sql, params = cursor.execute.call_args.args
    assert "payload = %s" in sql
    assert params == (event.payload, "evt-failed")
    assert event.status == "FAILED"
    conn.commit.assert_called_once()


def test_increment_retry_persists_failure_reason(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    event = OutboxEvent(
        event_id="evt-retry",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acme/users/alice/memories/profile",
        payload={"failure_reason": "temporary embedding failure"},
        status="PENDING",
    )

    store.increment_retry(event, event.uri)

    sql, params = cursor.execute.call_args.args
    assert "payload = %s" in sql
    assert params == (event.payload, None, "evt-retry")
    assert event.retry_count == 1
    conn.commit.assert_called_once()


def test_count_unfinished_counts_pending_and_processing_by_account(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (2, 3, 4)

    counts = store.count_unfinished(account_id="acme")

    assert counts == {"pending": 2, "processing": 3, "failed": 4, "total": 5}
    sql, params = cursor.execute.call_args.args
    assert "status IN ('PENDING', 'PROCESSING', 'FAILED')" in sql
    assert "account_id = %s" in sql
    assert params == ("acme",)
    conn.commit.assert_called_once()


def test_open_listener_connection_sets_autocommit_and_listens(
    mock_sql_outbox_store,
):
    store, mock_pg, _, _ = mock_sql_outbox_store
    listener_conn = MagicMock()
    listener_cursor = MagicMock()
    listener_conn.cursor.return_value.__enter__ = lambda s: listener_cursor
    listener_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.connect.return_value = listener_conn

    conn = store.open_listener_connection()

    assert conn is listener_conn
    listener_conn.set_session.assert_called_once_with(autocommit=True)
    listener_cursor.execute.assert_called_once_with(f"LISTEN {store.notify_channel}")


def test_processing_lock_uses_dedicated_session_advisory_lock(
    mock_sql_outbox_store,
):
    store, mock_pg, pooled_conn, _ = mock_sql_outbox_store
    lock_conn = MagicMock()
    lock_cursor = MagicMock()
    lock_conn.closed = 0
    lock_conn.cursor.return_value.__enter__ = lambda s: lock_cursor
    lock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    lock_cursor.fetchone.return_value = (True,)
    mock_pg.connect.return_value = lock_conn

    with store.processing_lock("ctx://acme/users/alice/memories/profile"):
        assert lock_cursor.execute.call_count == 1

    assert lock_conn is not pooled_conn
    mock_pg.connect.assert_called_once_with(
        store._connection_string,
        connect_timeout=5,
    )
    lock_conn.set_session.assert_called_once_with(autocommit=True)
    assert lock_cursor.execute.call_count == 2
    lock_sql, lock_params = lock_cursor.execute.call_args_list[0].args
    unlock_sql, unlock_params = lock_cursor.execute.call_args_list[1].args
    assert lock_sql == "SELECT pg_try_advisory_lock(%s)"
    assert unlock_sql == "SELECT pg_advisory_unlock(%s)"
    assert lock_params == unlock_params
    lock_conn.close.assert_called_once()


def test_processing_lock_times_out_without_unlocking_unowned_lock(
    mock_sql_outbox_store,
):
    store, mock_pg, _, _ = mock_sql_outbox_store
    lock_conn = MagicMock()
    lock_cursor = MagicMock()
    lock_conn.closed = 0
    lock_conn.cursor.return_value.__enter__ = lambda s: lock_cursor
    lock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    lock_cursor.fetchone.return_value = (False,)
    mock_pg.connect.return_value = lock_conn

    with patch(
        "commit.sql_outbox_store.time.monotonic",
        side_effect=[0.0, 6.0],
    ):
        with pytest.raises(TimeoutError, match="URI processing lock"):
            with store.processing_lock("ctx://acme/users/alice/memories/profile"):
                raise AssertionError("lock body must not run")

    assert lock_cursor.execute.call_count == 1
    assert lock_cursor.execute.call_args.args[0] == (
        "SELECT pg_try_advisory_lock(%s)"
    )
    lock_conn.close.assert_called_once()


def test_is_event_current_commits_read_transaction(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (1,)

    assert store.is_event_current("evt-1") is True

    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()


def test_build_delete_event_contains_complete_record_ids(mock_sql_outbox_store):
    store, _, _, _ = mock_sql_outbox_store
    uri = "ctx://acme/resources/chunk-a"

    event = store.build_delete_event(uri)

    assert event.event_type == "DELETE_CONTEXT"
    assert len(event.payload["ids_to_delete"]) == 3


def test_is_event_current_can_match_event_and_node_uri(mock_sql_outbox_store):
    store, _, conn, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (1,)

    assert store.is_event_current("evt-1", "ctx://acme/resources/chunk-a") is True

    sql, params = cursor.execute.call_args.args
    assert "event_id = %s AND uri = %s" in sql
    assert params == ("evt-1", "ctx://acme/resources/chunk-a")
    conn.commit.assert_called_once()


def test_only_dead_letters_are_reported_even_with_zero_active_total(mock_sql_outbox_store):
    store, _, _, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (0, 0, 1)
    assert store.count_unfinished('acme') == {
        'pending': 0, 'processing': 0, 'total': 0, 'failed': 1,
    }
