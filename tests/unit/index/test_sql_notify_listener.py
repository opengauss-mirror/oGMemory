"""Unit tests for SQLNotifyListener."""

from unittest.mock import Mock, call

from index.sql_notify_listener import SQLNotifyListener


class FakeConn:
    def __init__(self):
        self.notifies = []
        self.poll_count = 0
        self.closed = False

    def poll(self):
        self.poll_count += 1

    def fileno(self):
        return 0

    def close(self):
        self.closed = True


def _stats(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0):
    return {
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "moved_to_dlq": moved_to_dlq,
        "skipped": skipped,
    }


def test_run_cycle_drains_ready_events_after_notification(monkeypatch):
    conn = FakeConn()
    conn.notifies = [object(), object()]
    store = Mock()
    store.future_pending_delay.return_value = None
    worker = Mock()
    worker.run_once.side_effect = [
        _stats(processed=1, succeeded=1),
        _stats(),
    ]
    listener = SQLNotifyListener(store, worker, worker_id="worker-1")

    monkeypatch.setattr(
        "index.sql_notify_listener.select.select",
        lambda readers, writers, errors, timeout: ([conn], [], []),
    )

    stats = listener.run_cycle(conn)

    assert conn.poll_count == 1
    assert conn.notifies == []
    assert stats == _stats(processed=1, succeeded=1)
    assert worker.run_once.call_args_list == [
        call(outbox_store=store, account_ids=[], worker_id="worker-1"),
        call(outbox_store=store, account_ids=[], worker_id="worker-1"),
    ]


def test_run_cycle_timeout_triggers_fallback_sweep(monkeypatch):
    conn = FakeConn()
    store = Mock()
    store.future_pending_delay.return_value = None
    worker = Mock()
    worker.run_once.side_effect = [_stats(),]
    listener = SQLNotifyListener(store, worker, worker_id="worker-2")

    monkeypatch.setattr(
        "index.sql_notify_listener.select.select",
        lambda readers, writers, errors, timeout: ([], [], []),
    )

    stats = listener.run_cycle(conn)

    assert conn.poll_count == 0
    assert stats == _stats()
    worker.run_once.assert_called_once_with(
        outbox_store=store,
        account_ids=[],
        worker_id="worker-2",
    )


def test_run_forever_opens_listener_connection_and_closes_it(monkeypatch):
    conn = FakeConn()
    store = Mock()
    store.open_listener_connection.return_value = conn
    worker = Mock()
    listener = SQLNotifyListener(store, worker, worker_id="worker-3")

    monkeypatch.setattr(listener, "_drain_ready_events", Mock(return_value=_stats()))

    def fake_run_cycle(current_conn):
        assert current_conn is conn
        listener.stop()
        return _stats()

    monkeypatch.setattr(listener, "run_cycle", fake_run_cycle)

    listener.run_forever()

    store.open_listener_connection.assert_called_once_with()
    assert conn.closed is True
