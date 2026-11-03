"""Ownership, connection budgets and rejection need no live database."""

import threading
import time
from unittest.mock import Mock

import psycopg2
import pytest

from core.models import RequestContext
from fs.sql_adapter.session_mutation import (
    SessionMutationBusyError,
    SessionMutationLostError,
    SQLSessionGuard,
    guard_pool_methods,
    session_mutation_scope,
)

DSN = "dbname=test host=localhost"
CTX = RequestContext("account", "user", "agent", "session", "trace")


class Raw:
    closed = 0

    def __init__(self, *, delay=0):
        self.delay = delay
        self.calls = []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, args=None):
        self.calls.append((query, args))
        if "pg_try_advisory_lock" in query:
            time.sleep(self.delay)

    def fetchone(self):
        return (True,)

    def rollback(self):
        self.calls.append(("rollback", None))

    def commit(self):
        self.calls.append(("commit", None))

    def close(self):
        self.closed = 1


def connect_raw(monkeypatch, raw):
    monkeypatch.setattr("fs.sql_adapter.pool._resolved_hosts", lambda *args: {})
    monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: raw)


def test_acquisition_deadline_includes_slow_lock_query_and_late_connection_close(monkeypatch):
    raw = Raw(delay=.2)
    connect_raw(monkeypatch, raw)
    slots = threading.BoundedSemaphore(1)
    started = time.monotonic()
    with pytest.raises(SessionMutationBusyError), session_mutation_scope(DSN, CTX, 7, slots, .02):
        pytest.fail("ownership was delivered after its acquisition deadline")
    assert time.monotonic() - started < .15
    assert not slots.acquire(blocking=False)  # Abandoned acquisition remains bounded.
    time.sleep(.25)
    assert raw.closed and slots.acquire(blocking=False)
    slots.release()


def test_borrowed_transaction_rejects_nested_identity_lock_before_rollback():
    raw = Raw()
    guard = SQLSessionGuard(raw, DSN, CTX, 7)
    guard.borrow(DSN)
    with pytest.raises(SessionMutationLostError), guard.identity_lock(8, CTX, .1):
        pytest.fail("nested lock interrupted a borrowed transaction")
    assert raw.calls == []
    with pytest.raises(SessionMutationLostError):
        guard.connection.commit()


def test_pool_never_replaces_a_poisoned_owning_connection(monkeypatch):
    raw = Raw()
    connect_raw(monkeypatch, raw)
    store = Mock()
    store._sql_dialect = "postgresql"
    get, give_back = Mock(), Mock()
    store._get_connection, store._return_connection = get, give_back
    guard_pool_methods(store, DSN)
    with pytest.raises(SessionMutationLostError), session_mutation_scope(DSN, CTX, 7, threading.Semaphore(1)) as guard:
        conn = store._get_connection()
        conn.commit()
        store._return_connection(conn)
        guard.poison("deterministic connection loss")
        with pytest.raises(SessionMutationLostError):
            store._get_connection()
    get.assert_not_called()
    give_back.assert_not_called()
    assert raw.closed


def test_wrong_database_and_foreign_thread_poison_before_side_effects():
    guard = SQLSessionGuard(Raw(), DSN, CTX, 7)
    with pytest.raises(SessionMutationLostError):
        guard.borrow("dbname=other host=localhost")
    guard = SQLSessionGuard(Raw(), DSN, CTX, 7)
    errors = []
    def cross_thread():
        try:
            guard.connection.commit()
        except SessionMutationLostError as exc:
            errors.append(exc)
    worker = threading.Thread(target=cross_thread)
    worker.start()
    worker.join(1)
    assert len(errors) == 1 and guard.raw.calls == []
    with pytest.raises(SessionMutationLostError):
        guard.check_active()


@pytest.mark.parametrize("method", ["commit", "commit_snapshot"])
def test_first_manager_commit_checks_declared_backend_before_archiving(method):
    from session.session_manager import SessionManager

    class DeclaredSQL:
        def session_mutation_lock(self, *args, **kwargs):
            raise AssertionError("manager must require an existing guard")

    fs = DeclaredSQL()
    manager = SessionManager(get_context_fs=lambda: fs)
    manager.require_extraction_recovery(CTX.session_id)
    manager.add_message(CTX.session_id, "user", "An original source.", CTX, message_id="source")
    manager._process_snapshot = Mock(side_effect=AssertionError("unguarded archive was dispatched"))
    kwargs = {"session_id": CTX.session_id, "ctx": CTX, "wait": False}
    if method == "commit_snapshot":
        kwargs["snapshot"] = list(manager.get_or_create(CTX.session_id).messages)
    with pytest.raises(SessionMutationLostError):
        getattr(manager, method)(**kwargs)
    manager._process_snapshot.assert_not_called()
    assert len(manager.get_or_create(CTX.session_id).messages) == 1


def test_optional_generic_archive_failure_keeps_existing_in_memory_fallback():
    from session.session_manager import SessionManager

    manager = SessionManager(get_archive_store=Mock(side_effect=OSError("generic backend unavailable")))
    manager._compress = Mock(return_value=("Original source", "Original source"))
    manager.add_message(CTX.session_id, "user", "Original source.", CTX, message_id="source")
    result = manager.commit(CTX.session_id, CTX, wait=True)
    assert result["status"] == "completed" and result["archived"]


def test_mock_sql_archive_spec_does_not_declare_a_real_backend():
    from session.session_manager import SessionManager
    from session.sql_archive_store import SQLSessionArchiveStore

    manager = SessionManager(get_archive_store=Mock(return_value=Mock(spec=SQLSessionArchiveStore)))
    assert manager._require_sql_mutation(CTX) is False


@pytest.mark.parametrize("mock_sql_spec", [False, True])
def test_optional_non_sql_and_mock_spec_archive_rejection_retains_legacy_fallback(mock_sql_spec):
    from session.archive_store import SessionArchiveStore
    from session.models import ArchiveWriteResult
    from session.session_manager import SessionManager
    from session.sql_archive_store import SQLSessionArchiveStore
    from tests.unit.commit.test_memory_identity import MemoryFS

    store = Mock(spec=SQLSessionArchiveStore) if mock_sql_spec else SessionArchiveStore(MemoryFS())
    store.write_archive = Mock(return_value=ArchiveWriteResult(
        archive_id="archive", session_id=CTX.session_id, uri="", success=False,
        error="known non-SQL archive rejection"))
    manager = SessionManager(get_archive_store=lambda: store)
    result = manager._write_archive(CTX.session_id, "archive", "overview", "abstract", [], CTX)
    assert result == {"success": True, "uri": f"memory://{CTX.session_id}/archive", "archive_id": "archive"}
