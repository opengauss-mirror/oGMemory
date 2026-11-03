"""One thread-bound PostgreSQL session owns and executes a memory mutation."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar

from fs.sql_adapter.pool import psycopg2

_ACTIVE = ContextVar("sql_session_mutation", default=None)


class SessionMutationBusyError(TimeoutError):
    """No mutation began because a bounded ownership acquisition timed out."""


class SessionMutationLostError(RuntimeError):
    """The original owning SQL connection cannot be replaced or used again."""


def active_session_guard():
    return _ACTIVE.get()


def _dsn_identity(dsn):
    return tuple(sorted(psycopg2.extensions.parse_dsn(dsn).items()))


class _Cursor:
    def __init__(self, guard, cursor):
        self.guard, self.cursor = guard, cursor

    def __getattr__(self, name):
        return getattr(self.cursor, name)

    def __enter__(self):
        self.cursor.__enter__()
        return self

    def __exit__(self, *args):
        return self.cursor.__exit__(*args)

    def execute(self, *args, **kwargs):
        return self.guard.call(self.cursor.execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self.guard.call(self.cursor.executemany, *args, **kwargs)


class _Connection:
    def __init__(self, guard):
        self.guard = guard

    def __getattr__(self, name):
        return getattr(self.guard.raw, name)

    def cursor(self, *args, **kwargs):
        return _Cursor(self.guard, self.guard.call(self.guard.raw.cursor, *args, **kwargs))

    def commit(self):
        return self.guard.call(self.guard.raw.commit)

    def rollback(self):
        return self.guard.call(self.guard.raw.rollback)

    def close(self):
        self.guard.poison("owning connection was closed")
        self.guard.raw.close()


class SQLSessionGuard:
    def __init__(self, raw, dsn, ctx, key, dialect="postgresql"):
        self.raw, self.dsn, self.ctx, self.key = raw, _dsn_identity(dsn), ctx, key
        self.thread_id = threading.get_ident()
        self.dialect = dialect
        self.poison_reason = None
        self.borrowed = 0
        self.depths = {key: 1}
        self.post_commit_drain = False
        self.connection = _Connection(self)

    def poison(self, reason):
        self.poison_reason = self.poison_reason or reason

    def check_active(self):
        if threading.get_ident() != self.thread_id:
            self.poison("owning connection cannot cross threads")
        if self.raw.closed:
            self.poison("owning connection was lost")
        if self.poison_reason:
            raise SessionMutationLostError(self.poison_reason)

    def check_context(self, ctx):
        if (ctx.account_id, ctx.user_id, ctx.agent_id, ctx.session_id) != (
                self.ctx.account_id, self.ctx.user_id, self.ctx.agent_id, self.ctx.session_id):
            self.poison("mutation context does not match its owning session")
        self.check_active()

    def call(self, function, *args, **kwargs):
        self.check_active()
        try:
            return function(*args, **kwargs)
        except (psycopg2.InterfaceError, psycopg2.OperationalError) as exc:
            self.poison("owning SQL connection failed")
            raise SessionMutationLostError(self.poison_reason) from exc

    def ensure_owned(self):
        """Check before non-SQL side effects; SQL writes themselves use this session."""
        self.check_active()
        if self.borrowed:
            self.poison("ownership check cannot interrupt a borrowed transaction")
            self.check_active()
        unsigned = self.key & ((1 << 64) - 1)
        with self.raw.cursor() as cursor:
            self.call(cursor.execute, "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype='advisory' "
                      "AND pid=pg_backend_pid() AND classid=%s AND objid=%s AND objsubid=1)",
                      (unsigned >> 32, unsigned & ((1 << 32) - 1)))
            if not cursor.fetchone()[0]:
                self.poison("owning advisory lock was lost")
        self.call(self.raw.rollback)
        self.check_active()

    def borrow(self, dsn, dialect="postgresql"):
        self.check_database(dsn, dialect)
        if self.borrowed:
            self.poison("nested store transactions cannot share the mutation connection")
            self.check_active()
        self.borrowed += 1
        return self.connection

    def check_database(self, dsn, dialect="postgresql"):
        self.check_active()
        if _dsn_identity(dsn) != self.dsn or dialect != self.dialect:
            self.poison("guarded mutation cannot use another SQL database")
            self.check_active()

    def give_back(self, conn):
        if conn is not self.connection or self.borrowed != 1:
            self.poison("guarded connection return does not match its borrow")
            self.check_active()
        self.borrowed -= 1

    @contextmanager
    def identity_lock(self, key, ctx, timeout_seconds):
        self.check_context(ctx)
        if self.borrowed:
            self.poison("identity acquisition cannot interrupt a borrowed transaction")
            self.check_active()
        depth = self.depths.get(key, 0)
        if depth == 0:
            _try_lock(self, key, timeout_seconds)
        self.depths[key] = depth + 1
        try:
            yield self
        finally:
            self.depths[key] -= 1
            if self.depths[key] == 0:
                self.depths.pop(key)
                if not self.poison_reason and not self.raw.closed:
                    with self.raw.cursor() as cursor:
                        self.call(cursor.execute, "SELECT pg_advisory_unlock(%s)", (key,))
                    self.call(self.raw.rollback)


def _try_lock(guard, key, timeout_seconds):
    deadline = time.monotonic() + max(0, timeout_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SessionMutationBusyError("session mutation ownership is busy")
        with guard.raw.cursor() as cursor:
            guard.call(cursor.execute, "SET LOCAL statement_timeout = %s", (max(1, int(remaining * 1000)),))
            guard.call(cursor.execute, "SELECT pg_try_advisory_lock(%s)", (key,))
            acquired = cursor.fetchone()[0]
        guard.call(guard.raw.rollback)
        if acquired:
            if time.monotonic() > deadline:
                guard.poison("advisory acquisition exceeded its deadline")
                guard.check_active()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SessionMutationBusyError("session mutation ownership is busy")
        time.sleep(min(.025, remaining))


@contextmanager
def session_mutation_scope(dsn, ctx, key, slots, timeout_seconds=1.0, *, dialect="postgresql"):
    current = active_session_guard()
    if current is not None:
        current.check_context(ctx)
        if _dsn_identity(dsn) != current.dsn or key != current.key or dialect != current.dialect:
            current.poison("nested mutation must retain its database and session")
            current.check_active()
        yield current
        return
    deadline = time.monotonic() + max(0, timeout_seconds)
    if not slots.acquire(timeout=max(0, deadline - time.monotonic())):
        raise SessionMutationBusyError("session mutation connection budget is busy")
    raw = None
    connect_future = Future()
    connect_abandoned = False

    def connect():
        connection = None
        try:
            from fs.sql_adapter.pool import _resolved_hosts
            addresses = _resolved_hosts(dsn, deadline)
            connection = psycopg2.connect(dsn, connect_timeout=max(1, math.ceil(deadline - time.monotonic())),
                                          **addresses)
            _try_lock(SQLSessionGuard(connection, dsn, ctx, key, dialect), key, max(0, deadline - time.monotonic()))
            connect_future.set_result(connection)
        except BaseException as exc:
            if connection is not None:
                connection.close()
            connect_future.set_exception(exc)

    def close_abandoned(future):
        try:
            future.result().close()
        except BaseException:
            pass
        finally:
            slots.release()

    try:
        threading.Thread(target=connect, daemon=True, name="sql-session-mutation-connect").start()
        try:
            raw = connect_future.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError as exc:
            connect_abandoned = True
            connect_future.add_done_callback(close_abandoned)
            raise SessionMutationBusyError("session mutation connection deadline elapsed") from exc
        except psycopg2.Error as exc:
            raise SessionMutationBusyError("session mutation SQL connection was not established") from exc
        guard = SQLSessionGuard(raw, dsn, ctx, key, dialect)
        token = _ACTIVE.set(guard)
        try:
            yield guard
            guard.ensure_owned()
        finally:
            _ACTIVE.reset(token)
    finally:
        if raw is not None:
            raw.close()  # Closing releases every session advisory lock, even on failure.
        if not connect_abandoned:
            slots.release()


def guard_pool_methods(instance, dsn):
    """Wrap both shared-pool aliases and subclass standalone pool methods."""
    original_get, original_return = instance._get_connection, instance._return_connection

    def get(*args, **kwargs):
        guard = active_session_guard()
        return guard.borrow(dsn, instance._sql_dialect) if guard is not None else original_get(*args, **kwargs)

    def give_back(conn):
        guard = active_session_guard()
        if guard is not None:
            guard.give_back(conn)
        else:
            original_return(conn)

    instance._get_connection, instance._return_connection = get, give_back
