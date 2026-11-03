"""Thread-safe shared connection pool for all SQL stores.

All SQL-backed stores (SQLContextFS, SQLOutboxStore, SQLRelationStore,
SQLSessionArchiveStore) share one pool instance so that:

1. Connection count stays bounded (one pool, not N stores × pool_size).
2. Cross-store operations (e.g. write_node + outbox INSERT) can use the
   same connection for a single atomic transaction.

The pool enforces a hard cap on total connections (idle + active) via a
semaphore.  When ``pool_size`` connections are already checked out, the
next ``get_connection()`` call blocks until one is returned.
"""

import ipaddress
import logging
import os
import selectors
import socket
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager

from fs.sql_adapter.dialect import SQLDialect, normalize_sql_dialect, resolve_sql_dialect

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None

logger = logging.getLogger(__name__)
_dns_slots = threading.BoundedSemaphore(4)


def remaining_seconds(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('PostgreSQL operation deadline elapsed')
    return remaining


def _resolved_hosts(dsn, deadline):
    """Resolve TCP names before libpq, with bounded DNS workers and caller wait."""
    options = psycopg2.extensions.parse_dsn(dsn)
    hosts = options.get('host', os.environ.get('PGHOST', '')).split(',')
    addresses = options.get('hostaddr', os.environ.get('PGHOSTADDR', '')).split(',')
    ports = options.get('port', os.environ.get('PGPORT', '5432')).split(',')

    def needs_dns(index, host):
        if not host or host.startswith(('/', '@')) or (index < len(addresses) and addresses[index]):
            return False
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            return True

    if not any(needs_dns(index, host) for index, host in enumerate(hosts)):
        return {}
    if not _dns_slots.acquire(timeout=remaining_seconds(deadline)):
        raise TimeoutError('PostgreSQL DNS slot deadline elapsed')
    result = Future()

    def resolve():
        try:
            resolved = []
            resolution_error = None
            for index, host in enumerate(hosts):
                port = ports[index] if len(ports) > 1 else ports[0]
                address = addresses[index] if index < len(addresses) else ''
                if address or not host or host.startswith(('/', '@')):
                    resolved.append((host, address, port))
                    continue
                try:
                    infos = socket.getaddrinfo(host, port or 5432, type=socket.SOCK_STREAM)
                except socket.gaierror as exc:
                    resolution_error = exc
                    continue
                for address in dict.fromkeys(info[4][0] for info in infos):
                    resolved.append((host, address, port))
            if not resolved:
                raise resolution_error or psycopg2.OperationalError('No PostgreSQL host addresses')
            result.set_result({
                'host': ','.join(host for host, _, _ in resolved),
                'hostaddr': ','.join(address for _, address, _ in resolved),
                'port': ','.join(port for _, _, port in resolved),
            })
        except Exception as exc:
            result.set_exception(psycopg2.OperationalError(str(exc)))
        finally:
            _dns_slots.release()

    try:
        threading.Thread(target=resolve, daemon=True, name='postgres-dns').start()
    except Exception:
        _dns_slots.release()
        raise
    return result.result(timeout=remaining_seconds(deadline))


def wait_for_postgres(conn, deadline):
    from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE

    while True:
        remaining = remaining_seconds(deadline)
        state = conn.poll()
        if state == POLL_OK:
            return
        if state not in (POLL_READ, POLL_WRITE):
            raise psycopg2.OperationalError(f'Unexpected PostgreSQL poll state: {state}')
        with selectors.DefaultSelector() as selector:
            selector.register(conn, selectors.EVENT_READ if state == POLL_READ else selectors.EVENT_WRITE)
            if not selector.select(remaining):
                raise TimeoutError('PostgreSQL operation deadline elapsed')


@contextmanager
def _deadline_connection(connection_string, deadline):
    addresses = _resolved_hosts(connection_string, deadline)
    remaining_seconds(deadline)
    conn = psycopg2.connect(connection_string, async_=True, **addresses)
    try:
        wait_for_postgres(conn, deadline)
        yield conn
    finally:
        conn.close()


def resolve_connection_dialect(value, connection_string, connect_timeout_seconds=10, *, deadline=None):
    """Resolve an automatic dialect within one client-side connection/query budget."""
    if normalize_sql_dialect(value) != "auto":
        return resolve_sql_dialect(value)
    local_deadline = time.monotonic() + connect_timeout_seconds
    deadline = local_deadline if deadline is None else min(deadline, local_deadline)
    with _deadline_connection(connection_string, deadline) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT version()")
            wait_for_postgres(conn, deadline)
            version = str(cursor.fetchone()[0]).casefold()
        if "opengauss" in version:
            return resolve_sql_dialect("opengauss")
        if "postgresql" in version:
            return resolve_sql_dialect("postgresql")
        raise ValueError("Unrecognized SQL server; configure its dialect explicitly")


class SharedConnectionPool:
    """Process-level, thread-safe psycopg2 connection pool.

    ``pool_size`` caps the total number of live connections (idle + active).
    ``get_connection()`` blocks when the cap is reached until a connection
    is returned.

    Idle connections are validated by default before reuse. Callers whose
    operations are independently bounded and retried may skip the preflight.

    Usage::

        pool = SharedConnectionPool(dsn="...", pool_size=10)
        conn = pool.get_connection()
        try:
            ...
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            pool.return_connection(conn)
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        connect_timeout_seconds: int = 10,
        dialect: str | SQLDialect = "postgresql",
        *, deadline=None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SharedConnectionPool. "
                "Install with: pip install psycopg2-binary"
            )
        if connect_timeout_seconds < 1:
            raise ValueError("connect_timeout_seconds must be positive")
        self._connection_string = connection_string
        self._dialect = resolve_connection_dialect(dialect, connection_string, connect_timeout_seconds, deadline=deadline)
        self._connect_timeout_seconds = connect_timeout_seconds
        self._pool_size = pool_size
        self._pool: list = []
        self._lock = threading.Lock()
        # Semaphore limits total concurrent connections to pool_size
        self._semaphore = threading.Semaphore(pool_size)

    @property
    def connection_string(self) -> str:
        return self._connection_string

    @property
    def dialect(self) -> str:
        return self._dialect.name

    @property
    def sql_dialect(self) -> SQLDialect:
        return self._dialect

    def _is_connection_alive(self, conn) -> bool:
        """Check that an idle connection is still usable."""
        if conn.closed != 0:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.rollback()
            return True
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return False

    def get_connection(
        self,
        timeout: float | None = 30.0,
        *,
        validate: bool = True,
    ):
        """Borrow a connection (blocks if pool is exhausted).

        Args:
            timeout: Seconds to wait for a free slot. ``None`` waits forever.
                     Raises ``RuntimeError`` on timeout.

        Returns:
            A live psycopg2 connection.
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(
                f"SharedConnectionPool: timed out waiting for a connection "
                f"slot (pool_size={self._pool_size})"
            )

        while True:
            with self._lock:
                if not self._pool:
                    break
                conn = self._pool.pop()
            if conn.closed == 0 and (not validate or self._is_connection_alive(conn)):
                return conn
            # The acquired slot remains reserved while unusable idle
            # connections are discarded and replaced.

        try:
            return psycopg2.connect(
                self._connection_string,
                connect_timeout=self._connect_timeout_seconds,
            )
        except Exception:
            # Connection creation failed — release the slot
            self._semaphore.release()
            raise

    def return_connection(self, conn):
        """Return a connection to the pool for reuse."""
        with self._lock:
            if len(self._pool) < self._pool_size and conn.closed == 0:
                self._pool.append(conn)
                self._semaphore.release()
                return
        # Pool full or connection closed — discard
        if conn.closed == 0:
            conn.close()
        self._semaphore.release()

    @contextmanager
    def deadline_connection(self, deadline):
        """Use a bounded, nonblocking session within the shared connection limit."""
        if not self._semaphore.acquire(timeout=remaining_seconds(deadline)):
            raise TimeoutError('PostgreSQL connection slot deadline elapsed')
        try:
            # Replace one idle session so the physical connection cap also holds.
            with self._lock:
                idle = self._pool.pop() if self._pool else None
            if idle is not None:
                idle.close()
            with _deadline_connection(self._connection_string, deadline) as conn:
                yield conn
        finally:
            self._semaphore.release()

    def close_all(self):
        """Close every idle connection in the pool (for shutdown / tests)."""
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                if conn.closed == 0:
                    conn.close()


class PoolAdapterMixin:
    """Mixin that adds optional shared-pool support to a SQL store.

    Subclasses may rely on the mixin's standalone (no-pool) connection
    pool: a hard cap on **total live** connections (idle + checked-out)
    enforced with a semaphore, so N concurrent ``_get_connection`` calls
    queue for a slot instead of each opening their own connection.  When
    a shared ``SharedConnectionPool`` is provided, its bounded methods
    are bound to the instance instead.

    Usage in a store::

        class SQLSomeStore(PoolAdapterMixin):
            def __init__(self, connection_string=None, pool_size=5, pool=None):
                self._init_pool(pool, connection_string, pool_size)
                self._ensure_table()

    Subclasses that patch ``psycopg2`` at module level (for unit tests)
    override ``_create_connection`` to use their module reference.

    Timeout semantics (P3-2): the two pool wirings differ by design —
    the standalone path's ``_get_connection(timeout=None)`` **blocks
    forever** waiting for a free slot, while a shared
    ``SharedConnectionPool.get_connection`` defaults to ``timeout=30.0``
    and raises ``PoolTimeout`` after 30 s. The same call site can thus
    hang or raise depending on wiring; long-lived batch consumers that
    must not hang should pass an explicit ``timeout=``.
    """

    def _init_pool(
        self,
        pool,
        connection_string,
        pool_size,
        dialect="postgresql",
    ):
        """Initialise pool wiring. Call from ``__init__``."""
        self._shared_pool = pool
        if pool is not None:
            self._connection_string = pool.connection_string
            self._dialect = pool.sql_dialect
            self._sql_dialect = self._dialect.name
            self._get_connection = pool.get_connection
            self._return_connection = self._return_shared_connection
        else:
            if not connection_string:
                raise ValueError("connection_string is required when pool is not provided")
            self._connection_string = connection_string
            self._dialect = resolve_connection_dialect(dialect, connection_string)
            self._sql_dialect = self._dialect.name
            self._pool_size = pool_size
            self._pool: list = []
            # Guards the idle-queue length check + append/pop, and bounds
            # total live connections (idle + active) so concurrent
            # borrowers queue for a slot instead of over-connecting.
            self._lock = threading.Lock()
            self._semaphore = threading.Semaphore(max(1, pool_size))

        # Wrap both shared-pool aliases and the standalone pool methods
        # with the session-mutation guard (SQL archive safety net).
        from fs.sql_adapter.session_mutation import guard_pool_methods
        guard_pool_methods(self, self._connection_string)

    def _return_shared_connection(self, conn):
        self._shared_pool.return_connection(conn)

    # ------------------------------------------------------------------
    # Standalone (no shared pool) connection management
    # ------------------------------------------------------------------

    def _create_connection(self):
        """Open a new database connection.

        Subclasses whose unit tests patch ``psycopg2`` at their own module
        level must override this to use their module reference.
        """
        import psycopg2

        return psycopg2.connect(self._connection_string)

    def _standalone_get_connection(self, timeout=None):
        """Borrow a standalone connection (blocks if the cap is reached).

        ``timeout=None`` blocks until a slot frees up; a numeric ``timeout``
        raises ``RuntimeError`` when it elapses.  Objects built via
        ``__new__`` (no ``_init_pool``) fall back to the legacy unbounded
        pop-or-create path.
        """
        semaphore = getattr(self, "_semaphore", None)
        if semaphore is None:
            pool = getattr(self, "_pool", None)
            if pool:
                conn = pool.pop()
                if conn.closed == 0:
                    return conn
                # Defensive close for symmetry with the semaphore path
                # (the conn may already be closed; never raise here).
                try:
                    conn.close()
                except Exception:
                    pass
            return self._create_connection()
        if timeout is None:
            semaphore.acquire()
        elif not semaphore.acquire(timeout=timeout):
            raise RuntimeError(
                f"{type(self).__name__}: timed out waiting for a connection "
                f"slot (pool_size={self._pool_size})"
            )
        try:
            with self._lock:
                conn = self._pool.pop() if self._pool else None
            if conn is not None:
                if conn.closed == 0:
                    return conn
                try:
                    conn.close()
                except Exception:
                    pass
            return self._create_connection()
        except Exception:
            semaphore.release()
            raise

    def _standalone_return_connection(self, conn):
        """Return a standalone connection; always release its slot."""
        semaphore = getattr(self, "_semaphore", None)
        if semaphore is None:
            pool = getattr(self, "_pool", None)
            if (
                pool is not None
                and len(pool) < getattr(self, "_pool_size", 0)
                and conn.closed == 0
            ):
                pool.append(conn)
            elif conn.closed == 0:
                conn.close()
            return
        try:
            with self._lock:
                if len(self._pool) < self._pool_size and conn.closed == 0:
                    self._pool.append(conn)
                    return
            if conn.closed == 0:
                conn.close()
        finally:
            semaphore.release()

    def _get_connection(self, timeout=None):
        return self._standalone_get_connection(timeout=timeout)

    def _return_connection(self, conn):
        self._standalone_return_connection(conn)

    @property
    def sql_dialect(self) -> SQLDialect:
        dialect = getattr(self, "_dialect", None)
        if dialect is None:
            dialect = resolve_sql_dialect(getattr(self, "_sql_dialect", None))
            self._dialect = dialect
        return dialect

    @staticmethod
    def bind_context(ctx, *, session_id=None):
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            guard.check_context(ctx)
            if session_id is not None and session_id != guard.ctx.session_id:
                guard.poison("SQL archive operation changed its mutation session")
                guard.check_active()

    @staticmethod
    def bind_tenant(conn, account_id: str) -> None:
        """Bind tenant identity to the current transaction for RLS.

        Must be called at the start of every transaction that touches
        RLS-protected tables.  Uses ``SET LOCAL`` so the binding
        auto-resets when the transaction ends and the connection returns
        to the pool — no risk of leaking identity to the next request.
        """
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            if account_id != guard.ctx.account_id:
                guard.poison("mutation tenant does not match its owning session")
            guard.check_active()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL app.account_id = %s", (account_id,))
