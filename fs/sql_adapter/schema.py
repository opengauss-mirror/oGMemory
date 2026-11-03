"""Canonical schema application — single source of truth.

All SQL DDL lives in ``db_schema.sql`` (next to this file).  Every store
calls :func:`ensure_schema` from its ``_ensure_table`` method so that
the schema is applied exactly once, idempotently, regardless of which
store is instantiated first.
"""

import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from psycopg2.errors import DuplicateObject, DuplicateTable, UniqueViolation
from psycopg2.extensions import TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS

from fs.sql_adapter.dialect import SQLDialect, resolve_sql_dialect

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "db_schema.sql").read_text(encoding="utf-8")

# Serialize DDL only for the same resolved storage target, across store pools.
_schema_ensured: set[tuple] = set()
_schema_locks: dict[tuple, threading.Lock] = {}
_schema_locks_guard = threading.Lock()
_unidentified_schema_lock = threading.Lock()


def _schema_target(conn, raw, adapter, *, in_transaction):
    savepoint = "schema_lookup_" + uuid4().hex if in_transaction else None
    with conn.cursor() as cur:
        if savepoint:
            cur.execute(f"SAVEPOINT {savepoint}")
        try:
            cur.execute(
                "SELECT pg_catalog.inet_server_addr()::pg_catalog.text, pg_catalog.inet_server_port(), "
                "pg_catalog.current_database(), "
                "(SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()), "
                "current_user, pg_catalog.current_schema(), pg_catalog.current_schemas(true)"
            )
            row = cur.fetchone()
            namespaces = {}
            if row and len(row) == 7 and isinstance(row[6], (list, tuple)):
                cur.execute("SELECT nspname, oid FROM pg_catalog.pg_namespace "
                            "WHERE nspname = ANY(pg_catalog.current_schemas(true))")
                namespaces = dict(cur.fetchall())
        except Exception:
            if savepoint:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            if savepoint:
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
    if not row or len(row) != 7:
        return None
    address, port, database, database_oid, role, schema, search_path = row
    if not database or not database_oid or not role or not schema or not isinstance(search_path, (list, tuple)):
        return None
    if schema not in namespaces or any(name not in namespaces for name in search_path):
        return None
    if address and port:
        endpoint = ("tcp", address, port)
    else:
        info = raw.info
        if not info.host or not info.host.startswith(("/", "@")) or not info.port:
            return None
        endpoint = ("unix", info.host, info.port)
    resolved_path = tuple((name, namespaces[name]) for name in search_path)
    return adapter.name, endpoint, database, database_oid, role, schema, resolved_path


def _target_lock(target):
    if target is None:
        return _unidentified_schema_lock
    with _schema_locks_guard:
        return _schema_locks.setdefault(target, threading.Lock())


def ensure_schema(conn, dialect: str | SQLDialect = "postgresql") -> None:
    """Apply the canonical schema (idempotent).

    Cache only complete, durable initialization of a resolved server/database,
    effective search path and role. An existing caller transaction keeps its
    pending work and owns any uncached DDL; that DDL is not cached before commit.
    """
    adapter = resolve_sql_dialect(dialect)
    from fs.sql_adapter.session_mutation import active_session_guard

    guard = active_session_guard()
    raw = guard.raw if guard is not None and conn is guard.connection else conn
    if raw is not conn:
        guard.check_active()
    transaction_status = raw.get_transaction_status()
    if transaction_status not in (TRANSACTION_STATUS_IDLE, TRANSACTION_STATUS_INTRANS):
        raise RuntimeError("Schema initialization requires an idle connection or usable transaction")
    idle = transaction_status == TRANSACTION_STATUS_IDLE
    was_autocommit = raw.autocommit
    try:
        if idle and not was_autocommit:
            raw.autocommit = True
        target = _schema_target(conn, raw, adapter, in_transaction=not idle)
        # Caller transactions can already hold DDL locks awaited by an idle initializer.
        # Keep their subsequent DDL in database lock ordering alone.
        with _target_lock(target) if idle else nullcontext():
            if target is not None and target in _schema_ensured:
                return
            with conn.cursor() as cur:
                adapter.validate_connection(cur)
                ddl = adapter.prepare_schema(_SCHEMA_SQL)
                try:
                    cur.execute(ddl)
                except (DuplicateObject, DuplicateTable, UniqueViolation):
                    if not idle:
                        raise
                    # A transactional initializer may have committed while this
                    # autocommit DDL waited on its newly created catalog rows.
                    cur.execute(ddl)
            if idle and target is not None:
                _schema_ensured.add(target)
            logger.info("Schema ensured from db_schema.sql (dialect=%s)", adapter.name)
    finally:
        if idle and raw.autocommit != was_autocommit:
            raw.autocommit = was_autocommit
