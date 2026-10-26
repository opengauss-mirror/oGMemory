"""Canonical schema application — single source of truth.

All SQL DDL lives in ``db_schema.sql`` (next to this file).  Every store
calls :func:`ensure_schema` from its ``_ensure_table`` method so that
the schema is applied exactly once, idempotently, regardless of which
store is instantiated first.

Cold-start under parallel workers is guarded two ways:

* a session-scoped PostgreSQL advisory lock serialises the DDL so
  concurrent processes cannot deadlock on ``AccessExclusiveLock`` while
  racing ``CREATE TABLE IF NOT EXISTS``;
* a ``to_regclass('public.graph_nodes')`` presence check short-circuits
  the whole DDL re-run once the schema is already applied (the script is
  one idempotent block), so fresh worker processes never take the heavy
  locks at all.
"""

import logging
from pathlib import Path

from fs.sql_adapter.dialect import SQLDialect, resolve_sql_dialect

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "db_schema.sql").read_text(encoding="utf-8")

# Arbitrary stable int identifying the schema-DDL advisory lock.
_SCHEMA_LOCK_KEY = 741010016003217

# Process-level guard: ensure_schema only needs to run once.
# Without this, concurrent SQLContextFS instantiations all run DDL
# simultaneously, causing PG deadlocks (ACCESS EXCLUSIVE lock contention).
_schema_ensured: set[str] = set()


def _backend_is_postgres(conn) -> bool:
    """True only for real psycopg connections.

    Mock/fake connections lack an integral ``server_version`` and silently
    skip the advisory lock and short-circuit (their tests drive the DDL
    directly).
    """
    return isinstance(getattr(conn, "server_version", None), int)


def ensure_schema(conn, dialect: str | SQLDialect = "postgresql") -> None:
    """Apply the canonical schema (idempotent).

    Runs the full ``db_schema.sql`` within a single psycopg2 transaction.
    Uses libpq's ``PQexec`` via ``autocommit`` mode to correctly handle
    multi-statement SQL with ``DO $$ ... $$`` blocks.

    Guarded by a process-level flag so DDL only executes once, even when
    multiple SQLContextFS instances are created concurrently.  On real
    PostgreSQL backends a session-scoped advisory lock serialises DDL
    execution across processes, and the presence of ``graph_nodes``
    (the schema's keystone table) short-circuits the DDL re-run entirely.
    """
    adapter = resolve_sql_dialect(dialect)
    if adapter.name in _schema_ensured:
        return
    was_autocommit = conn.autocommit
    locked = False
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            if _backend_is_postgres(conn):
                # Session-scoped lock: safe under autocommit, each process
                # waits its turn before touching any DDL.
                cur.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_KEY,))
                locked = True
                cur.execute("SELECT to_regclass('public.graph_nodes') IS NOT NULL")
                row = cur.fetchone()
                if row and row[0]:
                    # db_schema.sql is one idempotent block: if the keystone
                    # table exists the whole schema is already applied.
                    _schema_ensured.add(adapter.name)
                    logger.info(
                        "Schema already applied (graph_nodes present, dialect=%s); DDL skipped",
                        adapter.name,
                    )
                    return
            adapter.validate_connection(cur)
            cur.execute(adapter.prepare_schema(_SCHEMA_SQL))
        _schema_ensured.add(adapter.name)
        logger.info("Schema ensured from db_schema.sql (dialect=%s)", adapter.name)
    finally:
        if locked:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_KEY,))
            except Exception:
                logger.warning("Failed to release schema advisory lock", exc_info=True)
        conn.autocommit = was_autocommit
