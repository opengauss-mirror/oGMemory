"""SQL-backed OutboxStore implementation.

Replaces file-based .outbox/*.json events with rows in outbox_events table.
Uses ``LISTEN/NOTIFY`` for low-latency wakeups and ``UPDATE … RETURNING`` for
atomic batch claims instead of filesystem lock files.
"""

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger
from core.models import ContextNode, OutboxEvent, RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = get_logger(__name__)

_PROCESSING_LOCK_CONNECT_TIMEOUT_SECONDS = 5
_PROCESSING_LOCK_TIMEOUT_SECONDS = 5.0
_PROCESSING_LOCK_POLL_SECONDS = 0.05


def _uri_processing_lock_key(uri: str) -> int:
    digest = hashlib.sha256(f"ogmemory:outbox:uri:{uri}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class SQLOutboxStore(PoolAdapterMixin):
    """Dialect-aware SQL outbox event storage.

    Stores OutboxEvents as rows in the ``outbox_events`` table.
    Locking uses ``UPDATE … RETURNING`` for atomic batch claims.
    """

    supports_batch_claim = True
    notify_channel = "ogmem_outbox"

    def __init__(
        self,
        connection_string: str | None = None,
        fs: ContextFS | None = None,
        pool_size: int = 5,
        pool=None,
        dialect: str = "postgresql",
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLOutboxStore. Install with: pip install psycopg2-binary"
            )
        self._fs = fs
        self._init_pool(pool, connection_string, pool_size, dialect)
        self.supports_batch_claim = self.sql_dialect.supports_batch_claim
        self._ensure_table()

    # ------------------------------------------------------------------
    # Connection pool: PoolAdapterMixin provides the bounded standalone
    # pool; _create_connection uses this module's psycopg2 so unit tests
    # that patch it keep working.
    # ------------------------------------------------------------------

    def _create_connection(self):
        return psycopg2.connect(self._connection_string)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_account_id(self, uri: str) -> str:
        from fs.sql_adapter.sql_context_fs import parse_uri

        return parse_uri(uri)["account"]

    def _node_uri_to_directory_uri(self, node_uri: str) -> str:
        parts = node_uri.rstrip("/").rsplit("/", 1)
        return parts[0] + "/"

    def _collect_sibling_abstracts(self, node_uri: str, ctx: RequestContext) -> list[str]:
        dir_uri = self._node_uri_to_directory_uri(node_uri)
        try:
            siblings = self._fs.list_children(dir_uri, ctx)
        except Exception:
            return []
        abstracts = []
        for sibling_uri in siblings[:20]:
            try:
                node = self._fs.read_node(sibling_uri, ctx)
                if node and node.abstract:
                    abstracts.append(node.abstract[:200])
            except Exception:
                continue
        return abstracts

    def _insert_event(self, event: OutboxEvent, account_id: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO outbox_events
                        (event_id, event_type, uri, account_id, payload,
                         status, retry_count, created_at, next_retry_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        event.event_type,
                        event.uri,
                        account_id,
                        Json(event.payload),
                        event.status,
                        event.retry_count,
                        event.created_at or datetime.now(UTC).isoformat(),
                        event.next_retry_at or None,
                    ),
                )
                self.sql_dialect.notify_outbox(
                    cur,
                    self.notify_channel,
                    json.dumps(
                        {
                            "event_id": event.event_id,
                            "account_id": account_id,
                            "event_type": event.event_type,
                        },
                        ensure_ascii=False,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _row_to_event(self, row) -> tuple[str, OutboxEvent]:
        # Rows from claim_batch include a trailing seq column;
        # rows from list_pending do not.  Slice uniformly.
        (
            event_id,
            event_type,
            uri,
            payload_raw,
            status,
            retry_count,
            created_at,
            next_retry_at,
        ) = row[:8]
        payload = (json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw) or {}
        event = OutboxEvent(
            event_id=event_id,
            event_type=event_type,
            uri=uri,
            payload=payload,
            status=status,
            retry_count=retry_count,
            created_at=str(created_at) if created_at else "",
            next_retry_at=str(next_retry_at) if next_retry_at else "",
        )
        return uri, event

    # ------------------------------------------------------------------
    # OutboxStore Protocol
    # ------------------------------------------------------------------

    def build_write_event(self, node: ContextNode) -> OutboxEvent:
        """Build an OutboxEvent for a node write without persisting it.

        Used by SQLContextFS.write_node_with_outbox to insert the event
        in the same transaction as the node write.
        """
        from commit.code_resource_events import build_code_resource_write_event

        return build_code_resource_write_event(node)

    def register_write(self, node: ContextNode, ctx: RequestContext) -> OutboxEvent:
        event = self.build_write_event(node)
        account_id = self._extract_account_id(node.uri)
        self._insert_event(event, account_id)
        return event

    def register_delete(self, uri: str, ctx: RequestContext) -> OutboxEvent:
        event = self.build_delete_event(uri)
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    @staticmethod
    def build_delete_event(uri: str) -> OutboxEvent:
        """Build a self-contained DELETE event without persisting it."""
        from commit.code_resource_events import build_code_resource_delete_event

        return build_code_resource_delete_event(uri)

    def register_archive(self, uri: str, ctx: RequestContext) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Pre-computes ids_to_delete so the worker can remove vector records
        for the archived node.

        Args:
            uri: URI of the archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        from core.models import IndexRecord

        l2_uri = uri.rstrip("/") + "/content.md"
        ids_to_delete = [
            IndexRecord.generate_id(uri, 0),
            IndexRecord.generate_id(uri, 1),
            IndexRecord.generate_id(l2_uri, 2),
        ]
        event = OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ARCHIVE_CONTEXT.value,
            uri=uri,
            payload={"ids_to_delete": ids_to_delete},
            status="PENDING",
        )
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    def register_directory(self, node: ContextNode, ctx: RequestContext) -> OutboxEvent:
        dir_uri = self._node_uri_to_directory_uri(node.uri)
        abstracts = self._collect_sibling_abstracts(node.uri, ctx)

        payload = {
            "directory_uri": dir_uri,
            "child_abstracts": abstracts,
            "filters": {
                "account_id": ctx.account_id,
                "owner_space": node.owner_space,
            },
        }

        event = OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.UPSERT_DIRECTORY.value,
            uri=dir_uri,
            payload=payload,
            status="PENDING",
        )
        account_id = self._extract_account_id(node.uri)
        self._insert_event(event, account_id)
        return event

    def list_pending(self, account_id: str) -> list[tuple[str, OutboxEvent]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, uri, payload, status,
                           retry_count, created_at, next_retry_at
                    FROM outbox_events
                    WHERE account_id = %s
                      AND status = 'PENDING'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at ASC
                    """,
                    (account_id,),
                )
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def list_recent(self, account_id: str, limit: int = 200) -> list[tuple[str, OutboxEvent]]:
        """Return recent live and failed index tasks for the task center."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, uri, payload, status,
                           retry_count, created_at, next_retry_at
                    FROM outbox_events
                    WHERE account_id = %s
                      AND status IN ('PENDING', 'PROCESSING', 'FAILED')
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (account_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def get_event(self, account_id: str, event_id: str) -> OutboxEvent | None:
        """Return one task scoped to an account."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, uri, payload, status,
                           retry_count, created_at, next_retry_at
                    FROM outbox_events
                    WHERE account_id = %s AND event_id = %s
                    """,
                    (account_id, event_id),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)
        return self._row_to_event(row)[1] if row else None

    def retry_failed(self, account_id: str, event: OutboxEvent) -> OutboxEvent | None:
        """Atomically requeue a failed task with a fresh retry budget."""
        payload = dict(event.payload)
        payload.pop("failure_reason", None)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PENDING', retry_count = 0, payload = %s,
                        next_retry_at = NULL, worker_id = NULL, locked_at = NULL
                    WHERE account_id = %s AND event_id = %s AND status = 'FAILED'
                    RETURNING event_id, event_type, uri, payload, status,
                              retry_count, created_at, next_retry_at
                    """,
                    (Json(payload), account_id, event.event_id),
                )
                row = cur.fetchone()
                if row:
                    self.sql_dialect.notify_outbox(
                        cur,
                        self.notify_channel,
                        json.dumps({
                            "event_id": event.event_id,
                            "account_id": account_id,
                            "event_type": event.event_type,
                        }, ensure_ascii=False),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return self._row_to_event(row)[1] if row else None

    def claim_batch(
        self,
        worker_id: str,
        limit: int = 100,
        timeout_seconds: int = 300,
    ) -> list[tuple[str, OutboxEvent]]:
        """Atomically claim ready events while preserving per-URI order."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH ready AS (
                        SELECT candidate.event_id
                        FROM outbox_events AS candidate
                        WHERE (
                                (
                                    candidate.status = 'PENDING'
                                    AND (
                                        candidate.next_retry_at IS NULL
                                        OR candidate.next_retry_at <= NOW()
                                    )
                                )
                                OR (
                                    candidate.status = 'PROCESSING'
                                    AND candidate.locked_at
                                        < NOW() - %s * interval '1 second'
                                )
                              )
                          AND NOT EXISTS (
                                SELECT 1
                                FROM outbox_events AS earlier
                                WHERE earlier.account_id = candidate.account_id
                                  AND earlier.uri = candidate.uri
                                  AND earlier.seq < candidate.seq
                                  AND earlier.status IN ('PENDING', 'PROCESSING')
                              )
                        ORDER BY candidate.seq ASC NULLS LAST,
                                 candidate.created_at ASC
                        FOR UPDATE OF candidate SKIP LOCKED
                        LIMIT %s
                    ),
                    claimed AS (
                        UPDATE outbox_events oe
                        SET status = 'PROCESSING',
                            worker_id = %s,
                            locked_at = NOW()
                        FROM ready
                        WHERE oe.event_id = ready.event_id
                        RETURNING oe.event_id, oe.event_type, oe.uri, oe.payload,
                                  oe.status, oe.retry_count, oe.created_at, oe.next_retry_at,
                                  oe.seq
                    )
                    SELECT * FROM claimed ORDER BY seq ASC NULLS LAST
                    """,
                    (timeout_seconds, limit, worker_id),
                )
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def count_unfinished(self, account_id: str | None = None) -> dict[str, int]:
        """Count pending/processing and dead-lettered events without claiming them."""
        conn = self._get_connection()
        params: list[str] = []
        where = "WHERE status IN ('PENDING', 'PROCESSING', 'FAILED')"
        if account_id:
            where += " AND account_id = %s"
            params.append(account_id)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'PENDING') AS pending,
                        COUNT(*) FILTER (WHERE status = 'PROCESSING') AS processing,
                        COUNT(*) FILTER (WHERE status = 'FAILED') AS failed
                    FROM outbox_events
                    {where}
                    """,
                    tuple(params),
                )
                row = cur.fetchone() or (0, 0, 0)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        pending = int(row[0] or 0)
        processing = int(row[1] or 0)
        failed = int(row[2] or 0)
        return {"pending": pending, "processing": processing,
                "failed": failed, "total": pending + processing}

    def open_listener_connection(self):
        """Create a dedicated autocommit connection for LISTEN/NOTIFY."""
        conn = psycopg2.connect(self._connection_string)
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute(f"LISTEN {self.notify_channel}")
        return conn

    @contextmanager
    def processing_lock(self, node_uri: str):
        """Serialize all side effects for one URI across worker processes."""
        lock_key = _uri_processing_lock_key(node_uri)
        conn = psycopg2.connect(
            self._connection_string,
            connect_timeout=_PROCESSING_LOCK_CONNECT_TIMEOUT_SECONDS,
        )
        acquired = False
        try:
            conn.set_session(autocommit=True)
            deadline = time.monotonic() + _PROCESSING_LOCK_TIMEOUT_SECONDS
            with conn.cursor() as cur:
                while True:
                    cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
                    acquired = bool(cur.fetchone()[0])
                    if acquired:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out acquiring URI processing lock")
                    time.sleep(min(_PROCESSING_LOCK_POLL_SECONDS, remaining))
            yield
        finally:
            if acquired and conn.closed == 0:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                except Exception:
                    conn.close()
            if conn.closed == 0:
                conn.close()

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET status = 'PROCESSING' WHERE event_id = %s",
                    (event.event_id,),
                )
            conn.commit()
            event.status = "PROCESSING"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def is_event_current(
        self,
        event_id: str,
        node_uri: str | None = None,
    ) -> bool:
        """Check whether an event still exists in the outbox table.

        Returns True if the event row exists, False if it was deleted
        (e.g. by move_node).  Raises on database errors so the caller
        can distinguish "tombstoned" from "DB unavailable".
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if node_uri is None:
                    cur.execute(
                        "SELECT 1 FROM outbox_events WHERE event_id = %s",
                        (event_id,),
                    )
                else:
                    cur.execute(
                        "SELECT 1 FROM outbox_events WHERE event_id = %s AND uri = %s",
                        (event_id, node_uri),
                    )
                exists = cur.fetchone() is not None
            conn.commit()
            return exists
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def mark_done(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM outbox_events WHERE event_id = %s",
                    (event.event_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def move_to_dlq(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'FAILED', payload = %s
                    WHERE event_id = %s
                    """,
                    (Json(event.payload), event.event_id),
                )
            conn.commit()
            event.status = "FAILED"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def increment_retry(
        self,
        event: OutboxEvent,
        node_uri: str,
        next_retry_at: datetime | None = None,
    ) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET retry_count = retry_count + 1,
                        status = 'PENDING',
                        payload = %s,
                        next_retry_at = %s,
                        worker_id = NULL,
                        locked_at = NULL
                    WHERE event_id = %s
                    """,
                    (Json(event.payload), next_retry_at, event.event_id),
                )
            conn.commit()
            event.retry_count += 1
            event.status = "PENDING"
            if next_retry_at:
                event.next_retry_at = next_retry_at.isoformat()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def try_acquire(
        self,
        event: OutboxEvent,
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Atomically claim an event using UPDATE … RETURNING."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PROCESSING',
                        worker_id = %s,
                        locked_at = NOW()
                    WHERE event_id = %s
                      AND (
                          status = 'PENDING'
                          OR (status = 'PROCESSING' AND locked_at < NOW() - %s * interval '1 second')
                      )
                    RETURNING event_id
                    """,
                    (worker_id, event.event_id, timeout_seconds),
                )
                acquired = cur.fetchone() is not None
            conn.commit()
            if acquired:
                event.status = "PROCESSING"
            return acquired
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def release(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PENDING', worker_id = NULL, locked_at = NULL
                    WHERE event_id = %s AND status = 'PROCESSING'
                    """,
                    (event.event_id,),
                )
            conn.commit()
            event.status = "PENDING"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
