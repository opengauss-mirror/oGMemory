"""SQL-backed OutboxStore implementation.

Replaces file-based .outbox/*.json events with rows in outbox_events table.
Uses ``LISTEN/NOTIFY`` for low-latency wakeups and ``UPDATE … RETURNING`` for
atomic batch claims instead of filesystem lock files.
"""

import hashlib
import json
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger
from core.models import ContextNode, OutboxEvent, RequestContext
from fs.sql_adapter.outbox_receipt import PROJECTION_RECEIPT_SQL
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
_PROCESSING_CONNECTIONS = ContextVar("sql_outbox_processing_connections", default=None)
_NO_UNFINISHED_PREDECESSOR = """
    NOT EXISTS (
        SELECT 1 FROM outbox_events AS earlier
        WHERE earlier.account_id = candidate.account_id
          AND earlier.uri = candidate.uri
          AND earlier.seq < candidate.seq
          AND earlier.status IN ('PENDING', 'PROCESSING')
          AND earlier.event_type <> ALL(%s)
          AND (earlier.event_type <> 'PROJECT_GRAPH' OR candidate.event_type = 'PROJECT_GRAPH')
    )
"""


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
    # Connection pool (standalone fallback — used when pool=None)
    # ------------------------------------------------------------------

    def _get_connection(self):
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn
        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn):
        if len(self._pool) < self._pool_size and conn.closed == 0:
            self._pool.append(conn)
        else:
            conn.close()

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
        self.bind_context(ctx)
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
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            if account_id != guard.ctx.account_id:
                guard.poison("outbox account does not match its mutation session")
            guard.check_active()
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
            from commit.outbox_scope import record_persisted_event
            record_persisted_event(event.uri, event.event_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _row_to_event(self, row) -> tuple[str, OutboxEvent]:
        # Claim rows append sequence and ownership fields after the event data.
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
        if len(row) >= 11:
            event._outbox_claim = (row[9], row[10])
        return uri, event

    @staticmethod
    def _claim_filter(event):
        claim = getattr(event, "_outbox_claim", None)
        if claim is None:
            return " AND status <> 'CANCELLED'", ()
        return " AND status = 'PROCESSING' AND worker_id = %s AND locked_at = %s", claim

    def ensure_processing_owned(self, event: OutboxEvent, node_uri: str) -> None:
        """Validate the claim within the URI lock that protects its side effects."""
        claim = getattr(event, "_outbox_claim", None)
        if claim is None:
            return
        conn = (_PROCESSING_CONNECTIONS.get() or {}).get((self._connection_string, node_uri))
        if conn is None or conn.closed:
            raise RuntimeError("Outbox URI processing lock is required")
        with conn.cursor() as cur:
            cur.execute("SELECT status, worker_id, locked_at FROM outbox_events "
                        "WHERE event_id = %s AND uri = %s", (event.event_id, node_uri))
            row = cur.fetchone()
        # A verified worker can finish its cleanup after source relocation.
        if (row is None or row[0] == "CANCELLED") and getattr(event, "_outbox_verified_connection", None) is conn:
            return
        if row != ("PROCESSING", *claim):
            raise RuntimeError("Outbox processing claim changed")
        event._outbox_verified_connection = conn

    def _check_claim_update(self, cur, event, node_uri):
        if getattr(event, "_outbox_claim", None) is not None and cur.rowcount == 0:
            self.ensure_processing_owned(event, node_uri)

    # ------------------------------------------------------------------
    # OutboxStore Protocol
    # ------------------------------------------------------------------

    def register_once(self, event: OutboxEvent, ctx: RequestContext) -> None:
        self.bind_context(ctx)
        if self.get_event_status(event.uri, event.event_id) is not None:
            return
        with self.processing_lock(event.uri):
            if self.get_event_status(event.uri, event.event_id) is None:
                self._insert_event(event, ctx.account_id)

    def save_payload(self, event: OutboxEvent) -> None:
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE outbox_events SET payload = %s WHERE event_id = %s" + ownership,
                            (Json(event.payload), event.event_id, *claim))
                self._check_claim_update(cur, event, event.uri)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def build_write_event(self, node: ContextNode) -> OutboxEvent:
        """Build an OutboxEvent for a node write without persisting it.

        Used by SQLContextFS.write_node_with_outbox to insert the event
        in the same transaction as the node write.
        """
        from commit.code_resource_events import build_code_resource_write_event

        return build_code_resource_write_event(node)

    def register_write(self, node: ContextNode, ctx: RequestContext) -> OutboxEvent:
        self.bind_context(ctx)
        event = self.build_write_event(node)
        account_id = self._extract_account_id(node.uri)
        self._insert_event(event, account_id)
        return event

    def register_delete(self, uri: str, ctx: RequestContext) -> OutboxEvent:
        self.bind_context(ctx)
        event = self.build_delete_event(uri)
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    @staticmethod
    def build_delete_event(uri: str) -> OutboxEvent:
        """Build a self-contained DELETE event without persisting it."""
        from commit.code_resource_events import build_code_resource_delete_event

        return build_code_resource_delete_event(uri)

    def register_archive(self, uri: str, ctx: RequestContext, *, record_ids: list[str] | None = None) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Pre-computes ids_to_delete so the worker can remove vector records
        for the archived node.

        Args:
            uri: URI of the archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        self.bind_context(ctx)
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
            payload={"ids_to_delete": record_ids if record_ids is not None else ids_to_delete},
            status="PENDING",
        )
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    def register_directory(self, node: ContextNode, ctx: RequestContext) -> OutboxEvent:
        self.bind_context(ctx)
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
        """Discover due events and claims whose five-minute lease elapsed."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, uri, payload, status,
                           retry_count, created_at, next_retry_at
                    FROM outbox_events
                    WHERE account_id = %s
                      AND (
                          (status = 'PENDING' AND (next_retry_at IS NULL OR next_retry_at <= NOW()))
                          OR (status = 'PROCESSING' AND locked_at < NOW() - INTERVAL '5 minutes')
                      )
                    ORDER BY created_at ASC
                    """,
                    (account_id,),
                )
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def future_pending_delay(
        self, account_ids: list[str] | None, *, exclude_event_types: list[str] | None = None,
    ) -> float | None:
        """Find the next retry. None scans all accounts and includes overdue retries."""
        if account_ids == []:
            return None
        excluded = exclude_event_types or []
        params = (excluded,)
        if account_ids is None:
            paused_projections = ["PROJECT_GRAPH"] if "PROJECT_GRAPH" in excluded else []
            scope = f"AND {_NO_UNFINISHED_PREDECESSOR}"
            params += (paused_projections,)
        else:
            scope = "AND candidate.account_id = ANY(%s) AND candidate.next_retry_at > NOW()"
            params += (account_ids,)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT EXTRACT(EPOCH FROM (MIN(candidate.next_retry_at) - NOW()))
                    FROM outbox_events AS candidate
                    WHERE candidate.status = 'PENDING' AND candidate.next_retry_at IS NOT NULL
                      AND candidate.event_type <> ALL(%s)
                      {scope}
                    """,
                    params,
                )
                row = cur.fetchone()
            conn.commit()
            return float(row[0]) if row and row[0] is not None else None
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def list_pending_events(self, references) -> list[tuple[str, OutboxEvent]]:
        """Load requested events and their unfinished per-URI predecessors."""
        references = tuple(references)
        if not references:
            return []
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT candidate.event_id, candidate.event_type,
                               candidate.uri, candidate.payload, candidate.status,
                               candidate.retry_count, candidate.created_at, candidate.next_retry_at
                            FROM outbox_events AS candidate
                            WHERE candidate.status IN ('PENDING', 'PROCESSING')
                              AND EXISTS (
                                SELECT 1 FROM outbox_events AS requested
                                WHERE (requested.uri, requested.event_id) IN %s
                                  AND requested.account_id = candidate.account_id
                                  AND requested.uri = candidate.uri
                                  AND (candidate.event_id = requested.event_id
                                       OR candidate.seq < requested.seq))
                            ORDER BY candidate.seq ASC NULLS LAST""", (references,))
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
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

    def get_event_status(self, node_uri: str, event_id: str) -> str | None:
        account_id = self._extract_account_id(node_uri)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM outbox_events WHERE account_id = %s AND uri = %s AND event_id = %s",
                    (account_id, node_uri, event_id),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)
        return row[0] if row else None

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
        *,
        exclude_event_types: list[str] | None = None,
        event_ids: list[str] | None = None,
    ) -> list[tuple[str, OutboxEvent]]:
        """Claim ready work with ordered node and revision-fenced graph lanes."""
        excluded = exclude_event_types or []
        # Graph wakeups represent an independent, revision-fenced projection.
        # Pausing them must not hold up ordinary indexing for the same URI.
        # Keep all other predecessor ordering, including agent-memory jobs.
        paused_projections = ["PROJECT_GRAPH"] if "PROJECT_GRAPH" in excluded else []
        scope = " AND candidate.event_id = ANY(%s)" if event_ids is not None else ""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                busy = []
                rows = []
                while (remaining := limit - len(rows)) > 0:
                    params = (timeout_seconds, busy, excluded)
                    if event_ids is not None:
                        params += (event_ids,)
                    params += (paused_projections, remaining)
                    cur.execute(
                        f"""
                            SELECT candidate.event_id, candidate.uri
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
                              AND candidate.event_id <> ALL(%s)
                              AND candidate.event_type <> ALL(%s)
                              {scope}
                              AND {_NO_UNFINISHED_PREDECESSOR}
                            ORDER BY candidate.seq ASC NULLS LAST,
                                     candidate.created_at ASC
                            FOR UPDATE OF candidate SKIP LOCKED
                            LIMIT %s
                        """,
                        params,
                    )
                    ready = cur.fetchall()
                    # Use the processing lock key before replacing an expired claim.
                    # The transaction lock is released when the claim commits.
                    for event_id, uri in ready:
                        cur.execute("SELECT pg_try_advisory_xact_lock(%s)",
                                    (_uri_processing_lock_key(uri),))
                        if not cur.fetchone()[0]:
                            busy.append(event_id)
                            continue
                        cur.execute(
                            """
                            UPDATE outbox_events SET status = 'PROCESSING',
                                worker_id = %s, locked_at = NOW()
                            WHERE event_id = %s
                            RETURNING event_id, event_type, uri, payload, status,
                                      retry_count, created_at, next_retry_at,
                                      seq, worker_id, locked_at
                            """, (worker_id, event_id),
                        )
                        rows.append(cur.fetchone())
                    if len(ready) < remaining:
                        break
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def count_unfinished(
        self, account_id: str | None = None, *, exclude_event_types: list[str] | None = None,
    ) -> dict[str, int]:
        """Count unfinished tasks, optionally excluding a paused feature for readiness.

        The default still includes every event type for task/status reporting.
        """
        conn = self._get_connection()
        params: list[object] = []
        where = "WHERE status IN ('PENDING', 'PROCESSING', 'FAILED')"
        if account_id:
            where += " AND account_id = %s"
            params.append(account_id)
        if exclude_event_types:
            where += " AND event_type <> ALL(%s)"
            params.append(exclude_event_types)
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
    def processing_lock(self, node_uri: str, *, timeout_seconds: float = _PROCESSING_LOCK_TIMEOUT_SECONDS):
        """Serialize all side effects for one URI across worker processes."""
        connections = _PROCESSING_CONNECTIONS.get() or {}
        current = connections.get((self._connection_string, node_uri))
        if current is not None:
            with current.cursor() as cur:
                cur.execute("SELECT 1")
            yield
            return
        lock_key = _uri_processing_lock_key(node_uri)
        conn = psycopg2.connect(
            self._connection_string,
            connect_timeout=_PROCESSING_LOCK_CONNECT_TIMEOUT_SECONDS,
        )
        acquired = False
        try:
            conn.set_session(autocommit=True)
            deadline = time.monotonic() + timeout_seconds
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
            token = _PROCESSING_CONNECTIONS.set({**connections, (self._connection_string, node_uri): conn})
            try:
                yield
            finally:
                _PROCESSING_CONNECTIONS.reset(token)
        finally:
            if acquired and conn.closed == 0:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                except Exception:
                    conn.close()
            if conn.closed == 0:
                conn.close()

    @contextmanager
    def claim_processing_event(self, event: OutboxEvent, node_uri: str, worker_id: str):
        with self.processing_lock(node_uri):
            yield self.try_acquire(event, node_uri, worker_id)

    @contextmanager
    def claim_agent_event(self, event: OutboxEvent, worker_id: str):
        """Reload checkpoints under the URI lock and hold it through acknowledgement."""
        with ExitStack() as stack:
            try:
                stack.enter_context(self.processing_lock(event.uri, timeout_seconds=0))
            except TimeoutError:
                yield False
                return
            if getattr(event, "_outbox_claim", None) is not None:
                try:
                    self.ensure_processing_owned(event, event.uri)
                except RuntimeError:
                    yield False
                    return
            ctx = RequestContext(**event.payload["context"])
            fresh = self.get_event(ctx.account_id, event.event_id)
            if fresh is None or fresh.status not in {"PENDING", "PROCESSING"}:
                yield False
                return
            event.payload, event.retry_count = fresh.payload, fresh.retry_count
            event.status, event.next_retry_at = fresh.status, fresh.next_retry_at
            if event.next_retry_at and datetime.fromisoformat(event.next_retry_at) > datetime.now(UTC):
                yield False
                return
            if getattr(event, "_outbox_claim", None) is None and not self.try_acquire(event, event.uri, worker_id):
                yield False
                return
            yield True

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET status = 'PROCESSING' WHERE event_id = %s" + ownership,
                    (event.event_id, *claim),
                )
                self._check_claim_update(cur, event, node_uri)
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
        """Return whether a persisted event remains current, including DONE and FAILED."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if node_uri is None:
                    cur.execute(
                        "SELECT 1 FROM outbox_events WHERE event_id = %s AND status <> 'CANCELLED'",
                        (event_id,),
                    )
                else:
                    cur.execute(
                        "SELECT 1 FROM outbox_events WHERE event_id = %s AND uri = %s AND status <> 'CANCELLED'",
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
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if event.event_type == EventType.EXTRACT_AGENT_MEMORY.value:
                    audit = {key: event.payload[key] for key in ("trace_uri", "value_reviews") if key in event.payload}
                    cur.execute("UPDATE outbox_events SET status = 'DONE', payload = %s WHERE event_id = %s" + ownership,
                                (Json(audit), event.event_id, *claim))
                else:
                    cur.execute(
                        f"UPDATE outbox_events SET status = 'DONE', payload = {PROJECTION_RECEIPT_SQL}, "
                        "worker_id = NULL, locked_at = NULL "
                        "WHERE event_id = %s AND uri = %s" + ownership,
                        (event.event_id, node_uri, *claim),
                    )
                self._check_claim_update(cur, event, node_uri)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def move_to_dlq(self, event: OutboxEvent, node_uri: str) -> None:
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'FAILED', payload = %s
                    WHERE event_id = %s
                    """ + ownership,
                    (Json(event.payload), event.event_id, *claim),
                )
                self._check_claim_update(cur, event, node_uri)
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
        ownership, claim = self._claim_filter(event)
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
                    """ + ownership,
                    (Json(event.payload), next_retry_at, event.event_id, *claim),
                )
                self._check_claim_update(cur, event, node_uri)
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

    def defer(
        self,
        event: OutboxEvent,
        node_uri: str,
        next_retry_at: datetime,
    ) -> None:
        """Reschedule an owned wakeup while its projection is progressing."""
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET status = 'PENDING', next_retry_at = %s, "
                    "worker_id = NULL, locked_at = NULL "
                    "WHERE event_id = %s AND uri = %s" + ownership,
                    (next_retry_at, event.event_id, node_uri, *claim),
                )
                self._check_claim_update(cur, event, node_uri)
            conn.commit()
            event.status = "PENDING"
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
                    WHERE event_id = %s AND uri = %s
                      AND (
                          (status = 'PENDING' AND (next_retry_at IS NULL OR next_retry_at <= NOW()))
                          OR (status = 'PROCESSING' AND locked_at < NOW() - %s * interval '1 second')
                      )
                    RETURNING worker_id, locked_at
                    """,
                    (worker_id, event.event_id, node_uri, timeout_seconds),
                )
                claim = cur.fetchone()
                acquired = claim is not None
            conn.commit()
            if acquired:
                event.status = "PROCESSING"
                event._outbox_claim = tuple(claim)
            return acquired
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def release(self, event: OutboxEvent, node_uri: str) -> None:
        if getattr(event, "_outbox_claim", None) is not None and event.status == "PENDING":
            return
        ownership, claim = self._claim_filter(event)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PENDING', worker_id = NULL, locked_at = NULL
                    WHERE event_id = %s AND status = 'PROCESSING'
                    """ + ownership,
                    (event.event_id, *claim),
                )
                self._check_claim_update(cur, event, node_uri)
            conn.commit()
            event.status = "PENDING"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
