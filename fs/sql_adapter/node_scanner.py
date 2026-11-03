"""Canonical-node scanner for the SQL storage adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

from core.enums import EventType
from core.models import ContextNode
from fs.sql_adapter.pool import PoolAdapterMixin


def _best_effort_rollback(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


class SQLCanonicalNodeScanner:
    """Page through active SQL nodes using the shared storage pool."""

    def __init__(self, pool, *, statement_timeout_ms: int = 30_000) -> None:
        if statement_timeout_ms < 1:
            raise ValueError("statement_timeout_ms must be positive")
        self._pool = pool
        self._statement_timeout_ms = statement_timeout_ms

    def _set_statement_timeout(self, cur) -> None:
        cur.execute(
            "SELECT set_config(%s, %s, true)",
            ("statement_timeout", str(self._statement_timeout_ms)),
        )

    def projection_is_quiescent(self, *, account_id: str) -> bool:
        """Return whether no canonical projection work can still change reads."""
        conn = self._pool.get_connection(validate=False)
        try:
            PoolAdapterMixin.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                self._set_statement_timeout(cur)
                cur.execute(
                    """
                    SELECT 1 FROM outbox_events
                    WHERE account_id=%s
                      AND status IN ('PENDING','PROCESSING')
                      AND event_type <> %s
                    LIMIT 1
                    """,
                    (account_id, EventType.EXTRACT_AGENT_MEMORY.value),
                )
                pending = cur.fetchone()
            conn.rollback()
            return pending is None
        except Exception:
            _best_effort_rollback(conn)
            raise
        finally:
            self._pool.return_connection(conn)

    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: Sequence[str],
        page_size: int,
    ) -> Iterator[ContextNode]:
        last_uri = ""
        scopes = list(dict.fromkeys(str(scope) for scope in owner_spaces if scope))
        while scopes:
            conn = self._pool.get_connection(validate=False)
            try:
                PoolAdapterMixin.bind_tenant(conn, account_id)
                with conn.cursor() as cur:
                    self._set_statement_timeout(cur)
                    cur.execute(
                        """
                        SELECT uri, context_type, category, level, owner_space,
                               abstract, overview, content, metadata
                        FROM context_nodes
                        WHERE account_id=%s AND status='ACTIVE'
                          AND owner_space=ANY(%s) AND uri>%s
                        ORDER BY uri LIMIT %s
                        """,
                        (account_id, scopes, last_uri, page_size),
                    )
                    rows = cur.fetchall()
                conn.rollback()
            except Exception:
                _best_effort_rollback(conn)
                raise
            finally:
                self._pool.return_connection(conn)
            if not rows:
                return
            for row in rows:
                metadata = row[8]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                yield ContextNode(
                    uri=str(row[0]),
                    context_type=str(row[1]),
                    category=str(row[2]),
                    level=int(row[3]),
                    owner_space=str(row[4]),
                    abstract=str(row[5] or ""),
                    overview=str(row[6] or ""),
                    content=str(row[7] or ""),
                    metadata=dict(metadata or {}),
                )
            last_uri = str(rows[-1][0])
            if len(rows) < page_size:
                return
