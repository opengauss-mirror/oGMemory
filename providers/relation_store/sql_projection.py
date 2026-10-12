"""Durable SQL relation-projection queue and persistence adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from psycopg2.extras import Json

from fs.sql_adapter.pool import PoolAdapterMixin
from fs.sql_adapter.schema import ensure_schema
from index.edge_projection import (
    AcceptedEdgeProjectionInput,
    EdgeProjection,
    EdgeProjectionWorker,
)


class SQLEdgeProjectionStore(PoolAdapterMixin):
    """Persists projections on canonical relation rows and owns retry jobs."""

    def __init__(self, connection_string: str | None = None, pool_size: int = 5, pool=None, dialect="postgresql"):
        if pool is None:
            from fs.sql_adapter.pool import SharedConnectionPool
            pool = SharedConnectionPool(connection_string,pool_size=pool_size,dialect=dialect)
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._claimed_accounts: dict[str, str] = {}
        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        finally:
            self._return_connection(conn)

    def put_many(self, projections: list[EdgeProjection]) -> None:
        if not projections:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for item in projections:
                    account_id = self._claimed_accounts.get(item.edge_id)
                    if account_id is None:
                        continue
                    self.bind_tenant(conn, account_id)
                    metadata = item.as_graph_metadata()
                    metadata["vector"] = list(item.vector)
                    cur.execute(
                        """
                        UPDATE relation_edges
                           SET edge_projection = %s, updated_at = NOW()
                         WHERE account_id = %s AND projection_edge_id = %s
                           AND EXISTS (SELECT 1 FROM edge_projection_jobs j
                               WHERE j.account_id=%s AND j.edge_id=%s
                                 AND j.source_revision=%s AND j.target_revision=%s
                                 AND j.embedding_profile=%s AND j.representation_version=%s)
                        """,
                        (Json(metadata),account_id,item.edge_id,account_id,item.edge_id,
                         item.source_revision,item.target_revision,item.embedding_profile,item.representation_version),
                    )
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'COMPLETE', last_error = NULL,
                               next_retry_at = NULL, updated_at = NOW()
                         WHERE account_id = %s AND edge_id = %s
                           AND source_revision=%s AND target_revision=%s
                           AND embedding_profile=%s AND representation_version=%s
                        """,
                        (account_id,item.edge_id,item.source_revision,item.target_revision,
                         item.embedding_profile,item.representation_version),
                    )
                    self._claimed_accounts.pop(item.edge_id, None)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def delete(self, edge_id: str) -> None:
        account_id = self._claimed_accounts.get(edge_id)
        if account_id is None:
            return
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE relation_edges SET edge_projection = NULL WHERE projection_edge_id = %s",
                    (edge_id,),
                )
                cur.execute("DELETE FROM edge_projection_jobs WHERE edge_id = %s", (edge_id,))
            conn.commit()
            self._claimed_accounts.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def claim_pending(self, account_id: str, limit: int = 128) -> list[AcceptedEdgeProjectionInput]:
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, source_uri, target_uri, relation_type, reason,
                           target_text, source_revision, target_revision
                      FROM edge_projection_jobs
                     WHERE account_id = %s
                       AND (status = 'PENDING'
                            OR (status = 'FAILED' AND
                                (next_retry_at IS NULL OR next_retry_at <= NOW()))
                            OR (status = 'PROCESSING' AND
                                updated_at <= NOW() - INTERVAL '5 minutes'))
                     ORDER BY created_at, edge_id
                     LIMIT %s
                     FOR UPDATE {skip_locked}
                    """.format(skip_locked="SKIP LOCKED" if self.sql_dialect.supports_batch_claim else ""),
                    (account_id, limit),
                )
                rows = cur.fetchall()
                self._claimed_accounts.update((row[0], account_id) for row in rows)
                ids = [row[0] for row in rows]
                if ids:
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'PROCESSING', attempt_count = attempt_count + 1,
                               updated_at = NOW()
                         WHERE account_id = %s AND edge_id = ANY(%s)
                        """,
                        (account_id, ids),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return [
            AcceptedEdgeProjectionInput(
                edge_id=row[0], source_uri=row[1], target_uri=row[2], relation=row[3],
                reason=row[4], target_text=row[5], source_revision=row[6], target_revision=row[7]
            )
            for row in rows
        ]

    def mark_failed(self, edge_ids: list[str], error: str) -> None:
        if not edge_ids:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                by_account: dict[str, list[str]] = {}
                for edge_id in edge_ids:
                    account_id = self._claimed_accounts.get(edge_id)
                    if account_id is not None:
                        by_account.setdefault(account_id, []).append(edge_id)
                for account_id, account_edge_ids in by_account.items():
                    self.bind_tenant(conn, account_id)
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'FAILED', last_error = %s,
                               next_retry_at = %s, updated_at = NOW()
                         WHERE account_id = %s AND edge_id = ANY(%s)
                        """,
                        (
                            error[:2000],
                            datetime.now(UTC) + timedelta(minutes=1),
                            account_id,
                            account_edge_ids,
                        ),
                    )
            conn.commit()
            for edge_id in edge_ids:
                self._claimed_accounts.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def drain_sync(
        self,
        account_id: str,
        embedder,
        profile: dict,
        limit: int = 128,
    ) -> int:
        """Drain projections inside the existing synchronous outbox lifecycle."""
        items = self.claim_pending(account_id, limit)
        if not items:
            return 0
        worker = EdgeProjectionWorker(embedder, self, profile)
        try:
            from fs.sql_adapter.sql_context_fs import parse_uri
            from index.edge_projection import build_searchable_text
            from providers.token_tracker import distributed_usage_scope, embedding_scope_weights
            inputs = []
            for item in items:
                owner = parse_uri(item.source_uri)
                scope = (account_id, f"{owner['owner_type'].rstrip('s')}:{owner['owner_id']}")
                inputs.append((scope,max(1,len(build_searchable_text(item)))))
            with distributed_usage_scope(embedding_scope_weights(inputs),inputs=inputs):
                worker.project_accepted_sync(items)
        except Exception as exc:
            self.mark_failed([item.edge_id for item in items], str(exc))
            raise
        return len(items)

    def unfinished(self, account_id: str, edge_id: str) -> bool:
        """Keep a durable outbox wakeup until its projection is complete."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn,account_id)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM edge_projection_jobs WHERE account_id=%s AND edge_id=%s AND status <> 'COMPLETE'",
                            (account_id,edge_id))
                return cur.fetchone() is not None
        finally:
            conn.rollback()
            self._return_connection(conn)

    async def drain(self, account_id: str, embedder, profile: dict, limit: int = 128) -> int:
        """Async adapter retained for explicit maintenance callers."""
        import asyncio

        return await asyncio.to_thread(
            self.drain_sync,
            account_id,
            embedder,
            profile,
            limit,
        )
