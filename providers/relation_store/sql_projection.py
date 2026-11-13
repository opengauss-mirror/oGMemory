"""Durable SQL relation-projection queue and persistence adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg2.extras import Json

from fs.sql_adapter.pool import PoolAdapterMixin
from fs.sql_adapter.schema import ensure_schema
from index.edge_projection import (
    AcceptedEdgeProjectionInput,
    EdgeProjection,
    EdgeProjectionWorker,
    bounded_projection_batch,
)


class SQLEdgeProjectionStore(PoolAdapterMixin):
    """Persists projections on canonical relation rows and owns retry jobs."""

    @staticmethod
    def _claimable_sql() -> str:
        """Shared eligibility for bounded mutation coalescing and durable retries."""
        return """(
            (status='PENDING' AND (
                mutation_batch_key IS NULL OR next_retry_at IS NULL OR next_retry_at<=NOW()
                OR NOT EXISTS (
                    SELECT 1 FROM pg_locks batch_lock
                    WHERE batch_lock.locktype='advisory' AND batch_lock.granted
                      AND batch_lock.objsubid=1
                      AND batch_lock.database=(SELECT oid FROM pg_database WHERE datname=current_database())
                      AND batch_lock.classid::bigint=(edge_projection_jobs.mutation_batch_key >> 32)
                      AND batch_lock.objid::bigint=(edge_projection_jobs.mutation_batch_key & 4294967295)
                )))
            OR (status='FAILED' AND (next_retry_at IS NULL OR next_retry_at<=NOW()))
            OR (status='PROCESSING' AND updated_at<=NOW()-INTERVAL '5 minutes')
        )"""

    def __init__(self, connection_string: str | None = None, pool_size: int = 5, pool=None,
                 dialect="postgresql", component_cache=None):
        if pool is None:
            from fs.sql_adapter.pool import SharedConnectionPool
            pool = SharedConnectionPool(connection_string,pool_size=pool_size,dialect=dialect)
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._claimed_accounts: dict[str, str] = {}
        self._claimed_tokens: dict[str, str] = {}
        self._component_cache = component_cache
        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        finally:
            self._return_connection(conn)

    @staticmethod
    def _lock_endpoints(cur, account_id, edge_ids):
        """Global write order: canonical nodes, then jobs, then relation rows."""
        cur.execute("""SELECT uri FROM context_nodes
            WHERE account_id=%s AND uri IN (
                SELECT source_uri FROM edge_projection_jobs WHERE account_id=%s AND edge_id=ANY(%s)
                UNION SELECT target_uri FROM edge_projection_jobs WHERE account_id=%s AND edge_id=ANY(%s))
            ORDER BY uri FOR UPDATE""", (account_id,account_id,edge_ids,account_id,edge_ids))
        cur.fetchall()

    def put_many(self, projections: list[EdgeProjection]) -> None:
        if not projections:
            return
        conn = self._get_connection()
        completed = []
        try:
            with conn.cursor() as cur:
                accounts = {}
                for item in projections:
                    account = self._claimed_accounts.get(item.edge_id)
                    if account is not None and item.claim_token:
                        accounts.setdefault(account, []).append(item.edge_id)
                for account, ids in sorted(accounts.items()):
                    self.bind_tenant(conn, account)
                    self._lock_endpoints(cur, account, ids)
                for item in sorted(projections, key=lambda p: p.edge_id):
                    account_id = self._claimed_accounts.get(item.edge_id)
                    if account_id is None or not item.claim_token:
                        continue
                    self.bind_tenant(conn, account_id)
                    cur.execute("""SELECT claim_token,source_revision,target_revision,
                        embedding_profile,representation_version FROM edge_projection_jobs
                        WHERE account_id=%s AND edge_id=%s AND status='PROCESSING' FOR UPDATE""",
                        (account_id,item.edge_id))
                    current = cur.fetchone()
                    if current is None or tuple(current) != (item.claim_token,item.source_revision,
                            item.target_revision,item.embedding_profile,item.representation_version):
                        continue
                    metadata = item.as_graph_metadata()
                    metadata["vector"] = list(item.vector)
                    cur.execute(
                        """
                        UPDATE relation_edges
                           SET edge_projection = %s, updated_at = NOW()
                         WHERE account_id = %s AND projection_edge_id = %s
                        """,
                        (Json(metadata),account_id,item.edge_id),
                    )
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'COMPLETE', last_error = NULL,
                               next_retry_at = NULL, claim_token = NULL, updated_at = NOW()
                         WHERE account_id = %s AND edge_id = %s
                           AND claim_token=%s
                        """,
                        (account_id,item.edge_id,item.claim_token),
                    )
                    completed.append((item.edge_id,item.claim_token))
            conn.commit()
            for edge_id, token in completed:
                if self._claimed_tokens.get(edge_id) == token:
                    self._claimed_accounts.pop(edge_id, None)
                    self._claimed_tokens.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def delete(self, edge_id: str) -> None:
        account_id = self._claimed_accounts.get(edge_id)
        token = self._claimed_tokens.get(edge_id)
        if account_id is None or token is None:
            return
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                self._lock_endpoints(cur, account_id, [edge_id])
                cur.execute("DELETE FROM edge_projection_jobs WHERE account_id=%s AND edge_id=%s AND claim_token=%s",
                            (account_id,edge_id,token))
                owned = cur.rowcount
                if not owned:
                    conn.rollback()
                    return
                cur.execute(
                    "UPDATE relation_edges SET edge_projection = NULL WHERE account_id=%s AND projection_edge_id = %s",
                    (account_id,edge_id),
                )
            conn.commit()
            if self._claimed_tokens.get(edge_id) == token:
                self._claimed_accounts.pop(edge_id, None)
                self._claimed_tokens.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    @staticmethod
    def _profile_filter(profile: dict | None) -> tuple[str, tuple]:
        if profile is None:
            return "", ()
        from providers.relation_store.projection_identity import REPRESENTATION_VERSION
        return (" AND embedding_profile=%s AND representation_version=%s",
                (str(profile["profile_id"]), profile.get("representation_version", REPRESENTATION_VERSION)))

    def claim_pending(self, account_id: str, limit: int = 128, *,
                      profile: dict | None = None) -> list[AcceptedEdgeProjectionInput]:
        profile_sql, profile_params = self._profile_filter(profile)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                # Scrub only unfinished pre-upgrade orphan payloads. Completed
                # history must not be revisited on every claim. New physical
                # deletes are covered atomically by the node delete trigger.
                cur.execute("""DELETE FROM edge_projection_jobs j
                    WHERE j.account_id=%s
                      AND j.status IN ('PENDING', 'FAILED', 'PROCESSING') AND (
                        NOT EXISTS (SELECT 1 FROM context_nodes s
                                    WHERE s.account_id=j.account_id AND s.uri=j.source_uri)
                        OR NOT EXISTS (SELECT 1 FROM context_nodes t
                                       WHERE t.account_id=j.account_id AND t.uri=j.target_uri))""",
                    (account_id,))
                cur.execute(
                    """
                    SELECT edge_id, source_uri, target_uri, relation_type, reason,
                           target_text, source_revision, target_revision
                      FROM edge_projection_jobs
                     WHERE account_id = %s
                        {profile_filter}
                       AND EXISTS (SELECT 1 FROM context_nodes s
                                    WHERE s.account_id=edge_projection_jobs.account_id
                                      AND s.uri=source_uri AND s.version=source_revision)
                       AND EXISTS (SELECT 1 FROM context_nodes t
                                    WHERE t.account_id=edge_projection_jobs.account_id
                                      AND t.uri=target_uri AND t.version=target_revision)
                       AND {claimable}
                     ORDER BY created_at, edge_id
                     LIMIT %s
                     FOR UPDATE {skip_locked}
                    """.format(claimable=self._claimable_sql(), profile_filter=profile_sql,
                               skip_locked="SKIP LOCKED" if self.sql_dialect.supports_batch_claim else ""),
                    (account_id, *profile_params, limit),
                )
                rows = cur.fetchall()
                candidates = [
                    AcceptedEdgeProjectionInput(
                        edge_id=row[0], source_uri=row[1], target_uri=row[2], relation=row[3],
                        reason=row[4], target_text=row[5], source_revision=row[6],
                        target_revision=row[7],
                    )
                    for row in rows
                ]
                rows = rows[:len(bounded_projection_batch(candidates))]
                token = uuid4().hex
                self._claimed_accounts.update((row[0], account_id) for row in rows)
                self._claimed_tokens.update((row[0], token) for row in rows)
                ids = [row[0] for row in rows]
                if ids:
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'PROCESSING', attempt_count = attempt_count + 1,
                               claim_token = %s, updated_at = NOW()
                         WHERE account_id = %s AND edge_id = ANY(%s)
                        """,
                        (token, account_id, ids),
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
                reason=row[4], target_text=row[5], source_revision=row[6], target_revision=row[7],
                claim_token=token,
            )
            for row in rows
        ]

    def mark_failed(self, edge_ids: list[str], error: str, *, claims=None) -> None:
        if not edge_ids:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                tokens = ({item.edge_id:item.claim_token for item in claims}
                          if claims is not None else dict(self._claimed_tokens))
                for edge_id in edge_ids:
                    account_id = self._claimed_accounts.get(edge_id)
                    token = tokens.get(edge_id)
                    if account_id is None or token is None:
                        continue
                    self.bind_tenant(conn, account_id)
                    cur.execute(
                        """
                        UPDATE edge_projection_jobs
                           SET status = 'FAILED', last_error = %s,
                               next_retry_at = %s, claim_token = NULL, updated_at = NOW()
                         WHERE account_id = %s AND edge_id = %s
                           AND status='PROCESSING' AND claim_token=%s
                        """,
                        (
                            error[:2000],
                            datetime.now(UTC) + timedelta(minutes=1),
                            account_id,
                            edge_id, token,
                        ),
                    )
            conn.commit()
            for edge_id in edge_ids:
                if self._claimed_tokens.get(edge_id) == tokens.get(edge_id):
                    self._claimed_accounts.pop(edge_id, None)
                    self._claimed_tokens.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def reconcile_profile(self, account_id: str, profile: dict, limit: int = 128) -> int:
        """Explicitly queue a bounded profile migration after old workers stop."""
        import json

        if not 1 <= limit <= 128:
            raise ValueError("Projection migration batch must contain 1..128 jobs")
        _, identity = self._profile_filter(profile)
        conn = self._get_connection()
        migrated = []
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute("""SELECT j.edge_id FROM edge_projection_jobs j
                    JOIN context_nodes s ON s.account_id=j.account_id AND s.uri=j.source_uri
                    JOIN context_nodes t ON t.account_id=j.account_id AND t.uri=j.target_uri
                    JOIN relation_edges r ON r.account_id=j.account_id AND r.projection_edge_id=j.edge_id
                    WHERE j.account_id=%s
                      AND (j.embedding_profile<>%s OR j.representation_version<>%s)
                    ORDER BY j.created_at,j.edge_id LIMIT %s""", (account_id,*identity,limit))
                ids = [row[0] for row in cur.fetchall()]
                if ids:
                    self._lock_endpoints(cur, account_id, ids)
                    cur.execute("""SELECT edge_id FROM edge_projection_jobs
                        WHERE account_id=%s AND edge_id=ANY(%s)
                          AND (embedding_profile<>%s OR representation_version<>%s)
                        ORDER BY edge_id FOR UPDATE""", (account_id,ids,*identity))
                    locked = [row[0] for row in cur.fetchall()]
                    for edge_id in locked:
                        cur.execute("""SELECT s.version,t.version,
                            concat_ws(E'\\n',t.abstract,t.overview,t.content),r.reason,j.source_uri
                            FROM edge_projection_jobs j
                            JOIN context_nodes s ON s.account_id=j.account_id AND s.uri=j.source_uri
                            JOIN context_nodes t ON t.account_id=j.account_id AND t.uri=j.target_uri
                            JOIN relation_edges r ON r.account_id=j.account_id AND r.projection_edge_id=j.edge_id
                            WHERE j.account_id=%s AND j.edge_id=%s""", (account_id,edge_id))
                        current = cur.fetchone()
                        if current is None:
                            continue
                        source_revision,target_revision,target_text,reason,source_uri = current
                        cur.execute("""UPDATE edge_projection_jobs
                            SET embedding_profile=%s,representation_version=%s,
                                source_revision=%s,target_revision=%s,target_text=%s,reason=%s,
                                status='PENDING',claim_token=NULL,last_error=NULL,
                                mutation_batch_key=NULL,next_retry_at=NULL,updated_at=NOW()
                            WHERE account_id=%s AND edge_id=%s""",
                            (*identity,source_revision,target_revision,target_text,reason,account_id,edge_id))
                        cur.execute("""UPDATE relation_edges SET edge_projection=NULL,updated_at=NOW()
                            WHERE account_id=%s AND projection_edge_id=%s""", (account_id,edge_id))
                        event_id = str(uuid4())
                        cur.execute("""INSERT INTO outbox_events
                            (event_id,event_type,uri,account_id,payload,status,retry_count)
                            VALUES (%s,'PROJECT_GRAPH',%s,%s,%s,'PENDING',0)""",
                            (event_id,source_uri,account_id,Json({"edge_id":edge_id})))
                        self.sql_dialect.notify_outbox(cur,"ogmem_outbox",json.dumps(
                            dict(event_id=event_id,account_id=account_id,event_type="PROJECT_GRAPH")))
                        migrated.append(edge_id)
            conn.commit()
            for edge_id in migrated:
                self._claimed_accounts.pop(edge_id, None)
                self._claimed_tokens.pop(edge_id, None)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return len(migrated)

    def has_due_work(self, account_id: str, *, profile: dict | None = None) -> bool:
        """Tenant-bound indexed probe, excluding all COMPLETE job history."""
        profile_sql, profile_params = self._profile_filter(profile)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute("""SELECT 1 FROM edge_projection_jobs
                    WHERE account_id=%s
                      AND status IN ('PENDING', 'FAILED', 'PROCESSING')
                      AND """ + self._claimable_sql() + profile_sql + " LIMIT 1",
                    (account_id, *profile_params))
                return cur.fetchone() is not None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def drain_sync(
        self,
        account_id: str,
        embedder,
        profile: dict,
        limit: int = 128,
    ) -> int:
        """Drain projections inside the existing synchronous outbox lifecycle."""
        items = self.claim_pending(account_id, limit, profile=profile)
        if not items:
            return 0
        worker = EdgeProjectionWorker(embedder, self, profile,
                                      component_cache=getattr(self, "_component_cache", None))
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
            self.mark_failed([item.edge_id for item in items], str(exc), claims=items)
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
