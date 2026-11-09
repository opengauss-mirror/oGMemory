"""PostgreSQL-backed RelationStore implementation.

Stores relation edges in the ``relation_edges`` table instead of
per-node .relations.json files.
"""

from __future__ import annotations

import json
import logging

from core.errors import AccessDeniedError
from core.models import RelationEdge, RelationNeighbor, RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


class SQLRelationStore(PoolAdapterMixin):

    """PostgreSQL implementation of RelationStore.

    Edges are stored as rows in ``relation_edges`` with a composite
    primary key ``(from_uri, to_uri, relation_type)`` for idempotent
    upsert.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
        dialect: str = "postgresql",
        edge_projection_profile_id: str | None = None,
        edge_projection_representation: str = "relation-target-v2",
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLRelationStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._ensure_table()
        self._edge_projection_profile_id = edge_projection_profile_id
        self._edge_projection_representation = edge_projection_representation

    _edge_projection_profile_id = None
    _edge_projection_representation = "relation-target-v2"

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
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to ensure schema: {exc}") from exc
        finally:
            self._return_connection(conn)

    def _extract_account_id(self, uri: str) -> str:
        from fs.sql_adapter.sql_context_fs import parse_uri

        return parse_uri(uri)["account"]

    def _ensure_accessible(self, uri: str, ctx: RequestContext, *, capability=None) -> None:
        from fs.sql_adapter.sql_context_fs import parse_uri

        try:
            components = parse_uri(uri)
        except ValueError as exc:
            raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format") from exc

        account_id = components["account"]
        if account_id != ctx.account_id:
            raise AccessDeniedError(
                uri,
                ctx.account_id,
                f"URI belongs to account '{account_id}'",
            )

        owner_type = components.get("owner_type", "")
        owner_id = components.get("owner_id", "")

        # Sessions have no owner isolation
        if not owner_id or owner_type == "sessions":
            return

        owner_space = f"{owner_type.rstrip('s')}:{owner_id}"

        vos = ctx.visible_owner_spaces
        if vos:
            if owner_space not in vos:
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    f"owner_space '{owner_space}' not accessible",
                )
        else:
            # Strict fallback: only own user/agent space
            if owner_type == "users" and owner_id != ctx.user_id:
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    "owner_space mismatch",
                )
            if owner_type == "agents" and (not ctx.agent_id or owner_id != ctx.agent_id):
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    "owner_space mismatch",
                )

    # ------------------------------------------------------------------
    # RelationStore Protocol
    # ------------------------------------------------------------------

    def _parse_relations_json(self, relations_raw) -> list[RelationEdge]:
        """Parse relations JSON from context_nodes.relations column."""
        if not relations_raw:
            return []
        rel_list = json.loads(relations_raw) if isinstance(relations_raw, str) else relations_raw
        return [
            RelationEdge(
                from_uri=r["from_uri"],
                to_uri=r["to_uri"],
                relation_type=r["relation_type"],
                weight=r["weight"],
                reason=r["reason"],
            )
            for r in rel_list
        ]

    def _fallback_from_context_nodes(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        """Fallback: read relations from context_nodes.relations column.

        Used when relation_edges table has no data (write path hasn't synced).
        Owner_space access is enforced before reading.
        """
        # Re-validate owner_space before fallback query
        self._ensure_accessible(uri, ctx)

        logger.warning(
            "relation_edges empty for uri=%s account=%s, falling back to context_nodes.relations",
            uri,
            ctx.account_id,
        )

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT relations FROM context_nodes
                    WHERE uri = %s AND account_id = %s
                    """,
                    (uri, ctx.account_id),
                )
                row = cur.fetchone()
        except Exception as exc:
            logger.error(
                "Fallback query failed for uri=%s account=%s: %s",
                uri,
                ctx.account_id,
                exc,
            )
            raise
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            logger.info(
                "Fallback found no context_nodes row for uri=%s account=%s",
                uri,
                ctx.account_id,
            )
            return []

        edges = self._parse_relations_json(row[0])
        logger.info(
            "Fallback returned %d edges for uri=%s account=%s",
            len(edges),
            uri,
            ctx.account_id,
        )
        return edges

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_uri, to_uri, relation_type, weight, reason
                    FROM relation_edges
                    WHERE from_uri = %s AND account_id = %s
                    """,
                    (uri, ctx.account_id),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        # If relation_edges has data, use it (authoritative source)
        if rows:
            return [
                RelationEdge(
                    from_uri=r[0],
                    to_uri=r[1],
                    relation_type=r[2],
                    weight=float(r[3]),
                    reason=r[4],
                )
                for r in rows
            ]

        # Fallback: read from context_nodes.relations column
        return self._fallback_from_context_nodes(uri, ctx)

    def get_traversal_neighbors(
        self, uri: str, ctx: RequestContext, *, limit: int = 128,
        embedding_profile: dict | None = None,
    ) -> list[dict]:
        """Read both orientations, limiting unique destinations before parallel edges.

        Endpoint authorization is still enforced on every returned relation;
        callers must hydrate current content through ContextReader as well.
        This is a read-only seam over the existing canonical relation table.
        With an embedding profile, match the experiment-33 collector's ordering:
        reject incompatible/stale projections and invalid weights BEFORE the
        unique-destination cap. Lifecycle/authorization remain hydration gates;
        moving those before the cap would be a different selection algorithm.
        """
        from core.uri_parser import strip_l2_suffix
        anchor = strip_l2_suffix(uri)
        self._ensure_accessible(anchor, ctx, capability="read")
        if not 1 <= limit <= 128:
            raise ValueError("Traversal destination limit must be between 1 and 128")
        projection_gate = ""
        params = (anchor, ctx.account_id, anchor, anchor)
        if embedding_profile is not None:
            identity = {key: embedding_profile.get(profile_key) for key, profile_key in (
                ("embedding_profile", "profile_id"), ("model", "model"),
                ("dimension", "dimension"), ("representation_version", "representation_version"))
                if embedding_profile.get(profile_key) is not None}
            # JSON comparisons deliberately avoid casts of untrusted projection
            # values. CASE keeps malformed/non-array vectors from making SQL fail.
            projection_gate = """
                           AND re.weight >= 0 AND re.weight < 'Infinity'::double precision
                           AND re.edge_projection @> %s::jsonb
                           AND re.edge_projection->'source_revision' = s.version::text::jsonb
                           AND re.edge_projection->'target_revision' = t.version::text::jsonb
                           AND jsonb_typeof(re.edge_projection->'vector') = 'array'
                           AND re.edge_projection->'dimension' = (
                               CASE WHEN jsonb_typeof(re.edge_projection->'vector') = 'array'
                                    THEN jsonb_array_length(re.edge_projection->'vector') ELSE -1 END)::text::jsonb
            """
            params += (json.dumps(identity, sort_keys=True),)
        params += (limit,)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH adjacent AS (
                        SELECT re.*,
                               CASE WHEN re.from_uri = %s THEN re.to_uri
                                    ELSE re.from_uri END AS destination
                          FROM relation_edges re
                          JOIN context_nodes s ON s.uri=re.from_uri AND s.account_id=re.account_id
                          JOIN context_nodes t ON t.uri=re.to_uri AND t.account_id=re.account_id
                         WHERE re.account_id=%s AND (re.from_uri=%s OR re.to_uri=%s)
                           AND lower(re.relation_type) NOT IN ('supersedes','conflict_alert')
                           {projection_gate}
                    ), destinations AS (
                        SELECT destination, max(weight) AS weight FROM adjacent
                         GROUP BY destination ORDER BY weight DESC, destination LIMIT %s
                    )
                    SELECT a.from_uri,a.to_uri,a.relation_type,a.weight,a.edge_projection,
                           s.version,t.version
                      FROM adjacent a JOIN destinations d USING (destination)
                      JOIN context_nodes s ON s.uri=a.from_uri AND s.account_id=a.account_id
                      JOIN context_nodes t ON t.uri=a.to_uri AND t.account_id=a.account_id
                     ORDER BY d.weight DESC,d.destination,a.relation_type,a.from_uri,a.to_uri
                    """.format(projection_gate=projection_gate), params,
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        result = []
        for source, target, relation, weight, projection, source_rev, target_rev in rows:
            try:
                self._ensure_accessible(source, ctx, capability="read")
                self._ensure_accessible(target, ctx, capability="read")
            except AccessDeniedError:
                continue
            result.append(dict(source_uri=source, target_uri=target, relation_type=relation,
                weight=float(weight), edge_projection=projection if isinstance(projection, dict)
                else json.loads(projection) if projection else None,
                source_revision=int(source_rev), target_revision=int(target_rev)))
        return result

    def get_relation_neighbors(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        limit: int = 10,
        min_weight: float = 0.0,
    ) -> list[RelationNeighbor]:
        """Return canonical SQL relations with target and projection metadata."""
        self._ensure_accessible(uri, ctx, capability="read")
        from core.uri_parser import strip_l2_suffix

        source_uri = strip_l2_suffix(uri)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT target.uri, target.abstract, target.overview,
                           target.category, re.relation_type, re.weight,
                           re.reason, re.edge_projection,
                           source.version, target.version
                      FROM relation_edges re
                      JOIN context_nodes source ON source.uri = re.from_uri AND source.account_id=re.account_id
                      JOIN context_nodes target ON target.uri = re.to_uri AND target.account_id=re.account_id
                     WHERE re.account_id = %s AND re.from_uri = %s
                       AND re.weight >= %s
                     ORDER BY re.weight DESC, re.to_uri, re.relation_type
                     LIMIT %s
                    """,
                    (ctx.account_id, source_uri, min_weight, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        authorized_rows = []
        for row in rows:
            try:
                self._ensure_accessible(row[0],ctx,capability="read")
            except AccessDeniedError:
                continue
            authorized_rows.append(row)
        return [
            RelationNeighbor(
                uri=row[0],
                abstract=row[1] or "",
                overview=row[2] or "",
                category=row[3] or "",
                relation_type=row[4],
                weight=float(row[5]),
                reason=row[6] or "",
                edge_projection=(
                    row[7]
                    if isinstance(row[7], dict)
                    else json.loads(row[7]) if row[7] else None
                ),
                source_revision=int(row[8]),
                target_revision=int(row[9]),
            )
            for row in authorized_rows
        ]

    def _enqueue_edge_projection(self, cur, edge: RelationEdge, ctx: RequestContext) -> None:
        """Queue in the canonical edge transaction, using portable SQL.

        Caller locks all endpoints first. Lock/update the job before writing
        its relation row, matching the node-update trigger and projector.
        """
        from providers.relation_store.projection_identity import canonical_edge_id
        edge_id = canonical_edge_id(ctx.account_id, edge.from_uri, edge.to_uri, edge.relation_type)
        cur.execute("""SELECT s.version, t.version, concat_ws(E'\\n', t.abstract, t.overview, t.content)
                       FROM context_nodes s JOIN context_nodes t ON t.account_id=s.account_id
                       WHERE s.account_id=%s AND s.uri=%s AND t.uri=%s""",
                    (ctx.account_id, edge.from_uri, edge.to_uri))
        endpoints = cur.fetchone()
        if endpoints is None:
            raise ValueError("Cannot project an edge with missing endpoints")
        source_rev, target_rev, target_text = endpoints
        signature = (edge.reason, target_text, source_rev, target_rev, self._edge_projection_representation,
                     self._edge_projection_profile_id)
        cur.execute("""SELECT reason,target_text,source_revision,target_revision,
                              representation_version,embedding_profile
                       FROM edge_projection_jobs WHERE account_id=%s AND edge_id=%s FOR UPDATE""",
                    (ctx.account_id, edge_id))
        old = cur.fetchone()
        changed = old is None or tuple(old) != signature
        if old is None:
            cur.execute("""INSERT INTO edge_projection_jobs
                (account_id,edge_id,source_uri,target_uri,relation_type,reason,target_text,
                 source_revision,target_revision,representation_version,embedding_profile,
                 mutation_batch_key,next_retry_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    ogmem_graph_projection_batch_key(),
                    CASE WHEN ogmem_graph_projection_batch_key() IS NOT NULL
                         THEN NOW()+INTERVAL '30 seconds' ELSE NULL END)""",
                (ctx.account_id,edge_id,edge.from_uri,edge.to_uri,edge.relation_type,*signature))
        elif changed:
            cur.execute("""UPDATE edge_projection_jobs SET reason=%s,target_text=%s,
                source_revision=%s,target_revision=%s,representation_version=%s,embedding_profile=%s,
                status='PENDING',last_error=NULL,
                mutation_batch_key=ogmem_graph_projection_batch_key(),
                next_retry_at=CASE
                    WHEN status='PENDING' AND next_retry_at IS NOT NULL THEN next_retry_at
                    WHEN ogmem_graph_projection_batch_key() IS NULL THEN NULL
                    ELSE NOW()+INTERVAL '30 seconds'
                END,
                claim_token=NULL,updated_at=NOW()
                WHERE account_id=%s AND edge_id=%s""", (*signature,ctx.account_id,edge_id))
        self.sql_dialect.upsert_relation_edge(cur, (
            edge.from_uri,edge.to_uri,edge.relation_type,edge.weight,edge.reason,ctx.account_id))
        cur.execute("""UPDATE relation_edges SET projection_edge_id=%s,
            edge_projection=CASE WHEN %s THEN NULL ELSE edge_projection END
            WHERE account_id=%s AND from_uri=%s AND to_uri=%s AND relation_type=%s""",
            (edge_id,changed,ctx.account_id,edge.from_uri,edge.to_uri,edge.relation_type))
        if changed:
            from uuid import uuid4
            from psycopg2.extras import Json
            # Reuse the existing durable outbox/notification lifecycle. Insert
            # in this transaction: a committed edge can never lose its wakeup.
            event_id = str(uuid4())
            cur.execute("""INSERT INTO outbox_events
                (event_id,event_type,uri,account_id,payload,status,retry_count)
                VALUES (%s,'PROJECT_GRAPH',%s,%s,%s,'PENDING',0)""",
                (event_id,edge.from_uri,ctx.account_id,Json({"edge_id":edge_id})))
            self.sql_dialect.notify_outbox(cur,"ogmem_outbox",json.dumps(
                dict(event_id=event_id,account_id=ctx.account_id,event_type="PROJECT_GRAPH")))

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        if not edges:
            return

        for edge in edges:
            self._ensure_accessible(edge.from_uri, ctx)
            if self._edge_projection_profile_id:
                self._ensure_accessible(edge.to_uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                if self._edge_projection_profile_id:
                    # Lock every endpoint before any job/edge row. Consistent
                    # URI order also prevents reversed A->B/B->A deadlocks.
                    uris = sorted({uri for edge in edges for uri in (edge.from_uri,edge.to_uri)})
                    cur.execute("SELECT uri FROM context_nodes WHERE account_id=%s AND uri=ANY(%s) ORDER BY uri FOR UPDATE",
                                (ctx.account_id,uris))
                    if len(cur.fetchall()) != len(uris):
                        raise ValueError("Cannot project an edge with missing endpoints")
                for edge in sorted(edges, key=lambda e:(e.from_uri,e.to_uri,e.relation_type)):
                    params = (
                        edge.from_uri,
                        edge.to_uri,
                        edge.relation_type,
                        edge.weight,
                        edge.reason,
                        ctx.account_id,
                    )
                    if self._edge_projection_profile_id:
                        self._enqueue_edge_projection(cur, edge, ctx)
                    else:
                        self.sql_dialect.upsert_relation_edge(cur, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_one_hop(self, uri: str, ctx: RequestContext, limit: int = 3) -> list[RelationEdge]:
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_uri, to_uri, relation_type, weight, reason
                    FROM relation_edges
                    WHERE from_uri = %s AND account_id = %s
                    ORDER BY weight DESC
                    LIMIT %s
                    """,
                    (uri, ctx.account_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        # If relation_edges has data, use it (authoritative source)
        if rows:
            return [
                RelationEdge(
                    from_uri=r[0],
                    to_uri=r[1],
                    relation_type=r[2],
                    weight=float(r[3]),
                    reason=r[4],
                )
                for r in rows
            ]

        # Fallback: read from context_nodes.relations column
        # Get all relations and sort/limit in Python since context_nodes
        # stores relations as JSON array (not queryable by weight)
        edges = self._fallback_from_context_nodes(uri, ctx)
        edges.sort(key=lambda e: e.weight, reverse=True)
        return edges[:limit]
