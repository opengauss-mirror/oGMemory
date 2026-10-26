"""PostgreSQL-backed RelationStore implementation.

Stores relation edges in the ``relation_edges`` table instead of
per-node .relations.json files.
"""

from __future__ import annotations

import json
import logging

from core.errors import AccessDeniedError
from core.models import RelationEdge, RequestContext
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
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLRelationStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size, dialect)
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
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to ensure schema: {exc}") from exc
        finally:
            self._return_connection(conn)

    def _extract_account_id(self, uri: str) -> str:
        from fs.sql_adapter.sql_context_fs import parse_uri

        return parse_uri(uri)["account"]

    def _ensure_accessible(self, uri: str, ctx: RequestContext) -> None:
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

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        if not edges:
            return

        for edge in edges:
            self._ensure_accessible(edge.from_uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                for edge in edges:
                    params = (
                        edge.from_uri,
                        edge.to_uri,
                        edge.relation_type,
                        edge.weight,
                        edge.reason,
                        ctx.account_id,
                    )
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
