"""PostgreSQL implementation of GraphStore protocol.

Stores heterogeneous graph nodes and edges in PostgreSQL with
incremental PageRank computation.

Usage::

    from providers.graph_store import SQLGraphStore
    from core.models import RequestContext

    store = SQLGraphStore("host=localhost dbname=ogmemory")
    ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                         session_id="s1", trace_id="t1")

    # Create nodes
    raw_node = create_raw_node(RawDataType.PROMPT, "Write a story", "acc1")
    store.create_node(raw_node.serialize(), ctx)
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any
from uuid import UUID

from core.vector_math import cosine_similarity
from core.errors import AccessDeniedError, NodeNotFoundError
from core.graph_enums import (
    ExtractionStatus,
    GraphEdgeType,
    GraphNodeType,
    NodeStatus,
    RawDataType,
)
from core.graph_models import (
    ExtractedNode,
    GraphEdge,
    GraphNode,
    GraphStats,
    MemoryNode,
    RawNode,
    SubgraphResult,
    SupportingChain,
    node_from_payload,
)
from core.interfaces import GraphStore
from core.models import RequestContext
from core.temporal import seconds, within_time
from fs.sql_adapter.pool import PoolAdapterMixin, SharedConnectionPool

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None
    Json = None

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length non-zero vectors.

    Returns 0.0 when either vector has zero norm or the lengths differ
    rather than raising — search stays robust against dirty data.
    Mirrors :meth:`MemoryGraphStore._cosine`; kept as a free function so
    the SQL store can rank without instantiating a graph store.
    """
    return cosine_similarity(a, b)


class SQLGraphStore(PoolAdapterMixin, GraphStore):
    """PostgreSQL-backed implementation of GraphStore.

    Stores nodes in `graph_nodes` table and edges in `graph_edges` table.
    Implements incremental PageRank updates on edge creation.

    Usage::

        store = SQLGraphStore("host=localhost dbname=ogmemory")
        store.create_node(raw_node.serialize(), ctx)
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool: SharedConnectionPool | None = None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLGraphStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection pool
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
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Ensure graph tables exist."""
        from pathlib import Path

        conn = self._get_connection()
        try:
            schema_path = Path(__file__).parent.parent.parent / "fs" / "sql_adapter" / "graph_schema.sql"
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    sql = f.read()
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning(f"Schema initialization warning: {exc}")
        finally:
            self._return_connection(conn)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_access(self, node_account_id: str, ctx: RequestContext) -> None:
        """Check if the request context has access to the node."""
        if node_account_id != ctx.account_id:
            raise AccessDeniedError(
                f"graph_node:{node_account_id}",
                ctx.account_id,
                f"Node belongs to account '{node_account_id}'",
            )

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Create a new graph node.

        Args:
            payload: JSON-ready node payload
            ctx: RequestContext for access control

        Returns:
            The created node with generated ID
        """
        node = node_from_payload(payload)
        self._check_access(node.account_id, ctx)
        payload = node.serialize()
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._insert_node(cur, payload)
            conn.commit()
            return node
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def create_node_with_edges(self, payload, edges, ctx):
        """Commit node, provenance edges and rank deltas in one transaction."""
        node = node_from_payload(payload)
        self._check_access(node.account_id, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._insert_node(cur, node.serialize())
                for edge in edges:
                    if UUID(node.node_id) not in (UUID(edge.source_id), UUID(edge.target_id)):
                        raise ValueError("Provenance edge must connect the new node")
                    self._insert_edge(cur, edge, ctx)
            conn.commit()
            return node
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _insert_node(self, cur, payload: dict[str, Any]) -> None:
        """Insert a node into the database."""

        cur.execute(
            """
            INSERT INTO graph_nodes
                (node_id, node_type, status, raw_type, content, summary,
                 properties, metadata, extracted_content,
                 source_raw_id, extractor_type, extraction_schema,
                 extraction_status, confidence,
                 supporting_node_ids, topic, tags, parent_memory_id,
                 pagerank, account_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s::text[], %s, %s, %s, %s, %s)
            """,
            (
                payload["node_id"],
                payload["node_type"],
                payload["status"],
                payload.get("raw_type"),
                payload.get("content"),
                payload.get("summary"),
                Json(payload.get("properties", {})),
                Json(payload.get("metadata", {})),
                Json(payload.get("extracted_content", {})),
                payload.get("source_raw_id"),
                payload.get("extractor_type"),
                payload.get("extraction_schema"),
                payload.get("extraction_status"),
                payload.get("confidence") if payload.get("node_type") == GraphNodeType.EXTRACTED.value else None,
                payload.get("supporting_node_ids", []),
                payload.get("topic"),
                payload.get("tags", []),
                payload.get("parent_memory_id"),
                payload["pagerank"],
                payload["account_id"],
                payload["created_at"],
                payload["updated_at"],
            ),
        )

    def get_node(self, node_id: str, ctx: RequestContext) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            GraphNode if found and accessible, None otherwise
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE node_id = %s AND account_id = %s
                    """,
                    (node_id, ctx.account_id),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        return self._row_to_node(row)

    def update_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Update an existing node.

        Args:
            payload: JSON-ready node payload
            ctx: RequestContext for access control

        Returns:
            The updated node
        """
        node = node_from_payload(payload)
        self._check_access(node.account_id, ctx)
        payload = node.serialize()
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE graph_nodes SET
                        status = %s,
                        content = %s,
                        summary = %s,
                        properties = %s,
                        metadata = %s,
                        extracted_content = %s,
                        extraction_status = %s,
                        confidence = %s,
                        supporting_node_ids = %s::uuid[],
                        topic = %s,
                        tags = %s::text[],
                        pagerank = %s,
                        updated_at = NOW()
                    WHERE node_id = %s AND account_id = %s
                    """,
                    (
                        payload["status"],
                        payload.get("content"),
                        payload.get("summary"),
                        Json(payload.get("properties", {})),
                        Json(payload.get("metadata", {})),
                        Json(payload.get("extracted_content", {})),
                        payload.get("extraction_status"),
                        payload.get("confidence") if payload.get("node_type") == GraphNodeType.EXTRACTED.value else None,
                        payload.get("supporting_node_ids", []),
                        payload.get("topic"),
                        payload.get("tags", []),
                        payload["pagerank"],
                        payload["node_id"],
                        ctx.account_id,
                    ),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(node.node_id)
            conn.commit()
            node.updated_at = self._get_current_timestamp()
            return node
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all its edges.

        Args:
            node_id: UUID of the node to delete
            ctx: RequestContext for access control
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_nodes WHERE node_id = %s AND account_id = %s",
                    (node_id, ctx.account_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def list_nodes(
        self,
        node_type: GraphNodeType | None,
        ctx: RequestContext,
        limit: int = 4096,
        offset: int = 0,
        raw_type: RawDataType | None = None,
        extractor_type: str | None = None,
        parent_audio_node_ids: list[str] | None = None,
    ) -> list[GraphNode]:
        """List nodes of a specific type, optionally filtered further by
        ``raw_type`` (RAW only), ``extractor_type`` (EXTRACTED only),
        or ``parent_audio_node_ids`` (EXTRACTED only — children of
        the named audio raw nodes).

        All three filters are pushed into the SQL WHERE clause so
        the LIMIT applies to the requested subset only. Without
        these push-downs, a generic ``node_type='extracted'`` query
        has to share its LIMIT budget across face / voice_asr /
        episodic_memory / etc., and the oldest extractor type
        silently disappears once the table grows past ``limit`` —
        this is the bug that dropped transcripts for the last 20
        of 23 videos in the M3-Bench 100-video run.

        ``limit=None`` translates to ``LIMIT ALL`` (Postgres /
        openGauss syntax extension), which drops the LIMIT clause
        entirely. Use this only after pre-narrowing via
        ``extractor_type`` or ``parent_audio_node_ids`` so the
        unfiltered scan stays bounded.
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Build a parameterized WHERE clause from the active
                # filters. We always carry the tenant scope; the
                # optional filters short-circuit to absent clauses
                # when unset so the SQL stays symmetric.
                where: list[str] = ["account_id = %s"]
                params: list[Any] = [ctx.account_id]
                if node_type is not None:
                    where.append("node_type = %s")
                    params.append(node_type.value)
                if raw_type is not None:
                    where.append("raw_type = %s")
                    params.append(raw_type.value)
                if extractor_type is not None:
                    where.append("extractor_type = %s")
                    params.append(extractor_type)
                if parent_audio_node_ids:
                    where.append(
                        "extracted_content->>'source_audio_node_id' = ANY(%s)"
                    )
                    params.append(list(parent_audio_node_ids))
                # ``limit=None`` means ``LIMIT ALL`` (Postgres /
                # openGauss syntax extension). Callers should set
                # this only after pre-narrowing via ``extractor_type``
                # or ``parent_audio_node_ids``.
                if limit is None:
                    limit_clause = "LIMIT ALL OFFSET %s"
                    params.append(offset)
                else:
                    limit_clause = "LIMIT %s OFFSET %s"
                    params.extend([limit, offset])
                cur.execute(
                    f"""
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE {' AND '.join(where)}
                    ORDER BY updated_at DESC
                    {limit_clause}
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def create_edge(self, edge: GraphEdge, ctx: RequestContext) -> GraphEdge:
        """Create a new edge between nodes.

        Automatically triggers incremental PageRank update.

        Args:
            edge: GraphEdge to create
            ctx: RequestContext for access control

        Returns:
            The created edge
        """
        self._check_access(edge.account_id, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._insert_edge(cur, edge, ctx)

            conn.commit()
            return edge
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _insert_edge(self, cur, edge, ctx):
        self._check_access(edge.account_id, ctx)
        # Lock and validate both endpoints in this transaction. A caller
        # cannot manufacture an own-account edge to a foreign node.
        endpoints = {str(UUID(edge.source_id)), str(UUID(edge.target_id))}
        cur.execute(
            """SELECT node_id FROM graph_nodes WHERE account_id = %s
               AND node_id = ANY(%s::uuid[]) ORDER BY node_id FOR UPDATE""",
            (ctx.account_id, list(endpoints)),
        )
        if {str(row[0]) for row in cur.fetchall()} != endpoints:
            raise NodeNotFoundError("edge endpoint")
        # Endpoint locks serialize updates to the same edge/target. Read the
        # previous weight so retries and weight changes only apply their delta.
        cur.execute(
            """SELECT weight FROM graph_edges WHERE source_id = %s
               AND target_id = %s AND edge_type = %s AND account_id = %s""",
            (edge.source_id, edge.target_id, edge.edge_type.value, ctx.account_id),
        )
        previous = cur.fetchone()
        old_weight = previous[0] if previous else 0.0
        # Insert edge
        cur.execute(
            """
            INSERT INTO graph_edges
                (edge_id, source_id, target_id, edge_type, weight, properties, account_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                weight = EXCLUDED.weight,
                properties = EXCLUDED.properties
            WHERE graph_edges.account_id = %s
            """,
            (
                edge.edge_id,
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.weight,
                Json(edge.properties),
                ctx.account_id,
                ctx.account_id,
            ),
        )

        # Incremental PageRank update
        self._update_pagerank_incremental(cur, edge.target_id, edge.weight - old_weight, ctx)

    def list_identity_pairs(
        self, source_ids: list[str], ctx: RequestContext,
    ) -> list[tuple[str, str]]:
        if not source_ids:
            return []
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT source_id, target_id FROM graph_edges
                       WHERE account_id = %s AND edge_type = %s
                       AND source_id = ANY(%s::uuid[])""",
                    (ctx.account_id, GraphEdgeType.SAME_IDENTITY.value, source_ids),
                )
                return [(str(s), str(t)) for s, t in cur.fetchall()]
        finally:
            conn.rollback()
            self._return_connection(conn)

    def ensure_identity_edges(self, edges: list[GraphEdge], ctx: RequestContext) -> int:
        """Batch insertion, with PageRank updates once per changed anchor.

        Existing pairs are true no-ops (including PageRank), unlike create_edge's
        upsert. Endpoint validation is explicitly tenant-scoped even for DB owners.
        """
        from psycopg2.extras import execute_values

        if not edges:
            return 0
        if any(e.account_id != ctx.account_id or e.edge_type != GraphEdgeType.SAME_IDENTITY
               for e in edges):
            raise ValueError("identity edge batch has wrong account or type")
        endpoints = {str(UUID(n)) for e in edges for n in (e.source_id, e.target_id)}
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT node_id FROM graph_nodes
                       WHERE account_id = %s AND node_id = ANY(%s::uuid[]) FOR KEY SHARE""",
                    (ctx.account_id, list(endpoints)),
                )
                if {str(row[0]) for row in cur.fetchall()} != endpoints:
                    raise ValueError("identity endpoint missing or outside account")
                inserted = execute_values(
                    cur,
                    """INSERT INTO graph_edges
                       (edge_id, source_id, target_id, edge_type, weight, properties, account_id)
                       VALUES %s ON CONFLICT (source_id, target_id, edge_type) DO NOTHING
                       RETURNING target_id""",
                    [(e.edge_id, e.source_id, e.target_id, e.edge_type.value,
                      e.weight, Json(e.properties), ctx.account_id) for e in edges],
                    page_size=500, fetch=True,
                )
                targets = sorted({str(row[0]) for row in inserted})
                if targets:
                    # Preserve the local incoming/outgoing contribution heuristic,
                    # but aggregate once per changed anchor rather than rescanning
                    # its growing edge set after every inserted member.
                    cur.execute(
                        """WITH incoming AS (
                            SELECT source_id, SUM(weight) AS weight
                            FROM graph_edges WHERE account_id = %s
                              AND target_id = ANY(%s::uuid[])
                            GROUP BY source_id
                        ), outgoing AS (
                            SELECT e.source_id, COUNT(*) AS n
                            FROM graph_edges e JOIN incoming i USING (source_id)
                            WHERE e.account_id = %s GROUP BY e.source_id
                        )
                        UPDATE graph_nodes n SET pagerank =
                            n.pagerank + i.weight / GREATEST(o.n, 1)
                        FROM incoming i JOIN outgoing o USING (source_id)
                        WHERE n.node_id = i.source_id AND n.account_id = %s""",
                        (ctx.account_id, targets, ctx.account_id, ctx.account_id),
                    )
                    cur.execute(
                        """UPDATE graph_nodes n SET pagerank = incoming.weight
                        FROM (
                            SELECT target_id, SUM(weight) AS weight
                            FROM graph_edges WHERE account_id = %s
                              AND target_id = ANY(%s::uuid[])
                            GROUP BY target_id
                        ) incoming
                        WHERE n.node_id = incoming.target_id AND n.account_id = %s""",
                        (ctx.account_id, targets, ctx.account_id),
                    )
            conn.commit()
            return len(inserted)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _update_pagerank_incremental(
        self, cur, target_node_id: str, incoming_weight: float, ctx: RequestContext
    ) -> None:
        """Apply the changed edge's incoming-weight delta in constant SQL work.

        The local heuristic need only update the target, not repeatedly credit
        all of its existing sources. Full PageRank remains compute_pagerank's job.
        """
        cur.execute(
            """UPDATE graph_nodes SET pagerank = pagerank + %s
               WHERE node_id = %s AND account_id = %s""",
            (incoming_weight, target_node_id, ctx.account_id),
        )

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge by ID."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_edges WHERE edge_id = %s AND account_id = %s",
                    (edge_id, ctx.account_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_edge(self, edge_id: str, ctx: RequestContext) -> GraphEdge | None:
        """Get an edge by ID."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, source_id, target_id, edge_type, weight, properties, account_id, created_at
                    FROM graph_edges
                    WHERE edge_id = %s AND account_id = %s
                    """,
                    (edge_id, ctx.account_id),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        return GraphEdge(
            edge_id=row[0],
            source_id=row[1],
            target_id=row[2],
            edge_type=GraphEdgeType(row[3]),
            weight=row[4],
            properties=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
            account_id=row[6],
            created_at=str(row[7]) if row[7] else "",
        )

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                results = []

                if direction in ("outgoing", "both"):
                    cur.execute(
                        """
                        SELECT n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at,
                               e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.target_id
                        JOIN graph_nodes origin ON origin.node_id = e.source_id
                        WHERE e.source_id = %s
                        AND e.account_id = %s AND n.account_id = %s AND origin.account_id = %s
                        AND (%s IS NULL OR e.edge_type = ANY(%s))
                        """,
                        (node_id, ctx.account_id, ctx.account_id, ctx.account_id,
                         [t.value for t in edge_types] if edge_types is not None else None,
                         [t.value for t in edge_types] if edge_types is not None else None),
                    )
                    for row in cur.fetchall():
                        neighbor = self._row_to_node(row[:22])
                        edge = self._row_to_edge(row[22:])
                        edge.source_id = node_id
                        edge.target_id = neighbor.node_id
                        results.append((neighbor, edge, "outgoing"))

                if direction in ("incoming", "both"):
                    cur.execute(
                        """
                        SELECT n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at,
                               e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.source_id
                        JOIN graph_nodes origin ON origin.node_id = e.target_id
                        WHERE e.target_id = %s
                        AND e.account_id = %s AND n.account_id = %s AND origin.account_id = %s
                        AND (%s IS NULL OR e.edge_type = ANY(%s))
                        """,
                        (node_id, ctx.account_id, ctx.account_id, ctx.account_id,
                         [t.value for t in edge_types] if edge_types is not None else None,
                         [t.value for t in edge_types] if edge_types is not None else None),
                    )
                    for row in cur.fetchall():
                        neighbor = self._row_to_node(row[:22])
                        edge = self._row_to_edge(row[22:])
                        edge.source_id = neighbor.node_id
                        edge.target_id = node_id
                        results.append((neighbor, edge, "incoming"))

        finally:
            conn.rollback()
            self._return_connection(conn)

        return results

    def traverse(
        self, start_node_id: str, depth: int, ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        from .traversal import traverse
        return traverse(self, start_node_id, depth, ctx, edge_types)

    def get_supporting_chain(self, memory_id: str, ctx: RequestContext) -> SupportingChain:
        from .traversal import get_supporting_chain
        return get_supporting_chain(self, memory_id, ctx)

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores for nodes."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                node_filter = "AND n.node_id = ANY(%s::uuid[])" if node_ids else ""
                params = [ctx.account_id, ctx.account_id, ctx.account_id]
                if node_ids:
                    params.append(node_ids)
                params.extend([damping, damping, ctx.account_id])
                cur.execute(
                    f"""
                    WITH node_scores AS (
                        SELECT n.node_id,
                               COALESCE(SUM(e.weight / NULLIF(outgoing.cnt, 0)), 0) AS pr_score
                        FROM graph_nodes n
                        LEFT JOIN graph_edges e
                          ON e.target_id = n.node_id AND e.account_id = %s
                        LEFT JOIN (
                            SELECT source_id, COUNT(*) AS cnt FROM graph_edges
                            WHERE account_id = %s GROUP BY source_id
                        ) outgoing ON outgoing.source_id = e.source_id
                        WHERE n.account_id = %s {node_filter}
                        GROUP BY n.node_id
                    )
                    UPDATE graph_nodes n
                    SET pagerank = (1 - %s) + %s * ns.pr_score
                    FROM node_scores ns
                    WHERE n.node_id = ns.node_id AND n.account_id = %s
                    RETURNING n.node_id, n.pagerank
                    """,
                    tuple(params),
                )

                rows = cur.fetchall()
                conn.commit()
                return {row[0]: row[1] for row in rows}
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a specific node."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pagerank FROM graph_nodes WHERE node_id = %s AND account_id = %s",
                    (node_id, ctx.account_id),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return row[0] if row else 0.0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memories; tags requires all specified tags (empty means no filter)."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Build query dynamically
                conditions = ["node_type = 'memory'", "account_id = %s"]
                params = [ctx.account_id]

                if query:
                    conditions.append("(summary ILIKE %s OR content ILIKE %s)")
                    params.extend([f"%{query}%", f"%{query}%"])

                if min_confidence > 0:
                    conditions.append("pagerank >= %s")
                    params.append(min_confidence)

                if topic:
                    conditions.append("topic = %s")
                    params.append(topic)

                if tags:
                    conditions.append("tags @> %s::text[]")
                    params.append(list(tags))

                where_clause = " AND ".join(conditions)

                cur.execute(
                    f"""
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE {where_clause}
                    ORDER BY pagerank DESC
                    LIMIT %s
                    """,
                    params + [top_k],
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by type and optional text query."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                conditions = ["node_type = 'raw'", "account_id = %s"]
                params = [ctx.account_id]

                if raw_type:
                    conditions.append("raw_type = %s")
                    params.append(raw_type.value)

                if query:
                    conditions.append("content ILIKE %s")
                    params.append(f"%{query}%")

                where_clause = " AND ".join(conditions)

                cur.execute(
                    f"""
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params + [top_k],
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    def search_by_embedding(
        self,
        query_vector: list[float],
        ctx: RequestContext,
        *,
        node_types: tuple[str, ...] = ("extracted",),
        extractor_types: tuple[str, ...] | None = None,
        video_id: str | None = None,
        before_clip: int | None = None,
        before_sec: float | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search over node embeddings.

        Implementation: filter candidates with JSONB predicates against
        the ``extracted_content`` column, then rank in Python by cosine
        similarity. For M3-Bench scale (~3K episodic nodes) this fits
        comfortably in memory; a pgvector upgrade is v2.

        Behaviour matches :meth:`MemoryGraphStore.search_by_embedding`:
        dimension-mismatched embeddings, missing keys, and malformed
        clip metadata are silently skipped so a single bad row cannot
        poison the result set.

        See :meth:`GraphStore.search_by_embedding` (Protocol) for parameter
        semantics.
        """
        if before_sec is not None:
            seconds(before_sec)
        if top_k <= 0 or not query_vector:
            return []

        type_filters: list[str] = []
        for nt in node_types:
            try:
                type_filters.append(GraphNodeType(nt).value)
            except ValueError:
                continue
        if not type_filters:
            return []

        conditions: list[str] = [
            "account_id = %s",
            "node_type = ANY(%s)",
            "(extracted_content ? 'embedding')",
        ]
        params: list[Any] = [ctx.account_id, type_filters]

        if extractor_types is not None:
            conditions.append("extractor_type = ANY(%s)")
            params.append(list(extractor_types))
        if video_id is not None:
            # EpisodicMemoryGenerator writes source_video_id / source_clip_id;
            # the legacy/_video_id schema is kept as a fallback (older
            # extractions, manual INSERTs) so the OR doesn't regress.
            conditions.append(
                "(extracted_content->>'_video_id' = %s "
                "OR extracted_content->>'source_video_id' = %s)"
            )
            params.extend([video_id, video_id])
        if before_clip is not None:
            conditions.append(
                "((CASE WHEN extracted_content->>'_clip_id' ~ '^-?[0-9]{1,18}$' "
                "THEN (extracted_content->>'_clip_id')::bigint END) <= %s "
                "OR (CASE WHEN extracted_content->>'source_clip_id' ~ '^-?[0-9]{1,18}$' "
                "THEN (extracted_content->>'source_clip_id')::bigint END) <= %s)"
            )
            params.extend([before_clip, before_clip])

        where_clause = " AND ".join(conditions)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT node_id, extracted_content
                    FROM graph_nodes
                    WHERE {where_clause}
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        dim = len(query_vector)
        ranked: list[tuple[str, float]] = []
        for row in rows:
            node_id, ext_content = row
            if isinstance(ext_content, str):
                # Driver compatibility: some psycopg2 builds return the JSONB
                # column as a raw string when there are no JSON codecs wired.
                try:
                    ext_content = json.loads(ext_content)
                except (TypeError, ValueError):
                    continue
            if not isinstance(ext_content, dict):
                continue
            if not within_time(ext_content, before_sec):
                continue
            embedding = ext_content.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != dim:
                continue
            score = _cosine(query_vector, embedding)
            ranked.append((str(node_id), score))

        # Stable, deterministic tie-break: on identical scores the
        # control loop must see the same node_id order from both the
        # SQL and Memory backends. Sort by (-score, node_id) so the
        # node_id is a strict ascending tiebreaker (matches memory).
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Count nodes by type
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE node_type = 'raw') as raw_count,
                        COUNT(*) FILTER (WHERE node_type = 'extracted') as extracted_count,
                        COUNT(*) FILTER (WHERE node_type = 'memory') as memory_count,
                        COALESCE(AVG(pagerank), 0) as avg_pagerank
                    FROM graph_nodes WHERE account_id = %s
                    """, (ctx.account_id,),
                )
                total, raw_count, extracted_count, memory_count, avg_pr = cur.fetchone()

                # Count edges
                cur.execute("SELECT COUNT(*) FROM graph_edges WHERE account_id = %s", (ctx.account_id,))
                total_edges = cur.fetchone()[0]

                # Top memories by pagerank
                cur.execute(
                    """
                    SELECT node_id, summary, pagerank
                    FROM graph_nodes
                    WHERE node_type = 'memory' AND account_id = %s
                    ORDER BY pagerank DESC
                    LIMIT 5
                    """, (ctx.account_id,),
                )
                top_memories = [
                    {"node_id": row[0], "summary": row[1], "pagerank": row[2]}
                    for row in cur.fetchall()
                ]

        finally:
            conn.rollback()
            self._return_connection(conn)

        return GraphStats(
            total_nodes=total,
            raw_nodes=raw_count,
            extracted_nodes=extracted_count,
            memory_nodes=memory_count,
            total_edges=total_edges,
            avg_pagerank=avg_pr,
            top_memories=top_memories,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode."""
        (
            node_id, node_type, status, raw_type, content, summary,
            properties, metadata, extracted_content,
            source_raw_id, extractor_type, extraction_schema,
            extraction_status, confidence,
            supporting_node_ids, topic, tags, parent_memory_id,
            pagerank, account_id, created_at, updated_at,
        ) = row

        properties = properties if isinstance(properties, dict) else json.loads(properties)
        metadata = metadata if isinstance(metadata, dict) else json.loads(metadata)

        node_type_enum = GraphNodeType(node_type)
        status_enum = NodeStatus(status) if status else NodeStatus.ACTIVE

        if node_type_enum == GraphNodeType.RAW:
            return RawNode(
                node_id=node_id,
                node_type=GraphNodeType.RAW,
                status=status_enum,
                raw_type=RawDataType(raw_type) if raw_type else RawDataType.PROMPT,
                content=content or "",
                metadata=metadata,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )
        elif node_type_enum == GraphNodeType.EXTRACTED:
            return ExtractedNode(
                node_id=node_id,
                node_type=GraphNodeType.EXTRACTED,
                status=status_enum,
                source_raw_id=source_raw_id or "",
                extractor_type=extractor_type or "",
                extraction_schema=extraction_schema or "",
                extracted_content=extracted_content if isinstance(extracted_content, dict) else json.loads(extracted_content),
                extraction_status=ExtractionStatus(extraction_status) if extraction_status else ExtractionStatus.PENDING,
                confidence=confidence or 0.0,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )
        else:  # MEMORY
            # psycopg2 returns uuid[] as PostgreSQL text unless a UUID array
            # caster is registered. Keep the GraphStore contract list[str]
            # without changing global connection typecasters.
            if isinstance(supporting_node_ids, str):
                values = supporting_node_ids.removeprefix("{").removesuffix("}")
                supporting_node_ids = [
                    str(UUID(value.strip('"'))) for value in values.split(",")
                    if value
                ]
            else:
                supporting_node_ids = [str(value) for value in supporting_node_ids or []]
            return MemoryNode(
                node_id=node_id,
                node_type=GraphNodeType.MEMORY,
                status=status_enum,
                summary=summary or "",
                content=content or "",
                supporting_node_ids=supporting_node_ids or [],
                topic=topic or "",
                tags=tags or [],
                parent_memory_id=parent_memory_id,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )

    def _row_to_edge(self, row: tuple) -> GraphEdge:
        """Convert a database row to a GraphEdge."""
        edge_id, edge_type, weight, properties, account_id, created_at = row
        return GraphEdge(
            edge_id=edge_id,
            source_id="",  # Will be set by caller if needed
            target_id="",  # Will be set by caller if needed
            edge_type=GraphEdgeType(edge_type),
            weight=weight,
            properties=properties if isinstance(properties, dict) else json.loads(properties),
            account_id=account_id,
            created_at=str(created_at) if created_at else "",
        )

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO8601 string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
