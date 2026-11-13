"""OpenGauss implementation of VectorIndex.

Supports efficient similarity search with HNSW indexing.
"""
import json
import logging
import re
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit, TypedQuery
from core.projection_ownership import current_projection_owner
from fs.sql_adapter.dialect import normalize_sql_dialect
from fs.sql_adapter.pool import PoolAdapterMixin, is_connection_alive, resolve_connection_dialect

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    OPENGAUSS_AVAILABLE = True
except ImportError:
    OPENGAUSS_AVAILABLE = False
    psycopg2 = None
    Json = None
    RealDictCursor = None

logger = logging.getLogger(__name__)


# Valid PostgreSQL identifier: letters, digits, underscore, must start with letter/underscore
_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_table_name(name: str) -> str:
    """Validate table name to prevent SQL injection.

    Args:
        name: Table name to validate

    Returns:
        The validated table name

    Raises:
        ValueError: If table name contains invalid characters
    """
    if not _TABLE_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid table name '{name}'. "
            "Table names must be valid PostgreSQL identifiers: "
            "start with letter or underscore, contain only letters, digits, and underscores."
        )
    # Also protect against SQL keywords
    upper_name = name.upper()
    sql_keywords = {
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'UNION', 'OR', 'AND', 'WHERE', 'FROM'
    }
    if upper_name in sql_keywords:
        raise ValueError(f"Table name '{name}' is a reserved SQL keyword")
    return name


def _vec_literal(vec: list[float]) -> str:
    """Convert vector to pgvector literal string."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _ensure_dict(val: Any) -> dict:
    """Ensure value is a dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


class OpenGaussVectorIndex(PoolAdapterMixin, VectorIndex):
    """Vector index for openGauss and PostgreSQL with pgvector.

    Supports HNSW cosine search, shared connection pools, tenant isolation,
    idempotent MERGE upserts and explicit legacy-row migration.
    """

    def __init__(
        self,
        connection_string: str = "",
        dimension: int = 1536,
        table_name: str = "vector_index",
        pool_size: int = 5,
        pool=None,
        dialect=None,
    ):
        """Initialize OpenGauss index.

        Args:
            connection_string: PostgreSQL connection string
            dimension: Embedding vector dimension (default 1536)
            table_name: Table name for vector storage (must be valid identifier)
            pool_size: Connection pool size (when pool is not provided)
            pool: Optional shared connection pool with matching SQL dialect
            dialect: Explicit dialect or auto detection; omitted inherits the pool or openGauss

        Raises:
            ImportError: If psycopg2 is not installed
            ValueError: If table_name is invalid
        """
        if not OPENGAUSS_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for OpenGaussVectorIndex. "
                "Install with: pip install psycopg2-binary"
            )

        self._dimension = dimension
        self._table_name = _validate_table_name(table_name)
        if dialect is None or (pool is not None and normalize_sql_dialect(dialect) == "auto"):
            dialect = pool.sql_dialect if pool is not None else "opengauss"
        dialect = resolve_connection_dialect(dialect, connection_string)
        if pool is not None and pool.sql_dialect.name != dialect.name:
            raise ValueError("vector index dialect must match shared pool dialect")
        self._init_pool(pool, connection_string, pool_size, dialect=dialect)

        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the table, security columns and search indexes.

        Legacy data backfills use scripts/migrate_vector_index.py.
        """
        tbl = self._table_name
        dim = self._dimension
        conn = self._get_connection()
        try:
            if self.sql_dialect.name == "postgresql":
                try:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    logger.debug("CREATE EXTENSION vector unavailable: %s", exc)
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        id            VARCHAR(16) PRIMARY KEY,
                        uri           VARCHAR(512) NOT NULL,
                        level         INTEGER NOT NULL,
                        text          TEXT NOT NULL,
                        embedding     vector({dim}) NOT NULL,
                        filters       JSONB NOT NULL,
                        metadata      JSONB NOT NULL DEFAULT '{{}}',
                        account_id    VARCHAR(128),
                        owner_space   VARCHAR(128),
                        node_uri      VARCHAR(512),
                        record_kind   VARCHAR(32) DEFAULT 'node_content',
                        node_status   VARCHAR(16) DEFAULT 'active',
                        space_id      UUID,
                        created_at    TIMESTAMPTZ DEFAULT NOW(),
                        updated_at    TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Upgrade path: add columns to pre-existing tables
                for col, col_type in [
                    ("account_id",  "VARCHAR(128)"),
                    ("owner_space", "VARCHAR(128)"),
                    ("node_uri",    "VARCHAR(512)"),
                    ("node_id",     "UUID"),
                    ("record_kind", "VARCHAR(32) DEFAULT 'node_content'"),
                    ("node_status", "VARCHAR(16) DEFAULT 'active'"),
                    ("space_id",    "UUID"),
                ]:
                    cur.execute(
                        f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    )

                # Legacy row backfills are explicitly run once by
                # scripts/migrate_vector_index.py, never during process startup.

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_account
                        ON {tbl} ((filters->>'account_id'))
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_level
                        ON {tbl} (level)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_filters_gin
                        ON {tbl} USING GIN (filters)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_embedding_hnsw
                        ON {tbl} USING hnsw (embedding vector_cosine_ops)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_space_id
                        ON {tbl} (account_id, space_id)
                        WHERE space_id IS NOT NULL
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_node_id
                        ON {tbl} (node_id)
                        WHERE node_id IS NOT NULL
                """)
                conn.commit()
            logger.info("Table '%s' ensured (dim=%d)", tbl, dim)
        except Exception as e:
            conn.rollback()
            raise RuntimeError(
                f"OpenGaussVectorIndex initialization failed for table "
                f"'{tbl}' (dim={dim}): {e}"
            ) from e
        finally:
            self._return_connection(conn)

    def migrate_legacy_rows(self) -> bool:
        """Run the legacy backfill once, atomically with its completion marker.

        Operators run this during a maintenance window, not at application boot.
        The ledger lock serializes concurrent migration commands; a failed
        migration rolls back both the data changes and the completion marker.
        """
        tbl = self._table_name
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS ogmem_vector_migrations (
                    table_name TEXT NOT NULL, migration TEXT NOT NULL,
                    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (table_name, migration))""")
                cur.execute("LOCK TABLE ogmem_vector_migrations IN EXCLUSIVE MODE")
                key = (tbl, "legacy_columns_v2")
                cur.execute("""SELECT 1 FROM ogmem_vector_migrations
                    WHERE table_name = %s AND migration = %s""", key)
                if cur.fetchone() is not None:
                    conn.commit()
                    return False
                cur.execute(f"""
                    UPDATE {tbl} SET
                        account_id = COALESCE(account_id, filters->>'account_id'),
                        owner_space = COALESCE(owner_space, filters->>'owner_space'),
                        node_uri = COALESCE(node_uri, uri),
                        node_status = COALESCE(node_status, 'active')
                    WHERE account_id IS NULL OR owner_space IS NULL
                       OR node_uri IS NULL OR node_status IS NULL
                """)
                cur.execute("""SELECT 1 FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = 'context_nodes'
                      AND column_name = 'node_id'""")
                if cur.fetchone() is not None:
                    cur.execute(f"""
                        SELECT DISTINCT account_id FROM {tbl}
                        WHERE node_id IS NULL AND account_id IS NOT NULL AND node_uri IS NOT NULL
                    """)
                    # Legacy context nodes are visible only within their tenant's RLS scope.
                    for (account_id,) in cur.fetchall():
                        self.bind_tenant(conn, account_id)
                        cur.execute(f"""
                            UPDATE {tbl} vi SET node_id = cn.node_id
                            FROM context_nodes cn
                            WHERE vi.node_uri = cn.uri AND vi.node_id IS NULL
                              AND vi.account_id = %s AND vi.account_id = cn.account_id
                        """, (account_id,))
                        cur.execute(f"""
                            UPDATE {tbl} vi SET node_id = cn.node_id
                            FROM context_nodes cn
                            WHERE vi.node_id IS NULL
                              AND vi.node_uri LIKE '%%/content.md'
                              AND cn.uri = substring(vi.node_uri from '^(.+)/content[.]md')
                              AND vi.account_id = %s AND vi.account_id = cn.account_id
                        """, (account_id,))
                cur.execute("""INSERT INTO ogmem_vector_migrations (table_name, migration)
                    VALUES (%s, %s)""", key)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        while self._pool:
            conn = self._pool.pop()
            if is_connection_alive(conn):
                return conn

        return psycopg2.connect(self._connection_string, connect_timeout=10)

    @staticmethod
    def _reset_connection(conn):
        """End the index transaction before the connection becomes idle."""
        if conn.closed != 0:
            return
        try:
            if conn.info.transaction_status != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                conn.rollback()
        except Exception:
            conn.close()

    def _return_shared_connection(self, conn):
        try:
            self._reset_connection(conn)
        finally:
            self._shared_pool.return_connection(conn)

    def _return_connection(self, conn):
        """Return a connection to the pool, rolling back dirty transactions."""
        self._reset_connection(conn)
        if conn.closed != 0:
            return
        if len(self._pool) < self._pool_size:
            self._pool.append(conn)
        else:
            conn.close()

    def _lock_projection(self, cur, ids):
        check = current_projection_owner()
        if check is not None:
            for record_id in sorted(set(ids)):
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                            (self._table_name, record_id))
            check()

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Uses MERGE INTO for openGauss compatibility (ON CONFLICT not supported).
        Per-record errors are logged and skipped to avoid poisoning the connection.
        """
        if not records:
            return

        conn = self._get_connection()
        try:
            for record in records:
                embedding = record.metadata.get("_embedding")
                if embedding is None:
                    logger.warning("Skipping record %s: missing '_embedding'", record.id)
                    continue

                embedding_str = _vec_literal(embedding)
                filters_json = Json(record.filters)
                metadata_json = Json(record.metadata)

                # Extract explicit security columns from filters
                acct = record.filters.get("account_id")
                owner = record.filters.get("owner_space")
                kind = record.metadata.get("record_kind", "node_content")
                space_id = record.filters.get("space_id")
                node_id = record.metadata.get("node_id")

                try:
                    with conn.cursor() as cur:
                        self._lock_projection(cur, [record.id])
                        # IDs should already be account-scoped. Fail closed on
                        # a legacy/global ID collision: it must hit the unique
                        # constraint, never overwrite another tenant's row.
                        cur.execute(f"""
                            MERGE INTO {self._table_name} t
                            USING (SELECT %s AS id, %s AS account_id) s
                            ON t.id = s.id AND (t.account_id = s.account_id
                                OR (t.account_id IS NULL AND s.account_id IS NULL))
                            WHEN MATCHED THEN UPDATE SET
                                uri = %s,
                                level = %s,
                                text = %s,
                                embedding = %s::vector,
                                filters = %s,
                                metadata = %s,
                                owner_space = %s,
                                node_uri = %s,
                                node_id = COALESCE(%s, t.node_id),
                                record_kind = %s,
                                node_status = 'active',
                                space_id = %s,
                                updated_at = NOW()
                            WHEN NOT MATCHED THEN INSERT
                                (id, uri, level, text, embedding, filters, metadata,
                                 account_id, owner_space, node_uri, node_id, record_kind,
                                 node_status, space_id)
                            VALUES (%s, %s, %s, %s, %s::vector, %s, %s,
                                    %s, %s, %s, %s, %s, 'active', %s)
                        """, (
                            record.id, acct,
                            record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                            owner, record.uri, node_id, kind, space_id,
                            record.id, record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                            acct, owner, record.uri, node_id, kind, space_id,
                        ))
                    conn.commit()
                except Exception as exc:
                    logger.warning(
                        "Upsert failed for record %s (uri=%s), rolling back: %s",
                        record.id, record.uri, exc,
                    )
                    conn.rollback()
                    if current_projection_owner() is not None:
                        raise
        finally:
            self._return_connection(conn)


    def delete(self, ids: list[str]) -> None:
        """Remove records from the index."""
        if not ids:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                self._lock_projection(cur, ids)
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                conn.commit()
        finally:
            self._return_connection(conn)

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Low-level vector search returning SeedHit."""
        where, params = self._build_where(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k

        sql = f"""
            SELECT id, uri, level, text, filters, metadata, node_id, owner_space,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table_name}
            WHERE  {where}
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_hit(r) for r in rows]


    def get_embeddings(
        self,
        uris: list[str],
        filters: dict[str, Any],
        level: int = 0,
        *, expected_texts: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        if not uris:
            return {}
        where, params = self._build_where({**filters, "level": level})
        params["uris"] = uris
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT uri, embedding::text{', text' if expected_texts is not None else ''} FROM {self._table_name} "
                    f"WHERE {where} AND uri=ANY(%(uris)s)",
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)
        output: dict[str, list[float]] = {}
        for uri, raw, *source in rows:
            if expected_texts is not None and (not source or source[0] != expected_texts.get(str(uri))):
                continue
            text = str(raw).strip().strip("[]")
            output[str(uri)] = [float(value) for value in text.split(",") if value]
        return output

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Search immediate children of parent_uri."""
        where, params = self._build_where(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k
        params["parent_uri"] = parent_uri

        sql = f"""
            SELECT id, uri, level, text, filters, metadata, node_id, owner_space,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table_name}
            WHERE  {where} AND metadata->>'parent_uri' = %(parent_uri)s
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_hit(r) for r in rows]

    @staticmethod
    def _build_where(f: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build WHERE clause with legacy tenant-filter compatibility.

        Older index writers populate JSON filters but not the denormalized
        tenant columns. Fall back only when a column is NULL: an explicit
        owner always takes precedence. Missing ownership remains excluded,
        and space grants still require their explicit columns.

        Pure function of ``f`` — reads no instance state, so it can be
        exercised directly (``OpenGaussVectorIndex._build_where(...)``)
        in unit tests without standing up a live PG connection. Mirrors
        the ``@staticmethod`` treatment of :meth:`_row_to_hit`.
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if "account_id" in f:
            clauses.append("COALESCE(account_id, filters->>'account_id') = %(f_account_id)s")
            params["f_account_id"] = f["account_id"]
        if "owner_space" in f:
            owner = f["owner_space"]
            if isinstance(owner, list):
                clauses.append("COALESCE(owner_space, filters->>'owner_space') = ANY(%(f_owner_space)s)")
            else:
                clauses.append("COALESCE(owner_space, filters->>'owner_space') = %(f_owner_space)s")
            params["f_owner_space"] = owner
        if "context_type" in f:
            ct = f["context_type"]
            if isinstance(ct, list):
                clauses.append("metadata->>'context_type' = ANY(%(f_ctx_type)s)")
            else:
                clauses.append("metadata @> %(f_ctx_type)s::jsonb")
                ct = json.dumps({"context_type": ct})
            params["f_ctx_type"] = ct
        if "memory_directory" in f:
            clauses.append("""CASE WHEN split_part(uri, '/', 6) = 'memories'
                THEN split_part(uri, '/', 7) ELSE split_part(uri, '/', 6) END
                = ANY(%(f_memory_directory)s)""")
            params["f_memory_directory"] = f["memory_directory"]
        if "category" in f:
            cat = f["category"]
            if isinstance(cat, list):
                clauses.append("metadata->>'category' = ANY(%(f_category)s)")
            else:
                clauses.append("metadata @> %(f_category)s::jsonb")
                cat = json.dumps({"category": cat})
            params["f_category"] = cat
        if "level" in f:
            lvl = f["level"]
            if isinstance(lvl, list):
                clauses.append("level = ANY(%(f_level)s)")
            else:
                clauses.append("level = %(f_level)s")
            params["f_level"] = lvl
        for key in ("record_kind", "representation_version"):
            if key in f:
                clauses.append(f"metadata->>'{key}' = %(f_{key})s")
                params[f"f_{key}"] = f[key]
        if "space_id" in f:
            sid = f["space_id"]
            if isinstance(sid, list):
                # Strict space isolation: legacy records with NULL space_id
                # are NOT auto-matched — they would silently bypass the
                # caller's space grant. Backfill them or exclude them.
                clauses.append("space_id = ANY(%(f_space_id)s)")
            else:
                clauses.append("space_id = %(f_space_id)s")
            params["f_space_id"] = sid
        return (" AND ".join(clauses) or "TRUE"), params

    @staticmethod
    def _row_to_hit(row: dict) -> SeedHit:
        """Convert database row to SeedHit."""
        meta = _ensure_dict(row.get("metadata"))
        filt = _ensure_dict(row.get("filters"))
        level = int(row.get("level", 2))
        # Prefer explicit column, fall back to JSONB for old rows
        owner = row.get("owner_space") or filt.get("owner_space", "")
        node_id = row.get("node_id")
        hit_metadata: dict[str, Any] = dict(meta)
        if node_id is not None:
            hit_metadata["node_id"] = str(node_id)
        return SeedHit(
            uri=row["uri"],
            score=float(row.get("score", 0)),
            level=level,
            parent_uri=meta.get("parent_uri"),
            context_type=meta.get("context_type", ""),
            category=meta.get("category", ""),
            owner_space=owner,
            abstract=meta.get("abstract", row.get("text", "")[:200]),
            has_overview=meta.get("has_overview", False),
            has_content=meta.get("has_content", False),
            active_count=meta.get("active_count", 0),
            updated_at=meta.get("updated_at"),
            metadata=hit_metadata,
        )

    def delete_account_data(self, account_id: str) -> int:
        """Delete all index records for an account.

        Args:
            account_id: Account ID to delete

        Returns:
            Count of deleted records
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE filters->>'account_id' = %(account_id)s",
                    {"account_id": account_id}
                )
                conn.commit()
                return cur.rowcount
        finally:
            self._return_connection(conn)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count index records, optionally constrained by filters."""
        where_sql, params = self._build_where(filters or {})
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table_name} WHERE {where_sql}", params)
                row = cur.fetchone()
                return int(row[0] if row else 0)
        finally:
            self._return_connection(conn)

    def delete_by_owner_space(
        self, account_id: str, owner_space: str
    ) -> int:
        """Delete all records matching account_id + owner_space.

        Args:
            account_id: Account ID to filter
            owner_space: Owner space in colon format "user:{id}" or "agent:{id}"

        Returns:
            Count of deleted records
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} "
                    "WHERE filters->>'account_id' = %(account_id)s "
                    "  AND filters->>'owner_space' = %(owner_space)s",
                    {"account_id": account_id, "owner_space": owner_space}
                )
                conn.commit()
                return cur.rowcount
        finally:
            self._return_connection(conn)

    def close(self) -> None:
        """Close owned connections, leaving a shared pool to its owner."""
        if getattr(self, "_shared_pool", None) is not None:
            return
        for conn in getattr(self, "_pool", []):
            if conn.closed == 0:
                conn.close()
        if hasattr(self, "_pool"):
            self._pool.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # Backward compatibility convenience method

    def search(self, query, embedder=None, top_k: int | None = None) -> list[SeedHit]:
        """Convenience method for backward compatibility with TypedQuery API."""
        if not isinstance(query, TypedQuery):
            raise TypeError(f"Expected TypedQuery, got {type(query)}")

        filters: dict[str, Any] = {}
        if query.account_id:
            filters["account_id"] = query.account_id
        if query.owner_space:
            filters["owner_space"] = query.owner_space
        if query.context_type:
            filters["context_type"] = query.context_type
        if query.categories:
            filters["category"] = query.categories

        if embedder:
            vectors = embedder.embed_texts([query.text])
            query_vector = vectors[0]
        else:
            import hashlib
            seed = int(hashlib.md5(query.text.encode()).hexdigest(), 16)
            query_vector = []
            for i in range(self._dimension):
                seed = (1103515245 * seed + 12345) & 0x7fffffff
                value = (seed / 0x7fffffff) * 2 - 1
                query_vector.append(value)

        return self.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=top_k or query.top_k,
        )

    def get_record(self, record_id: str) -> IndexRecord | None:
        """Fetch a complete stored index record by its identifier."""
        if not record_id:
            return None
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, uri, level, text, filters, metadata, embedding
                    FROM   {self._table_name}
                    WHERE  id = %(rid)s
                    """,
                    {"rid": record_id},
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)
        if row is None:
            return None
        return self._row_to_record(row)


    def get_records(self, record_ids: list[str]) -> list[IndexRecord | None]:
        """Read stored vectors in one round-trip, preserving input order."""
        if not record_ids:
            return []
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT id,uri,level,text,filters,metadata,embedding FROM {self._table_name} WHERE id=ANY(%s)",
                            (record_ids,))
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        records = {str(row["id"]):self._row_to_record(row) for row in rows}
        return [records.get(record_id) for record_id in record_ids]

    @staticmethod
    def _row_to_record(row):
        meta = _ensure_dict(row.get("metadata"))
        filt = _ensure_dict(row.get("filters"))
        embedding = row.get("embedding")
        # ``embedding`` comes back as a pgvector string like
        # ``[0.1,0.2,...]``; round-trip through the literal parser so
        # callers get a plain list[float]. The OpenGaussVectorIndex
        # stores it via ``_vec_literal`` which produces the same
        # format.
        emb_list: list[float] | None = None
        if isinstance(embedding, str):
            try:
                emb_list = [float(x) for x in embedding.strip("[]").split(",") if x.strip()]
            except ValueError:
                emb_list = None
        elif isinstance(embedding, (list, tuple)):
            emb_list = [float(x) for x in embedding]
        # Embedding lives in ``metadata["_embedding"]`` from the writer
        # — surface it there so callers see what they upserted. The
        # top-level ``embedding`` field on IndexRecord is reserved
        # for whatever the VectorIndex Protocol adds later.
        if emb_list is not None and "_embedding" not in meta:
            meta["_embedding"] = emb_list
        return IndexRecord(
            id=str(row["id"]),
            uri=str(row["uri"]),
            level=int(row["level"]),
            text=str(row["text"]),
            filters=filt,
            metadata=meta,
        )
