"""OpenGauss implementation of VectorIndex.

Supports efficient similarity search with HNSW indexing.
"""
import json
import logging
import re
from typing import Any

from core.clip_lookup import validate_clip_lookup
from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit, TypedQuery
from core.temporal import contains_time, seconds
from core.transcript_lexical import SEARCH_FIELD, indexed_metadata, validate_transcript_request
from fs.sql_adapter.pool import PoolAdapterMixin, resolve_connection_dialect
from fs.sql_adapter.dialect import normalize_sql_dialect

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
    """Vector-index implementation that targets openGauss (primary)
    and PostgreSQL+pgvector (single-DB convenience alias).

    The class is named ``OpenGaussVectorIndex`` because **openGauss
    is the recommended production target** — M3's benchmarks were
    validated against openGauss and deployments that already
    operate an openGauss cluster should keep using it without any
    code changes.

    As a convenience for deployments that prefer not to run a
    separate openGauss cluster, the same code path also runs
    against vanilla PostgreSQL 15+ with the pgvector extension.
    The ``m3.vector_index.backend`` config accepts both
    ``opengauss`` and ``pgvector`` (alias) and routes to this
    class either way; pick whichever backend you operate.

    The two backends diverge only on the extension bootstrap
    (in ``_ensure_table``):

    - **openGauss**: the configured dialect uses its built-in
      ``vector(N)`` type and HNSW ``vector_cosine_ops`` opclass.
    - **PostgreSQL 15+ with pgvector**: requires the
      ``pgvector/pgvector:pg16`` Docker image (or the
      ``pgvector`` package installed server-side). The CREATE
      EXTENSION call activates the extension; everything else is
      pgvector's standard SQL.

    ``psycopg2`` speaks the same wire protocol to both, so the rest
    of the class is backend-agnostic.

    Features:
        - HNSW indexing for fast approximate nearest neighbor search
        - Multi-tenant isolation via account_id filtering
        - Idempotent upsert with MERGE INTO (PG 15+ / openGauss)
        - Cosine similarity search
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
        """Create the vector_index table and indexes if they don't exist.

        Also adds missing columns. Legacy data backfills are an explicit
        one-time migration (scripts/migrate_vector_index.py).

        The ``video_id`` / ``extractor_type`` / ``before_clip`` columns
        are M3-specific scope columns: they let the index push the QA
        evaluation boundary (``ClipScope``) down to SQL instead of
        loading every candidate into Python and filtering post-hoc.
        Records written before this migration land with NULL on these
        columns; bounded searches exclude unknown provenance. Re-index
        legacy records before using clip or second-level boundaries.

        This class is named ``OpenGaussVectorIndex`` for historical
        reasons (the M3 pipeline was originally written against an
        openGauss deployment) but the same code runs against vanilla
        PostgreSQL 15+ with the pgvector extension. The ``CREATE
        EXTENSION vector`` statement below is the only piece that
        differs:

        - **PostgreSQL + pgvector**: required. The extension ships the
          ``vector(N)`` type and the HNSW ``vector_cosine_ops`` opclass
          referenced below. We try to create it idempotently and rely
          on the fact that ``CREATE EXTENSION IF NOT EXISTS`` is a
          no-op when the extension is already installed (e.g. via the
          ``pgvector/pgvector:pg16`` Docker image, where the bits are
          preloaded into ``shared_preload_libraries``).
        - **openGauss**: the configured dialect uses its built-in
          ``vector(N)`` type and HNSW opclass directly.

        ``psycopg2`` talks the same wire protocol to both servers, so
        nothing else in this class needs to special-case the backend.
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
                        video_id      VARCHAR(128),
                        extractor_type VARCHAR(64),
                        before_clip   INTEGER,
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
                    ("video_id",    "VARCHAR(128)"),
                    ("extractor_type", "VARCHAR(64)"),
                    ("before_clip", "INTEGER"),
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
                # M3 scope columns. Composite indexes keyed on account_id
                # first so the planner can use them in tandem with the
                # tenant filter without an extra sort.
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_video_id
                        ON {tbl} (account_id, video_id)
                        WHERE video_id IS NOT NULL
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_extractor_type
                        ON {tbl} (account_id, extractor_type)
                        WHERE extractor_type IS NOT NULL
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_before_clip
                        ON {tbl} (account_id, video_id, before_clip)
                        WHERE before_clip IS NOT NULL
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
                key = (tbl, "legacy_columns_v1")
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
                        UPDATE {tbl} vi SET node_id = cn.node_id
                        FROM context_nodes cn
                        WHERE vi.node_uri = cn.uri AND vi.node_id IS NULL
                          AND vi.account_id = cn.account_id
                    """)
                    cur.execute(f"""
                        UPDATE {tbl} vi SET node_id = cn.node_id
                        FROM context_nodes cn
                        WHERE vi.node_id IS NULL
                          AND vi.node_uri LIKE '%%/content.md'
                          AND cn.uri = substring(vi.node_uri from '^(.+)/content[.]md')
                          AND vi.account_id = cn.account_id
                    """)
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
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn

        return psycopg2.connect(self._connection_string)

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
                metadata_json = Json(indexed_metadata(record.metadata))

                # Extract explicit security columns from filters
                acct = record.filters.get("account_id")
                owner = record.filters.get("owner_space")
                kind = record.metadata.get("record_kind", "node_content")
                space_id = record.filters.get("space_id")
                node_id = record.metadata.get("node_id")
                # M3 scope columns. Prefer the record's scope filters,
                # with metadata as a fallback for older writers.
                # Mirror them to dedicated SQL columns so
                # the WHERE clause can index-scan instead of doing
                # ``metadata @> '{"video_id": ...}'::jsonb`` for every
                # call. None is preserved (NOT coerced to empty string)
                # so partial indexes still cover the populated rows.
                video_id = record.filters.get("video_id", record.metadata.get("video_id"))
                extractor_type = record.filters.get(
                    "extractor_type", record.metadata.get("extractor_type")
                )
                before_clip = record.metadata.get("before_clip")

                try:
                    with conn.cursor() as cur:
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
                                video_id = %s,
                                extractor_type = %s,
                                before_clip = %s,
                                updated_at = NOW()
                            WHEN NOT MATCHED THEN INSERT
                                (id, uri, level, text, embedding, filters, metadata,
                                 account_id, owner_space, node_uri, node_id, record_kind,
                                 node_status, space_id, video_id, extractor_type,
                                 before_clip)
                            VALUES (%s, %s, %s, %s, %s::vector, %s, %s,
                                    %s, %s, %s, %s, %s, 'active', %s,
                                    %s, %s, %s)
                        """, (
                            record.id, acct,
                            record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                            owner, record.uri, node_id, kind, space_id,
                            video_id, extractor_type, before_clip,
                            record.id, record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                            acct, owner, record.uri, node_id, kind, space_id,
                            video_id, extractor_type, before_clip,
                        ))
                    conn.commit()
                except Exception as exc:
                    logger.warning(
                        "Upsert failed for record %s (uri=%s), rolling back: %s",
                        record.id, record.uri, exc,
                    )
                    conn.rollback()
        finally:
            self._return_connection(conn)

    def get_records_by_clips(self, filters: dict[str, Any], clip_ids: list[int]) -> list[IndexRecord]:
        clips = validate_clip_lookup(filters, clip_ids)
        if not clips:
            return []
        where, params = self._build_where(filters)
        params["clip_ids"] = clips
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, uri, level, text, filters, metadata
                    FROM {self._table_name}
                    WHERE {where} AND before_clip = ANY(%(clip_ids)s)
                    ORDER BY before_clip ASC, uri ASC
                """, params)
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        return [IndexRecord(
            id=str(row["id"]), uri=str(row["uri"]), level=int(row["level"]),
            text=row.get("text") or "", filters=_ensure_dict(row.get("filters")),
            metadata=_ensure_dict(row.get("metadata")),
        ) for row in rows]

    def get_records_at_time(self, filters: dict[str, Any], timestamp: float) -> list[IndexRecord]:
        """Find crossing intervals using guarded numeric SQL, not vector top-k."""
        validate_clip_lookup(filters, [])
        timestamp = seconds(timestamp)
        where, params = self._build_where(filters)
        params["at_sec"] = timestamp

        def numeric(key):
            expr = f"metadata->>'{key}'"
            return (
                f"(CASE WHEN length({expr}) <= 64 AND "
                f"{expr} ~ '^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{{1,3}})?$' "
                f"THEN ({expr})::numeric ELSE NULL END)"
            )

        start = numeric("source_start_sec")
        duration = numeric("source_duration_sec")
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, uri, level, text, filters, metadata
                    FROM {self._table_name}
                    WHERE {where}
                      AND {start} <= %(at_sec)s
                      AND {start} + {duration} > %(at_sec)s
                    ORDER BY uri ASC
                """, params)
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        records = [IndexRecord(
            id=str(row["id"]), uri=str(row["uri"]), level=int(row["level"]),
            text=row.get("text") or "", filters=_ensure_dict(row.get("filters")),
            metadata=_ensure_dict(row.get("metadata")),
        ) for row in rows]
        # Reject inconsistent source_end_sec and invalid legacy provenance too.
        return [record for record in records if contains_time(record.metadata, timestamp)]

    def get_record(self, record_id: str) -> IndexRecord | None:
        """Fetch one :class:`IndexRecord` by id.

        Returns ``None`` when no row matches. Used by
        :class:`m3.search.EpisodicSearcher` to recover the full
        ``text`` field of an index hit — ``search_by_vector`` only
        surfaces a 200-char abstract to keep the SeedHit payload
        small, but the QA evidence renderer needs the entire text.

        One round-trip per top-k hit, expected small (top_k typically
        ≤ 16). Avoids forcing the EpisodicSearcher to maintain a
        local mirror of the index.
        """
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

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index."""
        if not ids:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
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

    def search_by_transcript(self, query: str, filters: dict[str, Any], top_k: int) -> list[SeedHit]:
        """Scoped token-coverage baseline over ASR, independent of vector top-k.

        Uses a derived JSONB field populated on upsert. Old rows must be
        re-upserted; no fallback to descriptions or unscoped Python scanning.
        This is a per-video SQL scan, not an inverted full-text index.
        """
        terms = validate_transcript_request(query, filters, top_k)
        if not terms:
            return []
        where, params = self._build_where(filters)
        params["topk"] = top_k
        parts = []
        for i, term in enumerate(terms):
            # Both SQL and Python match complete normalized word tokens.
            params[f"lex_{i}"] = f" {term} "
            parts.append(
                f"CASE WHEN POSITION(%(lex_{i})s IN "
                f"COALESCE(metadata->>'{SEARCH_FIELD}', '')) > 0 THEN 1.0 ELSE 0.0 END"
            )
        score = "(" + " + ".join(parts) + f") / {len(terms)}"
        sql = f"""
            SELECT id, uri, level, text, filters, metadata, node_id, {score} AS score
            FROM {self._table_name}
            WHERE {where} AND ({score}) > 0
            ORDER BY score DESC, uri ASC
            LIMIT %(topk)s
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        return [self._row_to_hit(row) for row in rows]

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
        and space/video/temporal grants still require their explicit columns.

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
        if "resource_type" in f:
            resource_type = f["resource_type"]
            if isinstance(resource_type, list):
                clauses.append("filters->>'resource_type' = ANY(%(f_resource_type)s)")
            else:
                clauses.append("filters->>'resource_type' = %(f_resource_type)s")
            params["f_resource_type"] = resource_type
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
        # --- M3 scope columns ---------------------------------------------
        # ``video_id`` is exact-match: M3-Bench evaluates one video at a
        # time, so cross-video bleed is a correctness bug, not a perf
        # issue.
        if "video_id" in f:
            clauses.append("video_id = %(f_video_id)s")
            params["f_video_id"] = f["video_id"]
        # ``extractor_type`` follows the same scalar/list convention as
        # ``space_id``: M3 currently passes a single string, but the
        # adapter may pass a tuple via ``extractor_types`` and we want
        # ONE-clause to do it.
        if "extractor_type" in f:
            et = f["extractor_type"]
            if isinstance(et, list):
                clauses.append("extractor_type = ANY(%(f_extractor_type)s)")
                params["f_extractor_type"] = et
            else:
                clauses.append("extractor_type = %(f_extractor_type)s")
                params["f_extractor_type"] = et
        # Unknown clip provenance is not safe under a clip bound.
        if "before_clip" in f:
            clauses.append(
                "before_clip <= %(f_before_clip)s"
            )
            params["f_before_clip"] = f["before_clip"]

        if "before_sec" in f:
            params["f_before_sec"] = seconds(f["before_sec"])
            # Canonical finite numbers written by the indexer. CASE guards
            # casts, so malformed legacy JSON cannot abort a whole search.
            for key in ("source_start_sec", "source_duration_sec", "source_end_sec"):
                expr = f"metadata->>'{key}'"
                clauses.append(
                    f"(CASE WHEN length({expr}) <= 64 AND "
                    f"{expr} ~ '^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{{1,3}})?$' "
                    f"THEN ({expr})::numeric ELSE NULL END) <= %(f_before_sec)s"
                )

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
        # M3 forward-channel: the M3 driver stamps
        # ``extracted_node_id`` / ``video_id`` / ``clip_id`` /
        # ``before_clip`` into the metadata JSONB so the
        # ``GraphService`` VectorIndex routing path can recover the
        # originating graph node without an extra lookup. Surface
        # them here so consumers (the GraphService adapter, the
        # legacy JSONL ``EpisodicSearcher``) see them on
        # ``SeedHit.metadata`` without re-querying.
        for key in ("extracted_node_id", "video_id", "clip_id", "before_clip",
                    "source_start_sec", "source_duration_sec", "source_end_sec",
                    "speech_transcript"):
            if key in meta and meta[key] is not None:
                hit_metadata[key] = meta[key]
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
        if query.resource_type:
            filters["resource_type"] = query.resource_type

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
