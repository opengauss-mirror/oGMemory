"""Durable SQL lifecycle store for database-native lexical projections."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.default_lexical_analyzer import DefaultLexicalAnalyzer
from core.lexical import (
    DEFAULT_MAX_QUERY_TERMS,
    LexicalAnalyzer,
    LexicalRecord,
)
from providers.lexical_index.connection_pool import LexicalConnectionPool

try:
    import psycopg2
    from psycopg2.extras import execute_batch

    SQL_LEXICAL_AVAILABLE = True
except ImportError:
    psycopg2 = None
    execute_batch = None
    SQL_LEXICAL_AVAILABLE = False

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_RESERVED = frozenset({
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "FROM",
    "WHERE",
})
_FILTER_COLUMNS = frozenset({"account_id", "owner_space", "context_type", "category", "level", "memory_directory"})


def _schema_lock_key(table_name: str) -> int:
    digest = hashlib.sha256(f"ogmemory:lexical:{table_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _acquire_schema_lock(cur, table_name: str) -> None:
    cur.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (_schema_lock_key(table_name),),
    )


_SCHEMA_VERSION = 1

DEFAULT_POOL_SIZE = 5
DEFAULT_TABLE_NAME = "lexical_index"
DEFAULT_MAX_WRITE_BATCH_SIZE = 1000
DEFAULT_REBUILD_LEASE_SECONDS = 3600
DEFAULT_MAX_DOCUMENT_CHARS = 262_144
MAX_WRITE_BATCH_SIZE = 5000
MAX_QUERY_TERMS = 256
MAX_DOCUMENT_CHARS = 1_048_576


def _validate_identifier(name: str, *, label: str = "identifier") -> str:
    value = str(name).strip()
    if not _IDENTIFIER_PATTERN.match(value) or value.upper() in _RESERVED:
        raise ValueError(
            f"invalid {label} {value!r}; use a non-reserved SQL identifier "
            "containing only letters, digits, and underscores"
        )
    return value


@dataclass(frozen=True)
class PreparedSQLLexicalRecord:
    record: LexicalRecord
    metadata_json: str
    content: str


class SQLNativeProjectionStore:
    """Persistent lifecycle and rebuild store for a native lexical engine."""

    supported_filters = _FILTER_COLUMNS
    _schema_version = _SCHEMA_VERSION

    @staticmethod
    def _rollback_preserving(conn) -> None:
        try:
            conn.rollback()
        except Exception:
            pass

    def __init__(
        self,
        *,
        connection_string: str,
        table_name: str = DEFAULT_TABLE_NAME,
        provider_name: str = "sql",
        pool_size: int = DEFAULT_POOL_SIZE,
        pool: LexicalConnectionPool | None = None,
        analyzer: LexicalAnalyzer | None = None,
        max_write_batch_size: int = DEFAULT_MAX_WRITE_BATCH_SIZE,
        max_query_terms: int = DEFAULT_MAX_QUERY_TERMS,
        rebuild_lease_seconds: int = DEFAULT_REBUILD_LEASE_SECONDS,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        connect_timeout_seconds: int | None = None,
        session_statement_timeout_ms: int | None = None,
    ) -> None:
        if not SQL_LEXICAL_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for SQLNativeProjectionStore. "
                "Install with: pip install psycopg2-binary"
            )
        if not connection_string:
            raise ValueError("connection_string is required for SQL lexical provider")
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        for name, value, maximum in (
            ("max_write_batch_size", max_write_batch_size, MAX_WRITE_BATCH_SIZE),
            ("max_query_terms", max_query_terms, MAX_QUERY_TERMS),
            ("max_document_chars", max_document_chars, MAX_DOCUMENT_CHARS),
        ):
            if not 1 <= int(value) <= maximum:
                raise ValueError(f"{name} must be between 1 and {maximum}")
        if rebuild_lease_seconds < 1:
            raise ValueError("rebuild_lease_seconds must be positive")
        self._provider_name = str(provider_name or "sql")
        self._table = _validate_identifier(table_name, label="table prefix")
        self._analyzer = analyzer or DefaultLexicalAnalyzer()
        self._max_write_batch_size = int(max_write_batch_size)
        self._max_query_terms = int(max_query_terms)
        self._rebuild_lease_seconds = int(rebuild_lease_seconds)
        self._max_document_chars = int(max_document_chars)
        self._limit_metrics_lock = threading.Lock()
        self._document_truncations = 0
        self._metrics_lock = threading.Lock()
        self._query_count = 0
        self._query_seconds_total = 0.0
        self._last_query_seconds = 0.0
        self._max_query_seconds = 0.0
        self._last_error: str | None = None
        self._closed = False
        self._closing = False
        self._close_lock = threading.Lock()
        self._shared_pool = pool or LexicalConnectionPool(
            connection_string,
            pool_size=pool_size,
            connect_fn=psycopg2.connect,
            connect_timeout_seconds=connect_timeout_seconds,
            session_statement_timeout_ms=session_statement_timeout_ms,
        )
        self._owns_pool = pool is None
        try:
            self._ensure_schema()
        except Exception:
            if self._owns_pool:
                try:
                    self._shared_pool.close_all()
                except Exception:
                    pass
            raise

    @property
    def _metadata_table(self) -> str:
        return f"{self._table}_metadata"

    @property
    def _generation_sequence(self) -> str:
        return f"{self._table}_generation_seq"

    @property
    def _docs_table(self) -> str:
        return f"{self._table}_docs"

    @property
    def _tombstones_table(self) -> str:
        return f"{self._table}_tombstones"

    @property
    def _rebuilds_table(self) -> str:
        return f"{self._table}_rebuilds"

    @property
    def _rebuild_docs_table(self) -> str:
        return f"{self._table}_rebuild_docs"

    @property
    def _scope_epochs_table(self) -> str:
        return f"{self._table}_scope_epochs"

    @property
    def _scope_tombstones_table(self) -> str:
        return f"{self._table}_scope_tombstones"

    @property
    def _schema_identity(self) -> str:
        return self._analyzer.identity

    def upsert_many(self, records: Sequence[LexicalRecord]) -> None:
        if not records:
            return
        if len(records) > self._max_write_batch_size:
            raise ValueError(
                f"lexical batch exceeds max_write_batch_size={self._max_write_batch_size}"
            )
        prepared = [self._prepare_record(record) for record in records]
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                self._acquire_mutation_locks(
                    cur,
                    account_ids=(item.record.account_id for item in prepared),
                    record_ids=(item.record.id for item in prepared),
                )
                generation = self._next_generation(cur)
                record_count_delta = 0
                for prepared_record in prepared:
                    record = prepared_record.record
                    inserted = self._upsert_doc(
                        cur,
                        record=record,
                        metadata_json=prepared_record.metadata_json,
                        content=prepared_record.content,
                        generation=generation,
                    )
                    if inserted:
                        record_count_delta += 1
                    cur.execute(
                        f"""
                        DELETE FROM {self._tombstones_table}
                        WHERE record_id=%s AND projection_generation<=%s
                        """,
                        (record.id, generation),
                    )
                self._adjust_record_count(cur, record_count_delta)
            conn.commit()
            self._last_error = None
        except Exception as exc:
            self._rollback_preserving(conn)
            self._last_error = type(exc).__name__
            raise
        finally:
            self._return(conn)

    def delete(self, record_ids: Sequence[str]) -> None:
        ids = tuple(dict.fromkeys(str(record_id) for record_id in record_ids if record_id))
        if not ids:
            return
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                accounts = self._record_accounts(cur, ids)
                self._acquire_mutation_locks(
                    cur,
                    account_ids=accounts,
                    record_ids=ids,
                )
                generation = self._next_generation(cur)
                record_count_delta = 0
                for record_id in ids:
                    deleted = self._delete_record(cur, record_id, generation)
                    if deleted:
                        record_count_delta -= 1
                    self._upsert_record_tombstone(cur, record_id, generation)
                self._adjust_record_count(cur, record_count_delta)
                self._prune_tombstones(cur)
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def delete_scope(self, *, account_id: str, owner_space: str | None = None) -> int:
        if not account_id:
            raise ValueError("account_id is required")
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                self._acquire_mutation_locks(cur, account_ids=(account_id,))
                generation = self._next_generation(cur)
                where_sql, params = self._scope_where(account_id, owner_space)
                cur.execute(
                    f"""
                    DELETE FROM {self._docs_table}
                    WHERE {where_sql} AND projection_generation<=%s
                    """,
                    (*params, generation),
                )
                deleted_count = int(cur.rowcount)
                scope_key = owner_space if owner_space is not None else "*"
                self._upsert_scope_tombstone(cur, account_id, scope_key, generation)
                self._adjust_record_count(cur, -deleted_count)
                self._prune_tombstones(cur)
            conn.commit()
            return deleted_count
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def begin_rebuild(self) -> str:
        rebuild_id = uuid.uuid4().hex
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                self._expire_stale_rebuilds(cur)
                snapshot_generation = self._next_generation(cur)
                rebuild_generation = self._next_generation(cur)
                cur.execute(
                    f"""
                    INSERT INTO {self._rebuilds_table}(
                        rebuild_id,snapshot_generation,rebuild_generation
                    ) VALUES(%s,%s,%s)
                    """,
                    (rebuild_id, snapshot_generation, rebuild_generation),
                )
            conn.commit()
            return rebuild_id
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def ready_scopes(
        self, account_id: str, owner_spaces: Sequence[str]
    ) -> frozenset[str]:
        spaces = tuple(dict.fromkeys(str(value) for value in owner_spaces if value))
        if not account_id or not spaces:
            return frozenset()
        placeholders = ",".join(["%s"] * len(spaces))
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT e.owner_space
                    FROM {self._scope_epochs_table} e
                    CROSS JOIN {self._metadata_table} m
                    WHERE e.account_id=%s
                      AND e.owner_space IN ({placeholders})
                      AND e.rebuild_generation >= e.dirty_generation
                      AND e.rebuild_generation >= m.all_scopes_dirty_generation
                    """,
                    (account_id, *spaces),
                )
                return frozenset(str(row[0]) for row in cur.fetchall())
        finally:
            self._return(conn)

    def dirty_generations(
        self, account_id: str, owner_spaces: Sequence[str]
    ) -> dict[str, int]:
        spaces = tuple(dict.fromkeys(str(value) for value in owner_spaces if value))
        if not account_id or not spaces:
            return {}
        placeholders = ",".join(["%s"] * len(spaces))
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT e.owner_space, e.dirty_generation
                    FROM {self._scope_epochs_table} e
                    WHERE e.account_id=%s AND e.owner_space IN ({placeholders})
                    """,
                    (account_id, *spaces),
                )
                scope_dirty = {str(row[0]): int(row[1]) for row in cur.fetchall()}
                cur.execute(
                    f"""
                    SELECT all_scopes_dirty_generation
                    FROM {self._metadata_table} WHERE singleton=1
                    """
                )
                row = cur.fetchone()
                global_dirty = int(row[0]) if row else 0
                return {
                    owner_space: max(global_dirty, scope_dirty.get(owner_space, 0))
                    for owner_space in spaces
                }
        finally:
            self._return(conn)

    def mark_scopes_dirty(self, scopes: Sequence[tuple[str, str]]) -> None:
        normalized = tuple(dict.fromkeys(
            (str(account_id), str(owner_space))
            for account_id, owner_space in scopes
            if account_id and owner_space
        ))
        if not normalized:
            return
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                self._acquire_mutation_locks(
                    cur,
                    account_ids=(account_id for account_id, _ in normalized),
                )
                dirty_generation = self._next_generation(cur)
                for account_id, owner_space in normalized:
                    self._upsert_scope_epoch(
                        cur,
                        account_id,
                        owner_space,
                        rebuild_generation=0,
                        dirty_generation=dirty_generation,
                    )
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def mark_all_scopes_dirty(self) -> None:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                dirty_generation = self._next_generation(cur)
                cur.execute(
                    f"""
                    UPDATE {self._metadata_table}
                    SET all_scopes_dirty_generation=
                        GREATEST(all_scopes_dirty_generation,%s)
                    WHERE singleton=1
                    """,
                    (dirty_generation,),
                )
                cur.execute(
                    f"""
                    UPDATE {self._scope_epochs_table}
                    SET dirty_generation=GREATEST(dirty_generation,%s)
                    WHERE dirty_generation < %s
                    """,
                    (dirty_generation, dirty_generation),
                )
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def replace_scopes(
        self,
        *,
        rebuild_id: str,
        account_id: str,
        owner_spaces: Sequence[str],
        records: Iterable[LexicalRecord],
    ) -> bool:
        scopes = tuple(dict.fromkeys(str(scope) for scope in owner_spaces if scope))
        scope_set = frozenset(scopes)
        if not account_id or not scopes:
            raise ValueError("account_id and owner_spaces are required")
        staged: list[PreparedSQLLexicalRecord] = []
        for record in records:
            if record.account_id != account_id or record.owner_space not in scope_set:
                raise ValueError("replacement records must belong to the requested scopes")
            staged.append(self._prepare_record(record))
            if len(staged) >= self._max_write_batch_size:
                self._stage_rebuild_batch(rebuild_id, staged)
                staged = []
        if staged:
            self._stage_rebuild_batch(rebuild_id, staged)
        try:
            published = self._publish_rebuild(rebuild_id, account_id, scopes)
            self._last_error = None
            return published
        except Exception as exc:
            self._last_error = type(exc).__name__
            try:
                self.abort_rebuild(rebuild_id)
            except Exception:
                # Cleanup failure must not replace the publish failure that
                # determines retry and incident diagnosis.
                pass
            raise

    def abort_rebuild(self, rebuild_id: str) -> None:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._rebuild_docs_table} WHERE rebuild_id=%s",
                    (rebuild_id,),
                )
                cur.execute(
                    f"DELETE FROM {self._rebuilds_table} WHERE rebuild_id=%s",
                    (rebuild_id,),
                )
                self._prune_tombstones(cur)
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def health(self) -> dict[str, Any]:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT record_count FROM {self._metadata_table} WHERE singleton=1")
                row = cur.fetchone()
                records = int(row[0]) if row else 0
            return {
                "available": not self._closed,
                "provider": self._provider_name,
                "records": records,
                "analyzer": self._analyzer.identity,
                "table_name": self._table,
                "document_truncations": self._document_truncations,
                "max_document_chars": self._max_document_chars,
                "query_count": self._query_count,
                "query_seconds_total": self._query_seconds_total,
                "last_query_seconds": self._last_query_seconds,
                "max_query_seconds": self._max_query_seconds,
                "last_error": self._last_error,
            }
        except Exception as exc:
            return {
                "available": False,
                "provider": self._provider_name,
                "table_name": self._table,
                "error_type": type(exc).__name__,
            }
        finally:
            self._return(conn)

    def close(self, timeout: float = 5.0) -> bool:
        with self._close_lock:
            self._closing = True
            if self._closed:
                return True
            ok = True
            if self._owns_pool:
                ok = self._shared_pool.close_all(timeout=timeout)
            if ok:
                self._closed = True
            return ok

    def _ensure_schema(self) -> None:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                _acquire_schema_lock(cur, self._table)
                cur.execute(
                    """
                    SELECT table_class.oid
                    FROM pg_class table_class
                    JOIN pg_namespace ns ON ns.oid=table_class.relnamespace
                    WHERE ns.nspname=current_schema()
                      AND table_class.relname=%s
                      AND table_class.relkind='r'
                    LIMIT 1
                    """,
                    (self._metadata_table,),
                )
                existing = cur.fetchone()
                if existing and existing[0] is not None:
                    cur.execute(
                        f"SELECT schema_version,analyzer_identity "
                        f"FROM {self._metadata_table} WHERE singleton=1"
                    )
                    identity = cur.fetchone()
                    if identity and (
                        int(identity[0]) != self._schema_version
                        or str(identity[1]) != self._schema_identity
                    ):
                        self._handle_schema_mismatch(cur, identity)
                cur.execute(
                    f"CREATE SEQUENCE IF NOT EXISTS {self._generation_sequence}"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._metadata_table} (
                        singleton INTEGER PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        analyzer_identity TEXT NOT NULL,
                        record_count BIGINT NOT NULL DEFAULT 0,
                        all_scopes_dirty_generation BIGINT NOT NULL DEFAULT 0,
                        CHECK(singleton=1)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._docs_table} (
                        record_id TEXT PRIMARY KEY,
                        uri TEXT NOT NULL,
                        account_id TEXT NOT NULL,
                        owner_space TEXT NOT NULL,
                        context_type TEXT,
                        category TEXT,
                        level INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        content TEXT,
                        projection_generation BIGINT NOT NULL DEFAULT 0
                    )
                    """
                )
                cur.execute(
                    f"ALTER TABLE {self._docs_table} ADD COLUMN IF NOT EXISTS content TEXT"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_scope ON {self._docs_table}(account_id, owner_space, level)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_type ON {self._docs_table}(account_id, context_type, category, level)"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._tombstones_table} (
                        record_id TEXT PRIMARY KEY,
                        projection_generation BIGINT NOT NULL
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._rebuilds_table} (
                        rebuild_id TEXT PRIMARY KEY,
                        snapshot_generation BIGINT NOT NULL,
                        rebuild_generation BIGINT NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._rebuild_docs_table} (
                        rebuild_id TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        account_id TEXT NOT NULL,
                        owner_space TEXT NOT NULL,
                        context_type TEXT,
                        category TEXT,
                        level INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        content TEXT,
                        PRIMARY KEY(rebuild_id, record_id)
                    )
                    """
                )
                cur.execute(
                    f"ALTER TABLE {self._rebuild_docs_table} ADD COLUMN IF NOT EXISTS content TEXT"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._scope_epochs_table} (
                        account_id TEXT NOT NULL,
                        owner_space TEXT NOT NULL,
                        rebuild_generation BIGINT NOT NULL,
                        dirty_generation BIGINT NOT NULL DEFAULT 0,
                        PRIMARY KEY(account_id, owner_space)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._scope_tombstones_table} (
                        account_id TEXT NOT NULL,
                        owner_space TEXT NOT NULL,
                        projection_generation BIGINT NOT NULL,
                        PRIMARY KEY(account_id, owner_space)
                    )
                    """
                )
                cur.execute(
                    f"SELECT schema_version, analyzer_identity "
                    f"FROM {self._metadata_table} WHERE singleton=1"
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        f"""
                        INSERT INTO {self._metadata_table}(
                            singleton,schema_version,analyzer_identity,
                            record_count,all_scopes_dirty_generation
                        ) VALUES(1,%s,%s,0,0)
                        """,
                        (self._schema_version, self._schema_identity),
                    )
                elif int(row[0]) != self._schema_version or str(row[1]) != self._schema_identity:
                    self._handle_schema_mismatch(cur, row)
                cur.execute(
                    f"SELECT singleton FROM {self._metadata_table} WHERE singleton=1 FOR UPDATE"
                )
                if cur.fetchone() is None:
                    raise RuntimeError("SQL lexical metadata row is missing")
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def _reset_projection(self, cur) -> None:
        for table in (
            self._docs_table,
            self._tombstones_table,
            self._rebuild_docs_table,
            self._rebuilds_table,
            self._scope_epochs_table,
            self._scope_tombstones_table,
        ):
            cur.execute(f"DELETE FROM {table}")
        cur.execute(
            f"""
            UPDATE {self._metadata_table}
            SET schema_version=%s, analyzer_identity=%s, record_count=0,
                all_scopes_dirty_generation=0
            WHERE singleton=1
            """,
            (self._schema_version, self._schema_identity),
        )

    def _handle_schema_mismatch(self, cur, row) -> None:
        """Reset a lifecycle store when its persisted identity changes."""
        self._reset_projection(cur)

    def _stage_rebuild_batch(
        self,
        rebuild_id: str,
        batch: list[PreparedSQLLexicalRecord],
    ) -> None:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                cur.execute(
                    f"SELECT 1 FROM {self._rebuilds_table} WHERE rebuild_id=%s",
                    (rebuild_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError("unknown or expired lexical rebuild")
                statement = f"""
                    DELETE FROM {self._rebuild_docs_table}
                    WHERE rebuild_id=%s AND record_id=%s;
                    INSERT INTO {self._rebuild_docs_table}(
                        rebuild_id,record_id,uri,account_id,owner_space,
                        context_type,category,level,metadata_json,content
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """
                params = []
                for prepared_record in batch:
                    record = prepared_record.record
                    params.append(
                        (
                            rebuild_id,
                            record.id,
                            rebuild_id,
                            record.id,
                            record.uri,
                            record.account_id,
                            record.owner_space,
                            record.metadata.get("context_type"),
                            record.metadata.get("category"),
                            record.level,
                            prepared_record.metadata_json,
                            prepared_record.content,
                        )
                    )
                execute_batch(cur, statement, params, page_size=100)
                cur.execute(
                    f"UPDATE {self._rebuilds_table} SET updated_at=CURRENT_TIMESTAMP WHERE rebuild_id=%s",
                    (rebuild_id,),
                )
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def _publish_rebuild(
        self,
        rebuild_id: str,
        account_id: str,
        owner_spaces: Sequence[str],
    ) -> bool:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._ensure_compatible(cur)
                self._acquire_mutation_locks(cur, account_ids=(account_id,))
                cur.execute(
                    f"""
                    SELECT snapshot_generation,rebuild_generation
                    FROM {self._rebuilds_table} WHERE rebuild_id=%s
                    """,
                    (rebuild_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError("unknown or expired lexical rebuild")
                snapshot_generation = int(row[0])
                rebuild_generation = int(row[1])
                placeholders = ",".join(["%s"] * len(owner_spaces))
                for owner_space in owner_spaces:
                    self._ensure_scope_epoch_exists(cur, account_id, owner_space)
                cur.execute(
                    f"""
                    SELECT owner_space FROM {self._scope_epochs_table}
                    WHERE account_id=%s AND owner_space IN ({placeholders})
                    FOR UPDATE
                    """,
                    (account_id, *owner_spaces),
                )
                locked_scopes = {str(row[0]) for row in cur.fetchall()}
                if locked_scopes != set(owner_spaces):
                    raise RuntimeError("failed to lock lexical scope epochs")
                cur.execute(
                    f"""
                    SELECT all_scopes_dirty_generation
                    FROM {self._metadata_table} WHERE singleton=1
                    """
                )
                meta = cur.fetchone()
                if meta is None:
                    raise RuntimeError("SQL lexical metadata row is missing")
                if int(meta[0]) > snapshot_generation:
                    self._discard_rebuild(cur, rebuild_id)
                    conn.commit()
                    return False
                cur.execute(
                    f"""
                    SELECT 1 FROM {self._scope_epochs_table}
                    WHERE account_id=%s AND owner_space IN ({placeholders})
                      AND (dirty_generation>%s OR rebuild_generation>=%s)
                    """,
                    (account_id, *owner_spaces, snapshot_generation, rebuild_generation),
                )
                if cur.fetchone() is not None:
                    self._discard_rebuild(cur, rebuild_id)
                    conn.commit()
                    return False
                cur.execute(
                    f"""
                    DELETE FROM {self._docs_table}
                    WHERE account_id=%s AND owner_space IN ({placeholders})
                      AND projection_generation<=%s
                    """,
                    (account_id, *owner_spaces, snapshot_generation),
                )
                record_count_delta = -int(cur.rowcount)
                cur.execute(
                    f"""
                    INSERT INTO {self._docs_table}(
                        record_id,uri,account_id,owner_space,context_type,
                        category,level,metadata_json,content,
                        projection_generation
                    )
                    SELECT r.record_id,r.uri,r.account_id,r.owner_space,
                           r.context_type,r.category,r.level,r.metadata_json,
                           r.content,%s
                    FROM {self._rebuild_docs_table} r
                    WHERE r.rebuild_id=%s
                      AND NOT EXISTS (
                          SELECT 1 FROM {self._docs_table} live
                          WHERE live.record_id=r.record_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM {self._tombstones_table} tomb
                          WHERE tomb.record_id=r.record_id
                            AND tomb.projection_generation>%s
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM {self._scope_tombstones_table} scope_tomb
                          WHERE scope_tomb.account_id=r.account_id
                            AND scope_tomb.owner_space IN (r.owner_space, '*')
                            AND scope_tomb.projection_generation>%s
                      )
                    """,
                    (
                        rebuild_generation,
                        rebuild_id,
                        snapshot_generation,
                        snapshot_generation,
                    ),
                )
                record_count_delta += int(cur.rowcount)
                for owner_space in owner_spaces:
                    self._upsert_scope_epoch(
                        cur,
                        account_id,
                        owner_space,
                        rebuild_generation=rebuild_generation,
                        dirty_generation=0,
                    )
                self._discard_rebuild(cur, rebuild_id)
                self._adjust_record_count(cur, record_count_delta)
                self._prune_tombstones(cur)
            conn.commit()
            return True
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def _prepare_record(self, record: LexicalRecord) -> PreparedSQLLexicalRecord:
        raise NotImplementedError("native lexical providers must prepare their records")

    def _upsert_doc(
        self,
        cur,
        *,
        record: LexicalRecord,
        metadata_json: str,
        generation: int,
        content: str = "",
    ) -> bool:
        if self._record_write_is_superseded(cur, record, generation):
            return False
        params = (
            record.uri,
            record.account_id,
            record.owner_space,
            record.metadata.get("context_type"),
            record.metadata.get("category"),
            record.level,
            metadata_json,
            content,
            generation,
            record.id,
            generation,
        )
        cur.execute(
            f"""
            UPDATE {self._docs_table}
            SET uri=%s, account_id=%s, owner_space=%s, context_type=%s,
                category=%s, level=%s, metadata_json=%s,
                content=%s, projection_generation=%s
            WHERE record_id=%s AND projection_generation<=%s
            """,
            params,
        )
        if cur.rowcount:
            return False
        cur.execute(
            f"SELECT projection_generation FROM {self._docs_table} WHERE record_id=%s",
            (record.id,),
        )
        if cur.fetchone() is not None:
            return False
        cur.execute("SAVEPOINT lexical_doc_insert")
        try:
            cur.execute(
                f"""
                INSERT INTO {self._docs_table}(
                    record_id,uri,account_id,owner_space,context_type,
                    category,level,metadata_json,content,
                    projection_generation
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    record.id,
                    record.uri,
                    record.account_id,
                    record.owner_space,
                    record.metadata.get("context_type"),
                    record.metadata.get("category"),
                    record.level,
                    metadata_json,
                    content,
                    generation,
                ),
            )
            cur.execute("RELEASE SAVEPOINT lexical_doc_insert")
            return True
        except psycopg2.IntegrityError:
            cur.execute("ROLLBACK TO SAVEPOINT lexical_doc_insert")
            if self._record_write_is_superseded(cur, record, generation):
                cur.execute("RELEASE SAVEPOINT lexical_doc_insert")
                return False
            cur.execute(
                f"""
                UPDATE {self._docs_table}
                SET uri=%s, account_id=%s, owner_space=%s, context_type=%s,
                    category=%s, level=%s, metadata_json=%s,
                    content=%s, projection_generation=%s
                WHERE record_id=%s AND projection_generation<=%s
                """,
                params,
            )
            if not cur.rowcount:
                cur.execute(
                    f"SELECT 1 FROM {self._docs_table} WHERE record_id=%s",
                    (record.id,),
                )
                if cur.fetchone() is None:
                    raise
            cur.execute("RELEASE SAVEPOINT lexical_doc_insert")
            return False

    def _record_write_is_superseded(
        self,
        cur,
        record: LexicalRecord,
        generation: int,
    ) -> bool:
        cur.execute(
            f"""
            SELECT 1
            WHERE EXISTS (
                SELECT 1 FROM {self._tombstones_table}
                WHERE record_id=%s AND projection_generation>%s
            ) OR EXISTS (
                SELECT 1 FROM {self._scope_tombstones_table}
                WHERE account_id=%s AND owner_space IN (%s, '*')
                  AND projection_generation>%s
            )
            """,
            (
                record.id,
                generation,
                record.account_id,
                record.owner_space,
                generation,
            ),
        )
        return cur.fetchone() is not None

    def _acquire_mutation_locks(
        self,
        cur,
        *,
        account_ids: Iterable[str] = (),
        record_ids: Iterable[str] = (),
    ) -> None:
        """Serialize projection mutations in a stable cross-process order."""
        lock_values = (
            *(f"account:{value}" for value in sorted(set(account_ids))),
            *(f"record:{value}" for value in sorted(set(record_ids))),
        )
        for value in lock_values:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_schema_lock_key(f"{self._table}:mutation:{value}"),),
            )

    def _record_accounts(
        self,
        cur,
        record_ids: Sequence[str],
    ) -> tuple[str, ...]:
        placeholders = ",".join(["%s"] * len(record_ids))
        cur.execute(
            f"""
            SELECT DISTINCT account_id
            FROM (
                SELECT account_id FROM {self._docs_table}
                WHERE record_id IN ({placeholders})
                UNION ALL
                SELECT account_id FROM {self._rebuild_docs_table}
                WHERE record_id IN ({placeholders})
            ) affected
            ORDER BY account_id
            """,
            (*record_ids, *record_ids),
        )
        return tuple(str(row[0]) for row in cur.fetchall())

    def _upsert_record_tombstone(
        self,
        cur,
        record_id: str,
        generation: int,
    ) -> None:
        cur.execute(
            f"""
            UPDATE {self._tombstones_table}
            SET projection_generation=GREATEST(projection_generation,%s)
            WHERE record_id=%s
            """,
            (generation, record_id),
        )
        if cur.rowcount:
            return
        cur.execute("SAVEPOINT lexical_tombstone_insert")
        try:
            cur.execute(
                f"""
                INSERT INTO {self._tombstones_table}
                    (record_id,projection_generation)
                VALUES(%s,%s)
                """,
                (record_id, generation),
            )
            cur.execute("RELEASE SAVEPOINT lexical_tombstone_insert")
        except psycopg2.IntegrityError:
            cur.execute("ROLLBACK TO SAVEPOINT lexical_tombstone_insert")
            cur.execute(
                f"""
                UPDATE {self._tombstones_table}
                SET projection_generation=GREATEST(projection_generation,%s)
                WHERE record_id=%s
                """,
                (generation, record_id),
            )
            if not cur.rowcount:
                raise
            cur.execute("RELEASE SAVEPOINT lexical_tombstone_insert")

    def _upsert_scope_tombstone(
        self,
        cur,
        account_id: str,
        owner_space: str,
        generation: int,
    ) -> None:
        cur.execute(
            f"""
            UPDATE {self._scope_tombstones_table}
            SET projection_generation=GREATEST(projection_generation,%s)
            WHERE account_id=%s AND owner_space=%s
            """,
            (generation, account_id, owner_space),
        )
        if cur.rowcount:
            return
        cur.execute("SAVEPOINT lexical_scope_tombstone_insert")
        try:
            cur.execute(
                f"""
                INSERT INTO {self._scope_tombstones_table}
                    (account_id,owner_space,projection_generation)
                VALUES(%s,%s,%s)
                """,
                (account_id, owner_space, generation),
            )
            cur.execute("RELEASE SAVEPOINT lexical_scope_tombstone_insert")
        except psycopg2.IntegrityError:
            cur.execute("ROLLBACK TO SAVEPOINT lexical_scope_tombstone_insert")
            cur.execute(
                f"""
                UPDATE {self._scope_tombstones_table}
                SET projection_generation=GREATEST(projection_generation,%s)
                WHERE account_id=%s AND owner_space=%s
                """,
                (generation, account_id, owner_space),
            )
            if not cur.rowcount:
                raise
            cur.execute("RELEASE SAVEPOINT lexical_scope_tombstone_insert")

    def _upsert_scope_epoch(
        self,
        cur,
        account_id: str,
        owner_space: str,
        *,
        rebuild_generation: int,
        dirty_generation: int,
    ) -> None:
        cur.execute(
            f"""
            UPDATE {self._scope_epochs_table}
            SET rebuild_generation=%s, dirty_generation=%s
            WHERE account_id=%s AND owner_space=%s
            """,
            (rebuild_generation, dirty_generation, account_id, owner_space),
        )
        if cur.rowcount:
            return
        cur.execute("SAVEPOINT lexical_scope_epoch_insert")
        try:
            cur.execute(
                f"""
                INSERT INTO {self._scope_epochs_table}(
                    account_id,owner_space,rebuild_generation,dirty_generation
                ) VALUES(%s,%s,%s,%s)
                """,
                (account_id, owner_space, rebuild_generation, dirty_generation),
            )
            cur.execute("RELEASE SAVEPOINT lexical_scope_epoch_insert")
        except psycopg2.IntegrityError:
            cur.execute("ROLLBACK TO SAVEPOINT lexical_scope_epoch_insert")
            cur.execute(
                f"""
                UPDATE {self._scope_epochs_table}
                SET rebuild_generation=%s, dirty_generation=%s
                WHERE account_id=%s AND owner_space=%s
                """,
                (rebuild_generation, dirty_generation, account_id, owner_space),
            )
            if not cur.rowcount:
                raise
            cur.execute("RELEASE SAVEPOINT lexical_scope_epoch_insert")

    def _ensure_scope_epoch_exists(
        self,
        cur,
        account_id: str,
        owner_space: str,
    ) -> None:
        cur.execute(
            f"""
            SELECT 1 FROM {self._scope_epochs_table}
            WHERE account_id=%s AND owner_space=%s
            """,
            (account_id, owner_space),
        )
        if cur.fetchone() is not None:
            return
        cur.execute("SAVEPOINT lexical_scope_epoch_ensure")
        try:
            cur.execute(
                f"""
                INSERT INTO {self._scope_epochs_table}(
                    account_id,owner_space,rebuild_generation,dirty_generation
                ) VALUES(%s,%s,0,0)
                """,
                (account_id, owner_space),
            )
            cur.execute("RELEASE SAVEPOINT lexical_scope_epoch_ensure")
        except psycopg2.IntegrityError:
            cur.execute("ROLLBACK TO SAVEPOINT lexical_scope_epoch_ensure")
            cur.execute("RELEASE SAVEPOINT lexical_scope_epoch_ensure")

    def _delete_record(self, cur, record_id: str, generation: int) -> bool:
        cur.execute(
            f"""
            DELETE FROM {self._docs_table}
            WHERE record_id=%s AND projection_generation<=%s
            """,
            (record_id, generation),
        )
        return bool(cur.rowcount)

    def _filter_where(self, filters: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, expected in filters.items():
            if key not in self.supported_filters:
                raise ValueError(f"unsupported lexical filter: {key}")
            values = (
                list(expected)
                if isinstance(expected, (list, tuple, set, frozenset))
                else [expected]
            )
            values = [value for value in values if value is not None]
            if not values:
                clauses.append("1=0")
                continue
            column = f"d.{key}"
            if key == "memory_directory":
                column = """CASE WHEN split_part(d.uri, '/', 6) = 'memories'
                    THEN split_part(d.uri, '/', 7) ELSE split_part(d.uri, '/', 6) END"""
            if len(values) == 1:
                clauses.append(f"{column}=%s")
                params.append(values[0])
            else:
                clauses.append(f"{column} IN ({','.join(['%s'] * len(values))})")
                params.extend(values)
        if not clauses:
            clauses.append("1=1")
        return " AND ".join(clauses), tuple(params)

    def _scope_where(self, account_id: str, owner_space: str | None) -> tuple[str, tuple[Any, ...]]:
        if owner_space is None:
            return "account_id=%s", (account_id,)
        return "account_id=%s AND owner_space=%s", (account_id, owner_space)

    def _ensure_compatible(self, cur) -> None:
        cur.execute(
            f"""
            SELECT schema_version, analyzer_identity
            FROM {self._metadata_table} WHERE singleton=1
            """
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("SQL lexical metadata row is missing")
        if int(row[0]) != self._schema_version or str(row[1]) != self._schema_identity:
            self._handle_schema_mismatch(cur, row)

    def _next_generation(self, cur) -> int:
        cur.execute("SELECT nextval(%s)", (self._generation_sequence,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("SQL lexical generation sequence returned no value")
        return int(row[0])

    def _adjust_record_count(self, cur, delta: int) -> None:
        if not delta:
            return
        cur.execute(
            f"""
            UPDATE {self._metadata_table}
            SET record_count=record_count+%s
            WHERE singleton=1
            """,
            (delta,),
        )

    def _discard_rebuild(self, cur, rebuild_id: str) -> None:
        cur.execute(
            f"DELETE FROM {self._rebuild_docs_table} WHERE rebuild_id=%s",
            (rebuild_id,),
        )
        cur.execute(
            f"DELETE FROM {self._rebuilds_table} WHERE rebuild_id=%s",
            (rebuild_id,),
        )

    def _expire_stale_rebuilds(self, cur) -> None:
        cur.execute(
            f"""
            DELETE FROM {self._rebuild_docs_table}
            WHERE rebuild_id IN (
                SELECT rebuild_id FROM {self._rebuilds_table}
                WHERE updated_at < CURRENT_TIMESTAMP
                    - %s * interval '1 second'
            )
            """,
            (self._rebuild_lease_seconds,),
        )
        cur.execute(
            f"""
            DELETE FROM {self._rebuilds_table}
            WHERE updated_at < CURRENT_TIMESTAMP
                - %s * interval '1 second'
            """,
            (self._rebuild_lease_seconds,),
        )

    def _prune_tombstones(self, cur) -> None:
        cur.execute(
            f"""
            SELECT COALESCE(
                MIN(snapshot_generation),
                9223372036854775807
            )
            FROM {self._rebuilds_table}
            """
        )
        threshold = int(cur.fetchone()[0])
        cur.execute(
            f"DELETE FROM {self._tombstones_table} WHERE projection_generation<=%s",
            (threshold,),
        )
        cur.execute(
            f"""
            DELETE FROM {self._scope_tombstones_table}
            WHERE projection_generation<=%s
            """,
            (threshold,),
        )

    @staticmethod
    def _loads_metadata(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if not raw:
            return {}
        try:
            parsed = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _borrow(self):
        if self._closed or self._closing:
            raise RuntimeError("lexical index is closed")
        return self._shared_pool.get_connection()

    def _return(self, conn) -> None:
        try:
            if (
                conn.get_transaction_status()
                != psycopg2.extensions.TRANSACTION_STATUS_IDLE
            ):
                conn.rollback()
        except Exception:
            conn.close()
            self._shared_pool.return_connection(conn)
            return
        self._shared_pool.return_connection(conn)

    def _record_query(self, seconds: float) -> None:
        with self._metrics_lock:
            self._query_count += 1
            self._query_seconds_total += seconds
            self._last_query_seconds = seconds
            self._max_query_seconds = max(self._max_query_seconds, seconds)
