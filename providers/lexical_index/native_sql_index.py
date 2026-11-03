"""Database-native lexical providers for PostgreSQL and openGauss."""

from __future__ import annotations

import json
import time
from typing import Any

from core.lexical import LexicalHit, LexicalRecord
from providers.lexical_index.sql_index import (
    PreparedSQLLexicalRecord,
    SQLNativeProjectionStore,
    _acquire_schema_lock,
)

_MAX_NATIVE_TABLE_PREFIX_CHARS = 39
DEFAULT_QUERY_TIMEOUT_MS = 5000
MAX_QUERY_TIMEOUT_MS = 300_000
DEFAULT_MAX_QUERY_CHARS = 32_768
MAX_QUERY_CHARS = 262_144
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
MAX_CONNECT_TIMEOUT_SECONDS = 300
DEFAULT_OPERATION_TIMEOUT_MS = 300_000
MAX_OPERATION_TIMEOUT_MS = 3_600_000


def _validate_native_table_prefix(value: str) -> str:
    prefix = str(value).strip()
    if len(prefix.encode("ascii", errors="ignore")) > _MAX_NATIVE_TABLE_PREFIX_CHARS:
        raise ValueError(
            "native lexical table_name must be at most 39 ASCII characters so "
            "all derived PostgreSQL/openGauss identifiers remain within 63 bytes"
        )
    return prefix


class DatabaseNativeLexicalIndex(SQLNativeProjectionStore):
    """Shared projection lifecycle for database-native lexical engines."""

    _schema_version = 2

    def __init__(
        self,
        *,
        query_timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        operation_timeout_ms: int = DEFAULT_OPERATION_TIMEOUT_MS,
        **kwargs: Any,
    ) -> None:
        if "table_name" in kwargs:
            kwargs["table_name"] = _validate_native_table_prefix(kwargs["table_name"])
        self._query_timeout_ms = int(query_timeout_ms)
        if not 1 <= self._query_timeout_ms <= MAX_QUERY_TIMEOUT_MS:
            raise ValueError(
                f"query_timeout_ms must be between 1 and {MAX_QUERY_TIMEOUT_MS}"
            )
        self._max_query_chars = int(max_query_chars)
        if not 1 <= self._max_query_chars <= MAX_QUERY_CHARS:
            raise ValueError(
                f"max_query_chars must be between 1 and {MAX_QUERY_CHARS}"
            )
        self._connect_timeout_seconds = int(connect_timeout_seconds)
        if not 1 <= self._connect_timeout_seconds <= MAX_CONNECT_TIMEOUT_SECONDS:
            raise ValueError(
                "connect_timeout_seconds must be between 1 and "
                f"{MAX_CONNECT_TIMEOUT_SECONDS}"
            )
        self._operation_timeout_ms = int(operation_timeout_ms)
        if not 1 <= self._operation_timeout_ms <= MAX_OPERATION_TIMEOUT_MS:
            raise ValueError(
                f"operation_timeout_ms must be between 1 and {MAX_OPERATION_TIMEOUT_MS}"
            )
        shared_pool = kwargs.get("pool")
        if shared_pool is not None:
            actual_connect_timeout = getattr(
                shared_pool, "connect_timeout_seconds", None
            )
            actual_operation_timeout = getattr(
                shared_pool, "session_statement_timeout_ms", None
            )
            if (
                actual_connect_timeout != self._connect_timeout_seconds
                or actual_operation_timeout != self._operation_timeout_ms
            ):
                raise ValueError(
                    "shared native lexical pools must be created with matching "
                    "connect_timeout_seconds and session_statement_timeout_ms"
                )
        self._engine = "uninitialized"
        self._query_truncations = 0
        super().__init__(
            connect_timeout_seconds=self._connect_timeout_seconds,
            session_statement_timeout_ms=self._operation_timeout_ms,
            **kwargs,
        )

    @property
    def _schema_identity(self) -> str:
        return f"native:stored-vector-v1:{self._engine}:chars={self._max_document_chars}"

    def _prepare_record(
        self, record: LexicalRecord
    ) -> PreparedSQLLexicalRecord:
        if not record.account_id or not record.owner_space:
            raise ValueError("lexical records require account_id and owner_space")
        metadata = dict(record.metadata or {})
        metadata.update({"uri": record.uri, "level": record.level})
        source = record.text or ""
        content = source[: self._max_document_chars]
        if len(source) > len(content):
            with self._limit_metrics_lock:
                self._document_truncations += 1
        return PreparedSQLLexicalRecord(
            record=record,
            metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            content=content,
        )

    def _handle_schema_mismatch(self, cur, row) -> None:
        del cur
        persisted = f"version={row[0]}, identity={row[1]!r}"
        expected = f"version={self._schema_version}, identity={self._schema_identity!r}"
        raise RuntimeError(
            f"native lexical table {self._table!r} has incompatible schema "
            f"({persisted}; expected {expected}); configure a new versioned table_name, "
            "rebuild it, and cut over explicitly"
        )

    def health(self) -> dict[str, Any]:
        result = super().health()
        result.update(
            {
                "engine": self._engine,
                "query_timeout_ms": self._query_timeout_ms,
                "max_query_chars": self._max_query_chars,
                "connect_timeout_seconds": self._connect_timeout_seconds,
                "operation_timeout_ms": self._operation_timeout_ms,
                "max_query_terms": self._max_query_terms,
                "query_truncations": self._query_truncations,
                "schema_version": self._schema_version,
                "schema_identity": self._schema_identity,
            }
        )
        result["analyzer"] = "opengauss_bm25"
        if result.get("available"):
            result.update(self._native_projection_health())
            result["available"] = bool(
                result["schema_compatible"] and result["index_ready"]
            )
        return result

    def _native_projection_health(self) -> dict[str, Any]:
        expected_index = f"idx_{self._table}_native_bm25"
        expected_fragment = "using bm25 (content"
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT schema_version,analyzer_identity "
                    f"FROM {self._metadata_table} WHERE singleton=1"
                )
                metadata = cur.fetchone()
                cur.execute(
                    """
                    SELECT pg_get_indexdef(index_class.oid),idx.indisvalid,
                           idx.indisready,am.amname
                    FROM pg_class index_class
                    JOIN pg_namespace ns ON ns.oid=index_class.relnamespace
                    JOIN pg_index idx ON idx.indexrelid=index_class.oid
                    JOIN pg_class table_class ON table_class.oid=idx.indrelid
                    JOIN pg_am am ON am.oid=index_class.relam
                    WHERE ns.nspname=current_schema() AND table_class.relname=%s
                      AND index_class.relname=%s
                    """,
                    (self._docs_table, expected_index),
                )
                index_row = cur.fetchone()
            compatible = bool(
                metadata
                and int(metadata[0]) == self._schema_version
                and str(metadata[1]) == self._schema_identity
            )
            index_ready = bool(
                index_row
                and index_row[1]
                and index_row[2]
                and str(index_row[3]).casefold() == "bm25"
                and expected_fragment in str(index_row[0]).casefold()
            )
            return {
                "schema_compatible": compatible,
                "index_ready": index_ready,
                "index_name": expected_index,
            }
        except Exception as exc:
            return {
                "schema_compatible": False,
                "index_ready": False,
                "index_name": expected_index,
                "projection_error_type": type(exc).__name__,
            }
        finally:
            self._return(conn)

    def _bounded_query(self, query: str) -> str:
        bounded = query[: self._max_query_chars]
        tokens = self._analyzer.analyze_query(bounded)
        truncated = len(query) > len(bounded)
        if len(tokens) > self._max_query_terms:
            bounded = " ".join(tokens[: self._max_query_terms])[
                : self._max_query_chars
            ]
            truncated = True
        if truncated:
            with self._limit_metrics_lock:
                self._query_truncations += 1
        return bounded

    def _recover_failed_search(self, conn, exc: Exception) -> None:
        self._last_error = type(exc).__name__
        try:
            conn.rollback()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

class OpenGaussLexicalIndex(DatabaseNativeLexicalIndex):
    """openGauss BM25 provider backed by the native BM25 access method."""

    def __init__(
        self,
        *,
        bm25_topk: int | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        bm25_taat: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self._bm25_topk = None if bm25_topk is None else int(bm25_topk)
        self._bm25_k1 = None if bm25_k1 is None else float(bm25_k1)
        self._bm25_b = None if bm25_b is None else float(bm25_b)
        self._bm25_taat = bm25_taat
        if self._bm25_topk is not None and not 5 <= self._bm25_topk <= 200:
            raise ValueError("bm25_topk must be between 5 and 200")
        if self._bm25_k1 is not None and not 0 <= self._bm25_k1 <= 3:
            raise ValueError("bm25_k1 must be between 0 and 3")
        if self._bm25_b is not None and not 0 <= self._bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1")
        if self._bm25_taat is not None and not isinstance(self._bm25_taat, bool):
            raise ValueError("bm25_taat must be a boolean")
        super().__init__(**kwargs)

    def _ensure_schema(self) -> None:
        if not self._database_supports_bm25():
            raise RuntimeError(
                "openGauss native BM25 access method and <&> operator are required"
            )
        self._engine = "opengauss_bm25"
        SQLNativeProjectionStore._ensure_schema(self)
        self._create_bm25_index()
        projection_health = self._native_projection_health()
        if not (
            projection_health.get("schema_compatible")
            and projection_health.get("index_ready")
        ):
            raise RuntimeError(
                "openGauss native BM25 index failed post-create validation: "
                f"{projection_health}"
            )

    def _database_supports_bm25(self) -> bool:
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        EXISTS(
                            SELECT 1
                            FROM pg_opclass opc
                            JOIN pg_am am ON am.oid=opc.opcmethod
                            WHERE am.amname='bm25'
                              AND opc.opcintype='pg_catalog.text'::regtype
                              AND opc.opcdefault
                        ),
                        EXISTS(
                            SELECT 1
                            FROM pg_operator opr
                            WHERE opr.oprname='<&>'
                              AND opr.oprleft='pg_catalog.text'::regtype
                              AND opr.oprright='pg_catalog.text'::regtype
                              AND pg_operator_is_visible(opr.oid)
                        )
                    """
                )
                row = cur.fetchone()
                return bool(row and row[0] and row[1])
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def _create_bm25_index(self) -> None:
        index_name = f"idx_{self._table}_native_bm25"
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                _acquire_schema_lock(cur, self._table)
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self._docs_table} USING bm25(content)
                    """
                )
            conn.commit()
        except Exception:
            self._rollback_preserving(conn)
            raise
        finally:
            self._return(conn)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[LexicalHit]:
        filters = dict(filters or {})
        unknown = set(filters) - self.supported_filters
        if unknown:
            raise ValueError(f"unsupported lexical filters: {sorted(unknown)}")
        if not str(filters.get("account_id", "")):
            raise ValueError("lexical search requires an account_id filter")
        if top_k < 1 or not query.strip():
            return []
        query = self._bounded_query(query)

        started = time.monotonic()
        conn = self._borrow()
        try:
            with conn.cursor() as cur:
                self._configure_bm25_session(cur, top_k)
                where_sql, params = self._filter_where(filters)
                index_name = f"idx_{self._table}_native_bm25"
                cur.execute(
                    f"""
                    SELECT /*+ indexscan (d {index_name}) */
                           d.record_id,d.metadata_json,d.content <&> %s AS rank
                    FROM {self._docs_table} d
                    WHERE {where_sql}
                    ORDER BY d.content <&> %s DESC
                    LIMIT %s
                    """,
                    (query, *params, query, top_k),
                )
                rows = cur.fetchall()
            conn.rollback()
            hits = [
                LexicalHit(
                    str(record_id),
                    float(rank or 0.0),
                    self._loads_metadata(metadata_json),
                )
                for record_id, metadata_json, rank in rows
            ]
            self._last_error = None
            return hits
        except Exception as exc:
            self._recover_failed_search(conn, exc)
            raise
        finally:
            self._record_query(time.monotonic() - started)
            self._return(conn)

    def _configure_bm25_session(self, cur, top_k: int) -> None:
        settings = [("statement_timeout", str(self._query_timeout_ms))]
        configured_topk = self._bm25_topk or max(5, min(200, top_k))
        # openGauss expands beyond bm25_topk when LIMIT requests more rows.
        settings.append(("bm25_topk", str(max(configured_topk, min(200, top_k)))))
        if self._bm25_k1 is not None:
            settings.append(("bm25_k1", str(self._bm25_k1)))
        if self._bm25_b is not None:
            settings.append(("bm25_b", str(self._bm25_b)))
        if self._bm25_taat is not None:
            settings.append(("enable_bm25_taat", "on" if self._bm25_taat else "off"))
        placeholders = ",".join(["(%s,%s)"] * len(settings))
        cur.execute(
            f"""
            SELECT set_config(setting_name, setting_value, true)
            FROM (VALUES {placeholders}) AS settings(setting_name, setting_value)
            """,
            tuple(value for setting in settings for value in setting),
        )

    def health(self) -> dict[str, Any]:
        result = super().health()
        result.update(
            {
                "bm25_available": self._engine == "opengauss_bm25",
                "bm25_topk": self._bm25_topk,
                "bm25_k1": self._bm25_k1,
                "bm25_b": self._bm25_b,
                "bm25_taat": self._bm25_taat,
            }
        )
        return result
