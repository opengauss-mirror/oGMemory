"""Transactional PostgreSQL tool-result artifact store."""

from __future__ import annotations

import hashlib
import json
import re

from core.models import RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin

from .models import ArtifactMetadata
from .store_base import (
    ArtifactNotVisibleError,
    ArtifactStoreUnavailableError,
    ToolResultArtifactStore,
    artifact_reuse_matches,
    decode_utf8_range,
    metadata_visible,
    require_session_owner,
)

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    psycopg2 = None
    Json = None


_ARTIFACT_ID = re.compile(r"^tr_[a-f0-9]{16,64}(?:_[a-f0-9]{8,16})?$")


class SQLToolResultArtifactStore(PoolAdapterMixin, ToolResultArtifactStore):
    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
        dialect: str = "postgresql",
    ):
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for SQLToolResultArtifactStore")
        self._init_pool(pool, connection_string, pool_size, dialect)
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        finally:
            self._return_connection(conn)

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

    @staticmethod
    def _rollback_failed_read(conn) -> None:
        try:
            conn.rollback()
        except Exception:
            conn.close()

    @staticmethod
    def _metadata(raw) -> ArtifactMetadata:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return ArtifactMetadata.from_dict(raw)

    def write(self, metadata: ArtifactMetadata, output_text: str, ctx: RequestContext) -> bool:
        self.bind_context(ctx, session_id=metadata.session_id)
        require_session_owner(metadata.session_id, ctx)
        if not metadata_visible(metadata, metadata.session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        encoded = output_text.encode("utf-8")
        if (
            len(encoded) != metadata.original_bytes
            or hashlib.sha256(encoded).hexdigest() != metadata.sha256
        ):
            raise ValueError("artifact output does not match metadata")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                params = (
                    ctx.account_id,
                    metadata.session_id,
                    ctx.user_id,
                    ctx.agent_id,
                    metadata.artifact_id,
                    output_text,
                    Json(metadata.to_dict()),
                    metadata.sha256,
                    metadata.created_at,
                )
                inserted = self.sql_dialect.insert_tool_result_artifact(cur, params)
                if inserted:
                    conn.commit()
                    return False
                # INSERT waits for a concurrent unique-key winner. A new
                # READ COMMITTED statement can then observe and verify it.
                cur.execute(
                    """
                    SELECT metadata, output_text FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s AND artifact_id=%s
                    """,
                    (
                        ctx.account_id,
                        metadata.session_id,
                        ctx.user_id,
                        ctx.agent_id,
                        metadata.artifact_id,
                    ),
                )
                existing = cur.fetchone()
                if existing is None:
                    raise ArtifactStoreUnavailableError(
                        "concurrent artifact write could not be verified"
                    )
                existing_metadata = self._metadata(existing[0])
                if artifact_reuse_matches(
                    existing_metadata,
                    metadata,
                    existing[1],
                    output_text,
                ):
                    conn.rollback()
                    return True
                raise FileExistsError("artifact identity conflict")
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def read_all(
        self, artifact_id: str, session_id: str, ctx: RequestContext
    ) -> tuple[ArtifactMetadata, str]:
        self.bind_context(ctx, session_id=session_id)
        require_session_owner(session_id, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata, output_text FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s AND artifact_id=%s
                    """,
                    (ctx.account_id, session_id, ctx.user_id, ctx.agent_id, artifact_id),
                )
                row = cur.fetchone()
            conn.rollback()
        except Exception:
            self._rollback_failed_read(conn)
            raise
        finally:
            self._return_connection(conn)
        if not row:
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        metadata, output = self._metadata(row[0]), row[1]
        encoded = output.encode("utf-8")
        if (
            not metadata_visible(metadata, session_id, ctx)
            or len(encoded) != metadata.original_bytes
            or hashlib.sha256(encoded).hexdigest() != metadata.sha256
        ):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata, output

    def read_metadata(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
    ) -> ArtifactMetadata:
        self.bind_context(ctx, session_id=session_id)
        require_session_owner(session_id, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s AND artifact_id=%s
                    """,
                    (ctx.account_id, session_id, ctx.user_id, ctx.agent_id, artifact_id),
                )
                row = cur.fetchone()
            conn.rollback()
        except Exception:
            self._rollback_failed_read(conn)
            raise
        finally:
            self._return_connection(conn)
        if not row:
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        metadata = self._metadata(row[0])
        if not metadata_visible(metadata, session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata

    def read_range(self, artifact_id, session_id, ctx, offset_bytes, length_bytes):
        self.bind_context(ctx, session_id=session_id)
        require_session_owner(session_id, ctx)
        if offset_bytes < 0 or length_bytes < 1:
            raise ValueError("offset_bytes must be non-negative and length_bytes must be positive")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata,
                           substring(convert_to(output_text, 'UTF8') from %s for %s),
                           octet_length(convert_to(output_text, 'UTF8'))
                    FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s AND artifact_id=%s
                    """,
                    (
                        # Up to three leading continuation bytes may be skipped
                        # and three trailing bytes may finish the last codepoint.
                        offset_bytes + 1,
                        length_bytes + 6,
                        ctx.account_id,
                        session_id,
                        ctx.user_id,
                        ctx.agent_id,
                        artifact_id,
                    ),
                )
                row = cur.fetchone()
            conn.rollback()
        except Exception:
            self._rollback_failed_read(conn)
            raise
        finally:
            self._return_connection(conn)
        if not row:
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        metadata = self._metadata(row[0])
        if not metadata_visible(metadata, session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        total_bytes = int(row[2])
        base_offset = min(offset_bytes, total_bytes)
        raw = bytes(row[1] or b"") if base_offset < total_bytes else b""
        decoded = decode_utf8_range(raw, 0, length_bytes)
        decoded["actual_offset_bytes"] += base_offset
        decoded["next_offset_bytes"] += base_offset
        decoded["eof"] = decoded["next_offset_bytes"] >= total_bytes
        return {"metadata": metadata.to_dict(), "ref": metadata.ref, **decoded}

    def list(
        self, session_id: str, ctx: RequestContext, limit: int = 50, cursor: str | None = None
    ) -> dict:
        self.bind_context(ctx, session_id=session_id)
        require_session_owner(session_id, ctx)
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if cursor is not None and not _ARTIFACT_ID.fullmatch(cursor):
            raise ValueError("invalid cursor")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                if cursor is not None:
                    cur.execute(
                        """
                        SELECT 1 FROM tool_result_artifacts
                        WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                          AND owner_agent_id=%s AND artifact_id=%s
                        """,
                        (ctx.account_id, session_id, ctx.user_id, ctx.agent_id, cursor),
                    )
                    if cur.fetchone() is None:
                        raise ValueError("invalid cursor")
                cur.execute(
                    """
                    SELECT artifact_id, metadata FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s AND (%s IS NULL OR artifact_id < %s)
                    ORDER BY artifact_id DESC LIMIT %s
                    """,
                    (
                        ctx.account_id,
                        session_id,
                        ctx.user_id,
                        ctx.agent_id,
                        cursor,
                        cursor,
                        limit + 1,
                    ),
                )
                rows = cur.fetchall()
            conn.rollback()
        except Exception:
            self._rollback_failed_read(conn)
            raise
        finally:
            self._return_connection(conn)
        truncated = len(rows) > limit
        rows = rows[:limit]
        artifacts = []
        for _, raw in rows:
            metadata = self._metadata(raw)
            if metadata_visible(metadata, session_id, ctx):
                artifacts.append(metadata.to_dict() | {"ref": metadata.ref})
        return {
            "artifacts": artifacts,
            "truncated": truncated,
            "next_cursor": rows[-1][0] if truncated and rows else None,
        }

    def delete_session_artifacts(self, session_id: str, ctx: RequestContext) -> int:
        self.bind_context(ctx, session_id=session_id)
        require_session_owner(session_id, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM tool_result_artifacts
                    WHERE account_id=%s AND session_id=%s AND owner_user_id=%s
                      AND owner_agent_id=%s
                    """,
                    (ctx.account_id, session_id, ctx.user_id, ctx.agent_id),
                )
                count = cur.rowcount
            conn.commit()
            return count
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
