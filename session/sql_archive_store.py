"""PostgreSQL-backed SessionArchiveStore implementation.

Stores session archives in a ``session_archives`` table instead of
per-archive AGFS directories.  Same public interface as
``session.archive_store.SessionArchiveStore``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from core.models import RequestContext, Role
from fs.sql_adapter.pool import PoolAdapterMixin
from session.models import ArchiveEntry, ArchiveWriteResult

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    Json = None
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


def _metadata_from_db(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _is_merged_metadata(metadata: dict) -> bool:
    return str(metadata.get("archive_status", metadata.get("status", ""))).upper() == "MERGED"


class SQLSessionArchiveStore(PoolAdapterMixin):
    """PostgreSQL-backed session archive storage.

    Provides the same ``write_archive``, ``list_archives``,
    ``read_archive``, and ``read_archive_abstract`` methods as the
    AGFS-based ``SessionArchiveStore``, but backed by a single
    ``session_archives`` table.
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
                "psycopg2 is required for SQLSessionArchiveStore. "
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

    # ------------------------------------------------------------------
    # Public API (same signatures as SessionArchiveStore)
    # ------------------------------------------------------------------

    def write_archive(
        self,
        session_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
        archive_id: str | None = None,
        metadata: dict | None = None,
    ) -> ArchiveWriteResult:
        """Write an archive idempotently using the configured SQL dialect."""
        self.bind_context(ctx, session_id=session_id)
        if archive_id is None:
            archive_id = f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        uri = f"ctx://{ctx.account_id}/sessions/{session_id}/history/{archive_id}"
        now = datetime.now(UTC).isoformat()

        archive_metadata = dict(metadata or {})
        archive_metadata.update(
            {
                "archive_id": archive_id,
                "session_id": session_id,
                "created_at": now,
                "message_count": len(messages),
                "owner_user_id": ctx.user_id,
                "owner_agent_id": ctx.agent_id,
            }
        )

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                params = (
                    archive_id,
                    session_id,
                    ctx.account_id,
                    abstract or "",
                    overview or "",
                    Json(messages),
                    Json(archive_metadata),
                    now,
                    ctx.user_id,
                    ctx.agent_id,
                )
                self.sql_dialect.insert_session_archive(cur, params)
            conn.commit()
            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=True,
                created_at=now,
            )
        except Exception as exc:
            conn.rollback()
            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=False,
                error=str(exc),
            )
        finally:
            self._return_connection(conn)

    def list_archives(self, session_id: str, ctx: RequestContext) -> list[ArchiveEntry]:
        """List all archives for a session, newest first."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE account_id = %s AND session_id = %s
                      AND UPPER(COALESCE(
                            metadata->>'archive_status', metadata->>'status', ''
                          )) <> 'MERGED'
                    ORDER BY created_at DESC
                    """,
                    (ctx.account_id, session_id),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            conn.rollback()
            self._return_connection(conn)

        entries: list[ArchiveEntry] = []
        for row in rows:
            (
                archive_id,
                sess_id,
                abstract,
                overview,
                messages_raw,
                metadata_raw,
                created_at,
            ) = row

            meta = (
                json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
            )
            if _is_merged_metadata(meta):
                continue

            entries.append(
                ArchiveEntry(
                    archive_id=archive_id,
                    session_id=sess_id,
                    overview=overview or "",
                    abstract=abstract or "",
                    messages=[],  # Don't include full messages in list
                    created_at=str(created_at) if created_at else "",
                    metadata=meta,
                )
            )
        return entries

    def list_archives_since(
        self,
        since: datetime,
        ctx: RequestContext,
        limit: int = 50,
    ) -> list[ArchiveEntry]:
        """List archives created since a given timestamp, across all sessions.

        Args:
            since: Only return archives created after this timestamp
            ctx: RequestContext for account_id
            limit: Maximum number of archives to return (default 50)

        Returns:
            List of ArchiveEntry with messages included, ordered by created_at DESC
        """
        self.bind_context(ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE account_id = %s AND created_at > %s
                      AND UPPER(COALESCE(
                            metadata->>'archive_status', metadata->>'status', ''
                          )) <> 'MERGED'
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (ctx.account_id, since, limit),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            conn.rollback()
            self._return_connection(conn)

        entries: list[ArchiveEntry] = []
        for row in rows:
            (
                archive_id,
                sess_id,
                abstract,
                overview,
                messages_raw,
                metadata_raw,
                created_at,
            ) = row

            # Parse messages
            if isinstance(messages_raw, str):
                messages = json.loads(messages_raw)
            elif messages_raw is not None:
                messages = list(messages_raw)
            else:
                messages = []

            meta = (
                json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
            )
            if _is_merged_metadata(meta):
                continue

            entries.append(
                ArchiveEntry(
                    archive_id=archive_id,
                    session_id=sess_id,
                    overview=overview or "",
                    abstract=abstract or "",
                    messages=messages,
                    created_at=str(created_at) if created_at else "",
                    metadata=meta,
                )
            )
        return entries

    def read_archive(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> ArchiveEntry | None:
        """Read a full session archive including messages."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                row = cur.fetchone()
        except Exception:
            return None
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        return self._entry_from_row(row)

    @staticmethod
    def _entry_from_row(row) -> ArchiveEntry:
        (
            aid,
            sess_id,
            abstract,
            overview,
            messages_raw,
            metadata_raw,
            created_at,
        ) = row

        # Parse messages
        if isinstance(messages_raw, str):
            messages = json.loads(messages_raw)
        elif messages_raw is not None:
            messages = list(messages_raw)
        else:
            messages = []

        meta = json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})

        return ArchiveEntry(
            archive_id=aid,
            session_id=sess_id,
            overview=overview or "",
            abstract=abstract or "",
            messages=messages,
            created_at=str(created_at) if created_at else "",
            metadata=meta,
        )

    def read_archive_by_id(self, archive_id: str, ctx: RequestContext) -> ArchiveEntry | None:
        """Resolve a unique archive within the requested account and owner."""
        if not (ctx.user_id or ctx.agent_id):
            return None
        self.bind_context(ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT archive_id, session_id, abstract, overview, messages, metadata, created_at "
                    "FROM session_archives WHERE account_id = %s AND archive_id = %s "
                    "AND owner_user_id = %s AND (%s = '' OR owner_agent_id = %s) LIMIT 2",
                    (ctx.account_id, archive_id, ctx.user_id, ctx.agent_id, ctx.agent_id),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)
        return self._entry_from_row(rows[0]) if len(rows) == 1 else None

    def read_archive_abstract(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> str | None:
        """Read only the abstract from an archive."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT abstract, metadata FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                row = cur.fetchone()
        except Exception:
            return None
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None
        return row[0] or None

    def delete_archive(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> bool:
        """Delete a single archive row for the current account and session."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                if cur.rowcount != 1:
                    conn.rollback()
                    return False
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def delete_session_archives(self, session_id: str, ctx: RequestContext) -> int:
        """Delete exact-owner rows plus ADMIN/ROOT-visible ownerless legacy rows."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                role = ctx.role if isinstance(ctx.role, Role) else Role(str(ctx.role).upper())
                if role in (Role.ADMIN, Role.ROOT):
                    cur.execute(
                        """
                        DELETE FROM session_archives
                        WHERE account_id = %s AND session_id = %s
                          AND (
                            (owner_user_id = %s AND owner_agent_id = %s)
                            OR (
                              COALESCE(owner_user_id, '') = ''
                              AND COALESCE(owner_agent_id, '') = ''
                            )
                          )
                        """,
                        (ctx.account_id, session_id, ctx.user_id, ctx.agent_id),
                    )
                else:
                    cur.execute(
                        """
                        DELETE FROM session_archives
                        WHERE account_id = %s AND session_id = %s
                          AND owner_user_id = %s AND owner_agent_id = %s
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

    def get_session_owner(self, session_id: str, ctx: RequestContext) -> tuple[str, str] | None:
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT COALESCE(owner_user_id, ''), COALESCE(owner_agent_id, '')
                    FROM session_archives
                    WHERE account_id = %s AND session_id = %s
                    """,
                    (ctx.account_id, session_id),
                )
                rows = cur.fetchall()
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        owners = {tuple(row) for row in rows}
        owners.discard(("", ""))
        if len(owners) > 1:
            raise PermissionError("conflicting durable session owners")
        owner = next(iter(owners)) if owners else None
        if owner is not None and not all(owner):
            raise PermissionError("invalid durable session owner metadata")
        return owner

    def unmark_archive_merged(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
        merged_into: str,
    ) -> bool:
        """Undo a merge marker when a merge operation rolls back."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    return False

                metadata = _metadata_from_db(row[0])
                if metadata.get("merged_into") != merged_into:
                    conn.rollback()
                    return False

                metadata.pop("archive_status", None)
                if str(metadata.get("status", "")).upper() == "MERGED":
                    metadata.pop("status", None)
                metadata.pop("merged_into", None)
                metadata.pop("merged_at", None)

                cur.execute(
                    """
                    UPDATE session_archives
                    SET metadata = %s
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (Json(metadata), archive_id, session_id, ctx.account_id),
                )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def mark_archive_merged(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
        merged_into: str,
    ) -> bool:
        """Mark an archive row as merged into another archive."""
        self.bind_context(ctx, session_id=session_id)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            marker = {
                "archive_status": "MERGED",
                "merged_into": merged_into,
                "merged_at": datetime.now(UTC).isoformat(),
            }
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE session_archives
                    SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (Json(marker), archive_id, session_id, ctx.account_id),
                )
                if cur.rowcount != 1:
                    conn.rollback()
                    return False
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)
