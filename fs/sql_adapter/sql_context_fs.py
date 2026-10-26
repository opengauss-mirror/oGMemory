"""Dialect-aware SQL implementation of ContextFS protocol.

Stores all node data (content, abstract, overview, relations, metadata)
in a single `context_nodes` table. Dialect-specific atomic upserts replace
the AGFS atomic 4-step write order.

Follows the same psycopg2 connection pool pattern as OpenGaussVectorIndex.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import unquote

from core.enums import ContextType, EventType, NodeStatus
from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    NodeBrokenError,
    NodeNotFoundError,
)
from core.models import ContextNode, OutboxEvent, RelationEdge, RequestContext
from core.resource_identity import (
    CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,
    RootBindingError,
    reserve_workspace_root_binding_state,
    resolve_workspace_root_binding_state,
    validate_resource_uri_slug,
)
from fs.access_control import check_uri_access
from fs.sql_adapter.pool import PoolAdapterMixin, SharedConnectionPool

# ---------------------------------------------------------------------------
# URI parsing (independent copy — no dependency on agfs_adapter)
# ---------------------------------------------------------------------------

_SKILL_URI_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/"
    r"(?P<skill_root>skills)/(?P<skill_source>generated|imported)/"
    r"(?P<skill_name>[^/]+)$"
)
_LEGACY_SKILL_URI_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/"
    r"(?P<skill_root>skills)/(?P<skill_name>[^/]+)$"
)
_SKILLS_DIR_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/"
    r"(?P<skill_root>skills)(?:/(?P<skill_source>generated|imported))?/?$"
)
_RESOURCE_URI_PATTERN = re.compile(r"^ctx://(?P<account>[^/]+)/resources/(?P<slug>[^/]+)$")
_RESOURCES_DIR_PATTERN = re.compile(r"^ctx://(?P<account>[^/]+)/resources/?$")
_PROFILE_URI_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/users/(?P<owner_id>[^/]+)/memories/profile$"
)
_MEMORY_URI_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/"
    r"memories/(?P<category>[^/]+)/(?P<slug>[^/]+)$"
)
_CATEGORY_DIR_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/"
    r"memories/(?P<category>[^/]+)/?$"
)
_MEMORIES_ROOT_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/memories/?$"
)
_SESSION_ARCHIVE_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/(?P<archive_id>[^/]+)$"
)
_SESSION_HISTORY_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/?$"
)
_SESSION_STATE_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/state(?:\.json)?$"
)


def parse_uri(uri: str) -> dict:
    """Parse a ContextEngine URI into its components."""
    match = _SKILL_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "agents",
            "owner_id": g["owner_id"],
            "category": f"{g['skill_root']}/{g['skill_source']}",
            "slug": unquote(g["skill_name"]),
        }

    match = _LEGACY_SKILL_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "agents",
            "owner_id": g["owner_id"],
            "category": g["skill_root"],
            "slug": unquote(g["skill_name"]),
        }

    stripped = uri.rstrip("/")
    match = _SKILLS_DIR_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "agents",
            "owner_id": g["owner_id"],
            "category": (
                f"{g['skill_root']}/{g['skill_source']}"
                if g.get("skill_source")
                else g["skill_root"]
            ),
            "slug": "",
        }

    match = _RESOURCE_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "resources",
            "owner_id": "",
            "category": "resource",
            "slug": validate_resource_uri_slug(g["slug"]),
        }

    match = _RESOURCES_DIR_PATTERN.match(stripped)
    if match:
        return {
            "account": match.group("account"),
            "owner_type": "resources",
            "owner_id": "",
            "category": "resource",
            "slug": "",
        }

    match = _PROFILE_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "users",
            "owner_id": g["owner_id"],
            "category": "profile",
            "slug": "profile",
        }

    match = _MEMORY_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": g["owner_type"],
            "owner_id": g["owner_id"],
            "category": g["category"],
            "slug": unquote(g["slug"]),
        }

    match = _CATEGORY_DIR_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": g["owner_type"],
            "owner_id": g["owner_id"],
            "category": g["category"],
            "slug": "",
        }

    match = _MEMORIES_ROOT_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": g["owner_type"],
            "owner_id": g["owner_id"],
            "category": "",
            "slug": "",
        }

    match = _SESSION_ARCHIVE_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "sessions",
            "owner_id": g["session_id"],
            "category": "history",
            "slug": g["archive_id"],
        }

    match = _SESSION_HISTORY_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "sessions",
            "owner_id": g["session_id"],
            "category": "history",
            "slug": "",
        }

    match = _SESSION_STATE_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {
            "account": g["account"],
            "owner_type": "sessions",
            "owner_id": g["session_id"],
            "category": "state",
            "slug": "state",
        }

    raise ValueError(f"Invalid URI format: {uri}")


try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None
    Json = None

logger = logging.getLogger(__name__)

_RESOURCE_MUTATION_LOCK_NAMESPACE = "ogmemory:sql:code-resource-mutation:v1"

# Tool-output retention holds a session-level advisory lock while it performs
# several independent ContextFS operations.  Keep those extra sessions
# bounded, and never wait indefinitely for either a connection slot or the
# database lock.  These values intentionally mirror the bounded URI lock used
# by ``SQLOutboxStore.processing_lock``.
_TOOL_OUTPUT_LOCK_CONNECT_TIMEOUT_SECONDS = 5
_TOOL_OUTPUT_LOCK_TIMEOUT_SECONDS = 5.0
_MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS = 120.0
_TOOL_OUTPUT_LOCK_POLL_SECONDS = 0.05
_TOOL_OUTPUT_LOCK_MAX_CONNECTIONS = 8
_TOOL_OUTPUT_LOCK_CONNECTION_SLOTS = threading.BoundedSemaphore(_TOOL_OUTPUT_LOCK_MAX_CONNECTIONS)
_MEMORY_IDENTITY_LOCK_STATE = threading.local()


def _resource_mutation_lock_key(account_id: str, uri: str) -> int:
    """Return the stable signed BIGINT used to serialize resource mutations."""

    normalized_account = str(account_id).strip()
    normalized_uri = str(uri).rstrip("/")
    payload = "\0".join((_RESOURCE_MUTATION_LOCK_NAMESPACE, normalized_account, normalized_uri))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class SQLContextFS(PoolAdapterMixin):
    """SQL-backed implementation of ContextFS.

    Each ContextNode maps to a single row in ``context_nodes``.
    The atomic 4-step write from AGFS is replaced by one SQL transaction.

    Usage::

        fs = SQLContextFS("host=localhost dbname=ogmemory")
        fs.write_node(node, ctx)
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool: SharedConnectionPool | None = None,
        dialect: str = "postgresql",
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLContextFS. Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Connection pool (same pattern as OpenGaussVectorIndex)
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

    def recovery_account_ids(self):
        """Read the content-free scheduler directory before binding tenant RLS."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', '30000', true)")
                cur.execute("SELECT account_id FROM extraction_work_accounts ORDER BY account_id")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.rollback()
            self._return_connection(conn)

    def find_archive_uri(self, archive_id: str, ctx: RequestContext) -> str | None:
        """Resolve a unique canonical archive under the caller's account and owner."""
        if not (ctx.user_id or ctx.agent_id):
            return None
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT uri FROM context_nodes WHERE account_id = %s AND category = 'history' "
                    "AND status = 'ACTIVE' AND metadata->>'archive_id' = %s "
                    "AND metadata->>'owner_user_id' = %s "
                    "AND (%s = '' OR metadata->>'owner_agent_id' = %s) LIMIT 2",
                    (ctx.account_id, archive_id, ctx.user_id, ctx.agent_id, ctx.agent_id),
                )
                rows = cur.fetchall()
            return rows[0][0] if len(rows) == 1 else None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def scan_session_recovery(self, account_id, *, cursor=None, limit=32):
        """Page through canonical session metadata for the service recovery worker."""
        limit = max(1, min(int(limit), 50))
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', '30000', true)")
                cur.execute(
                    "SELECT uri, owner_space, metadata, version, updated_at FROM context_nodes "
                    "WHERE account_id=%s AND category='state' AND status='ACTIVE' AND uri>%s "
                    "ORDER BY uri LIMIT %s",
                    (account_id, cursor or "", limit),
                )
                rows = [{"uri": row[0], "owner_space": row[1],
                         "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                         "version": row[3], "updated_at": str(row[4])}
                        for row in cur.fetchall()]
            return rows, rows[-1]["uri"] if len(rows) == limit else None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def read_code_workspace_root_binding(
        self,
        *,
        ctx: RequestContext,
        root_slug: str | None = None,
        root_fingerprint: str | None = None,
    ) -> tuple[str, str, str] | None:
        """Read one protected account-wide code-workspace root binding."""

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state FROM code_workspace_sync_roots "
                    "WHERE account_id = %s AND root_id = %s",
                    (ctx.account_id, CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID),
                )
                row = cur.fetchone()
            state = dict(row[0]) if row is not None and isinstance(row[0], Mapping) else None
            if row is not None and state is None:
                raise RootBindingError("workspace root binding state is invalid")
            return resolve_workspace_root_binding_state(
                state,
                account_id=ctx.account_id,
                root_slug=root_slug,
                root_fingerprint=root_fingerprint,
            )
        finally:
            conn.rollback()
            self._return_connection(conn)

    def reserve_code_workspace_root_binding(
        self,
        *,
        ctx: RequestContext,
        root_slug: str,
        root_fingerprint: str,
    ) -> tuple[str, str, str]:
        """Serialize one account-wide binding transition under a SQL row lock."""

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                self.sql_dialect.ensure_workspace_root(
                    cur,
                    ctx.account_id,
                    CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,
                )
                cur.execute(
                    "SELECT state FROM code_workspace_sync_roots "
                    "WHERE account_id = %s AND root_id = %s FOR UPDATE",
                    (ctx.account_id, CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID),
                )
                row = cur.fetchone()
                if row is None or not isinstance(row[0], Mapping):
                    raise RootBindingError("workspace root binding state is invalid")
                updated = reserve_workspace_root_binding_state(
                    dict(row[0]),
                    account_id=ctx.account_id,
                    root_slug=root_slug,
                    root_fingerprint=root_fingerprint,
                )
                cur.execute(
                    "UPDATE code_workspace_sync_roots "
                    "SET state = %s, updated_at = NOW() "
                    "WHERE account_id = %s AND root_id = %s",
                    (Json(updated), ctx.account_id, CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        binding = resolve_workspace_root_binding_state(
            updated,
            account_id=ctx.account_id,
            root_fingerprint=root_fingerprint,
        )
        if binding is None:
            raise RootBindingError("workspace root binding state is invalid")
        return binding

    # ------------------------------------------------------------------
    # Access control (reuses parse_uri from agfs_adapter)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_accessible(uri: str, ctx: RequestContext) -> bool:
        """Check URI accessibility with visible_owner_spaces support (SQL mode)."""
        try:
            components = parse_uri(uri)
            return check_uri_access(ctx, components, strict_mode=False)
        except ValueError:
            return False

    @classmethod
    def _ensure_accessible(cls, uri: str, ctx: RequestContext) -> None:
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            guard.check_context(ctx)
        if not cls._is_accessible(uri, ctx):
            try:
                components = parse_uri(uri)
                owner_id = components.get("owner_id", "")
                if owner_id:
                    raise AccessDeniedError(
                        uri,
                        ctx.account_id,
                        f"owner mismatch: URI owner={owner_id}, request user={ctx.user_id}",
                    )
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    f"URI belongs to account '{components['account']}'",
                )
            except ValueError:
                raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format")

    def _extract_account_id(self, uri: str) -> str:
        """Extract account_id from URI for row-level storage."""
        components = parse_uri(uri)
        return components["account"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_row_data(self, node: ContextNode) -> tuple[list, dict, int, str]:
        """Prepare relations JSON, metadata dict, new version, and timestamp."""
        now = datetime.now(UTC).isoformat()
        relations = node.metadata.get("_relations", [])
        relations_json = [
            {
                "from_uri": e.from_uri,
                "to_uri": e.to_uri,
                "relation_type": e.relation_type,
                "weight": e.weight,
                "reason": e.reason,
            }
            for e in relations
        ]

        extra_meta = {k: v for k, v in node.metadata.items() if not k.startswith("_")}
        extra_meta.setdefault("created_at", now)
        extra_meta["updated_at"] = now

        expected_version = node.metadata.get(
            "_expected_version", node.metadata.get("expected_version")
        )
        create_only = node.metadata.get("_expected_version") == 0
        if create_only:
            new_version = 1
        elif expected_version is not None:
            new_version = expected_version + 1
        else:
            new_version = node.metadata.get("version", 1)
        extra_meta["version"] = new_version

        return relations_json, extra_meta, new_version, now

    @staticmethod
    def _is_code_resource_node(node: ContextNode) -> bool:
        normalized_uri = str(node.uri).rstrip("/")
        return (
            node.context_type == ContextType.RESOURCE.value
            and node.metadata.get("resource_type") == "code"
            and _RESOURCE_URI_PATTERN.fullmatch(normalized_uri) is not None
        )

    def _acquire_resource_mutation_lock(self, cur, account_id: str, uri: str) -> None:
        """Serialize writes and durable deletes for one account-scoped resource URI."""

        cur.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (_resource_mutation_lock_key(account_id, uri),),
        )

    @contextmanager
    def session_mutation_lock(self, uri: str, ctx: RequestContext, *, timeout_seconds=1.0):
        """Fence session mutations on the same SQL connection as their writes."""
        from session.session_state import SessionState
        if uri != SessionState.state_uri(ctx.account_id, ctx.session_id):
            raise ValueError("session mutation lock requires its canonical state URI")
        self._ensure_accessible(uri, ctx)
        with self.configured_session_mutation_lock(
            self._connection_string, ctx, timeout_seconds=timeout_seconds, dialect=self._sql_dialect,
        ) as guard:
            yield guard

    def requires_thread_bound_access(self):
        """Native extraction must keep lazy reads on its owning SQL thread."""
        from fs.sql_adapter.session_mutation import active_session_guard
        return active_session_guard() is not None

    @staticmethod
    @contextmanager
    def configured_session_mutation_lock(dsn, ctx, *, timeout_seconds=1.0, dialect="postgresql"):
        """Acquire before constructing stores, which otherwise borrow from their pool."""
        from fs.sql_adapter.session_mutation import session_mutation_scope
        from session.session_state import SessionState
        uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
        with session_mutation_scope(dsn, ctx, _resource_mutation_lock_key(ctx.account_id, uri),
                                    _TOOL_OUTPUT_LOCK_CONNECTION_SLOTS, timeout_seconds,
                                    dialect=dialect) as guard:
            yield guard

    @contextmanager
    def tool_output_mutation_lock(
        self, uri: str, ctx: RequestContext, *, timeout_seconds: float = _TOOL_OUTPUT_LOCK_TIMEOUT_SECONDS,
    ):
        """Hold a shared account/Agent lock for Tool Output retention.

        Tool-output repository operations use separate ContextFS calls, so a
        transaction-scoped lock would be released before the write or prune
        completed.  A dedicated session-level lock keeps the whole repository
        mutation serialized without consuming a pooled connection.  The
        dedicated sessions are bounded per process and use a try-lock/deadline
        loop so a stalled holder cannot make later filter requests wait
        forever.  A timeout is intentionally surfaced to the caller; the HTTP
        hook treats that failure as a request to preserve the native output.
        """

        self._ensure_accessible(uri, ctx)
        from fs.sql_adapter.session_mutation import active_session_guard
        guard = active_session_guard()
        if guard is not None:
            with guard.identity_lock(
                _resource_mutation_lock_key(ctx.account_id, str(uri).rstrip("/")), ctx,
                min(timeout_seconds, 1.0),
            ):
                yield
            return
        connect = getattr(psycopg2, "connect", None)
        if not callable(connect):
            raise RuntimeError("PostgreSQL advisory locking is unavailable")

        deadline = time.monotonic() + timeout_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not _TOOL_OUTPUT_LOCK_CONNECTION_SLOTS.acquire(timeout=remaining):
            raise TimeoutError("timed out waiting for a SQL tool-output lock connection")

        conn = None
        lock_key = _resource_mutation_lock_key(
            ctx.account_id,
            str(uri).rstrip("/"),
        )
        acquired = False
        try:
            remaining = deadline - time.monotonic()
            connect_timeout = min(
                _TOOL_OUTPUT_LOCK_CONNECT_TIMEOUT_SECONDS,
                int(remaining),
            )
            if connect_timeout < 1:
                raise TimeoutError("timed out before opening a SQL tool-output lock connection")
            conn = connect(
                self._connection_string,
                connect_timeout=connect_timeout,
            )
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out acquiring SQL tool-output mutation lock")
                    cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
                    row = cur.fetchone()
                    acquired = bool(row and row[0])
                    if acquired:
                        break
                    time.sleep(min(_TOOL_OUTPUT_LOCK_POLL_SECONDS, remaining))
            yield conn
        finally:
            try:
                if acquired and conn is not None:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                    except Exception:
                        # Closing the dedicated session below still releases a
                        # session-level advisory lock.  Do not mask the
                        # mutation or timeout error with an unlock failure.
                        logger.warning(
                            "Failed to release SQL tool-output mutation lock",
                            exc_info=True,
                        )
            finally:
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        logger.debug(
                            "Failed to rollback SQL tool-output lock session",
                            exc_info=True,
                        )
                    try:
                        conn.close()
                    except Exception:
                        logger.debug(
                            "Failed to close SQL tool-output lock session",
                            exc_info=True,
                        )
                _TOOL_OUTPUT_LOCK_CONNECTION_SLOTS.release()

    @contextmanager
    def memory_identity_lock(self, directory: str, ctx: RequestContext):
        from fs.sql_adapter.session_mutation import SQLSessionGuard, active_session_guard
        active = active_session_guard()
        guard = active or getattr(_MEMORY_IDENTITY_LOCK_STATE, "guard", None)
        key = _resource_mutation_lock_key(ctx.account_id, directory)
        if guard is not None:
            self._ensure_accessible(directory, ctx)
            guard.check_database(self._connection_string, self.sql_dialect.name)
            with guard.identity_lock(key, ctx, 1.0 if active else _MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS):
                yield
            return
        with self.tool_output_mutation_lock(
            directory, ctx, timeout_seconds=_MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS,
        ) as conn:
            conn.rollback()
            # Reuse the lock session while ordinary store operations keep their own transactions.
            _MEMORY_IDENTITY_LOCK_STATE.guard = SQLSessionGuard(
                conn, self._connection_string, ctx, key, self.sql_dialect.name)
            try:
                yield
            finally:
                _MEMORY_IDENTITY_LOCK_STATE.guard = None

    def _guard_code_resource_write(
        self,
        cur,
        node: ContextNode,
        account_id: str,
    ) -> None:
        """Reject a code-resource write while its durable DELETE remains live."""

        if not self._is_code_resource_node(node):
            return

        normalized_uri = str(node.uri).rstrip("/")
        self._acquire_resource_mutation_lock(cur, account_id, normalized_uri)
        cur.execute(
            "SELECT 1 FROM outbox_events "
            "WHERE account_id = %s AND uri = %s AND event_type = %s "
            "AND status IN ('PENDING', 'PROCESSING', 'FAILED') LIMIT 1",
            (account_id, normalized_uri, EventType.DELETE_CONTEXT.value),
        )
        if cur.fetchone() is not None:
            raise RuntimeError("code resource URI is pending durable deletion")

    # ------------------------------------------------------------------
    # ContextFS Protocol implementation
    # ------------------------------------------------------------------

    def _execute_write(
        self,
        cur,
        node,
        account_id,
        expected_version,
        create_only,
        relations_json,
        extra_meta,
        new_version,
        now,
    ):
        """Execute the INSERT/UPDATE SQL for a node write on the given cursor.

        Shared by write_node and write_node_with_outbox to avoid duplication.
        """
        if node.category == "state":
            self.sql_dialect.ensure_extraction_work_account(cur, account_id)
        if expected_version is not None and not create_only:
            cur.execute(
                """
                UPDATE context_nodes SET
                    status       = 'ACTIVE',
                    content      = %s,
                    abstract     = %s,
                    overview     = %s,
                    relations    = %s,
                    metadata     = %s,
                    version      = version + 1,
                    updated_at   = NOW()
                WHERE uri = %s AND version = %s
                """,
                (
                    node.content,
                    node.abstract,
                    node.overview,
                    Json(relations_json),
                    Json(extra_meta),
                    node.uri,
                    expected_version,
                ),
            )
            if cur.rowcount == 0:
                cur.execute(
                    "SELECT version FROM context_nodes WHERE uri = %s",
                    (node.uri,),
                )
                row = cur.fetchone()
                actual = row[0] if row else 0
                raise ConcurrentModificationError(node.uri, expected_version, actual)
            return

        values = (
            node.uri,
            account_id,
            node.owner_space,
            node.category,
            node.context_type,
            node.level,
            node.content,
            node.abstract,
            node.overview,
            Json(relations_json),
            Json(extra_meta),
            1 if create_only else new_version,
            extra_meta.get("created_at", now),
        )
        written = self.sql_dialect.write_context_node(cur, values, create_only=create_only)
        if not written:
            cur.execute(
                "SELECT version FROM context_nodes WHERE uri = %s",
                (node.uri,),
            )
            row = cur.fetchone()
            raise ConcurrentModificationError(node.uri, 0, row[0] if row else 0)

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node — atomic upsert with optimistic locking.

        For merge operations (expected_version present), uses a single
        UPDATE … WHERE version = expected_version to avoid TOCTOU races.

        Initial-create semantics are delegated to the configured dialect.

        Args:
            node: ContextNode to write.
            ctx:  RequestContext for access control.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            ConcurrentModificationError: On optimistic lock failure.
            RuntimeError: If a code-resource DELETE remains unfinished.
        """
        self._ensure_accessible(node.uri, ctx)
        if self._is_code_resource_node(node):
            raise RuntimeError("code resource writes require write_node_with_outbox")
        account_id = self._extract_account_id(node.uri)
        expected_version = node.metadata.get(
            "_expected_version", node.metadata.get("expected_version")
        )
        create_only = node.metadata.get("_expected_version") == 0
        relations_json, extra_meta, new_version, now = self._build_row_data(node)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._guard_code_resource_write(cur, node, account_id)
                self._execute_write(
                    cur,
                    node,
                    account_id,
                    expected_version,
                    create_only,
                    relations_json,
                    extra_meta,
                    new_version,
                    now,
                )
            conn.commit()
        except ConcurrentModificationError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def write_node_with_outbox(
        self, node: ContextNode, ctx: RequestContext, outbox_event: OutboxEvent | None = None
    ) -> None:
        """Write a node and register an outbox event in a single transaction.

        Both the context_nodes INSERT/UPDATE and the outbox_events INSERT
        execute on the same connection within one commit, guaranteeing
        atomicity: either both succeed or both roll back.

        Args:
            node: ContextNode to write.
            ctx:  RequestContext for access control.
            outbox_event: Pre-built OutboxEvent (from SQLOutboxStore.build_write_event).
                         If None, only the node is written (delegates to write_node).

        Raises:
            AccessDeniedError: If account_id doesn't match.
            ConcurrentModificationError: On optimistic lock failure.
            RuntimeError: If a code-resource DELETE remains unfinished.
        """
        if outbox_event is None:
            return self.write_node(node, ctx)

        self._ensure_accessible(node.uri, ctx)
        account_id = self._extract_account_id(node.uri)
        expected_version = node.metadata.get(
            "_expected_version", node.metadata.get("expected_version")
        )
        create_only = node.metadata.get("_expected_version") == 0
        relations_json, extra_meta, new_version, now = self._build_row_data(node)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Step 1: Serialize against durable deletion and write the node.
                self._guard_code_resource_write(cur, node, account_id)
                self._execute_write(
                    cur,
                    node,
                    account_id,
                    expected_version,
                    create_only,
                    relations_json,
                    extra_meta,
                    new_version,
                    now,
                )

                # Step 2: Register outbox event in same transaction.
                # Wake capable outbox workers for prompt processing.
                cur.execute(
                    """
                    INSERT INTO outbox_events
                        (event_id, event_type, uri, account_id, payload,
                         status, retry_count, created_at, next_retry_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        outbox_event.event_id,
                        outbox_event.event_type,
                        outbox_event.uri,
                        account_id,
                        Json(outbox_event.payload),
                        outbox_event.status,
                        outbox_event.retry_count,
                        outbox_event.created_at or datetime.now(UTC).isoformat(),
                        None,
                    ),
                )
                self.sql_dialect.notify_outbox(
                    cur,
                    "ogmem_outbox",
                    json.dumps(
                        {
                            "event_id": outbox_event.event_id,
                            "account_id": account_id,
                            "event_type": outbox_event.event_type,
                        },
                        ensure_ascii=False,
                    ),
                )
            conn.commit()
        except ConcurrentModificationError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from PostgreSQL.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist or status != ACTIVE.
            NodeBrokenError: If node is in BROKEN state.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                return self.read_node_with_cursor(cur, uri, ctx)
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

    @classmethod
    def read_node_with_cursor(cls, cursor, uri: str, ctx: RequestContext) -> ContextNode:
        """Read canonical evidence inside the caller's tenant-bound transaction."""
        cls._ensure_accessible(uri, ctx)
        cursor.execute(
            "SELECT uri, context_type, category, level, owner_space, "
            "abstract, overview, content, relations, metadata, "
            "status, created_at, updated_at, version "
            "FROM context_nodes WHERE account_id=%s AND uri=%s",
            (ctx.account_id, uri),
        )
        row = cursor.fetchone()
        if row is None:
            raise NodeNotFoundError(uri)

        (
            uri_val,
            context_type,
            category,
            level,
            owner_space,
            abstract,
            overview,
            content,
            relations_raw,
            metadata_raw,
            status,
            created_at,
            updated_at,
            version,
        ) = row

        if status == NodeStatus.BROKEN.value:
            raise NodeBrokenError(uri)
        if status != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        # Parse relations
        relations = []
        if relations_raw:
            rel_list = (
                json.loads(relations_raw) if isinstance(relations_raw, str) else relations_raw
            )
            relations = [
                RelationEdge(
                    from_uri=r["from_uri"],
                    to_uri=r["to_uri"],
                    relation_type=r["relation_type"],
                    weight=r["weight"],
                    reason=r["reason"],
                )
                for r in rel_list
            ]

        # Parse metadata
        meta = json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
        meta.setdefault("created_at", str(created_at) if created_at else "")
        meta.setdefault("updated_at", str(updated_at) if updated_at else "")
        meta.setdefault("version", version)
        meta["_relations"] = relations

        return ContextNode(
            uri=uri_val,
            context_type=context_type,
            category=category,
            level=level,
            owner_space=owner_space,
            abstract=abstract or "",
            overview=overview or "",
            content=content or "",
            metadata=meta,
        )

    def read_node_fields(self, uri: str, ctx: RequestContext, level: str) -> ContextNode:
        """Read only fields needed by MCP read_memory(level=...)."""

        if level not in {"abstract", "overview", "content", "all"}:
            raise ValueError("level must be one of: abstract, overview, content, all")
        self._ensure_accessible(uri, ctx)

        abstract_expr = "abstract" if level in {"abstract", "all"} else "'' AS abstract"
        overview_expr = "overview" if level in {"overview", "all"} else "'' AS overview"
        content_expr = "content" if level in {"content", "all"} else "'' AS content"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT uri, context_type, category, level, owner_space,
                           {abstract_expr}, {overview_expr}, {content_expr},
                           metadata, status, created_at, updated_at, version,
                           COALESCE(content <> '', false) AS has_content
                    FROM context_nodes
                    WHERE uri = %s
                    """,
                    (uri,),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            raise NodeNotFoundError(uri)

        (
            uri_val,
            context_type,
            category,
            node_level,
            owner_space,
            abstract,
            overview,
            content,
            metadata_raw,
            status,
            created_at,
            updated_at,
            version,
            has_content,
        ) = row

        if status == NodeStatus.BROKEN.value:
            raise NodeBrokenError(uri)
        if status != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        meta = dict(metadata_raw or {})
        meta.setdefault("created_at", str(created_at) if created_at else "")
        meta.setdefault("updated_at", str(updated_at) if updated_at else "")
        meta.setdefault("version", version)
        meta["has_content"] = bool(has_content)

        return ContextNode(
            uri=uri_val,
            context_type=context_type,
            category=category,
            level=node_level,
            owner_space=owner_space,
            abstract=abstract or "",
            overview=overview or "",
            content=content or "",
            metadata=meta,
        )

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if node exists and is ACTIVE."""
        if not self._is_accessible(uri, ctx):
            return False

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM context_nodes WHERE uri = %s AND status = 'ACTIVE' LIMIT 1",
                    (uri,),
                )
                return cur.fetchone() is not None
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

    def archived_node_version(self, uri: str, ctx: RequestContext) -> int | None:
        self._ensure_accessible(uri, ctx)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version FROM context_nodes "
                    "WHERE uri = %s AND account_id = %s AND status = 'ARCHIVED'",
                    (uri, ctx.account_id),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                normalized_uri = str(uri).rstrip("/")
                if _RESOURCE_URI_PATTERN.fullmatch(normalized_uri) is not None:
                    account_id = self._extract_account_id(normalized_uri)
                    self._acquire_resource_mutation_lock(
                        cur,
                        account_id,
                        normalized_uri,
                    )
                    cur.execute(
                        "SELECT context_type, metadata FROM context_nodes "
                        "WHERE account_id = %s AND uri = %s FOR UPDATE",
                        (account_id, normalized_uri),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise NodeNotFoundError(uri)
                    raw_metadata = row[1]
                    metadata = (
                        json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
                    )
                    if (
                        row[0] == ContextType.RESOURCE.value
                        and isinstance(metadata, Mapping)
                        and metadata.get("resource_type") == "code"
                    ):
                        raise RuntimeError("code resource deletes require delete_nodes_with_outbox")
                cur.execute(
                    "DELETE FROM context_nodes WHERE uri = %s",
                    (uri,),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(uri)
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def delete_nodes_with_outbox(
        self,
        uris: list[str],
        ctx: RequestContext,
        outbox_events: list[OutboxEvent],
    ) -> int:
        """Atomically delete nodes, suppress stale UPSERTs, and retain DELETE markers."""

        targets = sorted({str(uri).rstrip("/") for uri in uris if str(uri).strip()})
        if not targets:
            return 0

        for uri in targets:
            self._ensure_accessible(uri, ctx)
        account_ids = {self._extract_account_id(uri) for uri in targets}
        if account_ids != {ctx.account_id}:
            raise AccessDeniedError(
                targets[0],
                ctx.account_id,
                "all deleted resources must belong to the request account",
            )

        events_by_uri: dict[str, OutboxEvent] = {}
        for event in outbox_events:
            event_uri = str(event.uri).rstrip("/")
            if event_uri in events_by_uri:
                raise ValueError("duplicate DELETE Outbox event URI")
            if event.event_type != EventType.DELETE_CONTEXT.value:
                raise ValueError("delete transaction accepts DELETE_CONTEXT events only")
            record_ids = event.payload.get("ids_to_delete")
            if not isinstance(record_ids, list) or len(record_ids) != 3:
                raise ValueError("DELETE Outbox event must contain complete record ids")
            events_by_uri[event_uri] = event
        if set(events_by_uri) != set(targets):
            raise ValueError("DELETE Outbox events must match target URIs exactly")

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Acquire deterministic locks in URI order.  Unlike row locks,
                # advisory locks also cover targets that do not exist yet.
                for uri in targets:
                    self._acquire_resource_mutation_lock(cur, ctx.account_id, uri)

                # Lock every currently present target row so concurrent writes
                # that acquired the mutation lock first finish before deletion.
                cur.execute(
                    "SELECT uri FROM context_nodes "
                    "WHERE account_id = %s AND uri = ANY(%s) ORDER BY uri FOR UPDATE",
                    (ctx.account_id, targets),
                )
                cur.fetchall()

                cur.execute(
                    "DELETE FROM context_nodes WHERE account_id = %s AND uri = ANY(%s)",
                    (ctx.account_id, targets),
                )
                deleted = cur.rowcount if isinstance(cur.rowcount, int) else 0

                cur.execute(
                    "DELETE FROM outbox_events "
                    "WHERE account_id = %s AND uri = ANY(%s) "
                    "AND event_type IN (%s, %s) "
                    "AND status IN ('PENDING', 'PROCESSING')",
                    (
                        ctx.account_id,
                        targets,
                        EventType.UPSERT_CONTEXT.value,
                        EventType.UPSERT_DIRECTORY.value,
                    ),
                )

                for uri in targets:
                    event = events_by_uri[uri]
                    cur.execute(
                        """
                        INSERT INTO outbox_events
                            (event_id, event_type, uri, account_id, payload,
                             status, retry_count, created_at, next_retry_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            event.event_id,
                            event.event_type,
                            uri,
                            ctx.account_id,
                            Json(event.payload),
                            event.status,
                            event.retry_count,
                            event.created_at or datetime.now(UTC).isoformat(),
                            event.next_retry_at or None,
                        ),
                    )
                    self.sql_dialect.notify_outbox(
                        cur,
                        "ogmem_outbox",
                        json.dumps(
                            {
                                "event_id": event.event_id,
                                "account_id": ctx.account_id,
                                "event_type": event.event_type,
                            },
                            ensure_ascii=False,
                        ),
                    )
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def archive_node(self, uri: str, ctx: RequestContext) -> None:
        """Archive a node (soft delete). Marks status as ARCHIVED.

        Args:
            uri: URI of the node to archive
            ctx: RequestContext for this operation

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE context_nodes SET status = 'ARCHIVED', "
                    "metadata = metadata || %s, "
                    "updated_at = NOW() "
                    "WHERE uri = %s AND status = 'ACTIVE'",
                    (
                        Json({"archived_at": datetime.now(UTC).isoformat()}),
                        uri,
                    ),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(uri)
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List immediate child URIs under a given parent URI.

        Finds all rows whose URI starts with ``parent_uri/`` and has
        exactly one additional path segment.
        """
        self._ensure_accessible(uri, ctx)
        account_id = self._extract_account_id(uri)

        # Normalise parent: ensure no trailing slash for prefix matching
        parent = uri.rstrip("/")
        prefix = parent + "/"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT uri FROM context_nodes
                    WHERE uri LIKE %s
                      AND account_id = %s
                      AND status = 'ACTIVE'
                    """,
                    (prefix + "%", account_id),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        children = []
        for (child_uri,) in rows:
            # Extract the segment after the parent prefix
            remainder = child_uri[len(prefix) :]
            # Only include immediate children (no further slashes)
            if remainder and "/" not in remainder:
                children.append(child_uri)

        return children

    def count_memory_children(self, ctx: RequestContext) -> dict[str, int]:
        """Count active memory items by parent directory in one scoped query."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT regexp_replace(uri, '/[^/]+$', '') AS parent, category, COUNT(*)
                    FROM context_nodes
                    WHERE account_id = %s AND status = 'ACTIVE'
                      AND category = ANY(%s)
                      AND (position('/memories/' in uri) > 0)
                    GROUP BY parent, category
                    """,
                    (
                        ctx.account_id,
                        ["profile", "preference", "entity", "event", "case", "pattern",
                         "tool", "experience", "session_summary", "session_archive"],
                    ),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        directories = {}
        allowed = {
            "profile": "profile", "preferences": "preference", "entities": "entity",
            "events": "event", "cases": "case", "patterns": "pattern",
            "tools": "tool", "tool_units": "tool", "experiences": "experience", "session_summaries": "session_summary",
            "session_archives": "session_archive",
        }
        for parent, category, count in rows:
            parts = parent.split("/")
            if len(parts) == 7 and parts[3] in {"users", "agents"} and parts[5] == "memories":
                if allowed.get(parts[6]) == category:
                    directories[parent] = int(count)
            elif len(parts) == 6 and parts[3] == "users" and parts[5] == "memories":
                if category == "profile":
                    directories[parent + "/profile"] = int(count)
        return directories

    def list_node_summaries(
        self, uri: str, ctx: RequestContext, limit: int = 50
    ) -> list[ContextNode]:
        """List immediate child summaries without selecting full content."""

        return self.list_node_summaries_page(uri, ctx, limit=limit, after_uri=None)

    def list_node_summaries_page(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        limit: int = 50,
        after_uri: str | None = None,
    ) -> list[ContextNode]:
        """List one stable URI-ordered page of immediate child summaries."""

        self._ensure_accessible(uri, ctx)
        if limit <= 0:
            return []
        account_id = self._extract_account_id(uri)
        parent = uri.rstrip("/")
        prefix = parent + "/"
        components = parse_uri(parent)
        owner_type = components.get("owner_type")
        owner_id = components.get("owner_id")
        owner_space = None
        if owner_type in {"users", "agents"} and owner_id:
            owner_space = f"{owner_type[:-1]}:{owner_id}"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT uri, context_type, category, level, owner_space,
                           abstract, metadata, created_at, updated_at, version,
                           COALESCE(content <> '', false) AS has_content
                    FROM context_nodes
                    WHERE LEFT(uri, CHAR_LENGTH(%s)) = %s
                      AND CHAR_LENGTH(uri) > CHAR_LENGTH(%s)
                      AND account_id = %s
                      AND (%s IS NULL OR owner_space = %s)
                      AND status = 'ACTIVE'
                      AND (%s IS NULL OR uri > %s)
                      AND POSITION('/' IN SUBSTRING(uri FROM %s)) = 0
                    ORDER BY uri
                    LIMIT %s
                    """,
                    (
                        prefix,
                        prefix,
                        prefix,
                        account_id,
                        owner_space,
                        owner_space,
                        after_uri,
                        after_uri,
                        len(prefix) + 1,
                        limit,
                    ),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        summaries: list[ContextNode] = []
        for row in rows:
            (
                child_uri,
                context_type,
                category,
                level,
                owner_space,
                abstract,
                metadata_raw,
                created_at,
                updated_at,
                version,
                has_content,
            ) = row
            remainder = child_uri[len(prefix) :]
            if "/" in remainder:
                continue
            metadata = dict(metadata_raw or {})
            metadata.setdefault("created_at", str(created_at) if created_at else "")
            metadata.setdefault("updated_at", str(updated_at) if updated_at else "")
            metadata.setdefault("version", version)
            metadata["has_content"] = bool(has_content)
            summaries.append(
                ContextNode(
                    uri=child_uri,
                    context_type=context_type,
                    category=category,
                    level=level,
                    owner_space=owner_space,
                    abstract=abstract or "",
                    overview="",
                    content="",
                    metadata=metadata,
                )
            )
            if len(summaries) >= limit:
                break

        return summaries

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        """Move/rename a node (and cascade to children + relation/outbox refs).

        Raises:
            AccessDeniedError: If either URI account doesn't match.
            NodeNotFoundError: If source node doesn't exist.
        """
        self._ensure_accessible(from_uri, ctx)
        self._ensure_accessible(to_uri, ctx)
        account_id = self._extract_account_id(from_uri)

        from_prefix = from_uri.rstrip("/") + "/"
        to_prefix = to_uri.rstrip("/") + "/"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Same write order as projection workers: nodes, jobs, edges.
                cur.execute("""SELECT uri FROM context_nodes WHERE account_id=%s
                    AND (uri=%s OR uri LIKE %s OR relations::text LIKE %s)
                    ORDER BY uri FOR UPDATE""",
                    (account_id,from_uri,from_prefix+"%","%"+from_uri+"%"))
                cur.fetchall()
                cur.execute("""SELECT edge_id,source_uri,target_uri,relation_type
                    FROM edge_projection_jobs WHERE account_id=%s
                    AND (source_uri=%s OR source_uri LIKE %s OR target_uri=%s OR target_uri LIKE %s)
                    ORDER BY edge_id FOR UPDATE""",
                    (account_id,from_uri,from_prefix+"%",from_uri,from_prefix+"%"))
                projection_jobs = cur.fetchall()
                # 1) Move the node itself
                cur.execute(
                    "UPDATE context_nodes SET uri = %s, updated_at = NOW() WHERE uri = %s",
                    (to_uri, from_uri),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(from_uri)

                # 2) Cascade: update children with the old prefix
                cur.execute(
                    "UPDATE context_nodes "
                    "SET uri = concat(%s, substring(uri from %s)), "
                    "    updated_at = NOW() "
                    "WHERE uri LIKE %s",
                    (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                )

                # 3) Cascade: update relation_edges (only if table exists)
                if self.sql_dialect.table_exists(cur, "relation_edges"):
                    cur.execute(
                        "UPDATE relation_edges SET from_uri = %s WHERE from_uri = %s",
                        (to_uri, from_uri),
                    )
                    cur.execute(
                        "UPDATE relation_edges SET to_uri = %s WHERE to_uri = %s",
                        (to_uri, from_uri),
                    )
                    cur.execute(
                        "UPDATE relation_edges "
                        "SET from_uri = concat(%s, substring(from_uri from %s)) "
                        "WHERE from_uri LIKE %s",
                        (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                    )
                    cur.execute(
                        "UPDATE relation_edges "
                        "SET to_uri = concat(%s, substring(to_uri from %s)) "
                        "WHERE to_uri LIKE %s",
                        (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                    )

                # 4) Cascade: rewrite URIs inside context_nodes.relations JSONB
                #
                # move_node updates context_nodes.uri and relation_edges, but the
                # per-node relations JSONB column still holds old from_uri /
                # to_uri values.  When relation_edges is empty the fallback path
                # in SQLRelationStore reads from this column, so stale URIs would
                # leak through.  Rewrite both fields in-place.
                self.sql_dialect.rewrite_moved_relations(
                    cur,
                    from_uri=from_uri,
                    to_uri=to_uri,
                    from_prefix=from_prefix,
                    to_prefix=to_prefix,
                    json_wrapper=Json,
                )
                from providers.relation_store.projection_identity import canonical_edge_id
                from uuid import uuid4
                def moved(uri):
                    if uri == from_uri:
                        return to_uri
                    return to_prefix+uri[len(from_prefix):] if uri.startswith(from_prefix) else uri
                for old_id,source,target,relation in projection_jobs:
                    source,target=moved(source),moved(target)
                    new_id=canonical_edge_id(account_id,source,target,relation)
                    cur.execute("""UPDATE edge_projection_jobs SET edge_id=%s,source_uri=%s,target_uri=%s,
                        status='PENDING',claim_token=NULL,last_error=NULL,next_retry_at=NULL,updated_at=NOW()
                        WHERE account_id=%s AND edge_id=%s""",(new_id,source,target,account_id,old_id))
                    cur.execute("""UPDATE relation_edges SET projection_edge_id=%s,edge_projection=NULL,updated_at=NOW()
                        WHERE account_id=%s AND projection_edge_id=%s""",(new_id,account_id,old_id))
                    cur.execute("""INSERT INTO outbox_events
                        (event_id,event_type,uri,account_id,payload,status,retry_count)
                        VALUES (%s,'PROJECT_GRAPH',%s,%s,%s,'PENDING',0)""",
                        (str(uuid4()),source,account_id,Json({"edge_id":new_id})))

                # 5) Cascade: update outbox_events (only if table exists)
                if self.sql_dialect.table_exists(cur, "outbox_events"):
                    import uuid as _uuid

                    # --- 5a) Delete PENDING and PROCESSING events for old URIs.
                    #
                    # We delete both PENDING and PROCESSING events.  The
                    # PROCESSING events may have been claimed by a worker that
                    # has read the payload into memory.  However, if we leave
                    # them, the worker will upsert stale records at the old URI
                    # *after* our DELETE_CONTEXT runs, undoing the move.
                    #
                    # By deleting the rows, the worker's final mark_done DELETE
                    # matches 0 rows (harmless), and we rely on the tombstone
                    # check in _process_upsert_context to catch the rare case
                    # where the worker has already started processing.
                    cur.execute(
                        "DELETE FROM outbox_events "
                        "WHERE uri = %s AND status IN ('PENDING', 'PROCESSING')",
                        (from_uri,),
                    )
                    cur.execute(
                        "DELETE FROM outbox_events "
                        "WHERE uri LIKE %s AND status IN ('PENDING', 'PROCESSING')",
                        (from_prefix + "%",),
                    )

                    # --- 5b) Emit DELETE_CONTEXT for old URIs.
                    #
                    # Build ids using the canonical IndexRecord.generate_id().
                    from core.models import IndexRecord as _IndexRecord

                    def _record_id(uri: str, level: int) -> str:
                        return _IndexRecord.generate_id(uri, level)

                    # Collect ALL descendants (not just immediate children).
                    # The context_nodes UPDATE already moved every row whose
                    # URI started with from_prefix, so we reconstruct old
                    # URIs by swapping the prefix back.
                    cur.execute(
                        "SELECT uri FROM context_nodes WHERE uri = %s OR uri LIKE %s",
                        (to_uri, to_prefix + "%"),
                    )
                    all_new_uris = [row[0] for row in cur.fetchall()]

                    for new_uri in all_new_uris:
                        old_uri = (
                            from_uri
                            if new_uri == to_uri
                            else from_prefix + new_uri[len(to_prefix) :]
                        )
                        # Build ids for L0, L1, L2 (with /content.md suffix)
                        ids_to_delete = [
                            _record_id(old_uri, 0),
                            _record_id(old_uri, 1),
                            _record_id(old_uri.rstrip("/") + "/content.md", 2),
                        ]
                        cur.execute(
                            """
                            INSERT INTO outbox_events
                                (event_id, event_type, uri, account_id,
                                 payload, status, retry_count, created_at)
                            VALUES (%s, 'DELETE_CONTEXT', %s, %s,
                                    %s, 'PENDING', 0, NOW())
                            """,
                            (
                                str(_uuid.uuid4()),
                                old_uri,
                                account_id,
                                Json({"ids_to_delete": ids_to_delete}),
                            ),
                        )

                    # --- 5c) Emit UPSERT_CONTEXT for new URIs.
                    #
                    # Reuse the complete builder, including L3 descriptors and
                    # canonical revisions. Keep the move path's existing L1.
                    cur.execute(
                        "SELECT uri, abstract, overview, content, "
                        "owner_space, category, context_type, metadata, version "
                        "FROM context_nodes "
                        "WHERE uri = %s OR uri LIKE %s",
                        (to_uri, to_prefix + "%"),
                    )
                    moved_rows = cur.fetchall()

                    for row in moved_rows:
                        r_uri, r_abstract, r_overview, r_content, r_owner, r_cat, r_ct, metadata, version = row
                        from index.index_record_builder import build_index_records
                        metadata = dict(metadata or {})
                        metadata.pop("_expected_version", None)
                        metadata.pop("expected_version", None)
                        metadata["version"] = version
                        node = ContextNode(uri=r_uri, context_type=r_ct, category=r_cat, level=2,
                            owner_space=r_owner, abstract=r_abstract or "", overview=r_overview or "",
                            content=r_content or "", metadata=metadata)
                        built = build_index_records(node)
                        if r_overview and built and not any(record.level == 1 for record in built):
                            built.append(_IndexRecord(id=_record_id(r_uri, 1), uri=r_uri, level=1,
                                text=r_overview, filters=dict(built[0].filters),
                                metadata={**built[0].metadata, "level":"overview", "record_kind":"node_overview"}))
                        from dataclasses import asdict
                        records = [asdict(record) for record in built]

                        if records:
                            cur.execute(
                                """
                                INSERT INTO outbox_events
                                    (event_id, event_type, uri, account_id,
                                     payload, status, retry_count, created_at)
                                VALUES (%s, 'UPSERT_CONTEXT', %s, %s,
                                        %s, 'PENDING', 0, NOW())
                                """,
                                (
                                    str(_uuid.uuid4()),
                                    r_uri,
                                    account_id,
                                    Json({"records": records}),
                                ),
                            )
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
