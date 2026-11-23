"""PostgreSQL-backed ControlPlaneStore.

Replaces JSON-file storage with SQL tables (cp_accounts, cp_users,
cp_agents, cp_audit_logs).  Exposes the same public interface as
ControlPlaneStore so consumers (APIKeyManager, AuditService,
TenantAdminService) work without changes.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from fs.sql_adapter.pool import PoolAdapterMixin
from server.control_plane_store import CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID
from server.space_locks import lock_space_account

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)

# Sentinel prefix — never collides with real filesystem paths.
_CP = "__cp__"
_registration_connection: ContextVar[tuple[object, Any] | None] = ContextVar(
    "workspace_registration_connection", default=None
)


class SQLControlPlaneStore(PoolAdapterMixin):
    """PostgreSQL-backed control-plane store.

    Path helpers return sentinel strings that ``read_json`` / ``write_json``
    dispatch to the correct table operations.
    """

    _registration_lock = threading.Lock()

    @contextmanager
    def code_workspace_registration_lock(self):
        """Use a separate connection so registration also works with pool_size=1."""
        with self._registration_lock:
            conn = psycopg2.connect(self._connection_string, connect_timeout=5)
            token = _registration_connection.set((self, conn))
            try:
                # The scan after the advisory lock must see the previous holder's commit.
                conn.set_session(isolation_level="READ COMMITTED")
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = 5000")
                    cur.execute("SELECT pg_advisory_xact_lock(%s)", (0x4F474D5753524547,))
                yield
                conn.commit()
            finally:
                _registration_connection.reset(token)
                conn.close()

    @contextmanager
    def _workspace_sync_connection(self, *, write=False):
        active = _registration_connection.get()
        if active is not None and active[0] is self:
            yield active[1]
            return
        conn = self._get_connection()
        try:
            yield conn
            if write:
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
        dialect: str = "postgresql",
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLControlPlaneStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._snapshots = threading.local()
        self._ensure_table()

    # ------------------------------------------------------------------
    # Connection pool: provided by PoolAdapterMixin (bounded standalone
    # pool; shared pool binds its own methods at __init__ time).
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    # ------------------------------------------------------------------
    # Path helpers — return sentinel strings
    # ------------------------------------------------------------------

    def global_accounts_path(self) -> str:
        return f"{_CP}:accounts"

    def users_path(self, account_id: str) -> str:
        return f"{_CP}:users:{account_id}"

    def agents_path(self, account_id: str) -> str:
        return f"{_CP}:agents:{account_id}"

    def audit_logs_path(self, account_id: str) -> str:
        return f"{_CP}:audit_logs:{account_id}"

    def code_workspace_sync_path(self, account_id: str, root_id: str) -> str:
        self._validate_workspace_sync_identity(account_id, root_id)
        return f"{_CP}:code_workspace_sync:{account_id}:{root_id}"

    def read_code_workspace_sync_state(
        self, account_id: str, root_id: str
    ) -> dict[str, Any] | None:
        """Read the protected durable state for one registered workspace."""

        self._validate_workspace_sync_identity(account_id, root_id)
        with self._workspace_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state, lease_owner, lease_expires_at "
                    "FROM code_workspace_sync_roots "
                    "WHERE account_id = %s AND root_id = %s",
                    (account_id, root_id),
                )
                row = cur.fetchone()
            if row is None:
                return None
            state = dict(row[0]) if isinstance(row[0], Mapping) else {}
            state["lease_owner"] = row[1]
            state["lease_expires_at"] = row[2].timestamp() if row[2] is not None else None
            return state

    def write_code_workspace_sync_state(
        self,
        account_id: str,
        root_id: str,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        replacement = copy.deepcopy(dict(state))
        return self.mutate_code_workspace_sync_state(
            account_id,
            root_id,
            lambda _current: replacement,
        )

    def list_code_workspace_sync_states(self, account_id: str) -> list[tuple[str, dict[str, Any]]]:
        """List one account's durable workspace states for reconciliation."""

        self._validate_workspace_sync_identity(account_id, "registry")
        with self._workspace_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT root_id, state, lease_owner, lease_expires_at "
                    "FROM code_workspace_sync_roots "
                    "WHERE account_id = %s AND root_id <> %s ORDER BY root_id",
                    (account_id, CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID),
                )
                rows = cur.fetchall()
            result: list[tuple[str, dict[str, Any]]] = []
            for root_id, raw_state, lease_owner, lease_expires_at in rows:
                state = dict(raw_state) if isinstance(raw_state, Mapping) else {}
                state["lease_owner"] = lease_owner
                state["lease_expires_at"] = (
                    lease_expires_at.timestamp() if lease_expires_at is not None else None
                )
                result.append((root_id, state))
            return result

    def list_code_workspace_sync_registrations(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """List durable workspace registrations for server-owned recovery."""

        with self._workspace_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT account_id, root_id, state, lease_owner, lease_expires_at "
                    "FROM code_workspace_sync_roots "
                    "WHERE root_id <> %s ORDER BY account_id, root_id",
                    (CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID,),
                )
                rows = cur.fetchall()
            result: list[tuple[str, str, dict[str, Any]]] = []
            for account_id, root_id, raw_state, lease_owner, lease_expires_at in rows:
                try:
                    self._validate_workspace_sync_identity(account_id, root_id)
                except ValueError:
                    continue
                state = dict(raw_state) if isinstance(raw_state, Mapping) else {}
                state["lease_owner"] = lease_owner
                state["lease_expires_at"] = (
                    lease_expires_at.timestamp() if lease_expires_at is not None else None
                )
                result.append((account_id, root_id, state))
            return result

    def mutate_code_workspace_sync_state(
        self,
        account_id: str,
        root_id: str,
        mutator: Callable[[dict[str, Any] | None], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Apply one workspace state transition under a SQL row lock."""

        self._validate_workspace_sync_identity(account_id, root_id)
        if not callable(mutator):
            raise ValueError("workspace sync mutator must be callable")
        with self._workspace_sync_connection(write=True) as conn:
            with conn.cursor() as cur:
                self.sql_dialect.ensure_workspace_root(cur, account_id, root_id)
                cur.execute(
                    "SELECT state, lease_owner, lease_expires_at "
                    "FROM code_workspace_sync_roots "
                    "WHERE account_id = %s AND root_id = %s FOR UPDATE",
                    (account_id, root_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("workspace sync state row was not created")
                current = dict(row[0]) if isinstance(row[0], Mapping) else {}
                current["lease_owner"] = row[1]
                current["lease_expires_at"] = row[2].timestamp() if row[2] is not None else None
                updated = mutator(copy.deepcopy(current))
                if not isinstance(updated, Mapping):
                    raise ValueError("workspace sync mutator must return a mapping")
                result = copy.deepcopy(dict(updated))
                lease_owner = result.get("lease_owner")
                lease_expires_at = result.get("lease_expires_at")
                cur.execute(
                    "UPDATE code_workspace_sync_roots "
                    "SET state = %s, lease_owner = %s, lease_expires_at = to_timestamp(%s), "
                    "updated_at = NOW() "
                    "WHERE account_id = %s AND root_id = %s",
                    (
                        Json(result),
                        str(lease_owner) if lease_owner else None,
                        float(lease_expires_at) if lease_expires_at is not None else None,
                        account_id,
                        root_id,
                    ),
                )
            return result

    @staticmethod
    def _validate_workspace_sync_identity(account_id: str, root_id: str) -> None:
        for label, value in (("account_id", account_id), ("root_id", root_id)):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or "/" in value
                or "\\" in value
                or value in {".", ".."}
            ):
                raise ValueError(f"workspace sync {label} is invalid")

    # ------------------------------------------------------------------
    # Sentinel parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_path(path: str) -> tuple[str, str]:
        """Parse sentinel path → (entity_type, account_id).

        Returns ("accounts", "") for global accounts.
        """
        if not path.startswith(_CP + ":"):
            raise ValueError(f"Invalid control-plane path: {path}")
        rest = path[len(_CP) + 1 :]
        parts = rest.split(":", 1)
        entity = parts[0]
        account_id = parts[1] if len(parts) > 1 else ""
        return entity, account_id

    # ------------------------------------------------------------------
    # read_json / write_json — interface-compatible with ControlPlaneStore
    # ------------------------------------------------------------------

    def read_json(self, path: str, default: Any) -> Any:
        result = self._read_json(path, default)
        snapshots = getattr(self._snapshots, "values", None)
        if snapshots is None:
            snapshots = self._snapshots.values = {}
        if self._parse_path(path)[0] in ('accounts', 'users', 'agents'):
            if path not in snapshots and len(snapshots) >= 8:
                snapshots.pop(next(iter(snapshots)))
            snapshots[path] = copy.deepcopy(result)
        return result

    def list_identity_page(
        self, kind: str, account_id: str, limit: int, after: str, query: str, *, owner_user_id: str | None = None,
    ) -> list[dict]:
        table, id_field, fields = {
            "users": ("cp_users", "user_id", "user_id, role, status, created_at"),
            "agents": ("cp_agents", "agent_id", "agent_id, owner_user_id, status, created_at"),
        }[kind]
        params = [account_id, after, query]
        owner_filter = ""
        if kind == "agents" and owner_user_id is not None:
            owner_filter = "AND owner_user_id=%s "
            params.append(owner_user_id)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {fields} FROM {table} WHERE account_id=%s AND status='active' "
                    f"AND {id_field}>%s AND POSITION(LOWER(%s) IN LOWER({id_field}))>0 "
                    f"{owner_filter}ORDER BY {id_field} LIMIT %s",
                    (*params, limit + 1),
                )
                columns = fields.split(", ")
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        finally:
            conn.rollback()
            self._return_connection(conn)

    def create_account_with_admin(self, account_id: str, user_id: str, key: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                lock_space_account(cur, "__accounts__")
                cur.execute("SELECT 1 FROM cp_accounts WHERE account_id=%s", (account_id,))
                if cur.fetchone():
                    raise ValueError(f"account already exists: {account_id}")
                cur.execute("INSERT INTO cp_accounts(account_id) VALUES (%s)", (account_id,))
                cur.execute(
                    "INSERT INTO cp_users(account_id,user_id,role,key) VALUES (%s,%s,'admin',%s)",
                    (account_id, user_id, key),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_member_key(self, account_id: str, user_id: str) -> str | None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT u.key FROM cp_users u JOIN cp_accounts a USING(account_id) "
                    "WHERE u.account_id=%s AND u.user_id=%s AND u.status='active' AND a.status='active' "
                    "AND LOWER(u.role) IN ('user', 'member')",
                    (account_id, user_id),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def resolve_user_key(self, api_key: str, account_hint: str | None = None) -> dict | None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT u.account_id,u.user_id,u.role FROM cp_users u "
                    "JOIN cp_accounts a USING(account_id) "
                    "WHERE u.key=%s AND u.status='active' AND a.status='active' "
                    "AND (%s IS NULL OR u.account_id=%s) LIMIT 1",
                    (api_key, account_hint, account_hint),
                )
                row = cur.fetchone()
                return dict(zip(('account_id', 'user_id', 'role'), row, strict=True)) if row else None
        finally:
            conn.rollback()
            self._return_connection(conn)

    def _read_json(self, path: str, default: Any) -> Any:
        entity, account_id = self._parse_path(path)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if entity == "accounts":
                    cur.execute("SELECT account_id, created_at, status FROM cp_accounts")
                    rows = cur.fetchall()
                    if not rows:
                        return default
                    return {
                        "accounts": {
                            r[0]: {
                                "created_at": r[1].isoformat() if r[1] else "",
                                "status": r[2],
                            }
                            for r in rows
                        }
                    }

                elif entity == "users":
                    cur.execute(
                        "SELECT user_id, role, key, created_at, status "
                        "FROM cp_users WHERE account_id = %s AND status = 'active'",
                        (account_id,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return default
                    return {
                        "users": {
                            r[0]: {
                                "role": r[1],
                                "key": r[2],
                                "created_at": r[3].isoformat() if r[3] else "",
                                "status": r[4],
                            }
                            for r in rows
                        }
                    }

                elif entity == "agents":
                    cur.execute(
                        "SELECT agent_id, owner_user_id, created_at, status "
                        "FROM cp_agents WHERE account_id = %s AND status = 'active'",
                        (account_id,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return default
                    return {
                        "agents": {
                            r[0]: {
                                "owner_user_id": r[1],
                                "created_at": r[2].isoformat() if r[2] else "",
                                "status": r[3],
                            }
                            for r in rows
                        }
                    }

                elif entity == "audit_logs":
                    cur.execute(
                        "SELECT log_id, actor, target, action, "
                        "timestamp, result, trace_id, details "
                        "FROM cp_audit_logs WHERE account_id = %s "
                        "ORDER BY timestamp",
                        (account_id,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return default
                    return {
                        "events": [
                            {
                                "log_id": r[0],
                                "actor": r[1],
                                "target": r[2],
                                "action": r[3],
                                "timestamp": r[4].isoformat() if r[4] else "",
                                "result": r[5],
                                "trace_id": r[6],
                                "details": r[7] if isinstance(r[7], dict) else (r[7] or {}),
                            }
                            for r in rows
                        ]
                    }

                else:
                    return default
        except Exception:
            logger.warning(
                "[SQLControlPlaneStore] read_json failed for %s",
                path,
                exc_info=True,
            )
            return default
        finally:
            conn.rollback()
            self._return_connection(conn)

    def write_json(self, path: str, data: Any, *, preserve_admin: bool = False) -> None:
        entity, account_id = self._parse_path(path)
        if entity not in ("accounts", "users", "agents"):
            return self._write_json(path, data)
        snapshots = getattr(self._snapshots, "values", {})
        previous = snapshots.pop(path, {}).get(entity, {})
        current = data.get(entity, {})
        table, id_column, fields = {
            "accounts": ("cp_accounts", "account_id", ("status", "created_at")),
            "users": ("cp_users", "user_id", ("role", "key", "status", "created_at")),
            "agents": ("cp_agents", "agent_id", ("owner_user_id", "created_at", "status")),
        }[entity]
        defaults = {"role": "user", "key": "", "status": "active", "owner_user_id": "", "created_at": None}
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                lock_space_account(cur, account_id or "__accounts__")
                had_admin = False
                if preserve_admin:
                    cur.execute(
                        "SELECT 1 FROM cp_users WHERE account_id=%s AND role='admin' "
                        "AND status='active' LIMIT 1", (account_id,),
                    )
                    had_admin = bool(cur.fetchone())
                for identity in previous.keys() - current.keys():
                    if entity == "users":
                        if preserve_admin:
                            cur.execute(
                                "SELECT 1 FROM cp_agents WHERE account_id=%s AND owner_user_id=%s "
                                "AND status='active' LIMIT 1", (account_id, identity),
                            )
                            if cur.fetchone():
                                raise ValueError("Reassign the user's Agents before removing this user")
                            cur.execute(
                                "UPDATE cp_users SET status='deleted', key='revoked' "
                                "WHERE account_id=%s AND user_id=%s",
                                (account_id, identity),
                            )
                        else:
                            self._delete_user_row(cur, account_id, identity)
                        continue
                    where = f"{id_column}=%s"
                    values = [identity]
                    if entity != "accounts":
                        where += " AND account_id=%s"
                        values.append(account_id)
                    cur.execute(f"UPDATE {table} SET status='deleted' WHERE {where}", values)
                for identity, info in current.items():
                    before = previous.get(identity)
                    owner = info.get("owner_user_id")
                    if entity == "agents" and owner and owner != (before or {}).get("owner_user_id"):
                        cur.execute(
                            "SELECT 1 FROM cp_users WHERE account_id=%s AND user_id=%s AND status='active'",
                            (account_id, owner),
                        )
                        if not cur.fetchone():
                            raise ValueError("Agent owner is no longer registered")
                    if before is None:
                        where = f"{id_column}=%s"
                        identity_values = [identity]
                        if entity != "accounts":
                            where += " AND account_id=%s"
                            identity_values.append(account_id)
                        cur.execute(f"SELECT status FROM {table} WHERE {where} FOR UPDATE", identity_values)
                        existing = cur.fetchone()
                        if existing and (entity == "users" or existing[0] != "deleted"):
                            raise ValueError(f"{entity[:-1]} already exists: {identity}")
                        field_values = [info.get(field, defaults[field]) or (None if field == 'created_at' else defaults[field]) for field in fields]
                        if existing:
                            cur.execute(
                                f"UPDATE {table} SET " + ','.join(f"{field}=%s" for field in fields) + f" WHERE {where}",
                                [*field_values, *identity_values],
                            )
                            continue
                        columns = [id_column, *fields]
                        values = [identity, *field_values]
                        if entity != "accounts":
                            columns.insert(0, "account_id")
                            values.insert(0, account_id)
                        placeholders = ','.join(['%s'] * len(columns))
                        cur.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})", values)
                    else:
                        changed = [field for field in fields if info.get(field) != before.get(field)]
                        if not changed:
                            continue
                        values = [info.get(field, defaults[field]) for field in changed]
                        where = f"{id_column}=%s"
                        values.append(identity)
                        if entity != "accounts":
                            where += " AND account_id=%s"
                            values.append(account_id)
                        if entity == "users":
                            where += " AND status = 'active'"
                        cur.execute(f"UPDATE {table} SET " + ','.join(f"{field}=%s" for field in changed) + f" WHERE {where}", values)
                        if entity == "users" and cur.rowcount == 0:
                            raise ValueError(f"user no longer exists: {identity}")
                if had_admin:
                    cur.execute(
                        "SELECT 1 FROM cp_users WHERE account_id=%s AND role='admin' "
                        "AND status='active' LIMIT 1",
                        (account_id,),
                    )
                    if not cur.fetchone():
                        raise ValueError("Keep at least one tenant administrator")
                if entity in ("users", "agents"):
                    cur.execute("UPDATE cp_accounts SET authz_rev=authz_rev+1 WHERE account_id=%s", (account_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    @staticmethod
    def _delete_user_row(cur, account_id, user_id):
        params = (account_id, user_id)
        cur.execute("SAVEPOINT cp_delete_user")
        try:
            cur.execute("DELETE FROM cp_users WHERE account_id = %s AND user_id = %s", params)
        except psycopg2.errors.ForeignKeyViolation:
            cur.execute("ROLLBACK TO SAVEPOINT cp_delete_user")
            cur.execute(
                "UPDATE cp_users SET status = 'deleted', key = 'revoked' "
                "WHERE account_id = %s AND user_id = %s",
                params,
            )
        cur.execute("RELEASE SAVEPOINT cp_delete_user")

    def _write_json(self, path: str, data: Any) -> None:
        entity, account_id = self._parse_path(path)
        if entity != 'audit_logs':
            raise ValueError(f'Unknown control-plane entity: {entity}')
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for event in data.get('events', []):
                    self.sql_dialect.insert_audit_log(cur, (
                        event.get('log_id'), account_id, event.get('actor', ''),
                        event.get('target', ''), event.get('action', ''),
                        event.get('timestamp') or None, event.get('result', ''),
                        event.get('trace_id', ''), Json(event.get('details', {})),
                    ))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
