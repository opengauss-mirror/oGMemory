"""PostgreSQL-backed ControlPlaneStore.

Replaces JSON-file storage with SQL tables (cp_accounts, cp_users,
cp_agents, cp_audit_logs).  Exposes the same public interface as
ControlPlaneStore so consumers (APIKeyManager, AuditService,
TenantAdminService) work without changes.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from typing import Any

from fs.sql_adapter.pool import PoolAdapterMixin
from server.control_plane_store import CODE_WORKSPACE_SYNC_BINDINGS_ROOT_ID

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)

# Sentinel prefix — never collides with real filesystem paths.
_CP = "__cp__"


class SQLControlPlaneStore(PoolAdapterMixin):
    """PostgreSQL-backed control-plane store.

    Path helpers return sentinel strings that ``read_json`` / ``write_json``
    dispatch to the correct table operations.
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
                "psycopg2 is required for SQLControlPlaneStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size, dialect)
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
        conn = self._get_connection()
        try:
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
        finally:
            conn.rollback()
            self._return_connection(conn)

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
        conn = self._get_connection()
        try:
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
        finally:
            conn.rollback()
            self._return_connection(conn)

    def list_code_workspace_sync_registrations(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """List durable workspace registrations for server-owned recovery."""

        conn = self._get_connection()
        try:
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
        finally:
            conn.rollback()
            self._return_connection(conn)

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
        conn = self._get_connection()
        try:
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
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

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
                        "FROM cp_users WHERE account_id = %s",
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
                        "SELECT agent_id, owner_user_id, created_at "
                        "FROM cp_agents WHERE account_id = %s",
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

    def write_json(self, path: str, data: Any) -> None:
        entity, account_id = self._parse_path(path)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if entity == "accounts":
                    cur.execute("DELETE FROM cp_accounts")
                    for aid, info in data.get("accounts", {}).items():
                        cur.execute(
                            "INSERT INTO cp_accounts (account_id, created_at, status) "
                            "VALUES (%s, %s, %s)",
                            (
                                aid,
                                info.get("created_at") or None,
                                info.get("status", "active"),
                            ),
                        )

                elif entity == "users":
                    cur.execute(
                        "DELETE FROM cp_users WHERE account_id = %s",
                        (account_id,),
                    )
                    for uid, info in data.get("users", {}).items():
                        cur.execute(
                            "INSERT INTO cp_users "
                            "(account_id, user_id, role, key, created_at, status) "
                            "VALUES (%s, %s, %s, %s, %s, %s)",
                            (
                                account_id,
                                uid,
                                info.get("role", "user"),
                                info.get("key", ""),
                                info.get("created_at") or None,
                                info.get("status", "active"),
                            ),
                        )

                elif entity == "agents":
                    cur.execute(
                        "DELETE FROM cp_agents WHERE account_id = %s",
                        (account_id,),
                    )
                    for aid, info in data.get("agents", {}).items():
                        cur.execute(
                            "INSERT INTO cp_agents "
                            "(account_id, agent_id, owner_user_id, created_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                account_id,
                                aid,
                                info.get("owner_user_id", ""),
                                info.get("created_at") or None,
                            ),
                        )

                elif entity == "audit_logs":
                    # Append-only: INSERT new events, skip existing by log_id
                    for ev in data.get("events", []):
                        params = (
                            ev.get("log_id"),
                            account_id,
                            ev.get("actor", ""),
                            ev.get("target", ""),
                            ev.get("action", ""),
                            ev.get("timestamp") or None,
                            ev.get("result", ""),
                            ev.get("trace_id", ""),
                            Json(ev.get("details", {})),
                        )
                        self.sql_dialect.insert_audit_log(cur, params)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
