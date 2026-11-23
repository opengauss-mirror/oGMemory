# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PostgreSQL-backed ControlPlaneStore.

Replaces JSON-file storage with SQL tables (cp_accounts, cp_users,
cp_agents, cp_audit_logs).  Exposes the same public interface as
ControlPlaneStore so consumers (APIKeyManager, AuditService,
TenantAdminService) work without changes.
"""

from __future__ import annotations

import logging
from typing import Any

from fs.sql_adapter.pool import PoolAdapterMixin

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
    # Connection pool (standalone fallback)
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
