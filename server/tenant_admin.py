# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Minimal control-plane service for V1 multi-tenant operations."""

from __future__ import annotations

from datetime import datetime, timezone

from core.models import RequestContext, Role
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore


class TenantAdminService:
    def __init__(self, key_manager: APIKeyManager, store: ControlPlaneStore, audit: AuditService):
        self._key_manager = key_manager
        self._store = store
        self._audit = audit

    def list_accounts(self, ctx: RequestContext) -> dict:
        self._require_root(ctx)
        rows = self._key_manager.get_accounts()
        for row in rows:
            row["user_count"] = len(self._key_manager.get_users(row["account_id"]))
        return {"accounts": rows}

    def get_account(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        users = self._key_manager.get_users(account_id)
        return {
            "account_id": account_id,
            "user_count": len(users),
            "status": "active",
        }

    def create_user(self, ctx: RequestContext, account_id: str, user_id: str, role: str = "user") -> dict:
        self._require_account_admin(ctx, account_id)
        if not self._account_exists(account_id):
            raise ValueError(f"account not found: {account_id}")
        stored_role = self._storage_role_name(role)
        key = self._key_manager.register_user(account_id, user_id, stored_role)
        self._audit.record(
            account_id,
            actor=ctx.user_id,
            target=f"user:{user_id}",
            action="user_created",
            result="success",
            trace_id=ctx.trace_id,
            details={"role": self._public_role_name(stored_role)},
        )
        return {
            "account_id": account_id,
            "user_id": user_id,
            "user_key": key,
            "role": self._public_role_name(stored_role),
        }

    def list_users(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        return {"users": self._key_manager.get_users(account_id)}

    def delete_user(self, ctx: RequestContext, account_id: str, user_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        deleted = self._key_manager.delete_user(account_id, user_id)
        self._audit.record(
            account_id,
            actor=ctx.user_id,
            target=f"user:{user_id}",
            action="user_deleted",
            result="success" if deleted else "not_found",
            trace_id=ctx.trace_id,
        )
        return {"deleted": deleted}

    def list_roles(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        users = self._key_manager.get_users(account_id)
        return {
            "roles": [
                {"user_id": row["user_id"], "role": self._public_role_name(row.get("role", "user"))}
                for row in users
            ]
        }

    def set_role(self, ctx: RequestContext, account_id: str, user_id: str, role: str) -> dict:
        self._require_root(ctx)
        stored_role = self._storage_role_name(role)
        ok = self._key_manager.set_role(account_id, user_id, stored_role)
        self._audit.record(
            account_id,
            actor=ctx.user_id,
            target=f"user:{user_id}",
            action="role_changed",
            result="success" if ok else "not_found",
            trace_id=ctx.trace_id,
            details={"role": self._public_role_name(stored_role)},
        )
        return {"updated": ok, "role": self._public_role_name(stored_role)}

    def list_agents(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        return {
            "agents": [
                {"agent_id": agent_id, **meta}
                for agent_id, meta in sorted(payload.get("agents", {}).items())
            ]
        }

    def create_agent(self, ctx: RequestContext, account_id: str, agent_id: str, owner_user_id: str | None = None) -> dict:
        self._require_account_admin(ctx, account_id)
        if owner_user_id is not None and not self._user_exists(account_id, owner_user_id):
            raise ValueError(f"owner user not found: {owner_user_id}")
        path = self._store.agents_path(account_id)
        payload = self._store.read_json(path, {"agents": {}})
        payload.setdefault("agents", {})
        if agent_id in payload["agents"]:
            raise ValueError(f"agent already exists: {agent_id}")
        now = datetime.now(timezone.utc).isoformat()
        payload["agents"][agent_id] = {
            "owner_user_id": owner_user_id or ctx.user_id,
            "created_at": now,
        }
        self._store.write_json(path, payload)
        self._audit.record(
            account_id,
            actor=ctx.user_id,
            target=f"agent:{agent_id}",
            action="agent_created",
            result="success",
            trace_id=ctx.trace_id,
            details={"owner_user_id": owner_user_id or ctx.user_id},
        )
        return {"agent_id": agent_id, **payload["agents"][agent_id]}

    def get_agent(self, ctx: RequestContext, account_id: str, agent_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        meta = payload.get("agents", {}).get(agent_id)
        if meta is None:
            return {"error": "not_found"}
        return {"agent_id": agent_id, **meta}

    def update_agent(self, ctx: RequestContext, account_id: str, agent_id: str, owner_user_id: str | None = None) -> dict:
        self._require_account_admin(ctx, account_id)
        if owner_user_id is not None and not self._user_exists(account_id, owner_user_id):
            raise ValueError(f"owner user not found: {owner_user_id}")
        path = self._store.agents_path(account_id)
        payload = self._store.read_json(path, {"agents": {}})
        payload.setdefault("agents", {})
        if agent_id not in payload["agents"]:
            raise FileNotFoundError(f"agent not found: {agent_id}")
        meta = payload["agents"][agent_id]
        if owner_user_id is not None:
            meta["owner_user_id"] = owner_user_id
        self._store.write_json(path, payload)
        self._audit.record(
            account_id,
            actor=ctx.user_id,
            target=f"agent:{agent_id}",
            action="agent_updated",
            result="success",
            trace_id=ctx.trace_id,
            details={"owner_user_id": meta.get("owner_user_id")},
        )
        return {"agent_id": agent_id, **meta}

    def get_agent_sharing_config(self, ctx: RequestContext, cfg) -> dict:
        self._require_admin_or_root(ctx)
        return {
            "role_control_enabled": cfg.role_control_enabled,
            "agent_shared_mode": cfg.agent_shared_mode,
            "agent_shared_list": cfg.agent_shared_list,
        }

    def list_audit_logs(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        return {"audit_logs": self._audit.list_logs(account_id)}

    def get_audit_log(self, ctx: RequestContext, account_id: str, log_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        entry = self._audit.get_log(account_id, log_id)
        return entry or {"error": "not_found"}

    def get_agent_owner(self, account_id: str, agent_id: str) -> str | None:
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        return payload.get("agents", {}).get(agent_id, {}).get("owner_user_id")

    def list_visible_agent_ids(self, account_id: str, user_id: str, cfg) -> list[str]:
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        visible: list[str] = []
        agents = payload.get("agents", {})
        shared = set(cfg.agent_shared_list) if cfg.role_control_enabled and cfg.agent_shared_mode == "user" else set()
        for agent_id, meta in sorted(agents.items()):
            if meta.get("owner_user_id") == user_id or agent_id in shared:
                visible.append(agent_id)
        return visible

    @staticmethod
    def is_agent_shared(cfg, agent_id: str) -> bool:
        return cfg.role_control_enabled and cfg.agent_shared_mode == "user" and agent_id in set(cfg.agent_shared_list)

    def _require_root(self, ctx: RequestContext) -> None:
        if self._normalize_role(ctx.role) != Role.ROOT:
            raise PermissionError("ROOT required")

    def _require_admin_or_root(self, ctx: RequestContext) -> None:
        role = self._normalize_role(ctx.role)
        if role not in (Role.ROOT, Role.ADMIN):
            raise PermissionError("ADMIN required")

    def _require_account_admin(self, ctx: RequestContext, account_id: str) -> None:
        role = self._normalize_role(ctx.role)
        if role == Role.ROOT:
            return
        if role == Role.ADMIN and ctx.account_id == account_id:
            return
        raise PermissionError("ADMIN for account required")

    @staticmethod
    def _normalize_role(role: Role | str) -> Role:
        if isinstance(role, Role):
            return role
        return Role(role)

    def _account_exists(self, account_id: str) -> bool:
        return any(row["account_id"] == account_id for row in self._key_manager.get_accounts())

    def _user_exists(self, account_id: str, user_id: str) -> bool:
        return any(row["user_id"] == user_id for row in self._key_manager.get_users(account_id))

    @staticmethod
    def _public_role_name(role: str) -> str:
        normalized = str(role).strip()
        if normalized == "admin":
            return "ADMIN"
        if normalized in ("user", "MEMBER", "member"):
            return "MEMBER"
        return normalized

    @staticmethod
    def _storage_role_name(role: str) -> str:
        normalized = str(role).strip()
        if normalized in ("ADMIN", "admin"):
            return "admin"
        if normalized in ("MEMBER", "member", "user"):
            return "user"
        return normalized
