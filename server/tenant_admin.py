"""Minimal control-plane service for V1 multi-tenant operations."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from itertools import islice

from core.models import RequestContext, Role
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore


class TenantAdminService:
    def __init__(self, key_manager: APIKeyManager, store: ControlPlaneStore, audit: AuditService):
        self._key_manager = key_manager
        self._store = store
        self._audit = audit

    def list_accounts(self, ctx: RequestContext, *, limit=None, after="", q="") -> dict:
        self._require_root(ctx)
        read_page = getattr(self._store, "list_accounts_page", None)
        if limit is not None and read_page is not None:
            return self._page("accounts", "account_id", read_page(limit, after, q), limit)
        page = self._filter_page("accounts", "account_id", self._key_manager.get_accounts(), limit, after, q)
        for row in page["accounts"]:
            row["user_count"] = len(self._key_manager.get_users(row["account_id"]))
        return page

    def create_account(self, ctx: RequestContext, account_id: str, admin_user_id: str) -> dict:
        self._require_root(ctx)
        self._validate_id(account_id, "account_id")
        self._validate_id(admin_user_id, "admin_user_id")
        key = self._key_manager.create_account(account_id, admin_user_id)
        self._audit.record(
            account_id, actor=ctx.user_id, target=f"account:{account_id}", action="account_created",
            result="success", trace_id=ctx.trace_id, details={"admin_user_id": admin_user_id},
        )
        return {"account_id": account_id, "user_id": admin_user_id, "user_key": key, "role": "ADMIN"}

    def get_account(self, ctx: RequestContext, account_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        users = self._key_manager.get_users(account_id)
        return {
            "account_id": account_id,
            "user_count": len(users),
            "status": "active",
        }

    def require_active_account(
        self, ctx: RequestContext, account_id: str, *, admin_only: bool = True
    ) -> None:
        if admin_only:
            self._require_account_admin(ctx, account_id)
        elif self._normalize_role(ctx.role) != Role.ROOT and ctx.account_id != account_id:
            raise PermissionError("account access denied")
        for account in self._key_manager.get_accounts():
            if account["account_id"] == account_id:
                if account.get("status", "active") != "active":
                    raise PermissionError("account is inactive")
                return
        raise FileNotFoundError("account not found")

    def create_user(self, ctx: RequestContext, account_id: str, user_id: str, role: str = "user") -> dict:
        self._require_account_admin(ctx, account_id)
        if not self._account_exists(account_id):
            raise ValueError(f"account not found: {account_id}")
        self._validate_id(user_id, "user_id")
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

    def list_users(self, ctx: RequestContext, account_id: str, *, limit=None, after="", q="") -> dict:
        self._require_account_admin(ctx, account_id)
        read_page = getattr(self._store, "list_identity_page", None)
        if limit is not None and read_page is not None:
            return self._page("users", "user_id", read_page("users", account_id, limit, after, q), limit)
        users = [
            {field: user[field] for field in ("user_id", "role", "status", "created_at") if field in user}
            for user in self._key_manager.get_users(account_id)
        ]
        return self._filter_page("users", "user_id", users, limit, after, q)

    def delete_user(self, ctx: RequestContext, account_id: str, user_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        deleted = self._key_manager.delete_user(account_id, user_id, preserve_admin=True)
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
        self._require_account_admin(ctx, account_id)
        stored_role = self._storage_role_name(role)
        ok = self._key_manager.set_role(account_id, user_id, stored_role, preserve_admin=True)
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

    def reveal_member_key(self, ctx: RequestContext, account_id: str, user_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        key = self._key_manager.get_member_key(account_id, user_id)
        if not key:
            raise FileNotFoundError("Active member key not found")
        self._audit.record(
            account_id, actor=ctx.user_id, target=f"user:{user_id}", action="user_key_revealed",
            result="success", trace_id=ctx.trace_id,
        )
        return {"account_id": account_id, "user_id": user_id, "user_key": key}

    def rotate_user_key(self, ctx: RequestContext, account_id: str, user_id: str) -> dict:
        self._require_account_admin(ctx, account_id)
        key = self._key_manager.rotate_key(account_id, user_id)
        if key is None:
            raise FileNotFoundError("user not found")
        self._audit.record(
            account_id, actor=ctx.user_id, target=f"user:{user_id}", action="user_key_rotated",
            result="success", trace_id=ctx.trace_id,
        )
        return {"account_id": account_id, "user_id": user_id, "user_key": key}

    def list_agents(
        self, ctx: RequestContext, account_id: str, *, limit=None, after="", q="", owner_user_id=None,
    ) -> dict:
        self._require_account_admin(ctx, account_id)
        read_page = getattr(self._store, "list_identity_page", None)
        if limit is not None and read_page is not None:
            rows = read_page("agents", account_id, limit, after, q, owner_user_id=owner_user_id)
            return self._page("agents", "agent_id", rows, limit)
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        agents = [
            {"agent_id": agent_id, **meta}
            for agent_id, meta in sorted(payload.get("agents", {}).items())
            if owner_user_id is None or meta.get("owner_user_id") == owner_user_id
        ]
        return self._filter_page("agents", "agent_id", agents, limit, after, q)

    @staticmethod
    def _page(resource, id_field, rows, limit):
        return {resource: rows[:limit], "next_after": rows[limit - 1][id_field] if len(rows) > limit else None}

    def _filter_page(self, resource, id_field, rows, limit, after, q):
        if limit is None:
            return {resource: rows}
        matches = (
            row for row in rows
            if row[id_field] > after and q.lower() in row[id_field].lower()
        )
        return self._page(resource, id_field, list(islice(matches, limit + 1)), limit)

    def create_agent(self, ctx: RequestContext, account_id: str, agent_id: str, owner_user_id: str | None = None) -> dict:
        self._require_account_admin(ctx, account_id)
        self._validate_id(agent_id, "agent_id")
        if not self._account_exists(account_id):
            raise ValueError(f"account not found: {account_id}")
        owner_user_id = "" if owner_user_id is None else owner_user_id
        with self._key_manager.identity_mutation(account_id):
            if owner_user_id != "" and not self._user_exists(account_id, owner_user_id):
                raise ValueError(f"owner user not found: {owner_user_id}")
            path = self._store.agents_path(account_id)
            payload = self._store.read_json(path, {"agents": {}})
            payload.setdefault("agents", {})
            if agent_id in payload["agents"]:
                raise ValueError(f"agent already exists: {agent_id}")
            now = datetime.now(UTC).isoformat()
            payload["agents"][agent_id] = {
                "owner_user_id": owner_user_id,
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
            details={"owner_user_id": owner_user_id},
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
        with self._key_manager.identity_mutation(account_id):
            if owner_user_id not in (None, "") and not self._user_exists(account_id, owner_user_id):
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
        return payload.get("agents", {}).get(agent_id, {}).get("owner_user_id") or None

    def list_visible_agent_ids(self, account_id: str, user_id: str, cfg) -> list[str]:
        payload = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
        visible: list[str] = []
        agents = payload.get("agents", {})
        shared = set(cfg.agent_shared_list) if cfg.role_control_enabled and cfg.agent_shared_mode == "user" else set()
        for agent_id, meta in sorted(agents.items()):
            if (user_id and meta.get("owner_user_id") == user_id) or agent_id in shared:
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
    def _validate_id(value: str, name: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError(f"{name} must be 1–128 ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit")

    @staticmethod
    def _storage_role_name(role: str) -> str:
        normalized = str(role).strip().lower()
        if normalized == "admin":
            return "admin"
        if normalized in ("member", "user"):
            return "user"
        raise ValueError("role must be ADMIN or MEMBER")
