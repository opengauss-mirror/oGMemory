"""API key and users registry management for V1 control plane."""

from __future__ import annotations

import secrets
from contextlib import nullcontext
from datetime import UTC, datetime

from server.control_plane_store import ControlPlaneStore


class APIKeyManager:
    def __init__(self, store: ControlPlaneStore):
        self._store = store

    def get_accounts(self) -> list[dict]:
        payload = self._store.read_json(self._store.global_accounts_path(), {"accounts": {}})
        accounts = payload.get("accounts", {})
        return [
            {"account_id": account_id, **meta}
            for account_id, meta in sorted(accounts.items())
        ]

    def create_account(self, account_id: str, admin_user_id: str) -> str:
        create = getattr(self._store, "create_account_with_admin", None)
        if create is not None:
            key = secrets.token_hex(16)
            create(account_id, admin_user_id, key)
            return key
        with self.identity_mutation():
            path = self._store.global_accounts_path()
            payload = self._store.read_json(path, {"accounts": {}})
            accounts = payload.setdefault("accounts", {})
            if account_id in accounts:
                raise ValueError(f"account already exists: {account_id}")
            accounts[account_id] = {"created_at": datetime.now(UTC).isoformat(), "status": "active"}
            key = self.register_user(account_id, admin_user_id, "admin")
            try:
                self._store.write_json(path, payload)
            except Exception:
                self.delete_user(account_id, admin_user_id)
                raise
            return key

    def get_users(self, account_id: str) -> list[dict]:
        payload = self._store.read_json(self._store.users_path(account_id), {"users": {}})
        users = payload.get("users", {})
        return [
            {"user_id": user_id, **meta}
            for user_id, meta in sorted(users.items())
            if meta.get("status", "active") == "active"
        ]

    def register_user(self, account_id: str, user_id: str, role: str = "user") -> str:
        key = secrets.token_hex(16)
        path = self._store.users_path(account_id)
        with self.identity_mutation(account_id):
            payload = self._store.read_json(path, {"users": {}})
            users = payload.setdefault("users", {})
            if user_id in users:
                raise ValueError(f"user already exists: {user_id}")
            users[user_id] = {
                "role": role,
                "key": key,
                "created_at": datetime.now(UTC).isoformat(),
                "status": "active",
            }
            self._store.write_json(path, payload)
        return key

    def delete_user(self, account_id: str, user_id: str, *, preserve_admin=False) -> bool:
        return self._update_user(account_id, user_id, None, preserve_admin=preserve_admin)

    def set_role(self, account_id: str, user_id: str, role: str, *, preserve_admin=False) -> bool:
        return self._update_user(account_id, user_id, {"role": role}, preserve_admin=preserve_admin)

    def get_member_key(self, account_id: str, user_id: str) -> str | None:
        lookup = getattr(self._store, "get_member_key", None)
        if lookup is not None:
            return lookup(account_id, user_id)
        payload = self._store.read_json(self._store.users_path(account_id), {"users": {}})
        user = payload["users"].get(user_id, {})
        if user.get("status", "active") == "active" and user.get("role", "").lower() in ("user", "member"):
            return user.get("key")
        return None

    def rotate_key(self, account_id: str, user_id: str) -> str | None:
        key = secrets.token_hex(16)
        return key if self._update_user(account_id, user_id, {"key": key}) else None

    def identity_mutation(self, account_id: str | None = None):
        return nullcontext() if hasattr(self._store, "create_account_with_admin") else self._store.identity_mutation(account_id)

    def _update_user(self, account_id, user_id, changes, *, preserve_admin=False):
        path = self._store.users_path(account_id)
        with self.identity_mutation(account_id):
            if changes is None and preserve_admin:
                agents = self._store.read_json(self._store.agents_path(account_id), {"agents": {}})
                if any(agent.get("owner_user_id") == user_id for agent in agents["agents"].values()):
                    raise ValueError("Reassign the user's Agents before removing this user")
            payload = self._store.read_json(path, {"users": {}})
            users = payload["users"]
            user = users.get(user_id)
            if user is None or user.get("status", "active") != "active":
                return False
            removes_admin = changes is None or changes.get("role", "admin") != "admin"
            guard_admin = preserve_admin and removes_admin
            sql_store = hasattr(self._store, "create_account_with_admin")
            if guard_admin and not sql_store and user["role"] == "admin" and not any(
                name != user_id and row["role"] == "admin" and row.get("status", "active") == "active"
                for name, row in users.items()
            ):
                raise ValueError("Keep at least one tenant administrator")
            if changes is None:
                if preserve_admin and not sql_store:
                    user.update(status="deleted", key="revoked")
                else:
                    del users[user_id]
            else:
                user.update(changes)
            if guard_admin and sql_store:
                self._store.write_json(path, payload, preserve_admin=True)
            else:
                self._store.write_json(path, payload)
            return True

    def resolve_user_key(self, api_key: str, account_hint: str | None = None) -> dict | None:
        lookup = getattr(self._store, "resolve_user_key", None)
        if lookup is not None:
            return lookup(api_key, account_hint)
        for account in self.get_accounts():
            account_id = account["account_id"]
            if account.get("status", "active") != "active" or (account_hint and account_hint != account_id):
                continue
            payload = self._store.read_json(self._store.users_path(account_id), {"users": {}})
            for user_id, meta in payload.get("users", {}).items():
                if meta.get("status", "active") != "active":
                    continue
                if meta.get("key") == api_key:
                    return {
                        "account_id": account_id,
                        "user_id": user_id,
                        "role": meta.get("role", "user"),
                    }
        return None
