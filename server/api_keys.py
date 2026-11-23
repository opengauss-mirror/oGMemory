# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""API key and users registry management for V1 control plane."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

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
        payload = self._store.read_json(self._store.global_accounts_path(), {"accounts": {}})
        payload.setdefault("accounts", {})
        if account_id in payload["accounts"]:
            raise ValueError(f"account already exists: {account_id}")
        now = datetime.now(timezone.utc).isoformat()
        payload["accounts"][account_id] = {"created_at": now, "status": "active"}
        self._store.write_json(self._store.global_accounts_path(), payload)
        return self.register_user(account_id, admin_user_id, "admin")

    def get_users(self, account_id: str) -> list[dict]:
        payload = self._store.read_json(self._store.users_path(account_id), {"users": {}})
        users = payload.get("users", {})
        return [
            {"user_id": user_id, **meta}
            for user_id, meta in sorted(users.items())
        ]

    def register_user(self, account_id: str, user_id: str, role: str = "user") -> str:
        now = datetime.now(timezone.utc).isoformat()
        key = secrets.token_hex(16)
        path = self._store.users_path(account_id)
        payload = self._store.read_json(path, {"users": {}})
        payload.setdefault("users", {})
        if user_id in payload["users"]:
            raise ValueError(f"user already exists: {user_id}")
        payload["users"][user_id] = {
            "role": role,
            "key": key,
            "created_at": now,
            "status": "active",
        }
        self._store.write_json(path, payload)
        return key

    def delete_user(self, account_id: str, user_id: str) -> bool:
        path = self._store.users_path(account_id)
        payload = self._store.read_json(path, {"users": {}})
        users = payload.setdefault("users", {})
        if user_id not in users:
            return False
        del users[user_id]
        self._store.write_json(path, payload)
        return True

    def set_role(self, account_id: str, user_id: str, role: str) -> bool:
        path = self._store.users_path(account_id)
        payload = self._store.read_json(path, {"users": {}})
        users = payload.setdefault("users", {})
        if user_id not in users:
            return False
        users[user_id]["role"] = role
        self._store.write_json(path, payload)
        return True

    def resolve_user_key(self, api_key: str, account_hint: str | None = None) -> dict | None:
        accounts = [account_hint] if account_hint else [row["account_id"] for row in self.get_accounts()]
        for account_id in filter(None, accounts):
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
