# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared storage helpers for V1 control-plane state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ControlPlaneStore:
    """JSON-backed store for accounts, users, agents and audit logs.

    Supports either a local filesystem root (tests) or an AGFS client.
    """

    def __init__(self, mount_prefix: str, client=None, local_root: str | None = None):
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")
        self._local_root = Path(local_root) if local_root else None

    def global_accounts_path(self) -> str:
        return self._join("_system", "accounts.json")

    def users_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "users.json")

    def agents_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "agents.json")

    def audit_logs_path(self, account_id: str) -> str:
        return self._join("accounts", account_id, "_system", "audit_logs.json")

    def read_json(self, path: str, default: Any) -> Any:
        try:
            raw = self._read_bytes(path)
        except FileNotFoundError:
            return default
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return default

    def write_json(self, path: str, data: Any) -> None:
        self._ensure_parent(path)
        self._write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

    def _join(self, *parts: str) -> str:
        if self._local_root is not None:
            return str(self._local_root.joinpath(*parts))
        base = self._mount_prefix if self._mount_prefix else ""
        return "/".join([base.rstrip("/"), *parts]).replace("//", "/")

    def _read_bytes(self, path: str) -> bytes:
        if self._local_root is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(path)
            return p.read_bytes()
        try:
            return self._client.read(path)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "no such file" in msg:
                raise FileNotFoundError(path) from exc
            raise

    def _write_bytes(self, path: str, data: bytes) -> None:
        if self._local_root is not None:
            Path(path).write_bytes(data)
            return
        self._client.write(path, data)

    def _ensure_parent(self, path: str) -> None:
        parent = str(Path(path).parent) if self._local_root is not None else path.rsplit("/", 1)[0]
        if not parent:
            return
        if self._local_root is not None:
            Path(parent).mkdir(parents=True, exist_ok=True)
            return

        current = ""
        for part in [p for p in parent.split("/") if p]:
            current += "/" + part
            try:
                self._client.mkdir(current)
            except Exception as exc:
                if "exists" not in str(exc).lower():
                    raise
