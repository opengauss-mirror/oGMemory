# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Minimal audit logging for V1 control-plane actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

from server.control_plane_store import ControlPlaneStore


@dataclass
class AuditLog:
    log_id: str
    actor: str
    target: str
    action: str
    timestamp: str
    result: str
    trace_id: str
    details: dict


class AuditService:
    def __init__(self, store: ControlPlaneStore):
        self._store = store

    def record(
        self,
        account_id: str,
        *,
        actor: str,
        target: str,
        action: str,
        result: str,
        trace_id: str,
        details: dict | None = None,
    ) -> AuditLog:
        event = AuditLog(
            log_id=str(uuid4()),
            actor=actor,
            target=target,
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            result=result,
            trace_id=trace_id,
            details=details or {},
        )
        path = self._store.audit_logs_path(account_id)
        payload = self._store.read_json(path, {"events": []})
        payload.setdefault("events", []).append(asdict(event))
        self._store.write_json(path, payload)
        return event

    def list_logs(self, account_id: str) -> list[dict]:
        payload = self._store.read_json(self._store.audit_logs_path(account_id), {"events": []})
        return payload.get("events", [])

    def get_log(self, account_id: str, log_id: str) -> dict | None:
        for entry in self.list_logs(account_id):
            if entry.get("log_id") == log_id:
                return entry
        return None
