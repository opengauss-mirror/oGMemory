"""Bounded discovery of recovery sources in canonical session state."""

import json
import logging
import re
import time
from collections import OrderedDict

from core.models import RequestContext

DISCOVERY_KEY = "memory_extraction_recovery_discovery"
_STATE_URI = re.compile(r"^ctx://([^/]+)/sessions/([^/]+)/state(?:\.json)?$")
logger = logging.getLogger(__name__)


def recovery_discovery_hint(payload):
    """Project the source owner and task identity into session metadata."""
    extension = payload.get("runtime_extensions", {}).get("memory_extraction_recovery", {})
    batch = extension.get("batch")
    if extension.get("version") != 1 or not isinstance(batch, dict) or "lifecycle" in batch:
        return None
    task = batch.get("retry_task")
    if isinstance(task, dict):
        task = {key: task[key] for key in ("id", "status", "attempts") if key in task}
    elif (task is None and isinstance(batch.get("snapshot"), list)
          and isinstance(batch.get("messages"), list) and isinstance(batch.get("owner"), dict)):
        task = {"id": batch.get("archive_id"), "status": "queued", "recovered_source": True}
    else:
        return None
    return {"owner": batch.get("owner"), "archive_id": batch.get("archive_id"), "task": task}


class ExtractionRecoveryDiscovery:
    """Visit one storage page per call, with a minimum interval per complete cycle."""

    def __init__(self, fs_getter, account_ids_getter, *, page_size=32, cycle_interval=60):
        self._fs_getter = fs_getter
        self._fs = None
        self._account_ids_getter = account_ids_getter
        self._page_size = max(1, min(int(page_size), 50))
        self._cycle_interval = max(60, cycle_interval)
        self._accounts = []
        self._account_index = 0
        self._cursor = None
        self._next_cycle = 0
        self._legacy = OrderedDict()

    def __call__(self):
        if not self._accounts:
            if time.monotonic() < self._next_cycle:
                return []
            self._accounts = list(dict.fromkeys(self._account_ids_getter()))
            self._account_index = 0
            self._next_cycle = time.monotonic() + self._cycle_interval
            if not self._accounts:
                return []
        if self._fs is None:
            self._fs = self._fs_getter()
        fs = self._fs
        scan = getattr(type(fs), "scan_session_recovery", None)
        if scan is None:
            self._advance(None)
            return []
        account_id = self._accounts[self._account_index]
        try:
            rows, cursor = scan(fs, account_id, cursor=self._cursor, limit=self._page_size)
        except Exception:
            logger.exception("Cannot scan extraction recovery account %s", account_id)
            self._advance(None)
            return []
        found = []
        for row in rows:
            try:
                candidate = self._candidate(fs, account_id, row)
                if candidate is not None:
                    found.append(candidate)
            except Exception:
                logger.exception("Cannot inspect extraction recovery source %s", row.get("uri"))
        self._advance(cursor)
        return found

    def _advance(self, cursor):
        self._cursor = cursor
        if cursor is None:
            self._account_index += 1
            if self._account_index == len(self._accounts):
                self._accounts = []
                self._next_cycle = time.monotonic() + self._cycle_interval

    def _candidate(self, fs, account_id, row):
        uri = row["uri"]
        match = _STATE_URI.fullmatch(uri)
        metadata = row["metadata"]
        if (match is None or match[1] != account_id
                or row["owner_space"] != "session:" + match[2]
                or metadata.get("session_id") != match[2]
                or not all(isinstance(metadata.get(key), str)
                           for key in ("owner_user_id", "owner_agent_id"))):
            return None
        owner = {"account_id": account_id, "user_id": metadata["owner_user_id"],
                 "agent_id": metadata["owner_agent_id"], "session_id": match[2]}
        ctx = RequestContext(**owner, trace_id="extraction-recovery-discovery")
        if "content" in row:
            # AGFS publishes the canonical content before its metadata projection.
            hint = recovery_discovery_hint(json.loads(row["content"]))
        elif DISCOVERY_KEY in metadata:
            hint = metadata[DISCOVERY_KEY]
        else:
            token = (row.get("version"), row.get("updated_at"))
            saved = self._legacy.get(uri)
            if saved is None or saved[0] != token:
                node = fs.read_node(uri, ctx)
                hint = recovery_discovery_hint(json.loads(node.content))
                self._legacy[uri] = (token, hint)
                if len(self._legacy) > 4096:
                    self._legacy.popitem(last=False)
            else:
                hint = saved[1]
                self._legacy.move_to_end(uri)
        if not isinstance(hint, dict) or hint.get("owner") != owner:
            return None
        task = hint.get("task")
        if (not isinstance(task, dict) or task.get("status") not in {"queued", "running", "failed"}
                or not isinstance(task.get("id"), str) or not task["id"]
                or task["id"] != hint.get("archive_id")):
            return None
        return ctx, dict(task)
