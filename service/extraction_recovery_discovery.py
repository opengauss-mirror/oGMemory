"""Bounded discovery of recovery sources in canonical session state."""

import json
import logging
import re
import time
from collections import OrderedDict

from core.models import RequestContext
from core.write_contract import OUTCOMES_KEY, needs_projection_refresh
from service.extraction_checkpoint import AGENT_TURN_PROJECTION_BARRIER

DISCOVERY_KEY = "memory_extraction_recovery_discovery"
AGENT_TURN_PENDING = "memory_agent_turn_pending"
AGENT_TURN_READY_AT = "agent_turn_ready_at_ms"
AGENT_TURN_TASK = "agent-turn"
_STATE_URI = re.compile(r"^ctx://([^/]+)/sessions/([^/]+)/state(?:\.json)?$")
logger = logging.getLogger(__name__)


def recovery_discovery_hint(payload):
    """Project the source owner and task identity into session metadata."""
    extensions = payload.get("runtime_extensions", {})
    hint = {
        "projection_pending": bool(extensions.get(AGENT_TURN_PROJECTION_BARRIER)) or any(needs_projection_refresh(outcome)
            for outcome in extensions.get(OUTCOMES_KEY, {}).get("entries", {}).values()),
        "agent_turn_pending": bool(extensions.get(AGENT_TURN_PENDING)),
    }
    if AGENT_TURN_READY_AT in extensions:
        hint["agent_ready_at_ms"] = extensions[AGENT_TURN_READY_AT]
    extension = extensions.get("memory_extraction_recovery", {})
    batch = extension.get("batch")
    if extension.get("version") != 1 or not isinstance(batch, dict) or "lifecycle" in batch:
        return hint
    task = batch.get("retry_task")
    if batch.get("agent_async") and (not isinstance(task, dict)
                                    or task.get("status") in {"queued", "running", "failed"}):
        hint["agent_turn_pending"] = True
        if "ready_at_ms" in batch:
            hint["agent_ready_at_ms"] = batch["ready_at_ms"]
        if task:
            hint["agent_next_attempt_at"] = max(task.get("next_attempt_at", 0), task.get("lease_until", 0))
    if isinstance(task, dict):
        task = {key: task[key] for key in ("id", "status", "attempts") if key in task}
    elif (task is None and isinstance(batch.get("snapshot"), list)
          and isinstance(batch.get("messages"), list) and isinstance(batch.get("owner"), dict)):
        task = {"id": batch.get("archive_id"), "status": "queued", "recovered_source": True}
    else:
        return hint
    return {**hint, "owner": batch.get("owner"), "archive_id": batch.get("archive_id"),
            "task": task, "agent_async": batch.get("agent_async", False)}


class ExtractionRecoveryDiscovery:
    """Visit one storage page per call, with a minimum interval per complete cycle."""

    def __init__(self, fs_getter, account_ids_getter, *, page_size=32, cycle_interval=60,
                 projection_handler=None):
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
        self._projection_handler = projection_handler

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
        elif isinstance(metadata.get(DISCOVERY_KEY), dict) and "projection_pending" in metadata[DISCOVERY_KEY]:
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
        if isinstance(hint, dict) and hint.get("projection_pending") and self._projection_handler:
            try:
                self._projection_handler(ctx)
            except Exception:
                logger.exception("Cannot refresh write projection receipts for %s", uri)
        if not isinstance(hint, dict):
            return None
        task = hint.get("task")
        if (hint.get("owner") == owner and not hint.get("agent_async")
                and isinstance(task, dict) and task.get("status") in {"queued", "running", "failed"}
                and isinstance(task.get("id"), str) and task["id"]
                and task["id"] == hint.get("archive_id")):
            return ctx, dict(task)
        if hint.get("agent_turn_pending"):
            return ctx, {"id": AGENT_TURN_TASK, "status": "queued",
                         "agent_async": hint.get("agent_async", False),
                         "ready_at_ms": hint.get("agent_ready_at_ms"),
                         "next_attempt_at": hint.get("agent_next_attempt_at", 0)}
        return None
