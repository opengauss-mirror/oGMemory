"""The shared oGMemory HTTP and canonical Trace contracts."""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)
_SECRET = re.compile(r"(?i)(authorization|cookie|password|passwd|secret|token|api[_-]?key)")
_TEXT_SECRET = re.compile(
    r"(?im)(?:\b(?:authorization|proxy-authorization|cookie|set-cookie)[\"']?\s*[:=]\s*[^\r\n]+|"
    r"\bBearer\s+\S+|\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key)[\w.-]*[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)|(?:sk-|ghp_|github_pat_|glpat-)[\w-]+)"
)


def redact(value, depth=0):
    """Bound and redact Trace content before it leaves the agent process."""
    if depth > 8:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        return {
            str(key)[:256]: "[REDACTED]" if _SECRET.search(str(key)) else redact(item, depth + 1)
            for key, item in list(value.items())[:64]
        }
    if isinstance(value, list):
        return [redact(item, depth + 1) for item in value[:64]]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except RecursionError:
            return "[DEPTH_LIMIT]"
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            return redact(decoded, depth + 1)
        return _TEXT_SECRET.sub("[REDACTED]", value)[:12000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value), depth + 1)


def event_id(*parts):
    return "qp_" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:48]


def memory_context(result):
    marked = [
        item["content"]
        for item in result.get("messages", [])
        if isinstance(item, dict)
        and item.get("_ogmem") is True
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]
    fields = ("identityContext", "episodicContext", "sessionContext", "retrievedEvidence")
    return "\n\n".join(
        marked
        or [
            result[key]
            for key in fields
            if isinstance(result.get(key), str) and result[key].strip()
        ]
    )[:64000]


class MemoryClient:
    def __init__(self, config=None, *, transport=None):
        config = config or {}

        def setting(key, env, legacy, default):
            return os.environ.get(env) or os.environ.get(legacy) or config.get(key) or default

        self.url = setting("endpoint", "OG_MEMORY_URL", "OGMEMORY_URL", "http://127.0.0.1:8090")
        self.account = setting("account_id", "OG_AUTH_ACCOUNT_ID", "OGMEMORY_ACCOUNT_ID", "default")
        self.user = setting("user_id", "OG_AUTH_USER_ID", "OGMEMORY_USER_ID", "")
        self.agent = setting("agent_id", "OG_AUTH_AGENT_ID", "OGMEMORY_AGENT_ID", "")
        self.key = setting("api_key", "OG_AUTH_API_KEY", "OGMEMORY_API_KEY", "")
        self.write_timeout = float(
            os.environ.get("OGMEMORY_TIMEOUT_SECONDS", config.get("write_timeout_seconds", 120))
        )
        self.compose_timeout = float(
            os.environ.get(
                "OGMEMORY_COMPOSE_TIMEOUT_SECONDS", config.get("compose_timeout_seconds", 5)
            )
        )
        self.bootstrap_timeout = float(
            os.environ.get(
                "OGMEMORY_BOOTSTRAP_TIMEOUT_SECONDS", config.get("bootstrap_timeout_seconds", 5)
            )
        )
        self.trace_timeout = float(
            os.environ.get("OGMEMORY_TRACE_TIMEOUT_SECONDS", config.get("trace_timeout_seconds", 3))
        )
        self.capture_content = str(
            os.environ.get(
                "OGMEM_TRACE_CAPTURE_CONTENT",
                config.get("trace_capture_content", True),
            )
        ).lower() in {"true", "1", "yes"}
        self.http = httpx.AsyncClient(base_url=self.url.rstrip("/"), transport=transport)
        self.sessions = {}
        self._trace_sessions = {}
        self._initializing = {}
        self._traces = asyncio.Queue(maxsize=256)
        self._worker = None
        self._closed = False

    def identity(self, ctx):
        return {
            "accountId": self.account,
            "userId": self.user or getattr(ctx.request, "user_id", None) or "default",
            "agentId": self.agent or ctx.agent_id,
            "sessionId": ctx.session_id,
        }

    async def post(self, path, identity, payload=None, *, timeout=None):
        headers = {"X-Account-ID": identity["accountId"], "X-User-ID": identity["userId"]}
        if self.key:
            headers["X-API-Key"] = self.key
        body = {**identity, **(payload or {})}
        response = await self.http.post(
            path, json=body, headers=headers, timeout=timeout or self.write_timeout
        )
        response.raise_for_status()
        result = response.json()
        if result.get("ok") is False:
            raise RuntimeError("oGMemory request was not accepted")
        return result

    async def ensure_session(self, identity, parent=""):
        key = tuple(identity.values())
        if key not in self._trace_sessions:
            self._trace_sessions[key] = identity
            self.emit(identity, "session_start", identity["sessionId"], parent_session_id=parent)
        if key in self.sessions:
            return
        task = self._initializing.get(key)
        if task is None:
            task = asyncio.create_task(self._start_session(identity, key))
            self._initializing[key] = task
        await asyncio.shield(task)

    async def _start_session(self, identity, key):
        try:
            await self.post("/api/v1/bootstrap", identity, timeout=self.bootstrap_timeout)
            self.sessions[key] = identity
        finally:
            self._initializing.pop(key, None)

    def emit(self, identity, kind, object_id, **fields):
        event = {"event_type": kind}
        content_fields = {
            "prompt",
            "assistant_message",
            "reasoning",
            "tool_input",
            "tool_response",
            "error",
        }
        for key, value in fields.items():
            if key in content_fields and not self.capture_content:
                continue
            clean = redact(value)
            if isinstance(value, str) and not isinstance(clean, str):
                clean = json.dumps(clean, ensure_ascii=False)[:12000]
            event[key] = clean
        body = {
            "request_id": event_id(*identity.values(), kind, object_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "account_id": identity["accountId"],
            "user_id": identity["userId"],
            "agent_id": identity["agentId"],
            "session_id": identity["sessionId"],
            "event": event,
        }
        if len(json.dumps(body, ensure_ascii=False).encode()) > 56000:
            event.update({key: "[TRUNCATED]" for key in content_fields if key in event})
            event["truncated"] = True
        try:
            self._traces.put_nowait((identity, body))
        except asyncio.QueueFull:
            logger.warning("oGMemory Trace queue full")
            return
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._send_traces())

    async def _send_traces(self):
        while not self._traces.empty():
            identity, body = self._traces.get_nowait()
            try:
                await self.post("/api/v1/trace/events", identity, body, timeout=self.trace_timeout)
            except Exception as exc:
                logger.warning("oGMemory Trace delivery failed: %s", type(exc).__name__)
            finally:
                self._traces.task_done()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(*self._initializing.values(), return_exceptions=True)
        for identity in self._trace_sessions.values():
            if self._traces.full():
                await self._traces.join()
            self.emit(identity, "session_end", identity["sessionId"])
        await self._traces.join()
        for identity in self._trace_sessions.values():
            try:
                await self.post("/api/v1/dispose", identity)
            except Exception as exc:
                logger.warning("oGMemory dispose failed: %s", type(exc).__name__)
        self.sessions.clear()
        self._trace_sessions.clear()
        await self.http.aclose()
