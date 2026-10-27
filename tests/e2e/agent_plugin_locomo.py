"""Native agent hosts and oGMemory plugin evaluation adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
QA_WRITE_PATHS = ("/after_turn", "/commit", "/compact", "/dispose")


def extract_response_text(response_json: dict) -> str:
    """Extract assistant text from the /v1/responses API response."""
    try:
        for item in response_json.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "")
        for item in response_json.get("output", []):
            if "text" in item:
                return item["text"]
            for content in item.get("content", []):
                if "text" in content:
                    return content["text"]
    except (KeyError, TypeError, IndexError) as e:
        print(f"Warning: Error extracting response text: {e}", file=sys.stderr)
    raise RuntimeError(f"Could not extract text from response: {response_json}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    return ""


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _user_message_fingerprints(payload: dict[str, Any]) -> list[str]:
    messages = payload.get("messages", [])
    start = payload.get("prePromptMessageCount", 0)
    start = max(0, start) if isinstance(start, int) else 0
    fingerprints = []
    for message in messages[start:]:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                block["text"] for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        if isinstance(content, str) and content.strip():
            fingerprints.append(text_fingerprint(content))
    return fingerprints


class ReadOnlyQaProxy:
    """Forward oGMemory traffic, suppressing lifecycle writes during QA."""

    def __init__(self, target_url: str) -> None:
        self.target_url = target_url.rstrip("/")
        self.read_only = False
        self.intercepted: list[str] = []
        self.accepted_writes: list[dict[str, Any]] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = self.path.split("?", 1)[0].rstrip("/")
                if proxy.read_only and any(path.endswith(suffix) for suffix in QA_WRITE_PATHS):
                    proxy.intercepted.append(path)
                    payload = json.dumps({"ok": True, "readOnly": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in {"host", "content-length", "connection"}
                }
                response = requests.request(
                    self.command,
                    f"{proxy.target_url}{self.path}",
                    headers=headers,
                    data=body,
                    timeout=1200,
                )
                if path.endswith("/after_turn") and response.ok:
                    result = response.json()
                    if (result.get("ok") is not False
                            and result.get("status") not in ("failed", "needs_confirmation")
                            and not result.get("error")):
                        payload = json.loads(body)
                        proxy.accepted_writes.append({
                            "session_id": payload.get("sessionId") or payload.get("session_id"),
                            "user_message_fingerprints": _user_message_fingerprints(payload),
                            "at": time.time(),
                        })
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def do_PUT(self) -> None:  # noqa: N802
                self._handle()

            def do_DELETE(self) -> None:  # noqa: N802
                self._handle()

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class AgentHost:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = (
            args.runtime_root.resolve()
            if args.runtime_root
            else args.output_dir / "runtime" / args.agent
        )
        self.logs = args.output_dir / "host-logs"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        invocation_path = args.output_dir / "host-invocations.json"
        if args.resume and invocation_path.exists():
            existing = json.loads(invocation_path.read_text(encoding="utf-8"))
            self.invocations = existing if isinstance(existing, list) else []
        else:
            self.invocations = []
        self.session_lock = threading.Lock()
        self.session_file = self.runtime / "host-sessions.json"
        self.sessions = json.loads(self.session_file.read_text()) if self.session_file.exists() else {}
        self.env = os.environ.copy()
        self.env.update(
            {
                "OGMEM_AGENT_LLM_API_KEY": args.llm_api_key,
                "OG_MEMORY_URL": args.ogmemory_url,
                "OG_AUTH_API_KEY": args.ogmemory_api_key,
                "OG_AUTH_ACCOUNT_ID": args.account_id,
                "OG_AUTH_USER_ID": args.user_id,
                "OG_AUTH_AGENT_ID": args.memory_agent_id,
                "OGMEM_TRACE_CAPTURE_CONTENT": "true",
            }
        )

    def remember_session(self, key: str, sid: str) -> None:
        with self.session_lock:
            self.sessions[key] = sid
            _write_json(self.session_file, self.sessions)

    def prepare(self) -> None:
        pass

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        raise NotImplementedError

    def close(self) -> None:
        _write_json(self.args.output_dir / "host-invocations.json", self.invocations)

    def _run(
        self,
        command: list[str],
        label: str,
        *,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
        timeout: int | None = None,
        check: bool = True,
        record: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=self.runtime,
                env=env or self.env,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout or self.args.agent_timeout,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            log_path = self.logs / f"{label}-{uuid.uuid4().hex[:8]}.json"
            _write_json(
                log_path,
                {
                    "command": command[:1] + ["<arguments-redacted>"],
                    "timed_out": True,
                    "duration_seconds": duration,
                },
            )
            if record:
                self.invocations.append(
                    {
                        "label": label,
                        "executable": command[0],
                        "timed_out": True,
                        "duration_seconds": duration,
                        "log": str(log_path.relative_to(self.args.output_dir)),
                    }
                )
            raise
        duration = time.monotonic() - started
        log_path = self.logs / f"{label}-{uuid.uuid4().hex[:8]}.json"
        _write_json(
            log_path,
            {
                "command": command[:1] + ["<arguments-redacted>"],
                "exit_code": result.returncode,
                "duration_seconds": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if record:
            self.invocations.append(
                {
                    "label": label,
                    "executable": command[0],
                    "exit_code": result.returncode,
                    "duration_seconds": duration,
                    "log": str(log_path.relative_to(self.args.output_dir)),
                }
            )
        if check and result.returncode != 0:
            detail = "\n".join(part for part in (result.stdout, result.stderr) if part)[-2000:]
            raise RuntimeError(f"{self.args.agent} host failed ({result.returncode}): {detail}")
        return result

    def _require(self, executable: str) -> None:
        if not shutil.which(executable):
            raise RuntimeError(f"required agent executable not found: {executable}")


def create_host(args: argparse.Namespace) -> AgentHost:
    from run_claude_code_plugin_locomo import ClaudeCodeHost
    from run_codex_plugin_locomo import CodexHost
    from run_dsh_plugin_locomo import DshHost
    from run_hermes_plugin_locomo import HermesHost
    from run_openclaw_plugin_locomo import OpenClawHost
    from run_opencode_plugin_locomo import OpenCodeHost
    from run_qwenpaw_plugin_locomo import QwenPawHost

    hosts = {
        "openclaw": OpenClawHost,
        "opencode": OpenCodeHost,
        "claude-code": ClaudeCodeHost,
        "hermes": HermesHost,
        "dsh": DshHost,
        "codex": CodexHost,
        "qwenpaw": QwenPawHost,
    }
    return hosts[args.agent](args)


class PluginAdapter:
    def __init__(self, args, common):
        self.args = args
        self.common = common
        self.host = self.proxy = None
        self.description = {
            "transport": "plugin",
            "agent": args.agent,
            "host_sessions": "per_invocation" if args.agent == "dsh" else "per_logical_key",
        }

    def open(self, phase):
        self.close()
        self.phase = phase
        self.proxy = ReadOnlyQaProxy(self.args.ogmemory_url)
        self.proxy.read_only = phase == "qa" and self.args.read_only_qa
        self.proxy.start()
        child = copy.copy(self.args)
        child.ogmemory_url = self.proxy.url
        child.output_dir = self.args.output_dir / self.args.agent / phase
        child.runtime_root = (
            (self.args.runtime_root or self.args.output_dir / "runtime") / self.args.agent / phase
        )
        self.host = create_host(child)
        self.host.prepare()

    def ingest(self, text, key):
        offset = len(self.proxy.accepted_writes)
        fingerprint = text_fingerprint(text)
        reply, usage = self.host.invoke(text, key)
        deadline = time.monotonic() + self.args.write_wait_timeout
        while True:
            writes = self.proxy.accepted_writes[offset:]
            ids = {
                row["session_id"]
                for row in writes
                if row["session_id"] and fingerprint in row["user_message_fingerprints"]
            }
            if ids and self.args.agent == "hermes":
                ids.update(
                    row["session_id"] for row in writes
                    if str(row["session_id"] or "").startswith("hermes-remember-")
                )
            if not self.args.verify_writes:
                return reply, usage, sorted(ids)
            if ids:
                for sid in ids:
                    self.common.wait_idle(self.args, sid)
                return reply, usage, sorted(ids)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"plugin write acknowledgement missing: {key}")
            time.sleep(1)

    def settle(self, sid):
        return self.common.flush_source(self.args, sid)

    def answer(self, prompt, key):
        return self.host.invoke(prompt, key)

    def close(self):
        try:
            if self.host is not None:
                self.host.close()
                self.host = None
        finally:
            if self.proxy is not None:
                path = self.args.output_dir / f"{self.phase}-write-observations.json"
                path.write_text(
                    json.dumps(
                        {
                            "accepted": self.proxy.accepted_writes,
                            "intercepted": self.proxy.intercepted,
                        },
                        indent=2,
                    )
                )
                self.proxy.close()
                self.proxy = None
