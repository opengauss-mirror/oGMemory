"""openclaw native plugin evaluation adapter."""

from __future__ import annotations

import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import requests
from agent_plugin_locomo import AgentHost, ROOT, _copy_tree, _write_json, extract_response_text


class OpenClawHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.openclaw_command)
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", self.args.openclaw_port)) == 0:
                raise RuntimeError(f"OpenClaw gateway port is already in use: {self.args.openclaw_port}")
        self.home = self.runtime / "home"
        self.profile = f"locomo-{self.args.run_id}"
        self.state = self.home / f".openclaw-{self.profile}"
        self.workspace = self.runtime / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)
        self.plugin_dir = self.runtime / "plugin"
        _copy_tree(ROOT / "plugins/openclaw", self.plugin_dir)
        for path in [self.runtime, self.home, self.state, self.workspace, self.plugin_dir]:
            path.chmod(0o700)
        for path in self.plugin_dir.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.token = uuid.uuid4().hex
        self.env["HOME"] = str(self.home)
        self.env["OPENCLAW_GATEWAY_TOKEN"] = self.token
        config = {
            "agents": {
                "defaults": {
                    "workspace": str(self.workspace),
                    "model": {"primary": f"locomo/{self.args.llm_model}"},
                    "timeoutSeconds": self.args.agent_timeout,
                    "params": {"thinking": "off", "reasoning": "off"},
                },
                "list": [{"id": self.args.memory_agent_id if self.args.agent == "openclaw" else "main",
                          "model": f"locomo/{self.args.llm_model}"}],
            },
            "gateway": {
                "mode": "local",
                "bind": "loopback",
                "port": self.args.openclaw_port,
                "auth": {"mode": "token", "token": self.token},
                "http": {"endpoints": {"responses": {"enabled": True}}},
            },
            "models": {
                "providers": {
                    "locomo": {
                        "baseUrl": self.args.llm_base_url,
                        "apiKey": "${OGMEM_AGENT_LLM_API_KEY}",
                        "api": self.args.openclaw_provider_api,
                        "models": [{"id": self.args.llm_model, "name": self.args.llm_model}],
                    }
                }
            },
            "tools": {"profile": "minimal"},
            "plugins": {
                "load": {"paths": [str(self.plugin_dir)]},
                "slots": {"contextEngine": "og-memory-context-engine"},
                "entries": {
                    "og-memory-context-engine": {
                        "enabled": True,
                        "hooks": {"allowConversationAccess": True},
                        "config": {
                            "mode": "remote",
                            "memoryApiBaseUrl": self.args.ogmemory_url,
                            "authApiKey": self.args.ogmemory_api_key,
                            "authAccountId": self.args.account_id,
                            "accountId": self.args.account_id,
                            "userId": self.args.user_id,
                            "agentId": self.args.memory_agent_id,
                            "prefetchEnabled": True,
                            "prefetchTopK": 5,
                            "compactTakeoverEnabled": False,
                            "shortTermIndexMode": "sync",
                        },
                    }
                },
            },
        }
        _write_json(self.state / "openclaw.json", config)
        command = [self.args.openclaw_command, "--profile", self.profile]
        self._run(command + ["config", "validate"], "prepare-validate", record=False)
        self.gateway_log = (self.logs / "gateway.log").open("w", encoding="utf-8")
        self.gateway = subprocess.Popen(
            command
            + [
                "gateway",
                "run",
                "--port",
                str(self.args.openclaw_port),
                "--bind",
                "loopback",
                "--auth",
                "token",
                "--token",
                self.token,
            ],
            cwd=self.runtime,
            env=self.env,
            text=True,
            stdout=self.gateway_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.gateway.poll() is not None:
                raise RuntimeError("OpenClaw gateway exited during startup")
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.args.openclaw_port}/health",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=2,
                )
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError("OpenClaw gateway did not become healthy")

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        started = time.monotonic()
        headers = {"Authorization": f"Bearer {self.token}", "X-OpenClaw-Session-Key": label}
        user = self.args.user_id
        response = requests.post(
            f"http://127.0.0.1:{self.args.openclaw_port}/v1/responses",
            headers=headers,
            json={
                "model": "openclaw",
                "input": prompt,
                "stream": False,
                "user": user,
            },
            timeout=self.args.agent_timeout,
        )
        duration = time.monotonic() - started
        if not response.ok:
            log_path = self.logs / f"{label}-{uuid.uuid4().hex[:8]}.json"
            _write_json(
                log_path,
                {
                    "duration_seconds": duration,
                    "status_code": response.status_code,
                    "response": response.text,
                },
            )
            response.raise_for_status()
        body = response.json()
        text = extract_response_text(body)
        usage = body.get("usage") or {}
        log_path = self.logs / f"{label}-{uuid.uuid4().hex[:8]}.json"
        _write_json(log_path, {"duration_seconds": duration, "response": body})
        self.invocations.append(
            {
                "label": label,
                "executable": self.args.openclaw_command,
                "transport": "/v1/responses",
                "exit_code": 0,
                "duration_seconds": duration,
                "log": str(log_path.relative_to(self.args.output_dir)),
            }
        )
        return text, {key: int(usage.get(key, 0) or 0) for key in ("input_tokens", "output_tokens", "cacheRead", "cacheWrite", "total_tokens")}

    def close(self) -> None:
        if hasattr(self, "gateway") and self.gateway.poll() is None:
            self.gateway.send_signal(signal.SIGTERM)
            try:
                self.gateway.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.gateway.kill()
                self.gateway.wait(timeout=5)
        if hasattr(self, "profile"):
            self._run(
                [self.args.openclaw_command, "--profile", self.profile, "gateway", "stop"],
                "gateway-stop",
                check=False,
                record=False,
            )
        if hasattr(self, "gateway_log"):
            self.gateway_log.close()
        super().close()

DEFAULT_AGENT_ID = "main"


api_logs = []


api_logs_lock = Lock()


def session_index_path(agent_id: str) -> Path:
    state = Path(os.environ.get("OPENCLAW_STATE_DIR") or os.environ.get("LOCOMO_OPENCLAW_STATE_DIR") or "~/.openclaw").expanduser()
    return state / "agents" / agent_id / "sessions" / "sessions.json"


def get_session_id_from_key(session_key: str, user: str, agent_id: str = "main") -> str | None:
    """Resolve the exact logical key in the selected Gateway agent's state."""
    sessions_file = session_index_path(agent_id)
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        print(f"[session] {sessions_file}: {error}", file=sys.stderr)
        return None
    entry = data.get(f"agent:{agent_id}:{session_key}", {})
    return entry.get("sessionFile")


def reset_ingest_session(session_key: str, user: str, agent_id: str = "main") -> None:
    """Start the selected sample with a fresh local Gateway session."""
    path = session_index_path(agent_id)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data.pop(f"agent:{agent_id}:{session_key}", None)
    if entry is None:
        return
    if entry.get("sessionFile"):
        reset_session(entry["sessionFile"], agent_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reset_session(session_path: str, agent_id: str = "main") -> str | None:
    """Rename the session .jsonl file with a timestamp suffix.
    Accepts either a session_id or a full path to the session file.
    Returns the new filename if successful, None otherwise.
    """
    # Check if session_path is already a full path
    if os.path.isabs(session_path):
        if os.path.exists(session_path):
            src = session_path
        else:
            # Already archived or gone; nothing to do.
            return None
    else:
        # Treat as session_id
        sessions_dir = os.path.expanduser(f"~/.openclaw/agents/{agent_id}/sessions")
        src = os.path.join(sessions_dir, f"{session_path}.jsonl")

    if not os.path.exists(src):
        print(f"    [backup] Session file not found: {src}", file=sys.stderr)
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dst = f"{src}.{timestamp}"
    try:
        os.rename(src, dst)
        new_filename = os.path.basename(dst)
        print(f"    [backup] renamed {os.path.basename(src)} -> {new_filename}", file=sys.stderr)
        return new_filename
    except OSError as e:
        print(f"    [backup] could not rename session file: {e}", file=sys.stderr)
        return None


def calculate_usage_from_jsonl(jsonl_filename: str, agent_id: str = "main") -> dict:
    """Calculate token usage from archived JSONL file."""
    # Check if jsonl_filename is already a full path
    if os.path.isabs(jsonl_filename) and os.path.exists(jsonl_filename):
        jsonl_full_path = jsonl_filename
    else:
        sessions_dir = os.path.expanduser(f"~/.openclaw/agents/{agent_id}/sessions")
        jsonl_full_path = os.path.join(sessions_dir, jsonl_filename)

    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "total_tokens": 0,
    }

    if not os.path.exists(jsonl_full_path):
        return usage

    try:
        with open(jsonl_full_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("type") == "message" and entry.get("message", {}).get("role") == "assistant":
                    entry_usage = entry.get("message", {}).get("usage", {})
                    usage["input_tokens"] += entry_usage.get("input", 0)
                    usage["output_tokens"] += entry_usage.get("output", 0)
                    usage["cacheRead"] += entry_usage.get("cacheRead", 0)
                    usage["cacheWrite"] += entry_usage.get("cacheWrite", 0)
                    usage["total_tokens"] += entry_usage.get("totalTokens", 0)
    except json.JSONDecodeError as e:
        print(f"    [usage] Error parsing JSONL file: {e}", file=sys.stderr)
    except OSError as e:
        print(f"    [usage] Error reading JSONL file: {e}", file=sys.stderr)

    return usage


def send_message(
    base_url: str, token: str, user: str, message: str,
    agent_id: str = DEFAULT_AGENT_ID, session_key: str | None = None, *, instructions: str | None = None
) -> tuple[str, dict]:
    """Send a single message to the OpenClaw responses API.

    Returns (reply_text, usage) where usage has input_tokens, output_tokens, total_tokens.
    """
    url = f"{base_url}/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-OpenClaw-Agent-ID": agent_id
    }
    if session_key:
        headers["X-OpenClaw-Session-Key"] = session_key
    payload = {
        "model": "openclaw",
        "input": message,
        "stream": False,
    }
    if instructions:
        payload["instructions"] = instructions
    if user:
        payload["user"] = user

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "target": "OpenClaw",
        "url": url,
        "method": "POST",
        "user": user,
        "agent_id": agent_id,
        "session_key": session_key or "",
        "request_payload": json.dumps(payload, ensure_ascii=False),
        "response_status": "",
        "response_body": "",
        "response_text": "",
        "error": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=1800)
        log_entry["response_status"] = str(resp.status_code)
        resp.raise_for_status()
        body = resp.json()
        log_entry["response_body"] = json.dumps(body, ensure_ascii=False)
    except requests.exceptions.ConnectionError as e:
        log_entry["error"] = f"Connection error: {e}"
        with api_logs_lock:
            api_logs.append(log_entry)
        raise RuntimeError(f"Connection error to {base_url}: {e}")
    except requests.exceptions.Timeout as e:
        log_entry["error"] = f"Timeout: {e}"
        with api_logs_lock:
            api_logs.append(log_entry)
        raise RuntimeError(f"Request timeout to {base_url}: {e}")
    except requests.exceptions.HTTPError as e:
        log_entry["error"] = f"HTTP error: {e}"
        with api_logs_lock:
            api_logs.append(log_entry)
        raise RuntimeError(f"HTTP error {e.response.status_code} from {base_url}: {e}")
    except json.JSONDecodeError as e:
        log_entry["error"] = f"JSON decode error: {e}"
        with api_logs_lock:
            api_logs.append(log_entry)
        raise RuntimeError(f"Error parsing response from {base_url}: {e}")

    usage = body.get("usage", {"input_tokens": 0, "output_tokens": 0, "cacheRead": 0, "total_tokens": 0})
    try:
        response_text = extract_response_text(body)
    except RuntimeError as e:
        log_entry["error"] = f"Response parse error: {e}"
        with api_logs_lock:
            api_logs.append(log_entry)
        raise
    log_entry["response_text"] = response_text
    log_entry["input_tokens"] = usage.get("input_tokens", 0)
    log_entry["output_tokens"] = usage.get("output_tokens", 0)
    log_entry["total_tokens"] = usage.get("total_tokens", 0)

    with api_logs_lock:
        api_logs.append(log_entry)

    return response_text, usage


def save_api_logs(args):
    """Save API request/response logs to CSV file."""
    if not api_logs:
        return

    log_dir = Path(os.environ.get("LOCOMO_API_LOG_DIR") or Path(getattr(args, "output", ".")).parent / "api_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"api_log_{timestamp}.csv"

    fieldnames = [
        "timestamp", "target", "url", "method", "user", "agent_id",
        "session_key", "request_payload", "response_status",
        "response_body", "response_text", "error",
        "input_tokens", "output_tokens", "total_tokens"
    ]

    try:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(api_logs)
        print(f"\n[API Log] Saved {len(api_logs)} API calls to {log_file}", file=sys.stderr)
    except OSError as e:
        print(f"[API Log] Error saving API logs: {e}", file=sys.stderr)


class CeAdapter:
    def __init__(self, args, common):
        self.args = args
        self.common = common
        self.url, self.token = args.base_url, args.token
        self.description = {"transport": "ce", "agent": "openclaw"}
        self.ingest_keys = set()

    def open(self, phase):
        self.phase = phase

    def _send(self, text, key, instructions=None):
        return send_message(
            self.url,
            self.token,
            self.args.user_id,
            text,
            self.args.agent_id,
            key,
            instructions=instructions,
        )

    def ingest(self, text, key):
        if key not in self.ingest_keys:
            reset_ingest_session(key, self.args.user_id, self.args.agent_id)
            self.ingest_keys.add(key)
        reply, usage = self._send(text, key)
        source_ids = []
        if self.args.verify_writes:
            session_file = get_session_id_from_key(
                key, self.args.user_id, self.args.agent_id
            )
            if session_file is None:
                raise RuntimeError(f"source session file unavailable: {key}")
            source_ids.append(Path(session_file).stem)
        return reply, usage, source_ids

    def settle(self, sid):
        return self.common.flush_source(self.args, sid)

    def answer(self, prompt, key):
        reply, usage = self._send(prompt, key)
        session_file = get_session_id_from_key(key, self.args.user_id, self.args.agent_id)
        if session_file:
            archived = reset_session(session_file, self.args.agent_id)
            if archived:
                usage = calculate_usage_from_jsonl(
                    str(Path(session_file).with_name(archived)), self.args.agent_id
                )
        return reply, usage

    def close(self):
        if api_logs:
            save_api_logs(SimpleNamespace(output=str(self.args.output_dir / "gateway")))
            api_logs.clear()


if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='openclaw'))
