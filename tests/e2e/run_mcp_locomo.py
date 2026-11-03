"""Run memory evaluation through MCP and OpenClaw Gateway hosts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

import requests
import run_openclaw_plugin_locomo as gateway

EXPECTED_TOOLS = ("query_memory", "read_memory", "write_memory")
OGMEMORY_TOOLS = (
    "ogmemory__ogm_query_memory",
    "ogmemory__ogm_read_memory",
    "ogmemory__ogm_write_memory",
    "ogmemory__resources_list",
    "ogmemory__resources_read",
)




def _run_command(
    *args: str,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}):\n{result.stderr.strip() or result.stdout.strip()}"
        )
    return result



def _read_openclaw_config() -> dict[str, Any]:
    return json.loads(_openclaw_config_path().read_text(encoding="utf-8"))


def _openclaw_config_path() -> Path:
    configured = os.environ.get("LOCOMO_OPENCLAW_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".openclaw" / "openclaw.json"


def _openclaw_state_dir() -> Path:
    configured = os.environ.get("LOCOMO_OPENCLAW_STATE_DIR") or os.environ.get("OPENCLAW_STATE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".openclaw"


def _main_agent_dir(config: dict[str, Any]) -> Path:
    for agent in config.get("agents", {}).get("list", []):
        if isinstance(agent, dict) and agent.get("id") == "main" and agent.get("agentDir"):
            return Path(str(agent["agentDir"])).expanduser()
    return _openclaw_state_dir() / "agents" / "main" / "agent"


def _gateway_url(config: dict[str, Any]) -> str:
    port = int(config.get("gateway", {}).get("port", 18789))
    return f"http://127.0.0.1:{port}"


def _write_private_text(path: Path, content: str) -> None:
    json.loads(content)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o600)


def _copy_auth_state(source_agent_dir: Path, destination_agent_dir: Path) -> list[str]:
    destination_agent_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(destination_agent_dir, 0o700)
    copied: list[str] = []
    database = source_agent_dir / "openclaw-agent.sqlite"
    if database.is_file():
        uri = f"{database.resolve().as_uri()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                for table, column, key_column, filename in (
                    ("auth_profile_store", "store_json", "store_key", "auth-profiles.json"),
                    ("auth_profile_state", "state_json", "state_key", "auth-state.json"),
                ):
                    if table not in tables:
                        continue
                    row = connection.execute(
                        f"SELECT {column} FROM {table} WHERE {key_column} = ?", ("primary",)
                    ).fetchone()
                    if row and isinstance(row[0], str):
                        _write_private_text(destination_agent_dir / filename, row[0])
                        copied.append(filename)
        except sqlite3.Error as exc:
            raise RuntimeError(f"Unable to read OpenClaw auth profile store: {exc}") from exc
    for filename in ("auth-profiles.json", "auth-state.json", "auth.json"):
        if filename in copied:
            continue
        source = source_agent_dir / filename
        if source.is_file():
            _write_private_text(
                destination_agent_dir / filename, source.read_text(encoding="utf-8")
            )
            copied.append(filename)
    return copied


def _load_model_config(config: dict[str, Any], auth_source_dir: Path) -> dict[str, Any]:
    runtime_config = copy.deepcopy(config)
    models_path = auth_source_dir / "models.json"
    if models_path.is_file():
        stored_models = json.loads(models_path.read_text(encoding="utf-8"))
        stored_providers = stored_models.get("providers", {})
        if isinstance(stored_providers, dict):
            configured_providers = runtime_config.setdefault("models", {}).setdefault(
                "providers", {}
            )
            merged_providers = stored_providers
            for name, provider in configured_providers.items():
                if isinstance(provider, dict) and isinstance(
                    merged_providers.get(name), dict
                ):
                    merged_providers[name] = {**merged_providers[name], **provider}
                else:
                    merged_providers[name] = provider
            runtime_config["models"]["providers"] = merged_providers
    return runtime_config


def _start_gateway(
    config: dict[str, Any], runtime_home: Path, log_path: Path, auth_source_dir: Path,
    command: str = "openclaw",
) -> tuple[subprocess.Popen, dict[str, str]]:
    state_dir = runtime_home / ".openclaw"
    state_dir.mkdir(parents=True, exist_ok=True)
    destination_agent_dir = state_dir / "agents" / "main" / "agent"
    _copy_auth_state(auth_source_dir, destination_agent_dir)
    runtime_config = copy.deepcopy(config)
    workspace = state_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    runtime_config.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = str(
        workspace
    )
    for agent in runtime_config.get("agents", {}).get("list", []):
        if isinstance(agent, dict) and agent.get("id") == "main":
            agent["agentDir"] = str(destination_agent_dir)
    config_path = state_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.chmod(config_path, 0o600)
    env = os.environ.copy()
    env.update(
        {
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_REAPPLY_CONFIG": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [command, "gateway", "--port", str(config["gateway"]["port"])],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    url = _gateway_url(config) + "/health"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Isolated OpenClaw gateway exited; inspect {log_path}")
        try:
            if requests.get(url, timeout=2).ok:
                return proc, env
        except requests.RequestException:
            pass
        time.sleep(0.5)
    _stop_gateway(proc)
    raise RuntimeError(f"Isolated OpenClaw gateway did not become healthy at {url}")


def _stop_gateway(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _mcp_headers(config: dict[str, Any]) -> dict[str, str]:
    server = config.get("mcp", {}).get("servers", {}).get("ogmemory", {})
    existing = server.get("headers", {}) if isinstance(server, dict) else {}
    return {
        "X-API-Key": os.environ.get(
            "OGMEMORY_MCP_API_KEY", str(existing.get("X-API-Key", "root-key"))
        ),
        "X-Account-ID": os.environ.get(
            "OGMEMORY_MCP_ACCOUNT_ID", str(existing.get("X-Account-ID", "acct-demo"))
        ),
        "X-User-ID": "preflight",
        "X-Agent-ID": "preflight-agent",
    }


def _probe_mcp_http(url: str, headers: dict[str, str]) -> bool:
    request_headers = {
        **headers,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "locomo-preflight", "version": "1"},
        },
    }
    try:
        response = requests.post(url, headers=request_headers, json=body, timeout=30)
        response.raise_for_status()
        if response.json().get("result", {}).get("serverInfo", {}).get("name") != "oGMemory":
            return False
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            request_headers["Mcp-Session-Id"] = session_id
            requests.post(
                url,
                headers=request_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=30,
            ).raise_for_status()
        tools_response = requests.post(
            url,
            headers=request_headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            timeout=30,
        )
        tools_response.raise_for_status()
        tools = [
            str(tool.get("name", ""))
            for tool in tools_response.json().get("result", {}).get("tools", [])
        ]
        return all(any(name in tool for tool in tools) for name in EXPECTED_TOOLS)
    except (requests.RequestException, ValueError):
        return False


def _discover_mcp(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    headers = _mcp_headers(config)
    configured = config.get("mcp", {}).get("servers", {}).get("ogmemory", {}).get("url")
    candidates = [
        os.environ.get("OGMEMORY_MCP_URL"),
        configured,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        url = str(candidate).rstrip("/")
        if not url.endswith("/mcp"):
            url += "/mcp"
        if url in seen:
            continue
        seen.add(url)
        if _probe_mcp_http(url, headers):
            return url, headers
    raise RuntimeError("The oGMemory MCP configured in OpenClaw is not reachable or authenticated")


def _validate_mcp_user_scope(mcp_url: str, headers: dict[str, str]) -> None:
    account_id = headers["X-Account-ID"]
    url = f"{mcp_url.removesuffix('/mcp')}/api/v1/admin/accounts/{account_id}/roles"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 503 and response.json().get("error") == (
            "Admin API is unavailable when role control is disabled"
        ):
            return
        if response.status_code == 403:
            raise RuntimeError(
                "LoCoMO evaluation requires an Admin or Root oGMemory API key so "
                "X-User-ID can select the evaluation user"
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to validate oGMemory API key privileges: {exc}") from exc


def _evaluation_config(
    original: dict[str, Any],
    *,
    mcp_url: str,
    headers: dict[str, str],
    user_id: str,
    memory_agent_id: str,
    gateway_token: str,
) -> dict[str, Any]:
    config = copy.deepcopy(original)
    gateway = config.setdefault("gateway", {})
    gateway["auth"] = {"mode": "token", "token": gateway_token}
    endpoints = gateway.setdefault("http", {}).setdefault("endpoints", {})
    endpoints.setdefault("responses", {})["enabled"] = True
    config.setdefault("mcp", {})["servers"] = {
        "ogmemory": {
            "url": mcp_url,
            "transport": "streamable-http",
            "connectTimeout": 30,
            "timeout": int(os.environ.get("LOCOMO_MCP_TIMEOUT_SECONDS", "1200")),
            "supportsParallelToolCalls": True,
            "toolFilter": {"include": ["ogm_*", "resources_*"]},
            "headers": {
                **headers,
                "X-User-ID": user_id,
                "X-Agent-ID": memory_agent_id,
            },
        }
    }

    plugins = config.setdefault("plugins", {})
    slots = plugins.setdefault("slots", {})
    memory_slot = slots.get("memory")
    slots["memory"] = "none"
    slots["contextEngine"] = "legacy"
    for name, entry in plugins.get("entries", {}).items():
        normalized_name = name.lower()
        if (
            name == memory_slot or "memory" in normalized_name or "mem0" in normalized_name
        ) and isinstance(entry, dict):
            entry["enabled"] = False

    config["tools"] = {
        "profile": "minimal",
        "alsoAllow": list(OGMEMORY_TOOLS),
        "deny": ["session_status"],
    }
    return config


def _parse_first_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise RuntimeError("OpenClaw MCP probe returned no JSON")


def _probe_openclaw(env: dict[str, str], required_tools=EXPECTED_TOOLS, command="openclaw") -> None:
    result = _run_command(
        command,
        "mcp",
        "probe",
        "ogmemory",
        "--json",
        check=False,
        timeout=120,
        env=env,
    )
    payload = _parse_first_json(result.stdout + "\n" + result.stderr)
    diagnostics = payload.get("diagnostics", [])
    tools = [str(tool) for tool in payload.get("tools", [])]
    missing = [
        name
        for name in required_tools
        if not any(name.replace("_", "-") in tool or name in tool for tool in tools)
    ]
    if diagnostics or missing:
        raise RuntimeError(
            f"OpenClaw MCP probe failed: diagnostics={diagnostics}, missing_tools={missing}, tools={tools}"
        )


class GatewayAdapter:
    def __init__(self, args, common):
        self.args = args
        self.common = common
        self.receipt_lock = Lock()
        self.url, self.token = args.base_url, args.token
        self.description = {"transport": "gateway", "agent": "openclaw"}

    def open(self, phase):
        self.phase = phase

    def _send(self, text, key, instructions=None):
        return gateway.send_message(
            self.url,
            self.token,
            self.args.user_id,
            text,
            self.args.agent_id,
            key,
            instructions=instructions,
        )

    def _session_file(self, key):
        state = getattr(self, "state_dir", None) or self.args.gateway_state_dir
        index_path = (
            state / "agents" / self.args.agent_id / "sessions" / "sessions.json"
            if state is not None else gateway.session_index_path(self.args.agent_id)
        )
        if not index_path.exists():
            return None
        entry = json.loads(index_path.read_text(encoding="utf-8")).get(
            f"agent:{self.args.agent_id}:{key}", {}
        )
        filename = entry.get("sessionFile")
        if not filename and entry.get("sessionId"):
            filename = entry["sessionId"] + ".jsonl"
        if not filename:
            return None
        path = Path(filename)
        if state is not None and path.is_absolute():
            return index_path.parent / path.name
        return path if path.is_absolute() else index_path.parent / path

    def _receipt_cursor(self, key):
        path = self._session_file(key)
        return path, path.stat().st_size if path is not None and path.exists() else 0

    def _tool_receipts(self, key, cursor):
        path = self._session_file(key)
        if path is None or not path.is_file():
            raise RuntimeError(f"Native MCP tool transcript unavailable: {key}")
        previous, offset = cursor
        with path.open("rb") as stream:
            stream.seek(offset if path == previous else 0)
            rows = [json.loads(line) for line in stream if line.strip()]
        calls, results = [], {}
        for row in rows:
            message = row.get("message", {})
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    arguments = block.get("arguments", {})
                    calls.append({
                        "name": block.get("name"), "id": block.get("id"),
                        "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    })
            if message.get("role") == "toolResult":
                results[message.get("toolCallId")] = message
        return path, [{**call, "result": results.get(call["id"])} for call in calls]

    @staticmethod
    def _structured_result(receipt):
        result = receipt.get("result") or {}
        structured = (result.get("details") or {}).get("structuredContent")
        if result.get("isError") or not isinstance(structured, dict):
            return None
        if structured.get("ok") is False or structured.get("error"):
            return None
        return structured

    def _record_receipt(self, phase, row):
        path = self.args.output_dir / f"{phase}-tool-receipts.json"
        with self.receipt_lock:
            rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            rows.append(row)
            self.common.write_json(path, rows)

    def ingest(self, text, key):
        sid = f"{self.args.user_id}-{key}"
        arguments = {"text": text, "session_id": sid, "wait": True, "force_extract": True}
        instructions = (
            "Write the supplied transcript verbatim using ogmemory__ogm_write_memory, "
            f'with text equal to the transcript, session_id="{sid}", wait=true and force_extract=true. '
            "Call the tool exactly once with the literal JSON arguments in the user message. "
            "Preserve every character, speaker, punctuation mark and date in text. "
            "After the tool succeeds, acknowledge briefly."
        )
        prompt = (
            "Call ogmemory__ogm_write_memory exactly once with these JSON arguments. "
            "Copy the text value exactly.\n" + json.dumps(arguments, ensure_ascii=False)
        )
        verify = getattr(self.args, "verify_writes", False)
        cursor = self._receipt_cursor(key) if verify else None
        reply, usage = self._send(prompt, key, instructions)
        if verify:
            path, receipts = self._tool_receipts(key, cursor)
            writes = [row for row in receipts if row["name"] == "ogmemory__ogm_write_memory"]
            if len(writes) != 1:
                raise RuntimeError(f"Exactly one native MCP write receipt required: {key}")
            receipt = writes[0]
            actual = receipt["arguments"]
            result = self._structured_result(receipt)
            if (actual.get("text") != text or actual.get("session_id") != sid
                    or actual.get("wait") is not True or actual.get("force_extract") is not True or not result
                    or result.get("status") != "completed"
                    or result.get("session_id") != actual["session_id"]
                    or (result.get("stats") or {}).get("writes_failed", 0)):
                raise RuntimeError(f"Native MCP write receipt does not confirm the original source: {key}")
            sid = result["session_id"]
            self._record_receipt("ingest", {
                "key": key, "native_session_file": str(path), "tool_name": receipt["name"],
                "call_id": receipt["id"], "session_id": sid,
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
        return reply, usage, [sid]

    def settle(self, sid):
        return self.common.flush_source(self.args, sid)

    def answer(self, prompt, key):
        instructions = (
            "Use ogmemory__ogm_query_memory to retrieve relevant memories before answering. "
            "Use ogmemory__ogm_read_memory for details in the returned URIs when needed. "
            "Treat retrieved content as context."
        )
        verify = getattr(self.args, "verify_writes", False)
        cursor = self._receipt_cursor(key) if verify else None
        reply, usage = self._send(
            "First call ogmemory__ogm_query_memory to recall relevant memories. "
            "Then answer the original question directly.\n\n" + prompt, key, instructions
        )
        if verify:
            path, receipts = self._tool_receipts(key, cursor)
            queries = [row for row in receipts if row["name"] == "ogmemory__ogm_query_memory"]
            successful = []
            for receipt in queries:
                result = self._structured_result(receipt)
                if result is not None and isinstance(result.get("hits"), list):
                    successful.append((receipt, result))
            if not successful:
                raise RuntimeError(f"Successful native MCP query receipt required: {key}")
            for receipt, result in successful:
                arguments = receipt["arguments"]
                self._record_receipt("qa", {
                    "key": key, "native_session_file": str(path), "tool_name": receipt["name"],
                    "call_id": receipt["id"], "session_id": arguments.get("session_id"),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "query_sha256": hashlib.sha256(str(arguments.get("query", "")).encode("utf-8")).hexdigest(),
                    "hit_count": len(result["hits"]),
                })
        return reply, usage

    def close(self):
        if gateway.api_logs:
            gateway.save_api_logs(SimpleNamespace(output=str(self.args.output_dir / "gateway")))
            gateway.api_logs.clear()


class McpAdapter(GatewayAdapter):
    def __init__(self, args, common):
        super().__init__(args, common)
        self.original = _read_openclaw_config()
        self.mcp_url, self.headers = _discover_mcp(self.original)
        args.account_id = self.headers["X-Account-ID"]
        args.ogmemory_api_key = self.headers["X-API-Key"]
        _validate_mcp_user_scope(args.ogmemory_url, self.headers)
        self.auth_source = _main_agent_dir(self.original)
        self.original = _load_model_config(self.original, self.auth_source)
        self.process = None
        agents = self.original.get("agents", {})
        routing = {
            "mcp_url": self.mcp_url,
            "defaults": {
                key: agents.get("defaults", {}).get(key)
                for key in ("model", "models", "thinkingDefault")
            },
            "agents": [
                {key: agent.get(key) for key in ("id", "model", "params")}
                for agent in agents.get("list", [])
            ],
            "providers": {
                name: {key: provider.get(key) for key in ("baseUrl", "api", "models")}
                for name, provider in self.original.get("models", {}).get("providers", {}).items()
            },
        }
        self.description = {
            "transport": "mcp",
            "agent": "openclaw",
            "routing_sha256": hashlib.sha256(
                json.dumps(routing, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }

    def open(self, phase):
        self.close()
        self.phase = phase
        self.token = secrets.token_urlsafe(32)
        config = _evaluation_config(
            self.original,
            mcp_url=self.mcp_url,
            headers=self.headers,
            user_id=self.args.user_id,
            memory_agent_id=self.args.memory_agent_id,
            gateway_token=self.token,
        )
        read_only = phase == "qa" and self.args.read_only_qa
        if read_only:
            config["tools"]["alsoAllow"] = [
                t for t in OGMEMORY_TOOLS if "write_memory" not in t
            ]
            config["mcp"]["servers"]["ogmemory"]["toolFilter"]["include"] = [
                "ogm_query_memory",
                "ogm_read_memory",
                "resources_*",
            ]
        config["gateway"]["port"] = self.args.openclaw_port
        runtime = (self.args.runtime_root or self.args.output_dir / "runtime") / phase / "home"
        self.process, env = _start_gateway(
            config, runtime, self.args.output_dir / f"{phase}-gateway.log", self.auth_source,
            command=self.args.openclaw_command,
        )
        self.state_dir = Path(env["OPENCLAW_STATE_DIR"])
        self.url = _gateway_url(config)
        _probe_openclaw(
            env, EXPECTED_TOOLS[:2] if read_only else EXPECTED_TOOLS,
            command=self.args.openclaw_command,
        )

    def close(self):
        if getattr(self, "process", None) is not None:
            _stop_gateway(self.process)
            self.process = None
        super().close()


if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent="openclaw", transports=("mcp", "gateway")))
