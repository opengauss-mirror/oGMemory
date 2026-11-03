"""QwenPaw native plugin host for the shared evaluation runner."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid

import requests
from agent_plugin_locomo import ROOT, AgentHost, _write_json


class QwenPawHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.qwenpaw_command)
        self.http = requests.Session()
        self.http.trust_env = False
        extra_body = json.loads(os.environ.get("LOCOMO_CHAT_EXTRA_BODY", "{}"))
        self.home = self.runtime / "home"
        self.state = self.home / ".qwenpaw"
        workspace = self.state / "workspaces" / "default"
        workspace.mkdir(parents=True, exist_ok=True)
        self.env.update(
            {
                "HOME": str(self.home),
                "QWENPAW_WORKING_DIR": str(self.state),
                "QWENPAW_SECRET_DIR": str(self.home / ".qwenpaw.secret"),
                "QWENPAW_ENABLED_CHANNELS": "console",
                "OGMEMORY_TIMEOUT_SECONDS": str(self.args.agent_timeout),
                "OGMEMORY_COMPOSE_TIMEOUT_SECONDS": str(self.args.compose_timeout_ms / 1000),
                "OGMEMORY_TRACE_TIMEOUT_SECONDS": str(self.args.trace_timeout_ms / 1000),
            }
        )
        active_model = {"provider_id": "locomo", "model": self.args.llm_model}
        _write_json(
            self.state / "config.json",
            {
                "agents": {
                    "active_agent": "default",
                    "profiles": {
                        "default": {"id": "default", "workspace_dir": str(workspace)},
                    },
                },
                "channels": {"console": {"enabled": True}},
            },
        )
        _write_json(
            workspace / "agent.json",
            {
                "id": "default",
                "name": "LoCoMo",
                "workspace_dir": str(workspace),
                "active_model": active_model,
                "language": "en",
                "approval_level": "AUTO",
                "running": {
                    "memory_manager_backend": "none",
                    "context_manager_backend": "light",
                    "auto_title_config": {"enabled": False},
                },
                "channels": {"console": {"enabled": True}},
                "system_prompt_files": ["AGENTS.md"],
            },
        )
        (workspace / "AGENTS.md").write_text(
            "You are evaluating long-term conversation memory. Preserve the supplied conversation "
            "facts. For conversation transcripts, acknowledge briefly. For questions, answer directly "
            "using the recalled memory and retain exact names and dates.\n",
            encoding="utf-8",
        )
        providers = self.home / ".qwenpaw.secret" / "providers"
        provider_file = providers / "custom" / "locomo.json"
        _write_json(
            provider_file,
            {
                "id": "locomo",
                "name": "LoCoMo",
                "is_custom": True,
                "base_url": self.args.llm_base_url,
                "api_key": self.args.llm_api_key,
                "chat_model": "OpenAIChatModel",
                "models": [
                    {
                        "id": self.args.llm_model,
                        "name": self.args.llm_model,
                        "max_input_length": 128000,
                        "max_tokens": 2048,
                        "generate_kwargs": {
                            "temperature": 0,
                            **({"extra_body": extra_body} if extra_body else {}),
                        },
                    }
                ],
            },
        )
        provider_file.chmod(0o600)
        _write_json(providers / "active_model.json", active_model)
        version = self._run(
            [self.args.qwenpaw_command, "--version"], "prepare-version", record=False
        )
        self.version = (version.stdout or version.stderr).strip()
        self._run(
            [
                self.args.qwenpaw_command,
                "plugin",
                "install",
                "--force",
                str(ROOT / "plugins/qwenpaw"),
            ],
            "prepare-plugin",
            record=False,
        )
        self.gateway_log = (self.logs / "qwenpaw.log").open("w", encoding="utf-8")
        self.gateway = subprocess.Popen(
            [
                self.args.qwenpaw_command,
                "app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.args.qwenpaw_port),
            ],
            cwd=self.runtime,
            env=self.env,
            stdout=self.gateway_log,
            stderr=subprocess.STDOUT,
        )
        self.base = f"http://127.0.0.1:{self.args.qwenpaw_port}/api/agents/default/console"
        deadline = time.monotonic() + min(self.args.agent_timeout, 180)
        while time.monotonic() < deadline:
            if self.gateway.poll() is not None:
                raise RuntimeError("QwenPaw exited during startup; see qwenpaw.log")
            try:
                response = self.http.get(self.base + "/chat/task/readiness", timeout=2)
                if response.status_code == 404 and "Task not found" in response.text:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("QwenPaw Console API did not become ready")

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        started = time.monotonic()
        sid = self.sessions.get(label) or "qp-" + uuid.uuid4().hex
        self.remember_session(label, sid)
        response = self.http.post(
            self.base + "/chat/task",
            json={
                "input": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "session_id": sid,
                "user_id": self.args.user_id,
                "channel": "console",
                "timeout": self.args.agent_timeout,
            },
            timeout=30,
        )
        response.raise_for_status()
        task_id = response.json()["task_id"]
        while time.monotonic() - started < self.args.agent_timeout:
            response = self.http.get(self.base + f"/chat/task/{task_id}", timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "finished":
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"QwenPaw task timed out: {label}")
        result = data["result"]
        log_path = self.logs / f"{label}-{uuid.uuid4().hex[:8]}.json"
        _write_json(log_path, result)
        if result.get("status") not in {"completed", "finished"}:
            raise RuntimeError(f"QwenPaw task failed; see {log_path}")
        text = "\n".join(
            block["text"]
            for message in result.get("output", [])
            for block in message.get("content", [])
            if block.get("type") == "text" and block.get("text")
        ).strip()
        if not text:
            raise RuntimeError(f"QwenPaw returned no assistant text; see {log_path}")
        self.remember_session(label, result.get("session_id") or sid)
        self.invocations.append(
            {
                "label": label,
                "duration_seconds": time.monotonic() - started,
                "log": str(log_path.relative_to(self.args.output_dir)),
                "exit_code": 0,
            }
        )
        return text, self._turn_usage(result.get("session_id") or sid)

    def _turn_usage(self, session_id: str) -> dict[str, int]:
        safe_sid = re.sub(r'[\\/:*?"<>|]', "--", session_id)
        safe_user = re.sub(r'[\\/:*?"<>|]', "--", self.args.user_id)
        filename = f"{safe_user}_{safe_sid}.json" if safe_user and safe_user != safe_sid else f"{safe_sid}.json"
        path = self.state / "workspaces/default/sessions/console" / filename
        state = json.loads(path.read_text(encoding="utf-8"))["agent"]["state"]
        reply_id = (state.get("reply_context") or {}).get("reply_id") or state.get("reply_id")
        calls = [
            message["usage"] for message in state["context"]
            if message["role"] == "assistant" and message["id"] == reply_id and message.get("usage")
        ]
        if not calls:
            raise RuntimeError(f"QwenPaw turn usage missing: {session_id}")
        input_tokens = sum(int(call.get("input_tokens", 0) or 0) for call in calls)
        output_tokens = sum(int(call.get("output_tokens", 0) or 0) for call in calls)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def close(self) -> None:
        if getattr(self, "gateway", None) is not None:
            self.gateway.terminate()
            try:
                self.gateway.wait(timeout=self.args.agent_timeout)
            except subprocess.TimeoutExpired:
                self.gateway.kill()
                self.gateway.wait()
            self.gateway_log.close()
        super().close()
        if getattr(self, "http", None) is not None:
            self.http.close()


if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent="qwenpaw", transports=("plugin",)))
