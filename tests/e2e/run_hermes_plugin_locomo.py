"""hermes native plugin evaluation adapter."""

from __future__ import annotations

import json
import uuid
from agent_plugin_locomo import AgentHost, ROOT, _copy_tree, _write_yaml


class HermesHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.hermes_command)
        self.home = self.runtime / "home"
        self.env["HOME"] = str(self.home)
        self.env["HERMES_HOME"] = str(self.home)
        self.env["OPENAI_API_KEY"] = self.args.llm_api_key
        self.env["OGMEM_AGENT_LLM_API_KEY"] = self.args.llm_api_key
        self.env.update(
            {
                "OGMEMORY_ENDPOINT": self.args.ogmemory_url,
                "OGMEMORY_API_KEY": self.args.ogmemory_api_key,
                "OGMEMORY_ACCOUNT_ID": self.args.account_id,
                "OGMEMORY_USER_ID": self.args.user_id,
                "OGMEMORY_AGENT_ID": self.args.memory_agent_id,
                "OGMEMORY_TIMEOUT_SECONDS": str(self.args.agent_timeout),
            }
        )
        _copy_tree(ROOT / "plugins/hermes" / "ogmemory", self.home / "plugins" / "ogmemory")
        _write_yaml(
            self.home / "config.yaml",
            {
                "model": {
                    "provider": "locomo",
                    "default": self.args.llm_model,
                    "base_url": self.args.llm_base_url,
                    "api_mode": "chat_completions",
                },
                "custom_providers": [
                    {
                        "name": "locomo",
                        "base_url": self.args.llm_base_url,
                        "key_env": "OGMEM_AGENT_LLM_API_KEY",
                        "api_mode": "chat_completions",
                        "model": self.args.llm_model,
                    }
                ],
                "memory": {"provider": "ogmemory"},
            },
        )
        result = self._run([self.args.hermes_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()
        self._run([self.args.hermes_command, "memory", "status"], "prepare-memory", record=False)

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        usage_file = self.logs / f"{label}-{uuid.uuid4().hex}-usage.json"
        result = self._run(
            [self.args.hermes_command, "--ignore-rules", "--usage-file", str(usage_file),
             *(["--resume", self.sessions[label]] if label in self.sessions else []),
             "-z", prompt],
            label,
        )
        answer = result.stdout.strip()
        if not answer:
            raise RuntimeError("Hermes returned no assistant text")
        raw_usage = json.loads(usage_file.read_text(encoding="utf-8"))
        sid = raw_usage.get("session_id")
        if (raw_usage.get("failed") or raw_usage.get("completed") is False
                or not isinstance(sid, str) or not sid.strip()):
            raise RuntimeError("Hermes returned no completed session")
        self.remember_session(label, sid)
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
            "cacheRead": int(raw_usage.get("cache_read_tokens", 0) or 0),
            "cacheWrite": int(raw_usage.get("cache_write_tokens", 0) or 0),
        }
        usage["total_tokens"] = int(
            raw_usage.get("total_tokens") or usage["input_tokens"] + usage["output_tokens"]
        )
        return answer, usage

if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='hermes', transports=("plugin",)))
