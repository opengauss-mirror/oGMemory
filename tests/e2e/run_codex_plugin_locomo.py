"""codex native plugin evaluation adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from agent_plugin_locomo import AgentHost, ROOT


class CodexHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.codex_command)
        self.home = self.runtime / "home"
        self.codex_home = self.home / ".codex"
        self.workspace = self.runtime / "workspace"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.env["HOME"] = str(self.home)
        self.env["CODEX_HOME"] = str(self.codex_home)
        self.env.update(
            {
                "OG_MEMORY_URL": self.args.ogmemory_url,
                "OG_MEMORY_ACCOUNT_ID": self.args.account_id,
                "OG_MEMORY_USER_ID": self.args.user_id,
                "OG_MEMORY_AGENT_ID": self.args.memory_agent_id,
                "OG_AUTH_API_KEY": self.args.ogmemory_api_key,
            }
        )
        config_path = self.codex_home / "config.toml"
        provider_config = (
            f"model = {json.dumps(self.args.llm_model)}\n"
            'model_provider = "locomo"\n\n'
            "[model_providers.locomo]\n"
            'name = "LoCoMO evaluator"\n'
            f"base_url = {json.dumps(self.args.llm_base_url)}\n"
            'env_key = "OGMEM_AGENT_LLM_API_KEY"\n'
            'wire_api = "responses"\n'
        )
        config_path.write_text(provider_config, encoding="utf-8")
        result = self._run([self.args.codex_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()
        self._run(
            [
                self.args.codex_command,
                "plugin",
                "marketplace",
                "add",
                str(ROOT),
                "--json",
            ],
            "prepare-marketplace",
            record=False,
        )
        self._run(
            [self.args.codex_command, "plugin", "add", "ogmemory@ogmemory", "--json"],
            "prepare-plugin",
            record=False,
        )

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.codex_command,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--dangerously-bypass-hook-trust",
                "--sandbox",
                "read-only",
                "--cd",
                str(self.workspace),
                *(["resume", self.sessions[label]] if label in self.sessions else []),
                prompt,
            ],
            label,
        )
        answer: list[str] = []
        usage = defaultdict(int)
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                self.remember_session(label, event["thread_id"])
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text")
                if text:
                    answer.append(str(text))
            if event.get("type") == "turn.completed":
                raw_usage = event.get("usage") or {}
                for key in ("input_tokens", "output_tokens"):
                    usage[key] += int(raw_usage.get(key, 0) or 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if not answer:
            raise RuntimeError("Codex returned no assistant text event")
        return "".join(answer).strip(), dict(usage)

if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='codex', transports=("plugin",)))
