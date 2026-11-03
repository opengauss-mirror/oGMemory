"""claude-code native plugin evaluation adapter."""

from __future__ import annotations

import json
import uuid
from agent_plugin_locomo import AgentHost, ROOT, _copy_tree


class ClaudeCodeHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.claude_command)
        self.env["HOME"] = str(self.runtime / "home")
        self.env.pop("ANTHROPIC_API_KEY", None)
        self.env.pop("ANTHROPIC_AUTH_TOKEN", None)
        auth_env = "ANTHROPIC_AUTH_TOKEN" if self.args.claude_auth_token else "ANTHROPIC_API_KEY"
        self.env[auth_env] = self.args.llm_api_key
        self.env["ANTHROPIC_BASE_URL"] = (
            self.args.claude_base_url or self.args.llm_base_url.removesuffix("/v1")
        )
        self.env.update(
            {
                "OG_MEMORY_URL": self.args.ogmemory_url,
                "OG_MEMORY_ACCOUNT_ID": self.args.account_id,
                "OG_MEMORY_USER_ID": self.args.user_id,
                "OG_MEMORY_AGENT_ID": self.args.memory_agent_id,
                "OG_AUTH_API_KEY": self.args.ogmemory_api_key,
            }
        )
        self.plugin_dir = self.runtime / "plugin"
        _copy_tree(ROOT / "plugins/claude-code", self.plugin_dir)
        result = self._run([self.args.claude_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.claude_command,
                "-p",
                "--output-format",
                "json",
                *(["--resume", self.sessions[label]] if label in self.sessions else
                  ["--session-id", str(uuid.uuid5(uuid.NAMESPACE_URL, self.args.run_id + label))]),
                "--plugin-dir",
                str(self.plugin_dir),
                "--model",
                self.args.llm_model,
                prompt,
            ],
            label,
        )
        try:
            body = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Claude Code returned invalid JSON") from error
        if body.get("session_id"):
            self.remember_session(label, body["session_id"])
        answer = str(body.get("result") or "").strip()
        if not answer:
            raise RuntimeError(f"Claude Code returned no result: {body.get('subtype', '')}")
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return answer, usage

if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='claude-code', transports=("plugin",)))
