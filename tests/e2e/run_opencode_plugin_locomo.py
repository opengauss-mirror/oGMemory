"""opencode native plugin evaluation adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from agent_plugin_locomo import AgentHost, ROOT, _write_json


class OpenCodeHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.opencode_command)
        self.env["HOME"] = str(self.runtime / "home")
        config = {
            "plugin": [str(ROOT / "plugins/opencode" / "index.mjs")],
            "provider": {
                "locomo": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "LoCoMO evaluator",
                    "options": {
                        "baseURL": self.args.llm_base_url,
                        "apiKey": "{env:OGMEM_AGENT_LLM_API_KEY}",
                    },
                    "models": {self.args.llm_model: {"name": self.args.llm_model}},
                },
            },
        }
        if getattr(self.args, "disable_host_tools", False):
            config["tools"] = {"*": False}
        self.env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        _write_json(
            Path(self.env["HOME"]) / ".config" / "opencode" / "ogmemory-config.json",
            {
                "composeTimeoutMs": getattr(self.args, "compose_timeout_ms", 60000),
                "traceTimeoutMs": getattr(self.args, "trace_timeout_ms", 60000),
            },
        )
        (self.runtime / "workspace").mkdir(parents=True, exist_ok=True)
        result = self._run([self.args.opencode_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()
        self._run([self.args.opencode_command, "db", "SELECT 1"], "prepare-database", record=False)

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.opencode_command,
                "run",
                "--format",
                "json",
                *(["--session", self.sessions[label]] if label in self.sessions else []),
                "--model",
                f"locomo/{self.args.llm_model}",
                "--dir",
                str(self.runtime / "workspace"),
            ],
            label,
            input_text=prompt,
        )
        answer: list[str] = []
        usage = defaultdict(int)
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error":
                raise RuntimeError(f"OpenCode host error: {event.get('error', event)}")
            if event.get("sessionID"):
                self.remember_session(label, event["sessionID"])
            part = event.get("part") or event.get("properties", {}).get("part") or {}
            if event.get("type") == "text" or part.get("type") == "text":
                text = part.get("text") or event.get("text")
                if text:
                    answer.append(str(text))
            tokens = part.get("tokens") or event.get("tokens") or {}
            cache = tokens.get("cache") or {}
            cache_read = int(cache.get("read", 0) or 0)
            cache_write = int(cache.get("write", 0) or 0)
            usage["input_tokens"] += int(tokens.get("input", 0) or 0) + cache_read + cache_write
            usage["output_tokens"] += int(tokens.get("output", 0) or 0)
            usage["cacheRead"] += cache_read
            usage["cacheWrite"] += cache_write
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        reply = "".join(answer).strip()
        if not reply:
            raise RuntimeError("OpenCode returned no assistant text")
        return reply, dict(usage)

if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='opencode', transports=("plugin",)))
