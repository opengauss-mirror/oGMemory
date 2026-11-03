"""dsh native plugin evaluation adapter."""

from __future__ import annotations

from pathlib import Path
from agent_plugin_locomo import AgentHost, ROOT, _write_yaml


class DshHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.dsh_command)
        self._require("pnpm")
        self.home = self.runtime / "home"
        self.env["HOME"] = str(self.home)
        self.env["DSH_HOME"] = str(self.home / ".dsh")
        self.env["DEEPSEEK_API_KEY"] = self.args.llm_api_key
        self.env.update(
            {
                "OGMEM_URL": self.args.ogmemory_url,
                "OGMEM_DSH_API_KEY": self.args.ogmemory_api_key,
                "OGMEM_ACCOUNT_ID": self.args.account_id,
                "OGMEM_USER_ID": self.args.user_id,
                "OGMEM_AGENT_ID": self.args.memory_agent_id,
                "OGMEM_DSH_TOOLS": "false",
                "OGMEM_DSH_FORCE_EXTRACT": "true",
                "DSH_PERMISSION_MODE": "read-only",
            }
        )
        if getattr(self.args, "disable_host_tools", False):
            self.tools_patch = self.runtime / "disable-host-tools.patch.yaml"
            _write_yaml(
                self.tools_patch,
                [
                    {"id": name, "disabled": True}
                    for name in (
                        "tool-bash", "tool-pwsh", "tool-jobs", "tool-fs", "tool-fs-search",
                        "tool-skill", "tool-subagent-control", "tool-subagent-list-agents",
                        "tool-subagent", "tool-subagent-fork", "tool-workflow", "tool-todo",
                        "tool-goal", "tool-ralph", "tool-web", "plan-mode",
                    )
                ],
            )
        package = self.runtime / "ogmemory-dsh-plugin.tgz"
        self._run(
            ["pnpm", "--dir", str(ROOT / "plugins/deepseek-harness"), "pack", "--out", str(package)],
            "prepare-package",
            timeout=max(self.args.agent_timeout, 600),
            record=False,
        )
        self._run(
            [
                self.args.dsh_command,
                "plugin",
                "--profile",
                "headless",
                "add",
                str(package),
            ],
            "prepare-plugin",
            timeout=max(self.args.agent_timeout, 600),
            record=False,
        )
        _write_yaml(
            Path(self.env["DSH_HOME"]) / "settings.yaml",
            {
                "agent-default-model": {"provider": "locomo", "model": self.args.llm_model},
                "llm-pi-ai": {
                    "providers": {
                        "locomo": {
                            "apiKeyEnv": "OGMEM_AGENT_LLM_API_KEY",
                            "api": "openai-completions",
                            "baseURL": self.args.llm_base_url,
                            "models": [{"id": self.args.llm_model}],
                        }
                    }
                },
            },
        )
        result = self._run([self.args.dsh_command, "--version"], "prepare-version", record=False)
        self.version = (result.stdout or result.stderr).strip()
        self._run(
            [self.args.dsh_command, "--profile", "headless", "--dump-config"],
            "prepare-config",
            record=False,
        )

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [self.args.dsh_command,
             *(["--patch", str(self.tools_patch)] if getattr(self.args, "disable_host_tools", False) else []),
             "--profile", "headless", prompt],
            label,
        )
        answer = result.stdout.strip()
        if not answer:
            raise RuntimeError("DSH returned no assistant text")
        return answer, {}

if __name__ == "__main__":
    from eval import main

    raise SystemExit(main(agent='dsh', transports=("plugin",)))
