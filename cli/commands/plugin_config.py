"""Export client configuration for an existing Agent and oGMemory service.

Only standard-library dependencies are needed. Exporting does not install
software, modify active Agent settings, contact services, or start processes.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cli.lib.colors import fail, info, ok
from cli.lib.prompt import input_masked, input_validated, select

NATIVE_AGENTS = ("openclaw", "claude-code", "codex", "opencode", "deepseek-harness", "hermes", "deepagents")
ALIASES = {"claude": "claude-code", "claude_hooks": "claude-code", "dsh": "deepseek-harness", "deepagent": "deepagents"}


@dataclass(frozen=True)
class Connection:
    url: str
    api_key: str
    account_id: str
    user_id: str
    agent_id: str


def normalize_url(value: str, *, mcp: bool = False) -> str:
    """Validate without echoing potentially secret URL components."""
    value = value.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("Expected an absolute HTTP(S) URL with a valid port") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or "\\" in value
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
        or port == 0
    ):
        raise ValueError("Use an absolute HTTP(S) URL without credentials, query, or fragment")
    value = value.rstrip("/")
    if not mcp and parsed.path.rstrip("/").endswith("/api/v1"):
        raise ValueError("--ogmem-url must be the API base URL without /api/v1")
    return value


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _environment_files(values: dict[str, str]) -> dict[str, str]:
    # Separate shell formats; never instruct users to source a dotenv file.
    shell = "# Source this file in the shell used to launch the Agent.\n"
    shell += "".join(f"export {name}={shlex.quote(value)}\n" for name, value in values.items())
    powershell = "# Dot-source this file before launching the Agent.\n"
    powershell += "".join(
        "$env:" + name + " = '" + value.replace("'", "''") + "'\n"
        for name, value in values.items()
    )
    return {"ogmemory.env.sh": shell, "ogmemory.env.ps1": powershell}


def build_artifacts(adapter: str, connection: Connection) -> dict[str, str]:
    """Render only formats consumed by the existing native adapters or MCP."""
    c = connection
    identity = {"accountId": c.account_id, "userId": c.user_id, "agentId": c.agent_id}
    if adapter == "agent-plugins":
        if urlsplit(c.url).scheme == "http" and urlsplit(c.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("The Agent Plugin requires HTTPS for non-loopback endpoints")
        files = {"agent-plugin.json": _json({
            "mcp_url": c.url, "api_key": c.api_key or "<OGMEM_TENANT_API_KEY>",
            "account_id": c.account_id, "user_id": c.user_id,
            "agent_id": c.agent_id, "timeout_ms": 30000,
        })}
        steps = (
            "1. Obtain the plugins/agent-plugins/ directory from the oGMemory repository or release ZIP. "
            "Install Node.js 22 or newer on the Agent machine.\n"
            "2. Copy agent-plugin.json to ~/.ogmemory/agent-plugin.json or set "
            "OGMEM_AGENT_PLUGIN_CONFIG to its absolute path. Replace any key placeholder.\n"
            "3. Load plugins/agent-plugins/ in an Agent Plugins 1.0 client. For a generic MCP client, "
            "configure command=node, args=[/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs] "
            "and load skills/ogmemory-memory/SKILL.md as a Skill or Agent instruction.\n"
            "4. Run node /absolute/path/plugins/agent-plugins/scripts/doctor.mjs. "
            "The expected tools are ogm_query_memory, ogm_read_memory and ogm_write_memory. "
            "The service must enable memory writes for the third tool.\n"
            "5. Follow docs/integrations/agent-plugins.md for OpenClaw, DSH, Claude Code and Codex. "
            "This is model-driven memory, without lifecycle hooks.\n"
        )
    elif adapter == "openclaw":
        files = {"openclaw.config.patch.json": _json({
            "plugins": {
                "slots": {"contextEngine": "og-memory-context-engine"},
                "entries": {"og-memory-context-engine": {
                    "enabled": True,
                    "hooks": {"allowConversationAccess": True},
                    "config": {
                        "mode": "remote", "memoryApiBaseUrl": c.url,
                        "authApiKey": c.api_key, "authAccountId": c.account_id,
                        **identity,
                    },
                }},
            },
        })}
        steps = (
            "1. If needed, install the plugin on the Agent machine:\n\n"
            "       openclaw plugins install --link /path/to/oGMemory/plugins/openclaw\n\n"
            "2. Merge openclaw.config.patch.json into the active OpenClaw config "
            "(OPENCLAW_CONFIG_PATH, or the selected profile's openclaw.json). "
            "Preserve unrelated settings and entries. If a plugins.allow list exists, "
            "append og-memory-context-engine. The contextEngine slot is exclusive; "
            "review any existing owner before replacing it.\n"
            "3. Restart the existing gateway using its normal service/container management.\n"
            "4. Check 'openclaw plugins list' and 'openclaw config get plugins.slots.contextEngine'.\n"
        )
    elif adapter == "claude-code":
        files = {"claude.settings.patch.json": _json({"env": {
            "OG_MEMORY_URL": c.url, "OG_AUTH_API_KEY": c.api_key,
            "OG_MEMORY_ACCOUNT_ID": c.account_id, "OG_MEMORY_USER_ID": c.user_id,
            "OG_MEMORY_AGENT_ID": c.agent_id,
        }})}
        steps = (
            "1. Merge the env entries from claude.settings.patch.json into the "
            "target project's .claude/settings.json or your user settings; preserve other settings.\n"
            "2. Load the oGMemory plugin. For a source checkout on the Agent machine:\n\n"
            "       claude --plugin-dir /path/to/oGMemory/plugins/claude-code\n\n"
            "   If already installed through your marketplace, reload that plugin instead.\n"
            "3. Review/trust its hooks and open a new session. Hook scripts require Python 3.\n"
        )
    elif adapter == "codex":
        files = {"codex-plugin.json": _json({
            "url": c.url, "api_key": c.api_key, "account_id": c.account_id,
            "user_id": c.user_id, "agent_id": c.agent_id,
        })}
        steps = (
            "1. Copy codex-plugin.json to ~/.ogmemory/codex-plugin.json, "
            "or set OG_MEMORY_CODEX_CONFIG to its path on the Agent machine.\n"
            "2. If needed, install the repository's Codex plugin:\n\n"
            "       codex plugin marketplace add https://gitcode.com/opengauss/oGMemory.git "
            "--ref dev --sparse .agents/plugins --sparse plugins/ogmemory\n"
            "       codex plugin add ogmemory@ogmemory\n\n"
            "3. Refresh Codex and trust its hooks in /hooks. Hook scripts require Python 3.\n"
        )
    elif adapter == "opencode":
        host = urlsplit(c.url).hostname or ""
        if urlsplit(c.url).scheme == "http" and not (
            host in {"localhost", "::1"} or re.fullmatch(r"127(?:\.\d{1,3}){3}", host)
        ):
            raise ValueError("The OpenCode adapter requires HTTPS for non-loopback endpoints")
        files = {
            "opencode.config.patch.json": _json({"plugin": ["@opengauss/ogmemory-opencode-plugin"]}),
            "ogmemory-config.json": _json({
                "enabled": True, "endpoint": c.url, "apiKey": c.api_key, **identity,
            }),
        }
        steps = (
            "1. Append the plugin from opencode.config.patch.json to the active "
            "OpenCode configuration's plugin list, preserving existing plugins.\n"
            "2. Copy ogmemory-config.json to ~/.config/opencode/ogmemory-config.json "
            "(Windows: %USERPROFILE%/.config/opencode/ogmemory-config.json).\n"
            "3. Restart OpenCode; it installs the declared npm plugin. Requires OpenCode >= 1.15.11 "
            "and Node.js >= 18. Source installation is described in plugins/opencode/INSTALL-ZH.md.\n"
        )
    elif adapter == "deepseek-harness":
        if urlsplit(c.url).scheme == "http" and urlsplit(c.url).hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("Use HTTPS for a remote DeepSeek Harness endpoint")
        files = _environment_files({
            "OGMEM_URL": c.url, "OGMEM_DSH_API_KEY": c.api_key,
            "OGMEM_ACCOUNT_ID": c.account_id, "OGMEM_USER_ID": c.user_id,
            "OGMEM_AGENT_ID": c.agent_id,
        })
        steps = (
            "1. If needed, install the plugin in the Agent's profile:\n\n"
            "       dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev\n\n"
            "2. Load one of the generated environment scripts, then launch 'dsh --profile web' "
            "from that shell. Substitute your existing profile for web if appropriate.\n"
            "3. Use 'dsh --profile web --dump-config' to check ogmemory-lifecycle is loaded.\n"
        )
    elif adapter == "deepagents":
        files = _environment_files({
            "OGMEM_URL": c.url, "OGMEM_API_KEY": c.api_key,
            "OGMEM_ACCOUNT_ID": c.account_id, "OGMEM_USER_ID": c.user_id,
            "OGMEM_AGENT_ID": c.agent_id,
        })
        files["ogmemory_deepagents.py"] = (
            "from deepagents import create_deep_agent\n"
            "from integrations.deepagents import OGMemoryMiddleware\n"
            "from langgraph.checkpoint.memory import InMemorySaver\n\n"
            "memory = OGMemoryMiddleware.from_env()\n"
            "agent = create_deep_agent(middleware=[memory], checkpointer=InMemorySaver())\n\n"
            "if __name__ == '__main__':\n"
            "    from uuid import uuid4\n"
            "    thread_id = str(uuid4())\n"
            "    result = agent.invoke(\n"
            "        {'messages': [{'role': 'user', 'content': 'What do you remember about me?'}]},\n"
            "        config={'configurable': {'thread_id': thread_id}},\n"
            "    )\n"
            "    print(result['messages'][-1].content)\n"
            "    memory.dispose(thread_id)\n"
        )
        steps = (
            "1. Install this oGMemory checkout on the Agent machine: "
            "python -m pip install '.[deepagents]'.\n"
            "2. Load one of the environment scripts and configure your Deep Agents "
            "model provider credentials separately.\n"
            "3. Run ogmemory_deepagents.py or pass OGMemoryMiddleware.from_env() "
            "in your existing create_deep_agent(middleware=[...]) call.\n"
            "4. Supply a stable configurable.thread_id per conversation; use a new ID "
            "for a new conversation. Call memory.dispose(thread_id) when it ends.\n"
        )
    elif adapter == "hermes":
        files = {"hermes.config.patch.yaml": "memory:\n  provider: ogmemory\n"}
        files.update(_environment_files({
            "OGMEMORY_ENDPOINT": c.url, "OGMEMORY_API_KEY": c.api_key,
            "OGMEMORY_ACCOUNT_ID": c.account_id, "OGMEMORY_USER_ID": c.user_id,
            "OGMEMORY_AGENT_ID": c.agent_id,
        }))
        steps = (
            "1. If needed, copy the repository's plugins/hermes/ogmemory/ directory "
            "to the active HERMES_HOME/plugins/ogmemory/ directory "
            "(default: ~/.hermes/plugins/ogmemory/) on the Agent machine.\n"
            "2. Merge hermes.config.patch.yaml into that profile's config.yaml. "
            "Review any existing memory provider before replacing it.\n"
            "3. Load one of the environment scripts before launching Hermes. For a managed "
            "service, persist those variables in its environment configuration.\n"
            "4. Run 'hermes memory status'.\n"
        )
    else:
        files = {"mcp.config.json": _json({"mcpServers": {"ogmemory": {
            "url": c.url,
            "headers": {
                "X-API-Key": c.api_key or "<OGMEM_TENANT_API_KEY>",
                "X-Account-ID": c.account_id, "X-User-ID": c.user_id,
                "X-Agent-ID": c.agent_id,
            },
        }}})}
        steps = (
            "1. The oGMemory Streamable HTTP MCP service must already be running "
            "with role control enabled. The MCP endpoint can have a different port "
            "or path from the oGMemory REST API.\n"
            "2. Merge mcp.config.json into an Agent supporting HTTP MCP and custom headers. "
            "Some clients require type: http or a different settings layout; use the "
            "provided URL and headers in that client's MCP settings. "
            "Replace <OGMEM_TENANT_API_KEY> if it is present.\n"
            "3. Reconnect MCP and list tools; verify an oGMemory memory query tool is available. "
            "An OAuth-only client needs a compatible authenticated gateway.\n"
            "4. MCP exposes tools. Automatic recall, turn capture, and compaction integration "
            "require a native Agent adapter and are not enabled by this export.\n"
        )

    files["README.md"] = (
        f"# oGMemory {adapter} connection configuration\n\n"
        "Status: configuration exported; installation and activation have not been verified.\n\n"
        "Use an existing Agent and an independently deployed oGMemory service. "
        "All application paths below refer to the Agent machine, which may differ from "
        "the machine generating this export. The URL must be reachable from the Agent "
        "process/container. Loopback addresses refer to that process's machine/container.\n\n"
        "Exports may contain a service API key. Keep them private. Existing environment "
        "overrides in the Agent can take precedence over these files.\n\n"
        "## Apply on the Agent machine\n\n" + steps +
        "\nEnvironment scripts, when supplied: POSIX 'source ./ogmemory.env.sh'; "
        "PowerShell '. ./ogmemory.env.ps1'. These scripts only set variables; launch "
        "the Agent separately in the same shell.\n\n"
        "## Verify\n\n"
        + (
            "Check MCP connection/authentication and tool listing from the Agent. "
            "A REST health check alone does not verify MCP.\n"
            if adapter in {"mcp", "agent-plugins"} else
            "Check /api/v1/health on the configured oGMemory API from the Agent machine, "
            "then complete a test conversation and verify its recall/writeback in a new session. "
            "A reachable server alone does not prove the plugin is installed or active.\n"
        )
    )
    return files


def _write_exports(directory: Path, files: dict[str, str]) -> None:
    # Check the complete set first so a re-run cannot partially replace an export.
    for name, content in files.items():
        target = directory / name
        if target.is_symlink() or (target.exists() and target.read_text(encoding="utf-8") != content):
            raise ValueError("Export files already exist with different content; choose a new --output-dir")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, content in files.items():
        target = directory / name
        if not target.exists():
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)


def run(args) -> int:
    """Export configuration without depending on server-side Python packages."""
    interactive = not getattr(args, "non_interactive", False) and sys.stdin.isatty()
    requested = getattr(args, "agent", None)
    if not requested:
        choices = [*NATIVE_AGENTS, "Other Agent (MCP)", "agent-plugins"]
        requested = choices[select("Existing Agent:", choices)] if interactive else "openclaw"
    requested = requested.strip().lower()
    agent = ALIASES.get(requested, requested)
    adapter = agent if agent in (*NATIVE_AGENTS, "agent-plugins") else "mcp"
    uses_mcp = adapter in {"mcp", "agent-plugins"}

    try:
        field = "mcp_url" if uses_mcp else "ogmem_url"
        url = getattr(args, field, None)
        if not url:
            if uses_mcp and not interactive:
                raise ValueError("MCP export requires --mcp-url with the full MCP port and path")
            default = "http://127.0.0.1:8091/mcp" if uses_mcp else f"http://127.0.0.1:{args.port}"
            url = input_validated("MCP URL" if uses_mcp else "oGMemory API base URL", default) if interactive else default
        url = normalize_url(url, mcp=uses_mcp)

        key = getattr(args, "ogmem_api_key", None)
        key_env = getattr(args, "ogmem_api_key_env", None)
        if key_env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env) or not os.environ.get(key_env):
                raise ValueError("--ogmem-api-key-env must name a non-empty environment variable")
            key = os.environ[key_env]
        elif key is None:
            key = input_masked("oGMemory service API key (Enter if not required)") if interactive else ""

        ids = []
        for name, default in (("account_id", "acct-demo"), ("user_id", "u-alice"), ("agent_id", adapter)):
            value = getattr(args, name, None)
            if value is None:
                value = input_validated(name.replace("_", " "), default) if interactive else default
            if not value.strip() or len(value) > 128 or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError("Identity fields must be non-empty, single-line strings up to 128 characters")
            ids.append(value.strip())
        if any(ord(c) < 32 or ord(c) == 127 for c in key):
            raise ValueError("The service API key must be a single-line string")
        connection = Connection(url, key, *ids)
        files = build_artifacts(adapter, connection)
        directory = Path(getattr(args, "output_dir", None) or Path.cwd() / "ogmem-client-config" / adapter).expanduser()
        _write_exports(directory, files)
    except (OSError, ValueError) as exc:
        print(fail(str(exc)))
        return 1

    if adapter == "mcp":
        print(info("MCP configuration selected; native lifecycle integration is not included."))
    print(ok(f"Connection configuration exported: {directory.resolve()}"))
    for name in files:
        print(f"  {name}")
    print(info("Apply the exported files using README.md on the Agent machine."))
    print(info("Status: exported; plugin installation, activation and service connectivity are unverified."))
    return 0
