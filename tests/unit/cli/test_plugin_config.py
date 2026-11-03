from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.commands import onboard, plugin_config
from cli.main import build_parser

ROOT = Path(__file__).resolve().parents[3]


def args_for(tmp_path, agent="openclaw", *extra):
    return build_parser().parse_args([
        "onboard", "--mode", "plugin", "--non-interactive", "--agent", agent,
        "--output-dir", str(tmp_path / "export"), *extra,
    ])


def forbid(*args, **kwargs):
    raise AssertionError("configuration export must not run backend operations")


def test_portable_plugin_export_uses_explicit_mcp_endpoint(tmp_path):
    assert plugin_config.run(args_for(tmp_path, "agent-plugins")) == 1
    assert plugin_config.run(args_for(
        tmp_path, "agent-plugins", "--mcp-url", "https://memory.example:18091/mcp",
        "--ogmem-api-key", "test-key", "--account-id", "a", "--user-id", "u", "--agent-id", "g",
    )) == 0
    config = json.loads((tmp_path / "export/agent-plugin.json").read_text(encoding="utf-8"))
    assert config == {"mcp_url": "https://memory.example:18091/mcp", "api_key": "test-key",
                      "account_id": "a", "user_id": "u", "agent_id": "g", "timeout_ms": 30000}
    assert not (tmp_path / "export/mcp.config.json").exists()


@pytest.mark.parametrize("agent", [*plugin_config.NATIVE_AGENTS, "agent-plugins", "unadapted-agent"])
def test_export_needs_no_backend_and_preserves_active_files(tmp_path, monkeypatch, agent, capsys):
    monkeypatch.chdir(tmp_path)
    for path in ("config/ogmem.yaml", ".claude/settings.json", "openclaw.plugin.json"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("existing-user-config")
    for name in ("_step1_check", "_step2_llm", "_step3_embedding", "_step4_database", "_step_install_deps"):
        monkeypatch.setattr(onboard, name, forbid)
    monkeypatch.setattr(onboard.subprocess, "run", forbid)
    monkeypatch.setattr(plugin_config, "input_validated", forbid)
    monkeypatch.setattr(plugin_config, "input_masked", forbid)
    args = args_for(tmp_path, agent, "--ogmem-url", "https://memory.example/team/",
                    "--mcp-url", "https://mcp.example:9443/team/mcp",
                    "--ogmem-api-key", "tenant-secret", "--account-id", "acct-test",
                    "--user-id", "user-test", "--agent-id", "agent-test")
    assert onboard.run(args) == 0
    assert "tenant-secret" not in capsys.readouterr().out
    for path in ("config/ogmem.yaml", ".claude/settings.json", "openclaw.plugin.json"):
        assert (tmp_path / path).read_text() == "existing-user-config"
    assert (tmp_path / "export" / "README.md").exists()
    for file in (tmp_path / "export").iterdir():
        if file.suffix == ".json":
            assert isinstance(json.loads(file.read_text()), dict)
        if os.name != "nt":
            assert stat.S_IMODE(file.stat().st_mode) == 0o600


def test_openclaw_custom_port_and_manifest_compatibility(tmp_path):
    assert onboard.run(args_for(tmp_path, "openclaw", "--port", "18090")) == 0
    patch = json.loads((tmp_path / "export/openclaw.config.patch.json").read_text())
    assert set(patch) == {"plugins"}
    entry = patch["plugins"]["entries"]["og-memory-context-engine"]
    cfg = entry["config"]
    assert cfg["memoryApiBaseUrl"] == "http://127.0.0.1:18090"
    assert cfg["mode"] == "remote"
    assert patch["plugins"]["slots"]["contextEngine"] == "og-memory-context-engine"
    schema = json.loads((ROOT / "plugins/openclaw/openclaw.plugin.json").read_text())["configSchema"]
    assert not (cfg.keys() - schema["properties"].keys())
    for key, value in cfg.items():
        assert schema["properties"][key]["type"] == "string"
        if "enum" in schema["properties"][key]:
            assert value in schema["properties"][key]["enum"]


@pytest.mark.parametrize("agent", ["mcp", "cursor", "a-new-agent"])
def test_mcp_preserves_separate_endpoint_and_auth(tmp_path, agent):
    args = args_for(tmp_path, agent, "--ogmem-url", "https://api.example:18090",
                    "--mcp-url", "https://memory.example:18091/proxy/mcp/",
                    "--ogmem-api-key", "tenant", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    server = json.loads((tmp_path / "export/mcp.config.json").read_text())["mcpServers"]["ogmemory"]
    assert server == {
        "url": "https://memory.example:18091/proxy/mcp",
        "headers": {"X-API-Key": "tenant", "X-Account-ID": "a", "X-User-ID": "u", "X-Agent-ID": "g"},
    }
    assert "command" not in server
    assert "native Agent adapter" in (tmp_path / "export/README.md").read_text()


def test_mcp_requires_explicit_endpoint_without_tty(tmp_path, capsys):
    assert onboard.run(args_for(tmp_path, "other")) == 1
    assert "--mcp-url" in capsys.readouterr().out
    assert not (tmp_path / "export").exists()


def test_mcp_without_key_exports_explicit_placeholder(tmp_path):
    assert onboard.run(args_for(tmp_path, "mcp", "--mcp-url", "https://memory.example/mcp")) == 0
    cfg = json.loads((tmp_path / "export/mcp.config.json").read_text())
    assert cfg["mcpServers"]["ogmemory"]["headers"]["X-API-Key"] == "<OGMEM_TENANT_API_KEY>"


def test_explicit_plugin_mode_is_honored_in_interactive_mode(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.non_interactive = False
    monkeypatch.setattr(onboard, "select", forbid)
    monkeypatch.setattr(onboard, "_step1_check", forbid)
    monkeypatch.setattr(plugin_config.sys.stdin, "isatty", lambda: False)
    assert onboard.run(args) == 0


def test_interactive_selects_plugin_before_any_backend_prompt(tmp_path, monkeypatch):
    args = args_for(tmp_path, "codex", "--ogmem-url", "https://memory.example",
                    "--ogmem-api-key", "", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    args.mode = None
    args.non_interactive = False
    monkeypatch.setattr(onboard, "select", lambda *args, **kwargs: 2)
    monkeypatch.setattr(onboard, "_step1_check", forbid)
    monkeypatch.setattr(plugin_config.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugin_config, "input_validated", forbid)
    monkeypatch.setattr(plugin_config, "input_masked", forbid)
    assert onboard.run(args) == 0
    assert args.mode == "plugin"


def test_interactive_other_agent_prompts_for_mcp_url(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.agent = None
    args.non_interactive = False
    monkeypatch.setattr(plugin_config.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugin_config, "select", lambda *args: len(plugin_config.NATIVE_AGENTS))
    monkeypatch.setattr(plugin_config, "input_masked", lambda *args: "")
    prompts = []
    def answer(label, default):
        prompts.append(label)
        return "http://localhost:18091/custom/mcp" if label == "MCP URL" else default
    monkeypatch.setattr(plugin_config, "input_validated", answer)
    assert onboard.run(args) == 0
    assert prompts[0] == "MCP URL"
    assert (tmp_path / "export/mcp.config.json").exists()


@pytest.mark.parametrize("agent,filename", [
    ("claude", "claude.settings.patch.json"), ("claude_hooks", "claude.settings.patch.json"),
    ("dsh", "ogmemory.env.sh"),
])
def test_native_aliases(tmp_path, agent, filename):
    assert onboard.run(args_for(tmp_path, agent)) == 0
    assert (tmp_path / "export" / filename).exists()


@pytest.mark.parametrize("url", [
    "file:///tmp/memory", "localhost:8090", "https://user:secret@memory.example",
    "https://memory.example?token=secret", "https://memory.example#fragment",
    "https://memory.example:99999", "http://memory.example:0", "https://memory.example/api/v1",
    "https://memory.example/\nheader",
])
def test_invalid_api_url_is_rejected_without_writing(tmp_path, url, capsys):
    assert onboard.run(args_for(tmp_path, "openclaw", "--ogmem-url", url)) == 1
    assert "secret" not in capsys.readouterr().out
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("agent", ["opencode", "deepseek-harness"])
def test_existing_adapter_url_restrictions_are_respected(tmp_path, agent):
    assert onboard.run(args_for(tmp_path, agent, "--ogmem-url", "http://memory.example")) == 1
    assert not (tmp_path / "export").exists()


def test_explicit_service_key_env_is_private(tmp_path, monkeypatch, capsys):
    secret = "quotes'\" dollars$ backslash\\ unicode密钥"
    monkeypatch.setenv("TEST_OGMEM_KEY", secret)
    assert onboard.run(args_for(tmp_path, "codex", "--ogmem-api-key-env", "TEST_OGMEM_KEY")) == 0
    assert secret not in capsys.readouterr().out
    cfg = json.loads((tmp_path / "export/codex-plugin.json").read_text())
    assert cfg["api_key"] == secret


def test_missing_key_env_fails_without_output(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    assert onboard.run(args_for(tmp_path, "openclaw", "--ogmem-api-key-env", "TEST_MISSING_KEY")) == 1
    assert not (tmp_path / "export").exists()


def test_repeat_export_is_idempotent_and_conflicts_preserve_files(tmp_path):
    args = args_for(tmp_path)
    assert onboard.run(args) == 0
    files = {p.name: p.read_bytes() for p in (tmp_path / "export").iterdir()}
    assert onboard.run(args) == 0
    args.port = 28090
    assert onboard.run(args) == 1
    assert files == {p.name: p.read_bytes() for p in (tmp_path / "export").iterdir()}


def test_codex_runtime_reads_export(tmp_path, monkeypatch):
    args = args_for(tmp_path, "codex", "--ogmem-url", "https://memory.example",
                    "--ogmem-api-key", "key", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    spec = importlib.util.spec_from_file_location("export_codex_hook", ROOT / "plugins/ogmemory/scripts/ogmemory_codex_hook.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    monkeypatch.setenv("OG_MEMORY_CODEX_CONFIG", str(tmp_path / "export/codex-plugin.json"))
    for name in ("OG_MEMORY_URL", "OGMEM_API_URL", "OG_AUTH_API_KEY", "OG_MEMORY_API_KEY",
                 "OG_MEMORY_ACCOUNT_ID", "OG_ACCOUNT_ID", "OG_MEMORY_USER_ID", "OG_USER_ID",
                 "OG_MEMORY_AGENT_ID", "OG_AGENT_ID"):
        monkeypatch.delenv(name, raising=False)
    assert hook._base_url() == "https://memory.example"
    assert hook._identity("s") == {"accountId": "a", "userId": "u", "agentId": "g", "sessionId": "s"}
    assert hook._headers()["X-API-Key"] == "key"


def test_claude_runtime_uses_export_over_backend_identity(tmp_path, monkeypatch):
    args = args_for(tmp_path, "claude-code", "--ogmem-url", "https://memory.example",
                    "--ogmem-api-key", "key", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    values = json.loads((tmp_path / "export/claude.settings.patch.json").read_text())["env"]
    spec = importlib.util.spec_from_file_location("export_claude_hook", ROOT / "plugins/claude-code/scripts/ogm_plugin_request.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    monkeypatch.setattr(hook, "_try_ogmem_config", lambda: SimpleNamespace(
        account_id="backend-account", user_id="backend-user", agent_id="backend-agent", http_port=8090))
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert hook.base_api_url() == "https://memory.example"
    assert hook.base_ctx("s") == {"accountId": "a", "userId": "u", "agentId": "g", "sessionId": "s"}
    assert hook.http_plugin_headers()["X-Account-ID"] == "a"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell round trip")
@pytest.mark.parametrize("agent,key_name", [
    ("deepseek-harness", "OGMEM_DSH_API_KEY"), ("hermes", "OGMEMORY_API_KEY"),
])
def test_shell_export_preserves_values_without_command_execution(tmp_path, agent, key_name):
    secret = "quoted' dollar$(touch SHOULD_NOT_EXIST) backslash\\ unicode密钥"
    assert onboard.run(args_for(tmp_path, agent, "--ogmem-api-key", secret)) == 0
    script = tmp_path / "export/ogmemory.env.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; exec "$2" -c "$3"', "export-test",
         str(script), sys.executable, f"import os; print(os.environ[{key_name!r}])"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert result.stdout.rstrip("\n") == secret
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_cli_runs_with_only_standard_library(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "cli.main", "onboard", "--mode", "plugin",
         "--non-interactive", "--agent", "custom", "--mcp-url", "https://memory.example/mcp",
         "--output-dir", str(tmp_path / "export")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "export/mcp.config.json").exists()
    assert "unverified" in result.stdout


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is unavailable")
def test_opencode_runtime_loads_export(tmp_path):
    args = args_for(tmp_path, "opencode", "--ogmem-url", "https://memory.example/team",
                    "--ogmem-api-key", "key", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    script = """
const { loadConfig } = await import(process.argv[1]);
const config = await loadConfig({env: {}, configPath: process.argv[2]});
console.log(JSON.stringify(config));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script,
         (ROOT / "plugins/opencode/lib/config.mjs").as_uri(),
         str(tmp_path / "export/ogmemory-config.json")],
        capture_output=True, text=True, check=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg["endpoint"] == "https://memory.example/team"
    assert (cfg["apiKey"], cfg["accountId"], cfg["userId"], cfg["agentId"]) == ("key", "a", "u", "g")


@pytest.mark.skipif(os.name == "nt" or not shutil.which("node"), reason="POSIX and Node.js required")
def test_dsh_manifest_consumes_exported_environment(tmp_path):
    args = args_for(tmp_path, "dsh", "--ogmem-url", "https://memory.example/team",
                    "--ogmem-api-key", "key", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    script = """
import {readFileSync} from 'node:fs';
const {resolveConfig} = await import(process.argv[1]);
const manifest = readFileSync(process.argv[2], 'utf8');
const fields = {};
for (const match of manifest.matchAll(/^\\s+(\\w+): !!js (.+)$/gm)) {
  fields[match[1]] = new Function('process', 'return (' + match[2] + ')')(process);
}
fields.apiKeyEnv = manifest.match(/apiKeyEnv: (\\w+)/)[1];
const config = resolveConfig(fields);
console.log(JSON.stringify({...config, resolvedKey: process.env[config.apiKeyEnv]}));
"""
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; exec node --input-type=module -e "$@"', "dsh-export",
         str(tmp_path / "export/ogmemory.env.sh"), script,
         (ROOT / "plugins/deepseek-harness/lib/config.js").as_uri(),
         str(ROOT / "plugins/deepseek-harness/cordis.patch.yml")],
        capture_output=True, text=True, check=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg["baseUrl"] == "https://memory.example/team"
    assert (cfg["resolvedKey"], cfg["accountId"], cfg["userId"], cfg["agentId"]) == ("key", "a", "u", "g")


@pytest.mark.skipif(os.name == "nt", reason="POSIX environment script")
def test_hermes_runtime_consumes_exported_environment(tmp_path):
    args = args_for(tmp_path, "hermes", "--ogmem-url", "https://memory.example/team",
                    "--ogmem-api-key", "key", "--account-id", "a", "--user-id", "u", "--agent-id", "g")
    assert onboard.run(args) == 0
    script = """
import importlib.util, json, sys, types
parent = types.ModuleType('agent')
base = types.ModuleType('agent.memory_provider')
base.MemoryProvider = object
sys.modules.update({'agent': parent, 'agent.memory_provider': base})
spec = importlib.util.spec_from_file_location('hermes_export', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module._load_settings()))
"""
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; exec "$@"', "hermes-export",
         str(tmp_path / "export/ogmemory.env.sh"), sys.executable, "-c", script,
         str(ROOT / "plugins/hermes/ogmemory/__init__.py")],
        env={**os.environ, "HERMES_HOME": str(tmp_path / "empty-hermes")},
        capture_output=True, text=True, check=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg["endpoint"] == "https://memory.example/team"
    assert (cfg["api_key"], cfg["account_id"], cfg["user_id"], cfg["agent_id"]) == ("key", "a", "u", "g")


def test_noninteractive_default_mode_never_prompts(monkeypatch):
    monkeypatch.setattr(onboard, "select", forbid)
    assert onboard._step5_target(SimpleNamespace(mode=None, non_interactive=True)) == "local"
