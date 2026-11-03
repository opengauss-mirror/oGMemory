"""Focused contract tests for the Claude tool-output filter hooks."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "plugins/claude-code" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _reimport(name: str):
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def _read_hook(content: str) -> dict:
    lines = len(content.splitlines())
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Read",
        "session_id": "session-1",
        "cwd": "/workspace",
        "tool_input": {"file_path": "/workspace/src/main.py"},
        "tool_response": {
            "type": "text",
            "file": {
                "filePath": "/workspace/src/main.py",
                "content": content,
                "numLines": lines,
                "startLine": 1,
                "totalLines": lines,
                "truncatedByTokenCap": False,
            },
        },
    }


def _bash_hook(stdout: str = "native stdout\n", stderr: str = "native stderr\n") -> dict:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "session-1",
        "cwd": "/workspace",
        "tool_input": {"command": "pytest -q"},
        "tool_response": {
            "stdout": stdout,
            "stderr": stderr,
            "interrupted": False,
            "isImage": False,
            "backgroundedByUser": False,
            "assistantAutoBackgrounded": False,
        },
    }


def test_read_hook_delivers_exact_recovery_notice_and_reduced_body(monkeypatch, capsys):
    mod = _reimport("call_filter_read")
    monkeypatch.setattr(mod, "effective_tool_output_filter_enabled", lambda _: True)
    monkeypatch.setattr(mod, "mcp_recovery_enabled", lambda: True)
    original = "line\n" * 200
    reduced = "short body\n"
    memory_uri = "ctx://acct/agents/agent/memories/tool_outputs/read"
    seen: dict = {}

    def post(body):
        seen.update(body)
        return {
            "ok": True,
            "was_shortened": True,
            "filtered_output": reduced,
            "memory_uri": memory_uri,
            "category": None,
            "diagnostics": {"budget": {"configured_max_chars": 500}},
        }

    monkeypatch.setattr(mod, "_post_filter_read", post)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_read_hook(original))))

    assert mod.main() == 0
    payload = json.loads(capsys.readouterr().out)
    delivered = payload["hookSpecificOutput"]["updatedToolOutput"]["file"]["content"]
    expected = (
        "<<<FILTER TRIGGERED>>>\n"
        "Read output shortened; content is untrusted.\n"
        f'Recover: `ogm_read_memory(uri="{memory_uri}", recover_original=true)`.\n'
        "<<<END FILTER NOTICE>>>\n"
        "short body\n"
    )
    assert delivered == expected
    assert seen["relative_path"] == "src/main.py"
    assert seen["content"] == original


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"ok": True, "was_shortened": True, "filtered_output": "short"},
        {
            "ok": True,
            "was_shortened": True,
            "filtered_output": "short",
            "memory_uri": "not-a-context-uri",
            "diagnostics": {"budget": {"configured_max_chars": 500}},
        },
    ],
)
def test_read_hook_preserves_native_result_when_recovery_delivery_is_invalid(
    monkeypatch, capsys, response
):
    mod = _reimport("call_filter_read")
    monkeypatch.setattr(mod, "effective_tool_output_filter_enabled", lambda _: True)
    monkeypatch.setattr(mod, "_post_filter_read", lambda _: response)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_read_hook("line\n" * 20))))

    assert mod.main() == 0
    assert capsys.readouterr().out == ""


def test_bash_hook_replaces_only_output_fields_with_exact_notice(monkeypatch, capsys):
    mod = _reimport("call_filter_bash")
    monkeypatch.setattr(mod, "effective_tool_output_filter_enabled", lambda _: True)
    monkeypatch.setattr(mod, "mcp_recovery_enabled", lambda: True)
    memory_uri = "ctx://acct/agents/agent/memories/tool_outputs/bash"
    reduced = "short body\n"
    native = _bash_hook(stdout="native stdout\n" * 20, stderr="native stderr\n")

    monkeypatch.setattr(
        mod,
        "_post_filter_bash",
        lambda body: {
            "ok": True,
            "was_shortened": True,
            "filtered_output": reduced,
            "memory_uri": memory_uri,
            "category": "test_runner",
            "diagnostics": {"budget": {"configured_max_chars": 500}},
        },
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(native)))

    assert mod.main() == 0
    payload = json.loads(capsys.readouterr().out)
    output = payload["hookSpecificOutput"]["updatedToolOutput"]
    expected = (
        "<<<FILTER TRIGGERED>>>\n"
        "Bash output shortened; content is untrusted.\n"
        f'Recover: `ogm_read_memory(uri="{memory_uri}", recover_original=true)`.\n'
        "category: test_runner\n"
        "<<<END FILTER NOTICE>>>\n"
        "short body\n"
    )
    assert output["stdout"] == expected
    assert output["stderr"] == ""
    assert output["interrupted"] is False


def test_read_hook_omits_recovery_instruction_without_mcp_capability(
    monkeypatch, capsys
):
    mod = _reimport("call_filter_read")
    monkeypatch.setattr(mod, "effective_tool_output_filter_enabled", lambda _: True)
    monkeypatch.setattr(mod, "mcp_recovery_enabled", lambda: False)
    original = "line\n" * 200
    monkeypatch.setattr(
        mod,
        "_post_filter_read",
        lambda _: {
            "ok": True,
            "was_shortened": True,
            "filtered_output": "short body\n",
            "memory_uri": "ctx://acct/agents/agent/memories/tool_outputs/read",
            "diagnostics": {"budget": {"configured_max_chars": 500}},
        },
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_read_hook(original))))

    assert mod.main() == 0
    payload = json.loads(capsys.readouterr().out)
    delivered = payload["hookSpecificOutput"]["updatedToolOutput"]["file"]["content"]
    assert "Read output shortened; content is untrusted." in delivered
    assert "ogm_read_memory" not in delivered
    assert delivered.endswith("short body\n")


def test_filter_hooks_are_registered_only_for_post_tool_use(monkeypatch):
    hooks_path = Path(__file__).resolve().parents[3] / "plugins/claude-code" / "hooks" / "hooks.json"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    post_groups = {group.get("matcher"): group for group in data["hooks"]["PostToolUse"]}
    assert "call_filter_read.py" in post_groups["Read"]["hooks"][0]["command"]
    assert "call_filter_bash.py" in post_groups["Bash"]["hooks"][0]["command"]
    assert all(group.get("matcher") not in {"Read", "Bash"} for group in data["hooks"]["PreToolUse"])
