"""Minimal Claude code-memory profile and policy coverage.

These tests exercise the plugin's local request/response projection with the
HTTP transport replaced by small recording doubles.  The Tool Output filter
hooks are covered by the parent Tool Output PR and are intentionally not
duplicated here.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "plugins/claude-code" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _reimport(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_code_policy_requires_search_before_edit():
    policy = _reimport("code_policy").render_code_policy(
        read_filter_enabled=True,
        bash_filter_enabled=True,
    )["text"]

    assert "call `ogm_search_code` at least once" in policy
    assert "to `ogm_edit_file` unchanged" in policy
    assert "instead of Claude's built-in `Edit` tool" in policy


def test_code_only_compose_posts_strict_flag_and_accepts_code_evidence(monkeypatch):
    monkeypatch.setenv("OGMEM_CODE_MEMORY_ONLY", "true")
    mod = _reimport("call_compose")
    seen: dict = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout

        class Response:
            status = 200

            def read(self):
                return json.dumps(
                    {
                        "messages": [],
                        "retrievedEvidence": "# code evidence",
                        "stats": {"code_memory_only": {"effective": True}},
                        "archiveCount": 0,
                        "archiveIncluded": False,
                        "identityContext": "",
                        "episodicContext": "",
                        "sessionContext": "",
                        "taskContext": "",
                        "systemPromptAddition": "",
                        "systemPromptSuffix": "",
                        "memoryUserMessage": "",
                        "openLoops": [],
                        "uncertainties": [],
                        "budgetUsedBySlot": {"retrieved": 3},
                    }
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod.call_compose("find parser", "session-1")

    assert result["retrievedEvidence"] == "# code evidence"
    assert seen["body"]["code_memory_only"] is True
    assert seen["timeout"] == mod.COMPOSE_TIMEOUT


def test_candidate_local_compose_timeout_fits_the_hook_supervisor(monkeypatch):
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_MODE", "candidate_local")
    mod = _reimport("call_compose")

    assert mod.effective_compose_timeout() == mod.CANDIDATE_LOCAL_COMPOSE_TIMEOUT
    assert mod.CANDIDATE_LOCAL_COMPOSE_TIMEOUT < mod.COMPOSE_HOOK_TIMEOUT


def test_candidate_local_mode_can_come_from_loaded_config(monkeypatch):
    monkeypatch.delenv("OGMEM_CODE_RETRIEVAL_MODE", raising=False)
    mod = _reimport("ogm_plugin_request")
    monkeypatch.setattr(
        mod,
        "_try_ogmem_config",
        lambda: SimpleNamespace(code_retrieval_mode="candidate_local"),
    )

    assert mod.effective_code_retrieval_mode() == "candidate_local"


def test_code_only_compose_rejects_generic_context(monkeypatch):
    monkeypatch.setenv("OGMEM_CODE_MEMORY_ONLY", "true")
    mod = _reimport("call_compose")
    response = {
        "retrievedEvidence": "code",
        "identityContext": "must be empty",
        "stats": {"code_memory_only": {"effective": True}},
        "archiveCount": 0,
        "archiveIncluded": False,
        "messages": [],
    }
    assert not mod._is_safe_code_only_response(response)


def test_code_only_trace_and_after_turn_hooks_skip_generic_transport(monkeypatch):
    monkeypatch.setenv("OGMEM_CODE_MEMORY_ONLY", "true")
    add_message = _reimport("call_add_session_message")
    after_turn = _reimport("call_after_turn")

    stdin = MagicMock()
    stdin.read.side_effect = AssertionError("code-only hooks must not read stdin")
    with (
        patch("sys.stdin", stdin),
        patch.object(add_message, "post_trace_event") as trace,
        patch.object(add_message, "post_session_message") as message,
    ):
        assert add_message.main() is None
    trace.assert_not_called()
    message.assert_not_called()

    with patch.object(after_turn, "post_after_turn") as post:
        assert after_turn.main() is None
    post.assert_not_called()


def test_hooks_keep_compose_and_parent_filter_entries():
    hooks_path = _SCRIPTS_DIR.parent / "hooks" / "hooks.json"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    user_prompt_commands = {
        item["command"]
        for item in data["hooks"]["UserPromptSubmit"][0]["hooks"]
    }
    post_groups = {group.get("matcher"): group for group in data["hooks"]["PostToolUse"]}

    assert any("call_compose.py" in command for command in user_prompt_commands)
    compose_hook = next(
        item for item in data["hooks"]["UserPromptSubmit"][0]["hooks"]
        if "call_compose.py" in item["command"]
    )
    assert compose_hook["timeout"] >= 365
    assert "Read" in post_groups and "Bash" in post_groups
    assert "call_filter_read.py" in post_groups["Read"]["hooks"][0]["command"]
    assert "call_filter_bash.py" in post_groups["Bash"]["hooks"][0]["command"]
