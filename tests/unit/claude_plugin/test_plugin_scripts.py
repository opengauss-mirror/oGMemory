"""Unit tests for claude-plugin hook scripts.

Covers: ogm_plugin_request, call_add_session_message,
        call_after_turn (parse/extract), call_compose (context builder).
"""

from __future__ import annotations

import importlib
import json
import os
import stat
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make plugins/claude-code/scripts importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "plugins/claude-code" / "scripts"
_REDACTION_FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures" / "trace" / "redaction_cases.json"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _reimport(name: str):
    """Force-reload a scripts module so monkeypatched env vars take effect."""
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def test_trace_keeps_exposed_model_reasoning(monkeypatch):
    monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "true")
    event = _reimport("call_add_session_message").claude_hook_to_event({
        "hook_event_name": "Stop", "last_assistant_message": "Use batches of 100.",
        "reasoning": "The contract sets the limit.",
    })
    assert event["assistant_message"] == "Use batches of 100."
    assert event["reasoning"] == "The contract sets the limit."

def test_stop_captures_current_transcript_thinking_and_text(monkeypatch, tmp_path):
    monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "true")
    transcript = tmp_path / "trace.jsonl"
    rows = [
        {"message": {"role": "user", "content": "Check the API."}},
        {"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Inspect the API contract."},
            {"type": "text", "text": "Use batches of 100."},
        ]}},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    event = _reimport("call_add_session_message").claude_hook_to_event({
        "hook_event_name": "Stop", "transcript_path": str(transcript),
    })
    assert event["assistant_message"] == "Use batches of 100."
    assert event["reasoning"] == "Inspect the API contract."


@pytest.mark.parametrize("hook_answer", [None, "Hook final answer."])
def test_stop_preserves_final_answer_after_long_intermediate_text(monkeypatch, tmp_path, hook_answer):
    monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "true")
    transcript = tmp_path / "trace.jsonl"
    rows = [
        {"message": {"role": "user", "content": "Check the API."}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "Planning. " * 2000}]}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Use batches of 100."},
            {"type": "text", "text": "Verify each response."},
        ]}},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    event = _reimport("call_add_session_message").claude_hook_to_event({
        "hook_event_name": "Stop", "transcript_path": str(transcript),
        "last_assistant_message": hook_answer,
    })
    assert event["assistant_message"] == (hook_answer or "Use batches of 100.\nVerify each response.")
    assert event["assistant_messages"].startswith("Planning. ")
    assert len(event["assistant_messages"]) < 10100


@pytest.mark.parametrize("character", ["字", "😀", "\x01"])
def test_stop_spool_bounds_complete_json_and_preserves_answer(monkeypatch, tmp_path, character):
    mod = _reimport("call_add_session_message")
    monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "true")
    monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)
    answer = "答" * 10000
    event = mod.claude_hook_to_event({
        "hook_event_name": "Stop", "last_assistant_message": answer,
        "reasoning": character * 10000, "assistant_messages": character * 10000,
    })
    with patch.object(mod.subprocess, "Popen"):
        assert mod.post_trace_event("sess-1", event)
    spool_files = list(tmp_path.rglob("*.json"))
    assert len(spool_files) == 1
    saved = json.loads(spool_files[0].read_text(encoding="utf-8"))
    envelope = mod._trace_envelope("sess-1", saved["event"], saved["event_id"], saved["event_time"])
    assert len(json.dumps(envelope, ensure_ascii=False).encode("utf-8")) <= mod._MAX_TRACE_EVENT_BODY_BYTES
    assert saved["event"]["assistant_message"] == answer
    assert saved["event"]["event_type"] == "turn_end"
    assert saved["event"]["truncated"] is True


# ===========================================================================
# ogm_plugin_request
# ===========================================================================


class TestPluginApiKey:
    def test_returns_og_auth_api_key_when_set(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "secret-root")
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() == "secret-root"

    def test_returns_none_when_no_key_set(self, monkeypatch):
        monkeypatch.delenv("OG_AUTH_API_KEY", raising=False)
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "  trimmed  ")
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() == "trimmed"


class TestHttpPluginHeaders:
    def test_no_key_returns_content_type_only(self, monkeypatch):
        monkeypatch.delenv("OG_AUTH_API_KEY", raising=False)
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod.http_plugin_headers() == {"Content-Type": "application/json"}

    def test_with_key_adds_auth_headers(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "my-key")
        monkeypatch.setenv("OG_ACCOUNT_ID", "acct-test")
        monkeypatch.setenv("OG_USER_ID", "u-test")
        mod = _reimport("ogm_plugin_request")
        monkeypatch.setattr(mod, "_try_ogmem_config", lambda: None)
        headers = mod.http_plugin_headers()
        assert headers["X-API-Key"] == "my-key"
        assert headers["X-Account-ID"] == "acct-test"
        assert headers["X-User-ID"] == "u-test"
        assert headers["Content-Type"] == "application/json"

    def test_no_agentid_in_headers(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "k")
        mod = _reimport("ogm_plugin_request")
        assert "X-Agent-ID" not in mod.http_plugin_headers()


class TestBaseCtx:
    def test_returns_required_keys(self, monkeypatch):
        monkeypatch.setenv("OG_ACCOUNT_ID", "acct-demo")
        monkeypatch.setenv("OG_USER_ID", "u-alice")
        monkeypatch.setenv("OG_AGENT_ID", "claude-code")
        mod = _reimport("ogm_plugin_request")
        monkeypatch.setattr(mod, "_try_ogmem_config", lambda: None)
        ctx = mod.base_ctx("sess-123")
        assert ctx["accountId"] == "acct-demo"
        assert ctx["userId"] == "u-alice"
        assert ctx["agentId"] == "claude-code"
        assert ctx["sessionId"] == "sess-123"

    def test_session_id_passed_through(self, monkeypatch):
        mod = _reimport("ogm_plugin_request")
        assert mod.base_ctx("my-session")["sessionId"] == "my-session"


class TestBaseApiUrl:
    def test_uses_og_memory_url_env(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://remote:9000")
        mod = _reimport("ogm_plugin_request")
        assert mod.base_api_url() == "http://remote:9000"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://remote:9000/")
        mod = _reimport("ogm_plugin_request")
        assert mod.base_api_url() == "http://remote:9000"

    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("OG_MEMORY_URL", raising=False)
        mod = _reimport("ogm_plugin_request")
        monkeypatch.setattr(mod, "_try_ogmem_config", lambda: None)
        assert mod.base_api_url() == "http://localhost:8090"

    def test_loads_monorepo_config_from_external_working_directory(self, monkeypatch, tmp_path):
        config_path = tmp_path / "ogmem.yaml"
        config_path.write_text(
            "service:\n"
            "  http_port: 19123\n"
            "identity:\n"
            "  account_id: configured-acct\n"
            "  user_id: configured-user\n"
            "  agent_id: configured-agent\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OGMEM_CONFIG", str(config_path))
        for name in (
            "OG_MEMORY_URL",
            "OG_MEMORY_ACCOUNT_ID",
            "OG_ACCOUNT_ID",
            "OG_MEMORY_USER_ID",
            "OG_USER_ID",
            "OG_MEMORY_AGENT_ID",
            "OG_AGENT_ID",
        ):
            monkeypatch.delenv(name, raising=False)

        mod = _reimport("ogm_plugin_request")

        assert mod._identity_and_url() == (
            "configured-acct",
            "configured-user",
            "configured-agent",
            "http://127.0.0.1:19123",
        )


class TestMcpRecovery:
    def test_recovery_is_disabled_without_explicit_opt_in(self, monkeypatch):
        monkeypatch.delenv("OGMEM_CLAUDE_MCP_RECOVERY_ENABLED", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod.mcp_recovery_enabled() is False

    def test_recovery_can_be_enabled_after_mcp_setup(self, monkeypatch):
        monkeypatch.setenv("OGMEM_CLAUDE_MCP_RECOVERY_ENABLED", "true")
        mod = _reimport("ogm_plugin_request")
        assert mod.mcp_recovery_enabled() is True


# ===========================================================================
# call_add_session_message
# ===========================================================================


class TestBuildToolText:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_add_session_message")

    def test_includes_tool_name(self):
        text = self.mod._build_tool_text(
            {"tool_name": "Bash", "tool_input": {}, "tool_response": {}}
        )
        assert "[PostToolUse] Bash" in text

    def test_includes_input_and_response(self):
        text = self.mod._build_tool_text(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/foo"},
                "tool_response": {"content": "bar"},
            }
        )
        assert "tool_input:" in text
        assert "tool_response:" in text
        assert "/foo" in text

    def test_handles_none_values(self):
        text = self.mod._build_tool_text(
            {"tool_name": "Edit", "tool_input": None, "tool_response": None}
        )
        assert "[PostToolUse] Edit" in text
        assert "null" in text

    def test_truncates_large_input(self):
        big = "x" * 20_000
        text = self.mod._build_tool_text(
            {"tool_name": "Bash", "tool_input": big, "tool_response": "ok"}
        )
        assert len(text) < 25_000

    def test_redacts_secrets_from_session_message(self):
        text = self.mod._build_tool_text(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "OPENAI_API_KEY=env-secret "
                        "AWS_SECRET_ACCESS_KEY='quoted-secret' "
                        "password=\"hunter2\" "
                        "tool --api-key cli-secret --token 'quoted-cli-secret' "
                        "curl -H 'Authorization: Bearer sk-secret'"
                    ),
                    "api_key": "raw-secret",
                },
                "tool_response": "ok",
            }
        )

        assert "sk-secret" not in text
        assert "env-secret" not in text
        assert "quoted-secret" not in text
        assert "hunter2" not in text
        assert "cli-secret" not in text
        assert "quoted-cli-secret" not in text
        assert "raw-secret" not in text
        assert text.count("{{") == 7

    def test_session_message_redaction_reuses_placeholders_across_chunks(self):
        text = self.mod._build_tool_text(
            {
                "tool_name": "Bash",
                "tool_input": "a@example.com",
                "tool_response": "b@example.com then a@example.com",
            }
        )

        assert "a@example.com" not in text
        assert "b@example.com" not in text
        assert text.count("{{email_1}}") == 2
        assert text.count("{{email_2}}") == 1


class TestBuildTraceEvent:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_add_session_message")

    @pytest.mark.parametrize(
        ("hook_name", "expected_type"),
        [
            ("SessionStart", "session_start"),
            ("UserPromptSubmit", "user_message"),
            ("PreToolUse", "tool_start"),
            ("PostToolUse", "tool_success"),
            ("PostToolUseFailure", "tool_failure"),
            ("Stop", "turn_end"),
            ("SubagentStop", "assistant_message"),
            ("PreCompact", "pre_compact"),
            ("SessionEnd", "session_end"),
        ],
    )
    def test_maps_supported_hook_events(self, hook_name, expected_type):
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": hook_name,
                "prompt": "hello",
                "tool_name": "Read",
                "last_assistant_message": "done",
                "agent_id": "claude-subagent",
            }
        )

        assert "hook_type" not in event
        assert "source" not in event
        assert "agent_id" not in event
        assert "subagent_id" not in event
        assert event["event_type"] == expected_type

    def test_bounds_tool_payloads(self, monkeypatch):
        monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "1")
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": "x" * 20_000,
                "tool_response": "y" * 20_000,
            }
        )

        assert len(event["tool_input"]) < 11_000
        assert len(event["tool_response"]) < 11_000

    def test_content_is_captured_by_default(self, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PostToolUse",
                "prompt": "private prompt",
                "tool_name": "Read",
                "tool_input": {"file_path": "/secret"},
                "tool_response": {"content": "private content"},
            }
        )

        assert event["prompt"] == "private prompt"
        assert "/secret" in event["tool_input"]
        assert "private content" in event["tool_response"]
        assert event["prompt_size"] > 0
        assert event["tool_input_size"] > 0
        assert event["tool_response_size"] > 0

    def test_size_fields_use_redacted_values(self, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PostToolUse",
                "prompt": "email a@example.com",
                "tool_name": "Read",
                "tool_input": {"api_key": "sk-secret"},
                "tool_response": {"content": "b@example.com"},
                "last_assistant_message": "done c@example.com",
            }
        )

        assert event["prompt_size"] == self.mod._content_size(event["prompt"])
        assert event["tool_input_size"] == self.mod._content_size(
            json.loads(event["tool_input"])
        )
        assert event["tool_response_size"] == self.mod._content_size(
            json.loads(event["tool_response"])
        )
        assert event["assistant_message_size"] == self.mod._content_size(
            event["assistant_message"]
        )
        assert event["tool_input_size"] != self.mod._content_size(
            {"api_key": "sk-secret"}
        )

    def test_size_fields_measure_full_redacted_payload_not_truncated_content(
        self, monkeypatch
    ):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "UserPromptSubmit",
                "prompt": "x" * 20_000,
            }
        )

        assert len(event["prompt"]) < event["prompt_size"]
        assert event["prompt_size"] == self.mod._content_size("x" * 20_000)

    def test_deep_payloads_are_depth_limited(self, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        payload = {}
        current = payload
        for _ in range(64):
            current["next"] = {}
            current = current["next"]

        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": payload,
            }
        )

        assert "[DepthLimit]" in event["tool_input"]

    def test_wide_payloads_are_node_limited(self, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        redacted = self.mod._redact({f"k{index}": index for index in range(3_000)})

        assert redacted["__truncated__"] == "[NodeLimit]"

    def test_zero_omits_content(self, monkeypatch):
        monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "0")
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PostToolUse",
                "prompt": "private prompt",
                "tool_name": "Read",
                "tool_input": {"file_path": "/secret"},
                "tool_response": {"content": "private content"},
            }
        )

        assert "prompt" not in event
        assert "tool_input" not in event
        assert "tool_response" not in event

        session_end = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "SessionEnd",
                "reason": "private@example.com finished",
            }
        )
        assert "reason" not in session_end

    def test_captured_content_redacts_secrets(self, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "curl -H 'authorization: bearer-value'",
                    "api_key": "raw-secret",
                },
            }
        )

        assert "raw-secret" not in event["tool_input"]
        assert "bearer-value" not in event["tool_input"]
        assert "{{api_key_1}}" in event["tool_input"]
        assert "{{authorization_1}}" in event["tool_input"]

    @pytest.mark.parametrize(
        "fixture",
        json.loads(_REDACTION_FIXTURES.read_text(encoding="utf-8")),
        ids=lambda fixture: fixture["name"],
    )
    def test_redacts_shared_fixture(self, fixture, monkeypatch):
        monkeypatch.delenv("OGMEM_TRACE_CAPTURE_CONTENT", raising=False)
        event = self.mod.claude_hook_to_event(
            {
                "hook_event_name": "UserPromptSubmit",
                "prompt": fixture["text"],
            }
        )

        serialized = json.dumps(event, ensure_ascii=False)
        for plaintext in (
            fixture["plaintext"]
            if isinstance(fixture["plaintext"], list)
            else [fixture["plaintext"]]
        ):
            assert plaintext not in serialized
        for placeholder in (
            fixture["placeholder"]
            if isinstance(fixture["placeholder"], list)
            else [fixture["placeholder"]]
        ):
            assert placeholder in serialized

    def test_repeated_plaintext_reuses_placeholder(self):
        redacted = self.mod._redact("one@example.com then one@example.com")

        assert redacted == "{{email_1}} then {{email_1}}"

    def test_circuit_state_is_isolated_by_endpoint_and_account(self):
        first = self.mod._trace_circuit_path("https://one.test/events", "acct-a")
        other_endpoint = self.mod._trace_circuit_path(
            "https://two.test/events", "acct-a"
        )
        other_account = self.mod._trace_circuit_path(
            "https://one.test/events", "acct-b"
        )

        assert len({first, other_endpoint, other_account}) == 3


class TestPostSessionMessage:
    def test_returns_true_on_200(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = mod.post_session_message("sess-1", "tool", "content")
        assert result is True

    def test_returns_false_on_url_error(self, monkeypatch):
        import urllib.error

        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = mod.post_session_message("sess-1", "tool", "content")
        assert result is False


class TestPostTraceEvent:
    def test_send_logs_event_type_without_removed_hook_type(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with (
            patch.object(mod.urllib.request, "urlopen", return_value=response),
            patch.object(mod, "log") as log,
        ):
            mod._send_trace_event("sess-1", {"event_type": "turn_end"})

        messages = [call.args[0] for call in log.call_args_list]
        assert any("event=turn_end" in message for message in messages)
        assert all("hook=None" not in message for message in messages)

    def test_send_serializes_stable_id_and_occurrence_time(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with patch.object(mod.urllib.request, "urlopen", return_value=response) as post:
            result = mod._send_trace_event(
                "sess-1",
                {"event_type": "tool_start"},
                "stable-id",
                "2026-07-15T08:00:00+00:00",
            )

        assert result == "sent"
        body = json.loads(post.call_args.args[0].data)
        assert body["request_id"] == "stable-id"
        assert body["timestamp"] == "2026-07-15T08:00:00+00:00"

    def test_private_spool_becomes_visible_only_after_complete_write(
        self, tmp_path
    ):
        mod = _reimport("call_add_session_message")
        target = tmp_path / "event.json"

        def assert_not_visible(_descriptor):
            assert list(tmp_path.glob("*.json")) == []

        with patch.object(mod.os, "fsync", side_effect=assert_not_visible):
            mod._write_private_spool(target, {"event": {"event_type": "tool_start"}})

        assert json.loads(target.read_text(encoding="utf-8"))["event"] == {
            "event_type": "tool_start"
        }
        assert list(tmp_path.glob("*.tmp")) == []

    def test_spools_and_starts_detached_sender(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with patch.object(mod.subprocess, "Popen") as popen:
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        spool_files = list(tmp_path.rglob("*.json"))
        assert len(spool_files) == 1
        payload = json.loads(spool_files[0].read_text(encoding="utf-8"))
        assert payload["event_id"] == spool_files[0].stem
        assert payload["event_time"].endswith("+00:00")
        assert payload["session_id"] == "sess-1"
        assert payload["event"] == {"event_type": "tool_start"}
        commands = [call.args[0] for call in popen.call_args_list]
        drain_command = next(command for command in commands if "--drain-traces" in command)
        watchdog_command = next(command for command in commands if "--watch-traces" in command)
        assert Path(drain_command[-2]).name == ".drain.lock"
        assert drain_command[-1]
        assert Path(watchdog_command[-3]).name == ".drain.lock"
        if os.name != "nt":
            assert stat.S_IMODE(spool_files[0].stat().st_mode) == 0o600
            assert stat.S_IMODE(spool_files[0].parent.stat().st_mode) == 0o700

    def test_oversized_trace_body_is_not_spooled(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with patch.object(mod.subprocess, "Popen") as popen:
            assert mod.post_trace_event(
                "sess-1",
                {
                    "event_type": "tool_success",
                    "tool_input": "😀" * 20_000,
                    "tool_response": "😀" * 20_000,
                },
            )

        assert list(tmp_path.rglob("*.json")) == []
        popen.assert_not_called()

    def test_permanent_http_error_drops_trace_without_opening_circuit(self, monkeypatch):
        mod = _reimport("call_add_session_message")

        class FailingOpener:
            def __call__(self, _req, timeout=None):
                raise urllib.error.HTTPError(
                    url="http://memory.test/api/v1/trace/events",
                    code=413,
                    msg="too large",
                    hdrs={},
                    fp=None,
                )

        monkeypatch.setattr(mod.urllib.request, "urlopen", FailingOpener())
        assert mod._send_trace_event("sess-1", {"event_type": "tool_start"}) == "drop"

    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    def test_transient_http_error_retries_trace_and_opens_circuit(
        self, status, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        circuit_states = []

        class FailingOpener:
            def __call__(self, _req, timeout=None):
                raise urllib.error.HTTPError(
                    url="http://memory.test/api/v1/trace/events",
                    code=status,
                    msg="transient",
                    hdrs={},
                    fp=None,
                )

        monkeypatch.setattr(mod.urllib.request, "urlopen", FailingOpener())
        monkeypatch.setattr(mod, "_set_trace_circuit", lambda _path, open_: circuit_states.append(open_))

        assert mod._send_trace_event("sess-1", {"event_type": "tool_start"}) == "retry"
        assert circuit_states == [True]

    def test_disappearing_old_spool_does_not_drop_new_event(
        self, tmp_path, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)
        partition = tmp_path / "partition"
        partition.mkdir()
        old = partition / "old.json"
        old.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(mod, "_trace_spool_partition", lambda *args: partition)
        original_sort_key = mod._spool_sort_key

        def vanish(path):
            path.unlink(missing_ok=True)
            return original_sort_key(path)

        with (
            patch.object(mod, "_spool_sort_key", side_effect=vanish),
            patch.object(mod, "_spawn_detached"),
        ):
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        spool_files = list(partition.glob("*.json"))
        assert len(spool_files) == 1
        assert json.loads(spool_files[0].read_text(encoding="utf-8"))["session_id"] == "sess-1"

    def test_lock_contention_starts_orphan_recovery_watchdog(
        self, tmp_path, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with (
            patch.object(mod, "_acquire_drain_lock", return_value=None),
            patch.object(mod, "_start_trace_watchdog", return_value=True) as watchdog,
        ):
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        watchdog.assert_called_once()

    def test_lock_contention_watchdog_failure_uses_fifo_fallback(
        self, tmp_path, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with (
            patch.object(mod, "_acquire_drain_lock", return_value=None),
            patch.object(mod, "_start_trace_watchdog", return_value=False),
            patch.object(mod, "_send_spooled_prefix", return_value=True) as fallback,
        ):
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        fallback.assert_called_once()

    def test_watchdog_takes_over_stale_drain_lease(self, tmp_path):
        mod = _reimport("call_add_session_message")
        (tmp_path / "event.json").write_text(
            json.dumps({"session_id": "s1", "event": {"event_type": "tool_start"}}),
            encoding="utf-8",
        )
        drain_lock = tmp_path / ".drain.lock"
        drain_lock.write_text("dead-worker", encoding="ascii")
        old = time.time() - mod._TRACE_LOCK_LEASE_SECONDS - 1
        os.utime(drain_lock, (old, old))
        watchdog_lock = tmp_path / ".watchdog.lock"
        watchdog_lock.write_text("watchdog-token", encoding="ascii")

        with patch.object(mod, "_drain_spooled_traces", return_value=0) as drain:
            assert (
                mod._watch_spooled_traces(
                    drain_lock, watchdog_lock, "watchdog-token"
                )
                == 0
            )

        drain.assert_called_once()
        assert not watchdog_lock.exists()

    def test_launch_failure_uses_bounded_direct_send(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with (
            patch.object(
                mod,
                "_spawn_detached",
                side_effect=[OSError("process limit"), OSError("process limit")],
            ),
            patch.object(mod, "_send_trace_event", return_value="sent") as send,
        ):
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        send.assert_called_once()
        assert list(tmp_path.rglob("*.json")) == []

    def test_watchdog_launch_failure_uses_ordered_fallback(
        self, tmp_path, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with (
            patch.object(
                mod,
                "_spawn_detached",
                side_effect=[None, OSError("watchdog process limit")],
            ),
            patch.object(mod, "_send_trace_event", return_value="sent") as send,
        ):
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        send.assert_called_once()
        assert list(tmp_path.rglob("*.json")) == []

    def test_double_launch_failure_never_sends_stop_ahead_of_backlog(
        self, tmp_path, monkeypatch
    ):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path.parent)
        monkeypatch.setattr(mod, "_trace_spool_partition", lambda *args: tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "older.json").write_text(
            json.dumps(
                {
                    "event_id": "older",
                    "event_time": "2020-01-01T00:00:00+00:00",
                    "session_id": "sess-1",
                    "event": {"event_type": "tool_start"},
                }
            ),
            encoding="utf-8",
        )
        sent = []

        def send(_session_id, event, _event_id, _event_time):
            sent.append(event["event_type"])
            return "sent"

        with (
            patch.object(
                mod,
                "_spawn_detached",
                side_effect=[OSError("drain limit"), OSError("watchdog limit")],
            ),
            patch.object(mod, "_send_trace_event", side_effect=send),
        ):
            assert mod.post_trace_event(
                "sess-1", {"event_type": "assistant_message", "hook_type": "Stop"}
            )

        assert sent == ["tool_start", "assistant_message"]
        assert list(tmp_path.glob("*.json")) == []

    def test_drain_retains_event_when_send_fails(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)
        spool = tmp_path / "event.json"
        spool.write_text(
            json.dumps({"session_id": "sess-1", "event": {"event_type": "tool_start"}}),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        with (
            patch.object(mod, "_send_trace_event", return_value=False),
            patch.object(mod, "_wait_with_lock_heartbeat", return_value=False),
        ):
            assert mod._drain_spooled_traces(lock, "owner-token") == 1

        assert spool.exists()
        assert not lock.exists()

    def test_drain_rescans_events_added_during_send(self, tmp_path):
        mod = _reimport("call_add_session_message")
        first = tmp_path / "first.json"
        first.write_text(
            json.dumps({"session_id": "s1", "event": {"event_type": "first"}}),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")
        sent = []

        sent_event_ids = []

        def send(session_id, event, event_id, event_time):
            sent.append(event["event_type"])
            sent_event_ids.append(event_id)
            assert event_time
            if event["event_type"] == "first":
                (tmp_path / "second.json").write_text(
                    json.dumps(
                        {"session_id": "s2", "event": {"event_type": "second"}}
                    ),
                    encoding="utf-8",
                )
            return "sent"

        with patch.object(mod, "_send_trace_event", side_effect=send):
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        assert sent == ["first", "second"]
        assert sent_event_ids == ["first", "second"]
        assert list(tmp_path.glob("*.json")) == []

    def test_drain_retries_without_a_later_hook(self, tmp_path):
        mod = _reimport("call_add_session_message")
        spool = tmp_path / "event.json"
        spool.write_text(
            json.dumps(
                {
                    "event_id": "stable-id",
                    "event_time": "2026-07-15T08:00:00+00:00",
                    "session_id": "s1",
                    "event": {"event_type": "first"},
                }
            ),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        calls = []

        def send(*args):
            calls.append(args)
            return "retry" if len(calls) == 1 else "sent"

        with (
            patch.object(mod, "_send_trace_event", side_effect=send),
            patch.object(mod, "_wait_with_lock_heartbeat", return_value=True),
        ):
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        assert not spool.exists()
        assert calls[0] == calls[1]
        assert calls[0][2:] == ("stable-id", "2026-07-15T08:00:00+00:00")

    def test_permanent_rejection_does_not_block_later_event(self, tmp_path):
        mod = _reimport("call_add_session_message")
        for name in ("first", "second"):
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"session_id": "s1", "event": {"event_type": name}}),
                encoding="utf-8",
            )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        with patch.object(mod, "_send_trace_event", side_effect=["drop", "sent"]):
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        assert list(tmp_path.glob("*.json")) == []

    def test_retry_reuses_spooled_event_id(self, tmp_path):
        mod = _reimport("call_add_session_message")
        spool = tmp_path / "stable-event-id.json"
        spool.write_text(
            json.dumps(
                {
                    "event_id": "stable-event-id",
                    "event_time": "2026-07-15T08:00:00+00:00",
                    "session_id": "s1",
                    "event": {"event_type": "tool_start"},
                }
            ),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        with patch.object(mod, "_send_trace_event", return_value="sent") as send:
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        send.assert_called_once_with(
            "s1",
            {"event_type": "tool_start"},
            "stable-event-id",
            "2026-07-15T08:00:00+00:00",
        )

    def test_spool_partition_isolated_by_endpoint_and_identity(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        partitions = {
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://two.test/events", "acct-a", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-b", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-b", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-b", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-a", "auth-b"
            ),
        }

        assert len(partitions) == 6


class TestAddSessionMessageMain:
    def test_skips_when_no_session_id(self, capsys):
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps({"tool_name": "Bash", "tool_input": {}, "tool_response": {}})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_read_only_tools_are_traced_but_not_added_to_messages(self, monkeypatch, capsys):
        """Read-only tools belong in traces but not the session message buffer."""
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps(
            {
                "session_id": "sess-1",
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x"},
                "tool_response": {"content": "y"},
            }
        )
        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "post_session_message") as mock_post,
            patch.object(mod, "post_trace_event", return_value=True) as mock_trace,
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0
        mock_post.assert_not_called()
        mock_trace.assert_called_once()
        assert "skip non-side-effect" in capsys.readouterr().err

    def test_logs_failure_when_post_fails(self, monkeypatch, capsys):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps(
            {
                "session_id": "sess-x",
                "tool_name": "Bash",
                "tool_input": {"command": "ls"},
                "tool_response": {"stdout": "file.txt"},
            }
        )
        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "post_session_message", return_value=False),
            patch.object(mod, "post_trace_event", return_value=True),
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit):
                mod.main()
        captured = capsys.readouterr()
        assert "Failed" in captured.err


class TestTraceHookRegistration:
    def test_registers_complete_trace_lifecycle(self):
        hooks_path = _SCRIPTS_DIR.parent / "hooks" / "hooks.json"
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))["hooks"]

        expected = {
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SubagentStop",
            "PreCompact",
            "SessionEnd",
        }
        assert expected <= hooks.keys()
        assert "matcher" not in hooks["PostToolUse"][0]

        for hook_name in expected:
            commands = [entry["command"] for entry in hooks[hook_name][0]["hooks"]]
            assert any("call_add_session_message.py" in command for command in commands)

    def test_readme_documents_trace_capture_contract(self):
        readme = (_SCRIPTS_DIR.parent / "README.md").read_text(encoding="utf-8")

        assert "/api/v1/trace/events" in readme
        assert "OGMEM_TRACE_CAPTURE_CONTENT=0" in readme
        assert "{{email_1}}" in readme
        assert "{{connection_string_1}}" in readme
        for hook_name, event_type in {
            "UserPromptSubmit": "user_message",
            "PreToolUse": "tool_start",
            "PostToolUse": "tool_success",
            "PostToolUseFailure": "tool_failure",
            "Stop": "turn_end",
            "SubagentStop": "assistant_message",
        }.items():
            assert hook_name in readme
            assert event_type in readme


# ===========================================================================
# call_after_turn — parse_messages / extract_text
# ===========================================================================


class TestExtractText:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_after_turn")

    def test_string_content_returned_as_is(self):
        assert self.mod.extract_text("hello") == "hello"

    def test_extracts_text_blocks_from_list(self):
        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert self.mod.extract_text(content) == "part1\npart2"

    def test_skips_non_text_blocks(self):
        content = [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "visible"}]
        assert self.mod.extract_text(content) == "visible"

    def test_empty_list_returns_empty(self):
        assert self.mod.extract_text([]) == ""


class TestParseMessages:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_after_turn")

    def _line(self, **kwargs) -> str:
        return json.dumps(kwargs)

    def test_parses_user_and_assistant(self):
        lines = "\n".join(
            [
                self._line(type="user", message={"role": "user", "content": "hi"}),
                self._line(
                    type="assistant",
                    message={"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                ),
            ]
        )
        msgs = self.mod.parse_messages(lines)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1] == {"role": "assistant", "content": "hello"}

    def test_filters_sidechain(self):
        lines = self._line(
            type="user", isSidechain=True, message={"role": "user", "content": "sub"}
        )
        assert self.mod.parse_messages(lines) == []

    def test_filters_api_error(self):
        lines = self._line(
            type="user", isApiErrorMessage=True, message={"role": "user", "content": "err"}
        )
        assert self.mod.parse_messages(lines) == []

    def test_filters_non_user_assistant_types(self):
        lines = self._line(type="tool_result", message={"role": "tool", "content": "data"})
        assert self.mod.parse_messages(lines) == []

    def test_filters_empty_content(self):
        lines = self._line(type="user", message={"role": "user", "content": [{"type": "tool_use"}]})
        assert self.mod.parse_messages(lines) == []

    def test_skips_malformed_json(self):
        lines = "not-json\n" + self._line(type="user", message={"role": "user", "content": "ok"})
        msgs = self.mod.parse_messages(lines)
        assert len(msgs) == 1


class TestOffsetReadWrite:
    def test_read_returns_zero_when_missing(self, tmp_path):
        mod = _reimport("call_after_turn")
        assert mod.read_offset(str(tmp_path / "missing.offset")) == 0

    def test_write_then_read(self, tmp_path):
        mod = _reimport("call_after_turn")
        path = str(tmp_path / "test.offset")
        mod.write_offset(path, 1234)
        assert mod.read_offset(path) == 1234


# ===========================================================================
# call_compose — build_additional_context
# ===========================================================================


class TestBuildAdditionalContext:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_compose")

    def test_includes_identity_and_evidence(self):
        data = {"identityContext": "I am Alice", "retrievedEvidence": "Past: X"}
        result = self.mod.build_additional_context(data)
        assert "I am Alice" in result
        assert "Past: X" in result
        assert result.startswith("[oG-Memory]")

    def test_returns_none_when_both_empty(self):
        assert self.mod.build_additional_context({}) is None
        assert (
            self.mod.build_additional_context({"identityContext": "", "retrievedEvidence": ""})
            is None
        )

    def test_works_with_only_identity(self):
        result = self.mod.build_additional_context({"identityContext": "profile data"})
        assert result is not None
        assert "profile data" in result

    def test_truncates_to_max_chars(self):
        data = {"identityContext": "x" * 20_000}
        result = self.mod.build_additional_context(data)
        assert len(result) <= self.mod.MAX_CONTEXT_CHARS + len("[oG-Memory]\n")


class TestCallComposeMain:
    def test_skips_short_prompt(self):
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "hi", "session_id": "s1"})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_skips_slash_command(self):
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "/og-compose test", "session_id": "s1"})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_outputs_additional_context_on_success(self, monkeypatch, capsys):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "what did we discuss", "session_id": "s1"})
        compose_response = {"identityContext": "Alice is a dev", "retrievedEvidence": ""}

        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "call_compose", return_value=compose_response),
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit):
                mod.main()

        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "Alice is a dev" in out["hookSpecificOutput"]["additionalContext"]
