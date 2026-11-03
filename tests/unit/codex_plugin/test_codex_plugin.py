from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins" / "ogmemory"
SCRIPT = PLUGIN / "scripts" / "ogmemory_codex_hook.py"

HOOK_EVENTS = {
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
}

def test_stop_captures_current_transcript_model_content(tmp_path):
    module = _load_module()
    transcript = tmp_path / "trace.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "task_started"}},
        {"type": "response_item", "payload": {
            "type": "reasoning", "summary": [{"type": "summary_text", "text": "Inspect the API contract."}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Use batches of 100."}],
        }},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    event = module._trace_event({"hook_event_name": "Stop", "transcript_path": str(transcript)})
    assert event["assistant_message"] == "Use batches of 100."
    assert event["reasoning"] == "Inspect the API contract."


@pytest.mark.parametrize("hook_answer", [None, "Hook final answer."])
def test_stop_preserves_final_answer_after_long_intermediate_text(tmp_path, hook_answer):
    module = _load_module()
    transcript = tmp_path / "trace.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "task_started"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Planning. " * 2000}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Use batches of 100."},
                        {"type": "output_text", "text": "Verify each response."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    event = module._trace_event({"hook_event_name": "Stop", "transcript_path": str(transcript),
                                 "last_assistant_message": hook_answer})
    assert event["assistant_message"] == (hook_answer or "Use batches of 100.\nVerify each response.")
    assert event["assistant_messages"].startswith("Planning. ")
    assert len(event["assistant_messages"]) <= module.MAX_TRACE_TEXT_CHARS


@pytest.mark.parametrize("character", ["字", "😀", "\x01"])
def test_stop_upload_bounds_complete_json_and_preserves_answer(monkeypatch, character):
    module = _load_module()
    requests = []
    monkeypatch.setattr(module, "_post_json", lambda path, body, timeout: requests.append(body))
    answer = "答" * 12000
    hook = {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": answer,
            "reasoning": character * 12000, "assistant_messages": character * 12000}
    module._capture_trace(hook)
    module._capture_trace(hook)
    assert requests[0]["request_id"] == requests[1]["request_id"]
    for body in requests:
        assert len(json.dumps(body, ensure_ascii=False).encode("utf-8")) <= module.MAX_TRACE_EVENT_BODY_BYTES
        assert body["event"]["assistant_message"] == answer
        assert body["event"]["event_type"] == "turn_end"
        assert body["event"]["truncated"] is True


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("recall", [True, False])
def test_independent_auto_controls(monkeypatch, tmp_path, capture, recall):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", str(capture))
    monkeypatch.setenv("OGMEM_AUTO_RECALL", str(recall))
    calls = []

    def post(path, body, timeout):
        calls.append(path)
        return {"ok": True, "prepareToken": "token", "identityContext": "memory"}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    for event in [
        "SessionStart", "UserPromptSubmit", "PreToolUse", "SubagentStart",
        "SubagentStop", "PreCompact", "PostCompact", "Stop", "SessionEnd",
    ]:
        module.handle_event({
            "hook_event_name": event, "session_id": "s", "turn_id": "t",
            "prompt": "remember this", "last_assistant_message": "done",
        })
    assert calls.count("/api/v1/trace/events") == 9
    assert calls.count("/api/v1/compose") == (4 if recall else 0)
    write_paths = set(calls) - {"/api/v1/trace/events", "/api/v1/compose"}
    assert write_paths == ({
        "/api/v1/bootstrap", "/api/v1/prepare_subagent_spawn",
        "/api/v1/on_subagent_ended", "/api/v1/prepare_compaction",
        "/api/v1/compact", "/api/v1/after_turn", "/api/v1/dispose",
    } if capture else set())
    if not capture:
        assert not list(tmp_path.rglob("*.json"))


def test_disabled_capture_preserves_pending_prompt(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    module._save_prompt("s", "t", "previous prompt")
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", "false")
    monkeypatch.setattr(module, "_post_json", lambda *args: {"ok": True})
    module._handle_stop({"session_id": "s", "turn_id": "t", "last_assistant_message": "done"})
    assert module._load_prompt("s", "t")[1] == "previous prompt"


@pytest.mark.parametrize("value, expected", [(" TRUE ", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("", False)])
def test_auto_control_values(monkeypatch, value, expected):
    module = _load_module()
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", value)
    assert module.auto_capture_enabled() is expected


def test_auto_control_config_and_env_precedence(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("OGMEM_AUTO_CAPTURE", raising=False)
    monkeypatch.setattr(module, "_plugin_config", lambda: {"auto_capture": "False"})
    assert module.auto_capture_enabled() is False
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", "true")
    assert module.auto_capture_enabled() is True


def _load_module():
    spec = importlib.util.spec_from_file_location("ogmemory_codex_hook", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_keeps_exposed_model_reasoning():
    event = _load_module()._trace_event({"hook_event_name": "Stop",
        "last_assistant_message": "Use batches of 100.", "reasoning": "The contract sets the limit."})
    assert event["assistant_message"] == "Use batches of 100."
    assert event["reasoning"] == "The contract sets the limit."


def _hook_timeout(hooks, event):
    return hooks["hooks"][event][0]["hooks"][0]["timeout"]


def test_plugin_package_contract():
    module = _load_module()
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())
    marketplace = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text())
    initialization_budget = (
        module.TRACE_TIMEOUT_SECONDS + module.WRITE_TIMEOUT_SECONDS + module.RECALL_TIMEOUT_SECONDS
    )

    assert manifest["name"] == "ogmemory"
    assert set(hooks["hooks"]) == HOOK_EVENTS
    assert _hook_timeout(hooks, "SessionStart") > initialization_budget
    assert _hook_timeout(hooks, "SubagentStart") > initialization_budget
    assert _hook_timeout(hooks, "SessionEnd") == 3
    assert (
        module.TRACE_TIMEOUT_SECONDS + module.RECALL_TIMEOUT_SECONDS + module.SKILL_CONTEXT_TIMEOUT_SECONDS
    ) < _hook_timeout(hooks, "UserPromptSubmit")
    assert marketplace["plugins"][0]["source"]["path"] == "./plugins/ogmemory"


def test_user_prompt_traces_recalls_and_caches_turn(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("OG_MEMORY_AGENT_ID", "codex")
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/compose":
            return {"identityContext": "Prefers Python", "retrievedEvidence": "Use uv"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    output = module.handle_event(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "prompt": "How should we package this project?",
        }
    )

    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/compose"]
    assert calls[0][1]["event"]["event_type"] == "user_message"
    assert calls[1][1]["agentId"] == "codex"
    assert "Prefers Python" in output["hookSpecificOutput"]["additionalContext"]
    assert module._load_prompt("session-1", "turn-1")[1] == "How should we package this project?"


def test_stop_traces_and_persists_completed_turn(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    module._save_prompt("session-1", "turn-1", "Remember this decision")
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    output = module.handle_event(
        {
            "hook_event_name": "Stop",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "last_assistant_message": "Saved.",
        }
    )

    assert output == {"continue": True}
    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/after_turn"]
    assert calls[1][1]["messages"] == [
        {"role": "user", "content": "Remember this decision"},
        {"role": "assistant", "content": "Saved."},
    ]
    assert calls[1][1]["forceExtract"] is True
    assert module._load_prompt("session-1", "turn-1") is None


def test_user_config_applies_to_http_calls(monkeypatch, tmp_path):
    config_path = tmp_path / "codex-plugin.json"
    config_path.write_text(
        json.dumps(
            {
                "url": "https://memory.example.com",
                "api_key": "tenant-key",
                "account_id": "account-1",
                "user_id": "user-1",
                "agent_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OG_MEMORY_CODEX_CONFIG", str(config_path))
    for name in (
        "OG_MEMORY_URL",
        "OGMEM_API_URL",
        "OG_AUTH_API_KEY",
        "OG_MEMORY_API_KEY",
        "OG_MEMORY_ACCOUNT_ID",
        "OG_ACCOUNT_ID",
        "OG_MEMORY_USER_ID",
        "OG_USER_ID",
        "OG_MEMORY_AGENT_ID",
        "OG_AGENT_ID",
    ):
        monkeypatch.setenv(name, "")
    module = _load_module()

    assert module._base_url() == "https://memory.example.com"
    assert module._identity("session-1") == {
        "accountId": "account-1",
        "userId": "user-1",
        "agentId": "agent-1",
        "sessionId": "session-1",
    }
    assert module._headers() == {
        "Content-Type": "application/json",
        "X-API-Key": "tenant-key",
        "X-Account-ID": "account-1",
        "X-User-ID": "user-1",
    }


def test_session_end_uses_three_second_hook_budget(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    module.handle_event(
        {
            "hook_event_name": "SessionEnd",
            "session_id": "session-1",
            "reason": "other",
        }
    )

    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/dispose"]
    assert [call[2] for call in calls] == [1, 1]


def test_compaction_prepare_token_is_committed(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/prepare_compaction":
            return {"ok": True, "prepareToken": "prepare-1"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    event = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "trigger": "auto",
    }
    module.handle_event({"hook_event_name": "PreCompact", **event})
    module.handle_event({"hook_event_name": "PostCompact", **event})

    compact_call = next(call for call in calls if call[0] == "/api/v1/compact")
    assert compact_call[1]["prepareToken"] == "prepare-1"


def test_tool_trace_redacts_credentials(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    module.handle_event(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "tool_name": "Bash",
            "tool_use_id": "tool-1",
            "tool_input": {"api_key": "secret-value"},
            "tool_response": "Authorization: Bearer secret-token",
        }
    )

    event = calls[0][1]["event"]
    assert event["tool_input"] == {"api_key": "[REDACTED]"}
    assert event["tool_response"] == "Authorization: Bearer [REDACTED]"


def test_permission_requests_use_distinct_trace_ids(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    base = {
        "hook_event_name": "PermissionRequest",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "permission_mode": "default",
    }
    module.handle_event(
        {
            **base,
            "tool_name": "Bash",
            "tool_input": {"command": "pwd"},
        }
    )
    module.handle_event(
        {
            **base,
            "tool_name": "apply_patch",
            "tool_input": {"patch": "*** Begin Patch"},
        }
    )

    assert calls[0][1]["request_id"] != calls[1][1]["request_id"]


@pytest.mark.parametrize(
    ("event", "extra", "expected_paths"),
    [
        (
            "SessionStart",
            {"source": "startup"},
            ["/api/v1/trace/events", "/api/v1/bootstrap", "/api/v1/compose"],
        ),
        ("SessionEnd", {"reason": "other"}, ["/api/v1/trace/events", "/api/v1/dispose"]),
        (
            "PreToolUse",
            {"tool_name": "Bash", "tool_input": {"command": "pwd"}},
            ["/api/v1/trace/events", "/api/v1/compose"],
        ),
        ("PermissionRequest", {"tool_name": "Bash"}, ["/api/v1/trace/events"]),
        ("PostToolUse", {"tool_name": "Bash", "tool_response": "done"}, ["/api/v1/trace/events"]),
        ("PreCompact", {"trigger": "auto"}, ["/api/v1/trace/events", "/api/v1/prepare_compaction"]),
        ("PostCompact", {"trigger": "auto"}, ["/api/v1/trace/events", "/api/v1/compact"]),
        (
            "SubagentStart",
            {"agent_id": "agent-1", "agent_type": "explorer"},
            ["/api/v1/trace/events", "/api/v1/prepare_subagent_spawn", "/api/v1/compose"],
        ),
        (
            "SubagentStop",
            {"agent_id": "agent-1", "agent_type": "explorer", "last_assistant_message": "Found it"},
            ["/api/v1/trace/events", "/api/v1/on_subagent_ended"],
        ),
    ],
)
def test_lifecycle_event_routes_to_ogmemory(
    monkeypatch,
    tmp_path,
    event,
    extra,
    expected_paths,
):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/compose":
            return {"retrievedEvidence": "Relevant project memory"}
        if path == "/api/v1/prepare_compaction":
            return {"ok": True, "prepareToken": "prepare-1"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    monkeypatch.setattr(module, "_get_json", lambda *args, **kwargs: None)
    module.handle_event(
        {
            "hook_event_name": event,
            "session_id": "session-1",
            "turn_id": "turn-1",
            "cwd": "/workspace",
            **extra,
        }
    )

    assert [call[0] for call in calls] == expected_paths
    assert calls[0][1]["event"]["event_type"] == module.EVENT_TYPES[event]


def test_user_prompt_skill_context_loads_best_published_skill(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("OG_MEMORY_AGENT_ID", "codex")
    paths = []

    def get(path, timeout=3):
        paths.append(path)
        if path.startswith("/api/v1/skills?"):
            return {
                "skills": [{
                    "name": "python-review",
                    "description": "Review Python code for security",
                    "score": 0.87,
                    "source": "generated",
                }]
            }
        return {"name": "python-review", "source": "generated", "content": "# Review\nRun tests."}

    monkeypatch.setattr(module, "_get_json", get)
    context = module._skill_context("review Python code for security")

    assert context is not None
    assert "python-review" in context
    assert "Run tests" in context
    assert paths == [
        "/api/v1/skills?q=review%20Python%20code%20for%20security&limit=3&agent_id=codex",
        "/api/v1/skills/python-review?source=generated&agent_id=codex&include_files=false",
    ]


def test_skill_context_reads_imported_source_and_agent_id(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("OG_MEMORY_AGENT_ID", "codex")
    paths = []

    def get(path, timeout=3):
        paths.append(path)
        if path.startswith("/api/v1/skills?"):
            return {"skills": [{"name": "same", "description": "Imported", "score": 0.9, "source": "imported"}]}
        return {"name": "same", "source": "imported", "content": "# Imported"}

    monkeypatch.setattr(module, "_get_json", get)
    context = module._skill_context("use same")

    assert "# Imported" in context
    assert paths == [
        "/api/v1/skills?q=use%20same&limit=3&agent_id=codex",
        "/api/v1/skills/same?source=imported&agent_id=codex&include_files=false",
    ]


def test_skill_http_request_carries_auth_and_agent_query(monkeypatch):
    module = _load_module()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"skills": []}'

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("OG_MEMORY_AGENT_ID", "agent-contract")
    monkeypatch.setenv("OG_AUTH_API_KEY", "")
    monkeypatch.setenv("OG_MEMORY_API_KEY", "unit-test-key")
    monkeypatch.setenv("OG_MEMORY_CODEX_CONFIG", "/tmp/ogmemory-test-config-missing.json")
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    assert module._get_json("/api/v1/skills?agent_id=agent-contract") == {"skills": []}
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["x-api-key"] == "unit-test-key"
    assert parse_qs(urlsplit(captured["url"]).query)["agent_id"] == ["agent-contract"]


def test_skill_query_redacts_secrets_and_bounds_encoded_url(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("OG_MEMORY_AGENT_ID", "codex")
    paths = []

    def get(path, timeout=3):
        paths.append(path)
        return {"skills": []}

    monkeypatch.setattr(module, "_get_json", get)
    module._skill_context("password=SECRET_TOKEN " + "中" * 2000)

    assert "SECRET_TOKEN" not in paths[0]
    assert len(paths[0].encode("ascii")) <= module.MAX_SKILL_QUERY_URL_BYTES + 80


def test_user_prompt_keeps_skill_when_memory_exceeds_output_budget(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("OGMEM_AUTO_RECALL", "true")
    monkeypatch.setattr(module, "_post_json", lambda path, body, timeout: {
        "identityContext": "M" * 3000,
    })
    monkeypatch.setattr(module, "_skill_context", lambda query: "[Matching Skill]\nSKILL-MARKER")

    output = module._recall("session", "prompt", "UserPromptSubmit")
    context = output["hookSpecificOutput"]["additionalContext"]

    assert "SKILL-MARKER" in context
    assert len(context) <= module.MAX_ADDITIONAL_CONTEXT_CHARS


@pytest.mark.parametrize("source", ["generated", "imported"])
@pytest.mark.parametrize("truncated", [None, "catalog", "detail"])
@pytest.mark.parametrize("frontmatter", [False, True])
@pytest.mark.parametrize("memory_size, body_size", [(0, 40), (3000, 2000)])
def test_prompt_http_preserves_body_after_long_skill_descriptions(tmp_path, source, memory_size, body_size, frontmatter, truncated):
    requests = []
    posts = []
    auth_keys = []
    body = "BODY-MARKER\n" + "步骤" * body_size
    document = body
    if frontmatter:
        document = "---\nname: skill-0\ndescription: " + "D" * 1024 + "\n---\n" + body

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            posts.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            auth_keys.append(self.headers.get("X-API-Key"))
            self.respond({"identityContext": "MEMORY-MARKER" + "M" * memory_size}
                         if self.path == "/api/v1/compose" else {"ok": True})

        def do_GET(self):
            requests.append(self.path)
            auth_keys.append(self.headers.get("X-API-Key"))
            stage = "catalog" if urlsplit(self.path).path == "/api/v1/skills" else "detail"
            if truncated == stage:
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'{"partial":')
                self.close_connection = True
                return
            if urlsplit(self.path).path == "/api/v1/skills":
                self.respond({"skills": [
                    {"name": f"skill-{i}", "description": "描述" * 350,
                     "score": 0.9, "source": source} for i in range(3)
                ]})
            else:
                self.respond({"content": document})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OG_", "OGMEM_"))}
    (tmp_path / "config.json").write_text(json.dumps({"agent_id": "中文-agent"}), encoding="utf-8")
    env.update(OG_MEMORY_URL=f"http://127.0.0.1:{server.server_port}",
               OG_MEMORY_CODEX_CONFIG=str(tmp_path / "config.json"),
               OG_MEMORY_API_KEY="test-key", OGMEM_AUTO_CAPTURE="false")
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], env=env, capture_output=True,
            input=json.dumps({"hook_event_name": "UserPromptSubmit",
                              "prompt": 'review code {"password": "SECRET with spaces"}'}),
            encoding="utf-8", timeout=15,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert result.returncode == 0, result.stderr
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert [path for path, _ in posts] == ["/api/v1/trace/events", "/api/v1/compose"]
    assert posts[0][1]["agent_id"] == "中文-agent"
    assert posts[1][1]["agentId"] == "中文-agent"
    if truncated:
        assert "MEMORY-MARKER" in context
        if truncated == "detail":
            assert "skill-0" in context
        return
    assert auth_keys == ["test-key"] * 4
    assert "BODY-MARKER" in context
    assert "MEMORY-MARKER" in context
    assert all(f"skill-{i}" in context for i in range(3))
    assert len(context) <= 2400
    skill_context = context.split("[oGMemory published Skills]", 1)[1]
    assert len("[oGMemory published Skills]" + skill_context) <= 800
    assert body[:300] in skill_context
    assert parse_qs(urlsplit(requests[0]).query)["q"] == [
        'review code {"password": [REDACTED]}'
    ]
    assert parse_qs(urlsplit(requests[1]).query) == {
        "source": [source], "agent_id": ["中文-agent"], "include_files": ["false"],
    }


@pytest.mark.parametrize("key", sorted(_load_module().SENSITIVE_KEYS))
@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("secret", ["SECRET with spaces", r"SECRET\"more", r"SECRET\'more"])
def test_skill_query_redacts_quoted_config_values(key, quote, secret):
    module = _load_module()
    prefix = f"Review {{{quote}{key}{quote}: "
    query = prefix + quote + secret + quote + "}"
    assert unquote(module._skill_query_url_part(query)) == prefix + "[REDACTED]}"

@pytest.mark.parametrize("query, expected", [
    ('Authorization: Basic dXNlcjpwYXNz', 'Authorization: Basic [REDACTED]'),
    ('OG_MEMORY_API_KEY=DEMO_SECRET', 'OG_MEMORY_API_KEY=[REDACTED]'),
    ('OG_AUTH_API_KEY=DEMO_SECRET', 'OG_AUTH_API_KEY=[REDACTED]'),
    ('{"authorization":"Bearer DEMO_SECRET_NOT_REAL"}', '{"authorization":[REDACTED]}'),
    ('{"token":"DEMO_SECRET_NOT_REAL"}', '{"token":[REDACTED]}'),
    ('token=DEMO_SECRET_NOT_REAL', 'token=[REDACTED]'),
    ('Cookie: session=DEMO_A; auth=DEMO_B', 'Cookie: [REDACTED]'),
])
def test_skill_query_redacts_review_credentials(query, expected):
    assert unquote(_load_module()._skill_query_url_part(query)) == expected


@pytest.mark.parametrize("stage", ["catalog", "detail"])
@pytest.mark.parametrize("slow_part", ["headers", "body"])
def test_skill_slow_stream_respects_wall_clock_budget(tmp_path, stage, slow_part):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, slow=False):
            body = json.dumps(payload).encode()
            headers = f"HTTP/1.0 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            try:
                for part, data in [("headers", headers), ("body", body)]:
                    if slow and part == slow_part:
                        for index in range(10):
                            self.wfile.write(data[len(data) * index // 10:len(data) * (index + 1) // 10])
                            self.wfile.flush()
                            time.sleep(0.4)
                    else:
                        self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond({"identityContext": "MEMORY-MARKER"})

        def do_GET(self):
            is_catalog = urlsplit(self.path).path == "/api/v1/skills"
            if is_catalog:
                if stage == "detail":
                    time.sleep(0.4)
                self.respond({"skills": [{"name": "review", "score": 0.9, "source": "generated"}]},
                             slow=stage == "catalog")
            else:
                self.respond({"content": "# Review code"}, slow=stage == "detail")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OG_", "OGMEM_"))}
    env.update(OG_MEMORY_URL=f"http://127.0.0.1:{server.server_port}",
               OG_MEMORY_CODEX_CONFIG=str(tmp_path / "config.json"), OGMEM_AUTO_CAPTURE="false")
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], env=env, capture_output=True,
            input=json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "review"}),
            encoding="utf-8", timeout=8,
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert result.returncode == 0, result.stderr
    assert "MEMORY-MARKER" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert elapsed < 3.2, f"Slow {stage} {slow_part} kept hook running for {elapsed:.2f}s"
