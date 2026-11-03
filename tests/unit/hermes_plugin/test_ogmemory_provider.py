from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins/hermes" / "ogmemory" / "__init__.py"


class _MemoryProvider:
    pass


agent_module = types.ModuleType("agent")
memory_provider_module = types.ModuleType("agent.memory_provider")
memory_provider_module.MemoryProvider = _MemoryProvider
sys.modules.setdefault("agent", agent_module)
sys.modules.setdefault("agent.memory_provider", memory_provider_module)

spec = importlib.util.spec_from_file_location("ogmemory_hermes_plugin", PLUGIN)
plugin = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(plugin)


class FakeClient:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []
        self.account_id = "acct"
        self.user_id = "user"
        self.agent_id = "hermes"

    def identity(self, session_id):
        return {
            "accountId": self.account_id,
            "userId": self.user_id,
            "agentId": self.agent_id,
            "sessionId": session_id,
        }

    def get(self, path):
        return {"status": "ok"}

    def post(self, path, payload, *, timeout=None):
        self.calls.append((path, payload))
        defaults = {
            "/api/v1/after_turn": {"ok": True},
            "/api/v1/call/wait_until_idle": {"idle": True},
            "/api/v1/compact": {"ok": True},
            "/api/v1/dispose": {"disposed": True},
        }
        return self.responses.get(path, defaults.get(path, {}))


@pytest.mark.parametrize("output", ["答" * 100_000, [{"type": "text", "text": "\u0001" * 100_000}]])
def test_sync_turn_bounds_native_tool_evidence_and_preserves_calls(output):
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient()
    provider._submit = provider._client.post
    messages = [
        {"role": "user", "content": "Read api.md."},
        {"role": "assistant", "content": "Reading.", "tool_calls": [
            {"id": "call-1", "function": {"name": "read_file", "arguments": '{"path":"api.md"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": output},
        {"role": "assistant", "content": "File verified."},
    ]
    original = json.dumps(messages)

    provider.sync_turn("Read api.md.", "File verified.", session_id="budget", messages=messages)

    _, payload = provider._client.calls[0]
    trace = payload["traceMessages"]
    assert len(json.dumps(payload).encode()) <= 128 * 1024
    assert trace[1]["tool_calls"] == messages[1]["tool_calls"]
    assert trace[2]["tool_call_id"] == "call-1"
    assert trace[-1]["content"] == "File verified."
    assert payload["messages"] == [messages[0], messages[-1]]
    assert json.dumps(messages) == original


def test_sync_turn_preserves_conversation_with_oversized_trace_structure(monkeypatch):
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient()
    provider._submit = provider._client.post
    monkeypatch.setattr(plugin, "_HTTP_BODY_BYTES", 1024)
    messages = [{"role": "user", "content": "Review."}] + [
        {"role": "assistant", "content": "Observation."} for _ in range(100)
    ]

    provider.sync_turn("Review.", "Verified.", session_id="wide", messages=messages)

    _, payload = provider._client.calls[0]
    assert len(json.dumps(payload).encode()) <= 1024
    assert payload["messages"] == [
        {"role": "user", "content": "Review."}, {"role": "assistant", "content": "Verified."},
    ]
    assert payload["clientRequestId"]


class FailingClient(FakeClient):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def post(self, path, payload, *, timeout=None):
        if path == "/api/v1/after_turn" and self.failures:
            self.calls.append((path, payload))
            self.failures -= 1
            raise RuntimeError("temporary failure")
        return super().post(path, payload, timeout=timeout)


class BlockingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def post(self, path, payload, *, timeout=None):
        if path == "/api/v1/after_turn":
            self.calls.append((path, payload))
            self.started.set()
            self.release.wait()
            return {"ok": True}
        return super().post(path, payload, timeout=timeout)


class BlockingCommitClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def post(self, path, payload, *, timeout=None):
        if path == "/api/v1/call/wait_until_idle":
            self.calls.append((path, payload))
            self.started.set()
            if not self.release.wait(timeout=timeout):
                raise TimeoutError("commit deadline expired")
            return {"idle": True}
        return super().post(path, payload, timeout=timeout)


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("recall", [True, False])
def test_independent_auto_controls(monkeypatch, capture, recall):
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", str(capture))
    monkeypatch.setenv("OGMEM_AUTO_RECALL", str(recall))
    provider = plugin.OgMemoryProvider()
    provider._session_id = "s"
    provider._client = FakeClient({"/api/v1/compose": {"identityContext": "memory"}})
    queued = []
    provider._submit = lambda path, payload: queued.append((path, payload))
    assert bool(provider.prefetch("query")) is recall
    provider.sync_turn("user", "assistant")
    provider.on_memory_write("add", "memory", "fact")
    assert len(queued) == (2 if capture else 0)
    provider.on_pre_compress([])
    provider.on_session_end([])
    provider.on_session_switch("new-session")
    paths = [path for path, _ in provider._client.calls]
    assert ("/api/v1/compact" in paths) is capture
    assert ("/api/v1/dispose" in paths) is capture


def test_disabled_capture_preserves_recovery_and_explicit_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", "false")
    monkeypatch.setenv("OGMEM_AUTO_RECALL", "false")
    client = FakeClient({"/api/v1/compose": {"identityContext": "memory"}})
    monkeypatch.setattr(plugin, "_OgMemoryClient", lambda *args, **kwargs: client)
    recovery = tmp_path / plugin._PENDING_WRITES_FILE
    contents = json.dumps({"dirty_sessions": ["old"], "writes": [{
        "path": "/api/v1/after_turn", "payload": {"sessionId": "old"},
    }]})
    recovery.write_text(contents)
    provider = plugin.OgMemoryProvider()
    provider.initialize("s", hermes_home=str(tmp_path))
    assert provider._worker is None
    assert not provider._dirty_sessions
    assert client.calls == []
    assert json.loads(provider.handle_tool_call("ogmemory_remember", {"content": "fact"}))["status"] == "stored"
    assert json.loads(provider.handle_tool_call("ogmemory_search", {"query": "fact"}))["context"] == ["memory"]
    provider.sync_turn("ignored", "ignored")
    provider.shutdown()
    assert recovery.read_text() == contents


@pytest.mark.parametrize("value, expected", [(" TRUE ", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("", False)])
def test_auto_control_values(monkeypatch, value, expected):
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", value)
    assert plugin.OgMemoryProvider()._auto_capture is expected


def test_auto_control_config_env_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OGMEM_AUTO_CAPTURE", raising=False)
    (tmp_path / "ogmemory.json").write_text(json.dumps({"auto_capture": False}))
    assert plugin.OgMemoryProvider()._auto_capture is False
    monkeypatch.setenv("OGMEM_AUTO_CAPTURE", "true")
    assert plugin.OgMemoryProvider()._auto_capture is True


def test_prefetch_maps_native_compose_slots_without_duplication():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient(
        {
            "/api/v1/compose": {
                "identityContext": "Profile facts",
                "episodicContext": "Earlier session",
                "sessionContext": "Current session",
                "retrievedEvidence": "Relevant preference",
            }
        }
    )

    result = provider.prefetch("What does the user prefer?")

    assert result == (
        "## oGMemory Context\n\nProfile facts\n\nEarlier session\n\n"
        "Current session\n\nRelevant preference"
    )
    path, payload = provider._client.calls[0]
    assert path == "/api/v1/compose"
    assert payload["sessionId"] == "session-1"
    assert payload["prompt"] == "What does the user prefer?"


def test_sync_turn_queues_only_completed_turn():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient()
    queued = []
    provider._submit = lambda path, payload: queued.append((path, payload))

    provider.sync_turn(
        "Remember that I use uv.",
        "Understood.",
        messages=[
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
        ],
    )

    assert queued[0][0] == "/api/v1/after_turn"
    assert queued[0][1]["messages"] == [
        {"role": "user", "content": "Remember that I use uv."},
        {"role": "assistant", "content": "Understood."},
    ]


def test_sync_turn_passes_the_native_tool_trace():
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient()
    queued = []
    provider._submit = lambda path, payload: queued.append(payload)
    messages = [{"role": "user", "content": "Read the contract."},
                {"role": "assistant", "content": "Reading.", "reasoning_content": "Check the limit."},
                {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "Limit: 100."}]
    provider.sync_turn("Read the contract.", "Use batches of 100.", messages=messages)
    assert queued[0]["traceMessages"] == messages + [{"role": "assistant", "content": "Use batches of 100."}]
    assert len(queued[0]["clientRequestId"]) == 32


@pytest.mark.parametrize("content", [
    "Read api.md.\nVerify the limit.",
    [{"type": "text", "text": "Read api.md.\nVerify the limit."}],
    [{"type": "text", "text": "Read api.md."}, {"type": "text", "text": "Verify the limit."}],
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
     {"type": "text", "text": "Read api.md.\nVerify the limit."}],
])
def test_sync_turn_selects_current_native_turn_by_text_content(content):
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient()
    provider._submit = provider._client.post
    prompt = "Read api.md.\nVerify the limit."
    current = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Reading.", "reasoning_content": "Check the contract.",
         "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"api.md"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Limit: 100."},
    ]
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": "Previous answer."}] + current
    original = json.dumps(messages)
    final = {"role": "assistant", "content": "Use batches of 100."}

    provider.sync_turn(prompt, final["content"], messages=messages)

    _, payload = provider._client.calls[0]
    assert payload["traceMessages"] == current + [final]
    assert payload["messages"] == [{"role": "user", "content": prompt}, final]
    assert json.dumps(messages) == original


def test_remember_uses_isolated_session_and_waiting_extraction():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "live-session"
    provider._client = FakeClient()

    result = json.loads(
        provider.handle_tool_call("ogmemory_remember", {"content": "Use uv for Python."})
    )

    assert result["status"] == "stored"
    after_path, after_payload = provider._client.calls[0]
    idle_path, idle_payload = provider._client.calls[1]
    assert after_path == "/api/v1/after_turn"
    assert after_payload["forceExtract"] is True
    assert after_payload["wait"] is True
    assert after_payload["sessionId"].startswith("hermes-remember-")
    assert idle_path == "/api/v1/call/wait_until_idle"
    assert idle_payload["drainOutbox"] is True
    assert provider._client.calls[2][0] == "/api/v1/dispose"


def test_remember_reports_after_turn_business_failure_and_disposes():
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient(
        {"/api/v1/after_turn": {"ok": False, "error": "extraction failed"}}
    )

    result = json.loads(provider.handle_tool_call("ogmemory_remember", {"content": "Use uv."}))

    assert result["error"] == "oGMemory remember failed"
    assert result["details"]["error"] == "extraction failed"
    assert [path for path, _payload in provider._client.calls] == [
        "/api/v1/after_turn",
        "/api/v1/dispose",
    ]


def test_remember_reports_idle_timeout_and_disposes():
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient(
        {"/api/v1/call/wait_until_idle": {"idle": False, "reason": "timeout"}}
    )

    result = json.loads(provider.handle_tool_call("ogmemory_remember", {"content": "Use uv."}))

    assert result["error"] == "oGMemory remember did not become idle"
    assert result["details"]["reason"] == "timeout"
    assert [path for path, _payload in provider._client.calls] == [
        "/api/v1/after_turn",
        "/api/v1/call/wait_until_idle",
        "/api/v1/dispose",
    ]


def test_remember_disposes_when_write_raises():
    provider = plugin.OgMemoryProvider()
    provider._client = FailingClient(failures=1)

    with pytest.raises(RuntimeError, match="temporary failure"):
        provider.handle_tool_call("ogmemory_remember", {"content": "Use uv."})

    assert [path for path, _payload in provider._client.calls] == [
        "/api/v1/after_turn",
        "/api/v1/dispose",
    ]


def test_save_config_restricts_new_and_existing_file_permissions(tmp_path):
    provider = plugin.OgMemoryProvider()
    config_path = tmp_path / "ogmemory.json"

    provider.save_config({"api_key": "secret"}, str(tmp_path))
    assert config_path.stat().st_mode & 0o777 == 0o600

    config_path.chmod(0o644)
    provider.save_config({"user_id": "user"}, str(tmp_path))
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_register_exposes_provider():
    registered = []
    ctx = types.SimpleNamespace(register_memory_provider=registered.append)

    plugin.register(ctx)

    assert len(registered) == 1
    assert registered[0].name == "ogmemory"


def test_commit_is_not_repeated_after_session_end():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient()
    provider._dirty_sessions.add("session-1")

    provider.on_session_end([])
    provider.shutdown()

    paths = [path for path, _payload in provider._client.calls]
    assert paths == [
        "/api/v1/call/wait_until_idle",
        "/api/v1/compact",
        "/api/v1/dispose",
    ]


def test_commit_stops_when_extraction_is_not_idle():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient(
        {"/api/v1/call/wait_until_idle": {"idle": False, "reason": "timeout"}}
    )
    provider._dirty_sessions.add("session-1")

    provider.on_session_end([])

    paths = [path for path, _payload in provider._client.calls]
    assert paths == ["/api/v1/call/wait_until_idle"]
    assert "session-1" in provider._dirty_sessions


def test_commit_stops_when_compaction_fails():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient({"/api/v1/compact": {"ok": False, "status": "failed"}})
    provider._dirty_sessions.add("session-1")

    provider.on_session_end([])

    paths = [path for path, _payload in provider._client.calls]
    assert paths == ["/api/v1/call/wait_until_idle", "/api/v1/compact"]
    assert "session-1" in provider._dirty_sessions


def test_background_write_retries_before_commit(monkeypatch):
    monkeypatch.setattr(plugin, "_WRITE_RETRY_DELAYS", (0, 0))
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FailingClient(failures=1)
    provider._worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker.start()

    provider.sync_turn("Remember this.", "OK")
    provider.on_session_end([])
    provider.shutdown()

    paths = [path for path, _payload in provider._client.calls]
    assert paths.count("/api/v1/after_turn") == 2
    assert "/api/v1/compact" in paths
    assert "session-1" not in provider._dirty_sessions


def test_failed_background_write_blocks_commit_until_recovered(monkeypatch):
    monkeypatch.setattr(plugin, "_WRITE_RETRY_DELAYS", (0, 0))
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FailingClient(failures=6)
    provider._worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker.start()

    provider.sync_turn("Remember this.", "OK")
    provider.on_session_end([])

    assert all(path != "/api/v1/compact" for path, _payload in provider._client.calls)
    assert "session-1" in provider._dirty_sessions
    assert provider._failed_writes["session-1"]

    provider._client.failures = 0
    provider.shutdown()

    paths = [path for path, _payload in provider._client.calls]
    assert "/api/v1/compact" in paths
    assert "session-1" not in provider._dirty_sessions


def test_background_business_failure_blocks_commit(monkeypatch):
    monkeypatch.setattr(plugin, "_WRITE_RETRY_DELAYS", (0, 0))
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient(
        {"/api/v1/after_turn": {"ok": False, "error": "extraction failed"}}
    )
    provider._worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker.start()

    provider.sync_turn("Remember this.", "OK")
    provider.on_session_end([])

    assert all(path != "/api/v1/compact" for path, _payload in provider._client.calls)
    assert "session-1" in provider._dirty_sessions
    assert provider._failed_writes["session-1"]
    provider.shutdown()


def test_lifecycle_commit_has_bounded_write_drain(monkeypatch):
    monkeypatch.setattr(plugin, "_WRITE_DRAIN_TIMEOUT", 0.05, raising=False)
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = BlockingClient()
    provider._worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker.start()
    provider.sync_turn("Remember this.", "OK")
    assert provider._client.started.wait(timeout=1)

    lifecycle = threading.Thread(target=provider.on_session_end, args=([],))
    lifecycle.start()
    try:
        lifecycle.join(timeout=0.2)
        assert not lifecycle.is_alive()
        assert "session-1" in provider._dirty_sessions
    finally:
        provider._client.release.set()
        lifecycle.join(timeout=1)
        provider.shutdown()


def test_lifecycle_commit_bounds_server_calls(monkeypatch):
    monkeypatch.setattr(plugin, "_WRITE_DRAIN_TIMEOUT", 0.05)
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = BlockingCommitClient()
    provider._dirty_sessions.add("session-1")

    lifecycle = threading.Thread(target=provider.on_session_end, args=([],))
    lifecycle.start()
    assert provider._client.started.wait(timeout=1)
    try:
        lifecycle.join(timeout=0.2)
        assert not lifecycle.is_alive()
        assert "session-1" in provider._dirty_sessions
    finally:
        provider._client.release.set()
        lifecycle.join(timeout=1)


def test_shutdown_timeout_persists_writes_for_next_start(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "_WRITE_DRAIN_TIMEOUT", 0.05)
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = BlockingClient()
    provider._recovery_path = tmp_path / "pending-writes.json"
    worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker = worker
    worker.start()
    provider.sync_turn("Remember this.", "OK")
    assert provider._client.started.wait(timeout=1)

    try:
        provider.shutdown()
        assert provider._worker is worker
        assert worker.is_alive()
        assert "session-1" in provider._dirty_sessions
        recovery = json.loads(provider._recovery_path.read_text(encoding="utf-8"))
        assert recovery["dirty_sessions"] == ["session-1"]
        assert recovery["writes"][0]["path"] == "/api/v1/after_turn"
    finally:
        provider._client.release.set()
        provider.shutdown()

    assert provider._worker is None
    assert "session-1" not in provider._dirty_sessions


def test_pending_writes_are_restored_and_cleared(tmp_path):
    recovery_path = tmp_path / "pending-writes.json"
    recovery_path.write_text(
        json.dumps(
            {
                "dirty_sessions": ["session-1"],
                "writes": [
                    {
                        "path": "/api/v1/after_turn",
                        "payload": FakeClient().identity("session-1"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provider = plugin.OgMemoryProvider()
    provider._client = FakeClient()
    provider._recovery_path = recovery_path
    provider._worker = threading.Thread(target=provider._run_writer, daemon=True)
    provider._worker.start()

    provider._restore_pending_writes()
    provider.shutdown()

    assert any(path == "/api/v1/after_turn" for path, _ in provider._client.calls)
    assert "session-1" not in provider._dirty_sessions
    assert not recovery_path.exists()


def test_session_end_retries_dirty_session_from_failed_switch():
    provider = plugin.OgMemoryProvider()
    provider._session_id = "session-1"
    provider._client = FakeClient({"/api/v1/compact": {"ok": False}})
    provider._dirty_sessions.add("session-1")

    provider.on_session_switch("session-2")
    assert "session-1" in provider._dirty_sessions

    provider._client.responses["/api/v1/compact"] = {"ok": True}
    provider.on_session_end([])

    assert "session-1" not in provider._dirty_sessions
