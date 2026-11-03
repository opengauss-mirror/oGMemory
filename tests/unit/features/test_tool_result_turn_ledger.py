from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from features.tool_result_externalization.store_local import (
    LocalToolResultArtifactStore,
)
from features.tool_result_externalization.turn_ledger import ToolResultTurnLedger
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.session_manager import SessionManager


def _ctx(session_id: str = "session") -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id=session_id,
        trace_id="trace",
    )


def _result_turn(
    output: str,
    *,
    prompt: str = "inspect",
    call_id: str = "call_ledger_1",
    answer: str = "done",
) -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": call_id,
                "name": "read",
                "arguments": {"path": "/tmp/fixture"},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "read",
            "content": [{"type": "text", "text": output}],
        },
        {"role": "assistant", "content": answer, "stopReason": "stop"},
    ]


def _six_result_turn(size: int = 18_000) -> list[dict]:
    return _multi_result_turn([size] * 6)


def _multi_result_turn(sizes: list[int]) -> list[dict]:
    messages = [{"role": "user", "content": "inspect six"}]
    for index, size in enumerate(sizes):
        call_id = f"call_{index}"
        messages.extend([
            {
                "role": "assistant",
                "content": [{
                    "type": "toolCall",
                    "id": call_id,
                    "name": "read",
                    "arguments": {"path": f"/tmp/{index}"},
                }],
                "stopReason": "toolUse",
            },
            {
                "role": "toolResult",
                "toolCallId": call_id,
                "toolName": "read",
                "content": chr(ord("A") + index) * size,
            },
        ])
    messages.append({"role": "assistant", "content": "all done"})
    return messages


class _MemorySessionStateFS:
    """Minimal durable ContextFS shared by independent service instances."""

    def __init__(self):
        self._nodes: dict[str, ContextNode] = {}
        self._lock = threading.RLock()

    def exists(self, uri: str, _ctx: RequestContext) -> bool:
        with self._lock:
            return uri in self._nodes

    def write_node(self, node: ContextNode, _ctx: RequestContext) -> None:
        with self._lock:
            self._nodes[node.uri] = deepcopy(node)

    def read_node(self, uri: str, _ctx: RequestContext) -> ContextNode:
        with self._lock:
            if uri not in self._nodes:
                raise NodeNotFoundError(uri)
            return deepcopy(self._nodes[uri])

    def delete_node(self, uri: str, _ctx: RequestContext) -> None:
        with self._lock:
            if uri not in self._nodes:
                raise FileNotFoundError(uri)
            del self._nodes[uri]

    def payload(self, uri: str) -> dict:
        with self._lock:
            return json.loads(self._nodes[uri].content)


def _service(tmp_path, **overrides) -> MemoryService:
    values = dict(
        tool_result_externalization_enabled=True,
        tool_result_semantic_turn_mode="user_message",
        tool_result_single_chars=20_000,
        after_turn_threshold=999_999,
    )
    values.update(overrides)
    cfg = OgMemConfig(**values)
    service = MemoryService(
        cfg,
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_archive_store = lambda: None
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _persistent_service(tmp_path, fs: _MemorySessionStateFS, **overrides) -> MemoryService:
    values = dict(
        tool_result_externalization_enabled=True,
        tool_result_semantic_turn_mode="user_message",
        tool_result_single_chars=20_000,
        after_turn_threshold=999_999,
    )
    values.update(overrides)
    service = MemoryService(
        OgMemConfig(**values),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._session_mgr = SessionManager(get_context_fs=lambda: fs)
    return service


def _after(service: MemoryService, turn: list[dict], *, start: int = 0):
    return service.after_turn({
        "sessionId": "session",
        "messages": turn,
        "prePromptMessageCount": start,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })


def _tool_contents(messages: list[dict]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "toolResult"
    ]
def test_user_message_medium_result_switches_only_after_idle_ttl(tmp_path):
    service = _service(
        tmp_path,
        tool_result_cache_delay_enabled=True,
        tool_result_cache_delay_chars=2_000,
        tool_result_cache_delay_idle_ttl_seconds=30,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = [start]
    service._tool_result_governance._tool_result_cache_delay.clock = lambda: now[0]
    turn = _result_turn("M" * 3_000)
    _after(service, turn)

    now[0] = start + timedelta(seconds=1)
    hot = service.compose({
        "sessionId": "session", "prompt": "hot", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    assert all(
        "[ogmemory-tool-result v=2]" not in value
        for value in _tool_contents(hot["messages"])
    )

    now[0] = start + timedelta(seconds=20)
    still_hot = service.compose({
        "sessionId": "session", "prompt": "hot", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    assert all(
        "[ogmemory-tool-result v=2]" not in value
        for value in _tool_contents(still_hot["messages"])
    )

    now[0] = start + timedelta(seconds=51)
    cold = service.compose({
        "sessionId": "session", "prompt": "cold", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    assert any(
        "[ogmemory-tool-result v=2]" in value
        for value in _tool_contents(cold["messages"])
    )


def test_real_session_state_restart_preserves_medium_result_ttl_cutover(tmp_path):
    fs = _MemorySessionStateFS()
    first = _persistent_service(
        tmp_path,
        fs,
        tool_result_cache_delay_enabled=True,
        tool_result_cache_delay_chars=2_000,
        tool_result_cache_delay_idle_ttl_seconds=30,
    )
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    first._tool_result_governance._tool_result_cache_delay.clock = lambda: started_at
    turn = _result_turn("M" * 3_000)
    _after(first, turn)

    first._tool_result_governance._tool_result_cache_delay.clock = (
        lambda: started_at + timedelta(seconds=1)
    )
    hot = first.compose({
        "sessionId": "session",
        "prompt": "hot",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert all(
        "[ogmemory-tool-result v=2]" not in value
        for value in _tool_contents(hot["messages"])
    )

    state_uri = "ctx://acct/sessions/session/state.json"
    persisted_extensions = fs.payload(state_uri)["runtime_extensions"]
    ledger_entry = persisted_extensions[ToolResultTurnLedger.STATE_KEY][
        "completed_turns"
    ][0]
    assert ledger_entry["results"][0]["hot_seen_at"]
    assert persisted_extensions["tool_result_cache_delay"][
        "hot_seen_artifacts"
    ] == {}
    assert persisted_extensions["tool_result_cache_delay"][
        "last_model_request_at"
    ] == (started_at + timedelta(seconds=1)).isoformat()

    restarted = _persistent_service(
        tmp_path,
        fs,
        tool_result_cache_delay_enabled=True,
        tool_result_cache_delay_chars=2_000,
        tool_result_cache_delay_idle_ttl_seconds=30,
    )
    restarted._tool_result_governance._tool_result_cache_delay.clock = (
        lambda: started_at + timedelta(seconds=40)
    )
    cold = restarted.compose({
        "sessionId": "session",
        "prompt": "cold",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert any(
        "[ogmemory-tool-result v=2]" in value
        for value in _tool_contents(cold["messages"])
    )

    restarted._tool_result_governance._tool_result_cache_delay.clock = (
        lambda: started_at + timedelta(seconds=41)
    )
    still_cold = restarted.compose({
        "sessionId": "session",
        "prompt": "still cold",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert any(
        "[ogmemory-tool-result v=2]" in value
        for value in _tool_contents(still_cold["messages"])
    )


def test_user_message_25k_is_full_before_afterturn_and_stub_on_next_compose(tmp_path):
    service = _service(tmp_path)
    turn = _result_turn("X" * 25_000)

    assert _tool_contents(turn) == [str(turn[2]["content"])]
    result = _after(service, turn)
    assert result["ok"] is True

    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["identity_strength"] == "strong"

    replay = service.compose({
        "sessionId": "session",
        "prompt": "next",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert any("[ogmemory-tool-result v=2]" in value for value in _tool_contents(replay["messages"]))
    assert all("_ogmem_source" not in message for message in replay["messages"])
def test_identical_outputs_in_different_turns_and_same_turn_do_not_alias(tmp_path):
    service = _service(tmp_path, tool_result_single_chars=10)
    first = _result_turn("same-output" * 3, prompt="turn one", call_id="call_1")
    second = _result_turn("same-output" * 3, prompt="turn two", call_id="call_2")
    _after(service, first, start=0)
    _after(service, first + second, start=len(first))

    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(artifacts) == 2
    assert len({row["artifact_id"] for row in artifacts}) == 2

    same_turn = [
        {"role": "user", "content": "two identical results"},
        {"role": "assistant", "content": [
            {"type": "toolCall", "id": "call_a", "name": "read", "arguments": {}},
            {"type": "toolCall", "id": "call_b", "name": "read", "arguments": {}},
        ]},
        {"role": "toolResult", "toolCallId": "call_a", "content": "Z" * 30},
        {"role": "toolResult", "toolCallId": "call_b", "content": "Z" * 30},
        {"role": "assistant", "content": "done"},
    ]
    _after(
        service,
        first + second + same_turn,
        start=len(first) + len(second),
    )
    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(artifacts) == 4
    assert len({row["artifact_id"] for row in artifacts}) == 4


def test_retry_and_restart_reuse_ledger_without_duplicate_artifacts(tmp_path):
    service = _service(tmp_path)
    turn = _result_turn("R" * 25_000)
    _after(service, turn)
    normalized_retry = deepcopy(turn)
    normalized_retry[1]["content"][0]["id"] = "callledger1"
    normalized_retry[2]["toolCallId"] = "callledger1"
    _after(service, normalized_retry)

    manager = service.get_session_manager()
    state = deepcopy(manager.get_or_create("session", ctx=_ctx()).runtime_extensions)
    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(state[ToolResultTurnLedger.STATE_KEY]["completed_turns"]) == 1
    assert len(artifacts) == 1

    restarted = _service(tmp_path)
    restarted.get_session_manager().get_or_create(
        "session", ctx=_ctx(),
    ).runtime_extensions = state
    replay = restarted.compose({
        "sessionId": "session", "prompt": "next", "messages": normalized_retry,
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    assert any("[ogmemory-tool-result v=2]" in value for value in _tool_contents(replay["messages"]))
def test_concurrent_afterturn_retries_create_one_artifact_and_one_turn(tmp_path):
    fs = _MemorySessionStateFS()
    service = _persistent_service(tmp_path, fs)
    turn = _result_turn("C" * 25_000)
    barrier = threading.Barrier(6)
    results = []
    errors = []
    result_lock = threading.Lock()

    def submit() -> None:
        try:
            barrier.wait(timeout=5)
            result = _after(service, deepcopy(turn))
            with result_lock:
                results.append(result)
        except Exception as exc:  # pragma: no cover - asserted below
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 6
    assert all(result["ok"] is True for result in results)
    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(artifacts) == 1
    persisted = fs.payload("ctx://acct/sessions/session/state.json")[
        "runtime_extensions"
    ][ToolResultTurnLedger.STATE_KEY]
    assert persisted["next_turn_sequence"] == 2
    assert len(persisted["completed_turns"]) == 1
def test_externalization_disabled_creates_no_turn_ledger(tmp_path):
    service = MemoryService(OgMemConfig(
        tool_result_externalization_enabled=False,
        tool_result_semantic_turn_mode="user_message",
        after_turn_threshold=999_999,
    ))
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    turn = _result_turn("D" * 25_000)
    service.after_turn({
        "sessionId": "session", "messages": turn,
        "prePromptMessageCount": 0, "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    buf = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert ToolResultTurnLedger.STATE_KEY not in buf.runtime_extensions
    assert [message.content for message in buf.messages if message.role == "toolResult"]
