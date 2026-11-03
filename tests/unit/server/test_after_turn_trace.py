import pytest

from commit.outbox_store import OutboxStore
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from tests.unit.server.test_agent_memory import _MemoryAGFS
from tests.unit.server.test_http_auth_v1 import _make_service, app_module


@pytest.fixture
def trace_client(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    service._cfg.index_workers = 0
    agfs = _MemoryAGFS()
    store = AGFSTraceEventStore(client=agfs, mount_prefix="/local")
    outbox = OutboxStore(agfs, AGFSContextFS(agfs))
    service._get_outbox_store = lambda fs: outbox
    service._trace_event_store = store
    service._get_context_fs = lambda: None
    service._get_archive_store = lambda: None
    manager = service.get_session_manager()
    manager.load_session_state = lambda *args, **kwargs: True
    manager.save_session_state = lambda *args, **kwargs: True
    monkeypatch.setattr(app_module, "_service", service)
    try:
        yield app_module.app.test_client(), store
    finally:
        service.shutdown()


@pytest.mark.parametrize("request_key", ["clientRequestId", "client_request_id"])
def test_after_turn_retry_reuses_normalized_trace_identity(trace_client, request_key):
    client, store = trace_client
    messages = [{"role": "user", "content": "Remember the limit."},
                {"role": "assistant", "content": "Use 100."}]
    body = {"accountId": "acct-1", "userId": "alice", "agentId": "agent", "sessionId": "retry",
            "messages": messages, "traceMessages": messages, request_key: " request-1 "}
    responses = []
    for identity in [" request-1 ", "request-1"]:
        body[request_key] = identity
        response = client.post("/api/v1/after_turn", json=body, headers={
            "X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "alice",
        })
        assert response.status_code == 200
        responses.append(response.get_json())
    assert [item["status"] for item in responses] == ["accumulating", "duplicate"]
    turns = store._read_state("acct-1", collections=("turns",), session_id="retry")["turns"]
    assert len(turns) == 1
    assert next(iter(turns.values()))["content"][-1]["assistant_message"] == "Use 100."


@pytest.mark.parametrize("trace_messages", [
    [
        {"role": "user", "content": "Read the contract."},
        {"role": "assistant", "content": None, "reasoning_content": "Check the limit.",
         "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": {"limit": 100}},
    ],
    [
        {"role": "user", "content": [{"type": "text", "text": "Read the contract."}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Check the limit."},
            {"type": "tool_use", "id": "c1", "name": "read_file", "input": {}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1",
                                       "content": [{"type": "text", "text": "Limit: 100."}]}]},
    ],
])
def test_after_turn_accepts_native_tool_and_reasoning_structures(trace_client, trace_messages):
    from skills.sanitizer import sanitize_trace

    client, store = trace_client
    final = {"role": "assistant", "content": "Use batches of 100."}
    body = {"accountId": "acct-1", "userId": "alice", "agentId": "agent", "sessionId": "native",
            "messages": [{"role": "user", "content": "Read the contract."}, final],
            "traceMessages": trace_messages + [final]}
    response = client.post("/api/v1/after_turn", json=body, headers={
        "X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "alice",
    })

    assert response.status_code == 200
    assert response.get_json()["trace_captured"] is True
    turns = store._read_state("acct-1", collections=("turns",), session_id="native")["turns"]
    events = next(iter(turns.values()))["content"]
    assert sanitize_trace(events).tools == {"read_file"}
    assert events[1]["reasoning"] == "Check the limit."
    assert events[-1]["assistant_message"] == final["content"]
