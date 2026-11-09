"""Asynchronous writers observe the durable identity returned by the service."""

import json
import threading
import time
from unittest.mock import Mock

import pytest

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.mcp_tools import write_memory
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("transport", ["http", "mcp"])
@pytest.mark.parametrize("mode", ["immediate", "buffered"])
def test_async_write_returns_its_durable_identity(native_agfs, monkeypatch, transport, mode):
    client, fs, ctx = _runtime(native_agfs[0], "async-identity")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._audit = Mock()
    service._trace_event_store = AGFSTraceEventStore(client=client, mount_prefix="/local")
    http = http_client(monkeypatch, service, ctx)
    started, finish = threading.Event(), threading.Event()
    extract = writer._pipeline.extract.side_effect

    def extract_turn(*args, **kwargs):
        started.set()
        assert finish.wait(20)
        return extract(*args, **kwargs)

    writer._pipeline.extract.side_effect = extract_turn
    source = [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]

    def submit(messages):
        if transport == "http":
            request = {"sessionId": ctx.session_id, "wait": False, "messages": messages}
            response = http.post("/api/v1/after_turn", json=request)
            record("POST", "/api/v1/after_turn", request, response)
            return response.json
        request = {"session_id": ctx.session_id, "wait": False, "messages": messages}
        result = write_memory(service, request_context=ctx, **request)
        print("MCP_EXCHANGE " + json.dumps({"tool": "ogm_write_memory", "arguments": request,
                                             "response": result}))
        return result

    try:
        buffered = unchanged = None
        if mode == "buffered":
            service._cfg.after_turn_threshold = 10
            buffered = submit(source)
            assert writer._pipeline.extract.call_count == 0
            if transport == "http":
                request = {"sessionId": ctx.session_id, "messages": source, "prePromptMessageCount": 1}
                response = http.post("/api/v1/after_turn", json=request)
                record("POST", "/api/v1/after_turn", request, response)
                unchanged = response.json
            source = [{"id": "msg-two", "role": "user", "content": "I also replaced the antenna."}]
        result = submit(source)
        assert started.wait(10)
        buffer = manager.get_or_create(ctx.session_id, ctx)
        batch = buffer.runtime_extensions[EXTENSION_KEY]["batch"]
        identity = batch["client_request_id"]
        print("DURABLE_IDENTITY " + json.dumps({"client_request_id": identity,
                                                "archive_id": batch["archive_id"]}))
        public_id = result.get("request_id", "")
        query = f"/api/v1/sessions/{ctx.session_id}/write-requests/{public_id}"
        pending = http.get(query)
        record("GET", query, None, pending)
        finish.set()
        deadline = time.monotonic() + 15
        while buffer.extraction_in_progress and time.monotonic() < deadline:
            time.sleep(.02)
        assert buffer.extraction_in_progress is False
        saved = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
        print("DURABLE_COMPLETION " + json.dumps(saved["runtime_extensions"][OUTCOMES_KEY]["entries"][identity]))
        events = store.list_pending(ctx.account_id)
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                             directory_summary_enabled=False).run_once(store, [ctx.account_id], pending_events=events)
        print("PROJECTION " + json.dumps(stats))
        completed = http.get(query)
        record("GET", query, None, completed)
        if buffered is not None:
            assert buffered["request_status"] == "ACCEPTED"
            assert buffered["source_status"] == "BUFFERED"
            assert buffered["semantic_status"] == "PENDING"
            assert buffered["retry_action"] == "none"
        if unchanged is not None:
            assert unchanged["status"] == "no_new_messages"
            assert unchanged["retry_action"] == "none"
        assert public_id == identity
        assert result["retry_action"] == "poll"
        assert pending.json["request_id"] == identity
        assert pending.json["request_status"] in {"RUNNING", "RECOVERING"}
        assert completed.json["request_status"] == "SUCCEEDED"
        assert completed.json["projection_status"] == "DONE"
        assert buffer.messages == []
        assert writer._pipeline.extract.call_count == 1
    finally:
        finish.set()
        service._extraction_retry.close(timeout=5)
        client.session.close()
