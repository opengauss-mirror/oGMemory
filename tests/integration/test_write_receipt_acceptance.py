"""Write acknowledgements retain acceptance and current completion policy."""

import json
from unittest.mock import Mock

import pytest

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from server.mcp_tools import write_memory
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("role,status,extractions", [
    ("user", "completed", 1), ("assistant", "no_extractable_messages", 0),
])
def test_synchronous_mcp_write_records_acceptance_and_trace(native_agfs, role, status, extractions):
    client, fs, ctx = _runtime(native_agfs[0], "mcp-acceptance")
    try:
        service, writer, manager = real_archives(fs, ctx)
        service._trace_event_store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        audit = Mock()
        service._audit = audit
        request = {"session_id": ctx.session_id, "wait": True,
                   "messages": [{"role": role, "content": "I repaired the receiver."}]}
        result = write_memory(service, request_context=ctx, **request)
        turns = service._trace_event_store._read_state(
            ctx.account_id, collections=("turns",), session_id=ctx.session_id)["turns"]
        recorded = audit.record.call_args.kwargs
        print("MCP_EXCHANGE " + json.dumps({"tool": "ogm_write_memory", "arguments": request,
              "response": result, "audit": recorded, "turns": list(turns.values())}, default=str))
        assert result["status"] == status
        assert result["accepted_messages"] == 1
        assert recorded["result"] == "success"
        assert len(turns) == 1
        assert next(iter(turns.values()))["status"] == "finalized"
        assert writer._pipeline.extract.call_count == extractions
        assert manager.get_or_create(ctx.session_id, ctx).messages == []
    finally:
        client.session.close()


def test_recovered_projection_receipt_uses_current_terminal_action(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "receipt-terminal-action")
    try:
        store = OutboxStore(client, fs)
        service, _, manager = real_archives(fs, ctx, outbox=store)
        service._get_outbox_store = lambda _fs: store
        http = http_client(monkeypatch, service, ctx)
        request = {"sessionId": ctx.session_id, "clientRequestId": "turn-one", "wait": True,
                   "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
        pending = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, pending)
        assert pending.json["request_status"] == "PROJECTING"
        buffer = manager.get_or_create(ctx.session_id, ctx)
        receipt = buffer.runtime_extensions[OUTCOMES_KEY]["entries"]["turn-one"]
        receipt["error_detail"] = {"code": "AGFS_DEPENDENCY_ERROR", "component": "agfs",
                                   "request_id": "turn-one", "retry_action": "poll",
                                   "upstream_http_status": 503}
        assert manager.save_session_state(ctx.session_id, ctx)
        print("RECOVERY_CHECKPOINT " + json.dumps(receipt))
        for uri, event in store.list_pending_events([tuple(ref) for ref in receipt["outbox_events"]]):
            with store.claim_processing_event(event, uri, "receipt-maintenance") as claimed:
                assert claimed
                store.move_to_dlq(event, uri)
            print("OUTBOX_TRANSITION " + json.dumps({"uri": uri, "event_id": event.event_id,
                  "status": store.get_event_status(uri, event.event_id)}))
        restarted, _, _ = real_archives(fs, ctx, outbox=store)
        restarted._get_outbox_store = lambda _fs: store
        http = http_client(monkeypatch, restarted, ctx)
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/turn-one"
        observed = http.get(path)
        replayed = http.post("/api/v1/after_turn", json=request)
        record("GET", path, None, observed)
        record("POST", "/api/v1/after_turn", request, replayed)
        for response in (observed, replayed):
            assert response.json["request_status"] == "NEEDS_CONFIRMATION"
            assert response.json["terminal"] is True
            assert response.json["retry_action"] == "manual_review"
            assert response.json["error_detail"]["retry_action"] == "manual_review"
    finally:
        client.session.close()
