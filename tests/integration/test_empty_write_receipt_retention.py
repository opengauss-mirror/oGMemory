"""Empty acknowledgements share the completed-request retention window."""

import json

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


def test_empty_receipts_follow_completed_window_and_preserve_pending_sources(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "empty-receipts")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    http = http_client(monkeypatch, service, ctx)
    first = {"id": "msg-one", "role": "user", "content": "I repaired the receiver."}
    second = {"id": "msg-two", "role": "user", "content": "I replaced its antenna."}
    third = {"id": "msg-three", "role": "user", "content": "The receiver now works."}

    def submit(request_id, message, **options):
        request = {"sessionId": ctx.session_id, "clientRequestId": request_id, "wait": True,
                   "messages": [message], **options}
        response = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, response)
        return response.json

    def observe(request_id):
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/{request_id}"
        response = http.get(path)
        record("GET", path, None, response)
        return response.json

    try:
        accepted = submit("pending-first", first)
        assert accepted["source_message_ids"] == ["msg-one"]
        for index in range(272):
            unchanged = submit(f"empty-{index:03d}", first, prePromptMessageCount=1)
            assert unchanged["status"] == "no_new_messages"
            assert unchanged["source_message_ids"] == []
            assert unchanged["retry_action"] == "none"
        accepted = submit("pending-second", second)
        assert accepted["source_message_ids"] == ["msg-two"]
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        saved = json.loads(fs.read_node(state_uri, ctx).content)
        entries = saved["runtime_extensions"][OUTCOMES_KEY]["entries"]
        retained = saved["completed_request_ids"]
        print("RECEIPT_WINDOW " + json.dumps({"entry_count": len(entries), "retained_count": len(retained),
            "entry_ids": list(entries), "retained_ids": retained,
            "buffer_ids": [message["id"] for message in saved["buffer_messages"]]}))
        first_receipt = observe("pending-first")
        second_receipt = observe("pending-second")
        assert first_receipt["source_message_ids"] == ["msg-one"]
        assert second_receipt["source_message_ids"] == ["msg-two"]
        assert first_receipt["request_status"] == second_receipt["request_status"] == "ACCEPTED"
        assert set(entries) == set(retained) | {"pending-first"}
        assert len(entries) == 257

        submit("complete-sources", third, forceExtract=True)
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
            directory_summary_enabled=False).run_once(store, [ctx.account_id])
        print("PROJECTION " + json.dumps(stats))
        completed = observe("pending-second")
        assert completed["request_status"] == "SUCCEEDED"
        assert completed["projection_status"] == "DONE"
        assert writer._pipeline.extract.call_count == 1
        assert manager.get_or_create(ctx.session_id, ctx).messages == []
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()
