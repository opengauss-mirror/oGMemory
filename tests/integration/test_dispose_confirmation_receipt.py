"""Buffered requests expose the durable outcome of their disposal batch."""

import json
import threading
import time
from dataclasses import replace

from commit.outbox_scope import capture_outbox_events
from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


def test_dispose_confirmation_survives_restart_in_source_receipt(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "dispose-confirmation")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    http = http_client(monkeypatch, service, ctx)
    restarted = None
    release = threading.Event()
    request = {"sessionId": ctx.session_id, "clientRequestId": "dispose-source", "wait": True,
               "messages": [{"id": "msg-one", "role": "user",
                             "content": "I repaired the receiver. " * 50}]}
    try:
        response = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, response)
        assert response.json["source_status"] == "BUFFERED"
        buffer = manager.get_or_create(ctx.session_id, ctx)

        @capture_outbox_events
        def saved_execution_prefix():
            journal = service._dispose_journal(manager, buffer, ctx, list(buffer.messages))
            journal.chunks(lambda: [journal.batch["messages"]])
            checkpoint = journal.checkpoint("chunk:0")
            checkpoint.accept([replace(candidate(), provenance_ids=[
                f"prov:1:archive:{journal.batch['archive_id']}:msg-one"])], 1, 0)
            checkpoint.begin(0)
            return journal.batch["archive_id"]

        archive_id = saved_execution_prefix()
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        print("RESTORED_CHECKPOINT " + fs.read_node(state_uri, ctx).content)
        dispose_request = {"sessionId": ctx.session_id}
        response = http.post("/api/v1/dispose", json=dispose_request)
        record("POST", "/api/v1/dispose", dispose_request, response)
        deadline = time.monotonic() + 20
        while buffer.dispose_in_progress and time.monotonic() < deadline:
            time.sleep(.02)
        assert buffer.dispose_in_progress is False
        saved = json.loads(fs.read_node(state_uri, ctx).content)
        batch = saved["runtime_extensions"][EXTENSION_KEY]["batch"]
        print("DISPOSE_CHECKPOINT " + json.dumps(batch))
        assert batch["lifecycle"]["last_error"]["status"] == "needs_confirmation"
        assert batch["slots"]["raw:0"]["operations"]["0"]["status"] == "complete"
        assert [message["id"] for message in saved["buffer_messages"]] == ["msg-one"]
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/dispose-source"
        response = http.get(path)
        record("GET", path, None, response)
        restarted, restored_writer, restored_manager = real_archives(fs, ctx, outbox=store)
        restarted._get_outbox_store = lambda _fs: store
        restored_response = http_client(monkeypatch, restarted, ctx).get(path)
        record("GET_AFTER_RESTART", path, None, restored_response)
        for result in (response.json, restored_response.json):
            assert result["request_status"] == "NEEDS_CONFIRMATION"
            assert result["retry_action"] == "manual_review"
            assert result["terminal"] is True
            assert result["accepted"] is True
            assert result["archive_id"] == archive_id
            assert result["request_id"] == "dispose-source"
            assert result["source_message_ids"] == ["msg-one"]
            detail = result["error_detail"]
            assert detail["code"] == "WRITE_CONFIRMATION_REQUIRED"
            assert detail["request_id"] == "dispose-source"
            assert detail["cause"]["type"] == "RecoveryNeedsConfirmationError"
            assert detail["cause"]["message"] == batch["lifecycle"]["last_error"]["error"]

        @capture_outbox_events
        def complete_saved_write():
            journal = service._dispose_journal(manager, buffer, ctx, list(buffer.messages))
            accepted, _, _ = journal.checkpoint("chunk:0").load()
            return writer._writer.write_candidate(accepted[0], ctx)

        print("CONFIRMED_WRITE " + str(complete_saved_write()))
        entered = threading.Event()
        write_raw = restored_writer.write_raw_chunk

        def observed_raw_write(*args, **kwargs):
            entered.set()
            assert release.wait(20)
            return write_raw(*args, **kwargs)

        restored_writer.write_raw_chunk = observed_raw_write
        response = http.post("/api/v1/dispose", json=dispose_request)
        record("POST", "/api/v1/dispose", dispose_request, response)
        try:
            assert entered.wait(10)
            running = http.get(path)
            record("GET_RESUMED", path, None, running)
        finally:
            release.set()
        resumed_buffer = restored_manager.get_or_create(ctx.session_id, ctx)
        deadline = time.monotonic() + 20
        while resumed_buffer.dispose_in_progress and time.monotonic() < deadline:
            time.sleep(.02)
        assert resumed_buffer.dispose_in_progress is False
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                             directory_summary_enabled=False).run_once(store, [ctx.account_id])
        print("PROJECTION " + json.dumps(stats))
        completed = http.get(path)
        record("GET_COMPLETED", path, None, completed)
        assert running.json["request_status"] == "RUNNING"
        assert running.json["retry_action"] == "poll"
        assert completed.json["source_status"] == "RAW_ARCHIVED"
        assert completed.json["semantic_status"] == "COMPLETE"
        assert completed.json["projection_status"] == "UNKNOWN"
        assert completed.json["request_status"] == "NEEDS_CONFIRMATION"
        assert restored_writer._pipeline.extract.call_count == 0
    finally:
        release.set()
        if restarted is not None:
            restarted._extraction_retry.close(timeout=5)
        service._extraction_retry.close(timeout=5)
        client.session.close()
