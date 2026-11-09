"""Buffered source owners observe the current durable recovery batch."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from commit.outbox_scope import capture_outbox_events
from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


def test_buffered_sources_follow_their_recovery_batch(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "buffered-batch")
    store = OutboxStore(client, fs)
    service, _, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    http = http_client(monkeypatch, service, ctx)
    restarted = None
    requests = [
        {"sessionId": ctx.session_id, "clientRequestId": request_id, "wait": True,
         "messages": [{"id": message_id, "role": "user", "content": content}]}
        for request_id, message_id, content in [
            ("source-a", "msg-a", "I repaired the receiver."),
            ("source-b", "msg-b", "I replaced its antenna."),
        ]
    ]
    try:
        response = http.post("/api/v1/after_turn", json=requests[0])
        record("POST", "/api/v1/after_turn", requests[0], response)
        assert response.json["source_status"] == "BUFFERED"
        ingest = {"sessionId": ctx.session_id, "messages": requests[1]["messages"]}
        response = http.post("/api/v1/ingest", json=ingest)
        record("POST", "/api/v1/ingest", ingest, response)
        assert response.json["ingested"] is True
        buffer = manager.get_or_create(ctx.session_id, ctx)
        source = [message for request in requests for message in request["messages"]]

        @capture_outbox_events
        def saved_execution_prefix():
            journal = ExtractionBatchJournal(
                manager, buffer, ctx, list(buffer.messages), archive_id="restored-source-b",
                messages=source, session_time=None, session_summary="", tool_stats_text="",
                client_request_id="source-b", request_messages=requests[1]["messages"],
                request_source_ids=["msg-b"],
            )
            journal.chunks(lambda: [source])
            checkpoint = journal.checkpoint("chunk:0")
            checkpoint.accept([replace(candidate(), provenance_ids=["prov:1:archive:restored-source-b:msg-a"])], 1, 0)
            checkpoint.begin(0)
            return journal

        journal = saved_execution_prefix()
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        print("RESTORED_CHECKPOINT " + fs.read_node(state_uri, ctx).content)
        restarted, _, restored_manager = real_archives(fs, ctx, outbox=store)
        restarted._get_outbox_store = lambda _fs: store
        restarted._load_extraction_recovery_before_accumulation(ctx.session_id, ctx)
        restored = restored_manager.get_or_create(ctx.session_id, ctx)
        restored_journal = ExtractionBatchJournal.resume(restored_manager, restored, ctx)
        restarted._extraction_retry.register_pending_source(ctx, restored_journal, wake=False)
        restored_journal.mutate(lambda batch: batch["retry_task"].update(next_attempt_at=0))
        http = http_client(monkeypatch, restarted, ctx)
        observations = []
        for request in requests:
            path = f"/api/v1/sessions/{ctx.session_id}/write-requests/{request['clientRequestId']}"
            response = http.get(path)
            record("GET", path, None, response)
            observations.append(response.json)
        restarted._extraction_retry.run_once(ctx, journal.batch["archive_id"])
        task = restored.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        print("RECOVERY_TASK " + json.dumps(task))
        final = []
        for request in requests:
            path = f"/api/v1/sessions/{ctx.session_id}/write-requests/{request['clientRequestId']}"
            response = http.get(path)
            record("GET", path, None, response)
            final.append(response.json)
        assert task["status"] == "needs_confirmation"
        assert [response["request_status"] for response in observations] == ["RECOVERING"] * 2
        for response in final:
            assert response["request_status"] == "NEEDS_CONFIRMATION"
            assert response["retry_action"] == "manual_review"
            assert response["archive_id"] == "restored-source-b"
            assert response["accepted"] is True
    finally:
        if restarted is not None:
            restarted._extraction_retry.close(timeout=5)
        service._extraction_retry.close(timeout=5)
        client.session.close()


def test_buffered_source_observes_live_extraction_and_completion(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "buffered-live")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    http = http_client(monkeypatch, service, ctx)
    entered, release = threading.Event(), threading.Event()
    original = writer._pipeline.extract.side_effect

    def extract(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)

    writer._pipeline.extract.side_effect = extract
    first = {"sessionId": ctx.session_id, "clientRequestId": "source-a", "wait": True,
             "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    second = {"sessionId": ctx.session_id, "clientRequestId": "source-b", "forceExtract": True,
              "messages": [{"id": "msg-two", "role": "user", "content": "I replaced its antenna."}]}
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/source-a"
    try:
        accepted = http.post("/api/v1/after_turn", json=first)
        record("POST", "/api/v1/after_turn", first, accepted)
        with ThreadPoolExecutor(max_workers=1) as pool:
            submitted = pool.submit(http.post, "/api/v1/after_turn", json=second)
            try:
                assert entered.wait(10)
                observed = http.get(path)
                record("GET", path, None, observed)
            finally:
                release.set()
            record("POST", "/api/v1/after_turn", second, submitted.result(timeout=20))
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                             directory_summary_enabled=False).run_once(store, [ctx.account_id])
        print("PROJECTION " + json.dumps(stats))
        completed = http.get(path)
        record("GET", path, None, completed)
        assert observed.json["request_status"] in {"RUNNING", "RECOVERING"}
        assert observed.json["retry_action"] == "poll"
        assert observed.json["source_message_ids"] == ["msg-one"]
        assert completed.json["request_status"] == "SUCCEEDED"
        assert completed.json["projection_status"] == "DONE"
        assert writer._pipeline.extract.call_count == 1
    finally:
        release.set()
        service._extraction_retry.close(timeout=5)
        client.session.close()
