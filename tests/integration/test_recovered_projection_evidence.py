"""Recovered writes expose the completion evidence retained by their checkpoint."""

import json
from dataclasses import replace

import pytest

from commit.outbox_scope import capture_outbox_events
from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("checkpoint_format", ["tracked", "legacy", "writing"])
def test_recovered_projection_reports_its_durable_evidence(native_agfs, monkeypatch, checkpoint_format):
    client, fs, ctx = _runtime(native_agfs[0], "evidence-" + checkpoint_format)
    try:
        store = OutboxStore(client, fs)
        service, writer, manager = real_archives(fs, ctx, outbox=store)
        service._get_outbox_store = lambda _fs: store
        service._cfg.after_turn_threshold = 100000
        http = http_client(monkeypatch, service, ctx)
        request = {"sessionId": ctx.session_id, "clientRequestId": "recovered-source", "wait": True,
                   "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
        buffered_request = {key: value for key, value in request.items() if key != "clientRequestId"}
        accepted = http.post("/api/v1/after_turn", json=buffered_request)
        record("POST", "/api/v1/after_turn", buffered_request, accepted)
        assert accepted.json["status"] == "accumulating"
        buffer = manager.get_or_create(ctx.session_id, ctx)

        @capture_outbox_events
        def checkpoint_source():
            journal = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
                archive_id="accepted-archive", messages=request["messages"],
                session_time=buffer.messages[0].created_at, session_summary="", tool_stats_text="",
                client_request_id="recovered-source", request_messages=request["messages"])
            journal.chunks(lambda: [request["messages"]])
            writer.write_raw_chunk(messages=request["messages"], ctx=ctx, chunk_index=0,
                session_id=ctx.session_id, session_time=buffer.messages[0].created_at,
                candidate_checkpoint=journal.checkpoint("raw:0"))
            checkpoint = journal.checkpoint("chunk:0")
            source = replace(candidate(), provenance_ids=["prov:1:archive:accepted-archive:msg-one"])
            entry = checkpoint.accept([source], 1, 0)[0]
            checkpoint.begin(0)
            plan = writer._writer.write_candidate(entry, ctx)
            if checkpoint_format != "writing":
                checkpoint.complete(0, plan)

        checkpoint_source()
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        state_node = fs.read_node(state_uri, ctx)
        saved = json.loads(state_node.content)
        if checkpoint_format == "legacy":
            saved["runtime_extensions"][EXTENSION_KEY]["batch"].pop("outbox_tracking_version")
            state_node.content = json.dumps(saved)
            fs.write_node(state_node, ctx)
        print("RESTORED_CHECKPOINT " + json.dumps(saved["runtime_extensions"][EXTENSION_KEY]))

        restarted, next_writer, next_manager = real_archives(fs, ctx, outbox=store)
        restarted._get_outbox_store = lambda _fs: store
        http = http_client(monkeypatch, restarted, ctx)
        replayed = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, replayed)
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/recovered-source"
        if checkpoint_format == "tracked":
            projecting = http.get(path)
            record("GET", path, None, projecting)
            assert projecting.json["request_status"] == "PROJECTING"
            assert projecting.json["retry_action"] == "poll"
        pending = store.list_pending(ctx.account_id)
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
            directory_summary_enabled=False).run_once(store, [ctx.account_id], pending_events=pending)
        statuses = [store.get_event_status(uri, event.event_id) for uri, event in pending]
        print("PROJECTION " + json.dumps({"stats": stats, "statuses": statuses}))
        assert statuses == ["DONE", "DONE"]
        observed = http.get(path)
        record("GET", path, None, observed)
        state = json.loads(fs.read_node(state_uri, ctx).content)
        receipt = state["runtime_extensions"][OUTCOMES_KEY]["entries"]["recovered-source"]
        print("DURABLE_RECEIPT " + json.dumps(receipt))
        assert receipt["source_status"] == "ARCHIVED"
        assert receipt["semantic_status"] == "COMPLETE"
        assert receipt["outbox_tracking_complete"] is (checkpoint_format == "tracked")
        assert next_manager.get_or_create(ctx.session_id, ctx).messages == []
        assert next_writer._pipeline.extract.call_count == 0
        if checkpoint_format == "tracked":
            assert observed.json["request_status"] == "SUCCEEDED"
            assert observed.json["projection_status"] == "DONE"
            assert observed.json["retry_action"] == "none"
        else:
            assert observed.json["request_status"] == "NEEDS_CONFIRMATION"
            assert observed.json["projection_status"] == "UNKNOWN"
            assert observed.json["retry_action"] == "manual_review"
            assert receipt["request_status"] == "NEEDS_CONFIRMATION"
    finally:
        client.session.close()
