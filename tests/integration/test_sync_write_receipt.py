"""Synchronous writes return the durable receipt for their accepted source."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.outbox_scope import capture_outbox_events
from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.mcp_tools import write_memory
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.extraction.test_node_processing import NodeLLM, operation
from tests.unit.server.test_extraction_retry_task import ready
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_memory_card_recovery import card, enable
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("mode", ["card_format", "node_processing"])
def test_sync_format_recovery_returns_its_durable_request_id(native_agfs, monkeypatch, mode):
    client, fs, ctx = _runtime(native_agfs[0], "sync-format-recovery")
    store = OutboxStore(client, fs)
    draft = replace(card(), overview="Nora is an engineer.") if mode == "card_format" else card()
    service, writer, manager = real_archives(fs, ctx, entries=[draft], outbox=store)
    if mode == "card_format":
        enable(service, writer)
    else:
        service._cfg.node_processing_enabled = writer._node_processing_enabled = True
        service._extraction_retry.wake = Mock()
        writer._llm = NodeLLM(lambda _: [operation([0], action="defer")])
    service._get_outbox_store = lambda _fs: store
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)
    service.drain_outbox_sync = lambda **kwargs: worker.run_once(store, [ctx.account_id])
    http = http_client(monkeypatch, service, ctx)
    request = {"sessionId": ctx.session_id, "wait": True, "forceExtract": True,
               "messages": [{"id": "msg-one", "role": "user", "content": "Nora is a radio engineer."}]}
    try:
        response = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, response)
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        state = json.loads(fs.read_node(state_uri, ctx).content)
        batch = state["runtime_extensions"][EXTENSION_KEY]["batch"]
        print("DURABLE_FORMAT_RECOVERY " + json.dumps({
            "mode": mode, "request_id": batch["client_request_id"], "archive_id": batch["archive_id"],
            "retry_task": batch["retry_task"],
        }))
        request_id = response.json["request_id"]
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests?clientRequestId={request_id}"
        observed = http.get(path)
        record("GET", path, None, observed)
        assert response.status_code == 202
        assert response.json["request_status"] == "RECOVERING"
        assert response.json["retry_action"] == "poll"
        assert request_id == batch["client_request_id"]
        assert request_id.startswith("card-")
        assert observed.status_code == 200
        assert observed.json["request_status"] == "RECOVERING"
        assert observed.json["archive_id"] == batch["archive_id"]

        if mode == "card_format":
            writer._pipeline.extract.side_effect = None
            writer._pipeline.extract.return_value = [card()]
        else:
            writer._llm = NodeLLM(lambda _: [operation([0])])
        with service._session_mutation_scope(ctx):
            ready(manager, ctx)
        service._extraction_retry.run_once(ctx, batch["retry_task"]["id"])
        completed = http.get(path)
        record("GET", path, None, completed)
        assert completed.status_code == 200
        assert completed.json["request_id"] == request_id
        assert completed.json["request_status"] == "SUCCEEDED"
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()


@pytest.mark.parametrize("transport", ["http", "mcp"])
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("mode", ["new", "restored", "legacy"])
def test_sync_write_returns_its_completed_receipt(native_agfs, monkeypatch, transport, role, mode):
    client, fs, ctx = _runtime(native_agfs[0], "sync-identity")
    store = OutboxStore(client, fs)
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)

    def runtime():
        service, writer, manager = real_archives(fs, ctx, outbox=store)
        service._get_outbox_store = lambda _fs: store
        service.drain_outbox_sync = lambda **kwargs: worker.run_once(store, [ctx.account_id])
        service._audit = Mock()
        service._trace_event_store = AGFSTraceEventStore(client=client, mount_prefix="/local")
        return service, writer, manager

    service, writer, manager = runtime()
    assert service._cfg.card_format_enabled is False
    assert service._cfg.node_processing_enabled is False
    http = http_client(monkeypatch, service, ctx)
    source = [{"id": "msg-one", "role": role, "content": "I repaired the receiver."}]
    try:
        if mode in {"restored", "legacy"}:
            request = {"sessionId": ctx.session_id, "messages": source}
            ingested = http.post("/api/v1/ingest", json=request)
            record("POST", "/api/v1/ingest", request, ingested)
            buffer = manager.get_or_create(ctx.session_id, ctx)

            @capture_outbox_events
            def checkpoint_source():
                return ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
                    archive_id="accepted-archive", messages=source if role == "user" else [],
                    session_time=buffer.messages[0].created_at, session_summary="", tool_stats_text="",
                    client_request_id="accepted-write" if mode == "restored" else "", request_messages=source)

            print("RESTORED_CHECKPOINT " + json.dumps(checkpoint_source().batch, default=str))
            service._extraction_retry.close(timeout=5)
            service, writer, manager = runtime()
            http = http_client(monkeypatch, service, ctx)
        if transport == "http":
            request = {"sessionId": ctx.session_id, "wait": True, "forceExtract": True, "messages": source}
            response = http.post("/api/v1/after_turn", json=request)
            record("POST", "/api/v1/after_turn", request, response)
            result = response.json
        else:
            request = {"session_id": ctx.session_id, "wait": True, "force_extract": True, "messages": source}
            result = write_memory(service, request_context=ctx, **request)
            print("MCP_EXCHANGE " + json.dumps({"tool": "ogm_write_memory", "arguments": request,
                                                 "response": result}))
        state = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
        receipts = state.get("runtime_extensions", {}).get(OUTCOMES_KEY, {}).get("entries", {})
        print("DURABLE_RECEIPTS " + json.dumps(receipts))
        identity = result["request_id"]
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/{identity}"
        observed = http.get(path)
        record("GET", path, None, observed)
        assert identity
        assert result["request_status"] == "SUCCEEDED"
        assert result["source_status"] == "ARCHIVED"
        assert result["semantic_status"] == "COMPLETE"
        assert result["projection_status"] == ("DONE" if role == "user" else "NOT_REQUIRED")
        assert result["retry_action"] == "none"
        assert observed.json["request_status"] == receipts[identity]["request_status"] == "SUCCEEDED"
        assert writer._pipeline.extract.call_count == (1 if role == "user" else 0)
        assert manager.get_or_create(ctx.session_id, ctx).messages == []
        if mode == "restored":
            assert identity == "accepted-write"
        elif mode == "legacy":
            assert identity == "extraction-retry-accepted-archive"
        if transport == "mcp":
            assert result["accepted_messages"] == 1
            assert service._audit.record.call_args.kwargs["result"] == "success"
            turns = service._trace_event_store._read_state(
                ctx.account_id, collections=("turns",), session_id=ctx.session_id)["turns"]
            assert len(turns) == 1
            assert next(iter(turns.values()))["status"] == "finalized"
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()


@pytest.mark.parametrize("source", ["completed", "buffered"])
@pytest.mark.parametrize("tail_id", ["msg-two", None])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_resumed_receipt_follows_the_submitted_source(native_agfs, monkeypatch, source, tail_id, role):
    client, fs, ctx = _runtime(native_agfs[0], "sync-source-owner")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)
    service.drain_outbox_sync = lambda **kwargs: worker.run_once(store, [ctx.account_id])
    http = http_client(monkeypatch, service, ctx)
    original = {"id": "msg-one", "role": role, "content": "I repaired the receiver."}
    tail = {"role": "user", "content": "I replaced its antenna."}
    if tail_id:
        tail["id"] = tail_id
    try:
        request = {"sessionId": ctx.session_id, "messages": [original]}
        record("POST", "/api/v1/ingest", request, http.post("/api/v1/ingest", json=request))
        buffer = manager.get_or_create(ctx.session_id, ctx)

        @capture_outbox_events
        def checkpoint_source():
            return ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
                archive_id="accepted-archive", messages=[original] if role == "user" else [],
                session_time=buffer.messages[0].created_at, session_summary="", tool_stats_text="",
                client_request_id="accepted-write", request_messages=[original])

        print("RESTORED_CHECKPOINT " + json.dumps(checkpoint_source().batch, default=str))
        if source == "completed":
            request = {"sessionId": ctx.session_id, "messages": [tail]}
            record("POST", "/api/v1/ingest", request, http.post("/api/v1/ingest", json=request))
        request = {"sessionId": ctx.session_id, "wait": True, "forceExtract": True,
                   "messages": [original if source == "completed" else tail]}
        response = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, response)
        saved = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
        print("BUFFER_AFTER_REQUEST " + json.dumps(saved["buffer_messages"]))
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/accepted-write"
        original_receipt = http.get(path)
        record("GET", path, None, original_receipt)
        assert [message["content"] for message in saved["buffer_messages"]] == [tail["content"]]
        assert original_receipt.json["request_status"] == "SUCCEEDED"
        assert original_receipt.json["archived_source_ids"] == ["msg-one"]
        assert writer._pipeline.extract.call_count == (1 if role == "user" else 0)
        if source == "completed":
            assert response.json["status"] == ("completed" if role == "user" else "no_extractable_messages")
            assert response.json["request_status"] == "SUCCEEDED"
            assert response.json["request_id"] == "accepted-write"
        else:
            assert response.json["status"] == "accumulating"
            assert response.json["request_status"] == "ACCEPTED"
            assert response.json["source_status"] == "BUFFERED"
            assert response.json["semantic_status"] == "PENDING"
            assert response.json["retry_action"] == "none"
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()
