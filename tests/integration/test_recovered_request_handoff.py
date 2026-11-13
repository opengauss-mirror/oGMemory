"""New request owners retain the archive evidence of their shared source."""

import json
from dataclasses import replace

import pytest

from commit.outbox_scope import capture_outbox_events
from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_checkpoint import ExtractionBatchJournal
from service.extraction_retry import source_only_recovery
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("sources", ["shared", "mixed", "independent"])
@pytest.mark.parametrize("previous_archive", [False, True])
def test_recovered_batch_settles_each_request_source(native_agfs, monkeypatch, sources, previous_archive):
    client, fs, ctx = _runtime(native_agfs[0], "recovery-handoff")
    store = OutboxStore(client, fs)
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)
    extractions = []

    def extract(messages, _ctx, **kwargs):
        extractions.append([message["id"] for message in messages])
        return [replace(candidate(body=message["content"]), routing_key="fact_" + message["id"],
                        title=message["content"], provenance_ids=[
                            f"prov:1:archive:{kwargs['archive_id']}:{message['id']}"])
                for message in messages]

    def runtime():
        service, writer, manager = real_archives(fs, ctx, outbox=store)
        service._get_outbox_store = lambda _fs: store
        service._cfg.after_turn_threshold = 100000
        service.drain_outbox_sync = lambda **kwargs: worker.run_once(store, [ctx.account_id])
        writer._pipeline.extract.side_effect = extract
        return service, writer, manager

    service, writer, manager = runtime()
    http = http_client(monkeypatch, service, ctx)
    older = {"id": "msg-zero", "role": "user", "content": "I painted the garage."}
    original = {"id": "msg-one", "role": "user", "content": "I repaired the receiver."}
    newer = {"id": "msg-two", "role": "user", "content": "I replaced its antenna."}

    def submit(request_id, messages, *, force=False, **options):
        request = {"sessionId": ctx.session_id, "wait": True,
                   "forceExtract": force, "messages": messages, **options}
        if request_id:
            request["clientRequestId"] = request_id
        response = http.post("/api/v1/after_turn", json=request)
        record("POST", "/api/v1/after_turn", request, response)
        assert response.json["ok"] is True
        return response.json

    @capture_outbox_events
    def checkpoint(request_id, messages, request_messages):
        buffer = manager.get_or_create(ctx.session_id, ctx)
        ids = {message["id"] for message in messages}
        journal = ExtractionBatchJournal(manager, buffer, ctx,
            [message for message in buffer.messages if message.id in ids],
            archive_id="archive-" + request_id, messages=messages,
            session_time=buffer.messages[0].created_at, session_summary="", tool_stats_text="",
            client_request_id=request_id, request_messages=request_messages,
            request_source_ids=[message["id"] for message in request_messages])
        print("RESTORED_CHECKPOINT " + json.dumps(journal.batch, default=str))

    def observe(request_id):
        path = f"/api/v1/sessions/{ctx.session_id}/write-requests/{request_id}"
        response = http.get(path)
        record("GET", path, None, response)
        return response.json

    try:
        if previous_archive:
            submit("request-r", [older])
        accepted_a = ([older] if previous_archive else []) + [original]
        submit("request-a", accepted_a)
        if previous_archive:
            checkpoint("request-r", [older], [older])
            submit(None, [older], force=True)
            prior_receipt = observe("request-r")
            if sources == "shared":
                for index in range(256):
                    submit(f"empty-{index:03d}", [original], prePromptMessageCount=1)
        checkpoint("request-a", [original], accepted_a)
        if previous_archive:
            service._extraction_retry.close(timeout=5)
            service, writer, manager = runtime()
            http = http_client(monkeypatch, service, ctx)
            print("RESTORED_SERVICE " + json.dumps(manager.get_or_create(ctx.session_id, ctx).runtime_extensions))
        submitted = ([original] if sources != "independent" else []) + ([newer] if sources != "shared" else [])
        first = submit("request-b", submitted, force=True)
        after_a = observe("request-a")
        observed = observe("request-b")
        replayed = submit("request-b", submitted, force=True)
        saved = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
        print("DURABLE_RECEIPTS_AFTER_HANDOFF " + json.dumps(saved["runtime_extensions"][OUTCOMES_KEY]))
        print("BUFFER_AFTER_HANDOFF " + json.dumps(saved["buffer_messages"]))
        if sources != "shared":
            submit("request-c", [newer], force=True)
        final = observe("request-b")
        expected_ids = sorted(message["id"] for message in submitted)
        expected_archived = ["msg-one"] if sources != "independent" else []
        expected_state = "SUCCEEDED" if sources == "shared" else "ACCEPTED"
        assert after_a["request_id"] == "request-a"
        assert after_a["request_status"] == "SUCCEEDED"
        for result in (first, observed, replayed):
            assert result["request_id"] == "request-b"
            assert result["request_status"] == expected_state
            assert result["source_message_ids"] == expected_ids
            assert result.get("archived_source_ids", []) == expected_archived
        assert [message["id"] for message in saved["buffer_messages"]] == (
            ["msg-two"] if sources != "shared" else [])
        assert final["request_id"] == "request-b"
        assert final["request_status"] == "SUCCEEDED"
        assert final["source_status"] == "ARCHIVED"
        assert final["archived_source_ids"] == expected_ids
        assert final["retry_action"] == "none"
        expected_archives = ({"archive-request-a"} if sources != "independent" else set())
        if sources != "shared":
            expected_archives.add(observe("request-c")["archive_id"])
        assert set(final["archive_receipts"]) == expected_archives
        if previous_archive:
            old_events = {tuple(event) for event in prior_receipt["outbox_events"]}
            assert old_events
            assert old_events <= {tuple(event) for event in after_a["outbox_events"]}
            assert old_events.isdisjoint(tuple(event) for event in final["outbox_events"])
        assert manager.get_or_create(ctx.session_id, ctx).messages == []
        assert len(extractions) == 1 + int(previous_archive) + int(sources != "shared")
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()


def test_split_source_keeps_manual_review_after_degraded_recovery(native_agfs, monkeypatch):
    client, fs, ctx = _runtime(native_agfs[0], "split-degraded")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)
    service.drain_outbox_sync = lambda **kwargs: worker.run_once(store, [ctx.account_id])
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = []
    http = http_client(monkeypatch, service, ctx)
    messages = [
        {"id": "msg-one", "role": "user", "content": "I repaired the receiver."},
        {"id": "msg-two", "role": "user", "content": "I replaced its antenna."},
    ]
    source = {"sessionId": ctx.session_id, "clientRequestId": "split-turn",
              "wait": True, "messages": messages}
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/split-turn"
    @capture_outbox_events
    def freeze_first_batch():
        buffer = manager.get_or_create(ctx.session_id, ctx)
        return ExtractionBatchJournal(
            manager, buffer, ctx, [buffer.messages[0]], archive_id="split-first",
            messages=messages[:1], session_time=None, session_summary="", tool_stats_text="",
            client_request_id="recover-turn", request_messages=messages[:1],
            request_source_ids=["msg-one"],
        )

    try:
        buffered = http.post("/api/v1/after_turn", json=source)
        record("POST", "/api/v1/after_turn", source, buffered)
        assert buffered.status_code == 202
        assert buffered.json["source_status"] == "BUFFERED"
        journal = freeze_first_batch()
        recovery = {**source, "clientRequestId": "recover-turn", "messages": messages[:1], "forceExtract": True}
        with source_only_recovery(ctx, journal.batch["archive_id"], "extraction retries exhausted"):
            recovered = http.post("/api/v1/after_turn", json=recovery)
        record("POST", "/api/v1/after_turn", recovery, recovered)
        assert recovered.json["request_status"] == "DEGRADED"
        assert recovered.json["retry_action"] == "manual_review"
        partial = http.get(path)
        record("GET", path, None, partial)
        assert partial.json["request_status"] == "ACCEPTED"
        assert partial.json["source_status"] == "BUFFERED"
        assert partial.json["archived_source_ids"] == ["msg-one"]
        assert partial.json["retry_action"] == "none"

        tail = {**source, "clientRequestId": "tail-turn", "messages": messages[1:], "forceExtract": True}
        tail_response = http.post("/api/v1/after_turn", json=tail)
        record("POST", "/api/v1/after_turn", tail, tail_response)
        assert tail_response.json["request_status"] == "SUCCEEDED"
        completed = http.get(path)
        record("GET", path, None, completed)
        assert completed.status_code == 200
        assert completed.json["request_status"] == "DEGRADED"
        assert completed.json["projection_status"] == "DONE"
        assert completed.json["archived_source_ids"] == ["msg-one", "msg-two"]
        assert completed.json["retry_action"] == "manual_review"
        assert completed.json["error_detail"]["retry_action"] == "manual_review"
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()
