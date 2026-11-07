"""Governed ingest handoff retains the source-to-receipt association."""

import json
import time

import pytest

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_recovery_discovery import ExtractionRecoveryDiscovery
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import setup_service


@pytest.mark.parametrize("mode", ["partial_handoff", "complete_handoff", "semantic_turn"])
def test_governed_ingest_handoff_reaches_durable_completion(
    native_agfs, monkeypatch, tmp_path, mode,
):
    import server.app as app_module
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "governed-receipt")
    store = OutboxStore(client, fs)
    service, writer, manager = setup_service(fs, ctx, outbox=store)
    manager._get_context_fs = lambda: fs
    service._cfg.after_turn_threshold = 100000
    service._cfg.tool_result_externalization_enabled = True
    service._initialize_tool_result_externalization(LocalToolResultArtifactStore(tmp_path / "artifacts"))
    service.get_llm = lambda: None
    service._generate_summary_candidate = lambda *args, **kwargs: None
    service._get_outbox_store = lambda _fs: store
    service._flush_session_tool_usage_async = lambda *args: None
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    http = app_module.app.test_client()
    source = [{"role": "user", "content": "I repaired the receiver. " * 50},
              {"role": "assistant", "content": "The repair is recorded."}]
    if mode == "semantic_turn":
        source[1:1] = [
            {"role": "assistant", "content": [{"type": "toolCall", "id": "repair-check",
             "name": "repair_log", "arguments": {}}], "stopReason": "toolUse"},
            {"role": "toolResult", "content": "Receiver repair complete.",
             "toolCallId": "repair-check", "toolName": "repair_log"},
        ]
    count = 1 if mode == "partial_handoff" else len(source)
    path = "/api/v1/after_turn" if mode == "semantic_turn" else "/api/v1/ingest"
    ingest = {"sessionId": ctx.session_id, "messages": source[:count]}
    print("REQUEST POST " + path + " " + json.dumps(ingest))
    response = http.post(path, json=ingest)
    print("RESPONSE ingest " + json.dumps({"status": response.status_code, "body": response.json}))
    assert response.status_code in (200, 202)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    ingested_ids = [item.id for item in buffer.messages]
    request = {"sessionId": ctx.session_id, "messages": source,
               "clientRequestId": "governed-source", "wait": True}
    print("REQUEST POST /api/v1/after_turn " + json.dumps(request))
    accepted = http.post("/api/v1/after_turn", json=request)
    print("RESPONSE after_turn " + json.dumps({"status": accepted.status_code, "body": accepted.json}))
    assert accepted.json["source_status"] == "BUFFERED"
    source_ids = [item.id for item in buffer.messages]
    assert source_ids[:len(ingested_ids)] == ingested_ids
    assert len(source_ids) == (3 if mode == "semantic_turn" else 2)
    assert set(accepted.json["source_message_ids"]) == set(source_ids)
    print("REQUEST POST /api/v1/dispose " + json.dumps({"sessionId": ctx.session_id}))
    disposed = http.post("/api/v1/dispose", json={"sessionId": ctx.session_id})
    print("RESPONSE dispose " + json.dumps({"status": disposed.status_code, "body": disposed.json}))
    deadline = time.monotonic() + 20
    while buffer.dispose_in_progress and time.monotonic() < deadline:
        time.sleep(.02)
    assert buffer.dispose_in_progress is False
    state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
    saved = json.loads(fs.read_node(state_uri, ctx).content)
    receipt = saved["runtime_extensions"][OUTCOMES_KEY]["entries"]["governed-source"]
    print("RESPONSE durable_receipt " + json.dumps(receipt))
    pending = store.list_pending(ctx.account_id)
    references = [(uri, event.event_id) for uri, event in pending]
    stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                         directory_summary_enabled=False).run_once(store, [ctx.account_id], pending_events=pending)
    print("RESPONSE projection " + json.dumps(stats))
    assert stats["succeeded"] == len(references)
    assert [store.get_event_status(*item) for item in references] == ["DONE"] * len(references)
    ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id],
                                projection_handler=service._refresh_write_outcomes)()
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/governed-source"
    print("REQUEST GET " + path)
    completed = http.get(path)
    print("RESPONSE status " + json.dumps({"status": completed.status_code, "body": completed.json}))
    assert completed.json["request_status"] == "SUCCEEDED"
    assert completed.json["projection_status"] == "DONE"
    assert set(receipt["source_message_ids"]) == set(source_ids)
    assert set(map(tuple, receipt["outbox_events"])) == set(references)
    assert saved["buffer_messages"] == []
    writer._pipeline.extract.assert_called_once()
    client.session.close()
