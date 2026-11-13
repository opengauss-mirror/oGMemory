"""Public disposal preserves write receipts through durable projection."""

import json
import threading
import time

import pytest

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_recovery_discovery import ExtractionRecoveryDiscovery
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_extraction_batch_recovery import setup_service


@pytest.mark.parametrize("repeats,state", [(1, "DEGRADED"), (50, "SUCCEEDED")])
def test_public_dispose_receipt_settles_after_projection(native_agfs, monkeypatch, repeats, state):
    import server.app as app_module

    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "dispose-receipt")
    store = OutboxStore(client, fs)
    service, writer, manager = setup_service(fs, ctx, outbox=store)
    manager._get_context_fs = lambda: fs
    service._cfg.after_turn_threshold = 100000
    service.get_llm = lambda: None
    service._generate_summary_candidate = lambda *args, **kwargs: None
    service._get_outbox_store = lambda _fs: store
    service._flush_session_tool_usage_async = lambda *args: None
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    http = app_module.app.test_client()
    source = [{"role": "user", "content": "I repaired the receiver. " * repeats},
              {"role": "assistant", "content": "The repair is recorded."}]
    request = {"sessionId": ctx.session_id, "clientRequestId": "dispose-source", "messages": source,
               "wait": True}
    print("REQUEST after_turn " + json.dumps(request))
    accepted = http.post("/api/v1/after_turn", json=request)
    print("RESPONSE after_turn " + json.dumps({"status": accepted.status_code, "body": accepted.json}))
    assert accepted.json["source_status"] == "BUFFERED"
    buffer = manager.get_or_create(ctx.session_id, ctx)
    print("REQUEST dispose " + json.dumps({"sessionId": ctx.session_id}))
    disposed = http.post("/api/v1/dispose", json={"sessionId": ctx.session_id})
    print("RESPONSE dispose " + json.dumps({"status": disposed.status_code, "body": disposed.json}))
    deadline = time.monotonic() + 20
    while buffer.dispose_in_progress and time.monotonic() < deadline:
        time.sleep(.02)
    assert buffer.dispose_in_progress is False
    state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
    saved = json.loads(fs.read_node(state_uri, ctx).content)
    receipt = saved["runtime_extensions"][OUTCOMES_KEY]["entries"]["dispose-source"]
    print("RESPONSE durable_receipt " + json.dumps(receipt))
    pending = store.list_pending(ctx.account_id)
    references = [(uri, event.event_id) for uri, event in pending]
    index = InMemoryVectorIndex(dimension=16)
    stats = OutboxWorker(index, MockEmbedder(16), fs=fs, directory_summary_enabled=False).run_once(
        store, [ctx.account_id], pending_events=pending)
    print("RESPONSE projection " + json.dumps(stats))
    assert stats["succeeded"] == len(references)
    assert [store.get_event_status(*item) for item in references] == ["DONE"] * len(references)
    ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id],
                                projection_handler=service._refresh_write_outcomes)()
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/dispose-source"
    print("REQUEST GET " + path)
    completed = http.get(path)
    print("RESPONSE status " + json.dumps({"status": completed.status_code, "body": completed.json}))
    assert completed.json["request_status"] == state
    assert completed.json["projection_status"] == "DONE"
    assert set(map(tuple, receipt["outbox_events"])) == set(references)
    refreshed = json.loads(fs.read_node(state_uri, ctx).content)
    persisted = refreshed["runtime_extensions"][OUTCOMES_KEY]["entries"]["dispose-source"]
    assert (persisted["request_status"], persisted["projection_status"]) == (state, "DONE")
    assert saved["buffer_messages"] == []
    archived = "\n".join(fs.read_node(uri, ctx).content for uri in receipt["raw_source_uris"])
    assert all(f"{message['role']}: {message['content']}" in archived for message in source)
    assert writer._pipeline.extract.call_count == (1 if repeats == 50 else 0)
    client.session.close()


def test_fast_dispose_projects_nondefault_agfs_account(native_agfs, monkeypatch):
    import server.app as app_module

    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "dispose-nondefault")
    store = OutboxStore(client, fs)
    index = InMemoryVectorIndex(dimension=16)
    embedder = MockEmbedder(16)
    service = MemoryService(OgMemConfig(
        agfs_base_url=address, account_id="dispose-default", role_control_enabled=False,
        session_state_bridge_enabled=False, after_turn_threshold=100000,
        directory_summary_enabled=False,
    ))
    service._write_api = MemoryWriteAPI(fs, None, store, directory_summary_enabled=False)
    service._vector_index = service._projection_index = index
    service._embedder = embedder
    upsert = index.upsert

    def project(records):
        assert fs.active_session_recovery() is None
        return upsert(records)

    monkeypatch.setattr(index, "upsert", project)
    service._get_outbox_store = lambda _fs: store
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    http = app_module.app.test_client()
    scans = []
    scanned = threading.Event()
    list_pending = store.list_pending

    def observe_scan(account_id):
        result = list_pending(account_id)
        if threading.current_thread().name == "outbox-worker":
            scans.append(account_id)
            scanned.set()
        return result

    monkeypatch.setattr(store, "list_pending", observe_scan)

    def request(method, path, **kwargs):
        response = http.open(path, method=method, **kwargs)
        print("HTTP " + json.dumps({"method": method, "path": path, "request": kwargs,
              "http_status": response.status_code, "response": response.json}))
        return response.json

    identity = {"accountId": ctx.account_id, "userId": ctx.user_id, "agentId": ctx.agent_id}
    lifecycle = {**identity, "sessionId": ctx.session_id}
    message = "Remember the blue receiver serial number R17."
    try:
        service._start_outbox_worker()
        assert scanned.wait(10)
        accepted = request("POST", "/api/v1/after_turn", json={**lifecycle, "wait": True,
            "clientRequestId": "dispose-short", "messages": [{"id": "source-one", "role": "user", "content": message}]})
        assert accepted["source_status"] == "BUFFERED"
        scanned.clear()
        disposed = request("POST", "/api/v1/dispose", json=lifecycle)
        assert disposed["reason"] == "below_token_threshold"
        assert scanned.wait(10)
        receipt = request("GET", f"/api/v1/sessions/{ctx.session_id}/write-requests/dispose-short",
                          query_string=identity)
        refs = receipt["outbox_events"]
        states = [store.get_event_status(*ref) for ref in refs]
        query = embedder.embed_texts([message])[0]
        hits = [hit.uri for hit in index.search_by_vector(query, {"account_id": ctx.account_id}, 10)]
        print("PROJECTION " + json.dumps({"background_accounts": scans, "events": refs,
              "states": states, "vector_hits": hits}))
        assert set(scans) == {"dispose-default"}
        assert receipt["source_status"] == "RAW_ARCHIVED"
        assert states == ["DONE"]
        assert receipt["projection_status"] == "DONE"
        assert set(receipt["raw_source_uris"]) & set(hits)
    finally:
        service.shutdown(join_timeout=5)
        client.session.close()
