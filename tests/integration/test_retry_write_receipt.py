"""A waiting concurrent writer resumes through durable request completion."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from commit.outbox_store import OutboxStore
from core.write_contract import OUTCOMES_KEY
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_lifecycle_batch_recovery import real_archives
from tests.unit.server.test_write_receipt_transitions import http_client, record


@pytest.mark.parametrize("background", [False, True])
def test_waiting_writer_finishes_with_tracked_projection(native_agfs, monkeypatch, background):
    import fs.agfs_adapter.agfs_context_fs as agfs_module
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "retry-capture")
    store = OutboxStore(client, fs)
    service, writer, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    manager._get_context_fs = lambda: fs
    http = http_client(monkeypatch, service, ctx)
    directory = writer._writer._identity_resolver.uris.get_directory_uri("event", ctx).rstrip("/")
    acquired, release = threading.Event(), threading.Event()
    monkeypatch.setattr(agfs_module, "_MEMORY_IDENTITY_LOCK_TIMEOUT_SECONDS", .2)

    def concurrent_writer():
        with fs.resource_mutation_lock(directory, ctx):
            acquired.set()
            assert release.wait(30)

    request = {"sessionId": ctx.session_id, "clientRequestId": "retry-source",
               "wait": True, "retryInBackground": background, "messages": [
                   {"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    path = "/api/v1/after_turn"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            owner = pool.submit(concurrent_writer)
            try:
                assert acquired.wait(10)
                response = http.post(path, json=request)
                record("POST", path, request, response)
                buffer = manager.get_or_create(ctx.session_id, ctx)
                batch = buffer.runtime_extensions.get(EXTENSION_KEY, {}).get("batch", {})
                print("CHECKPOINT registered " + json.dumps(batch))
            finally:
                release.set()
            owner.result(timeout=10)
        deadline = time.monotonic() + 25
        while buffer.runtime_extensions.get(EXTENSION_KEY) and time.monotonic() < deadline:
            time.sleep(.05)
        state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
        with buffer.session_state_save_lock:
            saved = json.loads(fs.read_node(state_uri, ctx).content)
        receipt = saved["runtime_extensions"][OUTCOMES_KEY]["entries"]["retry-source"]
        print("CHECKPOINT completed " + json.dumps(receipt))
        pending = store.list_pending(ctx.account_id)
        references = [(uri, event.event_id) for uri, event in pending]
        stats = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                             directory_summary_enabled=False).run_once(
                                 store, [ctx.account_id], pending_events=pending)
        print("PROJECTION " + json.dumps(stats))
        print("OUTBOX " + json.dumps([store.get_event_status(*ref) for ref in references]))
        query = f"/api/v1/sessions/{ctx.session_id}/write-requests/retry-source"
        final = http.get(query)
        record("GET", query, None, final)
        assert response.json["request_status"] == "RECOVERING"
        assert final.json["request_status"] == "SUCCEEDED"
        assert final.json["projection_status"] == "DONE"
        assert receipt["outbox_tracking_complete"] is True
        assert set(map(tuple, receipt["outbox_events"])) == set(references)
        assert writer._pipeline.extract.call_count == 1
    finally:
        service._extraction_retry.close(timeout=5)
        client.session.close()
