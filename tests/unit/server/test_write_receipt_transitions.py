"""Write receipts preserve retry decisions and advance with durable projections."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.write_contract import normalize_write_result, write_http_status
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.commit.test_request_outbox_events import make_runtime, request_context
from tests.unit.server.test_lifecycle_batch_recovery import real_archives


def http_client(monkeypatch, service, ctx):
    import server.app as app_module

    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    return app_module.app.test_client()


def record(method, path, payload, response):
    print(json.dumps({"method": method, "path": path, "request": payload,
                      "http_status": response.status_code, "response": response.json},
                     ensure_ascii=False, default=str))


@pytest.mark.parametrize("status,error", [
    ("failed", "idempotency state unavailable"),
])
def test_legacy_transient_acknowledgement_preserves_request_retry(status, error):
    result = normalize_write_result({"ok": False, "status": status, "retryable": True,
                                     "error": error}, "turn-retry")
    assert (result["request_status"], result["accepted"], result["retryable"],
            result["retry_action"], write_http_status(result)) == (
        "FAILED", False, True, "retry_same_request", 503)
    assert normalize_write_result(result) == result


def test_request_observation_follows_initial_submission(monkeypatch):
    _, fs, store = make_runtime()
    ctx = request_context("receipt-first-observation")
    service, _, _ = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    service._cfg.after_turn_threshold = 100000
    client = http_client(monkeypatch, service, ctx)
    path = f"/api/v1/sessions/{ctx.session_id}/write-requests/turn-one"
    initial = client.get(path)
    record("GET", path, None, initial)
    request = {"sessionId": ctx.session_id, "clientRequestId": "turn-one", "wait": True,
               "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    accepted = client.post("/api/v1/after_turn", json=request)
    observed = client.get(path)
    record("GET", path, None, observed)
    assert initial.status_code == 200
    assert initial.json["request_status"] == "UNKNOWN"
    assert accepted.json["accepted"] is True
    assert observed.json["request_status"] == "ACCEPTED"


def test_turn_retries_same_request_after_compaction_preparation(monkeypatch):
    fs = MemoryFS()
    ctx = request_context("receipt-busy")
    service, _, manager = real_archives(fs, ctx)
    service._cfg.after_turn_threshold = 100000
    client = http_client(monkeypatch, service, ctx)
    path = "/api/v1/after_turn"
    first = {"sessionId": ctx.session_id, "clientRequestId": "turn-one", "wait": True,
             "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    assert client.post(path, json=first).json["ok"] is True
    reached, release = threading.Event(), threading.Event()
    original = service._compaction_journal

    def prepare_journal(result):
        reached.set()
        assert release.wait(30)
        return original(result)

    monkeypatch.setattr(service, "_compaction_journal", prepare_journal)
    request = {**first, "clientRequestId": "turn-two",
               "messages": [{"id": "msg-two", "role": "user", "content": "I replaced the antenna."}]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        prepared = pool.submit(service.prepare_compaction, {"sessionId": ctx.session_id, "_ctx": ctx})
        try:
            assert reached.wait(30)
            busy = client.post(path, json=request)
            record("POST", path, request, busy)
        finally:
            release.set()
        preparation = prepared.result(timeout=30)
    compacted = service.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                                 "prepareToken": preparation["prepareToken"]})
    assert compacted["compacted"] is True
    resumed = client.post(path, json=request)
    record("POST", path, request, resumed)
    assert resumed.json["accepted"] is True
    assert [message.id for message in manager.get_or_create(ctx.session_id, ctx).messages] == ["msg-two"]
    assert (busy.status_code, busy.json["retry_action"], busy.json["accepted"]) == (
        409, "retry_same_request", False)
    assert busy.json["error_detail"]["code"] == "SESSION_WRITE_BUSY"
    assert busy.json["retryable"] is True


@pytest.mark.parametrize("completion,status,ok,http_status", [
    ("mark_done", "SUCCEEDED", True, 200),
    ("move_to_dlq", "NEEDS_CONFIRMATION", False, 409),
])
def test_projection_terminal_receipt_updates_query_and_replay(
        monkeypatch, completion, status, ok, http_status):
    _, fs, store = make_runtime()
    ctx = request_context("receipt-projection-" + completion)
    service, _, manager = real_archives(fs, ctx, outbox=store)
    service._get_outbox_store = lambda _fs: store
    client = http_client(monkeypatch, service, ctx)
    path = "/api/v1/after_turn"
    request = {"sessionId": ctx.session_id, "clientRequestId": "turn-one", "wait": True,
               "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    pending = client.post(path, json=request)
    record("POST", path, request, pending)
    assert pending.json["request_status"] == "PROJECTING"
    for uri, event in store.list_pending_events([tuple(ref) for ref in pending.json["outbox_events"]]):
        with store.claim_processing_event(event, uri, "receipt-worker") as claimed:
            assert claimed is True
            getattr(store, completion)(event, uri)
    query_path = f"/api/v1/sessions/{ctx.session_id}/write-requests/turn-one"
    queried = client.get(query_path)
    record("GET", query_path, None, queried)
    replayed = client.post(path, json=request)
    record("POST", path, request, replayed)
    for response in (queried, replayed):
        assert (response.json["request_status"], response.json["ok"], response.json["http_status"]) == (
            status, ok, http_status)
        assert response.json["terminal"] is True
        assert response.json["accepted"] is True
    assert queried.status_code == 200
    assert replayed.status_code == http_status
    assert manager.get_or_create(ctx.session_id, ctx).completed_request_ids == ["turn-one"]
