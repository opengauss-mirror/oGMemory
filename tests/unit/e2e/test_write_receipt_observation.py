"""Normal HTTP receipt observation and durable evaluation source snapshots."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e"))

from agent_plugin_locomo import PluginAdapter, ReadOnlyQaProxy, text_fingerprint
from core.models import RequestContext
import eval as evaluation
import evaluation_audit
from evaluation_settlement import collect_settlement, inventory


@pytest.fixture
def receipt_http():
    calls, receipts = [], {"state": "ACCEPTED", "buffered": True}

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            url = urlsplit(self.path)
            path = url.path
            request_id = parse_qs(url.query).get("clientRequestId", ["server-1"])[0]
            calls.append({"method": self.command, "path": path, "body": payload, "request_id": request_id})
            status = 200
            if path.endswith("/after_turn"):
                status = 202
                result = {"ok": True, "request_id": "server-1", "request_status": "ACCEPTED",
                          "source_status": "BUFFERED", "retry_action": "none"}
            elif "/write-requests/" in path or path.endswith("/write-requests"):
                result = receipts.get("settled") or {
                    "ok": True, "request_id": request_id, "request_status": receipts["state"],
                    "source_status": "BUFFERED" if receipts["buffered"] else "ARCHIVED",
                    "retry_action": "none" if receipts["state"] == "SUCCEEDED" else "poll",
                }
                receipts["state"] = "SUCCEEDED"
            elif path.endswith("/wait_until_idle"):
                result = {"ok": True, "idle": True, "outbox": {"pending": 0, "processing": 0, "total": 0}}
            elif path.endswith("/compact"):
                receipts.update(state="SUCCEEDED", buffered=False)
                result = receipts.get("compacted") or {"ok": True, "compacted": True}
            elif path.endswith("/dispose"):
                result = {"ok": True, "disposed": True}
            elif path.endswith("/context"):
                result = {"ok": True, "archive_count": 1}
            else:
                result = {"ok": True, "message_count": int(receipts["buffered"]),
                          "commit_count": int(not receipts["buffered"]), "pending_tokens": 0}
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET = handle_request

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = SimpleNamespace(ogmemory_url=f"http://127.0.0.1:{server.server_port}",
                           ogmemory_api_key="fixture-key", account_id="acme", user_id="alice",
                           memory_agent_id="worker", write_wait_timeout=5)
    try:
        yield SimpleNamespace(args=args, calls=calls, receipts=receipts)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_receipt_observation_records_each_http_get(receipt_http):
    receipt_http.receipts["buffered"] = False
    observations = []
    request_id = "turn/%2F"
    result = evaluation.wait_write_request(receipt_http.args, "session", request_id, observations=observations)

    assert result["request_status"] == "SUCCEEDED"
    assert [(row["method"], row["http_status"], row["request_id"]) for row in observations] == [
        ("GET", 200, request_id), ("GET", 200, request_id)]
    assert [call["method"] for call in receipt_http.calls] == ["GET", "GET"]
    assert [call["request_id"] for call in receipt_http.calls] == [request_id, request_id]


def test_buffered_source_settles_through_compact_and_dispose(receipt_http):
    result = evaluation.flush_source(receipt_http.args, "session")

    assert result["archive_count"] == 1
    assert result["pending_tokens"] == 0
    assert [row["path"].rsplit("/", 1)[1] for row in result["http_observations"]
            if row["method"] == "POST"] == ["wait_until_idle", "compact", "wait_until_idle", "dispose", "wait_until_idle"]
    assert len(result["http_observations"]) == len(receipt_http.calls)


def test_source_only_compaction_records_degradation_after_projection(receipt_http):
    degraded = {"ok": False, "request_id": "compact-1", "request_status": "DEGRADED",
                "source_status": "ARCHIVED", "semantic_status": "DEGRADED", "archive_id": "archive-1",
                "source_message_ids": ["msg-1"], "archived_source_ids": ["msg-1"],
                "recovery_mode": "source_only", "error_detail": {"code": "EXTRACTION_OUTPUT_INVALID"}}
    receipt_http.receipts.update(
        compacted={**degraded, "compacted": True, "projection_status": "PENDING"},
        settled={**degraded, "projection_status": "DONE"},
    )

    result = evaluation.flush_source(receipt_http.args, "session")

    receipt = result["degraded_writes"]["compact"]
    assert (receipt["request_id"], receipt["request_status"], receipt["projection_status"]) == (
        "compact-1", "DEGRADED", "DONE")
    assert receipt["error_code"] == "EXTRACTION_OUTPUT_INVALID"
    assert result["archive_count"] == 1
    assert [call["request_id"] for call in receipt_http.calls if call["path"].endswith("/write-requests")] == ["compact-1"]


def test_proxy_preserves_server_and_client_receipt_identity(receipt_http):
    proxy = ReadOnlyQaProxy(receipt_http.args.ogmemory_url)
    proxy.start()
    try:
        result = requests.post(proxy.url + "/api/v1/after_turn", json={
            "sessionId": "session", "clientRequestId": "client-1",
            "messages": [{"role": "user", "content": "Remember blue."}],
        }).json()
        requests.get(proxy.url + "/api/v1/sessions/session/write-requests",
                     params={"clientRequestId": "server-1"}).raise_for_status()
        assert result["request_id"] == "server-1"
        assert proxy.observed_writes[0]["client_request_id"] == "client-1"
        assert proxy.observed_writes[0]["request_id"] == "server-1"
        assert proxy.request_observations[0]["request_id"] == "server-1"
        assert proxy.observed_writes[0]["user_message_fingerprints"] == [text_fingerprint("Remember blue.")]
    finally:
        proxy.close()


def test_plugin_adapter_flushes_buffered_receipt_with_server_identity(receipt_http, tmp_path):
    args = receipt_http.args
    args.agent, args.verify_writes, args.output_dir = "openclaw", True, tmp_path
    adapter = PluginAdapter(args, evaluation)
    adapter.phase = "ingest"
    adapter.proxy = ReadOnlyQaProxy(args.ogmemory_url)
    adapter.proxy.start()

    def invoke(text, key):
        requests.post(adapter.proxy.url + "/api/v1/after_turn", json={
            "sessionId": "session", "clientRequestId": "client-1",
            "messages": [{"role": "user", "content": text}],
        }).raise_for_status()
        return "Saved.", {}

    adapter.host = SimpleNamespace(invoke=invoke, close=lambda: None)
    try:
        reply, usage, ids = adapter.ingest("Remember blue.", "ingest-1")
        settled = adapter.settle(ids[0])
        assert (reply, usage, ids) == ("Saved.", {}, ["session"])
        assert settled["requests"][0]["request_id"] == "server-1"
        assert settled["requests"][0]["request_status"] == "SUCCEEDED"
        assert sum(call["path"].endswith("/after_turn") for call in receipt_http.calls) == 1
    finally:
        adapter.close()
    evidence = json.loads((tmp_path / "ingest-write-observations.json").read_text())
    assert evidence["observed"][0]["request_id"] == "server-1"
    assert evidence["request_status"][-1]["outcome"]["request_status"] == "SUCCEEDED"


def write_node(root, directory, metadata, content):
    path = root / directory
    path.mkdir(parents=True)
    (path / ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "content.md").write_text(json.dumps(content), encoding="utf-8")


def test_new_account_baseline_then_normal_files_form_stable_source(tmp_path, monkeypatch):
    root = tmp_path / "acme"
    ctx = RequestContext("acme", "alice", "worker", "session", "trace")
    monkeypatch.setenv("NODE_AUDIT_ACCOUNT_ROOT", str(root))
    monkeypatch.setattr(evaluation_audit, "collect_database", lambda cfg, account_id: {
        "available": True, "records": {}, "records_sha256": evaluation_audit.digest({}), "record_count": 0,
    })
    baseline = evaluation_audit.collect_audit(SimpleNamespace(_cfg=None), ctx, {})
    assert baseline["baseline"]["nodes"] == {}
    assert baseline["checks"]["stable_source_snapshot"] is True
    metadata = {"account_id": "acme", "user_id": "alice", "agent_id": "worker", "session_id": "session"}
    write_node(root, "state", {"uri": "ctx://acme/sessions/session/state.json"}, {
        "session_meta": metadata, "completed_request_ids": ["client-1"], "buffer_messages": [],
        "runtime_extensions": {},
    })
    write_node(root, "archive", {"uri": "ctx://acme/users/alice/history/session/archive-1",
        "session_id": "session", "archive_id": "archive-1", "category": "history"},
        [{"role": "user", "content": "Remember blue."}])

    result = collect_settlement(None, ctx, {"expected_user_texts": ["Remember blue."], "client_request_ids": ["client-1"]})

    assert result["complete"] is True
    assert all(result["checks"].values())
    assert len(inventory(root)) == 4


def test_buffered_state_audit_exposes_status_and_counts():
    result = evaluation_audit.state_summary({
        "session_meta": {"account_id": "acme", "user_id": "alice", "agent_id": "worker", "session_id": "session"},
        "completed_request_ids": [], "buffer_messages": [{"role": "user", "content": "Remember blue."}],
        "runtime_extensions": {"memory_extraction_recovery": {"status": "RUNNING", "request_id": "server-1",
                                                               "pending_summary": "Deployment notes"}},
    })

    assert result == {
        "session_meta": {"account_id": "acme", "user_id": "alice", "agent_id": "worker", "session_id": "session"},
        "completed_request_ids": [], "buffer_message_count": 1,
        "pending_recovery": {"pending": True, "status": "RUNNING", "request_id": "server-1"},
    }


def test_projection_audit_matches_current_worker_embedding_identity():
    from dataclasses import asdict
    from core.models import ContextNode
    from core.transcript_lexical import indexed_metadata
    from index.index_record_builder import build_index_records
    from index.outbox_worker import OutboxWorker
    from providers.unified_config import OgMemConfig

    config = OgMemConfig(openai_embedding_model="bge-m3", opengauss_dimension=1024)
    embedder = SimpleNamespace(model=config.openai_embedding_model)
    node = ContextNode(uri="ctx://acme/users/alice/memories/entity/project", context_type="MEMORY",
        category="entity", level=2, owner_space="user:alice", abstract="Alice maintains the project.",
        overview="Alice maintains a database project.", content="Alice maintains the database project.",
        metadata={"version": 3, "provenance_ids": ["prov:1:archive:archive-1:message-1"]})
    records = build_index_records(node)
    vectors = [[0.25] * config.opengauss_dimension for _ in records]
    worker = OutboxWorker(vector_index=None, embedder=embedder, directory_summary_enabled=False)
    projected = worker._augment_with_embeddings(records, vectors)
    for expected, record in zip(records, projected):
        actual = {**asdict(record), "metadata": indexed_metadata(record.metadata)}
        result = evaluation_audit.validate_projection_record(
            asdict(expected), actual, config.opengauss_dimension, embedder=embedder)
        assert result["matches"] is True
        assert result["expected"] == actual
        assert result["expected"]["metadata"]["embedding_model"] == "bge-m3"
        assert result["expected"]["metadata"]["embedding_dimension"] == 1024
        assert result["embedding_validation"] == {"configured_model": "bge-m3",
            "configured_dimension": 1024, "actual_dimension": 1024,
            "dimension_matches": True, "numeric_finite": True, "nonzero": True}
