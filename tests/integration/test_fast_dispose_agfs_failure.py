"""Fast disposal preserves AGFS failure receipts and resumes committed source."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from werkzeug.serving import make_server

from commit.outbox_store import OutboxStore
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs


@contextmanager
def agfs_proxy(address):
    failed = threading.Event()
    recovered = threading.Event()
    resumed = threading.Event()
    release = threading.Event()
    faults = []
    upstream = requests.Session()
    upstream.trust_env = False

    class Handler(BaseHTTPRequestHandler):
        def relay(self):
            payload = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            path = parse_qs(urlsplit(self.path).query).get("path", [""])[0]
            target = self.command == "PUT" and "/session_archives/" in path and "/.outbox/" in path
            if target and not recovered.is_set():
                failed.set()
                status, content = 503, json.dumps({"error": "AGFS outbox temporarily unavailable"}).encode()
                faults.append({"method": self.command, "path": self.path, "file_path": path,
                               "request": json.loads(payload), "http_status": status,
                               "response": json.loads(content)})
                print("AGFS_FAULT " + json.dumps(faults[-1]))
            else:
                if target:
                    resumed.set()
                    assert release.wait(20)
                response = upstream.request(self.command, address + self.path, data=payload,
                    headers={"Content-Type": self.headers.get("Content-Type", "application/octet-stream")},
                    timeout=20)
                status, content = response.status_code, response.content
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = do_POST = do_PUT = do_DELETE = relay  # noqa: N815

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", failed, recovered, resumed, release, faults
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        upstream.close()


@pytest.mark.parametrize("role,content,reason", [
    ("user", "I prefer black coffee without sugar.", "below_token_threshold"),
    ("assistant", "The requested receiver diagnosis is ready. " * 30, "no_user_messages"),
])
def test_fast_dispose_agfs_failure_has_durable_receipt(native_agfs, monkeypatch, role, content, reason):
    import server.app as app_module

    services = []
    with agfs_proxy(native_agfs[0]) as (address, failed, recovered, resumed, release, faults):
        client, fs, ctx = _runtime(address, "dispose-agfs")
        store = OutboxStore(client, fs)
        index = InMemoryVectorIndex(dimension=16)
        embedder = MockEmbedder(16)

        def runtime():
            service = MemoryService(config=OgMemConfig(agfs_base_url=address,
                session_state_bridge_enabled=False, after_turn_threshold=100000))
            service._write_api = MemoryWriteAPI(fs, None, store, directory_summary_enabled=False)
            service._vector_index = service._projection_index = index
            service._embedder = embedder
            service._get_outbox_store = lambda _fs: store
            services.append(service)
            return service

        current = [runtime()]
        monkeypatch.setattr(app_module, "_get_service", lambda: current[0])
        monkeypatch.setattr(app_module, "_build_authenticated_context", lambda _params: ctx)
        server = make_server("127.0.0.1", 0, app_module.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        http = requests.Session()
        http.trust_env = False

        def request(method, path, payload=None):
            response = http.request(method, f"http://127.0.0.1:{server.server_port}" + path,
                                    json=payload, timeout=30)
            print("PUBLIC " + json.dumps({"method": method, "path": path, "request": payload,
                  "http_status": response.status_code, "response": response.json()}))
            if method == "GET":
                assert response.status_code == 200
            return response.json()

        try:
            source = {"sessionId": ctx.session_id, "clientRequestId": "fast-source", "wait": True,
                      "messages": [{"id": "source-one", "role": role, "content": content}]}
            assert request("POST", "/api/v1/after_turn", source)["source_status"] == "BUFFERED"
            dispose = {"sessionId": ctx.session_id}
            result = request("POST", "/api/v1/dispose", dispose)
            assert failed.is_set() and faults
            manager = current[0].get_session_manager()
            buffer = manager.get_or_create(ctx.session_id, ctx)
            print("OPERATIONS " + json.dumps({"dispose": buffer.dispose_in_progress,
                  "extraction": buffer.extraction_in_progress, "active": buffer.active_operation_count}))
            assert buffer.dispose_in_progress is False
            assert buffer.extraction_in_progress is False
            assert buffer.active_operation_count == 0
            state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
            saved = json.loads(fs.read_node(state_uri, ctx).content)
            batch = saved["runtime_extensions"][EXTENSION_KEY]["batch"]
            print("CHECKPOINT " + json.dumps(batch))
            path = f"/api/v1/sessions/{ctx.session_id}/write-requests/fast-source"
            stopped = request("GET", path)
            current[0]._extraction_retry.close(timeout=5)
            current[0] = runtime()
            cold = request("GET", path)
            recovered.set()
            with ThreadPoolExecutor(max_workers=1) as executor:
                retry = executor.submit(request, "POST", "/api/v1/dispose", dispose)
                try:
                    assert resumed.wait(20)
                    running = request("GET", path)
                finally:
                    release.set()
                retried = retry.result(timeout=20)
            completed = request("GET", path)
            assert result["error_detail"]["cause"]["type"] == "WriteOutcomeUnknownError"
            assert retried["reason"] == reason
            assert running["request_status"] == "RUNNING"
            assert completed["request_status"] == "DEGRADED"
            assert completed["source_status"] == "RAW_ARCHIVED"
            assert completed["projection_status"] == "UNKNOWN"
            assert completed["archive_id"] == batch["archive_id"]
            assert len(completed["raw_source_uris"]) == 1
            raw_uri = completed["raw_source_uris"][0]
            event_id = faults[-1]["request"]["event_id"]
            projection = store.get_event_status(raw_uri, event_id)
            print("PROJECTION " + json.dumps({"uri": raw_uri, "event_id": event_id, "status": projection}))
            assert projection == "DONE"
            hits = index.search_by_vector(embedder.embed_texts([content])[0], {"account_id": ctx.account_id}, 10)
            assert raw_uri in {hit.uri for hit in hits}
            raw = fs.read_node(raw_uri, ctx)
            assert raw.content == f"{role}: {content}"
            for receipt in (stopped, cold):
                assert receipt["request_status"] == "NEEDS_CONFIRMATION"
                assert receipt["terminal"] is True
                assert receipt["retry_action"] == "manual_review"
                assert receipt["accepted"] is True
                assert receipt["request_id"] == "fast-source"
                assert receipt["error_detail"]["cause"]["type"] == "WriteOutcomeUnknownError"
                assert receipt["error_detail"]["upstream_http_status"] == 503
            assert batch["lifecycle"]["last_error"]["status"] == "needs_confirmation"
        finally:
            release.set()
            for service in services:
                service._extraction_retry.close(timeout=5)
            server.shutdown()
            server.server_close()
            thread.join(5)
            http.close()
            client.session.close()
