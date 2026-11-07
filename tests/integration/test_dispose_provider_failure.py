"""Dispose receipts retain provider failures and recover through public disposal."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from werkzeug.serving import make_server

from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.llm.openai_llm import OpenAILLM
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs


@contextmanager
def model_provider(status):
    available = threading.Event()
    resumed = threading.Event()
    release = threading.Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tools = request.get("tools")
            response_status = status if tools and not available.is_set() else 200
            if response_status != 200:
                response = {"error": {"message": "model capacity temporarily exhausted",
                                      "type": "server_error", "code": "capacity_unavailable"}}
            else:
                message = {"role": "assistant"}
                if tools:
                    resumed.set()
                    assert release.wait(20)
                    candidate = {"routing_key": "coffee", "abstract": "Ben prefers black coffee without sugar.",
                                 "overview": "- Ben prefers black coffee without sugar.",
                                 "content": "# Coffee\n- Ben prefers black coffee without sugar.",
                                 "who": "ben", "confidence": .95,
                                 "evidence_quote": "I prefer black coffee without sugar."}
                    message["tool_calls"] = [{"id": "coffee-call", "type": "function", "function": {
                        "name": "extract_preference", "arguments": json.dumps(candidate)}}]
                else:
                    prompt = json.dumps(request["messages"])
                    if "Select all memory categories" in prompt:
                        result = {"categories": ["preference"], "shared_tools": []}
                    elif "spans" in prompt:
                        result = {"spans": [{"start": 0, "end": 0, "reason": "coffee preference",
                                              "categories": ["preference"]}]}
                    else:
                        result = {"events_with_dates": "", "facts_and_details": "", "preferences":
                                  "Ben prefers black coffee without sugar.", "plans_and_intentions": ""}
                    message["content"] = json.dumps(result)
                response = {"id": "local-provider-response", "object": "chat.completion", "created": 1,
                            "model": "test-model", "choices": [{"index": 0, "message": message,
                            "finish_reason": "tool_calls" if tools else "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
            calls.append({"method": "POST", "path": self.path, "request": request,
                          "http_status": response_status, "response": response})
            print("PROVIDER " + json.dumps(calls[-1]))
            data = json.dumps(response).encode()
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", available, resumed, release, calls
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def wait_dispose(buffer):
    workers = [thread for thread in threading.enumerate() if thread.name == f"dispose-{buffer.session_id[:8]}"]
    for thread in workers:
        thread.join(20)
    print("DISPOSE_WORKERS " + json.dumps({"threads": [{"name": t.name, "alive": t.is_alive()}
          for t in workers], "dispose_in_progress": buffer.dispose_in_progress,
          "extraction_in_progress": buffer.extraction_in_progress}))
    assert buffer.dispose_in_progress is False
    assert buffer.extraction_in_progress is False
    assert all(not thread.is_alive() for thread in workers)


@pytest.mark.parametrize("provider_status", [429, 503])
def test_dispose_provider_failure_recovers_via_dispose(native_agfs, monkeypatch, provider_status):
    import server.app as app_module

    client, fs, ctx = _runtime(native_agfs[0], "dispose-provider")
    store = OutboxStore(client, fs)
    worker = OutboxWorker(InMemoryVectorIndex(dimension=16), MockEmbedder(16), fs=fs,
                          directory_summary_enabled=False)
    services = []
    with model_provider(provider_status) as (url, available, resumed, release, calls):
        llm = OpenAILLM(api_key="local-test", base_url=url, model="test-model", request_timeout_seconds=10)

        def runtime():
            service = MemoryService(config=OgMemConfig(agfs_base_url=native_agfs[0],
                                    session_state_bridge_enabled=False, after_turn_threshold=100000,
                                    dual_pass_enabled=False))
            service._llm = llm
            service._write_api = MemoryWriteAPI(fs, llm, store, directory_summary_enabled=False,
                                               dual_pass_enabled=False)
            service._get_outbox_store = lambda _fs: store
            service.drain_outbox_sync = lambda **_kwargs: worker.run_once(store, [ctx.account_id])
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
                                    json=payload, timeout=20)
            print("PUBLIC " + json.dumps({"method": method, "path": path, "request": payload,
                  "http_status": response.status_code, "response": response.json()}))
            if method == "GET":
                assert response.status_code == 200
            return response.json()

        try:
            source = {"sessionId": ctx.session_id, "clientRequestId": "dispose-source", "wait": True,
                      "messages": [{"id": "coffee-source", "role": "user",
                                    "content": "I prefer black coffee without sugar. " * 30}]}
            assert request("POST", "/api/v1/after_turn", source)["source_status"] == "BUFFERED"
            buffer = current[0].get_session_manager().get_or_create(ctx.session_id, ctx)
            dispose = {"sessionId": ctx.session_id}
            assert request("POST", "/api/v1/dispose", dispose)["status"] == "background_flush_dispatched"
            wait_dispose(buffer)
            manager = current[0].get_session_manager()
            state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
            saved = json.loads(fs.read_node(state_uri, ctx).content)
            batch = saved["runtime_extensions"][EXTENSION_KEY]["batch"]
            print("CHECKPOINT " + json.dumps(batch))
            failure = batch["lifecycle"]["last_error"]
            assert failure["status"] == "failed"
            assert failure["error_detail"]["cause"]["type"] == "LLMTransientError"
            assert failure["error_detail"]["upstream_http_status"] == provider_status
            path = f"/api/v1/sessions/{ctx.session_id}/write-requests/dispose-source"
            failed = request("GET", path)
            current[0]._extraction_retry.close(timeout=5)
            current[0] = runtime()
            restored = request("GET", path)
            available.set()
            assert request("POST", "/api/v1/dispose", dispose)["status"] == "background_flush_dispatched"
            assert resumed.wait(20)
            running = request("GET", path)
            release.set()
            buffer = current[0].get_session_manager().get_or_create(ctx.session_id, ctx)
            wait_dispose(buffer)
            completed = request("GET", path)
            assert running["request_status"] == "RUNNING"
            assert running["retry_action"] == "poll"
            assert completed["request_status"] == "SUCCEEDED"
            assert completed["source_status"] == "RAW_ARCHIVED"
            assert completed["semantic_status"] == "COMPLETE"
            assert completed["projection_status"] == "DONE"
            assert completed["archive_id"] == batch["archive_id"]
            assert completed["archived_source_ids"] == ["coffee-source"]
            assert len(completed["raw_source_uris"]) == 1
            assert buffer.messages == []
            for result in (failed, restored):
                assert result["request_status"] == "FAILED"
                assert result["retry_action"] == "retry_dispose"
                assert result["terminal"] is True
                assert result["retryable"] is False
                assert result["accepted"] is True
                assert result["request_id"] == "dispose-source"
                assert result["http_status"] == 503
                assert result["source_status"] == "CHECKPOINTED"
                assert result["source_message_ids"] == ["coffee-source"]
                assert result["error_detail"]["code"] == "MODEL_TEMPORARILY_UNAVAILABLE"
                assert result["error_detail"]["request_id"] == "dispose-source"
                assert result["error_detail"]["upstream_http_status"] == provider_status
                assert result["error_detail"]["cause"] == failure["error_detail"]["cause"]
                assert result["error_detail"]["retry_action"] == "retry_dispose"
            assert any(call["http_status"] == provider_status for call in calls)
        finally:
            release.set()
            for dispose_thread in threading.enumerate():
                if dispose_thread.name == f"dispose-{ctx.session_id[:8]}":
                    dispose_thread.join(20)
            for service in services:
                service._extraction_retry.close(timeout=5)
            server.shutdown()
            server.server_close()
            thread.join(5)
            http.close()
            llm._client.close()
            client.session.close()
