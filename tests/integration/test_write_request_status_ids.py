"""Accepted opaque request IDs retain their identity across HTTP status routes."""

import json
import socket
import threading
import time
from urllib.parse import quote

import pytest
import requests
import uvicorn
from werkzeug.serving import make_server

from commit.outbox_store import OutboxStore
from server.auth import AuthService
from server.mcp_server import create_mcp_http_app, create_mcp_server
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_agfs_node_snapshot_reads import native_agfs as native_agfs
from tests.unit.server.test_lifecycle_batch_recovery import real_archives


@pytest.fixture
def status_servers(native_agfs, monkeypatch):
    import server.app as app_module

    client, fs, ctx = _runtime(native_agfs[0], "receipt-route")
    store = OutboxStore(client, fs)
    service, _, _ = real_archives(fs, ctx, outbox=store)
    service._cfg.after_turn_threshold = 100000
    service._cfg.role_control_enabled = True
    service._cfg.root_api_key = "route-test-root"
    service._auth = AuthService(service._cfg, None)
    service._get_outbox_store = lambda _fs: store
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    flask = make_server("127.0.0.1", 0, app_module.app, threaded=True)
    flask_thread = threading.Thread(target=flask.serve_forever, daemon=True)
    mcp = create_mcp_server(service=service, config=service._cfg)
    mcp_http = uvicorn.Server(uvicorn.Config(create_mcp_http_app(mcp, service),
        host="127.0.0.1", lifespan="off", log_level="error"))
    with socket.socket() as sock, requests.Session() as http:
        sock.bind(("127.0.0.1", 0))
        mcp_thread = threading.Thread(target=mcp_http.run, kwargs={"sockets": [sock]}, daemon=True)
        http.trust_env = False
        flask_thread.start()
        mcp_thread.start()
        try:
            deadline = time.monotonic() + 10
            while not mcp_http.started:
                assert mcp_thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
            yield http, f"http://127.0.0.1:{flask.server_port}", f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            mcp_http.should_exit = True
            mcp_thread.join(10)
            flask.shutdown()
            flask_thread.join(10)
            flask.server_close()
            service._extraction_retry.close(timeout=5)
            client.session.close()
            assert not mcp_thread.is_alive()
            assert not flask_thread.is_alive()


def exchange(http, method, url, **kwargs):
    response = http.request(method, url, timeout=10, allow_redirects=False, **kwargs)
    print("HTTP_EXCHANGE " + json.dumps({"method": method, "url": response.request.url,
        "request": kwargs.get("json"), "account": kwargs["headers"]["X-Account-ID"],
        "http_status": response.status_code, "response": response.text}, ensure_ascii=False))
    return response


@pytest.mark.parametrize("request_id,location", [
    pytest.param("turn-1", "path", id="ordinary"),
    pytest.param("turn/1", "path", id="slash"),
    pytest.param("turn%2F1", "path", id="percent-literal"),
    pytest.param("turn-1", "query", id="ordinary-query"),
    pytest.param("turn%2F1", "query", id="percent-literal-query"),
    pytest.param("turn//1", "path", id="repeated-slash"),
    pytest.param("/turn/1", "query", id="leading-slash"),
    pytest.param("//", "query", id="slash-only"),
    pytest.param("turn?part=1&next#你好 +%2F", "query", id="reserved-query"),
    pytest.param("turn/../1", "query", id="dot-segment"),
])
def test_accepted_request_id_is_queryable_on_both_http_servers(status_servers, request_id, location):
    http, flask_url, mcp_url = status_servers
    results = []
    for account in ("receipt-route-one", "receipt-route-two"):
        headers = {"X-API-Key": "route-test-root", "X-Account-ID": account,
                   "X-User-ID": "ben", "X-Agent-ID": "main"}
        session_id = account + "-session"
        source_id = account + "-source"
        payload = {"sessionId": session_id, "agentId": "main", "clientRequestId": request_id, "wait": True,
                   "messages": [{"id": source_id, "role": "user", "content": "I repaired the receiver."}]}
        accepted = exchange(http, "POST", flask_url + "/api/v1/after_turn", headers=headers, json=payload)
        assert accepted.status_code == 202
        assert accepted.json()["request_id"] == request_id
        for base_url in (flask_url, mcp_url):
            path = f"/api/v1/sessions/{session_id}/write-requests"
            params = {}
            if location == "path":
                path += "/" + quote(request_id, safe="")
                params["clientRequestId"] = "query-shadow"
            else:
                params["clientRequestId"] = request_id
            observed = exchange(http, "GET", base_url + path, headers=headers, params=params)
            results.append((source_id, observed))
    for source_id, observed in results:
        assert observed.status_code == 200, observed.text
        receipt = observed.json()
        assert receipt["request_id"] == request_id
        assert receipt["request_status"] == "ACCEPTED"
        assert receipt["source_status"] == "BUFFERED"
        assert receipt["source_message_ids"] == [source_id]
