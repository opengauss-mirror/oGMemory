import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from http.client import HTTPConnection
from io import BytesIO
from threading import Event, Thread

import pytest
from flask import Flask, request
from werkzeug.serving import make_server

from memorybot.config import BotConfig
from memorybot.http import create_blueprint
from memorybot.runtime import BotWorker
from memorybot.worker import create_service
from server.memory_viewer import create_memory_viewer_blueprint
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


def receive_live_reply(app, worker, llm, task_id, auth):
    reply = "青禾联调进展已整理。"
    delta_received = Event()

    def stream_reply(_messages, emit):
        emit(reply)
        assert delta_received.wait(12), "Client did not receive the reply delta"
        return [], reply

    llm.stream_hook = stream_reply
    with make_server("127.0.0.1", 0, app, threaded=True) as server:
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                running = executor.submit(worker.run_once)
                try:
                    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=8)
                    with closing(connection) as client:
                        client.request(
                            "GET", f"/api/v1/bot/tasks/{task_id}/events",
                            headers={**auth, "Accept": "text/event-stream"},
                        )
                        with client.getresponse() as response:
                            assert response.status == 200
                            assert response.headers.get_content_type() == "text/event-stream"
                            for line in response:
                                if line.rstrip(b"\r\n") == b"event: answer.delta":
                                    data = response.readline()
                                    assert data.startswith(b"data: ")
                                    assert response.readline() in (b"\n", b"\r\n"), (
                                        "The reply delta must be a complete SSE frame"
                                    )
                                    delta = json.loads(data.removeprefix(b"data: "))
                                    break
                            else:
                                pytest.fail("Event stream closed without a reply delta")
                    assert delta["task_id"] == task_id
                    assert delta["payload"]["text"] == reply
                    assert not running.done(), "The worker finished before the delta was received"
                finally:
                    delta_received.set()
                assert running.result(timeout=12)
        finally:
            server.shutdown()
            server_thread.join(timeout=5)
            llm.stream_hook = None
    return reply


def test_registered_user_http_worker_and_shared_file(runtime):
    memory, ctx = runtime.memory, runtime.ctx
    key = runtime.api_key
    memory._cfg.role_control_enabled = True
    memory._cfg.root_api_key = "integration-root-" + ctx.account_id
    config = BotConfig(
        enabled=True,
        connection_string=memory._cfg.sql_connection_string,
        files_base_url=runtime.worker.config.files_base_url,
    )
    service = create_service(memory, config)
    service.store.claim = partial(service.store.claim, account_id=ctx.account_id)
    worker = BotWorker(service.store, service.gateway, service.files, config)
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(create_blueprint(lambda: service))
    app.register_blueprint(
        create_memory_viewer_blueprint(
            lambda: memory,
            lambda params: memory.build_context(
                params,
                identity=memory.get_auth_service().resolve_identity(dict(request.headers)),
            ),
            lambda exc: ({"error": str(exc)}, 500),
        )
    )
    http = app.test_client()
    auth = {"Authorization": "Bearer " + key}

    def complete(task_id):
        for _ in range(20):
            worker.run_once()
            row = service.store.get_task(ctx.account_id, task_id)
            if row["state"] == "succeeded":
                return row
        pytest.fail("Task did not complete through the configured worker")

    try:
        assert http.get("/bot").status_code == 200
        assert http.get("/memory-viewer/").status_code == 200
        embedded = http.get("/bot?embedded=1")
        assert embedded.status_code == 200
        assert "frame-ancestors 'self'" in embedded.headers["Content-Security-Policy"]
        viewer_auth = {"X-API-Key": key}
        scope = http.get("/api/v1/memory-viewer/scope", headers=viewer_auth)
        assert scope.status_code == 200
        assert (scope.json["account_id"], scope.json["user_id"]) == (ctx.account_id, ctx.user_id)
        capabilities = http.get("/api/v1/bot/config", headers=viewer_auth)
        assert capabilities.status_code == 200
        templates = http.get("/api/v1/bot/templates", headers=auth)
        assert templates.status_code == 200
        assert "weekly_report" in {row["id"] for row in templates.json["data"]["items"]}
        created = http.post(
            "/api/v1/bot/sessions",
            headers={**auth, "Idempotency-Key": "session"},
            json={"title": "青禾联调"},
        )
        assert created.status_code == 201
        session_id = created.json["data"]["session_id"]
        submitted = http.post(
            f"/api/v1/bot/sessions/{session_id}/tasks",
            headers={**auth, "Idempotency-Key": "chat"},
            json={"text": "整理青禾进展"},
        )
        assert submitted.status_code == 202
        task_id = submitted.json["data"]["task_id"]
        reply = receive_live_reply(app, worker, runtime.llm, task_id, auth)
        completed = http.get(f"/api/v1/bot/tasks/{task_id}", headers=auth)
        assert completed.status_code == 200
        assert completed.json["data"]["state"] == "succeeded"
        assert completed.json["data"]["result"]["text"] == reply
        content = "项目,金额\n青禾,12.30\n".encode()
        uploaded = http.post(
            "/api/v1/bot/files",
            headers={**auth, "Idempotency-Key": "upload"},
            data={"file": (BytesIO(content), "qinghe.csv")},
        )
        assert uploaded.status_code == 202
        file = uploaded.json["data"]["file"]
        complete(uploaded.json["data"]["task"]["task_id"])
        downloaded = http.get(f"/api/v1/bot/files/{file['file_id']}/content", headers=auth)
        assert downloaded.status_code == 200
        assert downloaded.data == content
        detail = http.get(f"/api/v1/bot/files/{file['file_id']}", headers=auth)
        assert detail.json["data"]["state"] == "ready"
    finally:
        service.store.close()
