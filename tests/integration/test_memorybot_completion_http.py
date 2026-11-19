from __future__ import annotations

import io
import json
import threading

import pytest
from flask import Flask
from werkzeug.serving import make_server

from cli.main import main
from memorybot.http import create_blueprint
from memorybot.service import BotService
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


@pytest.fixture
def completion(runtime):
    memory, ctx = runtime.memory, runtime.ctx
    admin = memory.get_tenant_admin_service()
    member = admin.create_user(ctx, ctx.account_id, "viewer")
    admin.create_agent(ctx, ctx.account_id, "analyst", owner_user_id="viewer")
    memory._cfg.role_control_enabled = True
    memory._cfg.root_api_key = "test-root-" + ctx.account_id
    service = BotService(runtime.store, memory, runtime.files, runtime.worker.config)
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(create_blueprint(lambda: service))
    return app, service, member["user_key"], ctx.account_id


def test_native_agent_templates_and_background_stop(completion):
    app, service, key, account = completion
    http = app.test_client()
    auth = {"Authorization": "Bearer " + key}
    agents = http.get("/api/v1/bot/agents", headers=auth)
    assert agents.status_code == 200
    assert agents.json["data"] == {
        "items": [{"agent_id": "analyst"}],
        "default_agent_id": "analyst",
    }
    all_templates, cursor = [], None
    while True:
        params = {"agent_id": "analyst", "limit": 2}
        if cursor:
            params["cursor"] = cursor
        response = http.get("/api/v1/bot/templates", query_string=params, headers=auth)
        assert response.status_code == 200
        data = response.json["data"]
        all_templates.extend(data["items"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert [row["id"] for row in all_templates] == [
        "csv_summary",
        "handbook",
        "handover",
        "release_check",
        "weekly_report",
    ]
    assert all(
        set(row) == {"id", "name", "input", "output", "permission_requirements"}
        for row in all_templates
    )
    selected = http.get(
        "/api/v1/bot/templates?agent_id=analyst&q=周报&output_format=markdown&limit=1", headers=auth
    )
    assert selected.json["data"]["items"][0]["id"] == "weekly_report"
    session = http.post(
        "/api/v1/bot/sessions",
        headers={**auth, "Idempotency-Key": "agent-session"},
        json={"title": "分析会话", "agent_id": "analyst"},
    )
    assert session.status_code == 201
    assert session.json["data"]["agent_id"] == "analyst"
    remembered = http.post(
        "/api/v1/bot/memory-actions?agent_id=analyst",
        headers={**auth, "Idempotency-Key": "remember"},
        json={"action": "remember", "new_text": "偏好简洁回复"},
    )
    assert remembered.status_code == 202
    task_id = remembered.json["data"]["task"]["task_id"]
    stored = service.store.get_task(account, task_id, user_id="viewer")
    assert service.resolve_task_context(stored).agent_id == "analyst"
    stopped = http.post(
        f"/api/v1/bot/tasks/{task_id}/cancel",
        headers={**auth, "Idempotency-Key": "stop-memory"},
        json={},
    )
    assert stopped.status_code == 202
    assert stopped.json["data"]["state"] == "cancelled"
    uploaded = http.post(
        "/api/v1/bot/files?agent_id=analyst",
        headers={**auth, "Idempotency-Key": "source"},
        data={"file": (io.BytesIO("资料接收后停止处理。".encode()), "notes.txt")},
    )
    assert uploaded.status_code == 202
    task_id = uploaded.json["data"]["task"]["task_id"]
    stopped = http.post(
        f"/api/v1/bot/tasks/{task_id}/cancel",
        headers={**auth, "Idempotency-Key": "stop-file"},
        json={},
    )
    assert stopped.json["data"]["state"] == "cancelled"


def test_cli_native_agent_session_and_template_journey(completion, capsys):
    app, _, key, _ = completion
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = ["bot", "--base-url", f"http://127.0.0.1:{server.server_port}", "--api-key", key]

    def invoke(*arguments):
        assert main([*base, *arguments]) == 0
        return json.loads(capsys.readouterr().out)["data"]

    try:
        assert invoke("agent", "list")["items"] == [{"agent_id": "analyst"}]
        session = invoke("session", "create", "初始标题", "--agent-id", "analyst")
        session_id = session["session_id"]
        assert invoke("session", "rename", session_id, "每周回顾")["title"] == "每周回顾"
        assert invoke("session", "archive", session_id)["archived"] is True
        assert (
            invoke("session", "list", "--archived", "true", "--query", "回顾")["items"][0][
                "session_id"
            ]
            == session_id
        )
        assert invoke("session", "restore", session_id)["archived"] is False
        assert (
            invoke("session", "show", session_id, "--history-order", "desc")["session"]["agent_id"]
            == "analyst"
        )
        first = invoke("template", "list", "--agent-id", "analyst", "--limit", "2")
        second = invoke(
            "template",
            "list",
            "--agent-id",
            "analyst",
            "--limit",
            "2",
            "--cursor",
            first["next_cursor"],
        )
        assert [row["id"] for row in first["items"] + second["items"]] == [
            "csv_summary",
            "handbook",
            "handover",
            "release_check",
        ]
        pending = invoke("memory", "remember", "偏好简洁回复", "--agent-id", "analyst")
        assert invoke("task", "cancel", pending["task"]["task_id"])["state"] == "cancelled"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
