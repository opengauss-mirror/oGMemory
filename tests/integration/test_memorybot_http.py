from __future__ import annotations

import hashlib
import io
import json

import pytest
from flask import Flask

from memorybot.http import create_blueprint
from tests.integration.test_memorybot_service import service as service

pytestmark = pytest.mark.integration


class Files:
    def __init__(self):
        self.data = {}

    def publish(self, account, file_id, data, *, name, mime_type):
        digest = hashlib.sha256(data).hexdigest()
        key = f"files/{file_id}/{digest}/content"
        self.data[account, key] = data
        return {
            "storage_key": key,
            "sha256": digest,
            "size_bytes": len(data),
            "name": name,
            "mime_type": mime_type,
        }

    def read(self, account, key, *, offset=0, size=-1):
        value = self.data[account, key]
        return value[offset:] if size < 0 else value[offset : offset + size]


@pytest.fixture
def client(service):
    svc, ctx = service
    svc.files = Files()
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(lambda: svc))
    app.testing = True
    return app.test_client(), svc, ctx


def headers(key):
    return {"X-API-Key": "integration-user-key", "Idempotency-Key": key}


def test_http_memory_list_returns_native_category_owner_and_actions(client, monkeypatch):
    http, svc, ctx = client
    uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/qinghe"
    entry = {
        "uri": uri,
        "title": "青禾",
        "text": "青禾周四发布。",
        "source_refs": [],
        "updated_at": "2026-11-04T06:00:00Z",
        "category": "entity",
        "owner_space": ctx.user_space_name(),
    }
    scopes = []

    def list_memory_page(context, q, scope, hidden, *, limit, offset):
        scopes.append(scope["memory_scope"])
        return [entry], False

    monkeypatch.setattr(svc.gateway, "list_memory_page", list_memory_page, raising=False)
    for scope in ("user", "authorized"):
        result = http.get(f"/api/v1/bot/memories?memory_scope={scope}", headers=headers("memories"))
        assert result.status_code == 200
        assert result.json["data"]["items"] == [{**entry, "can_forget": True}]
    assert scopes == ["user", "authorized"]


def test_http_conversation_persistent_events_and_replay(client):
    http, svc, ctx = client
    created = http.post("/api/v1/bot/sessions", headers=headers("s"), json={"title": "青禾"})
    assert created.status_code == 201
    session_id = created.json["data"]["session_id"]
    path = f"/api/v1/bot/sessions/{session_id}/tasks"
    response = http.post(path, headers=headers("question"), json={"text": "汇总本周进展"})
    assert response.status_code == 202
    task_id = response.json["data"]["task_id"]
    replay = http.post(path, headers=headers("question"), json={"text": "汇总本周进展"})
    assert replay.json == response.json
    task = svc.store.claim("http-worker", account_id=ctx.account_id)
    step = svc.store.checkpoint(
        ctx.account_id,
        task_id,
        "http-worker",
        task["attempt"],
        1,
        "model",
        {"messages": [{"role": "system", "content": "依据用户资料整理报告"}]},
    )
    started_seq = step["public_progress"]["event_seq"]
    svc.store.checkpoint(
        ctx.account_id,
        task_id,
        "http-worker",
        task["attempt"],
        1,
        "model",
        {},
        state="succeeded",
        output={"text": "联调完成", "calls": []},
    )
    svc.store.finish(
        ctx.account_id,
        task_id,
        "http-worker",
        task["attempt"],
        {"text": "联调完成", "citations": [], "file_ids": []},
    )
    result = http.get(f"/api/v1/bot/tasks/{task_id}", headers=headers("read"))
    assert result.json["data"]["state"] == "succeeded"
    steps = result.json["data"]["steps"]
    assert steps[0]["label"] == "生成回复" and steps[0]["state"] == "succeeded"
    assert steps[0]["summary"] == {"character_count": 4, "tool_call_count": 0}
    stream = http.get(f"/api/v1/bot/tasks/{task_id}/events", headers=headers("events"))
    assert "event: task.completed" in stream.text
    assert f"id: {task_id}:" in stream.text
    history = http.get(f"/api/v1/bot/sessions/{session_id}", headers=headers("history"))
    assert history.json["data"]["messages"][0]["result"]["text"] == "联调完成"
    assert history.json["data"]["messages"][0]["steps"] == steps
    resumed = http.get(
        f"/api/v1/bot/tasks/{task_id}/events",
        headers={**headers("resume"), "Last-Event-ID": f"{task_id}:{started_seq}"},
    )
    events = [
        json.loads(line[6:]) for line in resumed.text.splitlines() if line.startswith("data: ")
    ]
    assert [event["event_seq"] for event in events] == list(
        range(started_seq + 1, result.json["data"]["last_event_seq"] + 1)
    )
    assert events[0]["payload"] == steps[0]
    with svc.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.delete("bot_event", {"task_id": task_id})
    retained = http.get(f"/api/v1/bot/tasks/{task_id}", headers=headers("retained"))
    assert retained.json["data"]["steps"] == steps


def test_http_upload_range_download_and_file_list(client):
    http, svc, ctx = client
    content = "青禾负责人林悦。\n".encode()
    result = http.post(
        "/api/v1/bot/files",
        headers=headers("upload"),
        data={
            "file": (io.BytesIO(content), "notes.md"),
            "scope": '{"memory_scope":"user"}',
        },
    )
    assert result.status_code == 202
    file = result.json["data"]["file"]
    assert file["sha256"] == hashlib.sha256(content).hexdigest()
    assert file["size_bytes"] == len(content)
    replay = http.post(
        "/api/v1/bot/files",
        headers=headers("upload"),
        data={
            "file": (io.BytesIO(content), "notes.md"),
            "scope": '{"memory_scope":"user"}',
        },
    )
    assert replay.json == result.json
    assert len(svc.files.data) == 1
    with svc.store.transaction(ctx.account_id, ctx.user_id) as tx:
        assert tx.list("bot_request", {"operation": "maintenance:upload_intent"}) == []
        tx.update("bot_file", {"file_id": file["file_id"]}, {"state": "ready"})
    path = f"/api/v1/bot/files/{file['file_id']}/content"
    downloaded = http.get(path, headers=headers("download"))
    assert downloaded.data == content
    assert downloaded.headers["ETag"] == '"' + file["sha256"] + '"'
    part = http.get(path, headers={**headers("range"), "Range": "bytes=0-5"})
    assert part.status_code == 206
    assert part.data == content[:6]
    assert part.headers["Content-Range"] == f"bytes 0-5/{len(content)}"
    listing = http.get("/api/v1/bot/files", headers=headers("files"))
    assert listing.json["data"]["items"][0]["file_id"] == file["file_id"]


def test_server_accepts_memorybot_upload_at_file_size_limit(service, monkeypatch):
    import server.app as server_app

    svc, _ = service
    svc.files = Files()
    monkeypatch.setattr(server_app, "_bot_service", svc)
    size = svc.config.max_file_bytes
    result = server_app.app.test_client().post(
        "/api/v1/bot/files",
        headers=headers("max-size-upload"),
        data={"file": (io.BytesIO(b"a" * size), "reference.txt")},
    )
    assert result.status_code == 202, result.json
    assert result.json["data"]["file"]["size_bytes"] == size


def test_http_schedule_template_and_web_result_flow(client):
    http, svc, ctx = client
    session = http.post(
        "/api/v1/bot/sessions", headers=headers("session"), json={"title": "周报"}
    ).json["data"]
    templates = http.get(
        "/api/v1/bot/templates?output_format=csv", headers=headers("templates")
    ).json["data"]["items"]
    assert any(row["id"] == "csv_summary" for row in templates)
    data = {
        "session_id": session["session_id"],
        "name": "周期周报",
        "task_input": {"text": "生成周报"},
        "rule": {"type": "interval", "interval_seconds": 120},
        "timezone": "Asia/Shanghai",
    }
    created = http.post("/api/v1/bot/schedules", headers=headers("schedule"), json=data)
    assert created.status_code == 201
    plan = created.json["data"]
    paused = http.patch(
        f"/api/v1/bot/schedules/{plan['schedule_id']}",
        headers=headers("pause"),
        json={"expected_revision": 1, "action": "pause"},
    )
    assert paused.json["data"]["state"] == "paused"
    resumed = http.patch(
        f"/api/v1/bot/schedules/{plan['schedule_id']}",
        headers=headers("resume"),
        json={"expected_revision": 2, "action": "resume"},
    )
    assert resumed.json["data"]["state"] == "active"
    assert resumed.json["data"]["task_input"]["delivery"]["channel"] == "web"


def test_http_completed_events_drain_all_batches_and_resume(client):
    http, svc, ctx = client
    session = http.post(
        "/api/v1/bot/sessions",
        headers=headers("event-session"),
        json={"title": "长任务进度"},
    ).json["data"]
    task_id = http.post(
        f"/api/v1/bot/sessions/{session['session_id']}/tasks",
        headers=headers("event-task"),
        json={"text": "汇总全部资料"},
    ).json["data"]["task_id"]
    task = svc.store.claim("event-worker", account_id=ctx.account_id)
    with svc.store.transaction(ctx.account_id, ctx.user_id) as tx:
        for completed in range(1, 206):
            svc.store.append_event(
                tx,
                task_id,
                "progress",
                {
                    "phase": "model",
                    "message": f"处理第 {completed} 段",
                    "completed": completed,
                    "total": 205,
                },
            )
    svc.store.finish(
        ctx.account_id,
        task_id,
        "event-worker",
        task["attempt"],
        {"text": "资料汇总完成", "citations": [], "file_ids": []},
    )
    snapshot = http.get(f"/api/v1/bot/tasks/{task_id}", headers=headers("snapshot")).json["data"]
    assert snapshot["last_event_seq"] > 200
    for after in (0, 123):
        response = http.get(
            f"/api/v1/bot/tasks/{task_id}/events?after={after}",
            headers={**headers("events"), "Last-Event-ID": f"{task_id}:{after}"},
        )
        assert response.status_code == 200
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert [event["event_seq"] for event in events] == list(
            range(after + 1, snapshot["last_event_seq"] + 1)
        )
        assert all(event["task_id"] == task_id for event in events)
        assert events[-1]["payload"]["state"] == "succeeded"
        assert "event: task.completed" in response.text
