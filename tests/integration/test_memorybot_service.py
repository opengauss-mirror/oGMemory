from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.models import RequestContext, Role
from memorybot.config import BotConfig
from memorybot.models import ChatInput, MemoryAction, ScheduleCreate, SessionCreate
from memorybot.service import BotService
from memorybot.store import BotStore, new_id
from server.auth import ResolvedIdentity

pytestmark = pytest.mark.integration


class IdentityProvider:
    def __init__(self, account_id, user_id):
        self.account_id, self.user_id = account_id, user_id

    def get_auth_service(self):
        return SimpleNamespace(
            role_control_active=lambda: True,
            resolve_identity=lambda headers: ResolvedIdentity(
                role=Role.MEMBER, account_id=self.account_id, user_id=self.user_id
            ),
        )

    def get_key_manager(self):
        return SimpleNamespace(
            get_users=lambda account: [
                {"user_id": self.user_id, "key": "integration-user-key", "status": "active"}
            ]
        )

    def build_context(self, payload, *, identity):
        return RequestContext(
            account_id=identity.account_id,
            user_id=identity.user_id,
            agent_id=payload.get("agentId") or "memorybot",
            session_id=payload.get("sessionId") or "memorybot",
            trace_id=new_id("trace"),
            role=identity.role,
        )


class MemoryGateway:
    def read_memory(self, ctx, uri):
        return {
            "uri": uri,
            "content": "Current progress",
            "source_refs": [{"uri": uri, "quote": "Original source"}],
        }


@pytest.fixture
def service():
    dsn = os.environ.get("MEMORYBOT_TEST_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("MEMORYBOT_TEST_DSN selects an isolated integration database")
    store = BotStore(dsn, dialect=os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql"))
    account = new_id("service")
    config = BotConfig()
    service = BotService(
        store,
        IdentityProvider(account, "user"),
        None,
        config,
        gateway=MemoryGateway(),
    )
    ctx = service.authenticate({"X-API-Key": "integration-user-key"})
    yield service, ctx
    with store.transaction(account) as tx:
        for table in (
            "bot_request",
            "bot_delivery",
            "bot_forget",
            "bot_file",
            "bot_event",
            "bot_step",
            "bot_task",
            "bot_schedule",
            "bot_session",
        ):
            tx.delete(table, {})
    store.close()


def create_session(service, ctx, request_id="session-1"):
    data = SessionCreate(title="Project review").model_dump(mode="json")
    status, body, duplicate = service.mutation(
        ctx,
        "POST /api/v1/bot/sessions",
        request_id,
        data,
        lambda tx: service.create_session(ctx, tx, data),
    )
    assert status == 201
    return body["data"]


def create_chat(service, ctx, session_id, request_id="chat-1"):
    data = ChatInput(text="Summarize the project").model_dump(mode="json")
    return service.mutation(
        ctx,
        f"POST /api/v1/bot/sessions/{session_id}/tasks",
        request_id,
        data,
        lambda tx: service.create_chat(ctx, tx, session_id, data, request_id),
    )


def test_session_chat_replay_and_persistent_history(service):
    app, ctx = service
    session = create_session(app, ctx)
    first = create_chat(app, ctx, session["session_id"])
    replay = create_chat(app, ctx, session["session_id"])
    assert first[:2] == replay[:2]
    assert replay[2] is True
    task = app.get_object(ctx, "task", {"task_id": first[1]["data"]["task_id"]})
    assert task["state"] == "queued"
    assert task["input"]["text"] == "Summarize the project"
    assert task["input"]["delivery"] == {"channel": "web", "target_id": session["session_id"]}
    history = app.page(
        ctx, "task", {"session_id": session["session_id"], "kind": "chat"}, history=True
    )
    assert [row["seq"] for row in history["items"]] == [1]
    original = app.store.get_task(ctx.account_id, task["task_id"])
    recovered = app.resolve_task_context(original)
    assert (recovered.account_id, recovered.user_id, recovered.session_id) == (
        ctx.account_id,
        ctx.user_id,
        session["session_id"],
    )


def test_session_seek_pagination(service):
    app, ctx = service
    sessions = [create_session(app, ctx, f"session-{index}") for index in range(3)]
    first = app.page(ctx, "session", {}, limit=2)
    second = app.page(ctx, "session", {}, limit=2, cursor=first["next_cursor"])
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert {row["session_id"] for row in first["items"] + second["items"]} == {
        row["session_id"] for row in sessions
    }


def test_uploaded_source_metadata_and_ingest_task_are_accepted_together(service):
    app, ctx = service
    scope = {"memory_scope": "user", "resource_ids": []}
    upload = {
        "file_id": new_id("f"),
        "name": "notes.md",
        "mime_type": "text/markdown",
        "size_bytes": 5,
        "storage_key": "sources/" + new_id("object"),
        "sha256": hashlib.sha256(b"notes").hexdigest(),
    }
    status, body, duplicate = app.mutation(
        ctx,
        "POST /api/v1/bot/files",
        "upload-1",
        {"scope": scope, "sha256": upload["sha256"]},
        lambda tx: app.create_file(ctx, tx, {"scope": scope}, "upload-1", upload=upload),
    )
    assert status == 202
    result = body["data"]
    assert result["file"]["file_id"] == upload["file_id"]
    assert result["file"]["state"] == "processing"
    assert result["task"]["kind"] == "ingest"
    task = app.store.get_task(ctx.account_id, result["task"]["task_id"])
    assert task["input"]["file_id"] == upload["file_id"]
    assert task["accepted_authz"]["memory_session_id"] == task["task_id"]


def test_schedule_create_pause_resume_and_update(service):
    app, ctx = service
    session = create_session(app, ctx)
    data = ScheduleCreate(
        session_id=session["session_id"],
        name="Periodic review",
        task_input=ChatInput(text="Review progress"),
        rule={"type": "interval", "interval_seconds": 120},
        timezone="Asia/Shanghai",
    ).model_dump(mode="json")
    _, body, _ = app.mutation(
        ctx,
        "POST /api/v1/bot/schedules",
        "schedule-1",
        data,
        lambda tx: app.create_schedule(ctx, tx, data),
    )
    item = body["data"]
    for action, expected_state in (("pause", "paused"), ("resume", "active"), ("update", "active")):
        change = {"action": action, "expected_revision": item["revision"]}
        if action == "update":
            change["name"] = "Updated review"
        _, body, _ = app.mutation(
            ctx,
            f"PATCH /api/v1/bot/schedules/{item['schedule_id']}",
            f"schedule-{action}",
            change,
            lambda tx: app.patch_schedule(ctx, tx, item["schedule_id"], change),
        )
        item = body["data"]
        assert item["state"] == expected_state
    assert item["revision"] == 4
    assert item["name"] == "Updated review"
    assert item["delivery"]["target_id"] == session["session_id"]


def test_queued_cancellation_preserves_user_input_and_completion_event(service):
    app, ctx = service
    session = create_session(app, ctx)
    _, body, _ = create_chat(app, ctx, session["session_id"])
    task_id = body["data"]["task_id"]
    _, body, _ = app.mutation(
        ctx,
        f"POST /api/v1/bot/tasks/{task_id}/cancel",
        "cancel-1",
        {},
        lambda tx: app.cancel_task(ctx, tx, task_id),
    )
    assert body["data"]["state"] == "cancelled"
    assert body["data"]["input"]["text"] == "Summarize the project"
    events = app.store.events(ctx.account_id, task_id)
    assert any(
        event["type"] == "task.completed" and event["payload"]["state"] == "cancelled"
        for event in events
    )


@pytest.mark.parametrize("action", ["remember", "correct", "forget"])
def test_memory_management_creates_persistent_operation(service, action):
    app, ctx = service
    values = {
        "action": action,
        "memory_uri": f"ctx://{ctx.account_id}/users/user/memories/profile/work",
        "new_text": "Project lead is Alice" if action != "forget" else None,
    }
    data = MemoryAction(**values).model_dump(mode="json")
    _, body, _ = app.mutation(
        ctx,
        "POST /api/v1/bot/memory-actions",
        "memory-1",
        data,
        lambda tx: app.memory_action(ctx, tx, data, "memory-1"),
    )
    item = body["data"]
    assert item["task"]["kind"] == "memory_write"
    assert item["task"]["memory_status"] == "pending"
    task = app.store.get_task(ctx.account_id, item["task"]["task_id"])
    assert task["input"]["action"] == action
    if action == "forget":
        record = app.store.get(ctx.account_id, "bot_forget", {"forget_id": item["forget_id"]})
        assert record["state"] == "pending"
        assert record["task_id"] == task["task_id"]


def test_fenced_intent_and_http_acceptance_commit_with_shared_capacity_order(service):
    app, ctx = service
    session = create_session(app, ctx)
    _, body, _ = create_chat(app, ctx, session["session_id"])
    task = app.store.claim("worker", account_id=ctx.account_id)
    execution_started, request_started = threading.Event(), threading.Event()

    def intent():
        with app.store.execution(ctx.account_id, task["task_id"], "worker", task["attempt"]) as tx:
            execution_started.set()
            request_started.wait(5)
            data = MemoryAction(action="remember", new_text="Project lead is Alice").model_dump(
                mode="json"
            )
            status, result = app.memory_action(ctx, tx, data, "managed-operation")
            tx.update(
                "bot_task",
                {"task_id": result["task"]["task_id"]},
                {"parent_task_id": task["task_id"]},
            )
            return status, result

    def accept():
        request_started.set()
        return create_chat(app, ctx, session["session_id"], "chat-2")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(intent)
        execution_started.wait(5)
        second = executor.submit(accept)
        status, result = first.result(timeout=15)
        assert status == 202
        assert second.result(timeout=15)[1]["data"]["seq"] == 2
    assert (
        app.store.get_task(ctx.account_id, result["task"]["task_id"])["parent_task_id"]
        == body["data"]["task_id"]
    )
