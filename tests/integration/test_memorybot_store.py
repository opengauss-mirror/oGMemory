from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from memorybot.maintenance import GC_OPERATION, UPLOAD_OPERATION, BotMaintenance
from memorybot.store import BotStore, new_id

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    dsn = os.environ.get("MEMORYBOT_TEST_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("MEMORYBOT_TEST_DSN selects an isolated integration database")
    storage = BotStore(dsn, dialect=os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql"))
    yield storage
    storage.close()


@pytest.fixture
def account(store):
    account_id = new_id("test")
    yield account_id
    with store.transaction(account_id) as tx:
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


def session(store, account, session_id="s", user_id="u"):
    with store.transaction(account, user_id) as tx:
        return tx.insert("bot_session", {"session_id": session_id, "title": "Integration"})


def chat(store, account, request_id="r", session_id="s", user_id="u"):
    with store.transaction(account, user_id) as tx:
        return store.create_task(
            tx,
            {
                "session_id": session_id,
                "kind": "chat",
                "request_id": request_id,
                "input": {"text": "Summarize progress"},
                "accepted_authz": {
                    "agent_id": "memorybot",
                    "memory_session_id": session_id,
                    "session_generation": 1,
                },
            },
        )


def test_request_replay_returns_original_task_and_response(store, account):
    session(store, account)
    request_hash = hashlib.sha256(b"request").hexdigest()

    def create(tx):
        task = store.create_task(
            tx,
            {
                "session_id": "s",
                "kind": "chat",
                "request_id": "request-1",
                "input": {"text": "Hello"},
                "accepted_authz": {},
            },
        )
        return 202, {"task_id": task["task_id"], "accepted_at": task["created_at"]}

    first = store.idempotent(account, "u", "POST /tasks", "request-1", request_hash, create)
    replay = store.idempotent(account, "u", "POST /tasks", "request-1", request_hash, create)
    assert first[:2] == replay[:2]
    assert replay[2] is True
    assert len(store.list(account, "bot_task")) == 1
    assert store.get(account, "bot_session", {"session_id": "s"})["next_seq"] == 2


def test_two_workers_execute_distinct_sessions_and_order_each_session(store, account):
    session(store, account, "a")
    session(store, account, "b")
    first_a = chat(store, account, "a1", "a")
    second_a = chat(store, account, "a2", "a")
    first_b = chat(store, account, "b1", "b")
    peer = BotStore(pool=store.pool, initialize=False)
    with ThreadPoolExecutor(max_workers=2) as workers:
        claimed = list(
            workers.map(
                lambda pair: pair[0].claim(pair[1], account_id=account),
                [(store, "worker-a"), (peer, "worker-b")],
            )
        )
    assert {task["task_id"] for task in claimed} == {first_a["task_id"], first_b["task_id"]}
    for task in claimed:
        store.finish(
            account,
            task["task_id"],
            task["owner_id"],
            task["attempt"],
            {"text": "Done", "citations": [], "file_ids": []},
        )
    assert store.claim("worker-a", account_id=account)["task_id"] == second_a["task_id"]


def test_restart_reuses_completed_step_and_stable_operation(store, account):
    session(store, account)
    chat(store, account)
    original = store.claim("first-process", account_id=account)
    step = store.checkpoint(
        account,
        original["task_id"],
        "first-process",
        original["attempt"],
        1,
        "retrieve",
        {"query": "progress"},
        state="succeeded",
        output={"sources": []},
    )
    with store.transaction(account) as tx:
        tx.update(
            "bot_task",
            {"task_id": original["task_id"]},
            {"lease_until": tx.now() - timedelta(seconds=1)},
        )
    restarted = BotStore(pool=store.pool, initialize=False)
    recovered = restarted.claim("new-process", account_id=account)
    assert recovered["attempt"] == original["attempt"] + 1
    saved = restarted.checkpoint(
        account,
        recovered["task_id"],
        "new-process",
        recovered["attempt"],
        1,
        "retrieve",
        step["input"],
    )
    assert saved["operation_id"] == step["operation_id"]
    assert saved["output"] == {"sources": []}
    assert (
        restarted.finish(account, recovered["task_id"], "new-process", recovered["attempt"])[
            "state"
        ]
        == "succeeded"
    )
    events = restarted.events(account, recovered["task_id"])
    assert [event["event_seq"] for event in events] == list(range(1, len(events) + 1))
    assert any(event["type"] == "answer.reset" for event in events)


def test_completion_persists_child_delivery_file_and_memory_state_together(store, account):
    session(store, account)
    chat(store, account)
    task = store.claim("worker", account_id=account)
    child_id, delivery_id, file_id = new_id(), new_id("d"), new_id("f")
    result = {"text": "Report", "citations": [], "file_ids": [file_id]}

    def mutate(tx):
        store.set_memory_status(tx, task["task_id"], "pending")

    finished = store.finish(
        account,
        task["task_id"],
        "worker",
        task["attempt"],
        result,
        mutate=mutate,
        children=[
            {
                "task_id": child_id,
                "kind": "delivery",
                "request_id": "delivery-1",
                "input": {"delivery_id": delivery_id},
            }
        ],
        deliveries=[
            {
                "delivery_id": delivery_id,
                "task_id": child_id,
                "source_task_id": task["task_id"],
                "channel": "web",
                "target_id": "s",
                "operation_id": new_id("op"),
                "content": result,
            }
        ],
        files=[
            {
                "file_id": file_id,
                "purpose": "artifact",
                "source_kind": "generated",
                "task_id": task["task_id"],
                "name": "report.md",
                "mime_type": "text/markdown",
                "size_bytes": 6,
                "storage_key": f"artifacts/{file_id}",
                "sha256": hashlib.sha256(b"Report").hexdigest(),
                "state": "ready",
            }
        ],
    )
    assert finished["memory_status"] == "pending"
    assert store.get_task(account, child_id)["parent_task_id"] == task["task_id"]
    assert (
        store.get(account, "bot_delivery", {"delivery_id": delivery_id})["source_task_id"]
        == task["task_id"]
    )
    assert store.get(account, "bot_file", {"file_id": file_id})["state"] == "ready"


def test_due_schedule_coalesces_and_advances_atomically(store, account):
    session(store, account)
    with store.transaction(account, "u") as tx:
        now = tx.now()
        created = now - timedelta(minutes=11)
        tx.insert(
            "bot_schedule",
            {
                "schedule_id": "schedule",
                "session_id": "s",
                "name": "Report",
                "task_input": {"text": "Weekly report"},
                "accepted_authz": {},
                "rule": {"type": "interval", "interval_seconds": 120},
                "timezone": "Asia/Shanghai",
                "created_at": created,
                "next_run_at": created + timedelta(seconds=120),
            },
        )
    emitted = [task for task in store.trigger_due(now) if task["account_id"] == account]
    assert len(emitted) == 1
    assert emitted[0]["scheduled_at"] == created + timedelta(minutes=10)
    schedule = store.get(account, "bot_schedule", {"schedule_id": "schedule"})
    assert schedule["next_run_at"] == created + timedelta(minutes=12)
    assert emitted[0]["schedule_snapshot"]["rule"] == schedule["rule"]


def test_schedule_admission_fills_configured_user_capacity(store, account):
    session(store, account)
    with store.transaction(account, "u") as tx:
        now = tx.now()
        for index in range(2):
            tx.insert(
                "bot_schedule",
                {
                    "schedule_id": f"plan-{index}",
                    "session_id": "s",
                    "name": "Review",
                    "task_input": {"text": "Report"},
                    "accepted_authz": {},
                    "rule": {"type": "interval", "interval_seconds": 120},
                    "timezone": "UTC",
                    "next_run_at": now - timedelta(seconds=1),
                },
            )
    emitted = [
        task for task in store.trigger_due(max_user_pending=1) if task["account_id"] == account
    ]
    assert len(emitted) == 1
    assert len(store.list(account, "bot_task", {"state": "queued"})) == 1


def test_worker_initialization_and_fenced_progress_complete_concurrently(store, account):
    session(store, account)
    chat(store, account)
    task = store.claim("worker", account_id=account)
    executing, initializing = threading.Event(), threading.Event()

    def progress():
        with store.execution(account, task["task_id"], "worker", task["attempt"]) as tx:
            executing.set()
            initializing.wait(5)
            return store.append_event(
                tx,
                task["task_id"],
                "progress",
                {"phase": "context", "completed": 1, "total": 1, "message": "Ready"},
            )

    def initialize():
        initializing.set()
        peer = BotStore(pool=store.pool)
        return peer.get_task(account, task["task_id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(progress)
        executing.wait(5)
        second = executor.submit(initialize)
        assert first.result(timeout=15)["type"] == "progress"
        assert second.result(timeout=15)["task_id"] == task["task_id"]


def test_retry_and_renewal_continue_same_task(store, account):
    session(store, account)
    chat(store, account)
    task = store.claim("worker", account_id=account)
    renewed = store.renew(account, task["task_id"], "worker", task["attempt"], 60)
    assert renewed["lease_until"] > task["lease_until"]
    waiting = store.retry(
        account,
        task["task_id"],
        "worker",
        task["attempt"],
        {"code": "upstream_pending", "message": "Retry scheduled", "details": {}},
        delay_seconds=0,
    )
    assert waiting["retry_count"] == 1
    resumed = store.claim("worker", account_id=account)
    assert resumed["task_id"] == task["task_id"]
    assert resumed["attempt"] == task["attempt"] + 1


def test_user_scoped_pagination_returns_owned_rows(store, account):
    session(store, account, "one", "u1")
    session(store, account, "two", "u1")
    session(store, account, "three", "u2")
    assert [
        row["session_id"]
        for row in store.list(
            account, "bot_session", user_id="u1", order_by="session_id ASC", limit=1, offset=1
        )
    ] == ["two"]


def test_concurrent_workers_fill_tenant_execution_capacity(store, account):
    for index in range(4):
        session_id = f"parallel-{index}"
        session(store, account, session_id)
        chat(store, account, f"request-{index}", session_id)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(
                lambda index: store.claim(f"worker-{index}", account_id=account, tenant_limit=2),
                range(4),
            )
        )
    active = [task for task in results if task]
    assert len(active) == 2
    assert len({task["task_id"] for task in active}) == 2
    assert len(store.list(account, "bot_task", {"state": "running"})) == 2


def test_retired_unreferenced_file_returns_object_for_shared_storage_gc(store, account):
    session(store, account)
    task = chat(store, account)
    claimed = store.claim("worker", account_id=account)
    store.finish(account, task["task_id"], "worker", claimed["attempt"])
    with store.transaction(account, "u") as tx:
        now = tx.now()
        tx.insert(
            "bot_file",
            {
                "file_id": "retired",
                "purpose": "artifact",
                "source_kind": "generated",
                "task_id": task["task_id"],
                "name": "retired.md",
                "storage_key": "retired.md",
                "created_at": now - timedelta(days=91),
                "updated_at": now - timedelta(days=91),
            },
        )
    removed = store.delete_unreferenced_file(
        account,
        "retired",
        before=now - timedelta(days=90),
        has_memory_reference=lambda tx, file: False,
    )
    assert removed["storage_key"] == "retired.md"


def test_user_memory_writes_resume_in_original_order(store, account):
    with store.transaction(account, "u") as tx:
        first = store.create_task(
            tx,
            {
                "kind": "memory_write",
                "request_id": "write-1",
                "input": {"action": "remember", "new_text": "First"},
                "accepted_authz": {},
            },
        )
        second = store.create_task(
            tx,
            {
                "kind": "memory_write",
                "request_id": "write-2",
                "input": {"action": "remember", "new_text": "Second"},
                "accepted_authz": {},
            },
        )
    active = store.claim("worker-a", kinds=["memory_write"], account_id=account)
    assert active["task_id"] == first["task_id"]
    store.retry(
        account,
        active["task_id"],
        active["owner_id"],
        active["attempt"],
        {"code": "pending", "message": "Continue", "details": {}},
        delay_seconds=0,
    )
    resumed = store.claim("worker-b", kinds=["memory_write"], account_id=account)
    assert resumed["task_id"] == first["task_id"]
    store.finish(account, resumed["task_id"], resumed["owner_id"], resumed["attempt"])
    assert (
        store.claim("worker-b", kinds=["memory_write"], account_id=account)["task_id"]
        == second["task_id"]
    )


def test_maintenance_reclaims_expired_objects_and_preserves_references(store, account):
    session(store, account)
    chat(store, account)
    task = store.claim("worker", account_id=account)
    store.finish(account, task["task_id"], task["owner_id"], task["attempt"])
    with store.transaction(account, "u") as tx:
        old = tx.now() - timedelta(days=91)
        for file_id, metadata in (
            ("reclaim", {}),
            ("backup", {"backup_refs": ["backup-2026"]}),
            ("linked", {}),
        ):
            tx.insert(
                "bot_file",
                {
                    "file_id": file_id,
                    "purpose": "artifact",
                    "source_kind": "generated",
                    "task_id": task["task_id"],
                    "name": file_id + ".md",
                    "storage_key": file_id + ".md",
                    "metadata": metadata,
                    "created_at": old,
                    "updated_at": old,
                },
            )
        tx.update(
            "bot_task",
            {"task_id": task["task_id"]},
            {"result": {"text": "Saved", "citations": [], "file_ids": ["linked"]}},
        )
        tx.update("bot_event", {"task_id": task["task_id"]}, {"created_at": old})
        tx.insert(
            "bot_request",
            {
                "operation": "POST /completed",
                "request_id": "old",
                "request_hash": "0" * 64,
                "response_status": 200,
                "response_body": {"data": {"accepted": True}},
                "expires_at": old,
            },
        )
        tx.insert(
            "bot_request",
            {
                "operation": "POST /files",
                "request_id": "protected",
                "request_hash": "1" * 64,
                "response_status": 200,
                "response_body": {"data": {"file_id": "backup"}},
                "expires_at": old,
            },
        )
    deleted = []

    class Files:
        def delete(self, account_id, storage_key):
            deleted.append((account_id, storage_key))

    class Gateway:
        def has_file_reference(self, tx, file):
            return False

    outcome = BotMaintenance(store, Files(), Gateway(), interval_seconds=0).run_once()
    assert {key: outcome[key] for key in ("events", "requests", "files", "objects")} == {
        "events": 3,
        "requests": 1,
        "files": 1,
        "objects": 1,
    }
    assert deleted == [(account, "reclaim.md")]
    assert store.get(account, "bot_file", {"file_id": "backup"})["metadata"]["backup_refs"] == [
        "backup-2026"
    ]
    assert store.get(account, "bot_file", {"file_id": "linked"})["storage_key"] == "linked.md"
    assert (
        store.get(
            account, "bot_request", {"operation": GC_OPERATION, "request_id": "reclaim"}, "u"
        )["response_body"]["state"]
        == "deleted"
    )


def test_maintenance_resumes_durable_file_deletion_after_restart(store, account):
    with store.transaction(account, "u") as tx:
        tx.insert(
            "bot_request",
            {
                "operation": GC_OPERATION,
                "request_id": "gc-restart",
                "request_hash": "2" * 64,
                "response_status": 200,
                "response_body": {
                    "state": "pending",
                    "file": {"file_id": "gc-restart", "storage_key": "gc-restart.md"},
                },
            },
        )
    deleted = []

    class Files:
        def delete(self, account_id, storage_key):
            deleted.append((account_id, storage_key))

    outcome = BotMaintenance(store, Files(), None).run_once()
    assert outcome["objects"] == 1
    assert deleted == [(account, "gc-restart.md")]


class RetentionGateway:
    def has_file_reference(self, tx, file):
        return False

    def has_task_reference(self, tx, task):
        return False


class RetentionFiles:
    def __init__(self, objects=()):
        self.objects, self.deleted = list(objects), []

    def list_objects(self, limit=32, cursor=None):
        return {"items": self.objects, "next_cursor": None}

    def delete(self, account_id, storage_key):
        self.deleted.append((account_id, storage_key))


def test_retention_reclaims_completed_task_dependencies_and_empty_containers(store, account):
    session(store, account)
    chat(store, account)
    parent = store.claim("worker", account_id=account)
    child_id, delivery_id = new_id(), new_id("d")
    store.checkpoint(
        account,
        parent["task_id"],
        "worker",
        parent["attempt"],
        1,
        "retrieve",
        {},
        state="succeeded",
        output={},
    )
    result = {"text": "Report", "citations": [], "file_ids": ["artifact"]}
    store.finish(
        account,
        parent["task_id"],
        "worker",
        parent["attempt"],
        result,
        files=[
            {
                "file_id": "artifact",
                "purpose": "artifact",
                "source_kind": "generated",
                "task_id": parent["task_id"],
                "name": "report.md",
                "storage_key": "report.md",
                "metadata": {},
            }
        ],
        children=[
            {
                "task_id": child_id,
                "kind": "delivery",
                "request_id": "delivery",
                "input": {"delivery_id": delivery_id},
            }
        ],
        deliveries=[
            {
                "delivery_id": delivery_id,
                "task_id": child_id,
                "source_task_id": parent["task_id"],
                "target_id": "s",
                "operation_id": new_id("op"),
                "content": result,
            }
        ],
    )
    child = store.claim("worker", account_id=account, kinds=["delivery"])
    store.finish(
        account,
        child_id,
        "worker",
        child["attempt"],
        result,
        mutate=lambda tx: tx.update(
            "bot_delivery",
            {"delivery_id": delivery_id},
            {"state": "sent", "provider_receipt": {"message_id": delivery_id}},
        ),
    )
    with store.transaction(account, "u") as tx:
        old = tx.now() - timedelta(days=91)
        tx.insert(
            "bot_schedule",
            {
                "schedule_id": "retired-plan",
                "session_id": "s",
                "name": "Retired",
                "task_input": {"text": "Review"},
                "accepted_authz": {},
                "rule": {"type": "once", "at": old.isoformat()},
                "timezone": "UTC",
                "state": "cancelled",
                "updated_at": old,
            },
        )
        for table in ("bot_task", "bot_step", "bot_file", "bot_delivery", "bot_session"):
            changes = {"created_at": old, "updated_at": old}
            if table == "bot_task":
                changes["finished_at"] = old
            tx.update(table, {}, changes)
        tx.update("bot_event", {}, {"created_at": old})
    files = RetentionFiles()
    maintenance = BotMaintenance(store, files, RetentionGateway(), interval_seconds=0)
    totals = {"tasks": 0, "sessions": 0, "schedules": 0, "deliveries": 0, "steps": 0}
    for _ in range(3):
        outcome = maintenance.run_once(force=True)
        for key in totals:
            totals[key] += outcome[key]
    assert totals == {
        "tasks": 2,
        "sessions": 1,
        "schedules": 1,
        "deliveries": 1,
        "steps": 1,
    }
    assert files.deleted == [(account, "report.md")]


def test_retention_keeps_effective_forgetting_and_its_task(store, account):
    with store.transaction(account, "u") as tx:
        task = store.create_task(
            tx,
            {
                "kind": "memory_write",
                "request_id": "forget",
                "input": {"action": "forget"},
                "accepted_authz": {},
            },
        )
    claimed = store.claim("worker", account_id=account)
    store.finish(account, task["task_id"], "worker", claimed["attempt"])
    with store.transaction(account, "u") as tx:
        old = tx.now() - timedelta(days=91)
        tx.update("bot_task", {"task_id": task["task_id"]}, {"updated_at": old, "finished_at": old})
        tx.insert(
            "bot_forget",
            {
                "forget_id": "effective",
                "task_id": task["task_id"],
                "memory_uri": "ctx://example/item",
                "state": "effective",
                "operation_id": new_id("op"),
                "updated_at": old,
            },
        )
    BotMaintenance(store, RetentionFiles(), RetentionGateway()).run_once()
    assert store.get(account, "bot_forget", {"forget_id": "effective"})["state"] == "effective"
    assert store.get_task(account, task["task_id"])["state"] == "succeeded"


def test_retention_reclaims_owned_unregistered_artifact_after_one_day(store, account):
    session(store, account)
    task = chat(store, account)
    claimed = store.claim("worker", account_id=account)
    text = "Recovered artifact"
    file_id = "orphan-artifact"
    key = f"files/{file_id}/{hashlib.sha256(text.encode()).hexdigest()}/content"
    store.checkpoint(
        account,
        task["task_id"],
        "worker",
        claimed["attempt"],
        1,
        "file_publish",
        {"file_id": file_id, "text": text, "name": "artifact.md"},
    )
    store.finish(account, task["task_id"], "worker", claimed["attempt"], state="failed")
    with store.transaction(account, "u") as tx:
        old = tx.now() - timedelta(days=2)
        tx.update("bot_task", {"task_id": task["task_id"]}, {"finished_at": old})
    files = RetentionFiles(
        [
            {
                "account_id": account,
                "file_id": file_id,
                "storage_key": key,
                "modified_at": old,
                "backup_protected": False,
            }
        ]
    )
    outcome = BotMaintenance(store, files, RetentionGateway()).run_once()
    assert outcome["objects"] == 1
    assert files.deleted == [(account, key)]


def test_expired_upload_intent_becomes_a_durable_file_reclamation(store, account):
    key = "files/upload-orphan/" + "a" * 64 + "/content"
    with store.transaction(account, "u") as tx:
        tx.insert(
            "bot_request",
            {
                "operation": UPLOAD_OPERATION,
                "request_id": "upload-orphan",
                "request_hash": "a" * 64,
                "response_status": 200,
                "response_body": {
                    "state": "pending",
                    "file": {"file_id": "upload-orphan", "storage_key": key, "user_id": "u"},
                },
                "expires_at": tx.now() - timedelta(seconds=1),
            },
        )
    files = RetentionFiles()
    outcome = BotMaintenance(store, files, RetentionGateway()).run_once()
    assert outcome["requests"] == 1
    assert outcome["objects"] == 1
    assert files.deleted == [(account, key)]
    assert (
        store.get(
            account, "bot_request", {"operation": GC_OPERATION, "request_id": "upload-orphan"}, "u"
        )["response_body"]["state"]
        == "deleted"
    )


def test_retention_cursor_advances_past_sessions_used_by_active_plans(store, account):
    with store.transaction(account, "u") as tx:
        now = tx.now()
        for session_id, days in (("protected-session", 92), ("retired-session", 91)):
            old = now - timedelta(days=days)
            tx.insert(
                "bot_session",
                {
                    "session_id": session_id,
                    "title": session_id,
                    "created_at": old,
                    "updated_at": old,
                },
            )
        tx.insert(
            "bot_schedule",
            {
                "schedule_id": "active-plan",
                "session_id": "protected-session",
                "name": "Ongoing",
                "task_input": {"text": "Review"},
                "accepted_authz": {},
                "rule": {"type": "interval", "interval_seconds": 120},
                "timezone": "UTC",
                "next_run_at": now + timedelta(days=1),
            },
        )
    maintenance = BotMaintenance(store, RetentionFiles(), RetentionGateway(), batch_size=1)
    first = maintenance.run_once(force=True)
    second = maintenance.run_once(force=True)
    assert first["sessions"] + second["sessions"] == 1
    assert (
        store.get(account, "bot_session", {"session_id": "protected-session"})["title"]
        == "protected-session"
    )
    assert store.get(account, "bot_schedule", {"schedule_id": "active-plan"})["state"] == "active"
