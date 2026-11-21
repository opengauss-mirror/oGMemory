from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from memorybot.config import BotConfig
from memorybot.service import BotService
from memorybot.store import BotStore, new_id
from tests.integration.test_memorybot_backup_transactions import database as database


@pytest.mark.integration
@pytest.mark.parametrize(
    "waiting_confirmation", [False, True], ids=["running", "awaiting-confirmation"]
)
def test_cancel_parent_stops_a_claimed_memory_child(database, waiting_confirmation):
    ctx = SimpleNamespace(account_id=new_id("cancel"), user_id="operator")
    store = BotStore(**database)
    service = BotService(store, None, None, BotConfig(), gateway=object())
    claimed, release = Event(), Event()
    try:
        with store.transaction(ctx.account_id, ctx.user_id) as tx:
            tx.insert("bot_session", {"session_id": "s_cancel", "title": "Cancel child"})
            parent = store.create_task(
                tx,
                {"kind": "chat", "session_id": "s_cancel", "request_id": "parent",
                 "input": {"text": "forget outdated preference"}, "accepted_authz": {},
                 "memory_status": "pending"},
            )
            memory_uri = "viking://user/memories/preferences/obsolete"
            child = store.create_task(
                tx,
                {"kind": "memory_write", "parent_task_id": parent["task_id"], "request_id": "child",
                 "input": {"action": "forget", "memory_uri": memory_uri},
                 "memory_status": "pending"},
            )
            tx.insert(
                "bot_forget",
                {"forget_id": "forget-cancel", "task_id": child["task_id"],
                 "memory_uri": memory_uri, "operation_id": "forget-operation"},
            )
        store.claim("chat-worker", account_id=ctx.account_id, kinds=["chat"])

        def worker():
            task = store.claim("memory-worker", account_id=ctx.account_id, kinds=["memory_write"])
            assert task["task_id"] == child["task_id"]
            identity = (ctx.account_id, task["task_id"], task["owner_id"], task["attempt"])
            if waiting_confirmation:
                store.wait_confirmation(*identity, {"code": "OUTCOME_UNKNOWN"})
            claimed.set()
            assert release.wait(timeout=10)
            if waiting_confirmation:
                store.claim("memory-worker", account_id=ctx.account_id, kinds=["memory_write"])
            else:
                store.finish(*identity)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker)
            try:
                assert claimed.wait(timeout=10)
                status, response, _ = service.mutation(
                    ctx,
                    "task.cancel",
                    "cancel-parent",
                    {},
                    lambda tx: service.cancel_task(ctx, tx, parent["task_id"]),
                )
                assert status == 202 and response["data"]["cancel_requested"]
                task = store.get_task(ctx.account_id, child["task_id"])
                assert task["cancel_requested"]
                assert task["state"] == ("queued" if waiting_confirmation else "running")
            finally:
                release.set()
            future.result(timeout=10)

        finished = store.get_task(ctx.account_id, child["task_id"])
        assert finished["state"] == "cancelled"
        assert finished["memory_status"] == "failed"
        assert store.get_task(ctx.account_id, parent["task_id"])["memory_status"] == "failed"
        with store.transaction(ctx.account_id, ctx.user_id) as tx:
            assert tx.get("bot_forget", {"forget_id": "forget-cancel"})["state"] == "failed"
        assert store.events(ctx.account_id, child["task_id"])[-1]["type"] == "task.completed"
    finally:
        store.close()
