import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from memorybot.files import AGFSFileStore
from memorybot.maintenance import BotMaintenance
from memorybot.store import GC_OPERATION, BotStore, new_id
from tests.integration.test_memorybot_backup_transactions import database as database


@pytest.mark.integration
def test_retention_reclaims_pending_shared_file_during_task_scan_timeout(database):
    url = os.environ.get("MEMORYBOT_TEST_AGFS_URL")
    if not url:
        pytest.skip("MEMORYBOT_TEST_AGFS_URL is not configured")
    account, operator, file_id = new_id("retention"), "operator", new_id("f")
    files = AGFSFileStore(url, os.getenv("MEMORYBOT_TEST_AGFS_MOUNT", "/local"))
    store = BotStore(**database)
    gateway = SimpleNamespace(has_task_reference=lambda *_: False)
    maintenance = BotMaintenance(store, files, gateway)
    publication = None
    try:
        publication = files.publish(account, file_id, b"retire me", name="source.md")
        with store.transaction(account, operator) as tx:
            task_id = store.create_task(tx, {
                "kind": "ingest", "request_id": "old-task", "input": {}, "accepted_authz": {},
            })["task_id"]
        task = store.claim("retention-fixture", account_id=account)
        store.finish(account, task_id, task["owner_id"], task["attempt"])
        with store.transaction(account, operator) as tx:
            tx.cursor.execute(
                "UPDATE bot_task SET updated_at=now()-interval '200 days', "
                "finished_at=now()-interval '200 days' WHERE account_id=%s AND task_id=%s",
                (account, task_id),
            )
            tx.cursor.execute(
                "UPDATE bot_event SET created_at=now()-interval '20 days' "
                "WHERE account_id=%s AND task_id=%s AND event_seq=1",
                (account, task_id),
            )
            maintenance._queue_gc(tx, {**publication, "file_id": file_id, "user_id": operator}, tx.now())

        with ThreadPoolExecutor(max_workers=1) as queue, store.transaction(account, operator) as tx:
            tx.cursor.execute("LOCK TABLE bot_step IN ACCESS EXCLUSIVE MODE")
            for _ in range(2):
                queue.submit(maintenance.run_once, force=True).result(timeout=5)
            receipt = store.get(
                account, "bot_request", {"operation": GC_OPERATION, "request_id": file_id}
            )
            assert [event["event_seq"] for event in store.events(account, task_id)] == [2, 3]
            assert receipt["response_body"]["state"] == "deleted"
    finally:
        if publication:
            files.delete(account, publication["storage_key"])
        store.close()
