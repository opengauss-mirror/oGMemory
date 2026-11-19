import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from memorybot.backup import BotFileBackup, _AGFSBackupFiles
from memorybot.store import BotStore, new_id

pytestmark = pytest.mark.integration


@pytest.fixture
def database():
    dsn = os.environ.get("MEMORYBOT_TEST_DSN")
    if not dsn:
        pytest.skip("MEMORYBOT_TEST_DSN is not configured")
    return {"connection_string": dsn, "dialect": os.getenv("MEMORYBOT_TEST_DIALECT", "postgresql")}


@pytest.mark.parametrize("share_pool", [True, False], ids=["one-pool", "two-instances"])
def test_concurrent_backups_preserve_each_others_protection(database, share_pool):
    url = os.environ.get("MEMORYBOT_TEST_AGFS_URL")
    if not url:
        pytest.skip("MEMORYBOT_TEST_AGFS_URL is not configured")
    mount = os.getenv("MEMORYBOT_TEST_AGFS_MOUNT", "/local")
    account_id, file_id = new_id("backup"), new_id("f")
    data = b"memoryBot backup integration file"
    digest = hashlib.sha256(data).hexdigest()
    storage_key = f"files/{file_id}/{digest}/content"
    files = _AGFSBackupFiles(url, mount, timeout=5)
    store = BotStore(**database, pool_size=1)
    peer = store if share_pool else BotStore(**database, pool_size=1, initialize=False)
    backup, other_backup = BotFileBackup(store, files), BotFileBackup(peer, files)
    try:
        path = files._path(account_id, storage_key)
        files._parents(path)
        files.client.write(path, data)
        with store.transaction(account_id, "operator") as tx:
            task = store.create_task(
                tx,
                {
                    "kind": "ingest",
                    "request_id": "ingest",
                    "input": {"file_id": file_id},
                    "accepted_authz": {},
                },
            )
            tx.insert(
                "bot_file",
                {
                    "file_id": file_id,
                    "task_id": task["task_id"],
                    "purpose": "source",
                    "source_kind": "upload",
                    "state": "ready",
                    "storage_key": storage_key,
                    "sha256": digest,
                    "size_bytes": len(data),
                    "name": "source.txt",
                    "mime_type": "text/plain",
                },
            )

        start = Barrier(2, timeout=5)

        def export(manager, backup_id):
            start.wait()
            return manager.export(account_id, backup_id)

        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(export, backup, "first")
            second = pool.submit(export, other_backup, "second")
            first_manifest, second_manifest = first.result(15), second.result(15)

        assert backup.verify(first_manifest)["size_bytes"] == len(data)
        assert backup.release(first_manifest)["released"]
        metadata = store.get(account_id, "bot_file", {"file_id": file_id})["metadata"]
        assert metadata["backup_refs"] == ["second"]
        assert other_backup.verify(second_manifest)["object_count"] == 1
        assert other_backup.release(second_manifest)["released"]
        metadata = store.get(account_id, "bot_file", {"file_id": file_id})["metadata"]
        assert (metadata["backup_refs"], metadata["backup_protected_until"]) == ([], None)
    finally:
        store.close()
        if peer is not store:
            peer.close()


def test_artifact_checkpoint_precedes_file_metadata(database):
    account_id, file_id = new_id("artifact"), new_id("f")
    store = BotStore(**database)
    try:
        with store.transaction(account_id, "operator") as tx:
            task = store.create_task(
                tx,
                {"kind": "ingest", "request_id": "checkpoint", "input": {}, "accepted_authz": {}},
            )
        attempt = store.claim("worker-a", account_id=account_id)
        identity = (account_id, task["task_id"], attempt["owner_id"], attempt["attempt"])
        step = store.checkpoint(
            *identity, 1, "file_publish", {"file_id": file_id, "name": "report.md"}
        )
        assert step["input"]["file_id"] == file_id
        publication = {"publication": {"file_id": file_id}}
        completed_step = store.checkpoint(
            *identity,
            1,
            "file_publish",
            {},
            state="succeeded",
            output=publication,
            external_receipt=publication,
        )
        assert completed_step["output"] == completed_step["external_receipt"] == publication
        assert completed_step["public_progress"]["state"] == "succeeded"
        finished = store.finish(
            *identity,
            {"file_ids": [file_id]},
            files=[
                {
                    "file_id": file_id,
                    "task_id": task["task_id"],
                    "purpose": "artifact",
                    "source_kind": "generated",
                    "name": "report.md",
                }
            ],
        )
        assert finished["result"]["file_ids"] == [file_id]
        assert store.get(account_id, "bot_file", {"file_id": file_id})["name"] == "report.md"
    finally:
        store.close()


def test_cancelled_confirmation_releases_next_chat(database):
    account_id = new_id("cancel")
    store = BotStore(**database)
    try:
        with store.transaction(account_id, "operator") as tx:
            session_id = "s_cancel"
            tx.insert("bot_session", {"session_id": session_id, "title": "Cancel confirmation"})
            for request_id in ("first", "following"):
                store.create_task(
                    tx,
                    {
                        "kind": "chat",
                        "session_id": session_id,
                        "request_id": request_id,
                        "input": {"text": request_id},
                        "accepted_authz": {},
                    },
                )
        first = store.claim("worker-a", account_id=account_id)
        task_id = first["task_id"]
        identity = (account_id, task_id, first["owner_id"], first["attempt"])
        store.checkpoint(*identity, 1, "tool", {"tool": "memory_action"})
        with store.transaction(account_id, "operator") as tx:
            tx.update("bot_task", {"task_id": task_id}, {"cancel_requested": True})

        result = store.wait_confirmation(*identity, {"code": "OUTCOME_UNKNOWN"})
        assert result["state"] == "cancelled"
        assert store.get(account_id, "bot_step", {"task_id": task_id, "step_no": 1})["state"] == (
            "cancelled"
        )
        assert store.events(account_id, task_id)[-1]["type"] == "task.completed"
        assert store.claim("worker-b", account_id=account_id)["request_id"] == "following"
    finally:
        store.close()
