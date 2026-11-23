import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

import pytest

from memorybot.backup import BotFileBackup
from memorybot.files import AGFSFileStore
from memorybot.store import BotStore, json_value
from tests.integration.test_memorybot_runtime import runtime as runtime
from tests.integration.test_memorybot_runtime import source

pytestmark = pytest.mark.integration


def marker(runtime, file):
    key = str(PurePosixPath(file["storage_key"]).with_name("backup.json"))
    return json.loads(runtime.files.read(runtime.ctx.account_id, key))


def release_backup(manager, account, backup_id):
    if manager._read_manifest(account, backup_id, pending=True) or manager._read_manifest(
        account, backup_id
    ):
        manager.release(account_id=account, backup_id=backup_id)


def test_export_replay_verify_restored_objects_and_release_audit(runtime):
    file = source(runtime, text="青禾负责人林悦，11月12日发布。")
    account = runtime.ctx.account_id
    manager = BotFileBackup(runtime.store, runtime.files)
    retained = datetime.now(UTC) + timedelta(days=90)
    with ExitStack() as cleanup:
        cleanup.callback(release_backup, manager, account, "release_snapshot")
        manifest = manager.export(account, "release_snapshot", retained_until=retained)
        fresh_store = BotStore(
            os.environ["MEMORYBOT_TEST_DSN"],
            dialect=os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql"),
            initialize=False,
        )
        cleanup.callback(fresh_store.close)
        restore_prefix = runtime.files.mount_prefix + "/restore_" + account
        runtime.files.client.mkdir(restore_prefix)
        restored = AGFSFileStore(os.environ["MEMORYBOT_TEST_AGFS_URL"], restore_prefix)
        cleanup.callback(restored.delete, account, file["storage_key"])
        assert manifest["retained_until"] == json_value(retained)
        assert len(manifest["files"]) == 1
        entry = manifest["files"][0]
        assert {PurePosixPath(row["storage_key"]).name for row in entry["objects"]} == {
            "content",
            "parsed.json",
        }
        saved = runtime.store.get(account, "bot_file", {"file_id": file["file_id"]})
        assert saved["metadata"]["backup_refs"] == ["release_snapshot"]
        assert marker(runtime, file)["retained_until"]["release_snapshot"] == json_value(retained)
        assert manager.export(account, "release_snapshot") == manifest
        for obj in entry["objects"]:
            path = restored._path(account, obj["storage_key"])
            restored._parents(path)
            restored.client.write(path, runtime.files.read(account, obj["storage_key"]))
        report = BotFileBackup(fresh_store, restored).verify(manifest)
        assert report["verified"] is True
        assert (report["file_count"], report["object_count"]) == (1, 2)
        assert report["size_bytes"] == sum(row["size_bytes"] for row in entry["objects"])
        assert manager.release(manifest)["released"] is True
        assert manager.release(manifest)["released"] is True
        saved = runtime.store.get(account, "bot_file", {"file_id": file["file_id"]})
        assert saved["metadata"]["backup_refs"] == []
        audit = json.loads(runtime.files.read(account, "backups/release_snapshot/manifest.json"))
        assert audit == manifest
        receipt = json.loads(runtime.files.read(account, "backups/release_snapshot/released.json"))
        assert receipt["released"] is True


def test_concurrent_backups_preserve_each_remaining_reference(runtime):
    file = source(runtime, text="青禾备份资料，负责人林悦。")
    account = runtime.ctx.account_id
    manager = BotFileBackup(runtime.store, runtime.files)
    retained = datetime.now(UTC) + timedelta(days=60)
    backup_ids = ("daily_one", "daily_two")
    with ExitStack() as cleanup:
        for backup_id in backup_ids:
            cleanup.callback(release_backup, manager, account, backup_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(manager.export, account, backup_id, retained_until=retained)
                for backup_id in backup_ids
            ]
            manifests = [job.result(timeout=30) for job in jobs]
        saved = runtime.store.get(account, "bot_file", {"file_id": file["file_id"]})
        assert saved["metadata"]["backup_refs"] == ["daily_one", "daily_two"]
        assert marker(runtime, file)["backup_refs"] == ["daily_one", "daily_two"]
        manager.release(manifests[0])
        saved = runtime.store.get(account, "bot_file", {"file_id": file["file_id"]})
        assert saved["metadata"]["backup_refs"] == ["daily_two"]
        assert marker(runtime, file)["backup_refs"] == ["daily_two"]
        assert saved["metadata"]["backup_protected_until"] == json_value(retained)
        assert manager.verify(manifests[1])["verified"] is True
        manager.release(manifests[1])
        saved = runtime.store.get(account, "bot_file", {"file_id": file["file_id"]})
        assert saved["metadata"]["backup_refs"] == []
        assert saved["metadata"]["backup_protected_until"] is None
        assert (
            runtime.files.read(account, file["storage_key"]).decode()
            == "青禾备份资料，负责人林悦。"
        )
