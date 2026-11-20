"""File manifests and shared-object protection for operator-managed database backups."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from memorybot.scheduling import as_utc
from memorybot.store import json_value

_FILE_FIELDS = (
    "account_id",
    "user_id",
    "file_id",
    "purpose",
    "source_kind",
    "state",
    "storage_key",
    "sha256",
    "size_bytes",
)


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _missing(error):
    return any(text in str(error).lower() for text in ("not found", "not exist", "404"))


def _sealed(manifest):
    manifest = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return {**manifest, "manifest_sha256": hashlib.sha256(_bytes(manifest)).hexdigest()}


class _AGFSBackupFiles:
    def __init__(self, base_url, mount_prefix, *, timeout):
        from pyagfs import AGFSClient

        self.client = AGFSClient(api_base_url=base_url, timeout=timeout)
        self.mount_prefix = mount_prefix.rstrip("/")

    def _path(self, account_id, key):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", account_id) or account_id in {".", ".."}:
            raise ValueError("租户标识无效")
        if "\\" in key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ValueError("文件键无效")
        return f"{self.mount_prefix}/{account_id}/{key}"

    def _parents(self, path):
        parent = self.mount_prefix
        for part in path[len(parent) + 1 :].split("/")[:-1]:
            parent += "/" + part
            try:
                self.client.mkdir(parent)
            except Exception as exc:
                if "exist" not in str(exc).lower():
                    raise

    def read(self, account_id, key):
        return self.client.read(self._path(account_id, key))


class BotFileBackup:
    def __init__(self, store, files):
        self.store, self.files = store, files

    def _serialized(self, account_id):
        return self.store.serialized_transactions(account_id, "memorybot.backup:" + account_id)

    def _read_json(self, account_id, key):
        try:
            return json.loads(self.files.read(account_id, key))
        except Exception as exc:
            if _missing(exc):
                return None
            raise

    def _write_json(self, account_id, key, value):
        path = self.files._path(account_id, key)
        self.files._parents(path)
        data = _bytes(value)
        self.files.client.write(path, data)
        if self.files.read(account_id, key) != data:
            raise OSError("备份清单或保护标记写入校验失败")

    def _remove(self, account_id, key):
        try:
            self.files.client.rm(self.files._path(account_id, key))
        except Exception as exc:
            if not _missing(exc):
                raise

    @staticmethod
    def _marker_key(row):
        return str(PurePosixPath(row["storage_key"]).with_name("backup.json"))

    @staticmethod
    def _manifest_key(backup_id, name="manifest.json"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", backup_id):
            raise ValueError("备份标识须为1至64位字母、数字、下划线或连字符")
        return f"backups/{backup_id}/{name}"

    def _read_manifest(self, account_id, backup_id, *, pending=False):
        name = "pending.json" if pending else "manifest.json"
        manifest = self._read_json(account_id, self._manifest_key(backup_id, name))
        if manifest is not None:
            self._validate(manifest, pending=pending)
            if (manifest["account_id"], manifest["backup_id"]) != (account_id, backup_id):
                raise ValueError("备份清单租户或标识不一致")
        return manifest

    @staticmethod
    def _rows(tx):
        tx.advisory_lock("memorybot.admission")
        tx.cursor.execute(
            "SELECT * FROM bot_file WHERE account_id=%s ORDER BY file_id FOR UPDATE",
            (tx.account_id,),
        )
        return [dict(row) for row in tx.cursor.fetchall()]

    @staticmethod
    def _protect_rows(tx, rows, manifest):
        backup_id = manifest["backup_id"]
        retained_until = as_utc(manifest["retained_until"])
        for row in rows:
            metadata = dict(row["metadata"])
            metadata["backup_refs"] = sorted(set(metadata.get("backup_refs", [])) | {backup_id})
            dates = [retained_until]
            if metadata.get("backup_protected_until"):
                dates.append(as_utc(metadata["backup_protected_until"]))
            metadata["backup_protected_until"] = json_value(max(dates))
            tx.update("bot_file", {"file_id": row["file_id"]}, {"metadata": metadata})

    def _object(self, account_id, key):
        data = self.files.read(account_id, key)
        return {
            "storage_key": key,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _protect_marker(self, account_id, row, backup_id, retained_until):
        key = self._marker_key(row)
        marker = self._read_json(account_id, key) or {}
        refs = set(marker.get("backup_refs", [])) | {backup_id}
        retained = dict(marker.get("retained_until", {}))
        retained[backup_id] = retained_until
        self._write_json(account_id, key, {"backup_refs": sorted(refs), "retained_until": retained})

    def export(self, account_id, backup_id, *, retained_until=None):
        key = self._manifest_key(backup_id)
        pending_key = self._manifest_key(backup_id, "pending.json")
        self.files._path(account_id, key)
        retained_until = (
            as_utc(retained_until) if retained_until else datetime.now(UTC) + timedelta(days=90)
        )
        with self._serialized(account_id) as transaction:
            if self._read_json(account_id, self._manifest_key(backup_id, "released.json")):
                raise ValueError("已解除保护的备份请使用新的备份标识")
            completed = self._read_manifest(account_id, backup_id)
            manifest = completed or self._read_manifest(account_id, backup_id, pending=True)
            with transaction() as tx:
                rows = self._rows(tx)
                descriptors = self._descriptors(rows)
                if manifest is not None:
                    self._match_rows(manifest, descriptors)
                else:
                    manifest = _sealed(
                        {
                            "schema_version": 1,
                            "account_id": account_id,
                            "backup_id": backup_id,
                            "created_at": json_value(datetime.now(UTC)),
                            "retained_until": json_value(retained_until),
                            "files": descriptors,
                            "pending": True,
                        }
                    )
                    self._validate(manifest, pending=True)
                    # Record the release information before committing the database protection.
                    self._write_json(account_id, pending_key, manifest)
                self._protect_rows(tx, rows, manifest)
            entries = []
            for row in descriptors:
                objects = []
                if row["storage_key"]:
                    content = self._object(account_id, row["storage_key"])
                    if (content["sha256"], content["size_bytes"]) != (
                        row["sha256"],
                        row["size_bytes"],
                    ):
                        raise ValueError("共享文件内容与数据库摘要不一致：" + row["file_id"])
                    objects.append(content)
                    directory = str(
                        PurePosixPath(self.files._path(account_id, row["storage_key"])).parent
                    )
                    if any(
                        item["name"] == "parsed.json" for item in self.files.client.ls(directory)
                    ):
                        objects.append(
                            self._object(
                                account_id,
                                str(PurePosixPath(row["storage_key"]).with_name("parsed.json")),
                            )
                        )
                entries.append({**row, "objects": objects})
            if completed and entries != manifest["files"]:
                raise ValueError("备份文件集合已变化，请使用新的备份标识")
            for row in entries:
                if row["storage_key"]:
                    self._protect_marker(account_id, row, backup_id, manifest["retained_until"])
            if completed is None:
                manifest.pop("pending")
                manifest = _sealed({**manifest, "files": entries})
                self._write_json(account_id, key, manifest)
            self._remove(account_id, pending_key)
            return manifest

    def _validate(self, manifest, *, pending=False):
        if (
            manifest.get("schema_version") != 1
            or manifest.get("pending", False) != pending
            or manifest.get("manifest_sha256") != _sealed(manifest)["manifest_sha256"]
        ):
            raise ValueError("备份清单摘要校验失败")
        account = manifest["account_id"]
        self.files._path(account, self._manifest_key(manifest["backup_id"]))
        as_utc(manifest["retained_until"])
        seen = set()
        for row in manifest["files"]:
            if (
                row["account_id"] != account
                or row["file_id"] in seen
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["file_id"])
            ):
                raise ValueError("备份清单文件归属或标识不一致")
            seen.add(row["file_id"])
            key = row["storage_key"]
            if key:
                expected = f"files/{row['file_id']}/{row['sha256']}/content"
                if key != expected:
                    raise ValueError("备份清单文件键与摘要不一致")
                self.files._path(account, key)
            if pending:
                continue
            if key:
                allowed = {key, str(PurePosixPath(key).with_name("parsed.json"))}
                keys = [item["storage_key"] for item in row["objects"]]
                if set(keys) - allowed or len(keys) != len(set(keys)) or key not in keys:
                    raise ValueError("备份清单对象超出文件范围")
                content = next(item for item in row["objects"] if item["storage_key"] == key)
                if (content["sha256"], content["size_bytes"]) != (row["sha256"], row["size_bytes"]):
                    raise ValueError("备份原件摘要与文件记录不一致")
            elif row["objects"]:
                raise ValueError("备份对象需要对应文件键")
            for item in row["objects"]:
                if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or item["size_bytes"] < 0:
                    raise ValueError("备份对象摘要或大小无效")

    @staticmethod
    def _descriptors(rows):
        return [{key: row[key] for key in _FILE_FIELDS} for row in rows]

    def _match_rows(self, manifest, descriptors):
        if descriptors != self._descriptors(manifest["files"]):
            raise ValueError("数据库文件清单与备份恢复点不一致")

    def verify(self, manifest):
        self._validate(manifest)
        with self.store.transaction(manifest["account_id"]) as tx:
            self._match_rows(manifest, self._descriptors(self._rows(tx)))
        count, size = 0, 0
        for row in manifest["files"]:
            for expected in row["objects"]:
                actual = self._object(manifest["account_id"], expected["storage_key"])
                if actual != expected:
                    raise ValueError("恢复文件摘要或字节数不一致：" + row["file_id"])
                count += 1
                size += actual["size_bytes"]
        return {
            "backup_id": manifest["backup_id"],
            "account_id": manifest["account_id"],
            "verified": True,
            "file_count": len(manifest["files"]),
            "object_count": count,
            "size_bytes": size,
        }

    def _release_marker(self, account_id, row, backup_id):
        if not row["storage_key"]:
            return {}
        key = self._marker_key(row)
        marker = self._read_json(account_id, key) or {}
        refs = set(marker.get("backup_refs", []))
        retained = dict(marker.get("retained_until", {}))
        if backup_id in refs:
            refs.remove(backup_id)
            retained.pop(backup_id, None)
            if refs:
                self._write_json(
                    account_id, key, {"backup_refs": sorted(refs), "retained_until": retained}
                )
            else:
                self._remove(account_id, key)
        return retained

    def release(self, manifest=None, *, account_id=None, backup_id=None):
        if manifest is not None:
            self._validate(manifest)
            account_id, backup_id = manifest["account_id"], manifest["backup_id"]
        if not account_id or not backup_id:
            raise ValueError("请指定备份清单或租户及备份标识")
        released_key = self._manifest_key(backup_id, "released.json")
        pending_key = self._manifest_key(backup_id, "pending.json")
        self.files._path(account_id, released_key)
        with self._serialized(account_id) as transaction:
            released = self._read_json(account_id, released_key)
            if manifest is None:
                manifest = self._read_manifest(account_id, backup_id, pending=True)
                if manifest is None:
                    manifest = self._read_manifest(account_id, backup_id)
                if manifest is None:
                    if released:
                        return released
                    raise ValueError("备份清单不存在")
            remaining = {
                row["file_id"]: self._release_marker(account_id, row, backup_id)
                for row in manifest["files"]
            }
            with transaction() as tx:
                tx.advisory_lock("memorybot.admission")
                for file_id, retained in remaining.items():
                    current = tx.get("bot_file", {"file_id": file_id}, for_update=True)
                    if current is None:
                        continue
                    metadata = dict(current["metadata"])
                    refs = set(metadata.get("backup_refs", []))
                    if backup_id not in refs:
                        continue
                    refs.remove(backup_id)
                    metadata["backup_refs"] = sorted(refs)
                    if not refs:
                        metadata["backup_protected_until"] = None
                    elif refs.issubset(retained):
                        metadata["backup_protected_until"] = json_value(
                            max(as_utc(retained[ref]) for ref in refs)
                        )
                    tx.update("bot_file", {"file_id": file_id}, {"metadata": metadata})
            report = released or {
                "account_id": account_id,
                "backup_id": backup_id,
                "released": True,
                "released_at": json_value(datetime.now(UTC)),
                "file_count": len(manifest["files"]),
            }
            self._write_json(account_id, released_key, report)
            self._remove(account_id, pending_key)
            return report


def run_backup(
    action, config_path=None, account_id=None, backup_id=None, manifest_path=None, retain_days=90
):
    from memorybot.config import BotConfig
    from memorybot.store import BotStore
    from providers.unified_config import get_config

    path = Path(manifest_path) if manifest_path else None
    by_id = action == "release" and account_id and backup_id and (path is None or not path.exists())
    if path is None and not by_id:
        raise ValueError("请指定备份清单路径")
    config = get_config(config_path)
    bot = BotConfig.load(config_path, config)
    files = _AGFSBackupFiles(
        bot.files_base_url, bot.files_mount_prefix, timeout=bot.http_timeout_seconds
    )
    store = BotStore(
        bot.connection_string, dialect=config.sql_dialect, pool_size=bot.pool_size, initialize=False
    )
    manager = BotFileBackup(store, files)
    try:
        if by_id:
            return manager.release(account_id=account_id, backup_id=backup_id)
        if action == "export":
            if not account_id or not backup_id or retain_days < 1:
                raise ValueError("请指定租户、备份标识及正整数保留天数")
            manifest = manager.export(
                account_id,
                backup_id,
                retained_until=datetime.now(UTC) + timedelta(days=retain_days),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(_bytes(manifest))
            os.replace(temporary, path)
            return {
                "backup_id": backup_id,
                "account_id": account_id,
                "manifest_path": str(path.resolve()),
                "file_count": len(manifest["files"]),
                "retained_until": manifest["retained_until"],
            }
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if action == "verify":
            return manager.verify(manifest)
        if action == "release":
            return manager.release(manifest)
        raise ValueError("备份操作须为export、verify或release")
    finally:
        store.close()
