"""Batched retention with durable shared-file deletion receipts."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import timedelta

from psycopg2 import sql
from psycopg2.errors import QueryCanceled

from core.errors import OperationDeadlineExceededError
from core.operation_deadline import check_deadline, operation_deadline
from memorybot.config import BotConfig
from memorybot.scheduling import as_utc
from memorybot.store import (
    GC_OPERATION,
    TERMINAL,
    UPLOAD_OPERATION,
    _advisory_lock_id,
    _table,
    json_value,
)

logger = logging.getLogger(__name__)
_CLEANUP_SECONDS = 0.5
_MAX_REFERENCE_MATCHES = 1000


def _gc_file(file):
    receipt = {"file_id": file["file_id"], "storage_key": file.get("storage_key")}
    if file.get("all_revisions") or file.get("source_kind") == "url":
        receipt["all_revisions"] = True
    return receipt


def _publication_key(source, output):
    result = output or {}
    storage_key = (result.get("publication") or result).get("storage_key")
    if storage_key:
        return storage_key
    if isinstance(source.get("text"), str) and source.get("file_id"):
        digest = hashlib.sha256(source["text"].encode()).hexdigest()
        return f"files/{source['file_id']}/{digest}/content"
    return None


def _object_references(value, name=None):
    check_deadline()
    keys = {
        "task_id": "task",
        "session_id": "session",
        "file_id": "file",
        "schedule_id": "schedule",
        "delivery_id": "delivery",
        "forget_id": "forget",
    }
    aliases = {
        "parent_task_id": "task_id",
        "source_task_id": "task_id",
        "previous_file_id": "file_id",
    }
    name = aliases.get(name, name)
    if name in keys and isinstance(value, str):
        return {(keys[name], name, value)}
    if name == "file_ids" and isinstance(value, list):
        return {("file", "file_id", item) for item in value if isinstance(item, str)}
    if isinstance(value, dict):
        return set().union(*(_object_references(item, key) for key, item in value.items()), set())
    if isinstance(value, list):
        return set().union(*(_object_references(item) for item in value), set())
    return set()


class BotMaintenance:
    def __init__(self, store, files, gateway, config=None, *, interval_seconds=60, batch_size=32):
        self.store, self.files, self.gateway = store, files, gateway
        self.config = config or BotConfig()
        self.interval_seconds, self.batch_size = interval_seconds, batch_size
        self.next_check = 0.0
        self.request_cursor = self.file_cursor = self.gc_cursor = None
        self.task_cursor = self.object_cursor = None
        self.container_cursors = {}

    @staticmethod
    def _related(tx, table, needles):
        columns = {
            "bot_task": ("task_id", "parent_task_id", "input", "result", "schedule_snapshot"),
            "bot_file": ("file_id", "task_id", "previous_file_id", "metadata"),
            "bot_step": ("task_id", "input", "output", "external_receipt"),
            "bot_schedule": ("session_id", "task_input", "baseline"),
            "bot_delivery": ("task_id", "source_task_id", "content"),
            "bot_forget": ("task_id", "source_refs"),
            "bot_request": ("response_body",),
        }[table]
        return tx._matching_reference_rows(table, columns, needles, limit=_MAX_REFERENCE_MATCHES)

    def _queue_gc(self, tx, file, now):
        key = {"operation": GC_OPERATION, "request_id": file["file_id"], "user_id": file["user_id"]}
        if tx.get("bot_request", key):
            return
        tx.insert(
            "bot_request",
            {
                **key,
                "request_hash": hashlib.sha256(
                    (file.get("storage_key") or file["file_id"]).encode()
                ).hexdigest(),
                "response_status": 200,
                "response_body": {"state": "pending", "file": _gc_file(file)},
                "expires_at": now + timedelta(days=self.config.retention_days),
            },
        )

    def _retire_task(self, tx, task_id, cutoff, now, counts):
        task = tx.get("bot_task", {"task_id": task_id}, for_update=True)
        if (
            not task
            or task["state"] not in TERMINAL
            or task["updated_at"] >= cutoff
            or task["finished_at"] >= cutoff
            or task["memory_status"] in {"pending", "stored"}
        ):
            return
        if tx.get("bot_task", {"parent_task_id": task_id}):
            return
        checker = getattr(self.gateway, "has_task_reference", None)
        if checker is None or checker(tx, task):
            return
        files = tx.list("bot_file", {"task_id": task_id}, limit=1001, for_update=True)
        if len(files) > 1000:
            return
        for file in files:
            metadata = file["metadata"]
            if (
                file["updated_at"] >= cutoff
                or metadata.get("memory_sources")
                or metadata.get("backup_refs")
                or (
                    metadata.get("backup_protected_until")
                    and as_utc(metadata["backup_protected_until"]) > now
                )
                or self.gateway.has_file_reference(tx, file)
            ):
                return
        targets = {("task", "task_id", task_id)} | {
            ("file", "file_id", file["file_id"]) for file in files
        }
        needles = [target[2] for target in targets]
        own_files = {file["file_id"] for file in files}
        expired_requests = []
        for table in (
            "bot_task",
            "bot_file",
            "bot_step",
            "bot_schedule",
            "bot_delivery",
            "bot_forget",
            "bot_request",
        ):
            for row in self._related(tx, table, needles):
                if not (_object_references(row) & targets):
                    continue
                if table in {"bot_task", "bot_step"} and row["task_id"] == task_id:
                    continue
                if table == "bot_file" and row["file_id"] in own_files:
                    continue
                if (
                    table == "bot_schedule"
                    and row["state"] in {"cancelled", "completed"}
                    and row["updated_at"] < cutoff
                ):
                    continue
                if (
                    table == "bot_delivery"
                    and row["task_id"] == task_id
                    and row["state"] in {"sent", "cancelled", "failed"}
                    and row["updated_at"] < cutoff
                ):
                    continue
                if (
                    table == "bot_forget"
                    and row["task_id"] == task_id
                    and row["state"] in {"failed", "superseded"}
                    and row["updated_at"] < cutoff
                ):
                    continue
                if table == "bot_request":
                    if row["operation"] == GC_OPERATION:
                        continue
                    if row["expires_at"] <= now:
                        expired_requests.append(row)
                        continue
                return
        for request in expired_requests:
            counts["requests"] += tx.delete(
                "bot_request", {key: request[key] for key in ("user_id", "operation", "request_id")}
            )
        object_keys = {}
        for step in tx.list("bot_step", {"task_id": task_id, "kind": "file_publish"}, limit=1000):
            file_id = step["input"].get("file_id")
            key = _publication_key(step["input"], step["output"])
            if file_id and key:
                object_keys.setdefault(file_id, key)
        for file in files:
            tx.delete("bot_file", {"file_id": file["file_id"]})
            published_key = object_keys.pop(file["file_id"], None)
            self._queue_gc(tx, {**file, "storage_key": file.get("storage_key") or published_key}, now)
            counts["files"] += 1
        for file_id, storage_key in object_keys.items():
            self._queue_gc(
                tx, {"file_id": file_id, "storage_key": storage_key, "user_id": task["user_id"]}, now
            )
        counts["deliveries"] += tx.delete("bot_delivery", {"task_id": task_id})
        tx.delete("bot_forget", {"task_id": task_id})
        counts["steps"] += tx.delete("bot_step", {"task_id": task_id})
        counts["events"] += tx.delete("bot_event", {"task_id": task_id})
        counts["tasks"] += tx.delete("bot_task", {"task_id": task_id})

    def _retire_containers(self, tx, now, counts):
        cutoff = now - timedelta(days=self.config.retention_days)
        definitions = (
            (
                "bot_schedule",
                "schedule_id",
                "schedules",
                "state IN ('cancelled','completed')",
                (("bot_task", "schedule_id"),),
            ),
            (
                "bot_session",
                "session_id",
                "sessions",
                "TRUE",
                (("bot_task", "session_id"), ("bot_schedule", "session_id")),
            ),
        )
        for table, key, counter, condition, dependencies in definitions:
            self._bind(tx, None)
            cursor = self.container_cursors.get(table)
            after = (
                sql.SQL(" AND (updated_at,account_id,{})>(%s,%s,%s)").format(sql.Identifier(key))
                if cursor
                else sql.SQL("")
            )
            tx.cursor.execute(
                sql.SQL(
                    "SELECT * FROM {} WHERE updated_at<%s AND {} {} ORDER BY updated_at,account_id,{} LIMIT %s"
                ).format(sql.Identifier(table), sql.SQL(condition), after, sql.Identifier(key)),
                [cutoff, *(cursor or ()), self.batch_size],
            )
            rows = [dict(item) for item in tx.cursor.fetchall()]
            self.container_cursors[table] = (
                tuple(rows[-1][field] for field in ("updated_at", "account_id", key))
                if len(rows) == self.batch_size
                else None
            )
            for row in rows:
                check_deadline()
                self._bind(tx, row["account_id"])
                row = tx.get(table, {key: row[key]}, for_update=True)
                if any(tx.get(child, {field: row[key]}) for child, field in dependencies):
                    continue
                requests = [
                    request
                    for request in self._related(tx, "bot_request", [row[key]])
                    if (table.removeprefix("bot_"), key, row[key]) in _object_references(request)
                ]
                if any(request["expires_at"] > now for request in requests):
                    continue
                for request in requests:
                    counts["requests"] += tx.delete(
                        "bot_request",
                        {field: request[field] for field in ("user_id", "operation", "request_id")},
                    )
                counts[counter] += tx.delete(table, {key: row[key]})

    def _orphan(self, tx, item, now):
        if item.get("backup_protected") or as_utc(item["modified_at"]) >= now - timedelta(hours=24):
            return
        self._bind(tx, item["account_id"])
        if tx.get("bot_file", {"file_id": item["file_id"]}) or tx.get(
            "bot_file", {"storage_key": item["storage_key"]}
        ):
            return
        tx.cursor.execute(
            "SELECT s.input,s.output,t.task_id,t.user_id FROM bot_step s JOIN bot_task t "
            "ON t.account_id=s.account_id AND t.task_id=s.task_id WHERE s.account_id=%s AND s.kind='file_publish' "
            "AND s.input->>'file_id'=%s AND t.state IN ('succeeded','failed','cancelled') AND t.finished_at<%s LIMIT 1",
            (tx.account_id, item["file_id"], now - timedelta(hours=24)),
        )
        owner = tx.cursor.fetchone()
        if owner is None:
            return
        if _publication_key(owner["input"], owner["output"]) != item["storage_key"]:
            return
        file = {**item, "user_id": owner["user_id"], "task_id": owner["task_id"]}
        if self.gateway.has_file_reference(tx, file):
            return
        for table in ("bot_task", "bot_schedule", "bot_forget", "bot_file"):
            for row in self._related(tx, table, [item["file_id"]]):
                if table == "bot_task" and row["task_id"] == owner["task_id"]:
                    continue
                if ("file", "file_id", item["file_id"]) in _object_references(row):
                    return
        self._queue_gc(tx, file, now)

    @staticmethod
    def _bind(tx, account_id, user_id=None):
        tx.account_id, tx.user_id = account_id, user_id
        tx.cursor.execute("SET LOCAL app.account_id = %s", (account_id or "",))
        tx.cursor.execute("SET LOCAL app.bot_worker = %s", ("on" if account_id is None else "off",))

    def _request_live(self, tx, request):
        for table, key, value in _object_references(request["response_body"]):
            check_deadline()
            if tx.get(_table(table), {key: value}):
                return True
        return False

    def run_once(self, *, force=False):
        now_monotonic = time.monotonic()
        counts = {
            key: 0
            for key in (
                "events",
                "requests",
                "files",
                "objects",
                "tasks",
                "steps",
                "deliveries",
                "sessions",
                "schedules",
            )
        }
        if not force and now_monotonic < self.next_check:
            return counts
        self.next_check = now_monotonic + self.interval_seconds
        for phase in (
            self._expire_events,
            self._expire_requests,
            self._retire_tasks,
            self._retire_containers,
            self._retire_files,
        ):
            self._run_phase(phase, counts)
        self._scan_objects(counts)
        pending = self._run_phase(self._pending_gc, counts, admission=False)
        self._delete_objects(pending or (), counts)
        return counts

    def _scan_objects(self, counts):
        if not hasattr(self.files, "list_objects"):
            return
        try:
            scanned = self.files.list_objects(limit=self.batch_size, cursor=self.object_cursor)
        except Exception:
            logger.exception("memoryBot shared-file scan failed")
            return
        self.object_cursor = scanned["next_cursor"]
        if scanned["items"]:
            self._run_phase(self._queue_orphans, counts, scanned["items"])

    def _run_phase(self, phase, counts, *args, admission=True):
        completed = dict.fromkeys(counts, 0)
        try:
            with operation_deadline(_CLEANUP_SECONDS), self.store.transaction(None) as tx:
                if admission:
                    for lock in ("memorybot.admission", "memorybot.maintenance"):
                        tx.cursor.execute(
                            "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (_advisory_lock_id(lock),)
                        )
                        if not tx.cursor.fetchone()["acquired"]:
                            return None
                tx.cursor.execute("SET LOCAL statement_timeout = %s", (int(_CLEANUP_SECONDS * 1000),))
                result = phase(tx, tx.now(), completed, *args)
                check_deadline()
        except (OperationDeadlineExceededError, QueryCanceled):
            return None
        for name, count in completed.items():
            counts[name] += count
        return result

    def _expire_events(self, tx, now, counts):
        tx.cursor.execute(
            "SELECT account_id,task_id,event_seq FROM bot_event WHERE created_at<%s ORDER BY created_at LIMIT %s",
            (now - timedelta(days=self.config.event_retention_days), self.batch_size),
        )
        events = [dict(row) for row in tx.cursor.fetchall()]
        for row in events:
            check_deadline()
            self._bind(tx, row["account_id"])
            counts["events"] += tx.delete(
                "bot_event", {"task_id": row["task_id"], "event_seq": row["event_seq"]}
            )

    def _expire_requests(self, tx, now, counts):
        request_clause = (
            " AND (expires_at,account_id,user_id,operation,request_id)>(%s,%s,%s,%s,%s)"
            if self.request_cursor
            else ""
        )
        tx.cursor.execute(
            sql.SQL(
                "SELECT * FROM bot_request WHERE expires_at<%s {} ORDER BY expires_at,account_id,user_id,operation,request_id LIMIT %s"
            ).format(sql.SQL(request_clause)),
            [now, *(self.request_cursor or ()), self.batch_size],
        )
        requests = [dict(row) for row in tx.cursor.fetchall()]
        self.request_cursor = (
            tuple(
                requests[-1][key]
                for key in ("expires_at", "account_id", "user_id", "operation", "request_id")
            )
            if len(requests) == self.batch_size
            else None
        )
        for request in requests:
            check_deadline()
            self._bind(tx, request["account_id"], request["user_id"])
            if request["operation"] == UPLOAD_OPERATION:
                file = {
                    **request["response_body"]["file"],
                    "account_id": request["account_id"],
                    "user_id": request["user_id"],
                }
                if tx.get("bot_file", {"file_id": file["file_id"]}):
                    continue
                target = ("file", "file_id", file["file_id"])
                if any(
                    target in _object_references(row)
                    for table in (
                        "bot_task",
                        "bot_step",
                        "bot_schedule",
                        "bot_forget",
                        "bot_file",
                        "bot_delivery",
                    )
                    for row in self._related(tx, table, [file["file_id"]])
                ):
                    continue
                if self.gateway.has_file_reference(tx, file):
                    continue
                self._queue_gc(tx, file, now)
            elif request["operation"] == GC_OPERATION:
                if request["response_body"].get("state") != "deleted":
                    continue
            elif self._request_live(tx, request):
                continue
            counts["requests"] += tx.delete(
                "bot_request",
                {"operation": request["operation"], "request_id": request["request_id"]},
            )

    def _retire_tasks(self, tx, now, counts):
        cutoff = now - timedelta(days=self.config.retention_days)
        task_clause = (
            " AND (updated_at,account_id,task_id)>(%s,%s,%s)" if self.task_cursor else ""
        )
        tx.cursor.execute(
            sql.SQL(
                "SELECT account_id,task_id,updated_at FROM bot_task WHERE state IN ('succeeded','failed','cancelled') "
                "AND updated_at<%s AND finished_at<%s {} ORDER BY updated_at,account_id,task_id LIMIT %s"
            ).format(sql.SQL(task_clause)),
            [cutoff, cutoff, *(self.task_cursor or ()), self.batch_size],
        )
        old_tasks = [dict(row) for row in tx.cursor.fetchall()]
        self.task_cursor = (
            tuple(old_tasks[-1][key] for key in ("updated_at", "account_id", "task_id"))
            if len(old_tasks) == self.batch_size
            else None
        )
        for task in old_tasks:
            check_deadline()
            self._bind(tx, task["account_id"])
            self._retire_task(tx, task["task_id"], cutoff, now, counts)

    def _retire_files(self, tx, now, counts):
        cutoff = now - timedelta(days=self.config.retention_days)
        file_clause = (
            " AND (updated_at,account_id,file_id)>(%s,%s,%s)" if self.file_cursor else ""
        )
        tx.cursor.execute(
            sql.SQL(
                "SELECT account_id,user_id,file_id,updated_at FROM bot_file WHERE updated_at<%s {} ORDER BY updated_at,account_id,file_id LIMIT %s"
            ).format(sql.SQL(file_clause)),
            [cutoff, *(self.file_cursor or ()), self.batch_size],
        )
        candidates = [dict(row) for row in tx.cursor.fetchall()]
        self.file_cursor = (
            tuple(candidates[-1][key] for key in ("updated_at", "account_id", "file_id"))
            if len(candidates) == self.batch_size
            else None
        )
        for candidate in candidates:
            check_deadline()
            self._bind(tx, candidate["account_id"])
            file = self.store.delete_file_metadata(
                tx,
                candidate["file_id"],
                before=cutoff,
                has_memory_reference=self.gateway.has_file_reference,
                reference_scan_limit=_MAX_REFERENCE_MATCHES,
            )
            if file is None:
                continue
            counts["files"] += 1
            self._queue_gc(tx, file, now)

    def _queue_orphans(self, tx, now, _counts, scanned):
        for item in scanned:
            check_deadline()
            self._orphan(tx, item, now)

    def _pending_gc(self, tx, _now, _counts):
        gc_clause = (
            " AND (created_at,account_id,user_id,request_id)>(%s,%s,%s,%s)"
            if self.gc_cursor
            else ""
        )
        tx.cursor.execute(
            sql.SQL(
                "SELECT * FROM bot_request WHERE operation=%s AND response_body->>'state'='pending' {} ORDER BY created_at,account_id,user_id,request_id LIMIT %s"
            ).format(sql.SQL(gc_clause)),
            [GC_OPERATION, *(self.gc_cursor or ()), self.batch_size],
        )
        pending = [dict(row) for row in tx.cursor.fetchall()]
        self.gc_cursor = (
            tuple(
                pending[-1][key]
                for key in ("created_at", "account_id", "user_id", "request_id")
            )
            if len(pending) == self.batch_size
            else None
        )
        return pending

    def _delete_objects(self, pending, counts):
        for request in pending:
            file = _gc_file(request["response_body"]["file"])
            try:
                if file.get("all_revisions") or not file.get("storage_key"):
                    self.files.delete_file_objects(request["account_id"], file["file_id"])
                else:
                    self.files.delete(request["account_id"], file["storage_key"])
                with self.store.transaction(request["account_id"], request["user_id"]) as tx:
                    tx.update(
                        "bot_request",
                        {"operation": GC_OPERATION, "request_id": request["request_id"]},
                        {
                            "response_body": {
                                "file": file,
                                "state": "deleted",
                                "deleted_at": json_value(tx.now()),
                            }
                        },
                    )
                counts["objects"] += 1
            except Exception:
                logger.exception("memoryBot file reclamation pending file=%s", file["file_id"])
