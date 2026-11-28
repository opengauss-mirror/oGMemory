"""Shared SQL state and fenced execution transactions for memoryBot."""

from __future__ import annotations

import builtins
import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from core.errors import OperationDeadlineExceededError
from core.operation_deadline import check_deadline, remaining_seconds
from fs.sql_adapter.pool import SharedConnectionPool
from memorybot.scheduling import as_utc, due_run, next_run

TERMINAL = frozenset({"succeeded", "cancelled", "failed"})
GC_OPERATION = "maintenance:file_gc"
UPLOAD_OPERATION = "maintenance:upload_intent"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
TABLES = frozenset(
    re.findall(r"CREATE TABLE IF NOT EXISTS (bot_\w+)", SCHEMA_PATH.read_text(encoding="utf-8"))
)
FILE_REFERENCE_COLUMNS = {
    "bot_task": ("input", "result"),
    "bot_schedule": ("task_input", "baseline"),
    "bot_forget": ("source_refs",),
    "bot_file": ("previous_file_id", "metadata"),
    "bot_step": ("input", "output", "external_receipt"),
    "bot_delivery": ("content",),
    "bot_request": ("response_body",),
}


class StoreError(RuntimeError):
    code = "store_error"


class NotFoundError(StoreError):
    code = "not_found"


class ConflictError(StoreError):
    code = "conflict"


NotFound = NotFoundError
Conflict = ConflictError


class LeaseLostError(ConflictError):
    code = "lease_lost"


LeaseLost = LeaseLostError


def new_id(prefix: str = "t") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_value(value):
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def step_label(kind, value):
    if value.get("phase", "").startswith("handbook_"):
        return {
            "handbook_segment": "整理资料片段",
            "handbook_chapter": "生成手册章节",
            "handbook_summary": "汇总手册",
            "handbook_merge": "合并手册",
        }.get(value["phase"])
    if kind == "retrieve":
        if "messages" in value:
            return "检索记忆"
        if "file_id" in value:
            return "读取资料"
        if "memory_uri" in value:
            return "读取记忆"
        if value.get("object") == "schedules":
            return "查询计划"
        if value.get("template_id"):
            return "准备模板"
        return None
    if kind == "tool":
        return {
            "search_memory": "查询记忆",
            "read_memory": "读取记忆",
            "list_schedules": "查询计划",
            "create_schedule": "创建计划",
            "update_schedule": "更新计划",
            "memory_action": "处理记忆",
        }.get(value.get("tool"), "统计表格" if value.get("template") else "执行工具")
    if kind == "memory_apply":
        return {"correct": "更正记忆", "forget": "遗忘记忆"}.get(value.get("action"), "收录资料")
    return {
        "model": "生成回复",
        "file_publish": "生成成果文件" if value.get("name") else "解析资料",
        "memory_submit": "提取并保存记忆",
        "memory_poll": "更新记忆索引",
    }.get(kind)


def step_summary(step):
    value, output = step.get("input") or {}, step.get("output") or {}
    summary = {}
    if value.get("tool"):
        summary["tool_name"] = value["tool"]
    for key in ("completed", "total", "unit", "segment", "segment_count", "budget_scope"):
        if key in value:
            summary[key] = value[key]
    if not isinstance(output, dict):
        return summary
    sources = output.get("sources")
    if isinstance(sources, list):
        summary["source_count"] = len(sources)
        summary["sources"] = [
            {key: str(row[key])[:512] for key in ("title", "uri") if row.get(key)}
            for row in sources[:10]
            if isinstance(row, dict)
        ]
    publication = output.get("publication", output)
    if not isinstance(publication, dict):
        publication = {}
    file_id = value.get("file_id") or publication.get("file_id")
    if file_id:
        summary["file_id"] = file_id
    file_name = publication.get("name") or value.get("name")
    if file_name:
        summary["file_name"] = str(file_name).replace("\\", "/").rsplit("/", 1)[-1][:255]
    if isinstance(output.get("file_ids"), list):
        summary["file_count"] = len(output["file_ids"])
    if isinstance(output.get("text"), str):
        summary["character_count"] = len(output["text"])
    if isinstance(output.get("calls"), list):
        summary["tool_call_count"] = len(output["calls"])
    if isinstance(output.get("schedules"), list):
        summary["schedule_count"] = len(output["schedules"])
    for key in ("schedule_id", "memory_status", "confirmed"):
        if key in output:
            summary[key] = output[key]
    for key in ("completed", "total", "unit"):
        if key in output:
            summary[key] = output[key]
    return summary


def _adapt(value):
    return (
        Json(json_value(value), dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
        if isinstance(value, (dict, list))
        else value
    )


def _table(name: str) -> str:
    name = name if name.startswith("bot_") else "bot_" + name
    if name not in TABLES:
        raise ValueError("unknown memoryBot table")
    return name


def _key(value: str) -> sql.Identifier:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("invalid column name")
    return sql.Identifier(value)


def _reference_ids(value, key=None) -> set[str]:
    if key in {"file_id", "previous_file_id"} and isinstance(value, str):
        return {value}
    if key == "file_ids" and isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    if isinstance(value, dict):
        return set().union(*(_reference_ids(item, name) for name, item in value.items()), set())
    if isinstance(value, list):
        return set().union(*(_reference_ids(item) for item in value), set())
    return set()


def _file_references(table: str, values: dict) -> set[str]:
    columns = FILE_REFERENCE_COLUMNS.get(table, ())
    return _reference_ids({column: values[column] for column in columns if column in values})


def _advisory_lock_id(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big", signed=True)


class UnitOfWork:
    """A short SQL transaction scoped to account and optionally user.

    ``cursor`` and ``connection`` allow core memory writes to share this commit.
    Returned rows are dictionaries; timestamp values are aware datetimes.
    """

    def __init__(
        self, store: BotStore, connection, account_id: str | None, user_id: str | None = None
    ):
        self.store, self.connection = store, connection
        self.account_id, self.user_id = account_id, user_id
        self.cursor = connection.cursor(cursor_factory=RealDictCursor)
        self.task: dict = {}
        self._pending_file_references: set[str] = set()

    def _scope(self, values: dict | None) -> dict:
        values = dict(values or {})
        for key, expected in (("account_id", self.account_id), ("user_id", self.user_id)):
            if expected is not None:
                if key in values and values[key] != expected:
                    raise Conflict(f"{key} differs from transaction identity")
                values[key] = expected
        return values

    def _where(self, filters: dict | None):
        scoped = self._scope(filters)
        clauses, params = [], []
        for key, value in scoped.items():
            clauses.append(
                sql.SQL("{} IS NULL").format(_key(key))
                if value is None
                else sql.SQL("{} = %s").format(_key(key))
            )
            if value is not None:
                params.append(_adapt(value))
        return sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("TRUE"), params

    def _matching_reference_rows(self, table, columns, needles, *, limit=None, fields=()):
        remaining = remaining_seconds()
        if remaining is not None:
            self.cursor.execute("SET LOCAL statement_timeout = %s", (max(1, int(remaining * 1000)),))
        where, params = self._where({})
        like = sql.SQL(" OR ").join(
            sql.SQL("CAST({} AS TEXT) LIKE %s").format(_key(column))
            for column in columns
            for _ in needles
        )
        projection = sql.SQL(", ").join(map(_key, fields)) if fields else sql.SQL("*")
        self.cursor.execute(
            sql.SQL("SELECT {} FROM {} WHERE {} AND ({}){}").format(
                projection, _key(table), where, like,
                sql.SQL(" LIMIT %s") if limit is not None else sql.SQL(""),
            ),
            params + [f"%{needle}%" for _ in columns for needle in needles]
            + ([limit + 1] if limit is not None else []),
        )
        rows = self.cursor.fetchall()
        if limit is not None and len(rows) > limit:
            raise OperationDeadlineExceededError("Maintenance reference limit exceeded")
        for row in rows:
            check_deadline()
            yield dict(row)

    def get(
        self, table: str, filters: dict, for_update: bool = False, *, nowait: bool = False
    ) -> dict | None:
        where, params = self._where(filters)
        suffix = (
            sql.SQL(" FOR UPDATE NOWAIT" if nowait else " FOR UPDATE")
            if for_update
            else sql.SQL("")
        )
        self.cursor.execute(
            sql.SQL("SELECT * FROM {} WHERE {} LIMIT 1{}").format(
                _key(_table(table)), where, suffix
            ),
            params,
        )
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def list(
        self,
        table: str,
        filters: dict | None = None,
        order_by: str = "created_at DESC",
        limit: int = 100,
        offset: int = 0,
        *,
        for_update: bool = False,
    ) -> builtins.list[dict]:
        where, params = self._where(filters)
        ordering = []
        for term in order_by.split(","):
            match = re.fullmatch(r"\s*([a-z][a-z0-9_]*)(?:\s+(ASC|DESC))?\s*", term, re.IGNORECASE)
            if not match:
                raise ValueError("invalid ordering")
            ordering.append(
                sql.SQL("{} {}").format(_key(match[1]), sql.SQL((match[2] or "ASC").upper()))
            )
        self.cursor.execute(
            sql.SQL("SELECT * FROM {} WHERE {} ORDER BY {} LIMIT %s OFFSET %s{}").format(
                _key(_table(table)),
                where,
                sql.SQL(", ").join(ordering),
                sql.SQL(" FOR UPDATE" if for_update else ""),
            ),
            params + [max(0, limit), max(0, offset)],
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def _lock_references(self, table: str, values: dict) -> set[str]:
        if self.account_id is None:
            return set()
        missing = set()
        for file_id in sorted(_file_references(table, values)):
            if self.get("bot_file", {"file_id": file_id}, for_update=True) is None:
                missing.add(file_id)
        return missing

    def _track_missing_references(
        self, table: str, missing: set[str], rows: builtins.list[dict]
    ) -> None:
        if not missing or not rows:
            return
        if table == "bot_request":
            for request in rows:
                if request["operation"] in {GC_OPERATION, UPLOAD_OPERATION}:
                    self._pending_file_references.update(missing - {request["request_id"]})
                else:
                    self._pending_file_references.update(missing)
            return
        if table != "bot_step":
            self._pending_file_references.update(missing)
            return
        for step in rows:
            publication = step["input"]
            if (
                step["kind"] != "file_publish"
                or not isinstance(publication, dict)
                or not publication.get("name")
            ):
                self._pending_file_references.update(missing)
                return
            # The file row is created with the task result, after its publication checkpoint.
            planned_file_id = publication.get("file_id")
            self._pending_file_references.update(
                file_id for file_id in missing if file_id != planned_file_id
            )

    def _check_file_references(self) -> None:
        # Ingest creates its task before inserting the file that belongs to the task.
        for file_id in sorted(self._pending_file_references):
            if self.get("bot_file", {"file_id": file_id}, for_update=True) is None:
                raise NotFound(f"File not found: {file_id}")

    def insert(self, table: str, values: dict) -> dict:
        table, values = _table(table), self._scope(values)
        missing = self._lock_references(table, values)
        columns = list(values)
        self.cursor.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
                _key(table),
                sql.SQL(", ").join(_key(key) for key in columns),
                sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            ),
            [_adapt(values[key]) for key in columns],
        )
        row = dict(self.cursor.fetchone())
        self._track_missing_references(table, missing, [row])
        return row

    def update(self, table: str, filters: dict, changes: dict) -> dict | None:
        table, changes = _table(table), dict(changes)
        self._scope(changes)
        if table == "bot_file" and isinstance(changes.get("metadata"), dict):
            current = self.get(table, filters, for_update=True)
            if current:
                metadata = dict(changes["metadata"])
                for key in ("backup_refs", "backup_protected_until"):
                    if key in current["metadata"]:
                        metadata.setdefault(key, current["metadata"][key])
                changes["metadata"] = metadata
        missing = self._lock_references(table, changes)
        if table != "bot_event":
            changes.setdefault("updated_at", self.now())
        if not changes:
            return self.get(table, filters)
        where, params = self._where(filters)
        self.cursor.execute(
            sql.SQL("UPDATE {} SET {} WHERE {} RETURNING *").format(
                _key(table),
                sql.SQL(", ").join(sql.SQL("{} = %s").format(_key(key)) for key in changes),
                where,
            ),
            [_adapt(value) for value in changes.values()] + params,
        )
        rows = self.cursor.fetchall()
        self._track_missing_references(table, missing, rows)
        return dict(rows[0]) if rows else None

    def delete(self, table: str, filters: dict) -> int:
        where, params = self._where(filters)
        self.cursor.execute(
            sql.SQL("DELETE FROM {} WHERE {}").format(_key(_table(table)), where), params
        )
        return cast(int, self.cursor.rowcount)

    def now(self) -> datetime:
        self.cursor.execute("SELECT clock_timestamp() AS now")
        return cast(datetime, self.cursor.fetchone()["now"])

    def advisory_lock(self, identity: str) -> None:
        self.cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_advisory_lock_id(identity),))


class BotStore:
    """PostgreSQL/openGauss persistence shared by API and worker instances."""

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        pool=None,
        initialize: bool = True,
        dialect: str = "postgresql",
        pool_size: int = 10,
    ):
        if pool is None and not connection_string:
            raise ValueError("connection_string is required when pool is omitted")
        self.pool = pool or SharedConnectionPool(
            cast(str, connection_string), pool_size=pool_size, dialect=dialect
        )
        self._owns_pool = pool is None
        self.recover_memory_write = None
        if initialize:
            self.initialize()

    def initialize(self) -> None:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        version = "memorybot-schema:" + hashlib.sha256(schema.encode()).hexdigest()
        for attempt in range(20):
            try:
                with self.transaction(None) as tx:
                    tx.advisory_lock("memorybot.admission")
                    tx.advisory_lock("memorybot.schema.v1")
                    tx.cursor.execute(
                        "SELECT obj_description(c.oid, 'pg_class') AS version FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relname='bot_task' AND n.nspname=current_schema()"
                    )
                    current = tx.cursor.fetchone()
                    if current and current["version"] == version:
                        return
                    tx.cursor.execute(
                        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                        "WHERE n.nspname=current_schema() AND c.relname=ANY(%s)",
                        (sorted(TABLES),),
                    )
                    existing = [row["relname"] for row in tx.cursor.fetchall()]
                    if existing:
                        tx.cursor.execute(
                            sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE NOWAIT").format(
                                sql.SQL(", ").join(_key(table) for table in sorted(existing))
                            )
                        )
                    with tx.connection.cursor() as cursor:
                        self.pool.sql_dialect.validate_connection(cursor)
                    tx.cursor.execute(self.pool.sql_dialect.prepare_schema(schema))
                    tx.cursor.execute("COMMENT ON TABLE bot_task IS %s", (version,))
                return
            except psycopg2.Error as error:
                if error.pgcode not in {"55P03", "40001", "40P01"} or attempt == 19:
                    raise
                time.sleep(0.05)

    def close(self) -> None:
        if self._owns_pool:
            self.pool.close_all()

    @contextmanager
    def transaction(
        self, account_id: str | None, user_id: str | None = None
    ) -> Iterator[UnitOfWork]:
        """Commit scoped changes together; ``None`` is reserved for internal discovery."""
        with self._connection() as conn, self._transaction_on(conn, account_id, user_id) as tx:
            yield tx

    @contextmanager
    def _connection(self):
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.return_connection(conn)

    @contextmanager
    def _transaction_on(self, conn, account_id, user_id=None) -> Iterator[UnitOfWork]:
        tx = UnitOfWork(self, conn, account_id, user_id)
        try:
            tx.cursor.execute("SET LOCAL app.account_id = %s", (account_id or "",))
            tx.cursor.execute(
                "SET LOCAL app.bot_worker = %s", ("on" if account_id is None else "off",)
            )
            yield tx
            tx._check_file_references()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            tx.cursor.close()

    @contextmanager
    def serialized_transactions(self, account_id: str, identity: str):
        """Run short transactions on one connection under a session advisory lock."""
        lock_id = _advisory_lock_id(identity)
        with self._connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
                conn.commit()
            except BaseException:
                conn.close()
                raise
            try:
                yield partial(self._transaction_on, conn, account_id)
            finally:
                try:
                    conn.rollback()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                    conn.commit()
                except BaseException:
                    conn.close()
                    raise

    def get(
        self,
        account_id: str,
        table: str,
        filters: dict,
        user_id: str | None = None,
        *,
        for_update: bool = False,
    ) -> dict | None:
        with self.transaction(account_id, user_id) as tx:
            return tx.get(table, filters, for_update)

    def list(
        self,
        account_id: str,
        table: str,
        filters: dict | None = None,
        user_id: str | None = None,
        *,
        order_by: str = "created_at DESC",
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[dict]:
        with self.transaction(account_id, user_id) as tx:
            return tx.list(table, filters, order_by, limit, offset)

    def get_task(self, account_id: str, task_id: str, user_id: str | None = None) -> dict | None:
        return self.get(account_id, "bot_task", {"task_id": task_id}, user_id)

    def task_steps(self, account_id, task_id, user_id=None, *, tx=None):
        if tx is None:
            with self.transaction(account_id, user_id) as transaction:
                return self.task_steps(account_id, task_id, user_id, tx=transaction)
        task = tx.get("bot_task", {"task_id": task_id})
        if task is None:
            raise NotFound("task not found")
        snapshots = []
        for step in tx.list("bot_step", {"task_id": task_id}, "step_no ASC", limit=10000):
            label = step_label(step["kind"], step["input"])
            if not label:
                continue
            snapshot = step.get("public_progress")
            if not snapshot:
                state = step["state"]
                if state in {"pending", "running", "waiting_confirmation"}:
                    if task["state"] in {"cancelled", "failed"}:
                        state = task["state"]
                    elif task["state"] == "retry_wait":
                        state = "retry_wait"
                finished = step["updated_at"] if state in TERMINAL else None
                snapshot = {
                    "step_id": f"{task_id}:{step['step_no']}",
                    "step_no": step["step_no"],
                    "kind": step["kind"],
                    "label": label,
                    "state": state,
                    "attempt": task["attempt"],
                    "event_seq": 0,
                    "started_at": step["created_at"],
                    "finished_at": finished,
                    "elapsed_ms": (
                        max(0, round((finished - step["created_at"]).total_seconds() * 1000))
                        if finished
                        else None
                    ),
                    "summary": step_summary(step),
                }
            snapshots.append(json_value(snapshot))
        return snapshots

    def step_progress(self, tx, step, state, attempt, *, error_code=None):
        label = step_label(step["kind"], step["input"])
        if not label:
            return step
        previous = step.get("public_progress") or {}
        if previous.get("state") == state and previous.get("attempt") == attempt:
            return step
        now = tx.now()
        started = (
            now if state == "running" else as_utc(previous.get("started_at") or step["created_at"])
        )
        finished = None if state == "running" else now
        summary = step_summary(step)
        if error_code:
            summary["error_code"] = error_code
        task = tx.get("bot_task", {"task_id": step["task_id"]}, for_update=True)
        snapshot = json_value(
            {
                "step_id": f"{step['task_id']}:{step['step_no']}",
                "step_no": step["step_no"],
                "kind": step["kind"],
                "label": label,
                "state": state,
                "attempt": attempt,
                "event_seq": task["next_event_seq"],
                "started_at": started,
                "finished_at": finished,
                "elapsed_ms": max(0, round((now - started).total_seconds() * 1000)),
                "summary": summary,
            }
        )
        event_type = {
            "running": "step.started",
            "succeeded": "step.completed",
            "failed": "step.failed",
            "cancelled": "step.cancelled",
        }.get(state, "step.updated")
        self.append_event(tx, step["task_id"], event_type, snapshot)
        return tx.update(
            "bot_step",
            {"task_id": step["task_id"], "step_no": step["step_no"]},
            {"public_progress": snapshot},
        )

    def pending_counts(self, tx: UnitOfWork, user_id: str) -> dict:
        """Count global admission and owned backlog while holding its transaction lock."""
        tx.advisory_lock("memorybot.admission")
        tx.cursor.execute("SET LOCAL app.bot_worker = 'on'")
        tx.cursor.execute(
            "SELECT count(*) AS total, count(CASE WHEN account_id=%s AND user_id=%s THEN 1 END) AS owned "
            "FROM bot_task WHERE state IN ('queued','running','retry_wait','waiting_confirmation')",
            (tx.account_id, user_id),
        )
        counts = dict(tx.cursor.fetchone())
        tx.cursor.execute(
            "SET LOCAL app.bot_worker = %s", ("on" if tx.account_id is None else "off",)
        )
        return counts

    def idempotent(
        self,
        account_id: str,
        user_id: str,
        operation: str,
        request_id: str,
        request_hash: str,
        callback: Callable[[UnitOfWork], tuple[int, dict]],
    ) -> tuple[int, dict, bool]:
        """Execute callback and persist its JSON-safe HTTP response in one transaction."""
        with self.transaction(account_id, user_id) as tx:
            # Serialize each request key so replay returns the response committed with its effects.
            tx.advisory_lock("memorybot.admission")
            request_lock_key = json.dumps(
                [account_id, user_id, operation, request_id], separators=(",", ":")
            )
            tx.advisory_lock(request_lock_key)
            key = {"operation": operation, "request_id": request_id}
            prior = tx.get("bot_request", key)
            if prior:
                if prior["request_hash"] != request_hash:
                    raise Conflict("request key already identifies another payload")
                return prior["response_status"], prior["response_body"], True
            status, body = callback(tx)
            body = json_value(body)
            tx.insert(
                "bot_request",
                {
                    **key,
                    "request_hash": request_hash,
                    "response_status": status,
                    "response_body": body,
                },
            )
            return status, body, False

    def create_task(self, tx: UnitOfWork, values: dict) -> dict:
        """Allocate the session sequence and persist a task with its first event."""
        values = dict(values)
        values.setdefault("task_id", new_id())
        values.setdefault("state", "queued")
        values.setdefault("created_at", tx.now())
        values.setdefault("updated_at", values["created_at"])
        if values["kind"] == "chat":
            # Allocate seq under the session lock so workers share one conversation order.
            session = tx.get("bot_session", {"session_id": values["session_id"]}, for_update=True)
            if session is None:
                raise NotFound("session not found")
            values.setdefault("user_id", session["user_id"])
            values["seq"] = session["next_seq"]
            tx.update(
                "bot_session",
                {"session_id": session["session_id"]},
                {"next_seq": session["next_seq"] + 1},
            )
        else:
            values["seq"] = None
        if values.get("parent_task_id"):
            parent = tx.get("bot_task", {"task_id": values["parent_task_id"]})
            if parent is None:
                raise NotFound("parent task not found")
            values.setdefault("user_id", parent["user_id"])
            values.setdefault("accepted_authz", parent["accepted_authz"])
            values.setdefault("session_id", parent["session_id"])
        task = tx.insert("bot_task", values)
        self.append_event(
            tx, task["task_id"], "task.updated", {"state": task["state"], "confirmation": None}
        )
        return cast(dict, tx.get("bot_task", {"task_id": task["task_id"]}))

    def append_event(self, tx: UnitOfWork, task_id: str, type: str, payload: dict) -> dict:
        task = tx.get("bot_task", {"task_id": task_id}, for_update=True)
        if task is None:
            raise NotFound("task not found")
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 65536:
            raise ValueError("event payload exceeds 64 KiB")
        event = tx.insert(
            "bot_event",
            {
                "user_id": task["user_id"],
                "task_id": task_id,
                "event_seq": task["next_event_seq"],
                "type": type,
                "payload": payload,
            },
        )
        tx.update("bot_task", {"task_id": task_id}, {"next_event_seq": task["next_event_seq"] + 1})
        return event

    def events(
        self,
        account_id: str,
        task_id: str,
        after: int = 0,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> builtins.list[dict]:
        with self.transaction(account_id, user_id) as tx:
            where, params = tx._where({"task_id": task_id})
            tx.cursor.execute(
                sql.SQL(
                    "SELECT * FROM bot_event WHERE {} AND event_seq > %s ORDER BY event_seq LIMIT %s"
                ).format(where),
                params + [after, limit],
            )
            return [dict(row) for row in tx.cursor.fetchall()]

    def claim(
        self,
        owner_id: str,
        *,
        kinds: builtins.list[str] | None = None,
        account_id: str | None = None,
        lease_seconds: int = 30,
        cluster_limit: int = 16,
        tenant_limit: int = 8,
    ) -> dict | None:
        with self.transaction(None) as tx:
            clauses = [
                "((state='queued' AND (next_attempt_at IS NULL OR next_attempt_at<=clock_timestamp())) OR "
                "(state='retry_wait' AND next_attempt_at<=clock_timestamp()) OR (state='running' AND lease_until<=clock_timestamp()))"
            ]
            params: builtins.list[Any] = []
            if account_id is not None:
                clauses.append("account_id=%s")
                params.append(account_id)
            if kinds is not None:
                clauses.append("kind=ANY(%s)")
                params.append(kinds)
            clauses.append(
                "(SELECT count(*) FROM bot_task active WHERE active.account_id=bot_task.account_id AND active.state='running' AND active.lease_until>clock_timestamp()) < %s"
            )
            params.append(tenant_limit)
            clauses.append(
                "(kind<>'chat' OR NOT EXISTS (SELECT 1 FROM bot_task earlier WHERE earlier.account_id=bot_task.account_id AND earlier.session_id=bot_task.session_id AND earlier.kind='chat' AND earlier.seq<bot_task.seq AND earlier.state NOT IN ('succeeded','cancelled','failed')))"
            )
            clauses.append(
                "(kind<>'memory_write' OR NOT EXISTS (SELECT 1 FROM bot_task earlier WHERE earlier.account_id=bot_task.account_id AND earlier.user_id=bot_task.user_id AND earlier.kind='memory_write' AND (earlier.created_at,earlier.task_id)<(bot_task.created_at,bot_task.task_id) AND earlier.state NOT IN ('succeeded','cancelled','failed')))"
            )
            tx.cursor.execute(
                sql.SQL(
                    "SELECT account_id,task_id,kind,session_id FROM bot_task WHERE {} ORDER BY created_at,task_id LIMIT 128"
                ).format(sql.SQL(" AND ").join(sql.SQL(clause) for clause in clauses)),
                params,
            )
            candidates = [dict(row) for row in tx.cursor.fetchall()]
        for candidate in candidates:
            try:
                with self.transaction(candidate["account_id"]) as tx:
                    tx.advisory_lock("memorybot.admission")
                    tx.advisory_lock("memorybot.execution.capacity")
                    tx.cursor.execute("SET LOCAL app.bot_worker = 'on'")
                    tx.cursor.execute(
                        "SELECT count(*) AS total, count(CASE WHEN account_id=%s THEN 1 END) AS owned "
                        "FROM bot_task WHERE state='running' AND lease_until>clock_timestamp()",
                        (candidate["account_id"],),
                    )
                    running_counts = tx.cursor.fetchone()
                    tx.cursor.execute("SET LOCAL app.bot_worker = 'off'")
                    if running_counts["total"] >= cluster_limit:
                        return None
                    if running_counts["owned"] >= tenant_limit:
                        continue
                    if candidate["kind"] == "chat":
                        # Keep later turns behind retries and pending confirmations.
                        tx.get(
                            "bot_session",
                            {"session_id": candidate["session_id"]},
                            True,
                            nowait=True,
                        )
                        tx.cursor.execute(
                            "SELECT task_id FROM bot_task WHERE account_id=%s AND session_id=%s AND kind='chat' AND state NOT IN ('succeeded','cancelled','failed') ORDER BY seq LIMIT 1",
                            (candidate["account_id"], candidate["session_id"]),
                        )
                        session_head = tx.cursor.fetchone()
                        if not session_head or session_head["task_id"] != candidate["task_id"]:
                            continue
                    task = tx.get("bot_task", {"task_id": candidate["task_id"]}, True, nowait=True)
                    now = tx.now()
                    if task is None or not self._eligible(task, now):
                        continue
                    if task["cancel_requested"]:
                        self._terminal(tx, task, "cancelled", task.get("result"), task.get("error"))
                        continue
                    previous_attempt = task["attempt"]
                    task = cast(
                        dict,
                        tx.update(
                            "bot_task",
                            {"task_id": task["task_id"]},
                            {
                                "state": "running",
                                "owner_id": owner_id,
                                "attempt": previous_attempt + 1,
                                "lease_until": now + timedelta(seconds=lease_seconds),
                                "next_attempt_at": None,
                            },
                        ),
                    )
                    if previous_attempt and task["kind"] == "chat":
                        self.append_event(
                            tx,
                            task["task_id"],
                            "answer.reset",
                            {"attempt": task["attempt"], "replaced_attempt": previous_attempt},
                        )
                    self.append_event(
                        tx,
                        task["task_id"],
                        "task.updated",
                        {"state": "running", "confirmation": None},
                    )
                    return cast(dict, tx.get("bot_task", {"task_id": task["task_id"]}))
            except psycopg2.Error as error:
                if error.pgcode in {"55P03", "40001", "40P01"}:
                    continue
                raise
        return None

    @staticmethod
    def _eligible(task: dict | None, now: datetime) -> bool:
        if not task:
            return False
        if task["state"] == "running":
            return bool(task["lease_until"] <= now)
        return bool(
            task["state"] in {"queued", "retry_wait"}
            and (task["next_attempt_at"] is None or task["next_attempt_at"] <= now)
        )

    def renew(
        self, account_id: str, task_id: str, owner_id: str, attempt: int, lease_seconds: int = 30
    ) -> dict:
        with self.transaction(account_id) as tx:
            tx.cursor.execute(
                "UPDATE bot_task SET lease_until=clock_timestamp()+(%s*INTERVAL '1 second'), updated_at=clock_timestamp() "
                "WHERE account_id=%s AND task_id=%s AND owner_id=%s AND attempt=%s AND state='running' AND lease_until>clock_timestamp() RETURNING *",
                (lease_seconds, account_id, task_id, owner_id, attempt),
            )
            row = tx.cursor.fetchone()
            if row is None:
                raise LeaseLost("execution lease changed or expired")
            return dict(row)

    @contextmanager
    def execution(
        self, account_id: str, task_id: str, owner_id: str, attempt: int
    ) -> Iterator[UnitOfWork]:
        """Lock and verify the current execution generation before committing work."""
        with self.transaction(account_id) as tx:
            # Fenced writes acquire locks in admission, schedule, session, then task order.
            tx.advisory_lock("memorybot.admission")
            identity = tx.get("bot_task", {"task_id": task_id})
            if identity and identity.get("schedule_id"):
                tx.get("bot_schedule", {"schedule_id": identity["schedule_id"]}, for_update=True)
            if identity and identity["kind"] == "chat":
                tx.get("bot_session", {"session_id": identity["session_id"]}, for_update=True)
            task = tx.get("bot_task", {"task_id": task_id}, for_update=True)
            # Check owner and attempt on entry, then recheck lease expiry before commit.
            if (
                not task
                or task["state"] != "running"
                or task["owner_id"] != owner_id
                or task["attempt"] != attempt
                or task["lease_until"] <= tx.now()
            ):
                raise LeaseLost("execution lease changed or expired")
            tx.user_id, tx.task = task["user_id"], task
            yield tx
            if task["lease_until"] <= tx.now():
                raise LeaseLost("execution transaction exceeded its lease")

    def checkpoint(
        self,
        account_id: str,
        task_id: str,
        owner_id: str,
        attempt: int,
        step_no: int,
        kind: str,
        input: dict,
        *,
        state: str = "running",
        output: dict | None = None,
        external_receipt: dict | None = None,
        operation_id: str | None = None,
    ) -> dict:
        """Freeze a step's operation key and input, then persist reusable progress."""
        with self.execution(account_id, task_id, owner_id, attempt) as tx:
            key = {"task_id": task_id, "step_no": step_no}
            step = tx.get("bot_step", key)
            # Reuse the original input and operation ID so retries replay the same request.
            if step and step["state"] == "succeeded":
                return step
            if step is None:
                step = tx.insert(
                    "bot_step",
                    {
                        **key,
                        "kind": kind,
                        "operation_id": operation_id or new_id("op"),
                        "input": input,
                    },
                )
            updated = tx.update(
                "bot_step",
                key,
                {"state": state, "output": output, "external_receipt": external_receipt},
            )
            return cast(dict, self.step_progress(tx, updated, state, attempt))

    def _cancel_related_state(
        self, tx: UnitOfWork, task: dict, result: dict | None
    ) -> dict | None:
        task_id = task["task_id"]
        kind = task["kind"]
        if kind == "ingest":
            tx.update("bot_file", {"task_id": task_id, "state": "processing"}, {"state": "failed"})
        elif kind == "delivery":
            filters = {"task_id": task_id}
            delivery = tx.get("bot_delivery", filters, for_update=True)
            if delivery and delivery["state"] in {
                "pending",
                "sending",
                "retry_wait",
                "waiting_confirmation",
            }:
                tx.update("bot_delivery", filters, {"state": "cancelled"})
        if kind != "memory_write":
            return result

        for related_id in (task_id, task.get("parent_task_id")):
            if related_id and tx.get(
                "bot_task", {"task_id": related_id, "memory_status": "pending"}, for_update=True
            ):
                self.set_memory_status(tx, related_id, "failed")

        if self._settle_interrupted_forget(tx, task):
            result = result or {
                "text": "已停止后续处理，已提交的记忆变更保留。",
                "citations": [],
                "file_ids": [],
            }
        return result

    def _settle_interrupted_forget(self, tx: UnitOfWork, task: dict) -> bool:
        applied = False
        changes: dict[str, Any] = {"state": "failed"}
        for step in tx.list("bot_step", {"task_id": task["task_id"], "kind": "memory_apply"}):
            if step["state"] == "succeeded":
                applied = True
                refs = (step.get("output") or {}).get("source_refs")
            elif self.recover_memory_write is not None:
                refs = self.recover_memory_write(task, step)
                applied = applied or refs is not None
            else:
                continue
            if refs:
                changes["source_refs"] = refs
        if applied:
            changes["state"] = "effective"
        tx.update("bot_forget", {"task_id": task["task_id"], "state": "pending"}, changes)
        return applied

    def _terminal(
        self, tx: UnitOfWork, task: dict, state: str, result: dict | None, error: dict | None
    ) -> dict:
        if state == "cancelled":
            result = self._cancel_related_state(tx, task, result)
        if state in {"cancelled", "failed"}:
            for step in tx.list(
                "bot_step", {"task_id": task["task_id"]}, "step_no ASC", limit=10000
            ):
                if step["state"] in {"pending", "running", "waiting_confirmation"}:
                    step = cast(
                        dict,
                        tx.update(
                            "bot_step",
                            {"task_id": task["task_id"], "step_no": step["step_no"]},
                            {"state": state},
                        ),
                    )
                    self.step_progress(
                        tx, step, state, task["attempt"], error_code=(error or {}).get("code")
                    )
        if state == "succeeded" and result is None:
            result = {"text": "", "citations": [], "file_ids": []}
        tx.update(
            "bot_task",
            {"task_id": task["task_id"]},
            {
                "state": state,
                "result": result,
                "error": error,
                "owner_id": None,
                "lease_until": None,
                "next_attempt_at": None,
                "finished_at": tx.now(),
            },
        )
        self.append_event(
            tx,
            task["task_id"],
            "task.completed",
            {"state": state, "result_url": f"/api/v1/bot/tasks/{task['task_id']}", "error": error},
        )
        return cast(dict, tx.get("bot_task", {"task_id": task["task_id"]}))

    def finish(
        self,
        account_id: str,
        task_id: str,
        owner_id: str,
        attempt: int,
        result: dict | None = None,
        *,
        state: str = "succeeded",
        error: dict | None = None,
        children: builtins.list[dict] | None = None,
        files: builtins.list[dict] | None = None,
        deliveries: builtins.list[dict] | None = None,
        mutate: Callable[[UnitOfWork], None] | None = None,
    ) -> dict:
        """Commit result, children, files, deliveries and caller changes atomically."""
        if state not in TERMINAL:
            raise ValueError("finish requires a terminal state")
        # Commit result metadata and follow-up work with the terminal state and event.
        with self.execution(account_id, task_id, owner_id, attempt) as tx:
            task = tx.task
            if task["cancel_requested"]:
                state = "cancelled"
            if mutate:
                mutate(tx)
            for record in files or []:
                tx.insert("bot_file", record)
            if state == "succeeded":
                for child in children or []:
                    self.create_task(tx, {"parent_task_id": task_id, **child})
                for delivery in deliveries or []:
                    tx.insert("bot_delivery", delivery)
            return self._terminal(tx, task, state, result, error)

    def retry(
        self,
        account_id: str,
        task_id: str,
        owner_id: str,
        attempt: int,
        error: dict,
        *,
        delay_seconds: float | None = None,
    ) -> dict:
        with self.execution(account_id, task_id, owner_id, attempt) as tx:
            task = tx.task
            if task["cancel_requested"]:
                return self._terminal(tx, task, "cancelled", task["result"], error)
            if task["retry_count"] >= 3:
                return self._terminal(tx, task, "failed", task["result"], error)
            delay = (5, 30, 120)[task["retry_count"]] if delay_seconds is None else delay_seconds
            tx.update(
                "bot_task",
                {"task_id": task_id},
                {
                    "state": "retry_wait",
                    "retry_count": task["retry_count"] + 1,
                    "next_attempt_at": tx.now() + timedelta(seconds=delay),
                    "owner_id": None,
                    "lease_until": None,
                    "error": error,
                },
            )
            self.append_event(
                tx, task_id, "task.updated", {"state": "retry_wait", "confirmation": None}
            )
            return cast(dict, tx.get("bot_task", {"task_id": task_id}))

    def wait_confirmation(
        self, account_id: str, task_id: str, owner_id: str, attempt: int, error: dict
    ) -> dict:
        with self.execution(account_id, task_id, owner_id, attempt) as tx:
            task = tx.task
            if task["cancel_requested"]:
                return self._terminal(tx, task, "cancelled", task["result"], error)
            tx.update(
                "bot_task",
                {"task_id": task_id},
                {
                    "state": "waiting_confirmation",
                    "error": error,
                    "owner_id": None,
                    "lease_until": None,
                },
            )
            self.append_event(
                tx,
                task_id,
                "task.updated",
                {"state": "waiting_confirmation", "confirmation": error.get("details")},
            )
            self.notify_memory_confirmation(tx, task)
            return cast(dict, tx.get("bot_task", {"task_id": task_id}))

    def notify_memory_confirmation(self, tx: UnitOfWork, task: dict) -> None:
        if task["kind"] == "memory_write" and task.get("parent_task_id"):
            parent = tx.get("bot_task", {"task_id": task["parent_task_id"]})
            self.append_event(
                tx,
                parent["task_id"],
                "task.updated",
                {
                    "state": parent["state"],
                    "confirmation": None,
                    "memory_confirmation_task_id": task["task_id"],
                },
            )

    def set_memory_status(
        self, tx: UnitOfWork, task_id: str, status: str, operation_id: str | None = None
    ) -> dict:
        changes = {"memory_status": status}
        if operation_id is not None:
            changes["memory_operation_id"] = operation_id
        task = tx.update("bot_task", {"task_id": task_id}, changes)
        if task is None:
            raise NotFound("task not found")
        self.append_event(
            tx,
            task_id,
            "memory.updated",
            {"memory_status": status, "memory_operation_id": task["memory_operation_id"]},
        )
        return task

    def trigger_due(
        self,
        now: datetime | str | None = None,
        limit: int = 32,
        *,
        max_pending: int = 10000,
        max_user_pending: int = 200,
    ) -> builtins.list[dict]:
        """Atomically create one coalesced task per due, idle schedule."""
        with self.transaction(None) as tx:
            now = as_utc(now) if now is not None else tx.now()
            tx.cursor.execute(
                "SELECT account_id,schedule_id FROM bot_schedule WHERE state='active' AND next_run_at<=%s "
                "AND NOT EXISTS (SELECT 1 FROM bot_task active WHERE active.account_id=bot_schedule.account_id AND active.schedule_id=bot_schedule.schedule_id "
                "AND active.state NOT IN ('succeeded','failed','cancelled')) ORDER BY next_run_at,schedule_id LIMIT %s",
                (now, limit),
            )
            candidates = [dict(row) for row in tx.cursor.fetchall()]
        created = []
        for candidate in candidates:
            try:
                with self.transaction(candidate["account_id"]) as tx:
                    tx.advisory_lock("memorybot.admission")
                    schedule = tx.get(
                        "bot_schedule", {"schedule_id": candidate["schedule_id"]}, True, nowait=True
                    )
                    if (
                        not schedule
                        or schedule["state"] != "active"
                        or schedule["next_run_at"] > now
                    ):
                        continue
                    tx.user_id = schedule["user_id"]
                    counts = self.pending_counts(tx, schedule["user_id"])
                    if counts["total"] >= max_pending or counts["owned"] >= max_user_pending:
                        continue
                    tx.cursor.execute(
                        "SELECT task_id FROM bot_task WHERE account_id=%s AND schedule_id=%s AND state NOT IN ('succeeded','failed','cancelled') LIMIT 1",
                        (tx.account_id, schedule["schedule_id"]),
                    )
                    if tx.cursor.fetchone():
                        continue
                    scheduled_at = due_run(
                        schedule["rule"],
                        schedule["timezone"],
                        schedule["next_run_at"],
                        now,
                        schedule["created_at"],
                    )
                    task = self.create_task(
                        tx,
                        {
                            "kind": "chat",
                            "session_id": schedule["session_id"],
                            "request_id": new_id("schedule"),
                            "input": schedule["task_input"],
                            "accepted_authz": schedule["accepted_authz"],
                            "schedule_id": schedule["schedule_id"],
                            "scheduled_at": scheduled_at,
                            "schedule_revision": schedule["revision"],
                            "schedule_snapshot": {
                                key: schedule[key]
                                for key in ("rule", "timezone", "condition", "baseline")
                            },
                        },
                    )
                    next_run_at = next_run(
                        schedule["rule"], schedule["timezone"], now, schedule["created_at"]
                    )
                    tx.update(
                        "bot_schedule",
                        {"schedule_id": schedule["schedule_id"]},
                        {
                            "next_run_at": next_run_at,
                            "state": "active" if next_run_at else "completed",
                        },
                    )
                    created.append(task)
            except psycopg2.Error as error:
                if error.pgcode in {"55P03", "40001", "40P01"}:
                    continue
                raise
        return created

    def update_schedule_baseline(self, tx: UnitOfWork, task: dict, baseline: dict) -> None:
        if task.get("schedule_id"):
            tx.update(
                "bot_schedule",
                {"schedule_id": task["schedule_id"], "revision": task["schedule_revision"]},
                {"baseline": baseline},
            )

    def prune_events(self, before: datetime | str, limit: int = 1000) -> int:
        with self.transaction(None) as tx:
            tx.cursor.execute(
                "SELECT account_id,task_id,event_seq FROM bot_event WHERE created_at<%s ORDER BY created_at LIMIT %s",
                (as_utc(before), limit),
            )
            rows = [dict(row) for row in tx.cursor.fetchall()]
        removed = 0
        for row in rows:
            with self.transaction(row["account_id"]) as tx:
                removed += tx.delete(
                    "bot_event", {"task_id": row["task_id"], "event_seq": row["event_seq"]}
                )
        return removed

    def delete_unreferenced_file(
        self,
        account_id: str,
        file_id: str,
        *,
        before: datetime | str,
        has_memory_reference: Callable[[UnitOfWork, dict], bool],
    ) -> dict | None:
        """Remove eligible SQL metadata and return the immutable object key for GC."""
        with self.transaction(account_id) as tx:
            return self.delete_file_metadata(
                tx, file_id, before=before, has_memory_reference=has_memory_reference
            )

    def delete_file_metadata(
        self,
        tx: UnitOfWork,
        file_id: str,
        *,
        before: datetime | str,
        has_memory_reference: Callable[[UnitOfWork, dict], bool],
        reference_scan_limit: int | None = None,
    ) -> dict | None:
        """Check references under the file lock in the caller's scoped transaction."""
        file = tx.get("bot_file", {"file_id": file_id}, for_update=True)
        if file is None or file["updated_at"] >= as_utc(before):
            return None
        metadata = file["metadata"]
        if (
            metadata.get("memory_sources")
            or metadata.get("backup_refs")
            or has_memory_reference(tx, file)
        ):
            return None
        if (
            metadata.get("backup_protected_until")
            and as_utc(metadata["backup_protected_until"]) > tx.now()
        ):
            return None
        task = tx.get("bot_task", {"task_id": file["task_id"]})
        if task and task["state"] not in TERMINAL:
            return None
        for table, columns in FILE_REFERENCE_COLUMNS.items():
            fields = (*columns, "file_id") if table == "bot_file" else columns
            for record in tx._matching_reference_rows(
                table, columns, [file_id], limit=reference_scan_limit, fields=fields
            ):
                if table == "bot_file" and record["file_id"] == file_id:
                    continue
                if file_id in _file_references(table, record):
                    return None
        tx.delete("bot_file", {"file_id": file_id})
        return file
