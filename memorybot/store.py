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
from pathlib import Path
from typing import Any, cast

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from fs.sql_adapter.pool import SharedConnectionPool
from memorybot.scheduling import as_utc, due_run, next_run

TERMINAL = frozenset({"succeeded", "cancelled", "failed"})
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
TABLES = frozenset(
    re.findall(r"CREATE TABLE IF NOT EXISTS (bot_\w+)", SCHEMA_PATH.read_text(encoding="utf-8"))
)


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

    def _lock_references(self, table: str, values: dict) -> None:
        if self.account_id is None:
            return
        refs = _reference_ids(values)
        if table == "bot_file":
            refs.discard(values.get("file_id", ""))
        for file_id in sorted(refs):
            self.get("bot_file", {"file_id": file_id}, for_update=True)

    def insert(self, table: str, values: dict) -> dict:
        table, values = _table(table), self._scope(values)
        self._lock_references(table, values)
        columns = list(values)
        self.cursor.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
                _key(table),
                sql.SQL(", ").join(_key(key) for key in columns),
                sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            ),
            [_adapt(values[key]) for key in columns],
        )
        return dict(self.cursor.fetchone())

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
        self._lock_references(table, changes)
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
        lock_id = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big", signed=True)
        self.cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))


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
        conn = self.pool.get_connection()
        tx = UnitOfWork(self, conn, account_id, user_id)
        try:
            tx.cursor.execute("SET LOCAL app.account_id = %s", (account_id or "",))
            tx.cursor.execute(
                "SET LOCAL app.bot_worker = %s", ("on" if account_id is None else "off",)
            )
            yield tx
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            tx.cursor.close()
            self.pool.return_connection(conn)

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
            return
        previous = step.get("public_progress") or {}
        if previous.get("state") == state and previous.get("attempt") == attempt:
            return
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
        tx.update(
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
