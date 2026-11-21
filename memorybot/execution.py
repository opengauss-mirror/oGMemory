"""Shared checkpoint, budget, permission and streaming state for one leased task."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

from core.operation_deadline import operation_deadline
from memorybot.gateway import OutcomeUnknownError, WorkPendingError
from memorybot.models import BotError, jsonable, new_id
from memorybot.store import LeaseLost
from memorybot.templates import TemplateRegistry


class _CancelledError(Exception):
    pass


class Execution:
    def __init__(self, worker, task):
        self.worker, self.task = (worker, task)
        self.store, self.gateway, self.files = (worker.store, worker.gateway, worker.files)
        self.config = worker.config
        self.identity = (task["account_id"], task["task_id"], task["owner_id"], task["attempt"])
        self.account, self.user = (task["account_id"], task["user_id"])
        self.number, self.model_count, self.tool_count = (0, 0, 0)
        self.operation_id = None
        self.partial = task.get("result")
        self.lost = threading.Event()
        self.deadline = None
        self.active_template = None
        self.template_scope = None

    def current(self):
        if self.lost.is_set():
            raise LeaseLost("execution lease transferred")
        if self.deadline and datetime.now(UTC) >= self.deadline:
            raise BotError("BUDGET_EXHAUSTED", "任务达到运行时限", details={"stage": "runtime"})
        with self.store.execution(*self.identity) as tx:
            if tx.task["cancel_requested"]:
                raise _CancelledError()
        ctx = self.gateway.context(self.task)
        if self.active_template:
            TemplateRegistry.authorize(
                self.active_template,
                ctx,
                self.template_scope,
                resource_reader=lambda uri: self.gateway.read_memory(ctx, uri),
            )
        return ctx

    def event(self, kind, payload):
        with self.store.execution(*self.identity) as tx:
            self.store.append_event(tx, self.task["task_id"], kind, payload)

    def step(self, kind, value, function, *, atomic=False, operation_id=None):
        self.current()
        # Each attempt rebuilds the same step sequence to reuse saved checkpoints.
        self.number += 1
        step_key = {"task_id": self.task["task_id"], "step_no": self.number}
        with self.store.execution(*self.identity) as tx:
            step = tx.get("bot_step", step_key)
            # Checkpoint reuse must respect current source access and forget records.
            if step and value.get("authorization_fingerprint") != step["input"].get(
                "authorization_fingerprint"
            ):
                raise BotError("ACCESS_CHANGED", "来源授权或遗忘范围已变化，请重新发起任务", 409)
            if step and step["state"] == "succeeded":
                return step["output"]
            if step is None:
                step = tx.insert(
                    "bot_step",
                    {
                        **step_key,
                        "kind": kind,
                        "operation_id": operation_id or new_id("op"),
                        "input": jsonable(value),
                    },
                )
            awaiting_confirmation = step["state"] == "waiting_confirmation"
            self.operation_id = step["operation_id"]
            step = tx.update("bot_step", step_key, {"state": "running"})
            progress_message = None
            if kind == "model":
                progress_message = "正在生成回复"
            elif kind == "file_publish":
                progress_message = "正在生成成果文件"
            elif kind == "retrieve" and "messages" in value:
                progress_message = "正在检索记忆"
            elif kind == "retrieve" and "file_id" in value:
                progress_message = "正在读取资料"
            elif kind == "tool":
                progress_message = {
                    "search_memory": "正在查询记忆",
                    "read_memory": "正在读取记忆",
                    "list_schedules": "正在查询计划",
                    "create_schedule": "正在创建计划",
                    "update_schedule": "正在更新计划",
                    "memory_action": "正在处理记忆",
                }.get(value.get("tool"), "正在处理资料")
            if progress_message:
                self.store.append_event(
                    tx,
                    self.task["task_id"],
                    "progress",
                    {"phase": kind, "message": progress_message},
                )
            if kind in {"model", "tool"}:
                budget = tx.get("bot_step", {"task_id": self.task["task_id"], "step_no": 1})
                usage = dict(budget["output"])
                counter = "model_calls" if kind == "model" else "tool_calls"
                limit = (
                    self.config.max_model_calls if kind == "model" else self.config.max_tool_calls
                )
                usage[counter] = usage.get(counter, 0) + 1
                consumed = usage[counter]
                scope = value.get("budget_scope") if kind == "model" else None
                if scope:
                    scoped = dict(usage.get("model_calls_by_scope", {}))
                    scoped[scope] = scoped.get(scope, 0) + 1
                    usage["model_calls_by_scope"] = scoped
                    consumed = scoped[scope]
                    if scope != "handbook:merge":
                        limit = self.config.handbook_max_model_calls_per_file
                if consumed > limit:
                    raise BotError(
                        "BUDGET_EXHAUSTED",
                        "达到本次调用额度，请缩小资料范围后继续",
                        details={"scope": scope, "limit": limit, "used": consumed - 1},
                    )
                tx.update(
                    "bot_step", {"task_id": self.task["task_id"], "step_no": 1}, {"output": usage}
                )
            self.store.step_progress(tx, step, "running", self.task["attempt"])
        started = time.monotonic()
        try:
            if atomic:
                # Commit writes and their receipts together for reliable replay.
                with self.store.execution(*self.identity) as tx:
                    with operation_deadline(self.remaining()):
                        output = function(
                            step["input"], step["operation_id"], awaiting_confirmation, tx
                        )
                    completed = tx.update(
                        "bot_step",
                        step_key,
                        {
                            "state": "succeeded",
                            "output": jsonable(output),
                            "external_receipt": jsonable(output),
                        },
                    )
                    self.record_time(tx, started)
                    self.store.step_progress(tx, completed, "succeeded", self.task["attempt"])
            else:
                # Release the task lock during external calls, then recheck the lease.
                with operation_deadline(self.remaining()):
                    output = function(step["input"], step["operation_id"], awaiting_confirmation)
                with self.store.execution(*self.identity) as tx:
                    completed = tx.update(
                        "bot_step",
                        step_key,
                        {
                            "state": "succeeded",
                            "output": jsonable(output),
                            "external_receipt": (
                                jsonable(output) if kind == "memory_submit" else None
                            ),
                        },
                    )
                    self.record_time(tx, started)
                    self.store.step_progress(tx, completed, "succeeded", self.task["attempt"])
            return jsonable(output)
        except OutcomeUnknownError:
            with self.store.execution(*self.identity) as tx:
                updated = tx.update("bot_step", step_key, {"state": "waiting_confirmation"})
                self.record_time(tx, started)
                self.store.step_progress(tx, updated, "waiting_confirmation", self.task["attempt"])
            raise
        except WorkPendingError:
            with self.store.execution(*self.identity) as tx:
                updated = tx.update("bot_step", step_key, {"state": "pending"})
                self.record_time(tx, started)
                self.store.step_progress(tx, updated, "retry_wait", self.task["attempt"])
            raise
        except LeaseLost:
            raise
        except _CancelledError:
            with self.store.execution(*self.identity) as tx:
                updated = tx.update("bot_step", step_key, {"state": "cancelled"})
                self.record_time(tx, started)
                self.store.step_progress(tx, updated, "cancelled", self.task["attempt"])
            raise
        except Exception as exc:
            with self.store.execution(*self.identity) as tx:
                updated = tx.update("bot_step", step_key, {"state": "failed"})
                self.record_time(tx, started)
                code = exc.code if isinstance(exc, BotError) else "STEP_FAILED"
                self.store.step_progress(
                    tx, updated, "failed", self.task["attempt"], error_code=code
                )
            raise

    def remaining(self):
        return (
            max(0.001, (self.deadline - datetime.now(UTC)).total_seconds())
            if self.deadline
            else None
        )

    def record_time(self, tx, started):
        if self.number == 1:
            return
        key = {"task_id": self.task["task_id"], "step_no": 1}
        budget = tx.get("bot_step", key)
        usage = dict(budget["output"])
        usage["used_seconds"] = usage.get("used_seconds", 0) + time.monotonic() - started
        tx.update("bot_step", key, {"output": usage})

    def run(self, handlers):
        # Step 1 carries elapsed time and call counts across task attempts.
        budget_usage = self.step(
            "retrieve",
            {"used_seconds": 0, "model_calls": 0, "tool_calls": 0},
            lambda value, *_: value,
        )
        timeout_seconds = (
            self.config.task_timeout_seconds
            if self.task["kind"] == "chat" and self.task["input"].get("template_id") != "handbook"
            else self.config.file_timeout_seconds
        )
        self.deadline = datetime.now(UTC) + timedelta(
            seconds=max(0, timeout_seconds - budget_usage["used_seconds"])
        )
        handlers[self.task["kind"]](self)

    def rows(self, table, filters=None, **kwargs):
        return self.store.list(self.account, table, filters, self.user, **kwargs)

    def row(self, table, filters):
        row = self.store.get(self.account, table, filters, self.user)
        if row is None:
            raise BotError("NOT_FOUND", "关联对象不可访问", 404)
        return row

    def excluded(self):
        excluded = []
        for state in ("pending", "effective"):
            offset = 0
            while True:
                page = self.rows(
                    "bot_forget", {"state": state}, order_by="created_at DESC, forget_id DESC",
                    limit=1000, offset=offset,
                )
                excluded.extend(page)
                if len(page) < 1000:
                    break
                offset += len(page)
        return excluded

    def model(self, messages, *, tools=None, progress=None, stream=True):
        if not (progress or {}).get("budget_scope"):
            self.model_count += 1
            if self.model_count > self.config.max_model_calls:
                raise BotError("BUDGET_EXHAUSTED", "达到模型调用次数上限")
        estimate = (
            sum(len(str(row.get("content", "")).encode("utf-8")) for row in messages) // 3 + 1
        )
        if estimate > self.config.max_input_tokens:
            raise BotError("BUDGET_EXHAUSTED", "资料超过本次上下文额度，请缩小资料范围")
        pending = []
        pending_size, received_bytes = (0, 0)
        last_flush = time.monotonic() - 0.1
        text_emitted = False
        visible_text = ""

        def flush():
            nonlocal pending_size, last_flush, text_emitted, visible_text
            if pending:
                self.current()
                chunk = "".join(pending)
                self.event("answer.delta", {"attempt": self.task["attempt"], "text": chunk})
                visible_text += chunk
                self.partial = {"text": visible_text, "citations": [], "file_ids": []}
                pending.clear()
                pending_size = 0
                last_flush = time.monotonic()
                text_emitted = True

        def receive(text):
            nonlocal pending_size, received_bytes
            received_bytes += len(text.encode("utf-8"))
            if received_bytes > min(524288, self.config.max_output_tokens * 12):
                raise BotError("BUDGET_EXHAUSTED", "模型输出超过本次额度")
            if not stream:
                return
            for offset in range(0, len(text), 512):
                chunk = text[offset : offset + 512]
                pending.append(chunk)
                pending_size += len(chunk)
                if pending_size >= 512 or time.monotonic() - last_flush >= 0.1:
                    flush()

        result = self.step(
            "model",
            {"messages": messages, "tools": tools, **(progress or {})},
            lambda value, *_: self.gateway.complete(
                self.task,
                value["messages"],
                tools=value["tools"],
                max_output_tokens=self.config.max_output_tokens,
                on_text_delta=receive,
            ),
        )
        if len(result["text"].encode("utf-8")) > min(524288, self.config.max_output_tokens * 12):
            raise BotError("BUDGET_EXHAUSTED", "模型输出超过本次额度")
        if result["calls"]:
            # Reset the visible draft so the next model turn starts with a clean reply.
            self.partial = None
            if text_emitted:
                self.event(
                    "answer.reset",
                    {"attempt": self.task["attempt"], "replaced_attempt": self.task["attempt"]},
                )
        elif stream:
            if received_bytes == 0 and result["text"]:
                receive(result["text"])
            flush()
        return result

    def project(self, receipts):
        if self.task["attempt"] > 1:
            self.gateway.service.drain_outbox_sync(account_id=self.account)

        def check(value, *_):
            if not self.gateway.projection_ready(self.task, value["sources"]):
                raise WorkPendingError("索引正在处理")
            return {"confirmed": True}

        return self.step("memory_poll", {"sources": receipts}, check)

    def _memory_status(self, tx, status):
        self.store.set_memory_status(tx, self.task["task_id"], status)
        if self.task.get("parent_task_id"):
            self.store.set_memory_status(tx, self.task["parent_task_id"], status)

    def memory_status(self, status):
        with self.store.execution(*self.identity) as tx:
            self._memory_status(tx, status)

    def fail_related(self, tx):
        if self.task["kind"] == "memory_write":
            self._memory_status(tx, "failed")
            # A committed forget operation still governs retrieval after later failures.
            applied = next(
                (
                    step["output"]
                    for step in tx.list(
                        "bot_step", {"task_id": self.task["task_id"], "state": "succeeded"}
                    )
                    if (step.get("output") or {}).get("action") == "forget"
                ),
                None,
            )
            if applied:
                tx.update(
                    "bot_forget",
                    {"task_id": self.task["task_id"], "state": "pending"},
                    {"state": "effective", "source_refs": applied["source_refs"]},
                )
            else:
                tx.update(
                    "bot_forget",
                    {"task_id": self.task["task_id"], "state": "pending"},
                    {"state": "failed"},
                )
        elif self.task["kind"] == "ingest":
            tx.update("bot_file", {"task_id": self.task["task_id"]}, {"state": "failed"})
        elif self.task["kind"] == "delivery":
            tx.update("bot_delivery", {"task_id": self.task["task_id"]}, {"state": "failed"})

    def cancel_related(self, tx):
        if self.task["kind"] == "delivery":
            tx.update("bot_delivery", {"task_id": self.task["task_id"]}, {"state": "cancelled"})
        elif self.task["kind"] == "memory_write":
            self.fail_related(tx)
        elif self.task["kind"] == "ingest":
            tx.update("bot_file", {"task_id": self.task["task_id"]}, {"state": "failed"})
