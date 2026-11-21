"""Authenticated application operations for MemoryBot."""

from __future__ import annotations

import base64
import hashlib
import json
from urllib.parse import urlsplit

from psycopg2 import sql

from memorybot.models import BotError, ChatInput, Scope, digest, jsonable, new_id, public
from memorybot.scheduling import as_utc, next_run
from memorybot.store import TERMINAL


def parsed_mime_type(file):
    return (file.get("metadata") or {}).get("parsed_mime_type") or (
        "text/plain" if file.get("source_kind") == "url" else file["mime_type"]
    )


class BotService:
    def __init__(self, store, memory_service, files, config, *, gateway=None):
        self.store, self.memory, self.files, self.config = store, memory_service, files, config
        if gateway is None:
            from memorybot.gateway import MemoryGateway

            gateway = MemoryGateway(memory_service, self.resolve_task_context)
        self.gateway = gateway

    def authenticate(self, headers, *, agent_id="memorybot", session_id="memorybot"):
        auth = self.memory.get_auth_service()
        if not auth.role_control_active():
            raise BotError("DEPENDENCY_UNAVAILABLE", "请先启用用户 API Key 鉴权", 503)
        normalized = {k.lower(): v for k, v in headers.items()}
        bearer = normalized.get("authorization", "")
        key = normalized.get("x-api-key")
        if key and bearer.lower().startswith("bearer ") and key != bearer[7:].strip():
            raise BotError("UNAUTHENTICATED", "凭据不一致", 401)
        identity = auth.resolve_identity(dict(headers))
        if identity is None:
            raise BotError("UNAUTHENTICATED", "需要用户 API Key", 401)
        return self.memory.build_context(
            {"agentId": agent_id, "sessionId": session_id}, identity=identity
        )

    def resolve_task_context(self, task):
        account, user = task["account_id"], task["user_id"]
        record = next(
            (
                r
                for r in self.memory.get_key_manager().get_users(account)
                if r["user_id"] == user and r.get("status", "active") == "active"
            ),
            None,
        )
        if not record or not record.get("key"):
            raise BotError("PERMISSION_DENIED", "后台任务用户授权已失效", 403)
        authz = task.get("accepted_authz") or {}
        expected = f"user:{account}:{user}"
        if authz.get("credential_ref") != expected:
            raise BotError("PERMISSION_DENIED", "任务凭据归属不匹配", 403)
        ctx = self.authenticate(
            {"X-API-Key": record["key"], "X-Account-ID": account},
            agent_id=authz.get("agent_id", ""),
            session_id=authz.get("memory_session_id", task["task_id"]),
        )
        if (ctx.account_id, ctx.user_id) != (account, user):
            raise BotError("PERMISSION_DENIED", "任务用户归属已变化", 403)
        return ctx

    def snapshot(self, ctx, *, session=None, task_id=None, scope=None, tool_ids=None):
        scope = scope or (session or {}).get("scope", Scope().model_dump())
        return {
            "agent_id": (session or {}).get("agent_id", ctx.agent_id),
            "memory_session_id": (session or {}).get("session_id", task_id),
            "session_generation": (session or {}).get("session_generation", 1),
            "resource_ids": scope.get("resource_ids", []),
            "tool_ids": tool_ids or [],
            "config_revision": 1,
            "credential_ref": f"user:{ctx.account_id}:{ctx.user_id}",
        }

    def validate_scope(self, ctx, scope):
        for uri in scope.get("resource_ids", []):
            self.gateway.read_memory(ctx, uri)

    def require(self, tx, table, identity, *, lock=False):
        row = tx.get(table, identity, for_update=lock)
        if row is None:
            raise BotError(
                "NOT_FOUND",
                "对象不存在或当前用户不可访问",
                404,
                {"object_type": table.removeprefix("bot_")},
            )
        return row

    def task_public(self, task, tx=None):
        if tx is None:
            with self.store.transaction(task["account_id"], task["user_id"]) as tx:
                return self.task_public(task, tx)
        task = dict(task)
        task["steps"] = self.store.task_steps(
            task["account_id"], task["task_id"], task["user_id"], tx=tx
        )
        events = tx.list(
            "bot_event", {"task_id": task["task_id"]}, order_by="event_seq ASC", limit=1
        )
        task["event_floor_seq"] = (
            events[0]["event_seq"] if events else task.get("next_event_seq", 1)
        )
        if task["state"] == "waiting_confirmation":
            steps = tx.list(
                "bot_step",
                {"task_id": task["task_id"], "state": "waiting_confirmation"},
                order_by="step_no ASC",
                limit=1,
            )
            operation_id = (
                steps[0]["operation_id"]
                if steps
                else (task.get("error") or {}).get("details", {}).get("operation_id")
            )
            task["confirmation"] = {
                "operation_id": operation_id,
                "reason": (task.get("error") or {}).get("message", "等待核对外部结果"),
                "actions": ["recheck", "cancel"],
            }
        return public("task", task)

    def mutation(self, ctx, operation, request_id, data, action):
        if (
            not request_id
            or len(request_id) > 64
            or any(ord(c) < 33 or ord(c) > 126 for c in request_id)
        ):
            raise BotError(
                "INVALID_ARGUMENT",
                "Idempotency-Key 需要 1–64 个可打印 ASCII 字符",
                details={"field": "Idempotency-Key"},
            )

        def execute(tx):
            status, result = action(tx)
            return status, {"request_id": request_id, "data": jsonable(result)}

        return self.store.idempotent(
            ctx.account_id,
            ctx.user_id,
            operation,
            request_id,
            digest({"operation": operation, "body": data}),
            execute,
        )

    def check_capacity(self, tx, user_id):
        tx.advisory_lock("memorybot.admission")
        counts = self.store.pending_counts(tx, user_id)
        if (
            counts["total"] >= self.config.max_pending
            or counts["owned"] >= self.config.max_user_pending
        ):
            raise BotError("CAPACITY_LIMIT", "任务队列已满", 429, {"retry_after_seconds": 5})

    def create_session(self, ctx, tx, data):
        session_id = new_id("s")
        ctx = self._agent_context(ctx, data["agent_id"], session_id)
        self.validate_scope(ctx, data["scope"])
        row = tx.insert(
            "bot_session",
            {
                **data,
                "session_id": session_id,
                "account_id": ctx.account_id,
                "user_id": ctx.user_id,
            },
        )
        return 201, public("session", row)

    def _agent_context(self, ctx, agent_id, session_id):
        from server.auth import ResolvedIdentity

        return self.memory.build_context(
            {"agentId": agent_id, "sessionId": session_id},
            identity=ResolvedIdentity(
                role=ctx.role, account_id=ctx.account_id, user_id=ctx.user_id
            ),
        )

    def update_session(self, ctx, tx, session_id, data):
        session = self.require(tx, "bot_session", {"session_id": session_id}, lock=True)
        if session["revision"] != data["expected_revision"]:
            raise BotError("REVISION_CONFLICT", "会话已更新，请刷新后操作", 409)
        changes = {key: data[key] for key in ("title", "archived") if data.get(key) is not None}
        changes["revision"] = session["revision"] + 1
        row = tx.update("bot_session", {"session_id": session_id}, changes)
        return 200, public("session", row)

    def prepare_input(self, ctx, tx, session, data):
        data = ChatInput.model_validate(data).model_dump(mode="json")
        ctx = self._agent_context(ctx, session["agent_id"], session["session_id"])
        self.validate_scope(ctx, session["scope"])
        documents = []
        for file_id in data["file_ids"]:
            item = self.require(tx, "bot_file", {"file_id": file_id}, lock=True)
            if item["state"] != "ready":
                raise BotError("FILE_NOT_READY", "资料正在处理", 409, {"file_id": file_id})
            self.validate_scope(ctx, item["scope"])
            documents.append({"mime_type": parsed_mime_type(item)})
        if data["template_id"]:
            self.templates(ctx).validate(
                data["template_id"],
                data["text"],
                documents,
                data["output_format"],
                ctx=ctx,
                scope=session["scope"],
                resource_reader=lambda uri: self.gateway.read_memory(ctx, uri),
            )
        data["delivery"]["target_id"] = session["session_id"]
        return data, ctx

    def create_chat(self, ctx, tx, session_id, data, request_id):
        self.check_capacity(tx, ctx.user_id)
        session = self.require(tx, "bot_session", {"session_id": session_id}, lock=True)
        if session["archived"]:
            session = tx.update(
                "bot_session",
                {"session_id": session_id},
                {"archived": False, "revision": session["revision"] + 1},
            )
        data, ctx = self.prepare_input(ctx, tx, session, data)
        row = self.store.create_task(
            tx,
            {
                "task_id": new_id("t"),
                "user_id": ctx.user_id,
                "session_id": session_id,
                "kind": "chat",
                "request_id": request_id,
                "input": data,
                "accepted_authz": self.snapshot(
                    ctx,
                    session=session,
                    tool_ids=[data["template_id"]] if data["template_id"] else [],
                ),
            },
        )
        return 202, self.task_public(row, tx)

    def get_object(self, ctx, kind, identity):
        table = "bot_" + kind
        with self.store.transaction(ctx.account_id, ctx.user_id) as tx:
            row = self.require(tx, table, identity)
            return self.task_public(row, tx) if kind == "task" else public(kind, row)

    def page(
        self, ctx, kind, filters, *, limit=20, cursor=None, history=False, history_order="asc"
    ):
        if not 1 <= limit <= 100:
            raise BotError("INVALID_ARGUMENT", "limit 范围为 1–100", details={"field": "limit"})
        if history_order not in {"asc", "desc"}:
            raise BotError(
                "INVALID_ARGUMENT",
                "history_order 请选择 asc 或 desc",
                details={"field": "history_order"},
            )
        table = "bot_" + kind
        pk = kind + "_id"
        fingerprint = digest([ctx.account_id, ctx.user_id, kind, filters, history, history_order])
        position = None
        if cursor:
            try:
                page_cursor = json.loads(
                    base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
                )
                if page_cursor["scope"] != fingerprint:
                    raise ValueError("cursor scope")
                position = page_cursor["position"]
                if history:
                    if type(position) is not int or not 1 <= position < 2**63:
                        raise ValueError("cursor sequence")
                else:
                    if not isinstance(position, list) or len(position) != 2:
                        raise ValueError("cursor position")
                    timestamp, identifier = position
                    if not isinstance(timestamp, str) or not isinstance(identifier, str):
                        raise ValueError("cursor position types")
                    if not 1 <= len(identifier) <= 64 or "\0" in identifier:
                        raise ValueError("cursor identifier")
                    position = (as_utc(timestamp), identifier)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise BotError(
                    "INVALID_ARGUMENT", "分页游标与查询不匹配", details={"field": "cursor"}
                ) from exc
        clauses = [sql.SQL("account_id=%s"), sql.SQL("user_id=%s")]
        params = [ctx.account_id, ctx.user_id]
        allowed = {
            "session_id",
            "schedule_id",
            "task_id",
            "source_task_id",
            "kind",
            "state",
            "purpose",
        }
        for name, value in filters.items():
            if value is None:
                continue
            if name == "archived" and kind == "session":
                clauses.append(sql.SQL("archived=%s"))
            elif name == "q" and kind == "session":
                if len(value) > 128:
                    raise BotError("INVALID_ARGUMENT", "搜索文字最多128字", details={"field": "q"})
                clauses.append(sql.SQL("POSITION(lower(%s) IN lower(title)) > 0"))
            elif name in allowed:
                clauses.append(sql.SQL("{}=%s").format(sql.Identifier(name)))
            else:
                raise BotError("INVALID_ARGUMENT", "查询字段无效", details={"field": name})
            params.append(value)
        time_column = "updated_at" if kind == "session" else "created_at"
        order = (
            sql.SQL("seq DESC" if history_order == "desc" else "seq ASC")
            if history
            else sql.SQL("{} DESC, {} DESC").format(sql.Identifier(time_column), sql.Identifier(pk))
        )
        if position is not None:
            if history:
                clauses.append(sql.SQL("seq<%s" if history_order == "desc" else "seq>%s"))
                params.append(position)
            else:
                clauses.append(
                    sql.SQL("({}, {}) < (%s, %s)").format(
                        sql.Identifier(time_column), sql.Identifier(pk)
                    )
                )
                params.extend(position)
        with self.store.transaction(ctx.account_id, ctx.user_id) as tx:
            query = sql.SQL("SELECT * FROM {} WHERE {} ORDER BY {} LIMIT %s").format(
                sql.Identifier(table), sql.SQL(" AND ").join(clauses), order
            )
            tx.cursor.execute(query, [*params, limit + 1])
            rows = [dict(r) for r in tx.cursor.fetchall()]
            next_cursor = None
            if len(rows) > limit:
                rows = rows[:limit]
                last = rows[-1]
                position = last["seq"] if history else [jsonable(last[time_column]), last[pk]]
                next_cursor = (
                    base64.urlsafe_b64encode(
                        json.dumps({"scope": fingerprint, "position": position}).encode()
                    )
                    .decode()
                    .rstrip("=")
                )
            if history and history_order == "desc":
                rows.reverse()
            items = [
                self.task_public(row, tx) if kind == "task" else public(kind, row) for row in rows
            ]
        return {"items": items, "next_cursor": next_cursor}

    def cancel_task(self, ctx, tx, task_id, *, operation_id=None, decision="cancel"):
        # Serialize parent-first cancellation with worker transactions that update the parent.
        tx.advisory_lock("memorybot.admission")
        task = self.require(tx, "bot_task", {"task_id": task_id}, lock=True)
        if task["state"] in TERMINAL:
            return 200, self.task_public(task, tx)
        if operation_id is not None:
            confirmation = self.task_public(task, tx)["confirmation"]
            if not confirmation or confirmation["operation_id"] != operation_id:
                raise BotError(
                    "TASK_STATE_CONFLICT", "待确认操作已变化", 409, {"state": task["state"]}
                )
        if decision == "recheck":
            tx.update(
                "bot_task", {"task_id": task_id}, {"state": "queued", "next_attempt_at": None}
            )
        else:
            self._request_cancel(tx, task)
            tx.cursor.execute(
                "SELECT * FROM bot_task WHERE account_id=%s AND user_id=%s "
                "AND parent_task_id=%s AND state NOT IN %s ORDER BY task_id FOR UPDATE",
                (task["account_id"], task["user_id"], task_id, tuple(sorted(TERMINAL))),
            )
            for row in tx.cursor.fetchall():
                child = dict(row)
                state = self._request_cancel(tx, child)
                if state != "cancelled":
                    self.store.append_event(
                        tx, child["task_id"], "task.updated", {"state": state, "confirmation": None}
                    )
        task = self.require(tx, "bot_task", {"task_id": task_id})
        self.store.append_event(
            tx, task_id, "task.updated", {"state": task["state"], "confirmation": None}
        )
        return 202, self.task_public(self.require(tx, "bot_task", {"task_id": task_id}), tx)

    def _request_cancel(self, tx, task):
        state = task["state"]
        changes = {"cancel_requested": True}
        if state == "waiting_confirmation":
            changes.update(state="queued", next_attempt_at=None)
        tx.update("bot_task", {"task_id": task["task_id"]}, changes)
        if state in {"queued", "retry_wait"}:
            self.store._terminal(tx, task, "cancelled", task.get("result"), None)
            return "cancelled"
        return changes.get("state", state)

    def create_file(self, ctx, tx, data, request_id, *, upload=None):
        self.check_capacity(tx, ctx.user_id)
        self.validate_scope(ctx, data["scope"])
        task_id, file_id = new_id("t"), (upload or {}).get("file_id", new_id("f"))
        task = self.store.create_task(
            tx,
            {
                "task_id": task_id,
                "user_id": ctx.user_id,
                "kind": "ingest",
                "request_id": request_id,
                "input": {"file_id": file_id, "scope": data["scope"]},
                "accepted_authz": self.snapshot(ctx, task_id=task_id, scope=data["scope"]),
            },
        )
        values = {
            "file_id": file_id,
            "task_id": task_id,
            "user_id": ctx.user_id,
            "purpose": "source",
            "scope": data["scope"],
            "source_kind": "upload" if upload else "url",
        }
        if upload:
            values.update({k: v for k, v in upload.items() if k != "file_id"})
        else:
            url = urlsplit(data["source_url"])
            if url.scheme not in {"http", "https"} or not url.hostname:
                raise BotError(
                    "INVALID_ARGUMENT", "请输入 HTTP(S) 网页地址", details={"field": "source_url"}
                )
            values.update(name=url.hostname, source_url=data["source_url"])
        item = tx.insert("bot_file", values)
        return 202, {"file": public("file", item), "task": self.task_public(task, tx)}

    def create_schedule(self, ctx, tx, data):
        session = self.require(tx, "bot_session", {"session_id": data["session_id"]}, lock=True)
        data = dict(data)
        data["task_input"], ctx = self.prepare_input(ctx, tx, session, data["task_input"])
        now = tx.now()
        due = next_run(data["rule"], data["timezone"], now, created_at=now)
        if due is None:
            raise BotError("INVALID_ARGUMENT", "计划执行时间须在未来", details={"field": "rule"})
        item = tx.insert(
            "bot_schedule",
            {
                **data,
                "user_id": ctx.user_id,
                "schedule_id": new_id("sc"),
                "next_run_at": due,
                "accepted_authz": self.snapshot(
                    ctx,
                    session=session,
                    tool_ids=(
                        [data["task_input"]["template_id"]]
                        if data["task_input"]["template_id"]
                        else []
                    ),
                ),
            },
        )
        return 201, public("schedule", item)

    def patch_schedule(self, ctx, tx, schedule_id, data):
        item = self.require(tx, "bot_schedule", {"schedule_id": schedule_id}, lock=True)
        if item["revision"] != data["expected_revision"]:
            raise BotError(
                "REVISION_CONFLICT",
                "计划已更新，请读取最新计划",
                409,
                {"revision": item["revision"]},
            )
        action = data["action"]
        valid = {
            "update": {"active", "paused"},
            "pause": {"active"},
            "resume": {"paused"},
            "cancel": {"active", "paused"},
        }
        if item["state"] not in valid[action]:
            raise BotError(
                "TASK_STATE_CONFLICT", "当前计划状态不支持此操作", 409, {"state": item["state"]}
            )
        changed = {k: v for k, v in data.items() if k not in {"action", "expected_revision"}}
        revision = item["revision"] + 1
        if changed.keys() == {"name"}:
            updated = tx.update(
                "bot_schedule", {"schedule_id": schedule_id}, {**changed, "revision": revision}
            )
            return 200, public("schedule", updated)
        next_item = {**item, **changed}
        session = self.require(tx, "bot_session", {"session_id": item["session_id"]}, lock=True)
        if "task_input" in changed:
            changed["task_input"], ctx = self.prepare_input(ctx, tx, session, changed["task_input"])
        next_state = {"pause": "paused", "resume": "active", "cancel": "cancelled"}.get(
            action, item["state"]
        )
        due = (
            next_run(
                next_item["rule"], next_item["timezone"], tx.now(), created_at=item["created_at"]
            )
            if next_state == "active"
            else None
        )
        if next_state == "active" and due is None:
            raise BotError("INVALID_ARGUMENT", "请选择未来执行时间", details={"field": "rule"})
        changed.update(state=next_state, revision=revision, next_run_at=due)
        if "condition" in changed:
            changed["baseline"] = {}
        for task in tx.list("bot_task", {"schedule_id": schedule_id}, limit=10000):
            if task["state"] in {"queued", "retry_wait"}:
                task = self.require(tx, "bot_task", {"task_id": task["task_id"]}, lock=True)
                if task["state"] in {"queued", "retry_wait"}:
                    self.store._terminal(tx, task, "cancelled", task.get("result"), None)
        return 200, public(
            "schedule", tx.update("bot_schedule", {"schedule_id": schedule_id}, changed)
        )

    def memory_action(self, ctx, tx, data, request_id):
        self.check_capacity(tx, ctx.user_id)
        if data.get("memory_uri") and data["action"] == "remember":
            forgotten = tx.get(
                "bot_forget",
                {"memory_uri": data["memory_uri"], "state": "effective"},
                for_update=True,
            )
            if forgotten is None:
                self.gateway.read_memory(ctx, data["memory_uri"])
        elif data.get("memory_uri"):
            memory = self.gateway.read_memory(ctx, data["memory_uri"])
            if not data["source_refs"]:
                data = {**data, "source_refs": memory.get("source_refs", [])[:100]}
        for ref in data["source_refs"]:
            if ref.get("file_id"):
                self.require(tx, "bot_file", {"file_id": ref["file_id"]}, lock=True)
            elif ref.get("task_id"):
                self.require(tx, "bot_task", {"task_id": ref["task_id"]})
            else:
                self.gateway.read_memory(ctx, ref["uri"])
        task_id = new_id("t")
        task = self.store.create_task(
            tx,
            {
                "task_id": task_id,
                "user_id": ctx.user_id,
                "kind": "memory_write",
                "request_id": request_id,
                "input": data,
                "memory_status": "pending",
                "accepted_authz": self.snapshot(ctx, task_id=task_id),
            },
        )
        forget_id = None
        if data["action"] == "forget":
            forget_id = new_id("fg")
            tx.insert(
                "bot_forget",
                {
                    "forget_id": forget_id,
                    "task_id": task_id,
                    "user_id": ctx.user_id,
                    "memory_uri": data["memory_uri"],
                    "source_refs": data["source_refs"],
                    "fact_scope": data["fact_scope"],
                    "operation_id": hashlib.sha256((task_id + ":forget").encode()).hexdigest(),
                },
            )
        return 202, {"task": self.task_public(task, tx), "forget_id": forget_id}

    def templates(self, ctx=None):
        from memorybot.templates import TemplateRegistry

        if ctx is not None and hasattr(self.gateway, "templates"):
            return self.gateway.templates(ctx)
        return TemplateRegistry()

    def template_list(self, ctx, scope=None):
        return self.templates(ctx).visible(
            ctx, scope, resource_reader=lambda uri: self.gateway.read_memory(ctx, uri)
        )
