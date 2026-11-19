"""MemoryBot HTTP contracts, authenticated file downloads and durable SSE."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from datetime import timedelta
from urllib.parse import quote

from flask import Blueprint, Response, g, jsonify, request, stream_with_context
from pydantic import ValidationError
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType
from werkzeug.wsgi import get_input_stream

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import Role
from memorybot import models
from memorybot.models import BotError, jsonable, new_id
from memorybot.store import TERMINAL, UPLOAD_OPERATION, Conflict, LeaseLost, NotFound
from retrieval.owner_scope import canonical_owner_space
from server.auth import AuthenticationError, AuthorizationError

logger = logging.getLogger("ogmem.memorybot.http")


def create_blueprint(get_service, *, inherit_upload_limit=True):
    bp = Blueprint("memorybot", __name__, static_folder="static", static_url_path="/bot/static")
    prefix = "/api/v1/bot"

    @bp.before_request
    def prepare():
        g.bot_request_id = request.headers.get("Idempotency-Key") or new_id("req")
        if request.path.startswith(prefix):
            svc = get_service()
            g.bot_ctx = svc.authenticate(
                dict(request.headers), agent_id=request.args.get("agent_id", "")
            )
        if request.path == prefix + "/files" and request.method == "POST":
            upload_limit = 51 * 1024 * 1024
            configured_limit = request.max_content_length if inherit_upload_limit else None
            limit = upload_limit if configured_limit is None else min(upload_limit, configured_limit)
            request.stream = get_input_stream(request.environ, max_content_length=limit)

    @bp.after_request
    def response_headers(response):
        response.headers["X-Request-ID"] = getattr(g, "bot_request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        if request.path.startswith("/bot"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'self'; base-uri 'self'"
            )
        return response

    def error(exc):
        if isinstance(exc, BotError):
            problem = exc
        elif isinstance(exc, ValidationError):
            first = exc.errors(include_input=False)[0]
            problem = BotError(
                "INVALID_ARGUMENT",
                first["msg"],
                details={"field": ".".join(str(k) for k in first["loc"])},
            )
        elif isinstance(exc, (AuthenticationError,)):
            problem = BotError("UNAUTHENTICATED", "用户凭据无效", 401)
        elif isinstance(exc, (AccessDeniedError, AuthorizationError, PermissionError)):
            problem = BotError("PERMISSION_DENIED", "当前用户无权执行此操作", 403)
        elif isinstance(exc, (NodeNotFoundError, NotFound, FileNotFoundError)):
            problem = BotError("NOT_FOUND", "对象不存在或不可访问", 404)
        elif isinstance(exc, UnsupportedMediaType):
            problem = BotError(
                "UNSUPPORTED_MEDIA_TYPE", "请使用受支持的请求格式", 415, {"field": "Content-Type"}
            )
        elif isinstance(exc, RequestEntityTooLarge):
            problem = BotError(
                "FILE_TOO_LARGE",
                "文件超过大小限制",
                413,
                {"max_size_bytes": 50 * 1024 * 1024},
            )
        elif isinstance(exc, LeaseLost):
            problem = BotError("TASK_STATE_CONFLICT", "任务执行状态已变化", 409)
        elif isinstance(exc, Conflict):
            code = "IDEMPOTENCY_CONFLICT" if "payload" in str(exc) else "TASK_STATE_CONFLICT"
            problem = BotError(code, "请求或对象状态冲突", 409)
        elif isinstance(exc, (ValueError, BadRequest)):
            problem = BotError("INVALID_ARGUMENT", "请求字段或格式无效")
        else:
            logger.exception(
                "memoryBot request failed request_id=%s",
                getattr(g, "bot_request_id", ""),
            )
            problem = BotError(
                "DEPENDENCY_UNAVAILABLE",
                "服务暂时不可用，请使用原请求标识重试",
                503,
                {"retry_after_seconds": 5},
            )
        response = jsonify(
            {
                "request_id": getattr(g, "bot_request_id", new_id("req")),
                "error": problem.payload(),
            }
        )
        response.status_code = problem.status
        if problem.status in {429, 503}:
            response.headers["Retry-After"] = str(problem.details.get("retry_after_seconds", 5))
        return response

    bp.register_error_handler(Exception, error)
    bp.register_error_handler(RequestEntityTooLarge, error)
    bp.register_error_handler(BadRequest, error)

    def ok(data, status=200):
        return jsonify({"request_id": g.bot_request_id, "data": jsonable(data)}), status

    def body(model):
        data = model.model_validate(request.get_json()).model_dump(mode="json")
        return data

    def mutate(data, callback):
        svc = get_service()
        request_id = request.headers.get("Idempotency-Key", "")
        status, payload, duplicate = svc.mutation(
            g.bot_ctx,
            request.method + ":" + request.path.removeprefix(prefix),
            request_id,
            data,
            callback,
        )
        response = jsonify(payload)
        response.status_code = status
        response.headers["X-Request-ID"] = payload["request_id"]
        target = payload["data"]
        task = target.get("task", target)
        if task.get("task_id") and status == 202:
            response.headers["Location"] = prefix + "/tasks/" + task["task_id"]
        return response

    def paging():
        return {
            "limit": int(request.args.get("limit", "20")),
            "cursor": request.args.get("cursor"),
        }

    def offset_page(fingerprint):
        args = paging()
        limit, token = args["limit"], args["cursor"]
        if not 1 <= limit <= 100:
            raise BotError("INVALID_ARGUMENT", "limit 范围为 1–100", details={"field": "limit"})
        if not token:
            return limit, 0
        try:
            cursor = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
            offset = cursor["offset"]
            if cursor["scope"] != fingerprint or type(offset) is not int or not 0 <= offset < 2**63:
                raise ValueError("invalid offset cursor")
        except (KeyError, TypeError, ValueError) as exc:
            raise BotError("INVALID_ARGUMENT", "分页游标无效", details={"field": "cursor"}) from exc
        return limit, offset

    def next_offset(fingerprint, offset, limit, has_more):
        if has_more:
            data = json.dumps({"scope": fingerprint, "offset": offset + limit}).encode()
            return base64.urlsafe_b64encode(data).decode().rstrip("=")
        return None

    def pages(kind, filters=None, history=False, history_order="asc"):
        return get_service().page(
            g.bot_ctx,
            kind,
            filters or {},
            history=history,
            history_order=history_order,
            **paging(),
        )

    @bp.get("/bot")
    @bp.get("/bot/")
    def home():
        return bp.send_static_file("index.html")

    @bp.get(prefix + "/config")
    def configuration():
        cfg = get_service().config
        return ok(
            {
                "enabled": cfg.enabled,
                "max_file_bytes": cfg.max_file_bytes,
            }
        )

    @bp.post(prefix + "/sessions")
    def create_session():
        data = body(models.SessionCreate)
        return mutate(data, lambda tx: get_service().create_session(g.bot_ctx, tx, data))

    @bp.get(prefix + "/agents")
    def agents():
        from server.memory_viewer import _visible_agents

        svc = get_service()
        values = sorted(set(_visible_agents(svc.memory, g.bot_ctx)))
        return ok(
            {
                "items": [{"agent_id": value} for value in values],
                "default_agent_id": (
                    "memorybot" if "memorybot" in values else next(iter(values), None)
                ),
            }
        )

    @bp.get(prefix + "/sessions")
    def sessions():
        archived = request.args.get("archived", "false").lower()
        if archived not in {"true", "false"}:
            raise BotError(
                "INVALID_ARGUMENT",
                "archived 请选择 true 或 false",
                details={"field": "archived"},
            )
        return ok(
            pages(
                "session",
                {
                    "q": request.args.get("q", "").strip() or None,
                    "archived": archived == "true",
                },
            )
        )

    @bp.patch(prefix + "/sessions/<session_id>")
    def update_session(session_id):
        data = body(models.SessionUpdate)
        return mutate(
            data,
            lambda tx: get_service().update_session(g.bot_ctx, tx, session_id, data),
        )

    @bp.get(prefix + "/sessions/<session_id>")
    def session_details(session_id):
        session = get_service().get_object(g.bot_ctx, "session", {"session_id": session_id})
        history = pages(
            "task",
            {"session_id": session_id, "kind": "chat"},
            history=True,
            history_order=request.args.get("history_order", "asc"),
        )
        return ok(
            {
                "session": session,
                "messages": history["items"],
                "next_cursor": history["next_cursor"],
            }
        )

    @bp.post(prefix + "/sessions/<session_id>/tasks")
    def create_task(session_id):
        data = body(models.ChatInput)
        return mutate(
            data,
            lambda tx: get_service().create_chat(g.bot_ctx, tx, session_id, data, g.bot_request_id),
        )

    @bp.get(prefix + "/tasks/<task_id>")
    def task_details(task_id):
        return ok(get_service().get_object(g.bot_ctx, "task", {"task_id": task_id}))

    @bp.post(prefix + "/tasks/<task_id>/cancel")
    def cancel(task_id):
        data = request.get_json(silent=True) or {}
        if data:
            raise BotError("INVALID_ARGUMENT", "取消请求体应为空对象")
        return mutate(data, lambda tx: get_service().cancel_task(g.bot_ctx, tx, task_id))

    @bp.post(prefix + "/tasks/<task_id>/confirm")
    def confirm(task_id):
        data = body(models.Confirmation)
        return mutate(data, lambda tx: get_service().cancel_task(g.bot_ctx, tx, task_id, **data))

    @bp.get(prefix + "/tasks/<task_id>/events")
    def events(task_id):
        svc, ctx = get_service(), g.bot_ctx
        after = int(request.args.get("after", "0"))
        last_event_id = request.headers.get("Last-Event-ID")
        if last_event_id:
            event_task_id, event_seq = last_event_id.rsplit(":", 1)
            if event_task_id != task_id or ("after" in request.args and int(event_seq) != after):
                raise BotError("INVALID_ARGUMENT", "事件游标与任务不匹配")
            after = int(event_seq)
        task = svc.get_object(ctx, "task", {"task_id": task_id})
        # A task snapshot provides a fresh cursor when older events are pruned.
        if after < task["event_floor_seq"] - 1:
            raise BotError(
                "EVENT_CURSOR_EXPIRED",
                "事件已归档，请读取任务快照",
                410,
                {
                    "snapshot_url": prefix + "/tasks/" + task_id,
                    "last_event_seq": task["last_event_seq"],
                    "event_floor_seq": task["event_floor_seq"],
                },
            )
        if after < 0 or after > task["last_event_seq"]:
            raise BotError("INVALID_ARGUMENT", "事件序号超出范围")

        @stream_with_context
        def generate():
            cursor, heartbeat, started = after, time.monotonic(), time.monotonic()
            while time.monotonic() - started < 60:
                rows = svc.store.events(
                    ctx.account_id, task_id, cursor, user_id=ctx.user_id, limit=200
                )
                for event in rows:
                    cursor = event["event_seq"]
                    payload = json.dumps(
                        jsonable(
                            {
                                "task_id": task_id,
                                "event_seq": cursor,
                                "payload": event["payload"],
                            }
                        ),
                        ensure_ascii=False,
                    )
                    yield f"id: {task_id}:{cursor}\nevent: {event['type']}\ndata: {payload}\n\n"
                current_task = svc.store.get_task(ctx.account_id, task_id, user_id=ctx.user_id)
                # Keep the stream open for memory updates after the reply completes.
                if (
                    current_task["state"] in TERMINAL
                    and current_task["memory_status"] not in {"pending", "stored"}
                    and cursor >= current_task["next_event_seq"] - 1
                ):
                    return
                if len(rows) == 200:
                    continue
                if time.monotonic() - heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    heartbeat = time.monotonic()
                time.sleep(0.5)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @bp.post(prefix + "/files")
    def upload():
        svc = get_service()
        if request.mimetype == "multipart/form-data":
            from memorybot.files import parse_text

            request_id = request.headers.get("Idempotency-Key", "")
            svc.validate_request_id(request_id)
            item = request.files.get("file")
            if item is None or not item.filename:
                raise BotError("INVALID_ARGUMENT", "请选择文件", details={"field": "file"})
            name = item.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if not name or len(name) > 255:
                raise BotError("INVALID_ARGUMENT", "文件名无效", details={"field": "file"})
            content = item.stream.read(svc.config.max_file_bytes + 1)
            if len(content) > svc.config.max_file_bytes:
                raise RequestEntityTooLarge()
            parsed = parse_text(content, name)
            scope = models.Scope.model_validate(
                json.loads(request.form.get("scope", "{}"))
            ).model_dump(mode="json")
            data = {
                "scope": scope,
                "name": name,
                "mime_type": parsed["mime_type"],
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            svc.validate_scope(g.bot_ctx, scope)
            request_hash = models.digest({"operation": "POST:/files", "body": data})
            file_id = new_id("f")
            intent = {"operation": UPLOAD_OPERATION, "request_id": file_id}
            with svc.store.transaction(g.bot_ctx.account_id, g.bot_ctx.user_id) as tx:
                tx.advisory_lock("memorybot.admission")
                prior = tx.get(
                    "bot_request",
                    {"operation": "POST:/files", "request_id": request_id},
                )
                if prior:
                    if prior["request_hash"] != request_hash:
                        raise BotError("IDEMPOTENCY_CONFLICT", "请求标识已用于其他上传内容", 409)
                    response = jsonify(prior["response_body"])
                    response.status_code = prior["response_status"]
                    response.headers["Location"] = (
                        prefix + "/tasks/" + prior["response_body"]["data"]["task"]["task_id"]
                    )
                    return response
                svc.check_capacity(tx, g.bot_ctx.user_id)
                tx.insert(
                    "bot_request",
                    {
                        **intent,
                        "request_hash": data["sha256"],
                        "response_status": 202,
                        "response_body": {
                            "state": "pending",
                            "file": {
                                "file_id": file_id,
                                "user_id": g.bot_ctx.user_id,
                                "storage_key": f"files/{file_id}/{data['sha256']}/content",
                                "name": name,
                                "size_bytes": len(content),
                                "sha256": data["sha256"],
                                "mime_type": parsed["mime_type"],
                                "purpose": "source",
                                "scope": scope,
                            },
                        },
                        "expires_at": tx.now() + timedelta(hours=24),
                    },
                )
            saved = svc.files.publish(
                g.bot_ctx.account_id,
                file_id,
                content,
                name=name,
                mime_type=parsed["mime_type"],
            )

            def accept_upload(tx):
                reservation = tx.get("bot_request", intent, for_update=True)
                if reservation is None or reservation["expires_at"] <= tx.now():
                    raise BotError("UPLOAD_EXPIRED", "上传已过期，请重新提交文件", 409)
                result = svc.create_file(
                    g.bot_ctx,
                    tx,
                    data,
                    g.bot_request_id,
                    upload={"file_id": file_id, **saved},
                )
                tx.delete("bot_request", intent)
                return result

            return mutate(data, accept_upload)
        data = body(models.FileCreate)
        return mutate(data, lambda tx: svc.create_file(g.bot_ctx, tx, data, g.bot_request_id))

    @bp.get(prefix + "/files")
    def files():
        return ok(
            pages(
                "file",
                {k: request.args.get(k) for k in ("purpose", "state", "task_id")},
            )
        )

    @bp.get(prefix + "/files/<file_id>")
    def file_details(file_id):
        return ok(get_service().get_object(g.bot_ctx, "file", {"file_id": file_id}))

    @bp.get(prefix + "/files/<file_id>/content")
    def download(file_id):
        svc, ctx = get_service(), g.bot_ctx
        with svc.store.transaction(ctx.account_id, ctx.user_id) as tx:
            tx.advisory_lock("memorybot.admission")
            file = svc.require(tx, "bot_file", {"file_id": file_id}, lock=True)
            svc.validate_scope(ctx, file["scope"])
            if file["state"] != "ready":
                raise BotError("FILE_NOT_READY", "文件正在处理", 409, {"file_id": file_id})
            tx.update("bot_file", {"file_id": file_id}, {"updated_at": tx.now()})
        size = file["size_bytes"]
        start, end, status = 0, size - 1, 200
        requested = request.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()) or size == 0:
                raise BotError("RANGE_NOT_SATISFIABLE", "文件范围无效", 416, {"size_bytes": size})
            left, right = match.groups()
            start = int(left) if left else max(0, size - int(right))
            end = min(size - 1, int(right)) if left and right else size - 1
            if start > end or start >= size:
                raise BotError(
                    "RANGE_NOT_SATISFIABLE",
                    "文件范围超出长度",
                    416,
                    {"size_bytes": size},
                )
            status = 206

        def chunks():
            position = start
            while position <= end:
                chunk = svc.files.read(
                    ctx.account_id,
                    file["storage_key"],
                    offset=position,
                    size=min(512 * 1024, end - position + 1),
                )
                if not chunk:
                    raise OSError("file ended before registered length")
                position += len(chunk)
                yield chunk

        headers = {
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
            "ETag": '"' + file["sha256"] + '"',
            "Content-Disposition": "attachment; filename=download; filename*=UTF-8''"
            + quote(file["name"], safe=""),
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return Response(
            stream_with_context(chunks()),
            status=status,
            mimetype=file["mime_type"],
            headers=headers,
        )

    @bp.get(prefix + "/templates")
    def templates():
        svc = get_service()
        agent_id = request.args.get("agent_id", "memorybot")
        ctx = svc._agent_context(g.bot_ctx, agent_id, g.bot_ctx.session_id)
        scope = None
        if "memory_scope" in request.args or "resource_uri" in request.args:
            scope = models.Scope(
                memory_scope=request.args.get("memory_scope", "user"),
                resource_ids=request.args.getlist("resource_uri"),
            ).model_dump()
        q, format_ = request.args.get("q", "").strip(), request.args.get("output_format")
        if len(q) > 128 or (format_ and format_ not in {"markdown", "csv"}):
            raise BotError("INVALID_ARGUMENT", "模板查询字段无效")
        fingerprint = models.digest([ctx.account_id, ctx.user_id, ctx.agent_id, q, format_, scope])
        limit, offset = offset_page(fingerprint)
        rows = [
            r
            for r in svc.template_list(ctx, scope=scope)
            if q.lower() in (r["id"] + r["name"]).lower()
            and (not format_ or format_ in r["output"]["formats"])
        ]
        rows.sort(key=lambda row: row["id"])
        next_cursor = next_offset(fingerprint, offset, limit, offset + limit < len(rows))
        return ok({"items": rows[offset : offset + limit], "next_cursor": next_cursor})

    @bp.post(prefix + "/schedules")
    def create_schedule():
        data = body(models.ScheduleCreate)
        return mutate(data, lambda tx: get_service().create_schedule(g.bot_ctx, tx, data))

    @bp.get(prefix + "/schedules")
    def schedules():
        return ok(pages("schedule", {k: request.args.get(k) for k in ("state", "session_id")}))

    @bp.get(prefix + "/schedules/<schedule_id>")
    def schedule_details(schedule_id):
        return ok(get_service().get_object(g.bot_ctx, "schedule", {"schedule_id": schedule_id}))

    @bp.patch(prefix + "/schedules/<schedule_id>")
    def patch_schedule(schedule_id):
        data = models.SchedulePatch.model_validate(request.get_json()).model_dump(
            mode="json", exclude_unset=True
        )
        return mutate(
            data,
            lambda tx: get_service().patch_schedule(g.bot_ctx, tx, schedule_id, data),
        )

    @bp.get(prefix + "/schedules/<schedule_id>/runs")
    def schedule_runs(schedule_id):
        get_service().get_object(g.bot_ctx, "schedule", {"schedule_id": schedule_id})
        return ok(pages("task", {"schedule_id": schedule_id, "state": request.args.get("state")}))

    @bp.get(prefix + "/memories")
    def memories():
        svc, ctx = get_service(), g.bot_ctx
        scope = models.Scope(
            memory_scope=request.args.get("memory_scope", "user"),
        ).model_dump()
        q = request.args.get("q", "")
        if len(q) > 2048:
            raise BotError("INVALID_ARGUMENT", "查询文字过长", details={"field": "q"})
        fingerprint = models.digest([ctx.account_id, ctx.user_id, q, scope])
        limit, offset = offset_page(fingerprint)
        uri = request.args.get("uri")
        forgotten = svc.store.list(
            ctx.account_id,
            "bot_forget",
            {"state": "effective"},
            user_id=ctx.user_id,
            limit=100000,
        )
        hidden = {row["memory_uri"] for row in forgotten}
        if uri:
            record = svc.gateway.read_memory(ctx, uri)
            rows = [record] if offset == 0 and record["uri"] not in hidden else []
            has_more = False
        else:
            rows, has_more = svc.gateway.list_memory_page(ctx, q, scope, hidden, limit=limit, offset=offset)
        next_cursor = next_offset(fingerprint, offset, limit, has_more)
        items = []
        can_forget_agent_memory = ctx.role in {Role.ADMIN, Role.ROOT}
        for row in rows:
            owner_space = canonical_owner_space(row.get("owner_space") or "")
            item = {
                key: row.get(key)
                for key in ("uri", "title", "text", "source_refs", "updated_at", "category", "owner_space")
            }
            item["can_forget"] = row.get("category") != "profile" and (
                owner_space == ctx.user_space_name()
                or (can_forget_agent_memory and owner_space.startswith("agent:"))
            )
            items.append(item)
        return ok({"items": items, "next_cursor": next_cursor})

    @bp.post(prefix + "/memory-actions")
    def memory_action():
        data = body(models.MemoryAction)
        return mutate(
            data,
            lambda tx: get_service().memory_action(g.bot_ctx, tx, data, g.bot_request_id),
        )

    return bp
