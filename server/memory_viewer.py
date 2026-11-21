"""Memory viewer HTTP routes."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

from core.models import RequestContext, Role
from fs.sql_adapter.sql_context_fs import parse_uri
from server.auth import AuthorizationError, ControlPlaneDisabledError, build_admin_target_context
from server.mcp_resources import (
    MEMORY_TYPES,
    external_to_internal_uri,
    external_to_internal_uri_candidates,
    internal_to_external_uri,
    parse_ogmemory_uri,
)
from server.mcp_security import is_not_found_or_not_visible_error
from server.mcp_tools import (
    _list_memory_node_entries_page,
    _memory_list_cursor,
    _memory_list_cursor_key,
    _memory_node_summary,
    query_memory,
)
from skills.artifact import canonical_skill_uri

_TASK_STEPS = {
    "session_commit": (
        "将会话归档写入 ContextFS",
        "会话提交 → 生成会话归档 → 写入 ContextFS",
    ),
    "skill_processing": (
        "将 Agent Skill 文档写入 ContextFS",
        "Agent Skill 处理 → 生成 Skill 文档 → 写入 ContextFS",
    ),
    "reindex": (
        "将目录摘要写入 ContextFS",
        "记忆写入 → 更新目录摘要 → 写入 ContextFS",
    ),
}
_DEFAULT_TASK_STEPS = (
    "ContextWriter 将记忆节点写入 ContextFS",
    "记忆写入 → ContextWriter 构建节点 → 写入 ContextFS",
)
_STATIC_DIR = Path(__file__).with_name("memory_viewer_static")


def create_memory_viewer_blueprint(
    get_service: Callable,
    build_context: Callable,
    error_response: Callable,
) -> Blueprint:
    blueprint = Blueprint("memory_viewer", __name__)

    @blueprint.get("/memory-viewer/")
    def index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @blueprint.get("/memory-viewer/<path:asset_path>")
    def asset(asset_path: str):
        return send_from_directory(_STATIC_DIR, asset_path)

    @blueprint.get("/api/v1/memories")
    def list_memories():
        try:
            params = _context_params()
            context = build_context(params)
            service = get_service()
            requested_type = request.args.get("type", "").strip()
            if requested_type:
                memory_type, source, limit, cursor = _directory_request_params()
                target = context if params.get("agentId") else replace(context, agent_id="")
                page, has_more, next_cursor = _list_target_page(
                    service,
                    target,
                    memory_type=memory_type,
                    source=source,
                    limit=limit,
                    cursor=cursor,
                )
                for memory in page:
                    memory["context_uri"] = memory.get("context_uri") or external_to_internal_uri(
                        memory["uri"], target
                    )
            else:
                limit = _bounded_int(request.args.get("limit"), default=100, minimum=1, maximum=500)
                cursor = request.args.get("cursor", "").strip() or None
                page, has_more, next_cursor = _list_visible_page(
                    service, context, limit=limit, cursor=cursor
                )
            return jsonify(
                {
                    "memories": page,
                    "count": len(page),
                    "truncated": has_more,
                    "next_cursor": next_cursor,
                }
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/scope")
    def viewer_scope():
        try:
            service = get_service()
            context = build_context({})
            role_control_active = service.get_auth_service().role_control_active()
            agents = _visible_agents(service, context)
            default_agent_id = ""
            if not role_control_active:
                default_agent_id = str(getattr(service, "default_agent_id", "") or "")
            return jsonify(
                {
                    "account_id": context.account_id,
                    "user_id": context.user_id,
                    "agents": [{"agent_id": agent_id} for agent_id in agents],
                    "default_agent_id": default_agent_id,
                    "role_control_active": role_control_active,
                    "admin_mode": role_control_active and _role(context) in {Role.ROOT, Role.ADMIN},
                }
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/admin/tree")
    def admin_tree():
        try:
            service = get_service()
            actor = build_context({})
            _require_admin(service, actor)
            admin = service.get_tenant_admin_service()
            role = _role(actor)
            if role == Role.ROOT:
                accounts = admin.list_accounts(actor).get("accounts", [])
                account_ids = [str(row["account_id"]) for row in accounts]
                requested_account = request.args.get("accountId", "").strip()
                account_id = requested_account or (account_ids[0] if account_ids else "")
                if account_id and account_id not in account_ids:
                    account_id = account_ids[0] if account_ids else ""
            else:
                account_id = actor.account_id
                accounts = [{"account_id": account_id}]
            if not account_id:
                return jsonify({"accounts": accounts, "account_id": "", "users": [], "agents": []})
            users = admin.list_users(actor, account_id).get("users", [])
            agents = admin.list_agents(actor, account_id).get("agents", [])
            agents_by_user: dict[str, list[dict]] = {}
            for agent in agents:
                owner_user_id = str(agent.get("owner_user_id") or "")
                agents_by_user.setdefault(owner_user_id, []).append(
                    {"agent_id": str(agent["agent_id"])}
                )
            return jsonify(
                {
                    "accounts": accounts,
                    "account_id": account_id,
                    "agents": [
                        {
                            "agent_id": str(agent["agent_id"]),
                            "owner_user_id": str(agent.get("owner_user_id") or ""),
                        }
                        for agent in agents
                    ],
                    "users": [
                        {
                            "user_id": str(user["user_id"]),
                            "role": user.get("role", "user"),
                            "agents": agents_by_user.get(str(user["user_id"]), []),
                        }
                        for user in users
                    ],
                }
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/inventory")
    def memory_inventory():
        try:
            service = get_service()
            actor = build_context({})
            if service.get_auth_service().role_control_active() and _role(actor) in {Role.ROOT, Role.ADMIN}:
                target = _admin_account_context(service, actor, request.args)
                admin = service.get_tenant_admin_service()
                users = [str(row["user_id"]) for row in admin.list_users(actor, target.account_id)["users"]]
                agents = [str(row["agent_id"]) for row in admin.list_agents(actor, target.account_id)["agents"]]
            else:
                target = actor
                users = [actor.user_id]
                agents = _visible_agents(service, actor)
            return jsonify(_inventory_counts(service, target, users, agents))
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/admin/memories")
    def admin_memories():
        try:
            service = get_service()
            actor = build_context({})
            _require_admin(service, actor)
            target = _admin_target_context(service, actor)
            requested_type, requested_source, limit, cursor = _directory_request_params()
            page, has_more, next_cursor = _list_target_page(
                service,
                target,
                memory_type=requested_type,
                source=requested_source,
                limit=limit,
                cursor=cursor,
            )
            for memory in page:
                memory["context_uri"] = memory.get("context_uri") or external_to_internal_uri(
                    memory["uri"], target
                )
            return jsonify(
                {
                    "memories": page,
                    "count": len(page),
                    "truncated": has_more,
                    "next_cursor": next_cursor,
                }
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/admin/detail")
    def admin_detail():
        try:
            service = get_service()
            actor = build_context({})
            _require_admin(service, actor)
            target = _admin_target_context(service, actor)
            return jsonify(_read_detail(service, target, request.args.get("uri", "")))
        except Exception as exc:
            return error_response(exc)

    @blueprint.post("/api/v1/memory-viewer/search")
    def search_memories():
        try:
            params = request.get_json(force=True, silent=True) or {}
            if not isinstance(params, dict):
                raise ValueError("request body must be an object")
            service = get_service()
            actor = build_context({})
            target = _search_target_context(service, actor, build_context, params)
            session_id = (
                str(params.get("sessionId") or params.get("session_id") or "").strip() or None
            )
            if session_id:
                target = replace(target, session_id=session_id)
            return jsonify(
                query_memory(
                    service,
                    query=params.get("query", ""),
                    types=params.get("types"),
                    top_k=_search_top_k(params.get("top_k")),
                    score_threshold=params.get("score_threshold"),
                    session_id=session_id,
                    agent_id=target.agent_id,
                    request_context=target,
                )
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/metrics")
    def model_usage_metrics():
        try:
            service = get_service()
            actor = build_context({})
            days = _bounded_int(request.args.get("days"), default=14, minimum=1, maximum=90)
            account_id, owner_spaces = _metrics_target(service, actor)
            return jsonify(
                service.get_model_usage_metrics(
                    days=days,
                    account_id=account_id,
                    owner_spaces=owner_spaces,
                )
            )
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/monitoring")
    def monitoring_overview():
        try:
            service = get_service()
            actor = build_context({})
            account_id, owner_spaces = _metrics_target(service, actor)
            target = replace(
                actor,
                account_id=account_id,
                agent_id="",
                visible_owner_spaces=owner_spaces,
            )
            return jsonify(service.get_monitoring_overview(target))
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memory-viewer/tasks")
    def background_tasks():
        try:
            service = get_service()
            actor = build_context({})
            account_id, owner_spaces = _metrics_target(service, actor)
            limit = _bounded_int(request.args.get("limit"), default=200, minimum=1, maximum=200)
            outbox = service._get_outbox_store(service._get_context_fs())
            if outbox is None:
                return jsonify({"tasks": [], "total": 0, "capabilities": {"retry_failed": False}})
            can_retry = all(hasattr(outbox, name) for name in ("get_event", "retry_failed"))
            if hasattr(outbox, "list_recent"):
                events = outbox.list_recent(account_id, limit=limit)
            else:
                events = outbox.list_pending(account_id)[:limit]
            tasks = [
                _task_record(event)
                for _, event in events
                if _task_visible(event, owner_spaces)
            ]
            tasks.sort(key=lambda task: task["updated_at"], reverse=True)
            return jsonify({
                "tasks": tasks,
                "total": len(tasks),
                "capabilities": {"retry_failed": can_retry},
            })
        except Exception as exc:
            return error_response(exc)

    @blueprint.post("/api/v1/memory-viewer/tasks/<task_id>/retry")
    def retry_background_task(task_id: str):
        try:
            service = get_service()
            actor = build_context({})
            account_id, owner_spaces = _metrics_target(service, actor)
            outbox = service._get_outbox_store(service._get_context_fs())
            if outbox is None or not all(
                hasattr(outbox, name) for name in ("get_event", "retry_failed")
            ):
                return jsonify({"error": "当前存储后端不支持手动重试"}), 409
            event = outbox.get_event(account_id, task_id)
            if event is None or not _task_visible(event, owner_spaces):
                raise FileNotFoundError("task not found")
            if str(event.status).upper() != "FAILED":
                return jsonify({"error": "仅失败任务可以重新触发"}), 409
            retried = outbox.retry_failed(account_id, event)
            if retried is None:
                return jsonify({"error": "任务状态已变化，请刷新后重试"}), 409
            return jsonify({"task": _task_record(retried)})
        except Exception as exc:
            return error_response(exc)

    @blueprint.get("/api/v1/memories/detail")
    def memory_detail():
        try:
            context = build_context(_context_params())
            service = get_service()
            return jsonify(_read_detail(service, context, request.args.get("uri", "")))
        except Exception as exc:
            return error_response(exc)

    return blueprint


def _context_params() -> dict[str, str]:
    params: dict[str, str] = {}
    agent_id = request.args.get("agentId", "").strip()
    if agent_id:
        params["agentId"] = agent_id
    return params


def _bounded_int(value: str | None, *, default: int, minimum: int, maximum: int) -> int:
    parsed = default if value is None else int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return parsed


def _inventory_counts(service, context, users: list[str], agents: list[str]) -> dict:
    account = context.account_id
    fs = service._get_context_fs()
    raw = fs.count_memory_children(context) if hasattr(fs, "count_memory_children") else None
    directories: dict[str, int] = {}
    for owner_type, owner_ids in (("users", users), ("agents", agents)):
        for owner_id in owner_ids:
            owner_context = replace(
                context,
                user_id=owner_id if owner_type == "users" else context.user_id,
                agent_id=owner_id if owner_type == "agents" else "",
            )
            for spec in MEMORY_TYPES.values():
                if spec.public_type == "skill" or (spec.owner != owner_type[:-1] and spec.public_type != "pattern"):
                    continue
                uri = f"ctx://{account}/{owner_type}/{owner_id}/memories/{spec.directory}"
                if raw is not None:
                    count = raw.get(uri, 0)
                elif spec.public_type == "profile":
                    count = int(fs.exists(uri, owner_context))
                else:
                    count = len(fs.list_children(uri, owner_context))
                directories[uri] = count
    total_memories = sum(directories.values())
    total_skills = 0
    for agent_id in agents:
        target = replace(context, agent_id=agent_id)
        counts = {"generated": 0, "imported": 0}
        cursor = None
        while True:
            page = service.list_published_skills(target, limit=100, cursor=cursor)
            for skill in page.get("skills", []):
                counts[skill.get("source") or "generated"] += 1
            if not page.get("truncated"):
                break
            next_cursor = page.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                raise RuntimeError("skill listing returned an invalid pagination cursor")
            cursor = next_cursor
        for source, count in counts.items():
            directories[f"ctx://{account}/agents/{agent_id}/skills/{source}"] = count
            total_skills += count
    return {
        "account_id": account,
        "directories": directories,
        "memories": total_memories,
        "skills": total_skills,
    }


def _directory_request_params() -> tuple[str, str | None, int, str | None]:
    memory_type = request.args.get("type", "").strip()
    if memory_type not in MEMORY_TYPES:
        raise ValueError("type must reference a supported memory directory")
    source = request.args.get("source", "").strip() or None
    if source not in {None, "generated", "imported"}:
        raise ValueError("source must be generated or imported")
    if memory_type != "skill" and source:
        raise ValueError("source is only valid for skill directories")
    if request.args.get("sort", "name") != "name":
        raise ValueError("ContextFS directories support name ordering only")
    if request.args.get("q", "").strip():
        raise ValueError("search is applied to loaded ContextFS nodes")
    if request.args.get("page") not in {None, "1"}:
        raise ValueError("use cursor for subsequent ContextFS pages")
    limit = _bounded_int(request.args.get("limit"), default=100, minimum=1, maximum=100)
    cursor = request.args.get("cursor", "").strip() or None
    return memory_type, source, limit, cursor


def _list_visible_page(service, context, *, limit: int, cursor: str | None):
    if cursor and cursor.startswith("ogmemory://skills/"):
        if not context.agent_id:
            raise ValueError("skill cursor requires agentId")
        skills, has_more = _list_skill_page(service, context, limit=limit, cursor=cursor)
        return skills, has_more, skills[-1]["uri"] if has_more and skills else None
    if cursor:
        parsed = parse_ogmemory_uri(cursor.partition("#")[0])
        if parsed.kind != "item" or parsed.memory_type == "skill":
            raise ValueError("cursor must reference a memory item")

    entries, memory_has_more = _list_memory_node_entries_page(
        service,
        context,
        limit=limit,
        cursor=cursor,
    )
    memories = [summary for _, summary in entries]
    memory_cursor = _memory_list_cursor(entries[-1]) if entries else None
    if memory_has_more or not context.agent_id:
        return memories, memory_has_more, memory_cursor if memory_has_more else None
    if len(memories) == limit:
        skills, _ = _list_skill_page(service, context, limit=1, cursor=None)
        return memories, bool(skills), memory_cursor if skills else None

    skills, skill_has_more = _list_skill_page(
        service,
        context,
        limit=limit - len(memories),
        cursor=None,
    )
    next_cursor = skills[-1]["uri"] if skill_has_more and skills else None
    return memories + skills, skill_has_more, next_cursor


def _visible_agents(service, context) -> list[str]:
    auth = service.get_auth_service()
    if not auth.role_control_active():
        default_agent_id = str(getattr(service, "default_agent_id", "") or "")
        return [default_agent_id] if default_agent_id else []
    role = context.role if isinstance(context.role, Role) else Role(context.role)
    admin = service.get_tenant_admin_service()
    if role in {Role.ROOT, Role.ADMIN}:
        rows = admin.list_agents(context, context.account_id).get("agents", [])
        return [str(row["agent_id"]) for row in rows]
    return admin.list_visible_agent_ids(context.account_id, context.user_id, service._cfg)


def _list_skill_page(
    service,
    context,
    *,
    limit: int,
    cursor: str | None,
    source: str | None = None,
):
    service_cursor = None
    if cursor:
        parsed = parse_ogmemory_uri(cursor)
        if parsed.kind != "item" or parsed.memory_type != "skill":
            raise ValueError("cursor must reference a skill")
        if source and parsed.skill_source != source:
            raise ValueError("cursor must reference the requested skill source")
        service_cursor = f"{parsed.memory_id}|{parsed.skill_source}"
    skills = []
    has_more = False
    while len(skills) < limit:
        payload = service.list_published_skills(
            context,
            limit=min(100, limit - len(skills)),
            cursor=service_cursor,
            source=source,
        )
        for skill in payload.get("skills", []):
            resolved_source = skill.get("source") or "generated"
            skills.append(
                {
                    "uri": canonical_skill_uri(skill["name"], resolved_source),
                    "type": "skill",
                    "id": skill["name"],
                    "abstract": skill.get("description", ""),
                    "updated_at": skill.get("updated_at", ""),
                    "has_content": True,
                    "source": resolved_source,
                }
            )
        has_more = bool(payload.get("truncated"))
        if not has_more:
            break
        next_cursor = payload.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == service_cursor:
            raise RuntimeError("skill listing returned an invalid pagination cursor")
        service_cursor = next_cursor
    return skills, has_more


def _require_admin(service, context) -> None:
    if not service.get_auth_service().role_control_active():
        raise ControlPlaneDisabledError("Memory Viewer admin mode requires role control")
    if _role(context) not in {Role.ROOT, Role.ADMIN}:
        raise AuthorizationError("ADMIN or ROOT required")


def _metrics_target(service, actor) -> tuple[str, tuple[str, ...]]:
    if not service.get_auth_service().role_control_active():
        return actor.account_id, ()

    role = _role(actor)
    if role not in {Role.ROOT, Role.ADMIN}:
        owner_spaces = tuple(actor.visible_owner_spaces) or (actor.user_space_name(),)
        return actor.account_id, owner_spaces

    account_id = request.args.get("accountId", "").strip() or actor.account_id
    if role != Role.ROOT and account_id != actor.account_id:
        raise AuthorizationError("account access denied")
    admin = service.get_tenant_admin_service()
    if role == Role.ROOT:
        account_ids = {
            str(row["account_id"])
            for row in admin.list_accounts(actor).get("accounts", [])
        }
        if account_id not in account_ids:
            raise FileNotFoundError("account not found")

    user_id = request.args.get("userId", "").strip()
    agent_id = request.args.get("agentId", "").strip()
    if not user_id and not agent_id:
        return account_id, ()

    users = admin.list_users(actor, account_id).get("users", [])
    agents = admin.list_agents(actor, account_id).get("agents", [])
    if not user_id and agent_id:
        user_id = next(
            (
                str(row.get("owner_user_id") or "")
                for row in agents
                if str(row["agent_id"]) == agent_id
            ),
            "",
        )
    if user_id not in {str(row["user_id"]) for row in users}:
        raise FileNotFoundError("user not found")
    owned_agents = {
        str(row["agent_id"])
        for row in agents
        if str(row.get("owner_user_id") or "") == user_id
    }
    if agent_id:
        if agent_id not in owned_agents:
            raise FileNotFoundError("agent not found under user")
        return account_id, (f"agent:{agent_id}",)
    return account_id, (f"user:{user_id}", *(f"agent:{item}" for item in sorted(owned_agents)))


def _role(context) -> Role:
    return context.role if isinstance(context.role, Role) else Role(context.role)


def _task_visible(event, owner_spaces: tuple[str, ...]) -> bool:
    if not owner_spaces:
        return True
    payload = event.payload if isinstance(event.payload, dict) else {}
    task_owner_spaces = payload.get("task_owner_spaces")
    if isinstance(task_owner_spaces, list):
        return any(str(item) in owner_spaces for item in task_owner_spaces)
    filters = payload.get("filters")
    if not isinstance(filters, dict):
        records = payload.get("records")
        if isinstance(records, list) and records and isinstance(records[0], dict):
            filters = records[0].get("filters")
    owner_space = str(filters.get("owner_space") or "") if isinstance(filters, dict) else ""
    if not owner_space:
        try:
            parts = parse_uri(event.uri)
            owner_type = parts.get("owner_type")
            if owner_type in {"users", "agents"}:
                owner_space = f"{owner_type[:-1]}:{parts['owner_id']}"
        except ValueError:
            return False
    return owner_space in owner_spaces


def _task_record(event) -> dict:
    event_type = str(event.event_type or "")
    uri = str(event.uri or "")
    if any(segment in uri for segment in ("/sessions/", "/session_summaries/", "/session_archives/")):
        task_type = "session_commit"
    elif event_type == "UPSERT_DIRECTORY":
        task_type = "reindex"
    elif "/skills/" in uri:
        task_type = "skill_processing"
    else:
        task_type = "memory_index"
    status = {
        "PENDING": "pending",
        "PROCESSING": "running",
        "FAILED": "failed",
    }.get(str(event.status or "").upper(), "unknown")
    error_code, error_params = _task_error(event, status, uri, task_type)
    return {
        "task_id": str(event.event_id or ""),
        "task_type": task_type,
        "event_type": event_type,
        "resource_id": uri,
        "status": status,
        "stage": "index_sync",
        "retry_count": int(event.retry_count or 0),
        "error_code": error_code,
        "error_params": error_params,
        "created_at": str(event.created_at or ""),
        "updated_at": str(event.next_retry_at or event.created_at or ""),
    }


def _task_error(
    event,
    status: str,
    resource_id: str,
    task_type: str,
) -> tuple[str, dict[str, object]]:
    if status != "failed":
        return "", {}
    payload = event.payload if isinstance(event.payload, dict) else {}
    reason = " ".join(str(payload.get("failure_reason") or "").split())
    if not reason:
        return "", {}
    reason = re.sub(r"(?i)(?:bearer\s+)?sk-[a-z0-9_-]{8,}", "[REDACTED]", reason)[:300]
    lowered = reason.lower()
    params: dict[str, object] = {"resource_id": resource_id}
    dependency = _task_error_dependency(lowered)
    if "资源不存在" in reason or "not found" in lowered or "no such file" in lowered:
        params["failed_step"], params["upstream_step"] = _TASK_STEPS.get(
            task_type,
            _DEFAULT_TASK_STEPS,
        )
        return "TASK_RESOURCE_NOT_FOUND", params
    if "向量数据库" in reason or "opengauss" in lowered:
        params["dependency"] = "openGauss"
        return "TASK_VECTOR_DB_UNAVAILABLE", params
    if "object has no attribute" in lowered or "traceback" in lowered:
        params["component"] = "任务处理组件"
        return "TASK_COMPONENT_ERROR", params
    if "max retry exceeded" in lowered:
        params["retry_count"] = int(event.retry_count or 0)
        return "TASK_MAX_RETRIES", params
    if "timeout" in lowered or "timed out" in lowered:
        params["dependency"] = dependency
        return "TASK_DEPENDENCY_TIMEOUT", params
    if "connection refused" in lowered or "service unavailable" in lowered:
        params["dependency"] = dependency
        return "TASK_DEPENDENCY_UNAVAILABLE", params
    if "401" in lowered or "invalid api key" in lowered:
        params["dependency"] = dependency
        return "TASK_AUTH_FAILED", params
    if "429" in lowered or "rate limit" in lowered:
        params["dependency"] = dependency
        return "TASK_RATE_LIMITED", params
    params["detail"] = reason
    return "TASK_UNKNOWN", params


def _task_error_dependency(reason: str) -> str:
    if "embedding" in reason:
        return "Embedding"
    if "llm" in reason or "model" in reason or "chat/completions" in reason:
        return "LLM"
    return "依赖服务"


def _admin_target_context(service, actor, params=None):
    params = request.args if params is None else params
    return build_admin_target_context(
        service, actor,
        account_id=str(params.get("accountId", "")).strip() or actor.account_id,
        user_id=str(params.get("userId", "")).strip(),
        agent_id=str(params.get("agentId", "")).strip(),
    )


def _search_target_context(service, actor, build_context, params):
    auth = service.get_auth_service()
    if auth.role_control_active() and _role(actor) in {Role.ROOT, Role.ADMIN}:
        if not str(params.get("userId", "")).strip() and not str(params.get("agentId", "")).strip():
            return _admin_account_context(service, actor, params)
        return _admin_target_context(service, actor, params)
    user_id = str(params.get("userId", "")).strip()
    agent_id = str(params.get("agentId", "")).strip()
    target = build_context({"agentId": agent_id})
    if user_id and user_id != target.user_id:
        raise AuthorizationError("user access denied")
    if not agent_id:
        if not user_id:
            return target
        return replace(
            target,
            agent_id="",
            visible_owner_spaces=(f"user:{target.user_id}",),
        )
    return replace(
        target,
        agent_id=agent_id,
        visible_owner_spaces=(f"user:{target.user_id}", f"agent:{agent_id}"),
    )


def _admin_account_context(service, actor, params):
    account_id = str(params.get("accountId", "")).strip() or actor.account_id
    if _role(actor) != Role.ROOT and account_id != actor.account_id:
        raise AuthorizationError("account access denied")
    admin = service.get_tenant_admin_service()
    users = admin.list_users(actor, account_id).get("users", [])
    agents = admin.list_agents(actor, account_id).get("agents", [])
    visible = [f"user:{user['user_id']}" for user in users]
    visible.extend(f"agent:{agent['agent_id']}" for agent in agents)
    return RequestContext(
        account_id=account_id,
        user_id=actor.user_id,
        agent_id="",
        session_id=actor.session_id,
        trace_id=actor.trace_id,
        role=actor.role,
        visible_owner_spaces=tuple(visible),
    )


def _search_top_k(value) -> int:
    try:
        return int(10 if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc


def _list_target_page(
    service,
    context,
    *,
    memory_type: str,
    source: str | None,
    limit: int,
    cursor: str | None,
):
    if memory_type == "skill":
        if not context.agent_id:
            raise ValueError("agentId is required for skill directories")
        skills, has_more = _list_skill_page(
            service,
            context,
            limit=limit,
            cursor=cursor,
            source=source,
        )
        return skills, has_more, skills[-1]["uri"] if has_more and skills else None

    spec = MEMORY_TYPES[memory_type]
    target_owner = "agent" if context.agent_id else "user"
    if spec.owner != target_owner and memory_type != "pattern":
        raise ValueError(f"{memory_type} is not available for this target")
    context_fs = service._get_context_fs()
    if memory_type == "profile":
        uri = f"ctx://{context.account_id}/users/{context.user_id}/memories/profile"
        if cursor or not context_fs.exists(uri, context):
            return [], False, None
        node = context_fs.read_node(uri, context)
        summary = _memory_node_summary(node)
        summary["context_uri"] = node.uri
        return [summary], False, None

    if cursor:
        parsed = parse_ogmemory_uri(cursor.partition("#")[0])
        if parsed.kind != "item" or parsed.memory_type != memory_type:
            raise ValueError("cursor must reference the requested memory directory")
        if memory_type == "pattern" and parsed.owner_scope != (
            "agent" if context.agent_id else None
        ):
            raise ValueError("cursor must reference the requested memory directory")

    owner_path = "agents" if context.agent_id else "users"
    owner_id = context.agent_id or context.user_id
    base_uri = f"ctx://{context.account_id}/{owner_path}/{owner_id}/memories/{spec.directory}"
    external_prefix = f"ogmemory://memories/{spec.public_type}/"
    if memory_type == "pattern" and context.agent_id:
        external_prefix += "agent/"
    cursor_key = _memory_list_cursor_key(context_fs, cursor, context)
    after_uri = None
    if cursor_key:
        if not cursor_key.startswith(external_prefix):
            raise ValueError("cursor must reference the requested memory directory")
        after_uri = f"{base_uri}/{cursor_key.removeprefix(external_prefix)}"
    fetch_limit = limit + 1
    if hasattr(context_fs, "list_node_summaries_page"):
        nodes = context_fs.list_node_summaries_page(
            base_uri,
            context,
            limit=fetch_limit,
            after_uri=after_uri,
        )
    else:
        child_uris = sorted(context_fs.list_children(base_uri, context))
        if after_uri:
            child_uris = [uri for uri in child_uris if uri > after_uri]
        nodes = [context_fs.read_node(uri, context) for uri in child_uris[:fetch_limit]]
    order_key = getattr(context_fs, "list_node_order_key", lambda uri: uri)
    entries = []
    for node in nodes[:limit]:
        summary = _memory_node_summary(node)
        summary["context_uri"] = node.uri
        entries.append((external_prefix + order_key(node.uri).rsplit("/", 1)[-1], summary))
    next_cursor = _memory_list_cursor(entries[-1]) if len(nodes) > limit and entries else None
    return [summary for _, summary in entries], len(nodes) > limit, next_cursor


def _read_detail(service, context, uri: str) -> dict:
    uri = uri.strip()
    if not uri:
        raise ValueError("uri is required")
    parsed = parse_ogmemory_uri(uri)
    if parsed.kind != "item":
        raise ValueError("uri must reference a memory item")
    if parsed.memory_type == "skill":
        if not context.agent_id:
            raise ValueError("agentId is required to read a skill")
        node = service.read_published_skill(
            context,
            parsed.memory_id,
            source=parsed.skill_source,
        )
    else:
        context_fs = service._get_context_fs()
        node = None
        last_error = None
        for internal_uri in external_to_internal_uri_candidates(uri, context):
            try:
                node = context_fs.read_node(internal_uri, context)
                break
            except Exception as exc:
                if not is_not_found_or_not_visible_error(exc):
                    raise
                last_error = exc
        if node is None:
            if last_error is not None:
                raise last_error
            raise FileNotFoundError(uri)
    return {
        "uri": internal_to_external_uri(node.uri, category=node.category),
        "context_uri": node.uri,
        "type": parsed.memory_type,
        "id": parsed.memory_id,
        "abstract": node.abstract,
        "overview": node.overview,
        "content": node.content,
        "metadata": node.metadata,
    }
