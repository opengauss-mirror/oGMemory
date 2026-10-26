#!/usr/bin/env python3
"""oG-Memory Standalone HTTP Server.

Exposes oG-Memory lifecycle methods as RESTful endpoints so that
multiple OpenClaw instances can share a single oG-Memory backend.

Usage:
    python server/app.py                       # dev (Flask built-in)
    gunicorn -c python:server.gunicorn_config -w 2 -b 0.0.0.0:8090 server.app:app
"""

from __future__ import annotations

import atexit
import ipaddress
import json
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from core.errors import AccessDeniedError, NodeNotFoundError, ValidationError
from core.logging_config import configure_third_party_loggers
from features.tool_result_externalization.store_base import (
    ArtifactNotVisibleError,
    ArtifactStoreUnavailableError,
    ToolResultExternalizationDisabledError,
)
from perf.recorder import SpanEvent, clocks, get_recorder
from providers.unified_config import get_config
from server.auth import AuthenticationError, AuthorizationError, ControlPlaneDisabledError
from server.mcp_security import with_untrusted_data_notice
from server.mcp_tools import (
    bound_tool_result_collection_response,
    bound_tool_result_read_response,
)
from server.memory_service import MemoryService
from server.memory_viewer import create_memory_viewer_blueprint
from server.scene_api import create_scene_blueprint
from session.session_manager import SessionOperationInProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
configure_third_party_loggers()
logger = logging.getLogger("ogmem.http")

app = Flask(__name__)
app.json.ensure_ascii = False
_MIB = 1024 * 1024
_MAX_SKILL_UPLOAD_BODY_BYTES = 21 * _MIB
_MAX_HTTP_BODY_BYTES = 32 * _MIB
_SKILL_ZIP_MIMETYPES = frozenset({"application/zip", "application/x-zip-compressed"})
_SKILL_UPLOAD_MIMETYPES = _SKILL_ZIP_MIMETYPES | {"multipart/form-data"}
app.config["MAX_CONTENT_LENGTH"] = _MAX_HTTP_BODY_BYTES
_service: MemoryService | None = None
_service_lock = threading.Lock()
_operator_service = None
_operator_service_lock = threading.Lock()
_perf_emit_warned = False  # log first perf failure as WARNING, subsequent as DEBUG
_MAX_TRACE_EVENT_BODY_BYTES = 64 * 1024
_MAX_TRACE_EVENT_DEPTH = 64
_MAX_TRACE_EVENT_NODES = 5_000
_CODE_MEMORY_CANONICAL_FACADES = frozenset(
    {
        "register_code_workspace",
        "sync_code_workspace",
        "refresh_code_resource",
        "upsert_code_resource",
        "search_code_resources",
        "read_focused_code_snippet",
        "edit_code_file",
    }
)


def _query_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("boolean query parameter is invalid")


def _jsonify_tool_result_read(payload: dict):
    """Preserve pagination metadata before a potentially large content field."""
    return app.response_class(
        app.json.dumps(payload, sort_keys=False),
        mimetype="application/json",
    )


def _trace_event_shape_ok(value: object) -> bool:
    """Bound trace event nesting after JSON parsing and before persistence."""
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_TRACE_EVENT_NODES or depth > _MAX_TRACE_EVENT_DEPTH:
            return False
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return True


def _get_service() -> MemoryService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                cfg = get_config()
                _service = MemoryService(config=cfg)
                _service.start_extraction_recovery()
                logger.info("MemoryService initialized (config loaded)")
                # Attach to perf recorder so token snapshots work
                try:
                    rec = get_recorder()
                    rec.attach(_service)
                except Exception as exc:
                    logger.warning("perf recorder attach failed (perf will be no-op): %s", exc)
    return _service


def _shutdown_service() -> None:
    """Close the process-wide service once, if it was initialized."""
    global _service
    service, _service = _service, None
    if service is None:
        return
    try:
        service.shutdown()
    except Exception:
        logger.exception("MemoryService shutdown failed")


atexit.register(_shutdown_service)


def _get_operator_service():
    """Lazily build an OperatorService for the multimodal operator API.

    Mirrors the lazy-init pattern of ``_get_service``. Built once per
    process; subsequent calls return the same instance.
    """
    global _operator_service
    if _operator_service is not None:
        return _operator_service
    with _operator_service_lock:
        if _operator_service is not None:
            return _operator_service
        from service.operator_service import OperatorService
        from service.graph_service import GraphService
        from extraction.operators import default_registry

        cfg = get_config()
        if cfg.storage_backend == "sql":
            from providers.graph_store import SQLGraphStore
            store = SQLGraphStore(
                connection_string=cfg.sql_connection_string,
                pool_size=cfg.sql_pool_size,
            )
        else:
            from providers.graph_store import MemoryGraphStore
            store = MemoryGraphStore()
        graph_service = GraphService(store)

        # Register built-in operators (uses MockBackend by default; deployments
        # can override by populating ``operators:`` in config/operators.yaml).
        registry = default_registry()
        try:
            from extraction.operators.builtin import register_all_defaults
            register_all_defaults(registry)
        except Exception as exc:
            logger.warning("Failed to register default operators: %s", exc)

        # Optionally load operators from YAML if configured.
        yaml_path = getattr(cfg, "operators_config_path", None) or "config/operators.yaml"
        if os.path.exists(yaml_path):
            try:
                registry.load_from_config(yaml_path, override=True)
            except Exception as exc:
                logger.warning("Failed to load operators from %s: %s", yaml_path, exc)

        _operator_service = OperatorService(
            graph_service=graph_service,
            registry=registry,
        )
        logger.info("OperatorService initialized (graph_store=%s)", type(store).__name__)
        return _operator_service


def _build_authenticated_context(params: dict):
    svc = _get_service()
    auth = svc.get_auth_service()
    identity = auth.resolve_identity(dict(request.headers)) if auth.role_control_active() else None
    return svc.build_context(params, identity=identity)


def _json_object_body() -> dict:
    """Decode an adapter request only when its JSON root is an object."""

    params = request.get_json(force=True, silent=True)
    if params is None:
        raise ValueError("request body must be a JSON object")
    if not isinstance(params, dict):
        raise ValueError("request body must be a JSON object")
    return params


def _error_response(exc: Exception):
    from core.errors import ConcurrentModificationError
    from scene.config import ConfigRevisionError
    from service.operator_service import OperatorServiceBusyError

    if isinstance(exc, OperatorServiceBusyError):
        return jsonify({"error": str(exc), "retryable": True}), 429

    if isinstance(exc, RequestEntityTooLarge):
        return jsonify({"error": exc.description}), 413
    if isinstance(exc, AuthenticationError):
        return jsonify({"error": str(exc)}), 401
    if isinstance(exc, ControlPlaneDisabledError):
        return jsonify({"error": str(exc)}), 503
    if isinstance(exc, ArtifactStoreUnavailableError):
        return jsonify({
            "ok": False,
            "error": "tool_result_artifact_store_unavailable",
            "retryable": True,
        }), 503
    if isinstance(exc, (
        FileNotFoundError,
        NodeNotFoundError,
        ArtifactNotVisibleError,
        ToolResultExternalizationDisabledError,
    )):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, FileExistsError):
        return jsonify({"error": str(exc)}), 409
    if isinstance(exc, (ConcurrentModificationError, ConfigRevisionError)):
        return jsonify({"error": str(exc)}), 409
    if isinstance(exc, SessionOperationInProgress):
        return jsonify({
            "ok": False,
            "status": 409,
            "error_code": "session_operation_in_progress",
            "conflict_type": exc.conflict_type,
            "retryable": exc.retryable,
        }), 409
    if isinstance(exc, (ValueError, ValidationError)):
        return jsonify({"error": str(exc)}), 400
    if isinstance(exc, (AuthorizationError, PermissionError, AccessDeniedError)):
        return jsonify({"error": str(exc)}), 403
    return jsonify({"error": str(exc)}), 500


app.register_blueprint(
    create_memory_viewer_blueprint(_get_service, _build_authenticated_context, _error_response)
)
app.register_blueprint(
    create_scene_blueprint(_get_service, _build_authenticated_context, _error_response)
)


def _lifecycle_result_status(result: dict) -> int:
    """Map structured service acknowledgements to transport retry semantics."""
    raw_status = result.get("status") if isinstance(result, dict) else None
    if isinstance(result, dict) and result.get("retryable") is True:
        return 503
    if isinstance(raw_status, int) and 400 <= raw_status <= 599:
        return raw_status
    if isinstance(result, dict) and result.get("ok") is False:
        return 400
    return 200


def _lifecycle_result_ok(result: dict) -> bool:
    return _lifecycle_result_status(result) < 400


def _lifecycle_result_response(result: dict):
    return jsonify(result), _lifecycle_result_status(result)


@app.errorhandler(RequestEntityTooLarge)
def _handle_request_entity_too_large(exc: RequestEntityTooLarge):
    return _error_response(exc)


@app.errorhandler(SessionOperationInProgress)
def _handle_session_operation_in_progress(exc: SessionOperationInProgress):
    return _error_response(exc)


def _skill_request_data():
    """Read a JSON Skill package or a ZIP upload from the current request."""

    from skills.management import MAX_SKILL_ARCHIVE_BYTES

    if request.mimetype == "multipart/form-data":
        upload = request.files.get("file")
        if upload is None:
            upload = request.files.get("skill")
        if upload is None:
            raise ValueError("ZIP file field 'file' is required")
        data = upload.stream.read(MAX_SKILL_ARCHIVE_BYTES + 1)
    elif request.mimetype in _SKILL_ZIP_MIMETYPES:
        data = request.stream.read(MAX_SKILL_ARCHIVE_BYTES + 1)
    else:
        params = request.get_json(force=True, silent=True)
        if not isinstance(params, dict) or "data" not in params:
            raise ValueError("data is required")
        return params["data"]
    if len(data) > MAX_SKILL_ARCHIVE_BYTES:
        raise RequestEntityTooLarge(
            description=f"skill archive exceeds {MAX_SKILL_ARCHIVE_BYTES} bytes"
        )
    if not data:
        raise ValueError("skill archive is empty")
    return data


@app.before_request
def _enforce_skill_upload_content_length():
    is_skill_upload = (
        request.method == "POST"
        and request.path in {"/api/v1/skills", "/api/v1/skills/validate"}
    ) or (
        request.method == "PUT"
        and request.path.startswith("/api/v1/skills/")
    )
    if (
        is_skill_upload
        and request.mimetype in _SKILL_UPLOAD_MIMETYPES
        and request.content_length is not None
        and request.content_length > _MAX_SKILL_UPLOAD_BODY_BYTES
    ):
        raise RequestEntityTooLarge(description="skill upload request exceeds 21 MiB")


def _ip_matches_allowlist(ip_text: str, allowlist: list[str], *, log_invalid_ip: bool = False) -> bool:
    if not allowlist:
        return True
    try:
        client_ip = ipaddress.ip_address(ip_text)
    except ValueError:
        if log_invalid_ip:
            logger.warning("Invalid client IP format: %s", ip_text or "<empty>")
        return False

    for raw_entry in allowlist:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            logger.warning("Ignoring invalid allowlist entry: %s", entry)
            continue
        if client_ip in network:
            return True
    return False


def _resolve_client_ip(cfg) -> str:
    remote_ip = (request.remote_addr or "").strip()
    if getattr(cfg, "http_ip_allowlist_trust_proxy", False):
        forwarded_header_present = bool(
            request.headers.get("X-Forwarded-For") or request.headers.get("X-Real-IP")
        )
        trusted_proxies = getattr(cfg, "http_trusted_proxies", [])
        if not trusted_proxies:
            if forwarded_header_present:
                logger.warning(
                    "Proxy IP trust is enabled, but no trusted proxies are configured; ignoring forwarded headers"
                )
            return remote_ip
        if not _ip_matches_allowlist(remote_ip, trusted_proxies, log_invalid_ip=True):
            if forwarded_header_present:
                logger.warning("Ignoring forwarded headers from untrusted proxy IP %s", remote_ip or "<unknown>")
            return remote_ip

        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            forwarded_ip = forwarded_for.split(",", 1)[0].strip()
            if forwarded_ip:
                return forwarded_ip
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    return remote_ip


def _ip_allowed(ip_text: str, allowlist: list[str]) -> bool:
    return _ip_matches_allowlist(ip_text, allowlist, log_invalid_ip=True)


@app.before_request
def _enforce_ip_allowlist():
    cfg = _service._cfg if _service is not None else get_config()
    allowlist = getattr(cfg, "http_ip_allowlist", [])
    if not allowlist:
        return None

    # Always allow health checks regardless of IP (for load balancers/monitors)
    if request.path == "/api/v1/health":
        return None

    client_ip = _resolve_client_ip(cfg)
    if _ip_allowed(client_ip, allowlist):
        return None

    logger.warning("Rejected request from IP %s", client_ip or "<unknown>")
    return jsonify({"error": "IP not allowed"}), 403


# ---------------------------------------------------------------------------
# Perf instrumentation helper
# ---------------------------------------------------------------------------

def _emit_perf_stage(
    stage: str,
    params: dict,
    before: dict,
    started_at: float,
    wall_start: float,
    cpu_start: float,
    ok: bool,
    error: str | None,
    tokens: dict,
    cost: dict,
    source: str,
) -> None:
    """Emit a perf stage event. Safe to call when recorder is disabled."""
    try:
        rec = get_recorder()
        if not rec.enabled:
            return
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cpu_ms = (time.process_time() - cpu_start) * 1000.0
        event = SpanEvent(
            run_id=rec.run_id,
            session_id=params.get("sessionId"),
            trace_id=None,
            stage=stage,
            span="",
            parent_span=None,
            started_at=started_at,
            wall_ms=round(wall_ms, 3),
            cpu_ms=round(cpu_ms, 3),
            ok=ok,
            error=error,
            llm_model=tokens.get("llm_model"),
            embed_model=tokens.get("embed_model"),
            tokens={"llm": tokens.get("llm", {}), "embed": tokens.get("embed", {})},
            cost_usd=cost,
            token_source=source,
            meta={},
        )
        rec.emit(event)
    except Exception as exc:
        global _perf_emit_warned
        if not _perf_emit_warned:
            logger.warning("perf stage emit failed stage=%s: %s (further failures at DEBUG)", stage, exc)
            _perf_emit_warned = True
        else:
            logger.debug("perf stage emit failed stage=%s: %s", stage, exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/v1/compose", methods=["POST"])
def handle_compose():
    """Search memory, return systemPromptAddition for the current turn."""
    params = request.get_json(force=True, silent=True) or {}
    logger.info("[assemble] userId=%s sessionId=%s", params.get("userId"), params.get("sessionId"))
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compose(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compose", params, before, started_at, wall_start, cpu_start, _lifecycle_result_ok(result), None, tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("compose failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compose", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/prefetch", methods=["POST"])
def handle_prefetch():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prefetch(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("prefetch failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/session_working_set", methods=["GET", "POST"])
def handle_session_working_set():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().session_working_set(params))
    except Exception as exc:
        logger.error("session_working_set failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/evict_idle_sessions", methods=["POST"])
def handle_evict_idle_sessions():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().evict_idle_sessions(params))
    except Exception as exc:
        logger.error("evict_idle_sessions failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/after_turn", methods=["POST"])
def handle_after_turn():
    """Commit conversation to long-term memory after an agent turn."""
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        from server.mcp_trace import capture_write_trace, validate_trace_messages
        trace_messages = params.get("traceMessages")
        if trace_messages is not None:
            validate_trace_messages(trace_messages)
        result = _get_service().after_turn(params)
        if trace_messages and _lifecycle_result_ok(result):
            captured = capture_write_trace(
                _get_service(), ctx, trace_messages, session_id=ctx.session_id,
                client_request_id=str(params.get("clientRequestId") or params.get("client_request_id") or "").strip(),
            )
            result["trace_captured"] = captured
            if not captured:
                result.update(ok=False, retryable=True, status="trace_capture_pending")
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            success = _lifecycle_result_ok(result)
            _emit_perf_stage("after_turn", params, before, started_at, wall_start, cpu_start, success, None if success else str(result.get("status", "rejected")), tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("after_turn failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("after_turn", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


def _memory_tool_request():
    params = request.get_json(force=True, silent=True) or {}
    if not isinstance(params, dict):
        raise ValueError("request body must be an object")
    return params, _build_authenticated_context(params)


def _memory_tool_integer(params: dict, name: str, default: int) -> int:
    try:
        return int(params.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


@app.route("/api/v1/memories/query", methods=["POST"])
def handle_query_memory():
    """Expose the native read-only oGMemory query contract over HTTP."""

    from server.mcp_tools import query_memory

    try:
        params, ctx = _memory_tool_request()
        return jsonify(
            query_memory(
                _get_service(),
                query=params.get("query", ""),
                types=params.get("types"),
                top_k=_memory_tool_integer(params, "top_k", 5),
                score_threshold=params.get("score_threshold"),
                session_id=params.get("sessionId") or params.get("session_id"),
                agent_id=ctx.agent_id,
                request_context=ctx,
            )
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/memories/read", methods=["POST"])
def handle_read_memory():
    """Read one native ogmemory:// memory URI."""

    from server.mcp_tools import read_memory

    try:
        params, ctx = _memory_tool_request()
        return jsonify(
            read_memory(
                _get_service(),
                uri=params.get("uri", ""),
                unit_key=params.get("unit_key"),
                level=params.get("level", "all"),
                max_chars=_memory_tool_integer(params, "max_chars", 12000),
                session_id=params.get("sessionId") or params.get("session_id"),
                agent_id=ctx.agent_id,
                request_context=ctx,
            )
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/memories/write", methods=["POST"])
def handle_write_memory():
    """Submit user-provided text through the native extraction pipeline."""

    from server.mcp_resources import mcp_write_tools_enabled
    from server.mcp_tools import write_memory

    try:
        params, ctx = _memory_tool_request()
        service = _get_service()
        if not mcp_write_tools_enabled(service._cfg):
            raise PermissionError("memory write tools are disabled")
        return jsonify(
            write_memory(
                service,
                text=params.get("text"),
                session_id=params.get("sessionId") or params.get("session_id"),
                agent_id=ctx.agent_id,
                client_request_id=params.get("client_request_id"),
                force_extract=_parse_boolean(params.get("force_extract"), "force_extract"),
                wait=_parse_boolean(params.get("wait"), "wait"),
                request_context=ctx,
            )
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/memories/delete", methods=["POST"])
def handle_delete_memory():
    """Soft-delete one explicit native ogmemory:// memory URI."""

    from server.mcp_resources import mcp_delete_tools_enabled
    from server.mcp_tools import delete_memory

    try:
        params, ctx = _memory_tool_request()
        service = _get_service()
        if not mcp_delete_tools_enabled(service._cfg):
            raise PermissionError("memory delete tools are disabled")
        return jsonify(
            delete_memory(
                service,
                uri=params.get("uri", ""),
                reason=params.get("reason"),
                session_id=params.get("sessionId") or params.get("session_id"),
                agent_id=ctx.agent_id,
                client_request_id=params.get("client_request_id"),
                request_context=ctx,
            )
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/ingest", methods=["POST"])
def handle_ingest():
    """Single message ingest (pass-through, real work in after_turn)."""
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            success = _lifecycle_result_ok(result)
            _emit_perf_stage("ingest", params, before, started_at, wall_start, cpu_start, success, None if success else str(result.get("status", "rejected")), tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("ingest failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/ingest_batch", methods=["POST"])
def handle_ingest_batch():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().ingest_batch(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            success = _lifecycle_result_ok(result)
            _emit_perf_stage("ingest_batch", params, before, started_at, wall_start, cpu_start, success, None if success else str(result.get("status", "rejected")), tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("ingest_batch failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("ingest_batch", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/govern_first_view_tool_result", methods=["POST"])
def handle_govern_first_view_tool_result():
    """Rewrite a trusted adapter ToolResult before its first model request."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().govern_first_view_tool_result(params))
    except Exception as exc:
        logger.error("first-view tool-result governance failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/compact", methods=["POST"])
def handle_compact():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().compact(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            success = _lifecycle_result_ok(result)
            _emit_perf_stage("compact", params, before, started_at, wall_start, cpu_start, success, None if success else str(result.get("status", "rejected")), tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("compact failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("compact", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/prepare_compaction", methods=["POST"])
def handle_prepare_compaction():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prepare_compaction(params)
        if not _lifecycle_result_ok(result):
            if rec.enabled:
                after = rec.snapshot_tokens()
                tokens, cost, source = rec.finalize_tokens(after, before)
                _emit_perf_stage("prepare_compaction", params, before, started_at, wall_start, cpu_start, False, str(result.get("status", "rejected")), tokens, cost, source)
            return _lifecycle_result_response(result)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("prepare_compaction", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify({
            "ok": True,
            "prepared": True,
            "prepareToken": result.get("prepareToken", ""),
            "messagesPrepared": len(result.get("messages", [])),
        })
    except Exception as exc:
        logger.error("prepare_compaction failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("prepare_compaction", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/bootstrap", methods=["POST"])
def handle_bootstrap():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().bootstrap(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("bootstrap", params, before, started_at, wall_start, cpu_start, True, None, tokens, cost, source)
        return jsonify(result)
    except Exception as exc:
        logger.error("bootstrap failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("bootstrap", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/resources/code/refresh", methods=["POST"])
def handle_code_resource_refresh():
    """Accept deterministic refresh or missing-file cleanup for one code file."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().refresh_code_resource(params))
    except Exception as exc:
        logger.error(
            "code resource refresh failed type=%s",
            type(exc).__name__,
        )
        return _error_response(exc)


@app.route("/api/v1/resources/code/sync/registrations", methods=["POST"])
def handle_code_workspace_registration():
    """Register a code workspace without returning its local path."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().register_code_workspace(params))
    except Exception as exc:
        logger.error("code workspace registration failed type=%s", type(exc).__name__)
        return _error_response(exc)


@app.route("/api/v1/resources/code/sync", methods=["POST"])
def handle_code_workspace_sync():
    """Run an explicit registered-workspace sync through its canonical service."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().sync_code_workspace(params))
    except Exception as exc:
        logger.error("code workspace sync failed type=%s", type(exc).__name__)
        return _error_response(exc)


@app.route("/api/v1/resources/code/upsert", methods=["POST"])
def handle_code_resource_upsert():
    """Refresh one code path through the existing lifecycle facade."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().upsert_code_resource(params))
    except Exception as exc:
        logger.error("code resource upsert failed type=%s", type(exc).__name__)
        return _error_response(exc)


@app.route("/api/v1/resources/code/search", methods=["POST"])
def handle_code_resource_search():
    """Search only imported code resources through active retrieval."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().search_code_resources(params))
    except Exception as exc:
        logger.error("code resource search failed type=%s", type(exc).__name__)
        return _error_response(exc)


@app.route("/api/v1/resources/code/snippet", methods=["POST"])
def handle_code_resource_snippet():
    """Return bounded evidence for one exact imported code URI."""
    try:
        params = _json_object_body()
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().read_focused_code_snippet(params))
    except Exception as exc:
        logger.error("code resource snippet failed type=%s", type(exc).__name__)
        return _error_response(exc)


@app.route("/api/v1/tool_output_original", methods=["POST"])
def handle_tool_output_original():
    """Return the current agent's stored original tool output."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().read_tool_output_original(params)
        return jsonify(with_untrusted_data_notice(result))
    except Exception as exc:
        logger.error("tool_output_original failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/filter_read", methods=["POST"])
def handle_filter_read():
    """Filter a complete Read result and return a bounded replacement."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().filter_read(params))
    except Exception as exc:
        logger.error("filter_read failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/filter_bash", methods=["POST"])
def handle_filter_bash():
    """Filter supported Bash output without executing the command."""
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        return jsonify(_get_service().filter_bash(params))
    except Exception as exc:
        logger.error("filter_bash failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/dispose", methods=["POST"])
def handle_dispose():
    params = request.get_json(force=True, silent=True) or {}
    rec = get_recorder()
    before = rec.snapshot_tokens() if rec.enabled else {}
    started_at, wall_start, cpu_start = clocks()
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().dispose(params)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            success = _lifecycle_result_ok(result)
            _emit_perf_stage("dispose", params, before, started_at, wall_start, cpu_start, success, None if success else str(result.get("status", "rejected")), tokens, cost, source)
        return _lifecycle_result_response(result)
    except Exception as exc:
        logger.error("dispose failed: %s", exc, exc_info=True)
        if rec.enabled:
            after = rec.snapshot_tokens()
            tokens, cost, source = rec.finalize_tokens(after, before)
            _emit_perf_stage("dispose", params, before, started_at, wall_start, cpu_start, False, f"{type(exc).__name__}: {exc}", tokens, cost, source)
        return _error_response(exc)


@app.route("/api/v1/prepare_subagent_spawn", methods=["POST"])
def handle_prepare_subagent_spawn():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().prepare_subagent_spawn(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("prepare_subagent_spawn failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/on_subagent_ended", methods=["POST"])
def handle_on_subagent_ended():
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = _get_service().on_subagent_ended(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("on_subagent_ended failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/token_stats", methods=["GET", "POST"])
def handle_token_stats():
    """Return cumulative LLM & embedding token usage across all after_turn calls.

    GET  → return current cumulative stats (non-destructive)
    POST → {"reset": true} to zero the counters and return the snapshot
    """
    svc = _get_service()
    if request.method == "POST":
        params = request.get_json(force=True, silent=True) or {}
        reset = bool(params.get("reset", False))
    else:
        reset = False
    return jsonify(svc.get_cumulative_token_usage(reset=reset))


@app.route("/stats/tool-usage", methods=["GET"])
@app.route("/api/v1/stats/tool-usage", methods=["GET"])
def handle_tool_usage_stats():
    """Return internal oGMem tool usage aggregates for diagnostics."""
    params = dict(request.args)
    return jsonify(_get_service().get_tool_usage_stats(params))


@app.route("/api/v1/health", methods=["GET"])
def handle_health():
    """Health check — verifies AGFS, LLM, vector DB connectivity."""
    result = _get_service().health()
    status_code = 200 if result.get("status") == "ok" else 503
    return jsonify(result), status_code


def _skill_request_kwargs() -> dict:
    return {
        "agent_id": request.args.get("agent_id") or request.headers.get("X-Agent-ID"),
        "transport_headers": dict(request.headers),
        "allow_stdio_env_auth": False,
    }


def _skill_include_files() -> bool:
    return _parse_boolean(request.args.get("include_files"), "include_files", default=True)


def _parse_boolean(value, name: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be true or false")
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be true or false")


@app.route("/api/v1/skills", methods=["GET", "POST"])
def handle_skills():
    """List/search published skills or install a managed Skill package."""

    from server.mcp_tools import list_skills, search_skills

    try:
        if request.method == "POST":
            ctx = _build_authenticated_context({
                "agentId": request.args.get("agent_id") or request.headers.get("X-Agent-ID")
            })
            return jsonify(_get_service().add_managed_skill(ctx, _skill_request_data())), 201
        query = str(request.args.get("q") or "").strip()
        limit = int(request.args.get("limit") or 50)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if query:
            return jsonify(search_skills(_get_service(), query=query, top_k=min(limit, 20), **_skill_request_kwargs()))
        return jsonify(
            list_skills(
                _get_service(),
                limit=limit,
                cursor=request.args.get("cursor"),
                **_skill_request_kwargs(),
            )
        )
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/skills/validate", methods=["POST"])
def handle_skill_validation():
    """Validate a complete managed Skill package without persisting it."""

    try:
        _build_authenticated_context({
            "agentId": request.args.get("agent_id") or request.headers.get("X-Agent-ID")
        })
        params = request.get_json(silent=True)
        if isinstance(params, dict):
            if "data" not in params:
                raise ValueError("data is required")
            data = params["data"]
            strict_value = params.get("strict")
            skill_dir_name = params.get("skill_dir_name")
            source_path = params.get("source_path")
        else:
            data = _skill_request_data()
            strict_value = request.form.get("strict") or request.args.get("strict")
            skill_dir_name = request.form.get("skill_dir_name") or request.args.get("skill_dir_name")
            source_path = request.form.get("source_path") or request.args.get("source_path")
        return jsonify(_get_service().validate_managed_skill(
            data,
            strict=_parse_boolean(strict_value, "strict"),
            skill_dir_name=skill_dir_name,
            source_path=source_path,
        ))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/skills/<name>", methods=["GET", "PUT", "DELETE"])
def handle_skill(name: str):
    """Read, update, or delete one published Skill."""

    from server.mcp_tools import read_skill

    try:
        if request.method == "PUT":
            ctx = _build_authenticated_context({
                "agentId": request.args.get("agent_id") or request.headers.get("X-Agent-ID")
            })
            return jsonify(
                _get_service().update_managed_skill(ctx, name, _skill_request_data())
            )
        if request.method == "DELETE":
            ctx = _build_authenticated_context({
                "agentId": request.args.get("agent_id") or request.headers.get("X-Agent-ID")
            })
            return jsonify(_get_service().delete_published_skill(
                ctx, name, source=request.args.get("source")
            ))
        return jsonify(read_skill(
            _get_service(),
            name=name,
            source=request.args.get("source"),
            include_files=_skill_include_files(),
            **_skill_request_kwargs(),
        ))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/skills/<name>/<path:file_path>", methods=["GET"])
def handle_skill_file(name: str, file_path: str):
    """Read one file from an installed managed Skill package."""

    try:
        ctx = _build_authenticated_context({
            "agentId": request.args.get("agent_id") or request.headers.get("X-Agent-ID")
        })
        return jsonify(_get_service().read_managed_skill_file(ctx, name, file_path))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/skill-candidates", methods=["GET"])
def handle_skill_candidates():
    """Inspect safe candidate summaries; raw traces are intentionally unavailable."""

    from server.mcp_tools import list_skill_candidates

    try:
        return jsonify(
            list_skill_candidates(
                _get_service(), state=request.args.get("state"), **_skill_request_kwargs()
            )
        )
    except Exception as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Trace / Perf event ingestion
# ---------------------------------------------------------------------------

@app.route("/api/v1/trace/events", methods=["POST"])
def handle_trace_event():
    """Persist one agent lifecycle event to the SQL trace store."""
    if (
        request.content_length is not None
        and request.content_length > _MAX_TRACE_EVENT_BODY_BYTES
    ):
        return jsonify({"error": "trace event body exceeds 64 KiB"}), 413
    raw_body = request.stream.read(_MAX_TRACE_EVENT_BODY_BYTES + 1)
    if len(raw_body) > _MAX_TRACE_EVENT_BODY_BYTES:
        return jsonify({"error": "trace event body exceeds 64 KiB"}), 413
    try:
        event_data = json.loads(raw_body)
    except RecursionError:
        return jsonify({"error": "trace event is too deeply nested"}), 413
    except (json.JSONDecodeError, UnicodeDecodeError):
        event_data = None
    if not isinstance(event_data, dict) or not event_data:
        return jsonify({"error": "invalid JSON body"}), 400
    event = event_data.get("event")
    session_id = str(event_data.get("session_id") or "").strip()
    event_id = str(event_data.get("request_id") or "").strip()
    if not isinstance(event, dict) or not event:
        return jsonify({"error": "event object is required"}), 400
    if not _trace_event_shape_ok(event):
        return jsonify({"error": "trace event is too large"}), 413
    legacy_hook_type = str(event.get("hook_type") or "unknown").strip() or "unknown"
    event_type = str(event.get("event_type") or "").strip()
    canonical_type = (
        "turn_end"
        if legacy_hook_type == "Stop"
        else (
            "session_end"
            if legacy_hook_type == "SessionEnd"
            else (event_type or legacy_hook_type or "hook_event")
        )
    )
    event = {**event, "event_type": canonical_type}
    for identity_field in (
        "hook_type",
        "hook",
        "hookName",
        "source",
        "producer",
        "producer_id",
        "producerId",
        "agent_id",
        "agentId",
        "agent",
        "subagent_id",
        "subagentId",
        "account_id",
        "accountId",
        "account",
        "user_id",
        "userId",
        "user",
        "session_id",
        "sessionId",
        "session",
    ):
        event.pop(identity_field, None)
    if not session_id or not event_id:
        return jsonify({"error": "request_id and session_id are required"}), 400

    svc = _get_service()
    params = {
        "accountId": event_data.get("account_id"),
        "userId": event_data.get("user_id"),
        "agentId": event_data.get("agent_id"),
        "sessionId": session_id,
    }
    try:
        ctx = _build_authenticated_context(params)
    except (AuthenticationError, AuthorizationError, PermissionError, ValueError) as exc:
        return _error_response(exc)

    trace_result = svc.add_session_trace(
        {
            "_ctx": ctx,
            "eventId": event_id,
            "sessionId": session_id,
            "hookType": canonical_type,
            "eventTime": event_data.get("timestamp"),
            "event": event,
        }
    )
    if not trace_result.get("ok"):
        status = 503 if trace_result.get("retryable", True) else 400
        return jsonify(trace_result), status

    if canonical_type == "session_end":
        svc.wake_skill_sedimentation()

    if canonical_type == "turn_end":
        finalized = svc.finalize_session_trace(
            {
                "_ctx": ctx,
                "sessionId": session_id,
                "stopEventIndex": trace_result["event_index"],
            }
        )
        if not finalized.get("ok"):
            status = 503 if finalized.get("retryable", True) else 400
            return jsonify(
                {"ok": False, "event": trace_result, "finalize": finalized}
            ), status
        trace_result["turn"] = finalized

    return jsonify(trace_result), 200


@app.route("/api/v1/perf/events", methods=["GET", "POST"])
def handle_perf_event():
    """Return accumulated perf events as JSON (GET) or append a perf event to JSONL (POST).

    P1 fix (GET/POST): when perf is enabled, enforce authentication via
    _build_authenticated_context so only authorized clients can read/write perf data.
    An unauthenticated POST would let anyone append arbitrary JSON to the perf log,
    which is a security risk when the service is externally reachable.

    P2 fix: GET semantics clarified — rec.get_all_events() returns events only when
    the sink is a MemorySink (e.g. during unit tests). With the default JsonlSink,
    events are already persisted to OGMEM_PERF_OUT and should be read via file tools
    or the perf.report CLI rather than this endpoint.
    """
    rec = get_recorder()

    # -- GET: return accumulated events --------------------------------------
    if request.method == "GET":
        events = rec.get_all_events()
        return jsonify({
            "run_id": rec.run_id,
            "enabled": rec.enabled,
            # P2 note: events is non-empty only when sink is MemorySink.
            # With JsonlSink (default) use file tools or perf.report to read persisted events.
            "events": events,
        })

    # -- POST: append perf event via sink ------------------------------------
    # P1 fix: enforce auth via _build_authenticated_context, catching exceptions.
    # P2 fix: POST returns explicit error when perf is disabled (not silent no-op).
    try:
        _build_authenticated_context({})
    except (AuthenticationError, AuthorizationError) as exc:
        return _error_response(exc)

    if not rec.enabled:
        return jsonify({"ok": False, "error": "perf is disabled"}), 409

    event_data = request.get_json(force=True, silent=True)
    if not event_data:
        return jsonify({"error": "invalid JSON body"}), 400

    from perf.recorder import SpanEvent
    try:
        event = SpanEvent(
            run_id=str(event_data.get("run_id", rec.run_id or "")),
            session_id=event_data.get("session_id"),
            trace_id=event_data.get("trace_id"),
            stage=str(event_data.get("stage", "")),
            span=str(event_data.get("span", "")),
            parent_span=event_data.get("parent_span"),
            started_at=float(event_data.get("started_at", 0.0)),
            wall_ms=float(event_data.get("wall_ms", 0.0)),
            cpu_ms=float(event_data.get("cpu_ms", 0.0)),
            ok=bool(event_data.get("ok", True)),
            error=event_data.get("error"),
            llm_model=event_data.get("llm_model"),
            embed_model=event_data.get("embed_model"),
            tokens=event_data.get("tokens", {}),
            cost_usd=event_data.get("cost_usd", {}),
            token_source=str(event_data.get("token_source", "tracker")),
            meta=event_data.get("meta", {}),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid event structure: {exc}"}), 400

    rec.emit(event)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# Session API (stateful session management)
# ---------------------------------------------------------------------------

@app.route("/api/v1/sessions/<session_id>/messages", methods=["POST"])
def handle_add_message(session_id):
    """Add a message to the session buffer."""
    params = request.get_json(force=True, silent=True) or {}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)

    content = params.get("content", "")
    if not content:
        return jsonify({"ok": False, "error": "content is required"}), 400

    params["_ctx"] = ctx
    result = svc.add_session_message(params)
    return _lifecycle_result_response(result)


@app.route("/api/v1/sessions/<session_id>", methods=["GET"])
def handle_get_session(session_id):
    """Get session meta + pending_tokens."""
    params = {k: v for k, v in request.args.items()}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    mgr = svc.get_session_manager()
    result = mgr.get_session(session_id, ctx)
    return jsonify(result)


@app.route("/api/v1/sessions/<session_id>", methods=["DELETE"])
def handle_delete_session(session_id):
    """Delete one Agent Session and all of its oGMemory-owned resources."""
    try:
        params = request.get_json(force=False, silent=True) or {}
        params["sessionId"] = session_id
        svc = _get_service()
        params["_ctx"] = _build_authenticated_context(params)
        return _lifecycle_result_response(svc.delete_session(params))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/sessions/<session_id>/commit", methods=["POST"])
def handle_commit_session(session_id):
    """Commit session: archive + extract memories."""
    params = request.get_json(force=True, silent=True) or {}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    params["_ctx"] = ctx
    result = svc.commit_session_buffer(params)
    return _lifecycle_result_response(result)


@app.route("/api/v1/sessions/<session_id>/tool-results", methods=["GET"])
def handle_list_tool_results(session_id):
    """List verified tool-result artifacts without reading their bodies."""
    try:
        params = {key: value for key, value in request.args.items()}
        params["sessionId"] = session_id
        svc = _get_service()
        if not getattr(svc._cfg, "tool_result_externalization_enabled", False):
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        params["_ctx"] = _build_authenticated_context(params)
        payload = svc.list_tool_result_artifacts(params)
        payload = bound_tool_result_collection_response(
            payload,
            "artifacts",
            svc._cfg.tool_result_http_max_response_chars,
        )
        return jsonify(payload)
    except Exception as exc:
        return _error_response(exc)


@app.route(
    "/api/v1/sessions/<session_id>/tool-results/<artifact_id>",
    methods=["GET"],
)
def handle_read_tool_result(session_id, artifact_id):
    """Read a bounded UTF-8 byte range from one artifact."""
    try:
        params = {key: value for key, value in request.args.items()}
        params.update({"sessionId": session_id, "artifactId": artifact_id})
        svc = _get_service()
        if not getattr(svc._cfg, "tool_result_externalization_enabled", False):
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        full = _query_bool(params.pop("full", None))
        response_limit = svc._cfg.tool_result_http_max_response_chars
        requested_length = response_limit if full else int(
            params.get("length_bytes", svc._cfg.tool_result_read_default_bytes)
        )
        if requested_length < 1:
            raise ValueError("length_bytes must be positive")
        params["length_bytes"] = min(
            response_limit if full else requested_length,
            response_limit,
        )
        if full:
            params["offset_bytes"] = 0
        params["_ctx"] = _build_authenticated_context(params)
        payload = svc.read_tool_result_artifact(params)
        payload = bound_tool_result_read_response(payload, response_limit)
        payload = svc.attest_tool_result_source(payload, params["_ctx"])
        return _jsonify_tool_result_read(payload)
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/sessions/<session_id>/tool-results/search", methods=["POST"])
def handle_search_tool_results(session_id):
    """Search bounded relevant chunks without injecting complete artifacts."""
    try:
        params = request.get_json(force=True, silent=True) or {}
        params["sessionId"] = session_id
        svc = _get_service()
        if not getattr(svc._cfg, "tool_result_externalization_enabled", False):
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        top_k = int(params.get("top_k", svc._cfg.tool_result_search_top_k))
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        response_limit = svc._cfg.tool_result_http_max_response_chars
        max_chars = int(params.get("max_chars", 20_000))
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        params["top_k"] = top_k
        params["max_chars"] = min(max_chars, response_limit)
        params["_ctx"] = _build_authenticated_context(params)
        payload = svc.search_tool_result_artifacts(params)
        payload = bound_tool_result_collection_response(
            payload,
            "results",
            response_limit,
        )
        payload["results"] = [
            svc.attest_tool_result_source(row, params["_ctx"])
            for row in payload.get("results", [])
        ]
        return jsonify(payload)
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/sessions/<session_id>/context", methods=["GET"])
def handle_get_session_context(session_id):
    """Get assembled session context (for assemble/compact)."""
    token_budget = int(request.args.get("token_budget", 128_000))
    params = {k: v for k, v in request.args.items()}
    params["sessionId"] = session_id
    svc = _get_service()
    ctx = _build_authenticated_context(params)
    mgr = svc.get_session_manager()
    result = mgr.get_context(session_id, token_budget, ctx)
    return jsonify(result)


# Generic dispatch endpoint (for forward-compat with new methods)
@app.route("/api/v1/call/<method>", methods=["POST"])
def handle_call(method: str):
    if method == "delete_session" or method in _CODE_MEMORY_CANONICAL_FACADES:
        return jsonify({"error": f"unknown method: {method}"}), 404
    svc = _get_service()
    handler = getattr(svc, method, None)
    if handler is None or not callable(handler):
        return jsonify({"error": f"unknown method: {method}"}), 404
    params = request.get_json(force=True, silent=True) or {}
    try:
        ctx = _build_authenticated_context(params)
        params["_ctx"] = ctx
        result = handler(params)
        return jsonify(result)
    except Exception as exc:
        logger.error("call/%s failed: %s", method, exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/admin/accounts", methods=["GET"])
def handle_admin_accounts():
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_accounts(ctx))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>", methods=["GET"])
def handle_admin_account(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_account(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users", methods=["GET", "POST"])
def handle_admin_users(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.list_users(ctx, account_id))
        return jsonify(admin.create_user(ctx, account_id, params["user_id"], params.get("role", "user")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users/<user_id>", methods=["DELETE"])
def handle_admin_delete_user(account_id, user_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().delete_user(ctx, account_id, user_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/users/<user_id>/role", methods=["PATCH"])
def handle_admin_set_role(account_id, user_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        return jsonify(svc.get_tenant_admin_service().set_role(ctx, account_id, user_id, params["role"]))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/roles", methods=["GET"])
def handle_admin_roles(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_roles(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/agents", methods=["GET", "POST"])
def handle_admin_agents(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.list_agents(ctx, account_id))
        return jsonify(admin.create_agent(ctx, account_id, params["agent_id"], params.get("owner_user_id")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/agents/<agent_id>", methods=["GET", "PATCH"])
def handle_admin_agent(account_id, agent_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        params = request.get_json(force=True, silent=True) or {}
        ctx = _build_authenticated_context(params)
        admin = svc.get_tenant_admin_service()
        if request.method == "GET":
            return jsonify(admin.get_agent(ctx, account_id, agent_id))
        return jsonify(admin.update_agent(ctx, account_id, agent_id, params.get("owner_user_id")))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/config/agent-sharing", methods=["GET"])
def handle_admin_agent_sharing():
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_agent_sharing_config(ctx, svc._cfg))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/audit-logs", methods=["GET"])
def handle_admin_audit_logs(account_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().list_audit_logs(ctx, account_id))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/admin/accounts/<account_id>/audit-logs/<log_id>", methods=["GET"])
def handle_admin_audit_log(account_id, log_id):
    svc = _get_service()
    try:
        if not svc.get_auth_service().role_control_active():
            raise ControlPlaneDisabledError("Admin API is unavailable when role control is disabled")
        ctx = _build_authenticated_context({})
        return jsonify(svc.get_tenant_admin_service().get_audit_log(ctx, account_id, log_id))
    except Exception as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Operator API (multimodal extraction)
# ---------------------------------------------------------------------------

@app.route("/api/v1/operators", methods=["GET"])
def handle_operators_list():
    """List registered operators, optionally filtered by raw_type."""
    raw_type = request.args.get("raw_type")
    try:
        svc = _get_operator_service()
        ops = svc.list_operators(raw_type=raw_type) if raw_type else svc.list_operators()
        return jsonify({"operators": ops, "count": len(ops)})
    except Exception as exc:
        logger.error("operators list failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/operators/<operator_type>", methods=["GET"])
def handle_operators_describe(operator_type):
    """Get details (description, config_schema) for one operator."""
    try:
        svc = _get_operator_service()
        return jsonify(svc.describe_operator(operator_type))
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/v1/operators/<operator_type>/run", methods=["POST"])
def handle_operators_run_single(operator_type):
    """Run a single operator against a raw node."""
    params = request.get_json(force=True, silent=True) or {}
    raw_node_id = params.get("raw_node_id")
    if not raw_node_id:
        return jsonify({"error": "raw_node_id is required"}), 400
    ctx = _build_authenticated_context(params)
    try:
        svc = _get_operator_service()
        result = svc.run_operator(
            raw_node_id=raw_node_id,
            operator_type=operator_type,
            ctx=ctx,
            params=params.get("params"),
            write_to_graph=params.get("write_to_graph", True),
        )
        return jsonify(result)
    except Exception as exc:
        logger.error("operator run failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/operators/pipelines/run", methods=["POST"])
def handle_operators_run_pipeline():
    """Run a multi-step pipeline against a raw node (sync)."""
    params = request.get_json(force=True, silent=True) or {}
    raw_node_id = params.get("raw_node_id")
    pipeline = params.get("pipeline")
    if not raw_node_id or not pipeline:
        return jsonify({"error": "raw_node_id and pipeline are required"}), 400
    ctx = _build_authenticated_context(params)
    try:
        svc = _get_operator_service()
        result = svc.run_pipeline(
            raw_node_id=raw_node_id,
            pipeline_spec=pipeline,
            ctx=ctx,
            write_to_graph=params.get("write_to_graph", True),
        )
        return jsonify(result)
    except Exception as exc:
        logger.error("pipeline run failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/operators/pipelines/run_async", methods=["POST"])
def handle_operators_run_pipeline_async():
    """Run a pipeline on a background thread; return task_id."""
    params = request.get_json(force=True, silent=True) or {}
    raw_node_id = params.get("raw_node_id")
    pipeline = params.get("pipeline")
    if not raw_node_id or not pipeline:
        return jsonify({"error": "raw_node_id and pipeline are required"}), 400
    ctx = _build_authenticated_context(params)
    try:
        svc = _get_operator_service()
        task_id = svc.run_pipeline_async(
            raw_node_id=raw_node_id,
            pipeline_spec=pipeline,
            ctx=ctx,
            write_to_graph=params.get("write_to_graph", True),
        )
        return jsonify({"task_id": task_id, "status": "processing"})
    except Exception as exc:
        logger.error("async pipeline run failed: %s", exc, exc_info=True)
        return _error_response(exc)


@app.route("/api/v1/operators/tasks/<task_id>", methods=["GET"])
def handle_operators_task_status(task_id):
    """Poll the status of an async pipeline task owned by this account."""
    try:
        ctx = _build_authenticated_context(request.args.to_dict())
        svc = _get_operator_service()
        rec = svc.get_task_status(task_id, ctx)
        if rec is None:
            return jsonify({"error": "task not found"}), 404
        return jsonify(rec)
    except Exception as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_http_server() -> None:
    cfg = get_config()
    _get_service()
    logger.info("Starting oG-Memory HTTP server on :%d", cfg.http_port)
    try:
        app.run(host="0.0.0.0", port=cfg.http_port, threaded=True)
    finally:
        _shutdown_service()


if __name__ == "__main__":
    _run_http_server()
