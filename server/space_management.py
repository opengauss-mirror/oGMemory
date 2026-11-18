"""Shared-space management routes and the public retrieval endpoint."""

from __future__ import annotations

import logging
import re
from functools import wraps
from uuid import uuid4

from flask import Blueprint, jsonify, request

from server.auth import AuthenticationError, ControlPlaneDisabledError

logger = logging.getLogger(__name__)
_REVISION = re.compile(r'"(0|[1-9][0-9]*)"')


def _context(service):
    auth = service.get_auth_service()
    if not auth.role_control_active():
        raise ControlPlaneDisabledError("Shared spaces require role control")
    identity = auth.resolve_identity(dict(request.headers))
    if identity is None:
        raise AuthenticationError("A registered identity is required")
    return service.build_context(
        {"agentId": request.headers.get("X-Agent-Id", "")}, identity=identity,
    )


def _body():
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required")
    return value


def _integer(name, default=None, *, maximum=None):
    raw = request.args.get(name)
    if raw is None:
        return default
    if not raw.isdecimal():
        raise ValueError(f"{name} must be a positive integer")
    value = int(raw)
    if value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _page():
    return {"limit": _integer("limit", 50, maximum=100), "cursor": request.args.get("cursor")}


def _mutation(*, create=False):
    key = request.headers.get("Idempotency-Key", "")
    if not 1 <= len(key) <= 128 or any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError("Idempotency-Key must contain 1 to 128 printable ASCII characters")
    if create:
        return {"request_id": key}
    match = _REVISION.fullmatch(request.headers.get("If-Match", ""))
    if match:
        revision = int(match.group(1))
    elif request.headers.get("If-None-Match") == "*":
        revision = 0
    else:
        from server.space_rules import SpaceError

        raise SpaceError(428, "PRECONDITION_REQUIRED", 'Provide If-Match: "revision" or If-None-Match: *')
    return {"request_id": key, "expected_rev": revision}


def _endpoint(error_response):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            correlation = str(uuid4())
            try:
                result = function(*args, **kwargs)
                payload, status = result if isinstance(result, tuple) else (result, 200)
                response = jsonify({**payload, "request_id": correlation})
                response.status_code = status
                revision = payload.get("rule_rev", payload.get("grant_rev", payload.get("authz_rev")))
                if revision is not None:
                    response.set_etag(str(revision))
            except Exception as exc:
                from server.space_repository import SpaceError

                if isinstance(exc, SpaceError):
                    response = jsonify({"error": {"code": exc.code, "message": str(exc)}, "request_id": correlation})
                    response.status_code = exc.status
                elif isinstance(exc, (ValueError, AuthenticationError, ControlPlaneDisabledError, PermissionError)):
                    response = error_response(exc)
                    from flask import make_response

                    response = make_response(response)
                else:
                    logger.exception("Shared-space request failed: %s", correlation)
                    response = jsonify({"error": {"code": "SPACE_UNAVAILABLE", "message": "Shared-space service is unavailable"}, "request_id": correlation})
                    response.status_code = 503
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Request-Id"] = correlation
            if response.status_code in {429, 503}:
                response.headers["Retry-After"] = "5"
            return response
        return wrapped
    return decorate


def create_space_management_blueprint(get_service, error_response):
    """Mount on the authenticated management listener at /spaces."""
    blueprint = Blueprint("space_management", __name__, url_prefix="/spaces")
    endpoint = _endpoint(error_response)

    def repository_context():
        service = get_service()
        ctx = _context(service)
        return service.get_space_repository(), ctx

    @blueprint.route("", methods=["GET", "POST"])
    @endpoint
    def spaces():
        repo, ctx = repository_context()
        if request.method == "POST":
            return repo.create_space(ctx, _body(), **_mutation(create=True)), 201
        return repo.list_spaces(ctx, space_id=request.args.get("space_id"), **_page())

    @blueprint.get("/principals")
    @endpoint
    def principals():
        repo, ctx = repository_context()
        return repo.list_principals(ctx, kind=request.args.get("kind"), q=request.args.get("q"), space_id=request.args.get("space_id"), **_page())

    @blueprint.patch("/<space_id>")
    @endpoint
    def update(space_id):
        repo, ctx = repository_context()
        return repo.update_space(ctx, space_id, _body(), **_mutation())

    @blueprint.delete("/<space_id>")
    @endpoint
    def delete_space(space_id):
        repo, ctx = repository_context()
        return repo.delete_space(ctx, space_id, **_mutation())

    @blueprint.put("/<space_id>/members")
    @endpoint
    def members(space_id):
        repo, ctx = repository_context()
        return repo.set_members(ctx, space_id, _body(), **_mutation())

    @blueprint.put("/<space_id>/sources/<binding_id>")
    @endpoint
    def sources(space_id, binding_id):
        repo, ctx = repository_context()
        mutation = _mutation()
        return repo.put_source(ctx, space_id, binding_id, _body(), **mutation), 201 if mutation["expected_rev"] == 0 else 200

    @blueprint.delete("/<space_id>/sources/<binding_id>")
    @endpoint
    def delete_source(space_id, binding_id):
        repo, ctx = repository_context()
        return repo.delete_source(ctx, space_id, binding_id, **_mutation())

    @blueprint.put("/<space_id>/acl")
    @endpoint
    def acl(space_id):
        repo, ctx = repository_context()
        return repo.set_acl(ctx, space_id, _body(), **_mutation())

    @blueprint.get("/<space_id>/permissions")
    @endpoint
    def permissions(space_id):
        repo, ctx = repository_context()
        return {**repo.permissions(ctx, space_id, item_id=request.args.get("item_id")), "viewer_user_id": ctx.user_id}

    @blueprint.put("/<space_id>/rules/<rule_id>")
    @endpoint
    def save_rule(space_id, rule_id):
        repo, ctx = repository_context()
        mutation = _mutation()
        return repo.save_rule(ctx, space_id, rule_id, _body(), **mutation), 201 if mutation["expected_rev"] == 0 else 200

    @blueprint.delete("/<space_id>/rules/<rule_id>")
    @endpoint
    def delete_rule(space_id, rule_id):
        repo, ctx = repository_context()
        return repo.delete_rule(ctx, space_id, rule_id, **_mutation())

    @blueprint.get("/<space_id>/rules")
    @endpoint
    def rules(space_id):
        repo, ctx = repository_context()
        return repo.list_rules(ctx, space_id, rule_id=request.args.get("rule_id"), revision=_integer("revision"), **_page())

    @blueprint.get("/<space_id>/items")
    @endpoint
    def items(space_id):
        repo, ctx = repository_context()
        return repo.list_items(ctx, space_id, item_id=request.args.get("item_id"), state=request.args.get("state"), file=request.args.get("file"), **_page())

    @blueprint.get("/<space_id>/jobs")
    @endpoint
    def jobs(space_id):
        repo, ctx = repository_context()
        return repo.list_jobs(ctx, space_id, rule_id=request.args.get("rule_id"), generation=_integer("generation"), job_id=request.args.get("job_id"), state=request.args.get("state"), **_page())

    @blueprint.get("/<space_id>/audit")
    @endpoint
    def audit(space_id):
        repo, ctx = repository_context()
        return repo.list_audit(ctx, space_id, action=request.args.get("action"), rule_id=request.args.get("rule_id"), item_id=request.args.get("item_id"), from_time=request.args.get("from"), until=request.args.get("until"), **_page())

    return blueprint


def create_shared_search_blueprint(get_service, error_response):
    blueprint = Blueprint("shared_memory_search", __name__)

    @blueprint.post("/api/spaces/search")
    @_endpoint(error_response)
    def search():
        service = get_service()
        ctx = _context(service)
        body = _body()
        return service.search_shared_memory(
            ctx, query=body.get("query"), space_ids=body.get("space_ids"),
            kind=body.get("kind"), top_k=body.get("top_k", 10),
            max_context_tokens=body.get("max_context_tokens", 4096),
            mode=body.get("mode", "hybrid"), limit=body.get("limit"), cursor=body.get("cursor"),
        )

    return blueprint
