"""Authenticated scene management and runtime configuration endpoints."""

from flask import Blueprint, jsonify, request

from core.models import Role
from server.auth import build_admin_target_context


def _target_context(service, build_context, payload):
    if not service.get_auth_service().role_control_active():
        return build_context(payload)
    actor = build_context({**payload, "agentId": ""})
    if Role(actor.role) not in {Role.ROOT, Role.ADMIN}:
        return build_context(payload)
    account_id = str(payload.get("accountId") or payload.get("account_id") or actor.account_id).strip()
    user_id = str(payload.get("userId") or payload.get("user_id") or "").strip()
    agent_id = str(payload.get("agentId") or "").strip()
    owner = payload.get("owner_space")
    if owner:
        kind, separator, identifier = str(owner).partition(":")
        if not separator or kind not in {"user", "agent"} or not identifier:
            raise ValueError("owner_space must identify a user or agent")
        if kind == "user":
            user_id, agent_id = identifier, ""
        else:
            agent_id = identifier
    if agent_id and not user_id:
        agent = service.get_tenant_admin_service().get_agent(actor, account_id, agent_id)
        user_id = str(agent.get("owner_user_id") or "")
    return build_admin_target_context(
        service, actor, account_id=account_id,
        user_id=user_id or actor.user_id, agent_id=agent_id,
    )


def create_scene_blueprint(get_service, build_context, error_response):
    blueprint = Blueprint("scene_memory", __name__, url_prefix="/api/v1/scenes")

    def dispatch(operation):
        try:
            payload = dict(request.args)
            if request.method in {"POST", "PUT", "DELETE"}:
                body = request.get_json(silent=True)
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                payload.update(body)
            service = get_service()
            ctx = _target_context(service, build_context, payload)
            return jsonify(operation(service.get_scene_service(), ctx, payload))
        except Exception as exc:
            return error_response(exc)

    @blueprint.route("", methods=["GET", "POST"])
    def collection():
        return dispatch(lambda service, ctx, payload: (
            service.save(ctx, payload) if request.method == "POST" else
            service.list(ctx, owner_space=payload.get("owner_space"),
                         offset=int(payload.get("offset", 0)), limit=int(payload.get("limit", 20)))
        ))

    @blueprint.route("/configuration", methods=["GET", "PUT"])
    def configuration():
        return dispatch(lambda service, ctx, payload: (
            service.update_configuration(ctx, payload) if request.method == "PUT" else
            service.configuration(ctx, payload.get("owner_space"))
        ))

    @blueprint.route("/status", methods=["GET"])
    def status():
        return dispatch(lambda service, ctx, payload: service.status(ctx, payload.get("owner_space")))

    @blueprint.route("/flush", methods=["POST"])
    def flush():
        return dispatch(lambda service, ctx, payload: service.flush(ctx, payload.get("owner_space")))

    @blueprint.route("/<scene_id>", methods=["PUT", "DELETE"])
    def item(scene_id):
        return dispatch(lambda service, ctx, payload: (
            service.save(ctx, payload, scene_id=scene_id) if request.method == "PUT" else
            service.archive(ctx, scene_id, revision=payload.get("revision"),
                            owner_space=payload.get("owner_space"))
        ))

    return blueprint
