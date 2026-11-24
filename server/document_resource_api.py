"""Authenticated HTTP entry points for knowledge documents."""

from dataclasses import asdict
from io import BytesIO
from threading import Lock
from uuid import UUID

from flask import Blueprint, jsonify, request, send_file

from core.document_backend import DocumentBackendError
from core.resource_models import DocumentResource
from server.document_resource_store import DocumentResourceStoreError


def _read_upload() -> tuple[str, str, bytes]:
    files = request.files.getlist("file")
    if len(files) != 1:
        raise ValueError("one multipart file field is required")
    upload = files[0]
    filename = upload.filename
    if not filename:
        raise ValueError("filename is required")
    return filename, upload.mimetype or "application/octet-stream", upload.read()


def _public(resource: DocumentResource) -> dict[str, str]:
    return {
        "id": str(resource.resource_id),
        "uri": resource.uri,
        "filename": resource.filename,
        "media_type": resource.media_type,
    }


def _json_body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    return body


def create_document_resource_blueprint(create_runtime, error_response):
    blueprint = Blueprint("document_resources", __name__, url_prefix="/api/v1")
    lock = Lock()
    runtime = None

    @blueprint.after_request
    def prevent_caching(response):
        response.headers["Cache-Control"] = "private, no-store"
        return response

    def authenticated_runtime():
        nonlocal runtime
        with lock:
            if runtime is None:
                runtime = create_runtime()
        return runtime, runtime.authenticate(dict(request.headers))

    def operation_error(exc):
        if isinstance(exc, DocumentBackendError):
            return jsonify(error="Knowledge base operation failed"), 502
        if isinstance(exc, DocumentResourceStoreError):
            return jsonify(error="Knowledge base catalog is unavailable"), 503
        return error_response(exc)

    @blueprint.route("/knowledge-bases", methods=["GET", "POST"])
    @blueprint.route("/knowledge-bases/<uuid:knowledge_base_id>", methods=["PATCH", "DELETE"])
    def catalog(knowledge_base_id: UUID | None = None):
        try:
            runtime, ctx = authenticated_runtime()
            if knowledge_base_id is None:
                if request.method in {"GET", "HEAD"}:
                    return jsonify(runtime.list_knowledge_bases(ctx))
                return jsonify(runtime.create_knowledge_base(ctx, _json_body())), 201
            if request.method == "PATCH":
                return jsonify(runtime.update_knowledge_base(ctx, knowledge_base_id, _json_body()))
            runtime.delete_knowledge_base(ctx, knowledge_base_id)
            return "", 204
        except Exception as exc:
            return operation_error(exc)

    @blueprint.route("/resources/documents", methods=["GET", "POST"])
    @blueprint.route("/resources/documents/<uuid:resource_id>", methods=["GET", "PUT", "DELETE"])
    @blueprint.get("/resources/documents/<uuid:resource_id>/content")
    @blueprint.get("/resources/documents/scope")
    @blueprint.post("/resources/documents/search")
    def dispatch(resource_id: UUID | None = None):
        try:
            runtime, ctx = authenticated_runtime()
            if request.path.endswith("/scope"):
                account_id, accounts = runtime.scope(ctx)
                return jsonify(account_id=account_id, accounts=accounts)
            raw_knowledge_base = request.args.get("knowledge_base_id")
            knowledge_base_id = UUID(raw_knowledge_base) if raw_knowledge_base is not None else None
            searching = request.path.endswith("/search")
            write = request.method in {"POST", "PUT", "DELETE"} and not searching
            with runtime.open_documents(ctx, knowledge_base_id, write=write) as service:
                if searching:
                    body = _json_body()
                    query, limit = body.get("query"), body.get("limit", 10)
                    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
                        raise ValueError("query must contain 1 to 2000 characters")
                    if type(limit) is not int or not 1 <= limit <= 100:
                        raise ValueError("limit must be an integer between 1 and 100")
                    hits = service.search(query.strip(), limit=limit)
                    return jsonify(hits=[asdict(hit) for hit in hits])
                if resource_id is None:
                    if request.method in {"GET", "HEAD"}:
                        page = int(request.args.get("page", "1"))
                        page_size = int(request.args.get("page_size", "20"))
                        if not 1 <= page <= 2_147_483_647 or not 1 <= page_size <= 100:
                            raise ValueError(
                                "page must be between 1 and 2147483647 and page_size between 1 and 100"
                            )
                        resources, total = service.list_documents(page=page, page_size=page_size)
                        return jsonify(
                            items=[_public(resource) for resource in resources],
                            page=page,
                            page_size=page_size,
                            total=total,
                        )
                    resource = service.ingest(*_read_upload())
                    return jsonify(**_public(resource), indexing_submitted=True), 201
                if request.method in {"GET", "HEAD"}:
                    if request.path.endswith("/content"):
                        resource, data = service.download(resource_id)
                        return send_file(
                            BytesIO(data),
                            mimetype=resource.media_type,
                            download_name=resource.filename,
                            as_attachment=True,
                            max_age=0,
                        )
                    resource, status = service.status(resource_id)
                    return jsonify(
                        **_public(resource),
                        index_state=status.state,
                        chunk_count=status.chunk_count,
                    )
                if request.method == "PUT":
                    resource, pending = service.replace(resource_id, *_read_upload())
                    return jsonify(
                        **_public(resource), indexing_submitted=True, cleanup_pending=pending
                    )
                if request.method == "DELETE":
                    service.delete(resource_id)
                    return "", 204
                return "", 405
        except Exception as exc:
            return operation_error(exc)

    return blueprint
