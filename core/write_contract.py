"""Shared write acknowledgements for HTTP, MCP and durable request receipts."""

from copy import deepcopy
from functools import wraps

from core.errors import (
    ConcurrentModificationError, EmbeddingError, ExtractionBudgetError,
    ExtractionIncompleteError, LLMTransientError, OperationDeadlineExceededError,
    ValidationError, VectorIndexError,
)

OUTCOMES_KEY = "memory_write_outcomes"
IN_PROGRESS = {"ACCEPTED", "RUNNING", "RECOVERING", "PROJECTING"}
PUBLIC_FIELDS = (
    "request_id", "request_status", "source_status", "semantic_status",
    "projection_status", "retry_action", "error_detail", "archive_id",
    "accepted", "terminal", "http_status",
    "paused_projection_count",
)


def error_fields(exc, request_id=""):
    """Preserve the originating failure independently from recovery policy."""
    name = type(exc).__name__
    code, component, status, action = "WRITE_INTERNAL_ERROR", "memory", 500, "manual_review"
    if isinstance(exc, ExtractionBudgetError):
        code, component, status = "EXTRACTION_CONTEXT_BUDGET", "extraction", 500
    elif isinstance(exc, LLMTransientError):
        code, component, status, action = "MODEL_TEMPORARILY_UNAVAILABLE", "model", 503, "retry_same_request"
    elif isinstance(exc, OperationDeadlineExceededError):
        code, component, status, action = "WRITE_DEADLINE_EXCEEDED", "memory", 504, "retry_same_request"
    elif isinstance(exc, ExtractionIncompleteError):
        code, component, status = "EXTRACTION_SEMANTIC_INCOMPLETE", "extraction", 500
    elif isinstance(exc, ConcurrentModificationError):
        code, status, action = "WRITE_VERSION_CONFLICT", 409, "retry_same_request"
    elif isinstance(exc, (EmbeddingError, VectorIndexError)):
        code, component = "VECTOR_DEPENDENCY_ERROR", "vector"
    elif name in {"AGFSClientError", "AGFSError"}:
        code, component = "AGFS_DEPENDENCY_ERROR", "agfs"
    elif name in {"SessionMutationBusyError", "SessionOperationInProgress"}:
        code, status, action = "SESSION_WRITE_BUSY", 409, "retry_same_request"
        if getattr(exc, "retryable", True) is False:
            action = "manual_review"
    elif isinstance(exc, FileExistsError) or name == "ConfigRevisionError":
        code, status, action = "WRITE_VERSION_CONFLICT", 409, "retry_same_request"
    elif name == "UsagePersistenceError":
        code, component = "MODEL_USAGE_PERSISTENCE_FAILED", "model_usage"
    elif name in {"RecoveryNeedsConfirmationError", "RecoveryPersistenceError",
                  "WriteOutcomeUnknownError", "SessionMutationLostError"}:
        code, component, status = "WRITE_CONFIRMATION_REQUIRED", "persistence", 409
    elif isinstance(exc, (ValueError, ValidationError)):
        code, status = "WRITE_INVALID_ARGUMENT", 400
    elif isinstance(exc, (ConnectionError, TimeoutError)):
        code, status, action = "WRITE_DEPENDENCY_UNAVAILABLE", 503, "retry_same_request"
    causes = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen and len(causes) < 5:
        seen.add(id(current))
        causes.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    upstream = next((value for cause in causes
                     for value in (getattr(cause, "status_code", None),
                                   getattr(getattr(cause, "response", None), "status_code", None))
                     if type(value) is int and 100 <= value <= 599), None)
    if component in {"agfs", "vector"}:
        if upstream in {401, 403, 404, 409}:
            status = upstream
            action = "retry_same_request" if upstream == 409 else "manual_review"
        elif upstream in {408, 425, 429, 502, 503, 504} or any(
                isinstance(cause, (ConnectionError, TimeoutError)) or type(cause).__name__ in
                {"ConnectTimeout", "ReadTimeout", "ConnectError", "ConnectionError"} for cause in causes):
            status, action = 503, "retry_same_request"
    return {"http_status": status, "error_detail": {
        "code": code, "component": component, "request_id": request_id,
        "retry_action": action, "upstream_http_status": upstream,
        "cause": {"type": name, "message": str(exc)[:1000],
                  "underlying": [{"type": type(cause).__name__, "message": str(cause)[:1000]}
                                 for cause in causes[1:]]},
    }}


def normalize_write_result(result, request_id=""):
    result = dict(result)
    request_id = str(result.get("request_id") or result.get("client_request_id") or request_id)
    result["request_id"] = request_id
    state = result.get("request_status")
    if state is None:
        if result.get("degraded") or result.get("recovery_mode") == "source_only":
            state = "DEGRADED"
        elif result.get("recovery_task"):
            state = "RECOVERING"
        elif result.get("status") == "needs_confirmation":
            state = "NEEDS_CONFIRMATION"
        elif result.get("ok") is False:
            state = "FAILED"
        elif result.get("status") in {"queued", "processing"}:
            state = "RUNNING"
        elif result.get("status") in {"accumulating", "no_new_messages"}:
            state = "ACCEPTED"
        else:
            state = "UNKNOWN"
    result["request_status"] = state
    result.setdefault("source_status", "UNKNOWN")
    result.setdefault("semantic_status", "UNKNOWN")
    result.setdefault("projection_status", "UNKNOWN")
    result.setdefault("accepted", state in IN_PROGRESS | {"SUCCEEDED", "DEGRADED"}
                      or (state == "UNKNOWN" and result.get("ok") is True))
    result["terminal"] = state in {"SUCCEEDED", "DEGRADED", "NEEDS_CONFIRMATION", "FAILED"}
    detail = deepcopy(result.get("error_detail"))
    failure_action = "retry_same_request" if state == "FAILED" and result.get("retryable") is True else "manual_review"
    if state in IN_PROGRESS:
        action = "poll"
    elif state == "SUCCEEDED":
        action = "none"
    elif state in {"DEGRADED", "NEEDS_CONFIRMATION"}:
        action = "manual_review"
    else:
        action = detail.get("retry_action", failure_action) if detail else failure_action
    if state == "ACCEPTED" and (result.get("source_status") == "BUFFERED"
                                or result.get("status") == "no_new_messages"):
        action = "none"
    result["retry_action"] = action
    if detail:
        detail.update(request_id=request_id, retry_action=action)
        result["error_detail"] = detail
    if state in IN_PROGRESS:
        result.update(ok=True, retryable=False, http_status=202)
    elif state == "SUCCEEDED":
        result.update(ok=True, retryable=False, http_status=200)
    elif state == "DEGRADED":
        result.update(ok=False, retryable=False, status="degraded", http_status=200)
    elif state in {"FAILED", "NEEDS_CONFIRMATION"}:
        result["ok"] = False
        status = next((result[key] for key in ("http_status", "status")
                       if type(result.get(key)) is int and 400 <= result[key] <= 599), None)
        result["http_status"] = status or (409 if state == "NEEDS_CONFIRMATION" else
                                          503 if action == "retry_same_request" else 500)
    else:
        result.setdefault("http_status", 200)
    return result


def write_http_status(result):
    status = result.get("http_status")
    if isinstance(status, int) and 200 <= status <= 599:
        return status
    status = result.get("status")
    if isinstance(status, int) and 400 <= status <= 599:
        return status
    if result.get("retryable") is True:
        return 503
    return 400 if result.get("ok") is False else 200


def write_request_contract(function):
    @wraps(function)
    def wrapped(self, params):
        request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
        result = function(self, params)
        if "outbox_tracking_complete" in result:
            result = self._resolve_write_outcome(result)
        return normalize_write_result(result, request_id)
    return wrapped


def needs_projection_refresh(outcome):
    return (outcome.get("outbox_tracking_complete") is True
            and outcome.get("source_status") in {"ARCHIVED", "RAW_ARCHIVED"}
            and outcome.get("semantic_status") in {"COMPLETE", "DEGRADED"}
            and (outcome.get("projection_status", "UNKNOWN") in {"PENDING", "UNKNOWN"}
                 or outcome.get("paused_projection_count", 0) > 0))


def store_outcome(buffer, request_id, outcome, *, resolve_outcome=None):
    entries = buffer.runtime_extensions.setdefault(OUTCOMES_KEY, {"version": 1, "entries": {}})["entries"]
    outcome = deepcopy(outcome)
    projected = (outcome.get("outbox_tracking_complete") is True
                 and outcome.get("projection_status") in {"DONE", "NOT_REQUIRED"}
                 and outcome.get("source_status") in {"ARCHIVED", "RAW_ARCHIVED"}
                 and (outcome.get("semantic_status"), outcome.get("request_status"))
                 in {("COMPLETE", "SUCCEEDED"), ("DEGRADED", "DEGRADED")})
    if resolve_outcome is not None and "outbox_tracking_complete" in outcome and not projected:
        outcome = resolve_outcome(outcome)
    entries.pop(request_id, None)
    entries[request_id] = outcome
    retained = set(buffer.completed_request_ids)
    for key, value in list(entries.items()):
        if len(entries) <= 256:
            break
        if key in retained:
            continue
        if resolve_outcome is not None and value.get("request_status") == "PROJECTING":
            value = resolve_outcome(value)
            entries[key] = value
        empty_buffer = (value.get("request_status") == "ACCEPTED" and value.get("source_status") == "BUFFERED"
                        and value.get("source_message_ids") == [])
        if empty_buffer or value.get("request_status") in {"SUCCEEDED", "DEGRADED", "FAILED", "NEEDS_CONFIRMATION"}:
            entries.pop(key)
    return outcome


def merge_source_outcome(pending, receipt, batch_ids):
    result = {**pending, **receipt}
    source_ids = set(pending.get("source_message_ids", []))
    archived_ids = set(pending.get("archived_source_ids", [])) | (source_ids & set(batch_ids))
    references = [tuple(item) for item in pending.get("outbox_events", []) + receipt.get("outbox_events", [])]
    result.update(source_message_ids=sorted(source_ids), archived_source_ids=sorted(archived_ids),
                  outbox_events=[list(item) for item in dict.fromkeys(references)],
                  outbox_tracking_complete=pending.get("outbox_tracking_complete", True) and receipt.get("outbox_tracking_complete", False))
    result["raw_source_uris"] = list(dict.fromkeys(pending.get("raw_source_uris", []) + receipt.get("raw_source_uris", [])))
    archives = dict(pending.get("archive_receipts", {}))
    archives[receipt["archive_id"]] = receipt["source_sha256"]
    result["archive_receipts"] = archives
    degraded = pending.get("semantic_degraded", False) or receipt.get("semantic_status") == "DEGRADED"
    result["semantic_degraded"] = degraded
    if source_ids - archived_ids:
        result.update(source_status="BUFFERED", semantic_status="PENDING", request_status="ACCEPTED")
    elif degraded:
        result.update(semantic_status="DEGRADED", request_status="DEGRADED")
    return result


def projection_outcome(outcome, store, *, paused_event_types=()):
    """Derive completion from exact durable event identities."""
    result = deepcopy(outcome)
    result.pop("paused_projection_count", None)
    if not result.get("outbox_tracking_complete"):
        projection = "UNKNOWN"
    else:
        references = result.get("outbox_events", [])
        if not references:
            projection = "NOT_REQUIRED"
        else:
            statuses, paused = [], 0
            for uri, event_id in references:
                if paused_event_types:
                    event = store.get_event(uri.split("/")[2], event_id)
                    if event is not None and event.uri == uri:
                        if event.event_type in paused_event_types and event.status in {"PENDING", "PROCESSING"}:
                            paused += 1
                            continue
                        statuses.append(event.status)
                    else:
                        statuses.append(None)
                else:
                    statuses.append(store.get_event_status(uri, event_id))
            if paused:
                result["paused_projection_count"] = paused
            projection = ("NOT_REQUIRED" if not statuses else
                          "DONE" if all(status == "DONE" for status in statuses) else
                          "FAILED" if any(status in {"FAILED", "DLQ", "CANCELLED"} for status in statuses) else
                          "UNKNOWN" if any(status is None for status in statuses) else "PENDING")
    result["projection_status"] = projection
    if result.get("semantic_status") == "DEGRADED":
        result["request_status"] = "DEGRADED"
    elif result.get("semantic_status") == "COMPLETE" and result.get("source_status") in {"ARCHIVED", "RAW_ARCHIVED"}:
        result["request_status"] = ("SUCCEEDED" if projection in {"DONE", "NOT_REQUIRED"} else
                                    "NEEDS_CONFIRMATION" if projection == "FAILED" or not result.get("outbox_tracking_complete")
                                    else "PROJECTING")
    return normalize_write_result(result)
