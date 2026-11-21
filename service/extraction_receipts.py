"""Compact extraction receipts backed by native durable write outcomes."""

from copy import deepcopy

from core.write_contract import OUTCOMES_KEY, normalize_write_result

KEY = "memory_extraction_receipts"


def compact_plans(plans):
    result = []
    for plan in plans:
        entry = {
            "action": plan.get("action", "skip"),
            "target_uri": plan.get("target_uri", ""),
            "merged_fields": {"status": (plan.get("merged_fields") or {}).get("status", "")},
        }
        if entry not in result:
            result.append(entry)
    return result


def read(buffer, ctx, request_id, *, include_accepted=False):
    """Read an owner-scoped completed receipt or its buffered source continuation."""
    owner = {name: getattr(ctx, name) for name in ("account_id", "user_id", "agent_id")}
    if buffer.session_id != ctx.session_id or any(
        getattr(buffer.meta, name) != value for name, value in owner.items()
    ):
        return None
    receipt = buffer.runtime_extensions.get(OUTCOMES_KEY, {}).get("entries", {}).get(request_id)
    if receipt is not None:
        result = normalize_write_result(deepcopy(receipt), request_id)
        result.update(client_request_id=request_id, plans=compact_plans(result.get("plans", [])))
        if result.get("source_status") == "ARCHIVED" and result.get("semantic_status") == "COMPLETE":
            result["status"] = "completed"
            return result
        message_ids = set(result.get("source_message_ids", [])) - set(result.get("archived_source_ids", []))
        if (include_accepted and result.get("source_status") == "BUFFERED" and message_ids
                and message_ids <= {message.id for message in buffer.messages}):
            result.update(status="accepted", message_ids=sorted(message_ids))
            return result
        return None
    legacy = buffer.runtime_extensions.get(KEY, {}).get(request_id)
    if legacy and legacy.get("owner") == {**owner, "session_id": ctx.session_id}:
        result = legacy["result"]
        if result.get("status") == "completed" and buffer.request_completed(request_id):
            return deepcopy(result)
        message_ids = set(result.get("message_ids", []))
        if (include_accepted and result.get("status") == "accepted" and message_ids
                and message_ids <= {message.id for message in buffer.messages}):
            return deepcopy(result)
    return None
