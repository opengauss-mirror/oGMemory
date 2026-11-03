"""Capture messages submitted through the MCP memory-write boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import uuid4

from core.models import RequestContext
from skills.sanitizer import sanitize_trace_text

logger = logging.getLogger(__name__)
_CONTENT_CHARS = 10_000


def validate_trace_messages(messages: Any) -> None:
    """Validate the containers consumed by native-message conversion."""
    if not isinstance(messages, list):
        raise ValueError("traceMessages must be a list")
    for index, message in enumerate(messages):
        path = f"traceMessages[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{path} must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role.strip() not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{path}.role must be system, user, assistant, or tool")
        content = message.get("content")
        if role.strip() != "tool" and content is not None:
            if not isinstance(content, (str, list)):
                raise ValueError(f"{path}.content must be text or a list of blocks")
            if isinstance(content, list) and any(
                not isinstance(block, dict) or not isinstance(block.get("type"), str) for block in content
            ):
                raise ValueError(f"{path}.content blocks must be objects with a string type")
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ValueError(f"{path}.tool_calls must be a list")
            if any(not isinstance(call, dict) or not isinstance(call.get("function", {}), dict) for call in tool_calls):
                raise ValueError(f"{path}.tool_calls entries must be objects with a function object")


def message_events(messages: list[dict[str, Any]]) -> list[dict]:
    """Convert ordered message blocks to canonical lifecycle events."""
    events, calls = [], {}
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = message.get("content", "")
        call_id = str(message.get("tool_call_id", ""))
        if role == "tool" and call_id and call_id in calls:
            events.append({
                "event_type": "tool_failure" if message.get("is_error") else "tool_success",
                "tool_name": calls.pop(call_id), "tool_use_id": call_id,
                "tool_response": content,
            })
            continue
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        texts, reasoning = [], []
        model_reasoning = message.get("reasoning_content") or message.get("reasoning")
        if role == "assistant" and model_reasoning:
            reasoning.append(str(model_reasoning))

        def flush_text():
            text = "\n".join(texts)
            if role == "assistant" and (text or reasoning):
                events.append({"event_type": "assistant_message", "assistant_message": text,
                               "reasoning": "\n".join(reasoning)})
            elif role in {"user", "system"} and text:
                events.append({"event_type": f"{role}_message", "prompt": text})
            elif role == "tool" and text:
                call_id = str(message.get("tool_call_id", ""))
                events.append({"event_type": "tool_message", "tool_name": message.get("name") or calls.get(call_id, "unknown_tool"),
                               "tool_use_id": call_id, "content": text})
            texts.clear()
            reasoning.clear()

        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in {"text", "output_text", "input_text"}:
                if block.get("text") is not None:
                    texts.append(str(block["text"]))
            elif kind in {"thinking", "reasoning"} and role == "assistant":
                reasoning.append(str(block.get("thinking") or block.get("text") or ""))
            elif kind == "tool_use" and role == "assistant":
                flush_text()
                call_id = str(block.get("id", ""))
                calls[call_id] = block.get("name", "unknown_tool")
                events.append({"event_type": "tool_start", "tool_name": calls[call_id],
                               "tool_use_id": call_id, "tool_input": block.get("input", {})})
            elif kind == "tool_result":
                flush_text()
                call_id = str(block.get("tool_use_id", ""))
                events.append({"event_type": "tool_failure" if block.get("is_error") else "tool_success",
                               "tool_name": calls.get(call_id, "unknown_tool"), "tool_use_id": call_id,
                               "tool_response": block.get("content", "")})
        flush_text()
        for call in (message.get("tool_calls") or []) if role == "assistant" else []:
            function = call.get("function", {})
            call_id = str(call.get("id", ""))
            calls[call_id] = function.get("name", "unknown_tool")
            events.append({"event_type": "tool_start", "tool_name": calls[call_id],
                           "tool_use_id": call_id, "tool_input": function.get("arguments", "")})
    return events


def capture_write_trace(
    service: Any,
    ctx: RequestContext,
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    client_request_id: str | None = None,
) -> bool:
    """Persist an ordered, bounded write batch and finalize its turn."""
    try:
        request_id = client_request_id or uuid4().hex
        identity = [ctx.account_id, ctx.user_id, ctx.agent_id, session_id, request_id]
        batch = hashlib.sha256(
            json.dumps([identity, messages], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        events = message_events(messages)
        assistant = ""
        for event in events:
            for field in ("prompt", "assistant_message", "reasoning", "tool_input", "tool_response", "content"):
                if field not in event:
                    continue
                value = event[field]
                content = sanitize_trace_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
                event[field] = content[:_CONTENT_CHARS]
                if len(content) > _CONTENT_CHARS:
                    event["truncated"] = True
            if event.get("assistant_message"):
                assistant = event["assistant_message"]

        end = {"event_type": "turn_end", "status": "ok", "boundary": "mcp_write"}
        if assistant:
            end["assistant_message"] = assistant
        events.append(end)
        for index, event in enumerate(events):
            stored = _append(service, ctx, batch, index, event, session_id=session_id)
        result = service.finalize_session_trace({
            "_ctx": ctx,
            "sessionId": session_id,
            "stopEventIndex": stored["event_index"],
        })
        if not result.get("ok"):
            raise RuntimeError("trace finalization unavailable")
        return True
    except Exception as exc:
        logger.warning("MCP write trace capture failed: %s", type(exc).__name__)
        return False


def _append(
    service: Any, ctx: RequestContext, batch: str, index: int, event: dict[str, Any],
    *, session_id: str,
) -> dict[str, Any]:
    event_id = hashlib.sha256(f"mcp-write:{batch}:{index}".encode()).hexdigest()
    result = service.add_session_trace({
        "_ctx": ctx,
        "sessionId": session_id,
        "eventId": event_id,
        "hookType": event["event_type"],
        "event": {**event, "write_batch_id": batch},
    })
    if not result.get("ok"):
        raise RuntimeError("trace persistence unavailable")
    return result
