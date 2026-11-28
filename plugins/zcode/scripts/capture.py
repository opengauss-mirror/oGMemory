import ipaddress
import json
import re
import time
from pathlib import Path

from state import digest


EVENT_TYPES = {
    "SessionStart": "session_start", "UserPromptSubmit": "user_message",
    "PreToolUse": "tool_start", "PermissionRequest": "permission_request",
    "PostToolUse": "tool_success", "PostToolUseFailure": "tool_failure", "Stop": "turn_end",
}
BODY_FIELDS = {
    "prompt": ("prompt",), "assistant_message": ("responseText", "last_assistant_message", "responsePreview"),
    "tool_input": ("toolInput", "tool_input"), "tool_response": ("toolResponse", "tool_response"),
    "error": ("error_details", "error"), "reason": ("reason",),
}
TRACE_RULES = tuple(
    (re.compile(rule["pattern"], re.ASCII | (re.IGNORECASE if "i" in rule["flags"] else 0)),
     rule["replacement"].replace("$1", r"\g<1>"), rule.get("validator"))
    for rule in json.loads(Path(__file__).with_name("trace_redaction_rules.json").read_text(encoding="utf-8"))
)
SECRET = re.compile(
    r"(?i)\b((?:authorization\s*[:=]\s*(?:bearer|basic)\s+|bearer\s+)|"
    r"(?:[\w-]*(?:api[_-]?key|password|secret|token|cookie)[\w-]*)[\"']?\s*[:=]\s*)"
    r"""(?:"[^"]*"|'[^']*'|[^\s,;]+)"""
)
SENSITIVE_KEY = re.compile(r"(?i)(api[_-]?key|password|authorization|secret|token|cookie)")
KEY_LITERAL = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|(?:xox[a-z]?|xapp)-[A-Za-z0-9-]{20,})\b"
)
CLI_SECRET = re.compile(r"""(?i)(--(?:api[-_]?key|password|secret|token)[=\s]+)(?:"[^"]*"|'[^']*'|[^\s,;]+)""")
IDENTITY_DOCUMENT = re.compile(r"\b(?:[1-9]\d{16}[\dXx]|[EeGg]\d{8})\b", re.ASCII)
HOOK_IDENTIFIER = re.compile(r"[A-Za-z0-9._:-]{1,512}")


def get(event, *names):
    return next((event[name] for name in names if event.get(name) is not None), None)


def redact_ipv6(match):
    try:
        ipaddress.IPv6Address(match[0])
    except ValueError:
        return match[0]
    return "[REDACTED_IP]"


def sanitize(value):
    if isinstance(value, str):
        for pattern, replacement, validator in TRACE_RULES:
            value = pattern.sub(redact_ipv6 if validator == "ipv6" else replacement, value)
        value = CLI_SECRET.sub(r"\1[REDACTED]", value)
        value = SECRET.sub(r"\1[REDACTED]", value)
        value = KEY_LITERAL.sub("[REDACTED]", value)
        return IDENTITY_DOCUMENT.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {sanitize(key): "[REDACTED]" if SENSITIVE_KEY.search(key) else sanitize(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def text_value(value, limit):
    value = sanitize(value)
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return value.encode()[:limit].decode(errors="ignore")


def hook_identifier(value):
    if isinstance(value, str) and HOOK_IDENTIFIER.fullmatch(value):
        return value
    return None


def build_trace(event, name, identity, event_id, turn, trace_id, body_fields, content):
    tool = hook_identifier(get(event, "toolCallId", "tool_use_id"))
    fields = {
        "event_type": EVENT_TYPES[name],
        "tool_name": get(event, "toolName", "tool_name"),
        "terminal_state": {"PostToolUse": "success", "PostToolUseFailure": "failure"}.get(name),
        "start_source": event.get("source"),
        "model": event.get("model"),
        "agent_type": get(event, "agentName", "agent_type"),
        "is_interrupt": get(event, "isInterrupt", "is_interrupt"),
        "stop_hook_active": get(event, "stopHookActive", "stop_hook_active"),
    }
    fields = {key: text_value(value, 512) if isinstance(value, str) else value
              for key, value in fields.items() if value is not None}
    fields["trace_id"] = trace_id
    if turn := hook_identifier(turn):
        fields["turn_id"] = turn
    if tool:
        fields.update(tool_use_id=tool, object_id=tool)
    for field, value in body_fields.items():
        rendered = text_value(value, 12000)
        fields[field + "_size"] = len(value.encode()) if isinstance(value, str) else len(
            json.dumps(value, ensure_ascii=False).encode())
        if content and fields[field + "_size"] > 12000:
            fields["truncated"] = True
        if content:
            fields[field] = rendered
    payload = {
        "account_id": identity["accountId"], "user_id": identity["userId"],
        "agent_id": identity["agentId"], "session_id": identity["sessionId"],
        "request_id": event_id, "trace_id": trace_id,
        "timestamp": event.get("timestamp"), "operation": "zcode_hook", "event": fields,
    }
    # JSON escaping can enlarge the wire payload beyond the sum of the text lengths.
    for field in BODY_FIELDS:
        if len(json.dumps(payload, ensure_ascii=False).encode()) <= 63000:
            break
        if field in fields:
            fields.pop(field)
            fields["truncated"] = True
    return payload


def capture(queue, event, identity, trace_enabled, write_enabled, trace_content, force_extract):
    name = get(event, "hookEventName", "hook_event_name")
    session = identity["sessionId"]
    turn = get(event, "turnId", "turn_id")
    if name not in EVENT_TYPES or not session or not (trace_enabled or write_enabled):
        return
    # Exclude compatibility aliases that contain a new temporary path on every invocation.
    source = {key: value for key, value in event.items() if key not in {"transcriptPath", "transcript_path"}}
    event_id = digest([queue.partition, source])
    trace_id = hook_identifier(get(event, "traceId", "trace_id")) or event_id
    body_fields = {field: value for field, aliases in BODY_FIELDS.items()
                   if (value := get(event, *aliases)) is not None}
    with queue.transaction() as db:
        if name == "SessionStart":
            queue.prune(db)
        count = db.execute("SELECT count(*) FROM delivery WHERE partition=? AND state!='done'",
                           (queue.partition,)).fetchone()[0]
        if count > queue.limit - 2:
            if trace_enabled:
                db.execute("INSERT OR IGNORE INTO capture_gap VALUES(?,?)", (queue.partition, session))
            return False
        if not db.execute("INSERT OR IGNORE INTO seen(partition,id,created) VALUES(?,?,?)",
                          (queue.partition, event_id, time.time())).rowcount:
            return
        if trace_enabled:
            trace = build_trace(event, name, identity, event_id, turn, trace_id, body_fields, trace_content)
            if name == "Stop" and db.execute("DELETE FROM capture_gap WHERE partition=? AND session=?",
                                             (queue.partition, session)).rowcount:
                trace["event"]["capture_incomplete"] = True
            queue.enqueue(db, session, "trace", trace)
        if not (write_enabled and turn):
            return
        if name == "UserPromptSubmit":
            prompt = text_value(body_fields.get("prompt") or "", 64000)
            db.execute("INSERT OR IGNORE INTO turn(partition,session,id,prompt,created) VALUES(?,?,?,?,?)",
                       (queue.partition, session, turn, prompt, time.time()))
        if name != "Stop":
            return
        row = db.execute("SELECT prompt,stopped FROM turn WHERE partition=? AND session=? AND id=?",
                         (queue.partition, session, turn)).fetchone()
        assistant = body_fields.get("assistant_message")
        if row is None or not isinstance(assistant, str) or not assistant.strip():
            return
        prompt, stopped = row
        messages = []
        if not stopped and prompt.strip():
            messages.append({"role": "user", "content": prompt, "id": digest([queue.partition, session, turn, "user"])})
        messages.append({"role": "assistant", "content": text_value(assistant, 64000),
                         "id": digest([event_id, "assistant"])})
        queue.enqueue(db, session, "write", {
            **identity, "client_request_id": event_id, "messages": messages,
            "forceExtract": force_extract or (not stopped and any(
                marker in prompt.lower() for marker in ("remember", "记住", "记下", "保存到记忆"))),
        })
        db.execute("UPDATE turn SET stopped=stopped+1,prompt='' WHERE partition=? AND session=? AND id=?",
                   (queue.partition, session, turn))
