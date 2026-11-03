#!/usr/bin/env python3
"""Claude ``Bash`` PostToolUse adapter for oGMemory.

Claude authorizes and executes the original command before this hook runs. The
adapter filters only a complete foreground textual ``BashOutput`` and, when
the service confirms actual shortening after durable recovery storage, replaces
only the model-visible stdout/stderr fields of that structured result.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ogm_plugin_request import (  # noqa: E402, I001
    base_api_url,
    base_ctx,
    effective_tool_output_filter_enabled,
    http_plugin_headers,
    mcp_recovery_enabled,
)
from tool_output_delivery import build_filter_delivery  # noqa: E402


FILTER_BASH_TIMEOUT = 10.0
_CANDIDATE_PATTERNS = (
    (
        "test_runner",
        re.compile(
            r"(^|[\s;&|()])(?:python(?:\d+(?:\.\d+)?)?\s+-m\s+)?pytest\b",
            re.I,
        ),
    ),
    ("test_runner", re.compile(r"(^|[\s;&|()])py\.test\b", re.I)),
    (
        "test_runner",
        re.compile(r"(^|[\s;&|()])(?:tox|npm\s+(?:run\s+)?test)\b", re.I),
    ),
    (
        "test_runner",
        re.compile(
            r"(^|[\s;&|()])(?:python(?:\d+(?:\.\d+)?)?\s+)?"
            r"(?:\./)?tests/runtests\.py\b",
            re.I,
        ),
    ),
    ("diff", re.compile(r"(^|[\s;&|()])git\s+diff\b", re.I)),
    ("diff_maybe", re.compile(r"(^|[\s;&|()])git\s+(?:show|log\s+-p)\b", re.I)),
    (
        "diff",
        re.compile(r"(^|[\s;&|()])(?:cat|sed|head|tail)\s+[^;&|]*\.(?:patch|diff)\b", re.I),
    ),
    ("python_maybe", re.compile(r"(^|[\s;&|()])python(?:\d+(?:\.\d+)?)?\b", re.I)),
)
_BASH_OUTPUT_FIELDS = frozenset(
    {
        "stdout",
        "stderr",
        "rawOutputPath",
        "interrupted",
        "isImage",
        "backgroundTaskId",
        "backgroundedByUser",
        "assistantAutoBackgrounded",
        "dangerouslyDisableSandbox",
        "returnCodeInterpretation",
        "noOutputExpected",
        "structuredContent",
        "persistedOutputPath",
        "persistedOutputSize",
        "staleReadFileStateHint",
        "ghRateLimitHint",
        "gitOperation",
    }
)
_INCOMPLETE_OUTPUT_FIELDS = (
    "rawOutputPath",
    "structuredContent",
    "persistedOutputPath",
    "persistedOutputSize",
)


def log(message: str) -> None:
    print(f"[call_filter_bash] {message}", file=sys.stderr)


def classify_bash_candidate(command: str) -> str:
    """Return the server pre-category without changing command semantics."""

    compact = " ".join(str(command or "").strip().split())
    if not compact:
        return ""
    for category, pattern in _CANDIDATE_PATTERNS:
        if pattern.search(compact):
            return category
    return ""


# Preserve the private alias while using the neutral public classifier name.
_bash_filter_candidate = classify_bash_candidate


def _safe_reason(data: object) -> str:
    if not isinstance(data, dict):
        return "invalid_response"
    reason = data.get("reason")
    if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,80}", reason):
        return reason
    return "unknown"


def _post_filter_bash(body: dict) -> dict | None:
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            f"{base_api_url()}/api/v1/filter_bash",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=http_plugin_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=FILTER_BASH_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            status = getattr(response, "status", None) or response.getcode()
        data = json.loads(raw) if raw.strip() else {}
        log(
            f"state=response status={status} chars={len(raw)} latency_ms="
            f"{int((time.monotonic() - started) * 1000)}"
        )
        return data if isinstance(data, dict) else None
    except urllib.error.HTTPError as exc:
        log(
            f"state=http_error status={exc.code} latency_ms="
            f"{int((time.monotonic() - started) * 1000)}"
        )
    except urllib.error.URLError:
        log(f"state=transport_error latency_ms={int((time.monotonic() - started) * 1000)}")
    except TimeoutError:
        log(f"state=timeout latency_ms={int((time.monotonic() - started) * 1000)}")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
        log(f"state=parse_error latency_ms={int((time.monotonic() - started) * 1000)}")
    except Exception as exc:  # noqa: BLE001 - hooks must never replace native Bash on failure
        log(f"state=error type={type(exc).__name__}")
    return None


def _response(updated_tool_output: dict) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": updated_tool_output,
        }
    }


def _optional_bool(value: object) -> bool:
    return value is None or isinstance(value, bool)


def _complete_text_bash(hook: dict) -> tuple[dict, str] | None:
    """Return one complete text result with stdout followed by stderr."""

    tool_response = hook.get("tool_response")
    if not isinstance(tool_response, dict):
        log("state=skip reason=malformed_bash_result")
        return None
    if set(tool_response) - _BASH_OUTPUT_FIELDS:
        log("state=skip reason=unsupported_bash_schema")
        return None

    stdout = tool_response.get("stdout")
    stderr = tool_response.get("stderr")
    interrupted = tool_response.get("interrupted")
    if (
        not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or not isinstance(interrupted, bool)
    ):
        log("state=skip reason=malformed_bash_result")
        return None
    if interrupted:
        log("state=skip reason=interrupted_bash_result")
        return None

    is_image = tool_response.get("isImage")
    backgrounded_by_user = tool_response.get("backgroundedByUser")
    assistant_auto_backgrounded = tool_response.get("assistantAutoBackgrounded")
    if not all(
        _optional_bool(value)
        for value in (is_image, backgrounded_by_user, assistant_auto_backgrounded)
    ):
        log("state=skip reason=malformed_bash_result")
        return None
    if is_image is True:
        log("state=skip reason=image_bash_result")
        return None
    background_task_id = tool_response.get("backgroundTaskId")
    if background_task_id is not None and not isinstance(background_task_id, str):
        log("state=skip reason=malformed_bash_result")
        return None
    if background_task_id or backgrounded_by_user is True or assistant_auto_backgrounded is True:
        log("state=skip reason=background_bash_result")
        return None
    if any(
        field in tool_response and tool_response[field] is not None
        for field in _INCOMPLETE_OUTPUT_FIELDS
    ):
        log("state=skip reason=indirect_bash_output")
        return None

    separator = "\n" if stdout and stderr and not stdout.endswith("\n") else ""
    combined = stdout + separator + stderr
    if not combined:
        log("state=skip reason=empty_bash_output")
        return None
    return tool_response, combined


def _updated_bash_output(tool_response: dict, filtered_output: str) -> dict:
    updated = dict(tool_response)
    updated["stdout"] = filtered_output
    updated["stderr"] = ""
    return updated


def main() -> int:
    try:
        enabled = effective_tool_output_filter_enabled("Bash")
    except Exception as exc:  # noqa: BLE001 - configuration failure preserves native Bash
        log(f"state=fallback reason=config_unavailable type={type(exc).__name__}")
        return 0
    if not enabled:
        log("state=disabled")
        return 0

    try:
        raw = sys.stdin.read()
        hook = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        log("state=malformed")
        return 0
    if (
        not isinstance(hook, dict)
        or hook.get("tool_name") != "Bash"
        or hook.get("hook_event_name") not in {None, "PostToolUse"}
    ):
        log("state=skip reason=not_bash")
        return 0

    tool_input = hook.get("tool_input")
    if not isinstance(tool_input, dict):
        log("state=skip reason=malformed_input")
        return 0
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        log("state=skip reason=missing_command")
        return 0
    category = classify_bash_candidate(command)
    if not category:
        log("state=not_candidate")
        return 0

    complete_bash = _complete_text_bash(hook)
    if complete_bash is None:
        return 0
    tool_response, original_output = complete_bash
    session_id = str(hook.get("session_id") or "unknown")
    try:
        body = {
            **base_ctx(session_id),
            "tool_name": "Bash",
            "command": command,
            "output": original_output,
            "pre_category": category,
        }
    except Exception as exc:  # noqa: BLE001 - identity failure preserves native Bash
        log(f"state=fallback reason=request_context_unavailable type={type(exc).__name__}")
        return 0

    data = _post_filter_bash(body)
    delivered_output = build_filter_delivery(
        data,
        original_output=original_output,
        tool_name="Bash",
        recovery_available=mcp_recovery_enabled(),
    )
    if delivered_output is None:
        log(f"state=skip reason={_safe_reason(data)} category={category}")
        return 0

    if isinstance(data.get("original_chars"), int):
        log(
            f"state=filtered category={category} "
            f"chars={data['original_chars']}->{len(delivered_output)}"
        )
    else:
        log(f"state=filtered category={category}")
    try:
        sys.stdout.write(
            json.dumps(
                _response(_updated_bash_output(tool_response, delivered_output)),
                ensure_ascii=False,
            )
        )
        sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001 - output failure must preserve native Bash
        log(f"state=output_error type={type(exc).__name__}")
    return 0


hook_filter_bash = main


if __name__ == "__main__":
    raise SystemExit(main())
