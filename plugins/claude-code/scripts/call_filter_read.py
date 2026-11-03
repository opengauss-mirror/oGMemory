#!/usr/bin/env python3
"""Claude ``Read`` PostToolUse adapter for oGMemory.

Native authorization and Read execution happen first. The hook filters only a
complete text ``FileReadOutput`` and replaces that structured result without
rereading the file or changing the authorized input.
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


FILTER_READ_TIMEOUT = 10.0


def log(message: str) -> None:
    print(f"[call_filter_read] {message}", file=sys.stderr)


def _safe_reason(data: object) -> str:
    if not isinstance(data, dict):
        return "invalid_response"
    # ``error`` may contain a path, URI, or upstream response text.  Only the
    # bounded diagnostic enum is safe to echo in hook stderr.
    reason = data.get("reason")
    if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,80}", reason):
        return reason
    return "unknown"


def _workspace_root(hook: dict) -> Path:
    raw = hook.get("cwd")
    try:
        return Path(str(raw or Path.cwd())).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return Path.cwd().resolve()


def _post_filter_read(body: dict) -> dict | None:
    started = time.monotonic()
    try:
        url = f"{base_api_url()}/api/v1/filter_read"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=http_plugin_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=FILTER_READ_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            status = getattr(response, "status", None) or response.getcode()
        data = json.loads(raw) if raw.strip() else {}
        log(
            f"state=response status={status} chars={len(raw)} latency_ms="
            f"{int((time.monotonic() - started) * 1000)}"
        )
        return data if isinstance(data, dict) else None
    except urllib.error.HTTPError as exc:
        log(f"state=http_error status={exc.code} latency_ms={int((time.monotonic() - started) * 1000)}")
    except urllib.error.URLError:
        log(f"state=transport_error latency_ms={int((time.monotonic() - started) * 1000)}")
    except TimeoutError:
        log(f"state=timeout latency_ms={int((time.monotonic() - started) * 1000)}")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
        log(f"state=parse_error latency_ms={int((time.monotonic() - started) * 1000)}")
    except Exception as exc:  # noqa: BLE001 - hooks must never block native Read
        log(f"state=error type={type(exc).__name__}")
    return None


def _response(updated_tool_output: dict) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": updated_tool_output,
        }
    }


def _present_read_range(tool_input: dict) -> bool:
    return any(
        field in tool_input
        and tool_input[field] is not None
        and str(tool_input[field]).strip() != ""
        for field in ("offset", "limit")
    )


def _resolved_read_path(raw_path: str, workspace: Path) -> Path | None:
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _complete_text_read(
    hook: dict,
    tool_input: dict,
) -> tuple[dict, str, str] | None:
    """Return ``(tool_response, relative_path, content)`` for one complete Read."""

    if _present_read_range(tool_input):
        log("state=skip reason=explicit_read_range")
        return None
    tool_response = hook.get("tool_response")
    if not isinstance(tool_response, dict) or tool_response.get("type") != "text":
        log("state=skip reason=unsupported_read_result")
        return None
    file_result = tool_response.get("file")
    if not isinstance(file_result, dict):
        log("state=skip reason=malformed_read_result")
        return None

    output_path = file_result.get("filePath")
    content = file_result.get("content")
    num_lines = _nonnegative_int(file_result.get("numLines"))
    start_line = _nonnegative_int(file_result.get("startLine"))
    total_lines = _nonnegative_int(file_result.get("totalLines"))
    if (
        not isinstance(output_path, str)
        or not output_path.strip()
        or not isinstance(content, str)
        or num_lines is None
        or start_line != 1
        or total_lines is None
        or num_lines != total_lines
        or file_result.get("truncatedByTokenCap") is True
    ):
        log("state=skip reason=partial_or_malformed_read_result")
        return None

    input_path = tool_input.get("file_path")
    if not isinstance(input_path, str) or not input_path.strip():
        log("state=skip reason=missing_file_path")
        return None
    workspace = _workspace_root(hook)
    resolved_input = _resolved_read_path(input_path, workspace)
    resolved_output = _resolved_read_path(output_path, workspace)
    if resolved_input is None or resolved_output is None or resolved_input != resolved_output:
        log("state=skip reason=read_path_mismatch")
        return None
    try:
        relative_path = resolved_output.relative_to(workspace).as_posix()
    except ValueError:
        log("state=skip reason=outside_workspace")
        return None
    if not relative_path or relative_path == ".":
        log("state=skip reason=unsafe_locator")
        return None
    return tool_response, relative_path, content


def _updated_text_read(tool_response: dict, filtered_output: str) -> dict:
    updated = dict(tool_response)
    updated_file = dict(tool_response["file"])
    updated_file["content"] = filtered_output
    updated_file["numLines"] = len(filtered_output.splitlines())
    updated["file"] = updated_file
    return updated


def main() -> int:
    try:
        enabled = effective_tool_output_filter_enabled("Read")
    except Exception as exc:  # noqa: BLE001 - configuration failure must preserve native Read
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
        or hook.get("tool_name") != "Read"
        or hook.get("hook_event_name") not in {None, "PostToolUse"}
    ):
        log("state=skip reason=not_read")
        return 0

    tool_input = hook.get("tool_input")
    if not isinstance(tool_input, dict):
        log("state=skip reason=malformed_input")
        return 0
    complete_read = _complete_text_read(hook, tool_input)
    if complete_read is None:
        return 0
    tool_response, relative_path, original_content = complete_read
    session_id = str(hook.get("session_id") or "unknown")
    try:
        body = {
            **base_ctx(session_id),
            "tool_name": "Read",
            "relative_path": relative_path,
            "content": original_content,
        }
    except Exception as exc:  # noqa: BLE001 - identity failure must preserve native Read
        log(f"state=fallback reason=request_context_unavailable type={type(exc).__name__}")
        return 0
    data = _post_filter_read(body)
    delivered_output = build_filter_delivery(
        data,
        original_output=original_content,
        tool_name="Read",
        recovery_available=mcp_recovery_enabled(),
    )
    if delivered_output is None:
        log(f"state=skip reason={_safe_reason(data)}")
        return 0

    if isinstance(data.get("original_chars"), int):
        log(f"state=filtered chars={data['original_chars']}->{len(delivered_output)}")
    else:
        log("state=filtered")
    try:
        sys.stdout.write(
            json.dumps(
                _response(_updated_text_read(tool_response, delivered_output)),
                ensure_ascii=False,
            )
        )
        sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001 - output failure must not break native Read
        log(f"state=output_error type={type(exc).__name__}")
    return 0


hook_filter_read = main


if __name__ == "__main__":
    raise SystemExit(main())
