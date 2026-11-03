#!/usr/bin/env python3
"""Claude Code hook: call oG-Memory after_turn on Stop and PreCompact.

Reads new JSONL lines since last offset, transforms to messages format,
and POSTs to oG-Memory /api/v1/after_turn.

Usage: called by Claude Code Stop/PreCompact hooks (stdin = hook JSON).
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ogm_plugin_request import (  # noqa: E402
    _env_bool,
    auto_capture_enabled,
    base_api_url,
    base_ctx,
    http_plugin_headers,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OG_MEMORY_TIMEOUT = 10  # seconds, hardcoded

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print to stderr so stdout stays clean for Claude Code."""
    print(f"[call_after_turn] {msg}", file=sys.stderr)


def code_memory_only_enabled():
    """Return whether generic transcript persistence must be skipped."""

    return _env_bool("OGMEM_CODE_MEMORY_ONLY", False)


def read_offset(sidecar_path):
    """Read last byte offset from sidecar file. Returns 0 if missing."""
    try:
        with open(sidecar_path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def write_offset(sidecar_path, offset):
    """Write byte offset to sidecar file."""
    with open(sidecar_path, "w") as f:
        f.write(str(offset))


def extract_text(content):
    """Extract plain text from message content (string or structured blocks).

    User messages: content is a string.
    Assistant messages: content is [{type: "text", text: "..."}, ...].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def parse_messages(jsonl_text):
    """Parse JSONL lines into [{role, content}] messages.

    Filters out:
    - isSidechain: true (sub-agent calls, have their own transcript)
    - isApiErrorMessage: true
    - non-message types (progress, tool_result, etc.)
    - messages with no text content
    """
    messages = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log(f"Skipping malformed JSONL line: {line[:80]}...")
            continue

        # Filter out noise
        if entry.get("isSidechain"):
            continue
        if entry.get("isApiErrorMessage"):
            continue

        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue

        msg = entry.get("message")
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue

        content = extract_text(msg.get("content", ""))
        if not content:
            continue

        messages.append({"role": role, "content": content})

    return messages


def post_after_turn(session_id, messages, hook_event):
    """POST messages to oG-Memory /api/v1/after_turn."""
    if code_memory_only_enabled() or not auto_capture_enabled():
        return True
    url = f"{base_api_url()}/api/v1/after_turn"
    body = {**base_ctx(session_id), "messages": messages, "hook_event_name": hook_event}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers=http_plugin_headers(),
        method="POST",
    )

    log(f"POST {url} session={session_id} msgs={len(messages)} event={hook_event}")

    try:
        with urllib.request.urlopen(req, timeout=OG_MEMORY_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            log(f"Response {resp.status}: {body[:200]}")
            return resp.status >= 200 and resp.status < 300
    except urllib.error.URLError as e:
        log(f"HTTP error: {e}")
        return False
    except Exception as e:
        log(f"Request failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if code_memory_only_enabled():
        log("code_memory_only skip")
        return

    if not auto_capture_enabled():
        log("auto_capture disabled, skipping transcript sync")
        return

    # 1. Read hook input from stdin
    hook_input = sys.stdin.read()
    try:
        hook = json.loads(hook_input)
    except json.JSONDecodeError:
        log("Failed to parse hook input JSON")
        sys.exit(0)

    session_id = hook.get("session_id", "unknown")
    transcript_path = hook.get("transcript_path", "")
    hook_event = hook.get("hook_event_name", "unknown")

    log(f"Hook: event={hook_event} session={session_id}")

    # 2. Check transcript exists
    if not transcript_path or not os.path.isfile(transcript_path):
        log(f"Transcript not found: {transcript_path}")
        sys.exit(0)

    # 3. Read offset
    sidecar_path = transcript_path + ".ingest-offset"
    last_offset = read_offset(sidecar_path)

    # 4. Get file size
    file_size = os.path.getsize(transcript_path)
    if file_size <= last_offset:
        log(f"No new content (offset={last_offset}, size={file_size})")
        sys.exit(0)

    # 5. Read incremental content
    with open(transcript_path, encoding="utf-8", errors="replace") as f:
        f.seek(last_offset)
        new_text = f.read()

    log(f"Read {len(new_text)} bytes from offset {last_offset}")

    # 6. Parse and filter messages
    messages = parse_messages(new_text)
    if not messages:
        log("No valid messages after filtering, updating offset")
        write_offset(sidecar_path, file_size)
        sys.exit(0)

    # 7. POST to oG-Memory
    success = post_after_turn(session_id, messages, hook_event)

    # 8. Update offset only on success
    if success:
        write_offset(sidecar_path, file_size)
        log(f"Offset updated to {file_size}")
    else:
        log("Ingest failed, offset NOT updated (will retry next hook)")

    sys.exit(0)


if __name__ == "__main__":
    main()
