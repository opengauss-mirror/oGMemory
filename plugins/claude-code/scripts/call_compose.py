#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Claude Code hook: call oG-Memory compose on UserPromptSubmit.

Calls oG-Memory /api/v1/compose with the user's prompt, extracts
identityContext (profile) and retrievedEvidence (working set), and injects
them as additionalContext into Claude's context.

Usage: called by Claude Code UserPromptSubmit hook (stdin = hook JSON).
"""

import json
import sys
from pathlib import Path
import urllib.request
import urllib.error

# Sibling module must resolve even when cwd is not plugins/claude-code/scripts
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ogm_plugin_request import (  # noqa: E402, I001
    _env_bool,
    auto_recall_enabled,
    base_api_url,
    base_ctx,
    http_plugin_headers,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPOSE_TIMEOUT = 13          # seconds; below the configured prompt hook deadline
MAX_CONTEXT_CHARS = 9500     # leave 500 for JSON wrapper
MIN_PROMPT_LEN = 4           # skip queries shorter than this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[call_compose] {msg}", file=sys.stderr)


def call_compose(prompt, session_id):
    """POST to oG-Memory compose endpoint. Returns parsed JSON or None."""
    url = f"{base_api_url()}/api/v1/compose"
    body = {**base_ctx(session_id), "prompt": prompt}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers=http_plugin_headers(),
        method="POST",
    )

    log(f"POST {url} session={session_id} prompt_len={len(prompt)}")

    try:
        with urllib.request.urlopen(req, timeout=COMPOSE_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            log(f"Response {resp.status}: {body[:200]}")
            return json.loads(body)
    except urllib.error.URLError as e:
        log(f"HTTP error: {e}")
        return None
    except Exception as e:
        log(f"Request failed: {e}")
        return None


def build_additional_context(data):
    """Extract identityContext + retrievedEvidence, truncate to MAX_CONTEXT_CHARS."""
    parts = []

    identity = (data.get("identityContext") or "").strip()
    if identity:
        parts.append(identity)

    evidence = (data.get("retrievedEvidence") or "").strip()
    if evidence:
        parts.append(evidence)

    if not parts:
        return None

    # Header + body
    body = "\n\n".join(parts)

    total = body
    if len(total) > MAX_CONTEXT_CHARS:
        total = total[:MAX_CONTEXT_CHARS]

    return f"[oG-Memory]\n{total}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not auto_recall_enabled():
        log("auto_recall disabled, skipping automatic compose")
        return

    # 1. Read hook input from stdin
    hook_input = sys.stdin.read()
    try:
        hook = json.loads(hook_input)
    except json.JSONDecodeError:
        log("Failed to parse hook input JSON")
        sys.exit(0)

    prompt = (hook.get("prompt") or "").strip()
    session_id = hook.get("session_id", "unknown")

    # 2. Skip short / empty / slash commands
    if len(prompt) < MIN_PROMPT_LEN:
        log(f"Prompt too short ({len(prompt)} chars), skipping")
        sys.exit(0)

    if prompt.startswith("/"):
        log("Slash command, skipping")
        sys.exit(0)

    # 3. Call compose
    data = call_compose(prompt, session_id)
    if data is None:
        sys.exit(0)

    # 4. Build additionalContext
    additional = build_additional_context(data)
    if additional is None:
        log("No relevant context returned")
        sys.exit(0)

    # 5. Output JSON (method B: additionalContext)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional,
        }
    }
    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
