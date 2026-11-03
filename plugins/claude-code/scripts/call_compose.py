#!/usr/bin/env python3
"""Claude Code hook: call oG-Memory compose on UserPromptSubmit.

Calls oG-Memory /api/v1/compose with the user's prompt, extracts
identityContext (profile) and retrievedEvidence (working set), and injects
them as additionalContext into Claude's context.

Usage: called by Claude Code UserPromptSubmit hook (stdin = hook JSON).
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

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
    effective_code_retrieval_mode,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPOSE_TIMEOUT = 13          # seconds; default/full transport deadline
CANDIDATE_LOCAL_COMPOSE_TIMEOUT = 360
# Claude uses this upper bound for the command itself.  Keep one second of
# margin so the HTTP client can return before the hook supervisor terminates it.
COMPOSE_HOOK_TIMEOUT = 365
MAX_CONTEXT_CHARS = 9500     # leave 500 for JSON wrapper
MIN_PROMPT_LEN = 4           # skip queries shorter than this
_CODE_RETRIEVAL_DIAGNOSTICS_PREFIX = "code_retrieval_diagnostics "
_SAFE_CODE_RETRIEVAL_MODES = {"full", "candidate_local"}
_SAFE_CODE_RETRIEVAL_STATUSES = {
    "disabled", "ok", "zero_result", "partial_failure", "error",
}
_SAFE_CODE_RETRIEVAL_ROUTE_STATES = {
    "disabled",
    "ok",
    "zero_result",
    "partial_failure",
    "error",
    "timeout",
    "not_run",
}
_CANDIDATE_ROUTES = ("semantic", "bm25", "symbolic", "graph")
_CODE_ONLY_EMPTY_TEXT_FIELDS = (
    "identityContext",
    "episodicContext",
    "sessionContext",
    "taskContext",
    "systemPromptAddition",
    "systemPromptSuffix",
    "memoryUserMessage",
)
_CODE_ONLY_EMPTY_LIST_FIELDS = ("openLoops", "uncertainties")
_CODE_ONLY_ALLOWED_RESPONSE_FIELDS = frozenset(
    {
        "messages",
        "systemPromptAddition",
        "systemPromptSuffix",
        "memoryUserMessage",
        "estimatedTokens",
        "archiveCount",
        "archiveIncluded",
        "stats",
        "identityContext",
        "episodicContext",
        "sessionContext",
        "taskContext",
        "retrievedEvidence",
        "openLoops",
        "uncertainties",
        "budgetUsedBySlot",
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[call_compose] {msg}", file=sys.stderr)


def effective_compose_timeout():
    """Choose only the plugin transport deadline; server mode remains config-owned."""

    mode = effective_code_retrieval_mode()
    if mode == "candidate_local":
        return min(CANDIDATE_LOCAL_COMPOSE_TIMEOUT, COMPOSE_HOOK_TIMEOUT - 1)
    return COMPOSE_TIMEOUT


def code_memory_only_enabled():
    """Return whether the explicit plugin code-only profile is enabled."""

    return _env_bool("OGMEM_CODE_MEMORY_ONLY", False)


def _safe_nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _candidate_code_retrieval_diagnostics_payload(data):
    """Project one complete, content-free candidate diagnostic event or fail closed."""

    stats = data.get("stats") if isinstance(data, dict) else None
    code_retrieval = stats.get("code_retrieval") if isinstance(stats, dict) else None
    if not isinstance(code_retrieval, dict):
        return None
    diagnostics = code_retrieval.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    if (
        code_retrieval.get("mode") != "candidate_local"
        or diagnostics.get("mode") != "candidate_local"
    ):
        return None

    status = code_retrieval.get("status")
    if not isinstance(status, str) or status not in _SAFE_CODE_RETRIEVAL_STATUSES:
        return None
    raw_top_level_counts = {
        "duration_ms": diagnostics.get("duration_ms"),
        "evidence_count": code_retrieval.get("evidence_count"),
        "evidence_chars": diagnostics.get("evidence_chars"),
        "line_span_count": diagnostics.get("line_span_count"),
    }
    top_level_counts = {}
    for name, raw_value in raw_top_level_counts.items():
        value = _safe_nonnegative_int(raw_value)
        if value is None:
            return None
        top_level_counts[name] = value

    routes = diagnostics.get("routes")
    if not isinstance(routes, dict):
        return None
    safe_routes = {}
    for name in _CANDIDATE_ROUTES:
        route = routes.get(name)
        if not isinstance(route, dict):
            return None
        state = route.get("state")
        count = _safe_nonnegative_int(route.get("count"))
        duration_ms = _safe_nonnegative_int(route.get("duration_ms"))
        if (
            not isinstance(state, str)
            or state not in _SAFE_CODE_RETRIEVAL_ROUTE_STATES
            or count is None
            or duration_ms is None
        ):
            return None
        safe_routes[name] = {
            "state": state,
            "count": count,
            "duration_ms": duration_ms,
        }

    embedding = diagnostics.get("embedding")
    if not isinstance(embedding, dict):
        return None
    safe_embedding = {}
    for name in ("logical_inputs", "provider_calls", "batches"):
        value = _safe_nonnegative_int(embedding.get(name))
        if value is None:
            return None
        safe_embedding[name] = value

    fusion = diagnostics.get("fusion")
    if not isinstance(fusion, dict):
        return None
    safe_fusion = {}
    for name in ("fused_count", "route_fill_count", "literal_fallback_count"):
        value = _safe_nonnegative_int(fusion.get(name))
        if value is None:
            return None
        safe_fusion[name] = value

    retention = diagnostics.get("retention")
    if not isinstance(retention, dict) or retention.get("scope") != "request":
        return None
    safe_retention = {"scope": "request"}
    for name in ("retained_candidates", "retained_vectors"):
        value = _safe_nonnegative_int(retention.get(name))
        if value is None:
            return None
        safe_retention[name] = value

    return {
        "mode": "candidate_local",
        "status": status,
        "duration_ms": top_level_counts["duration_ms"],
        "evidence_count": top_level_counts["evidence_count"],
        "evidence_chars": top_level_counts["evidence_chars"],
        "line_span_count": top_level_counts["line_span_count"],
        "routes": safe_routes,
        "embedding": safe_embedding,
        "fusion": safe_fusion,
        "retention": safe_retention,
    }


def _log_candidate_code_retrieval_diagnostics(data):
    """Emit the frozen candidate artifact line only when its whole schema is safe."""

    payload = _candidate_code_retrieval_diagnostics_payload(data)
    if payload is None:
        return
    log(
        _CODE_RETRIEVAL_DIAGNOSTICS_PREFIX
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _log_code_retrieval_summary(data):
    """Log bounded response counters without response content or protected locators."""

    stats = data.get("stats") if isinstance(data, dict) else None
    code_retrieval = stats.get("code_retrieval") if isinstance(stats, dict) else None
    if not isinstance(code_retrieval, dict):
        return
    diagnostics = code_retrieval.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    embedding = diagnostics.get("embedding")
    embedding = embedding if isinstance(embedding, dict) else {}
    mode = diagnostics.get("mode", code_retrieval.get("mode"))
    status = code_retrieval.get("status")
    if (
        not isinstance(mode, str)
        or mode not in _SAFE_CODE_RETRIEVAL_MODES
        or not isinstance(status, str)
        or status not in _SAFE_CODE_RETRIEVAL_STATUSES
    ):
        return
    fields = [f"mode={mode}", f"status={status}"]
    for key, label in (
        ("duration_ms", "duration_ms"),
        ("logical_inputs", "embedding_inputs"),
        ("provider_calls", "embedding_calls"),
        ("batches", "embedding_batches"),
    ):
        source = diagnostics if key == "duration_ms" else embedding
        value = _safe_nonnegative_int(source.get(key))
        if value is not None:
            fields.append(f"{label}={value}")
    log("code_retrieval " + " ".join(fields))


def _has_content(value):
    """Return whether an unknown response field could carry injected context."""

    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return True


def _is_safe_code_only_response(data):
    """Validate the code-only acknowledgement and reject generic context fail-closed."""

    if not isinstance(data, dict):
        return False

    stats = data.get("stats")
    code_only = stats.get("code_memory_only") if isinstance(stats, dict) else None
    if not isinstance(code_only, dict) or code_only.get("effective") is not True:
        return False

    for field in _CODE_ONLY_EMPTY_TEXT_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or value.strip():
            return False

    for field in _CODE_ONLY_EMPTY_LIST_FIELDS:
        if data.get(field) != []:
            return False

    archive_count = data.get("archiveCount")
    if (
        isinstance(archive_count, bool)
        or not isinstance(archive_count, int)
        or archive_count != 0
        or data.get("archiveIncluded") is not False
    ):
        return False

    if not isinstance(data.get("retrievedEvidence"), str):
        return False

    budget_used_by_slot = data.get("budgetUsedBySlot")
    if not isinstance(budget_used_by_slot, dict):
        return False
    for slot, value in budget_used_by_slot.items():
        if (
            not isinstance(slot, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (slot != "retrieved" and value != 0)
        ):
            return False

    for field, value in data.items():
        if field not in _CODE_ONLY_ALLOWED_RESPONSE_FIELDS and _has_content(value):
            return False
    return True


def call_compose(prompt, session_id):
    """POST to oG-Memory compose endpoint. Returns parsed JSON or None."""
    url = f"{base_api_url()}/api/v1/compose"
    body = {**base_ctx(session_id), "prompt": prompt}
    if code_memory_only_enabled():
        # Keep this a JSON boolean; the server rejects string/number lookalikes.
        body["code_memory_only"] = True
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers=http_plugin_headers(),
        method="POST",
    )

    log(f"POST /api/v1/compose session={session_id} prompt_len={len(prompt)}")

    try:
        with urllib.request.urlopen(req, timeout=effective_compose_timeout()) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, dict):
                log(
                    f"Response status={resp.status} chars={len(body)} "
                    f"identity={bool((data.get('identityContext') or '').strip())} "
                    f"evidence={bool((data.get('retrievedEvidence') or '').strip())}"
                )
                _log_code_retrieval_summary(data)
                _log_candidate_code_retrieval_diagnostics(data)
            else:
                log(f"Response status={resp.status} chars={len(body)} object=false")
            return data
    except urllib.error.URLError as e:
        log(f"HTTP error type={type(e).__name__}")
        return None
    except Exception as e:
        log(f"Request failed type={type(e).__name__}")
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


def build_code_only_additional_context(data):
    """Return only acknowledged code evidence, never generic Compose slots."""

    if not _is_safe_code_only_response(data):
        return None

    evidence = data["retrievedEvidence"].strip()
    if not evidence:
        return None
    return f"[oG-Memory]\n{evidence[:MAX_CONTEXT_CHARS]}"


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
    if code_memory_only_enabled():
        additional = build_code_only_additional_context(data)
        if additional is None:
            log("code_memory_only context_not_injected")
    else:
        additional = build_additional_context(data)
    if additional is None:
        if not code_memory_only_enabled():
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
