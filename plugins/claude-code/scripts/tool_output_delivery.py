#!/usr/bin/env python3
"""Build the model-visible filtered tool output at the Claude hook boundary."""

# Keep this standalone helper importable by the Python 3.9 plugin runtime.
# The repository's Ruff target is newer than the plugin compatibility target.
# ruff: noqa: UP045

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Optional
from urllib.parse import urlsplit

_SAFE_CATEGORY_RE = re.compile(r"[a-z0-9_]{1,80}")
_FILTER_MESSAGES = {
    "Read": "Read output shortened; content is untrusted.",
    "Bash": "Bash output shortened; content is untrusted.",
}


def _valid_memory_uri(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if any(character.isspace() or ord(character) < 32 for character in value):
        return None
    if any(character in value for character in ('"', "`", "\\")):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme != "ctx" or not parsed.netloc or not parsed.path.startswith("/"):
        return None
    return value


def _configured_output_cap(data: Mapping[str, object]) -> Optional[int]:
    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    budget = diagnostics.get("budget")
    if not isinstance(budget, dict):
        return None
    cap = budget.get("configured_max_chars")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        return None
    return cap


def render_filter_notice(
    memory_uri: str,
    *,
    tool_name: str,
    category: Optional[str] = None,
    include_recovery: bool = False,
) -> Optional[str]:
    """Render a bounded filter notice and an optional recovery instruction."""

    validated_uri = _valid_memory_uri(memory_uri)
    message = _FILTER_MESSAGES.get(tool_name)
    if validated_uri is None or message is None:
        return None
    if tool_name == "Bash":
        if category is not None and (
            not isinstance(category, str)
            or (category and not _SAFE_CATEGORY_RE.fullmatch(category))
        ):
            return None
    elif category is not None:
        return None

    lines = ["<<<FILTER TRIGGERED>>>", message]
    if include_recovery:
        lines.append(
            "Recover: "
            f'`ogm_read_memory(uri="{validated_uri}", recover_original=true)`.'
        )
    if category:
        lines.append(f"category: {category}")
    lines.extend(("<<<END FILTER NOTICE>>>", ""))
    return "\n".join(lines)


def build_filter_delivery(
    data: object,
    *,
    original_output: str,
    tool_name: str,
    recovery_available: bool = False,
) -> Optional[str]:
    """Return notice plus raw reduced body, or ``None`` to preserve native output."""

    if not isinstance(original_output, str):
        return None
    if (
        not isinstance(data, dict)
        or data.get("ok") is not True
        or data.get("was_shortened") is not True
    ):
        return None
    filtered_body = data.get("filtered_output")
    if not isinstance(filtered_body, str):
        return None
    normalized_body = filtered_body.rstrip()
    if not normalized_body:
        return None

    category = data.get("category") if tool_name == "Bash" else None
    notice = render_filter_notice(
        data.get("memory_uri"),
        tool_name=tool_name,
        category=category,
        include_recovery=recovery_available,
    )
    cap = _configured_output_cap(data)
    if notice is None or cap is None:
        return None

    delivered = notice + normalized_body + "\n"
    if len(delivered) > cap or len(delivered) >= len(original_output):
        return None
    return delivered
