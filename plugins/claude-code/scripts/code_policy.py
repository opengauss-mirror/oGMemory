"""Canonical oGMemory code-policy rendering.

This module is stdlib-only because Claude hooks may import it with the checked
out repository's Python rather than the development interpreter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_POLICY_PATH = _PLUGIN_ROOT / "prompts" / "code_policy_injection.txt"
_FILTERING_PROMPT_HEADING = "## Filtering Strategy"
_FILTERING_PROMPT_END_MARKER = "Keep the workflow compact:"
_FILTERING_PROMPT_BULLETS = (
    "- Native `Read` on long logs/traces may be filtered through oGMemory before content is returned. "
    "The original output is stored in owner-scoped tool-output memory and can be recovered with "
    "`ogm_read_memory(uri=..., recover_original=true)`.",
)
def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _render_policy_prompt(source: str, *, include_filtering: bool) -> str:
    """Render the canonical policy with the configured filtering section."""

    if include_filtering:
        return source.strip()
    rendered = source
    for bullet in _FILTERING_PROMPT_BULLETS:
        rendered = rendered.replace(bullet + "\n", "")
    start = rendered.find(_FILTERING_PROMPT_HEADING)
    if start == -1:
        return rendered.strip()
    end = rendered.find(_FILTERING_PROMPT_END_MARKER, start)
    if end == -1:
        return rendered[:start].rstrip()
    return (rendered[:start].rstrip() + "\n\n" + rendered[end:].lstrip()).strip()


def render_code_policy(*, read_filter_enabled: bool, bash_filter_enabled: bool) -> dict[str, Any]:
    """Return canonical text plus stable source/rendered hashes."""

    source = _POLICY_PATH.read_text(encoding="utf-8")
    rendered = _render_policy_prompt(
        source,
        include_filtering=bool(read_filter_enabled or bash_filter_enabled),
    )
    return {
        "text": rendered,
        "source_sha256": _sha256_text(source),
        "rendered_sha256": _sha256_text(rendered),
        "read_filter_enabled": bool(read_filter_enabled),
        "bash_filter_enabled": bool(bash_filter_enabled),
    }
