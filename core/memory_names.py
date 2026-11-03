"""Shared memory-name formatting and validation."""

import re
import unicodedata

from core.candidate_dedup import case_draft_signature, event_draft_signature


def memory_directory(uri: str) -> str:
    """Return the type directory from a memory or skill context URI."""
    parts = uri.split("/", 7)
    if len(parts) > 6 and parts[0] == "ctx:" and parts[3] in {"users", "agents"}:
        if parts[5] == "memories":
            return parts[6]
        if parts[5] == "skills":
            return "skills"
    return ""


def candidate_identity_key(candidate) -> tuple[str, str, str, str, str]:
    """Keep owner, source scope and subject separate during extraction deduplication."""
    if candidate.target_uri:
        return (candidate.category, candidate.owner_scope, candidate.target_uri, "", "")
    subject = ""
    if candidate.category == "entity":
        subject = canonical_object_name(candidate.object_name)
    elif candidate.category == "preference":
        subject = canonical_scope(candidate.who or "")
    elif candidate.category == "event":
        subject = event_draft_signature(candidate)
    elif candidate.category == "case":
        subject = case_draft_signature(candidate)
    return (candidate.category, candidate.owner_scope,
            candidate.routing_key,
            canonical_scope(candidate.scope_ref), subject)


def canonical_object_name(value: str) -> str:
    """Normalize a subject label while retaining its identifying symbols."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def canonical_scope(value: str) -> str:
    """Normalize scope word separators while retaining identifying symbols."""
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    return re.sub(r"[\s_-]+", "_", value.casefold()).strip("_")


def canonical_memory_key(value: str) -> str:
    """Format a proposed human-readable memory name."""
    value = unicodedata.normalize("NFKC", value).strip()
    if value.isascii():
        value = re.sub(r"\s+", "_", value.casefold())
        value = re.sub(r"[^a-z0-9_-]", "", value)
        return re.sub(r"[_-]+", "_", value).strip("_")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    return re.sub(r"[\W_]+", "_", value.casefold()).strip("_")


def is_safe_memory_key(value: str) -> bool:
    """Accept one URI path segment without encoded separators or controls."""
    return bool(value and not value.startswith(".") and any(char.isalnum() for char in value)
                and not any(char in value for char in "/\\%")
                and all(unicodedata.category(char) not in {"Cc", "Cs"} for char in value))


def is_meaningful_memory_key(value: str, *, max_length: int = 80) -> bool:
    """Require a descriptive name suitable for a public memory URI."""
    if not value or not value[0].isalpha() or value != canonical_memory_key(value):
        return False
    if any(not part or not part.isalnum() for part in value.split("_")):
        return False
    if not 2 <= len(value) <= max_length:
        return False
    if value in {"unnamed", "unknown", "memory", "entity", "event", "preference", "md", "json"}:
        return False
    return not re.fullmatch(r"(?:auto|memory|node)_[0-9a-f]{8,}", value)


def qualify_memory_key(key: str, scope: str) -> str:
    """Include the normalized scope once, wherever it already occurs in the name."""
    if scope and f"_{scope}_" not in f"_{key}_":
        return f"{scope}_{key}"
    return key


def unscoped_memory_key(key: str, scope: str) -> str:
    """Drop the scope tokens so differently placed scoped spellings compare equal."""
    if not scope:
        return key
    return re.sub(rf"(?:^|_){re.escape(scope)}(?=_|$)", "", key).strip("_") or key
