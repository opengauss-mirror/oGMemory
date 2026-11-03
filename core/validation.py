"""Input validation for ContextEngine.

Provides validators for all external inputs to prevent:
- DoS via oversized payloads
- Data corruption via invalid encodings
"""

import hashlib
import logging
import re
from dataclasses import replace

from core.errors import ValidationError
from core.memory_names import (
    canonical_memory_key,
    is_meaningful_memory_key,
    is_safe_memory_key,
    unscoped_memory_key,
)
from core.models import CandidateMemory

logger = logging.getLogger(__name__)


# Validation limits
MAX_ABSTRACT_LEN = 200
MAX_OVERVIEW_LEN = 5000
MAX_CONTENT_LEN = 100000
MAX_SLUG_LEN = 100
MAX_PATH_SEGMENT_BYTES = 255
MAX_MEMORY_KEY_BYTES = MAX_PATH_SEGMENT_BYTES - len(".update-lock-") - 1 - 32
APPEND_ONLY_KEY_SUFFIX_BYTES = 1 + 14 + 1 + 8

SAFE_SLUG_PATTERN = re.compile(r'^[a-z0-9_-]+$')


def validate_text_length(text: str, max_len: int, field_name: str) -> str:
    """Reject oversized text without changing its value.

    Args:
        text: Input text
        max_len: Maximum allowed length
        field_name: Field name for error reporting

    Returns:
        Original text within the limit

    Raises:
        ValidationError: If text exceeds max_len
    """
    if not text:
        return text

    if len(text) > max_len:
        raise ValidationError(
            field_name,
            f"Length {len(text)} exceeds maximum {max_len}"
        )
    return text


def validate_utf8(text: str, field_name: str) -> str:
    """Validate text is valid UTF-8.

    Args:
        text: Input text
        field_name: Field name for error reporting

    Returns:
        Validated text

    Raises:
        ValidationError: If text contains invalid UTF-8
    """
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError as e:
        raise ValidationError(
            field_name,
            f"Invalid UTF-8 encoding: {e}"
        )


def validate_slug(slug: str, field_name: str = "slug") -> str:
    """Validate slug format (lowercase alphanumeric with hyphens/underscores).

    Args:
        slug: Slug to validate
        field_name: Field name for error reporting

    Returns:
        Validated slug

    Raises:
        ValidationError: If slug format is invalid
    """
    if not slug:
        raise ValidationError(field_name, "Slug cannot be empty")

    if len(slug) > MAX_SLUG_LEN:
        raise ValidationError(
            field_name,
            f"Length {len(slug)} exceeds maximum {MAX_SLUG_LEN}"
        )

    if not SAFE_SLUG_PATTERN.match(slug):
        raise ValidationError(
            field_name,
            "Must contain only lowercase letters, numbers, hyphens, and underscores"
        )

    return slug


def normalize_slug(slug: str) -> str:
    """Normalize slug to lowercase with hyphens instead of spaces.

    Args:
        slug: Raw slug input

    Returns:
        Normalized slug
    """
    if not slug:
        return slug

    # Convert to lowercase
    slug = slug.lower().strip()

    # Replace spaces and special chars with hyphens
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9_-]', '', slug)

    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)

    # Remove leading/trailing hyphens
    slug = slug.strip('-')

    return slug


def _legacy_routing_key(value: str) -> str:
    try:
        return validate_slug(normalize_slug(value), "routing_key")
    except ValidationError:
        return f"auto-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def validate_candidate(candidate: CandidateMemory) -> CandidateMemory:
    """Validate a CandidateMemory before processing.

    Args:
        candidate: CandidateMemory to validate

    Returns:
        Validated CandidateMemory with original text preserved in a new copy

    Raises:
        ValidationError: If validation fails critically
    """
    if (not isinstance(candidate.source_occurrence_id, str)
            or candidate.source_occurrence_id and not re.fullmatch(r"[0-9a-f]{64}", candidate.source_occurrence_id)):
        raise ValidationError("source_occurrence_id", "Must be an internal source partition digest")
    # Validate abstract first (may raise)
    validated_abstract = None
    if candidate.abstract:
        validated_abstract = validate_utf8(candidate.abstract, "abstract")
        validated_abstract = validate_text_length(
            validated_abstract, MAX_ABSTRACT_LEN, "abstract"
        )

    # Validate overview
    validated_overview = None
    if candidate.overview:
        validated_overview = validate_utf8(candidate.overview, "overview")
        validated_overview = validate_text_length(
            validated_overview, MAX_OVERVIEW_LEN, "overview"
        )

    # Validate content length and encoding
    validated_content = None
    if candidate.content:
        validated_content = validate_utf8(candidate.content, "content")
        validated_content = validate_text_length(
            validated_content, MAX_CONTENT_LEN, "content"
        )

    # Validate and normalize routing_key (slug)
    validated_routing_key = validate_utf8(candidate.routing_key, "routing_key").strip()
    legacy_routing_key = candidate.legacy_routing_key
    if candidate.category in {"entity", "preference", "pattern", "event", "case"}:
        normalized_key = canonical_memory_key(validated_routing_key)
        if is_meaningful_memory_key(normalized_key):
            validated_routing_key = normalized_key
        elif validated_routing_key.startswith(".") and is_safe_memory_key(normalized_key):
            validated_routing_key = normalized_key
        elif not is_safe_memory_key(validated_routing_key):
            raise ValidationError("routing_key", "Must be one safe URI path segment")
        max_key_bytes = MAX_MEMORY_KEY_BYTES
        if candidate.category in {"event", "case"}:
            max_key_bytes -= APPEND_ONLY_KEY_SUFFIX_BYTES
        # Allocated upsert names also contain scope/holder labels. Keep the
        # short-name limit on the topic and the byte budget on the entire name.
        short_key = validated_routing_key
        if candidate.category in {"entity", "preference", "pattern"}:
            labels = [canonical_memory_key(candidate.scope_ref)]
            if candidate.category == "preference":
                labels.append(canonical_memory_key(candidate.who or ""))
            # A holder can include the scope (or vice versa); remove it whole.
            for label in sorted(labels, key=len, reverse=True):
                short_key = unscoped_memory_key(short_key, label)
        if (len(short_key) > MAX_SLUG_LEN
                or len(validated_routing_key.lower().encode("utf-8")) > max_key_bytes):
            raise ValidationError("routing_key", "Memory name exceeds path segment length limit")
        if not legacy_routing_key:
            legacy_routing_key = _legacy_routing_key(candidate.routing_key)
        if legacy_routing_key == validated_routing_key:
            legacy_routing_key = ""
    elif candidate.routing_key:
        validated_routing_key = _legacy_routing_key(candidate.routing_key)

    # Validate confidence
    validated_confidence = candidate.confidence
    if not 0.0 <= validated_confidence <= 1.0:
        # Clamp to valid range rather than reject
        validated_confidence = max(0.0, min(1.0, validated_confidence))

    # Validate category (raises if invalid)
    valid_categories = {
        "profile", "preference", "entity", "event",
        "case", "pattern", "skill", "tool"
    }
    if candidate.category not in valid_categories:
        raise ValidationError(
            "category",
            f"Must be one of {valid_categories}, got '{candidate.category}'"
        )

    # Validate routing_key != category (would write to parent directory)
    if candidate.routing_key == candidate.category:
        raise ValidationError(
            "routing_key",
            f"routing_key cannot equal category name '{candidate.category}'"
        )

    # Validate owner_scope (raises if invalid)
    if candidate.owner_scope not in ("user", "agent"):
        raise ValidationError(
            "owner_scope",
            f"Must be 'user' or 'agent', got '{candidate.owner_scope}'"
        )

    # Create a NEW copy with validated fields (thread-safe, no mutation)
    return replace(
        candidate,
        abstract=validated_abstract,
        overview=validated_overview,
        content=validated_content,
        routing_key=validated_routing_key,
        legacy_routing_key=legacy_routing_key,
        confidence=validated_confidence,
    )


# ---------------------------------------------------------------------------
# Speaker attribution validation for profile candidates
# ---------------------------------------------------------------------------

# Basis values that are strong enough for profile extraction
VALID_PROFILE_BASES = {"self_first_person", "self_named"}

# Tokens that indicate the speaker IS the user
_USER_SPEAKER_TOKENS = {"user", "<user>"}


def _normalize_speaker(s: str) -> str:
    """Normalize a speaker identifier for comparison."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def validate_attribution(
    candidate: CandidateMemory,
    user_id: str | None = None,
) -> CandidateMemory:
    """Validate speaker attribution for profile candidates.

    If attribution indicates a non-user speaker, downgrade from 'profile'
    to 'entity' to prevent polluting identityContext. The information is
    preserved as an entity rather than discarded.

    Args:
        candidate: CandidateMemory to validate
        user_id: The user's identifier, used to verify speaker identity

    Returns:
        CandidateMemory, possibly with category changed from profile to entity
    """
    if candidate.category != "profile":
        return candidate

    basis = candidate.attribution_basis
    speaker = candidate.attributed_speaker

    # No attribution provided (legacy / backward compat) — allow through
    if basis is None and speaker is None:
        logger.info(
            "Profile candidate '%s' has no attribution fields — allowing (backward compat)",
            candidate.routing_key,
        )
        return candidate

    # 1) Check basis strength — if LLM identified self-reference, trust it
    if basis in VALID_PROFILE_BASES:
        return candidate

    # 2) Basis is missing or non-self — check speaker identity matches user
    if basis is not None and basis not in VALID_PROFILE_BASES:
        logger.info(
            "Downgrading profile→entity: routing_key='%s' attribution_basis='%s' speaker='%s'",
            candidate.routing_key, basis, speaker,
        )
        return _downgrade_to_entity(candidate, speaker)

    # 3) No basis provided — fall back to speaker token match
    user_tokens = set(_USER_SPEAKER_TOKENS)
    if user_id:
        user_tokens.add(_normalize_speaker(user_id))

    if speaker and _normalize_speaker(speaker) not in user_tokens:
        logger.info(
            "Downgrading profile→entity: routing_key='%s' speaker='%s' does not match user_id='%s'",
            candidate.routing_key, speaker, user_id,
        )
        return _downgrade_to_entity(candidate, speaker)

    return candidate


def _downgrade_to_entity(candidate: CandidateMemory, speaker: str | None) -> CandidateMemory:
    """Downgrade a profile candidate to entity, preserving information."""
    prefix = (speaker or "unknown").lower().replace(" ", "_")
    new_routing_key = f"{prefix}_{candidate.routing_key}"
    return replace(
        candidate,
        category="entity",
        owner_scope="user",
        routing_key=new_routing_key,
    )
