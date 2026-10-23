"""Versioned ingestion-only activation records; never ordinary memory layers."""

import hashlib
import json

from core.models import ContextNode, IndexRecord, RetrievedBlock, SeedHit

ACTIVATION_LEVEL = 3
ACTIVATION_VERSION = "c1-integrated-activation-v1"


def candidate_activation(candidate) -> str | None:
    """Return only a valid opt-in descriptor; absence does not clear old data."""
    value = candidate.metadata.get("update_activation")
    if (candidate.metadata.get("update_activation_version") == ACTIVATION_VERSION
            and isinstance(value, str) and value.strip()
            and "\n" not in value.strip() and "\r" not in value.strip()):
        return value.strip()
    return None


def is_current_activation(record: IndexRecord, node: ContextNode) -> bool:
    """Fail closed on stale projections; caller must authorize the canonical node."""
    expected = build_activation_record(node, record.filters)
    return (expected is not None and record.id == expected.id
            and record.uri == node.uri and record.level == ACTIVATION_LEVEL
            and record.text == expected.text
            and all(record.metadata.get(k) == v for k, v in expected.metadata.items()))


def activation_revision(node: ContextNode) -> str:
    """Bind a projection to both the stored memory and its activation."""
    payload = [node.uri, node.abstract, node.overview, node.content,
               node.metadata.get("update_activation"),
               node.metadata.get("update_activation_version")]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def build_activation_record(node: ContextNode, filters: dict) -> IndexRecord | None:
    text = node.metadata.get("update_activation")
    if (node.metadata.get("update_activation_version") != ACTIVATION_VERSION
            or not isinstance(text, str) or not text.strip()
            or "\n" in text.strip() or "\r" in text.strip()):
        return None
    return IndexRecord(
        id=IndexRecord.generate_id(node.uri, ACTIVATION_LEVEL),
        uri=node.uri, level=ACTIVATION_LEVEL, text=text.strip(),
        filters=dict(filters),
        metadata={"record_kind": "update_activation",
                  "representation_version": ACTIVATION_VERSION,
                  "memory_revision": activation_revision(node)},
    )


def activation_hit_is_compatible(
    hit: SeedHit,
    block: RetrievedBlock,
    *,
    embedding_model: str,
    embedding_dimension: int,
) -> bool:
    """Validate a hydrated activation hit against its canonical memory."""
    metadata = dict(block.metadata or {})
    node = ContextNode(
        uri=hit.uri,
        context_type=str(metadata.get("context_type", "MEMORY")),
        category=block.category,
        level=2,
        owner_space=block.owner_space,
        abstract=block.abstract or "",
        overview=block.overview or "",
        content=block.content_excerpt or "",
        metadata=metadata,
    )
    expected = build_activation_record(node, {})
    return bool(
        expected is not None
        and hit.level == ACTIVATION_LEVEL
        and all(
            hit.metadata.get(key) == value
            for key, value in expected.metadata.items()
        )
        and hit.metadata.get("embedding_model") == embedding_model
        and hit.metadata.get("embedding_dimension") == embedding_dimension
    )
