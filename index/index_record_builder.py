"""IndexRecordBuilder: Expand ContextNode into L0 and optional L2 records.

Indexable ContextNodes produce an L0 abstract record and, when content is
present, an L2 content record. This builder does not emit L1 records;
DirectoryEventHandler emits L0/L1 records for directory summaries.
CRITICAL: All IndexRecords MUST contain account_id + owner_space in filters.
This is the LAST layer of tenant isolation at the vector level.

Missing filters cause SILENT cross-tenant leakage (no runtime error).
"""

import re
from typing import Final

from core.models import ContextNode, IndexRecord
from core.resource_identity import code_index_metadata

# URI pattern to extract account_id: ctx://{account}/...
_URI_ACCOUNT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^ctx://([^/]+)/"
)

# Required filter keys for tenant isolation
_REQUIRED_FILTER_KEYS: Final[tuple[str, ...]] = ("account_id", "owner_space")


class FilterValidationError(Exception):
    """Raised when IndexRecord filters fail validation."""

    def __init__(self, missing_keys: tuple[str, ...]) -> None:
        self.missing_keys = missing_keys
        super().__init__(
            f"IndexRecord.filters missing required keys: {missing_keys}. "
            "This causes SILENT cross-tenant leakage!"
        )


def _extract_account_id(uri: str) -> str:
    """Extract account_id from ContextNode URI.

    Args:
        uri: ContextNode URI like "ctx://acme/users/alice/memories/profile"

    Returns:
        Account ID (e.g., "acme")

    Raises:
        ValueError: If URI format is invalid
    """
    match = _URI_ACCOUNT_PATTERN.match(uri)
    if not match:
        raise ValueError(f"Invalid URI format, cannot extract account_id: {uri}")
    return match.group(1)


def agent_memory_index_ids(node: ContextNode) -> list[str]:
    """Enumerate the layers and all recorded unit identities for index cleanup."""
    return [
        IndexRecord.generate_id(node.uri + "#agent-unit=" + key, 2)
        for key in node.metadata["agent_memory"]["units"]
    ] + [
        IndexRecord.generate_id(node.uri, 0),
        IndexRecord.generate_id(node.uri, 1),
        IndexRecord.generate_id(node.uri + "/content.md", 2),
    ]


def build_index_records(node: ContextNode) -> list[IndexRecord]:
    """Expand ContextNode into an L0 record and optional L2 record.

    For nodes with indexing enabled:
    - Level 0: abstract (L0) - always emitted for first-pass recall
    - Level 2: content (L2) - emitted when content is present for deep reading

    Typed Agent nodes emit L0, L1 and complete L2 units. DirectoryEventHandler
    owns L1 overview records for directory summaries. Nodes with
    ``metadata["indexing_disabled"] is True`` produce no records.

    CRITICAL: All records include account_id + owner_space in filters
    to prevent cross-tenant leakage at the vector index level.
    """
    if node.metadata.get("indexing_disabled") is True:
        return []

    # Extract account_id from URI for tenant isolation
    account_id = _extract_account_id(node.uri)

    # Build base filters shared by every emitted record
    base_filters: dict[str, str | int | float | bool] = {
        "account_id": account_id,
        "owner_space": node.owner_space,
        "category": node.category,
        "context_type": node.context_type,
    }

    code_metadata = code_index_metadata(node.metadata)
    resource_type = code_metadata.get("resource_type")
    if isinstance(resource_type, str):
        base_filters["resource_type"] = resource_type

    # Build metadata shared by emitted L0/L2 records
    base_metadata: dict[str, str | int | float | bool] = {
        "abstract": node.abstract,
        "category": node.category,
        "context_type": node.context_type,
        "parent_uri": node.parent_uri,
        "has_overview": bool(node.overview),
        "has_content": bool(node.content),
    }
    base_metadata.update(code_metadata)

    if node.metadata.get("agent_memory_version") == 1:
        from extraction.agent_memory import memory_units, node_memory_fields, render_agent_memory

        fields = node_memory_fields(node)
        units = memory_units(node.category, fields)
        base_metadata.update(
            agent_memory_version=1,
            routing_key=node.metadata["routing_key"], abstract=node.abstract,
        )
        records = [
            IndexRecord(IndexRecord.generate_id(node.uri, level), node.uri, level, text,
                        dict(base_filters), {**base_metadata, "level": label})
            for level, label, text in ((0, "abstract", node.abstract), (1, "overview", node.overview))
            if units and text
        ]
        records.extend(
            IndexRecord(
                IndexRecord.generate_id(node.uri + "#agent-unit=" + item["key"], 2),
                node.uri + "/content.md", 2,
                render_agent_memory(
                    node.category,
                    {**fields, "units": [item]} if node.category == "tool" else item,
                    grounding=item.get("grounding"),
                ),
                dict(base_filters),
                {**base_metadata, "level": "content", "parent_uri": node.uri,
                 "abstract": item["title"], "has_overview": False,
                 "unit_key": item["key"], "agent_memory_unit_key": item["key"],
                 "grounding": item.get("grounding", []),
                 "unit_revision": node.metadata["agent_memory"]["units"][item["key"]]["revision"]},
            )
            for item in units
        )
        _validate_filters(records)
        return records

    # Enrich metadata with structured extraction fields (when/who/where)
    for key in ("when", "who", "where", "routing_key"):
        val = node.metadata.get(key)
        if val:
            base_metadata[key] = val

    records: list[IndexRecord] = []

    # Level 0: Abstract (L0) - ≤100 char summary for first-pass recall
    records.append(IndexRecord(
        id=IndexRecord.generate_id(node.uri, 0),
        uri=node.uri,
        level=0,
        text=node.abstract,
        filters=dict(base_filters),
        metadata={"level": "abstract", **base_metadata},
    ))

    # Level 2: Content (L2) - full content for deep reading
    # L2 URI points to the actual .md file; parent_uri is the entity directory
    # so that search_children(parent_uri=<entity_dir>) can find L2 leaves.
    if node.content:
        l2_uri = node.uri.rstrip("/") + "/content.md"
        l2_metadata = {
            **base_metadata,
            "parent_uri": node.uri,
        }
        records.append(IndexRecord(
            id=IndexRecord.generate_id(l2_uri, 2),
            uri=l2_uri,
            level=2,
            text=node.content,
            filters=dict(base_filters),
            metadata={"level": "content", **l2_metadata},
        ))

    # Validate every emitted record has the required filters
    _validate_filters(records)

    return records


def _validate_filters(records: list[IndexRecord]) -> None:
    """Validate that all IndexRecords have required filter keys.

    This is the LAST line of defense against cross-tenant leakage.
    Missing filters cause SILENT leakage (no runtime error in vector search).

    Args:
        records: IndexRecords to validate

    Raises:
        FilterValidationError: If any record is missing required keys
    """
    for record in records:
        missing_keys = tuple(
            key for key in _REQUIRED_FILTER_KEYS
            if key not in record.filters
        )
        if missing_keys:
            raise FilterValidationError(missing_keys)


def build_single_record(
    uri: str,
    level: int,
    text: str,
    account_id: str,
    owner_space: str,
    category: str,
    context_type: str,
) -> IndexRecord:
    """Build a single IndexRecord with explicit parameters.

    Utility function for building individual records.
    Used by OutboxWorker when processing events.

    Args:
        uri: ContextNode URI
        level: Index level (0, 1, or 2)
        text: Text content for embedding
        account_id: Tenant ID (REQUIRED for isolation)
        owner_space: user_space or agent_space (REQUIRED for isolation)
        category: Memory category (profile, preference, etc.)
        context_type: MEMORY | SKILL | RESOURCE

    Returns:
        IndexRecord with mandatory filters

    Raises:
        FilterValidationError: If account_id or owner_space is empty
    """
    if not account_id:
        raise FilterValidationError(("account_id",))
    if not owner_space:
        raise FilterValidationError(("owner_space",))

    return IndexRecord(
        id=IndexRecord.generate_id(uri, level),
        uri=uri,
        level=level,
        text=text,
        filters={
            "account_id": account_id,
            "owner_space": owner_space,
            "category": category,
            "context_type": context_type,
        },
        metadata={
            "level": str(level),
            "category": category,
            "context_type": context_type,
        },
    )


def build_record_id(uri: str, level: int) -> str:
    """Build IndexRecord ID from URI and level.

    Args:
        uri: ContextNode URI
        level: Index level (0, 1, or 2)

    Returns:
        IndexRecord ID (sha256 hash)
    """
    return IndexRecord.generate_id(uri, level)
