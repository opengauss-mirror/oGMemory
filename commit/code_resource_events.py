"""Pure Outbox event builders for durable code RESOURCE mutations.

The builders have no storage/client dependency.  Both SQL and the later AGFS
adapter can use the same payload shape without importing one another.
"""

from __future__ import annotations

import uuid

from core.enums import EventType
from core.models import ContextNode, IndexRecord, OutboxEvent


def build_code_resource_write_event(node: ContextNode) -> OutboxEvent:
    """Build a complete UPSERT event for one code RESOURCE node."""
    from index.index_record_builder import agent_memory_index_ids, build_index_records

    records_data = [
        {
            "id": record.id,
            "uri": record.uri,
            "level": record.level,
            "text": record.text,
            "filters": record.filters,
            "metadata": record.metadata,
        }
        for record in build_index_records(node)
    ]
    payload = {"records": records_data, "task_owner_spaces": _task_owner_spaces(node)}
    if node.metadata.get("agent_memory_version") == 1:
        payload["agent_memory_version"] = 1
        payload["ids_to_delete"] = agent_memory_index_ids(node)
    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.UPSERT_CONTEXT.value,
        uri=node.uri,
        payload=payload,
        status="PENDING",
    )


def _task_owner_spaces(node: ContextNode) -> list[str]:
    """Return the user and agent scopes allowed to inspect this task."""
    spaces = []
    if node.owner_space.startswith(("user:", "agent:")):
        spaces.append(node.owner_space)
    for metadata_key, prefix in (
        ("owner_user_id", "user"),
        ("owner_agent_id", "agent"),
    ):
        owner_id = str(node.metadata.get(metadata_key) or "")
        owner_space = f"{prefix}:{owner_id}" if owner_id else ""
        if owner_space and owner_space not in spaces:
            spaces.append(owner_space)
    return spaces


def build_code_resource_delete_event(uri: str) -> OutboxEvent:
    """Build a self-contained DELETE event for one code RESOURCE URI."""
    normalized_uri = str(uri).rstrip("/")
    l2_uri = normalized_uri + "/content.md"
    ids_to_delete = [
        IndexRecord.generate_id(normalized_uri, 0),
        IndexRecord.generate_id(normalized_uri, 1),
        IndexRecord.generate_id(l2_uri, 2),
    ]
    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.DELETE_CONTEXT.value,
        uri=normalized_uri,
        payload={"ids_to_delete": ids_to_delete},
        status="PENDING",
    )


__all__ = [
    "build_code_resource_delete_event",
    "build_code_resource_write_event",
]
