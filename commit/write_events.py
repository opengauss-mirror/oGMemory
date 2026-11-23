# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Build memory write events with viewer ownership metadata."""

from __future__ import annotations

import uuid

from core.enums import EventType
from core.models import ContextNode, OutboxEvent


def build_write_event(node: ContextNode) -> OutboxEvent:
    """Build an UPSERT event for a memory node."""
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
