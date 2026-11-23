# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Archive merge helpers for compact session history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from session.models import ArchiveEntry


@dataclass
class MergedArchive:
    overview: str
    abstract: str
    messages: list[dict]
    metadata: dict[str, Any] = field(default_factory=dict)


class ArchiveMerger:
    """Merge multiple session archives into one compact archive payload."""

    def __init__(self, max_messages: int = 200):
        self.max_messages = max(1, max_messages)

    def merge(self, archives: list[ArchiveEntry]) -> MergedArchive:
        if not archives:
            return MergedArchive(
                overview="",
                abstract="",
                messages=[],
                metadata={"source_archive_ids": []},
            )

        ordered = sorted(archives, key=lambda entry: entry.created_at or "")
        latest = ordered[-1]
        source_ids = [entry.archive_id for entry in ordered]
        return MergedArchive(
            overview=self._merge_overview(ordered),
            abstract=self._merge_abstract(ordered),
            messages=self._merge_messages(ordered, latest),
            metadata={
                "source_archive_ids": source_ids,
                "latest_archive_id": latest.archive_id,
                "merged_at": datetime.now(timezone.utc).isoformat(),
                "source_count": len(ordered),
            },
        )

    def _merge_overview(self, archives: list[ArchiveEntry]) -> str:
        sections = ["## Merged Archive History"]
        for entry in archives:
            text = (entry.overview or entry.abstract or "").strip()
            if text:
                sections.append(f"### {entry.archive_id}\n{text}")
        return "\n\n".join(sections)

    def _merge_abstract(self, archives: list[ArchiveEntry]) -> str:
        abstracts = [entry.abstract.strip() for entry in archives if entry.abstract.strip()]
        if not abstracts:
            return f"Merged {len(archives)} session archives"
        return " | ".join(abstracts)

    def _merge_messages(
        self,
        archives: list[ArchiveEntry],
        latest: ArchiveEntry,
    ) -> list[dict]:
        latest_messages = list(latest.messages or [])
        remaining_budget = max(0, self.max_messages - len(latest_messages))
        older_messages: list[dict] = []
        for entry in archives:
            if entry.archive_id == latest.archive_id:
                continue
            older_messages.extend(list(entry.messages or []))
        older_tail = older_messages[-remaining_budget:] if remaining_budget else []
        return older_tail + latest_messages
