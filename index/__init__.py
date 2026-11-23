# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Index package: Async index synchronization and repair.

This package handles:
- ContextNode → IndexRecord expansion (3-level)
- Outbox event processing with retry/DLQ
- Directory summary generation
- Repair job for corrupted nodes

Key invariant: All IndexRecords MUST contain account_id + owner_space in filters.
"""

from index.index_record_builder import (
    build_index_records,
    build_single_record,
    build_record_id,
    _extract_account_id,
    _validate_filters,
    FilterValidationError,
)

from index.outbox_worker import (
    OutboxWorker,
    WorkerResult,
    WorkerConfig,
    create_upsert_event,
)

from index.sql_notify_listener import SQLNotifyListener

from index.directory_summarizer import (
    DirectorySummarizer,
    DirectorySummary,
    ChildSummary,
    is_directory_uri,
)

from index.directory_event_handler import (
    DirectoryEventHandler,
    DirectoryEventResult,
    DirNode,
    DagStats,
)

from index.repair_job import (
    RepairJob,
    RepairReport,
    RepairJobSummary,
    AGFSSnapshot,
    MemoryAGFSSnapshot,
    OutboxEventSink,
    MemoryOutboxEventSink,
)

__all__ = [
    "build_index_records",
    "build_single_record",
    "build_record_id",
    "_extract_account_id",
    "_validate_filters",
    "FilterValidationError",
    "OutboxWorker",
    "WorkerResult",
    "WorkerConfig",
    "create_upsert_event",
    "SQLNotifyListener",
    "DirectorySummarizer",
    "DirectorySummary",
    "ChildSummary",
    "is_directory_uri",
    "DirectoryEventHandler",
    "DirectoryEventResult",
    "DirNode",
    "DagStats",
    "RepairJob",
    "RepairReport",
    "RepairJobSummary",
    "AGFSSnapshot",
    "MemoryAGFSSnapshot",
    "OutboxEventSink",
    "MemoryOutboxEventSink",
]
