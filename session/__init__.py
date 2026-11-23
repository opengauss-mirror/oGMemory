# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Session archival system for ContextEngine.

Provides session compression and archival storage for long-term
conversation history persistence.

Components:
- SessionArchiveStore: AGFS-backed archive storage
- SQLSessionArchiveStore: PostgreSQL-backed archive storage
- SessionCompressor: LLM-based session compression
- SessionState: Task state and commitment tracking (RCA pipeline)
- SessionTopicBuffer: Prefetch result caching (RCA pipeline)
"""

from session.models import ArchiveEntry, ArchiveWriteResult, SessionMessage, SessionMeta, SessionWindowState
from session.archive_store import SessionArchiveStore
from session.sql_archive_store import SQLSessionArchiveStore
from session.archive_merger import ArchiveMerger, MergedArchive
from session.compressor import SessionCompressor
from session.rolling_compressor import RollingCompressor
from session.session_manager import SessionManager, SessionBuffer
from session.topic_buffer import SessionTopicBuffer, SlotContent
from session.session_state import SessionState, TaskState, Commitment
from session.topic_detector import TopicDetection, TopicDetector
from session.compression_quality import CompressionQualityEvaluator

__all__ = [
    "ArchiveEntry",
    "ArchiveWriteResult",
    "SessionMessage",
    "SessionMeta",
    "SessionWindowState",
    "SessionArchiveStore",
    "SQLSessionArchiveStore",
    "ArchiveMerger",
    "MergedArchive",
    "SessionCompressor",
    "RollingCompressor",
    "SessionManager",
    "SessionBuffer",
    "SessionTopicBuffer",
    "SlotContent",
    "SessionState",
    "TaskState",
    "Commitment",
    "TopicDetection",
    "TopicDetector",
    "CompressionQualityEvaluator",
]
