# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Commit package - Write path orchestration.

Orchestrates the pipeline from extraction to persistence:
- Extract candidates via CandidateExtractor
- Plan write actions via MergePolicy
- Build ContextNode via ArchiveBuilder
- Write to storage via ContextFS
- Register OutboxEvents for async indexing
"""

from .merge_policies import (
    ProfilePolicy,
    AggregateTopicPolicy,
    AppendOnlyPolicy,
    SkillToolPolicy,
)
from .policy_router import PolicyRouter
from .archive_builder import ArchiveBuilder
from .candidate_pipeline import CandidatePipeline
from .context_writer import ContextWriter
from .outbox_store import OutboxStore

__all__ = [
    "ProfilePolicy",
    "AggregateTopicPolicy",
    "AppendOnlyPolicy",
    "SkillToolPolicy",
    "PolicyRouter",
    "ArchiveBuilder",
    "CandidatePipeline",
    "ContextWriter",
    "OutboxStore",
]
