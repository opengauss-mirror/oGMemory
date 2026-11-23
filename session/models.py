# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Data models for session archival system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SessionMessage:
    """A single message in the session buffer."""

    id: str  # msg_{uuid4_hex}
    role: str  # "user" | "assistant"
    content: str  # message text
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.content) // 4)


@dataclass
class SessionMeta:
    """Session metadata persisted alongside the buffer."""

    session_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    message_count: int = 0
    commit_count: int = 0
    last_commit_at: str = ""

    # Identity bound at session creation — used to inherit userId
    # when subsequent requests (e.g. compose) don't carry it.
    account_id: str = ""
    user_id: str = ""
    agent_id: str = ""


@dataclass
class ArchiveEntry:
    """A single archived session.

    Represents a compressed snapshot of a session's conversation history.
    """

    archive_id: str  # Unique identifier (uuid4 or timestamp-based)
    session_id: str  # Original session identifier
    overview: str  # Structured overview of the session (→ overview.md)
    abstract: str  # Brief summary (≤100 chars) (→ abstract.md)
    messages: list[dict]  # Full message history (→ messages.json)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveWriteResult:
    """Result of writing an archive to storage."""

    archive_id: str
    session_id: str
    uri: str  # Full AGFS URI where archive was written
    success: bool
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionWindowState:
    """Rolling compressed window for session state layer (Layer 2).

    Structured state object for cognitive state assembly, not just
    chat history compression. Updated every N turns (not every turn like
    Layer 3, not never like Layer 1). Provides the LLM awareness of recent
    conversation flow without re-sending all messages. Goes into
    systemPromptSuffix for KV cache friendliness.
    """

    compressed_text: str = ""  # Fallback: full summary text
    turn_count_at_last_compress: int = 0  # Turns when last compressed
    token_count_at_last_compress: int = 0  # Token count when last compressed
    skills_text: str = ""  # Skills retrieved from compressed content

    # Structured cognitive state fields
    active_task: str = ""  # Current task or sub-task
    confirmed_constraints: list[str] = field(default_factory=list)  # Known constraints
    recent_decisions: list[str] = field(default_factory=list)  # Decisions made this session
    open_loops: list[str] = field(default_factory=list)  # Unresolved items
    uncertainties: list[str] = field(default_factory=list)  # Things we're unsure about
    last_accessed_at: str = ""  # Last compose/session-state access timestamp
    session_state_version: int = 0
    session_state_synced_at: str = ""
    session_state_sync_turn_count: int = 0
