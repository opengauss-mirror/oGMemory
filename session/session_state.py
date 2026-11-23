# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Session-level task state tracking for RCA pipeline.

S1 of the RuntimeContextAssembly pipeline: manages session-level task state
and commitments. This is the foundation for context-aware assembly.

Task state and commitments are tracked per session and used by the
assembly pipeline to provide relevant context.
"""

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

from core.models import ContextNode, RequestContext


logger = logging.getLogger("ogmem.session")


@dataclass
class TaskState:
    """Task state for a session.

    Tracks the current objective, stage, next step, and any blockers
    for the task being worked on in a session.

    Fields:
        objective: Overall task goal (e.g., "Debug Python performance issue")
        current_stage: Current stage of work (e.g., "Investigating root cause")
        next_step: Planned next action (e.g., "Profile memory usage")
        blockers: List of blocking items preventing progress
    """

    objective: str | None = None
    current_stage: str | None = None
    next_step: str | None = None
    blockers: list[str] = field(default_factory=list)


@dataclass
class Commitment:
    """A commitment made during a session.

    Commitments are promises or action items that emerge during conversation.
    They can be open (pending), fulfilled (completed), or expired (no longer relevant).

    Fields:
        content: Description of the commitment
        status: Current status - "open" | "fulfilled" | "expired"
        created_at: ISO timestamp when commitment was made
        resolved_at: ISO timestamp when commitment was resolved (for fulfilled)
    """

    content: str
    status: str = "open"  # open | fulfilled | expired
    created_at: str = ""
    resolved_at: str | None = None
    kind: str = ""  # constraint | decision | loop | empty for generic commitment

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.kind = str(self.kind or "").strip().lower()


class SessionState:
    """S1: Session-level task state tracking.

    Manages task state and commitments for sessions using in-memory storage.
    This is the foundation for context-aware assembly in the RCA pipeline.

    The session state provides:
    - Task context: What the user is working on
    - Commitments: Promises made during conversation
    - Open loops: Unresolved commitments that need attention
    """

    def __init__(self):
        """Initialize session state manager."""
        self._task_states: dict[str, TaskState] = {}
        self._commitments: dict[str, list[Commitment]] = {}
        self._versions: dict[str, int] = {}

    def _bump_version(self, session_id: str) -> None:
        self._versions[session_id] = self._versions.get(session_id, 0) + 1

    def get_version(self, session_id: str) -> int:
        return self._versions.get(session_id, 0)

    def get_task_state(self, session_id: str) -> TaskState:
        """Get task state for a session.

        Returns existing TaskState or creates a new empty one.
        Never returns None.

        Args:
            session_id: Session identifier

        Returns:
            TaskState for the session (empty if not previously set)
        """
        if session_id not in self._task_states:
            self._task_states[session_id] = TaskState()
        return self._task_states[session_id]

    def get_commitments(
        self, session_id: str, status: str = "open"
    ) -> list[Commitment]:
        """Get commitments for a session, optionally filtered by status.

        Args:
            session_id: Session identifier
            status: Filter by status ("open", "fulfilled", "expired").
                    If None, returns all commitments.

        Returns:
            List of commitments matching the status filter
        """
        if session_id not in self._commitments:
            return []

        commitments = self._commitments[session_id]

        if status is None:
            return commitments

        return [c for c in commitments if c.status == status]

    def update_task_state(self, session_id: str, state: TaskState) -> None:
        """Update task state for a session.

        Merges with existing state, only overwriting non-None fields.
        This allows partial updates without losing existing information.

        Args:
            session_id: Session identifier
            state: New task state (partial or complete)
        """
        current = self.get_task_state(session_id)

        # Merge: only overwrite non-None fields
        if state.objective is not None:
            current.objective = state.objective
        if state.current_stage is not None:
            current.current_stage = state.current_stage
        if state.next_step is not None:
            current.next_step = state.next_step
        if state.blockers:
            # Replace blockers list entirely
            current.blockers = list(state.blockers)

        self._task_states[session_id] = current
        self._bump_version(session_id)

    def add_commitment(self, session_id: str, commitment: Commitment) -> None:
        """Add a commitment to a session.

        Args:
            session_id: Session identifier
            commitment: Commitment to add
        """
        if session_id not in self._commitments:
            self._commitments[session_id] = []

        self._commitments[session_id].append(commitment)
        self._bump_version(session_id)

    def resolve_commitment(self, session_id: str, content: str) -> None:
        """Resolve a commitment by marking it as fulfilled.

        Finds commitment by content match (exact or substring) and sets
        status to "fulfilled" with resolved_at timestamp.

        Args:
            session_id: Session identifier
            content: Content string to match commitment
        """
        if session_id not in self._commitments:
            return

        now = datetime.now(timezone.utc).isoformat()

        changed = False
        for commitment in self._commitments[session_id]:
            if commitment.status == "open" and content in commitment.content:
                commitment.status = "fulfilled"
                commitment.resolved_at = now
                changed = True
        if changed:
            self._bump_version(session_id)

    def get_open_loops(self, session_id: str) -> list[Commitment]:
        """Get open (unresolved) commitments for a session.

        Open loops are commitments with status="open" that represent
        unresolved promises or action items.

        Args:
            session_id: Session identifier

        Returns:
            List of open commitments
        """
        return self.get_commitments(session_id, status="open")

    @staticmethod
    def _strip_prefix(content: str, prefix: str) -> str:
        lowered = content.lower()
        marker = f"{prefix.lower()}:"
        if lowered.startswith(marker):
            return content[len(marker):].strip()
        return content

    def get_confirmed_constraints(self, session_id: str) -> list[str]:
        result: list[str] = []
        for item in self.get_commitments(session_id, status="fulfilled"):
            content = item.content.strip()
            if item.kind == "constraint":
                result.append(content)
            elif content.lower().startswith("constraint:"):
                result.append(self._strip_prefix(content, "constraint"))
        return result

    def get_recent_decisions(self, session_id: str) -> list[str]:
        result: list[str] = []
        for item in self.get_commitments(session_id, status="fulfilled"):
            content = item.content.strip()
            if item.kind == "decision":
                result.append(content)
            elif content.lower().startswith("decision:"):
                result.append(self._strip_prefix(content, "decision"))
        return result

    @staticmethod
    def _optional_dataclass_payload(value: Any) -> dict | None:
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        return None

    def has_state(self, session_id: str) -> bool:
        task_state = self._task_states.get(session_id)
        has_task = bool(
            task_state
            and (
                task_state.objective
                or task_state.current_stage
                or task_state.next_step
                or task_state.blockers
            )
        )
        return has_task or bool(self.get_commitments(session_id, status=None))

    def to_dict(
        self,
        session_id: str,
        *,
        session_meta: Any = None,
        window_state: Any = None,
        extraction_summary: str | None = None,
    ) -> dict:
        task_state = self._task_states.get(session_id, TaskState())
        commitments = self.get_commitments(session_id, status=None)
        payload = {
            "task_state": asdict(task_state),
            "commitments": [asdict(c) for c in commitments],
        }
        meta_payload = self._optional_dataclass_payload(session_meta)
        if meta_payload is not None:
            payload["session_meta"] = meta_payload
        window_payload = self._optional_dataclass_payload(window_state)
        if window_payload is not None:
            payload["window_state"] = window_payload
        if extraction_summary is not None:
            payload["extraction_summary"] = extraction_summary
        return payload

    def load_dict(self, session_id: str, payload: dict) -> None:
        if not isinstance(payload, dict):
            payload = {}

        task_payload = payload.get("task_state") or {}
        if not isinstance(task_payload, dict):
            task_payload = {}

        blockers = task_payload.get("blockers")
        if isinstance(blockers, (list, tuple)):
            blockers = list(blockers)
        else:
            blockers = []

        task_state = TaskState(
            objective=task_payload.get("objective"),
            current_stage=task_payload.get("current_stage"),
            next_step=task_payload.get("next_step"),
            blockers=blockers,
        )

        commitments = []
        commitments_payload = payload.get("commitments") or []
        if not isinstance(commitments_payload, list):
            commitments_payload = []

        for item in commitments_payload:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            commitments.append(
                Commitment(
                    content=str(item.get("content", "")),
                    status=str(item.get("status", "open")),
                    created_at=str(item.get("created_at", "")),
                    resolved_at=item.get("resolved_at"),
                    kind=str(item.get("kind", "")),
                )
            )

        self._task_states[session_id] = task_state
        self._commitments[session_id] = commitments
        self._bump_version(session_id)

    @staticmethod
    def _has_malformed_commitments(payload: dict) -> bool:
        commitments = payload.get("commitments") or []
        if not isinstance(commitments, list):
            return True
        return any(not isinstance(item, dict) or not item.get("content") for item in commitments)

    @staticmethod
    def state_uri(account_id: str, session_id: str) -> str:
        return f"ctx://{account_id}/sessions/{session_id}/state.json"

    @staticmethod
    def legacy_state_uri(account_id: str, session_id: str) -> str:
        return f"ctx://{account_id}/sessions/{session_id}/state"

    def save(
        self,
        session_id: str,
        fs,
        ctx: RequestContext,
        *,
        session_meta: Any = None,
        window_state: Any = None,
        extraction_summary: str | None = None,
    ) -> bool:
        try:
            now = datetime.now(timezone.utc).isoformat()
            payload = self.to_dict(
                session_id,
                session_meta=session_meta,
                window_state=window_state,
                extraction_summary=extraction_summary,
            )
            node = ContextNode(
                uri=self.state_uri(ctx.account_id, session_id),
                context_type="RESOURCE",
                category="state",
                level=0,
                owner_space=f"session:{session_id}",
                abstract="Session task state and commitments",
                overview="Session task state and commitments",
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={
                    "session_id": session_id,
                    "owner_user_id": ctx.user_id,
                    "owner_agent_id": ctx.agent_id,
                    "updated_at": now,
                },
            )
            fs.write_node(node, ctx)
            return True
        except Exception as exc:
            logger.warning("SessionState save failed session=%s: %s", session_id, exc)
            return False

    def _load_payload_with_status(
        self,
        session_id: str,
        fs,
        ctx: RequestContext,
    ) -> tuple[str, dict | None]:
        uris = [
            self.state_uri(ctx.account_id, session_id),
            self.legacy_state_uri(ctx.account_id, session_id),
        ]
        had_error = False
        for uri in uris:
            try:
                if hasattr(fs, "exists") and not fs.exists(uri, ctx):
                    continue
                node = fs.read_node(uri, ctx)
                payload = json.loads(node.content or "{}")
                if not isinstance(payload, dict):
                    return "error", None
                if self._has_malformed_commitments(payload):
                    return "error", None
                return "loaded", payload
            except Exception as exc:
                had_error = True
                logger.warning(
                    "SessionState load failed session=%s uri=%s: %s",
                    session_id,
                    uri,
                    exc,
                )
        if hasattr(fs, "exists") and not had_error:
            return "missing", None
        return "error", None

    def load_payload(self, session_id: str, fs, ctx: RequestContext) -> dict | None:
        _status, payload = self._load_payload_with_status(session_id, fs, ctx)
        return payload

    def load_with_payload_status(
        self,
        session_id: str,
        fs,
        ctx: RequestContext,
    ) -> tuple[str, dict | None]:
        status, payload = self._load_payload_with_status(session_id, fs, ctx)
        if status == "loaded" and payload is not None:
            self.load_dict(session_id, payload)
        return status, payload

    def load_with_payload(self, session_id: str, fs, ctx: RequestContext) -> tuple[bool, dict | None]:
        status, payload = self.load_with_payload_status(session_id, fs, ctx)
        return status == "loaded", payload

    def load(self, session_id: str, fs, ctx: RequestContext) -> bool:
        loaded, _payload = self.load_with_payload(session_id, fs, ctx)
        return loaded

    def clear_session(self, session_id: str) -> None:
        """Clear all state for a session.

        Useful for session cleanup or testing.

        Args:
            session_id: Session identifier
        """
        changed = False
        if session_id in self._task_states:
            del self._task_states[session_id]
            changed = True
        if session_id in self._commitments:
            del self._commitments[session_id]
            changed = True
        if changed:
            self._bump_version(session_id)
