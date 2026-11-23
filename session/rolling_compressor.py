# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Rolling compressor for session state layer (Layer 2).

Generates a full-rewrite compressed summary of recent conversation turns
every N turns or when token count exceeds threshold. The compressed text
replaces (not appends to) the previous window summary.

Requires LLM — no fallback. If LLM is unavailable, compression is skipped
and the window state remains unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from core.interfaces import LLM
from session.models import SessionMessage, SessionWindowState
from session.session_state import SessionState

logger = logging.getLogger("ogmem.session")


class RollingCompressor:
    """Compresses recent session messages into a rolling window summary.

    L0 (full rewrite) compression: regenerate the entire window summary
    from scratch on each compression cycle. Requires LLM — no fallback.
    """

    def __init__(self, llm: Optional[LLM] = None, fallback_enabled: bool = False):
        self._llm = llm
        self._fallback_enabled = fallback_enabled

    def compress(
        self,
        messages: list[SessionMessage],
        window_state: SessionWindowState,
        session_state: SessionState | None = None,
        session_id: str | None = None,
    ) -> SessionWindowState:
        """Compress messages into a new window state.

        Args:
            messages: All messages in the current session buffer.
            window_state: Current window state (updated in place).
            session_state: Optional durable session task/commitment state.
            session_id: Session identifier for durable state lookup.

        Returns:
            Updated SessionWindowState with new compressed text.
            If LLM is unavailable or returns empty, window_state is unchanged.
        """
        if not messages:
            return window_state

        if self._llm is None:
            if not self._fallback_enabled:
                logger.debug("RollingCompressor: no LLM, skipping compression")
                return window_state
            result = self._fallback_compress(messages)
        else:
            result = self._llm_compress(messages)
            if not result:
                if not self._fallback_enabled:
                    logger.warning("RollingCompressor: LLM returned empty result, skipping")
                    return window_state
                logger.warning("RollingCompressor: LLM unavailable, using fallback")
                result = self._fallback_compress(messages)

        turn_count = sum(1 for m in messages if m.role == "user")
        token_count = sum(m.estimated_tokens for m in messages)

        # Populate structured fields from LLM response
        window_state.active_task = result.get("active_task", "")
        window_state.confirmed_constraints = result.get("confirmed_constraints", [])
        window_state.recent_decisions = result.get("recent_decisions", [])
        window_state.open_loops = result.get("open_loops", [])
        window_state.uncertainties = result.get("uncertainties", [])
        window_state.compressed_text = result.get("summary", "")
        window_state.turn_count_at_last_compress = turn_count
        window_state.token_count_at_last_compress = token_count

        if session_state is not None and session_id:
            try:
                from session.session_state_bridge import apply_session_state_bridge

                apply_session_state_bridge(
                    window_state,
                    session_state,
                    session_id,
                    turn_count=sum(1 for m in messages if m.role == "user"),
                    force=True,
                )
            except Exception as exc:
                logger.warning("RollingCompressor SessionState bridge failed: %s", exc)

        return window_state

    def _llm_compress(self, messages: list[SessionMessage]) -> dict:
        """Use LLM to generate a full-rewrite window summary with structured fields."""
        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )

        prompt = (
            "Analyze this conversation and extract structured state.\n"
            "Output in the same language as the conversation.\n"
            "Preserve proper nouns, version numbers, file names, and error codes verbatim.\n"
            "Keep 'summary' field under 150 words.\n\n"
            "Fields to extract:\n"
            "- active_task: What is the user currently working on?\n"
            "- confirmed_constraints: What constraints have been established?\n"
            "- recent_decisions: What key decisions were made?\n"
            "- open_loops: What remains unresolved?\n"
            "- uncertainties: What is unclear or tentative?\n"
            "- summary: A concise narrative summary\n\n"
            f"CONVERSATION:\n{conversation}"
        )

        schema = {
            "type": "object",
            "properties": {
                "active_task": {
                    "type": "string",
                    "description": "Current task or sub-task being worked on",
                },
                "confirmed_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints that have been confirmed",
                },
                "recent_decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key decisions made in this segment",
                },
                "open_loops": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Unresolved items or questions",
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things we're not sure about",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise narrative summary of the conversation",
                },
            },
            "required": ["summary"],
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            return result
        except Exception as exc:
            logger.warning("RollingCompressor LLM failed: %s", exc)
            return {}

    def _fallback_compress(self, messages: list[SessionMessage]) -> dict:
        user_texts = [
            m.content.strip()
            for m in messages
            if m.role == "user" and m.content.strip()
        ]
        latest_user = user_texts[-1] if user_texts else ""
        active_task = ""
        if latest_user:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", latest_user)
                if sentence.strip()
            ]
            active_task = sentences[0] if sentences else latest_user.strip()

        open_loops: list[str] = []
        for text in user_texts:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text)
                if sentence.strip()
            ]
            for sentence in sentences:
                if sentence.endswith("?"):
                    open_loops.append(sentence)

        summary = self._truncate_summary_tail(" ".join(user_texts))
        return {
            "active_task": active_task,
            "confirmed_constraints": [],
            "recent_decisions": [],
            "open_loops": open_loops,
            "uncertainties": [],
            "summary": summary,
        }

    def _truncate_summary_tail(self, text: str, limit: int = 600) -> str:
        if len(text) <= limit:
            return text

        tail = text[-limit:]
        if text[-limit - 1].isspace():
            return tail.lstrip()

        boundary = re.search(r"\s+", tail)
        if boundary is None:
            return tail.lstrip()
        return tail[boundary.end():].lstrip()
