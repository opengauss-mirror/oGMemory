"""Session Summary Generator — L0 structured summaries via compact-style prompt.

Generates structured summaries of conversation sessions to capture micro-details
(names, items, dates, feelings) that structured extraction may miss.
These summaries (L0) are injected into sessionContext at retrieval time,
complementing structured extraction nodes in retrievedEvidence.

Inspired by Claude Code's compact summarization approach.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from core.errors import LLMTransientError
from core.models import CandidateMemory, RequestContext

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "events_with_dates": {
            "type": "string",
            "description": "Events with absolute dates: [date] description. One per line.",
        },
        "facts_and_details": {
            "type": "string",
            "description": "Specific facts: names, items, locations, numbers mentioned.",
        },
        "preferences": {
            "type": "string",
            "description": "Likes, dislikes, attitudes expressed by participants.",
        },
        "plans_and_intentions": {
            "type": "string",
            "description": "Future plans, goals, intentions mentioned.",
        },
        "emotional_reactions": {
            "type": "string",
            "description": "How participants felt about events or topics.",
        },
        "key_details": {
            "type": "string",
            "description": "Items, books, movies, music, places mentioned with specifics.",
        },
    },
    "required": [
        "events_with_dates",
        "facts_and_details",
        "preferences",
        "plans_and_intentions",
    ],
}

SUMMARY_PROMPT = """\
Summarize this conversation, preserving ALL specific details that someone might ask about later.
Session date: {session_time}. Participants: {participants}

IMPORTANT: Convert ALL relative times to absolute dates using the session date.
For example: "last weekend" → calculate from session date, "next month" → session month + 1.

Return a JSON object with these fields:

- "events_with_dates": Events with absolute dates. Format: "[YYYY-MM-DD] description". One per line.
- "facts_and_details": Specific facts about people, things, places. Include exact names, titles, numbers.
- "preferences": What participants like, dislike, or feel strongly about.
- "plans_and_intentions": Future plans, goals, or things they intend to do.
- "emotional_reactions": How participants felt about specific events or topics.
- "key_details": Lists of specific items: books (with titles), music (with artist names), items bought, places visited, etc.

BE EXHAUSTIVE — include every specific detail, name, title, and number. Do not summarize away details.

Conversation:
{conversation}"""


class SessionSummaryGenerator:
    """Generate structured L0 summaries from conversation sessions."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def generate(
        self,
        messages_text: str,
        session_time: datetime | None = None,
        participants: str = "",
    ) -> dict | None:
        """Generate L0 summary from formatted conversation text.

        Args:
            messages_text: Full conversation text (all messages concatenated).
            session_time: Session date for resolving relative times.
            participants: Comma-separated participant names.

        Returns:
            Dict with structured summary fields, or None on failure.

        Raises:
            LLMTransientError: Temporary provider failure requiring recovery.
        """
        if not messages_text or not messages_text.strip():
            return None

        time_str = session_time.strftime("%Y-%m-%d") if session_time else "unknown"

        # Truncate very long sessions to stay within LLM context
        text = messages_text[:8000]

        prompt = SUMMARY_PROMPT.format(
            session_time=time_str,
            participants=participants or "unknown",
            conversation=text,
        )

        try:
            result = self._llm.complete_json(prompt, schema=SUMMARY_SCHEMA)
            if result and isinstance(result, dict):
                return result
        except LLMTransientError:
            raise
        except Exception as exc:
            logger.warning("SessionSummaryGenerator failed: %s", exc)

        return None

    @staticmethod
    def format_summary(summary: dict) -> str:
        """Format structured summary dict into readable text for sessionContext injection."""
        parts = []
        section_map = {
            "events_with_dates": "Events",
            "facts_and_details": "Facts",
            "preferences": "Preferences",
            "plans_and_intentions": "Plans",
            "emotional_reactions": "Reactions",
            "key_details": "Key Details",
        }
        for key, label in section_map.items():
            value = summary.get(key, "")
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            if value and str(value).strip():
                parts.append(f"### {label}\n{value}")

        return "\n\n".join(parts)

    @staticmethod
    def build_candidate(
        summary: dict,
        formatted_text: str,
        session_time: datetime | None = None,
        participants: str = "",
        session_id: str = "",
        chunk_index: int = 0,
    ) -> CandidateMemory:
        """Build a CandidateMemory from generated summary for writing to storage."""
        time_str = ""
        if session_time:
            try:
                if isinstance(session_time, datetime):
                    time_str = session_time.strftime("%Y%m%d")
            except Exception:
                pass

        sid = session_id[:8] if session_id else "unknown"
        routing_key = f"summary_{time_str}_{sid}_{chunk_index}"

        abstract = formatted_text[:200].replace("\n", " ")
        if len(formatted_text) > 200:
            abstract += "..."

        return CandidateMemory(
            category="session_summary",
            owner_scope="user",
            routing_key=routing_key,
            abstract=abstract,
            overview=formatted_text[:500],
            content=formatted_text,
            confidence=1.0,
            when=str(session_time) if session_time else None,
            who=participants or None,
        )
