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
            "description": "Events with source-supported year, month, day or range precision: [time] description. Keep ambiguous periods with their session anchor, plus stated status and uncertainty. One per line.",
        },
        "facts_and_details": {
            "type": "string",
            "description": "Specific facts and item details: exact names, book/movie titles, artists, objects, locations and numbers. Each fact appears once.",
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

IMPORTANT: Resolve relative times using the session date, preserving the source's year, month,
day or range precision. Use a specific day only when the source supports that day.
Resolve relative years to calendar years (YYYY), and relative months to calendar months (YYYY-MM).
Resolve day-relative expressions such as yesterday against the session date (YYYY-MM-DD).
Weeks and weekends denote time ranges. Resolve last week or last weekend to a range only when
the source and context establish its boundaries; otherwise retain the original expression
together with the session-date anchor. Keep explicit dates and ranges at their stated precision.
Preserve approximate or unresolved times and their stated uncertainty.
Retain each event's planned, attempted, ongoing or completed status and qualifications such as
"tried to", "might" and "plans to" in every section that mentions it.

Return a JSON object with these fields:

- "events_with_dates": Events at the source-supported precision. Format: "[YYYY] description", "[YYYY-MM] description", "[YYYY-MM-DD] description", or "[start to end] description". Keep uncertain times as stated. One per line.
- "facts_and_details": Specific facts about people, things and places, including books, movies, music, artists, items bought and places visited. Preserve every exact name, title and number here.
- "preferences": What participants like, dislike, or feel strongly about.
- "plans_and_intentions": Future plans, goals, or things they intend to do.
- "emotional_reactions": How participants felt about specific events or topics.

BE EXHAUSTIVE — include every specific detail, name, title, and number. Do not summarize away details.
Assign each fact to its most specific section exactly once, retaining its names, dates and qualifiers together.

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
