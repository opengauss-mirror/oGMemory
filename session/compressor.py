"""Session compression using LLM.

Compresses conversation history into overview and abstract summaries.
"""

import logging
from typing import Optional

from core.interfaces import LLM

logger = logging.getLogger(__name__)


class SessionCompressor:
    """Compresses session messages into condensed summaries.

    Uses LLM to generate:
    - overview: Structured summary with key topics, decisions, outcomes
    - abstract: Brief one-sentence summary (≤100 chars)

    Falls back to simple concatenation if LLM is not available.
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the compressor.

        Args:
            llm: Optional LLM instance for smart compression
        """
        self.llm = llm

    def compress(
        self,
        messages: list[dict],
        prev_overview: str = "",
        prev_abstract: str = "",
    ) -> tuple[str, str]:
        """Compress a session message history, optionally fusing with previous archive.

        Args:
            messages: List of message dicts with 'role' and 'content'
            prev_overview: Overview from the previous archive (L0)
            prev_abstract: Abstract from the previous archive (L1)

        Returns:
            Tuple of (overview, abstract)
            - overview: Structured summary of the session
            - abstract: Brief one-sentence summary (≤100 chars)
        """
        if not messages:
            return "", "Empty session"

        if self.llm is None:
            return self._fallback_compress(messages, prev_overview, prev_abstract)

        return self._llm_compress(messages, prev_overview, prev_abstract)

    def _llm_compress(
        self,
        messages: list[dict],
        prev_overview: str = "",
        prev_abstract: str = "",
    ) -> tuple[str, str]:
        """Use LLM to generate compressed summaries, fusing with previous archive context.

        Args:
            messages: List of message dicts
            prev_overview: Overview from the previous archive (L0)
            prev_abstract: Abstract from the previous archive (L1)

        Returns:
            Tuple of (overview, abstract)
        """
        conversation = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )

        prev_section = ""
        if prev_overview or prev_abstract:
            prev_section = (
                "PREVIOUS ARCHIVE (for continuity — integrate relevant context, "
                "do NOT simply copy):\n"
            )
            if prev_abstract:
                prev_section += f"Previous abstract: {prev_abstract}\n"
            if prev_overview:
                prev_section += f"Previous overview:\n{prev_overview}\n"
            prev_section += "\n"

        prompt = f"""{prev_section}Compress the following conversation into two summaries.
If previous archive context is provided, fuse its key context into the new
summary — preserving continuity while emphasizing the latest developments.

CONVERSATION:
{conversation}

Return JSON with this schema:
{{
    "overview": "Use EXACTLY these markdown sections:\\n## Main Topics\\n(bullet list of topics discussed)\\n## Decisions Made\\n(bullet list of decisions)\\n## Outcomes & Progress\\n(bullet list of outcomes)\\n## Action Items\\n(bullet list of next steps, or 'None' if empty)\\nIf previous archive is provided, integrate its key context under the relevant sections.",
    "abstract": "One-sentence summary (maximum 100 characters) capturing the combined essence"
}}

IMPORTANT: The overview MUST use the four markdown sections (Main Topics, Decisions Made, Outcomes & Progress, Action Items) with bullet lists. Do NOT write a single paragraph.
"""

        schema = {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "abstract": {"type": "string", "maxLength": 100},
            },
            "required": ["overview", "abstract"],
        }

        try:
            result = self.llm.complete_json(prompt, schema)
            overview = result.get("overview", "")
            abstract = result.get("abstract", "")
            if not isinstance(overview, str) or not isinstance(abstract, str):
                # Some compatible endpoints do not enforce nested JSON types.
                # Never pass a dict/list into archive TEXT columns: use the
                # existing deterministic fallback and retain the raw messages.
                logger.warning(
                    "Session summary fields must be strings (overview=%s, abstract=%s); "
                    "using deterministic compression fallback",
                    type(overview).__name__, type(abstract).__name__,
                )
                return self._fallback_compress(messages, prev_overview, prev_abstract)
            return overview, abstract[:100]

        except Exception:
            # Fall back to simple compression on error
            return self._fallback_compress(messages, prev_overview, prev_abstract)

    def _fallback_compress(
        self,
        messages: list[dict],
        prev_overview: str = "",
        prev_abstract: str = "",
    ) -> tuple[str, str]:
        """Simple concatenation-based compression (no LLM).

        Args:
            messages: List of message dicts
            prev_overview: Overview from the previous archive (L0)
            prev_abstract: Abstract from the previous archive (L1)

        Returns:
            Tuple of (overview, abstract)
        """
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

        overview_parts = []
        if prev_abstract:
            previous_lines = ["Previous context:"]
            previous_lines.append(f"Previous abstract: {prev_abstract}")
            overview_parts.append("\n".join(previous_lines))
        overview_parts.append(
            f"Session with {len(messages)} messages "
            f"({user_count} from user, {assistant_count} from assistant)."
        )

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    overview_parts.append(f"Started with: {preview}")
                    break

        overview = "\n\n".join(overview_parts)

        first_content = ""
        last_content = ""
        for msg in messages:
            if msg.get("role") == "user" and not first_content:
                first_content = msg.get("content", "")
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")

        if first_content and last_content:
            abstract = f"{first_content[:50]}... → {last_content[:30]}"
        elif first_content:
            abstract = first_content[:97] + "..."
        else:
            abstract = f"Session with {len(messages)} messages"

        if prev_abstract:
            abstract = f"{prev_abstract[:45]} | {abstract[:52]}"
        abstract = abstract[:100]

        return overview, abstract
