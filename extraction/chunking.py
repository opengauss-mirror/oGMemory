"""
Conversation chunking module - EverOS style MemCell boundary detection.

Implements three-phase chunking pipeline:
1. Force-split: Safety net for exceeding hard limits
2. LLM boundary detection: Semantic splitting based on topic/time changes
3. Flush tail: Pack remaining messages into final chunk
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from core.interfaces import LLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default limits for force splitting
DEFAULT_MAX_TOKENS = 8000
DEFAULT_MAX_MESSAGES = 500
DEFAULT_MIN_CHUNK_TOKENS = 300

# Time gap threshold for boundary detection (4 hours)
TIME_GAP_THRESHOLD = timedelta(hours=4)

# JSON schema for boundary detection response
_BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "boundaries": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "should_wait": {"type": "boolean"},
    },
    "required": ["reasoning", "boundaries", "should_wait"],
}

# LLM prompt for batch boundary detection (adapted from EverOS)
_CONV_BATCH_BOUNDARY_DETECTION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an episodic memory boundary detection expert. Your task is to find all natural "episode boundaries" in a continuous conversation and split it into meaningful, independently memorable segments (MemCells). Your core principle is **"default to merging, split cautiously"**.

### Input Format
The following is a complete chronological conversation log. Each message is prefixed with a 1-based index and timestamp:

```
{messages}
```

### When to split

Add a boundary (by message number) only when **clear signals** appear:
- **Cross-day split (highest priority):** Adjacent messages have different calendar dates — MUST split at the date boundary.
- **Substantive topic change:** Conversation shifts from one concrete topic to a completely unrelated one (e.g., project architecture → weekend plans).
- **Task completion + new topic:** A closing message ("migration done") belongs to its task's episode; split only when the **next** message opens a genuinely unrelated topic.
- **Long gap + new topic:** Time gap > 4 hours AND new messages have no clear connection to prior conversation.

**Do NOT split for:**
- Greetings, farewells ("bye", "thanks") — keep with the main episode
- Transition phrases ("by the way", "oh also") — usually continue the current episode
- Brief pauses (< 4 hours) followed by the same topic

### `should_wait`
Set to `true` when the **last segment** has insufficient information to determine its episode context:
- **Non-text messages:** Only media placeholders (`[image]`, `[video]`, `[file]`) with no accompanying text
- **Intent-free short replies:** Minimal responses like "ok", "sure", "got it", "😂"
- **System or non-conversational messages:** System notifications (join/leave group, payment reminders, etc.) cannot themselves determine episode boundaries — wait for the next human message to decide
- **Ambiguous intermediate state:** Gap of 30 min–4 hours with content that is neither clearly continuing nor clearly starting a new topic

### Decision Principles
- **Merge by default:** When in doubt, do not split; only split on clear signals
- **Content over form:** Greetings and farewells belong to the episode they serve, not their own
- **Process continuity:** Consecutive actions toward the same goal (e.g., create group → post first instruction) form one episode
- **System messages don't trigger splits:** The episode context of a system message is determined by the next human message that follows it

### Examples

**Example 1 — one boundary:**
Input messages:
```
[1] [2024-03-10 09:00:00+00:00] Alice: Can you help me debug the login issue?
[2] [2024-03-10 09:01:00+00:00] Bob: Sure, let me check the logs.
[3] [2024-03-10 09:05:00+00:00] Bob: Found it — a null pointer in AuthService line 42.
[4] [2024-03-10 09:06:00+00:00] Alice: Fixed, thanks!
[5] [2024-03-11 10:00:00+00:00] Alice: Hey, are you free for lunch today?
[6] [2024-03-11 10:01:00+00:00] Bob: Sure, 12:30?
```
Output:
```json
{{
    "reasoning": "Messages 1-4 are a complete bug-fix episode; message 5 starts a new day with an unrelated lunch topic.",
    "boundaries": [4],
    "should_wait": false
}}
```

**Example 2 — no boundary:**
Input messages:
```
[1] [2024-03-10 14:00:00+08:00] Alice: What's the status of the Q2 roadmap?
[2] [2024-03-10 14:02:00+08:00] Bob: About 60% done. Need to finalize the API specs.
[3] [2024-03-10 14:10:00+08:00] Alice: OK, let's review the specs tomorrow.
```
Output:
```json
{{
    "reasoning": "All messages are part of the same Q2 roadmap discussion with no topic change.",
    "boundaries": [],
    "should_wait": false
}}
```

### Output Format
Return strictly in the following JSON format:
```json
{{
    "reasoning": "<one sentence explaining all boundary decisions>",
    "boundaries": [<1-indexed message numbers after which to split>],
    "should_wait": <boolean, whether the last segment has insufficient information>
}}
```

**`boundaries: []` means all messages belong to the same episode — no split.**

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkingResult:
    """Result from conversation chunking."""
    chunks: list[list[dict]]
    should_wait: bool = False


# ---------------------------------------------------------------------------
# Conversation Chunker
# ---------------------------------------------------------------------------

class ConversationChunker:
    """EverOS-style conversation chunker with three-phase pipeline.

    Phase 1: Force-split - safety net when exceeding hard limits
    Phase 2: LLM boundary detection - semantic splitting based on topic/time
    Phase 3: Flush tail - pack remaining messages into final chunk
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
        use_llm_boundary: bool = True,
    ):
        """Initialize the chunker.

        Args:
            llm: LLM instance for boundary detection. If None, only force-split.
            max_tokens: Maximum tokens per chunk before force-split triggers.
            max_messages: Maximum messages per chunk before force-split triggers.
            min_chunk_tokens: Minimum tokens to avoid merging too-small chunks.
            use_llm_boundary: Whether to use LLM for semantic boundary detection.
        """
        self._llm = llm
        self._max_tokens = max_tokens
        self._max_messages = max_messages
        self._min_chunk_tokens = min_chunk_tokens
        self._use_llm_boundary = use_llm_boundary and (llm is not None)

    def chunk_messages(
        self,
        messages: list[dict],
        flush: bool = False,
    ) -> ChunkingResult:
        """Split conversation into chunks using three-phase pipeline.

        Args:
            messages: List of message dicts with "role", "content", "timestamp"
            flush: If True, force pack remaining messages into final chunk

        Returns:
            ChunkingResult with chunks and should_wait flag
        """
        if not messages:
            return ChunkingResult(chunks=[], should_wait=False)

        # Step 1: If small enough, return single chunk
        if self._is_small_batch(messages):
            return ChunkingResult(chunks=[messages], should_wait=True)

        chunks: list[list[dict]] = []
        remaining = messages[:]
        should_wait = False

        # Step 2: Force-split loop
        while remaining:
            total_tokens = self._estimate_tokens_batch(remaining)
            total_messages = len(remaining)

            exceeds_token = total_tokens >= self._max_tokens
            exceeds_count = total_messages >= self._max_messages

            if not exceeds_token and not exceeds_count:
                # Within limits, exit force-split loop
                break

            split_at = self._find_force_split_point(remaining)
            trigger = "token_limit" if exceeds_token else "message_limit"

            logger.debug(
                f"[Chunking] Force split: tokens={total_tokens}/{self._max_tokens}, "
                f"messages={total_messages}/{self._max_messages}, split_at={split_at}, "
                f"trigger={trigger}"
            )

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        # Step 3: LLM boundary detection
        if remaining and self._use_llm_boundary:
            try:
                llm_result = self._llm_boundary_detect(remaining)
                prev = 0
                for boundary in llm_result.boundaries:
                    # boundary is 1-indexed, convert to 0-indexed exclusive end
                    end_idx = boundary  # 1-indexed, use as exclusive end index
                    segment = remaining[prev:end_idx]
                    if segment:
                        chunks.append(segment)
                    prev = end_idx

                remaining = remaining[prev:]
                should_wait = llm_result.should_wait
            except Exception as e:
                logger.warning(f"[Chunking] LLM boundary detection failed: {e}, treating as single chunk")
                should_wait = True

        # Step 4: Flush tail
        if flush and remaining:
            logger.info(f"[Chunking] Flush: packing {len(remaining)} messages into final chunk")
            chunks.append(remaining)
            remaining = []
            should_wait = False  # Flush mode always returns should_wait=False

        # Step 5: Pack remaining (non-flush mode)
        if remaining:
            chunks.append(remaining)

        # Step 6: Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        return ChunkingResult(chunks=chunks, should_wait=should_wait)

    def _is_small_batch(self, messages: list[dict]) -> bool:
        """Check if batch is small enough to skip chunking.

        Only returns True for genuinely small batches that don't need
        force-split or LLM boundary detection.
        """
        # Check token limit first (handles small message count with large content)
        total_tokens = self._estimate_tokens_batch(messages)
        if total_tokens >= self._max_tokens * 0.3:
            return False

        # Check message count threshold
        msg_count = len(messages)
        if msg_count >= self._max_messages * 0.3:
            return False

        # Very small message count AND low token count
        if msg_count < 10 and total_tokens < self._max_tokens * 0.3:
            return True

        # Otherwise check if under 50% of token limit
        return total_tokens < self._max_tokens * 0.5

    def _find_force_split_point(self, messages: list[dict]) -> int:
        """Find how many messages to include in a force-split chunk.

        Starts with max_messages - 1, then reduces if token limit is exceeded.
        Guaranteed to return at least 1 and at most len(messages) - 1.

        Args:
            messages: Messages to split

        Returns:
            Number of messages to include (exclusive end index)
        """
        if len(messages) <= 1:
            return len(messages)

        # Start with message limit (leave at least 1 for next iteration)
        candidate = max(1, min(self._max_messages - 1, len(messages) - 1))

        # Reduce if token limit exceeded
        while (
            candidate > 1
            and self._estimate_tokens_batch(messages[:candidate]) >= self._max_tokens
        ):
            candidate = max(1, candidate // 2)

        return candidate

    def _llm_boundary_detect(self, messages: list[dict]) -> "BatchBoundaryResult":
        """Use LLM to detect semantic boundaries.

        Args:
            messages: Messages to analyze (already within limits)

        Returns:
            BatchBoundaryResult with boundaries (1-indexed) and should_wait
        """
        if not self._llm:
            return BatchBoundaryResult(boundaries=[], should_wait=True)

        messages_text = self._format_messages_with_indices(messages)

        prompt = _CONV_BATCH_BOUNDARY_DETECTION_PROMPT.format(
            messages=messages_text
        )

        logger.debug(
            f"[Chunking] LLM boundary detection: {len(messages)} messages, "
            f"prompt_len={len(prompt)}"
        )

        try:
            result = self._llm.complete_json(prompt, schema=_BOUNDARY_SCHEMA)
            boundaries = result.get("boundaries", [])

            # Validate boundaries: must be 1-indexed and within range
            valid_boundaries = [
                b for b in boundaries
                if isinstance(b, int) and 1 <= b < len(messages)
            ]

            if len(valid_boundaries) != len(boundaries):
                logger.warning(
                    f"[Chunking] Filtered {len(boundaries) - len(valid_boundaries)} "
                    f"out-of-range boundaries (total messages: {len(messages)})"
                )

            return BatchBoundaryResult(
                boundaries=sorted(valid_boundaries),
                should_wait=bool(result.get("should_wait", False)),
            )
        except Exception as e:
            logger.error(f"[Chunking] LLM boundary detection error: {e}")
            raise

    def _format_messages_with_indices(self, messages: list[dict]) -> str:
        """Format messages with 1-based indices and timestamps for LLM.

        Format: [N] [YYYY-MM-DD HH:MM:SS+TZ] role: content

        Args:
            messages: List of message dicts

        Returns:
            Formatted string with numbered messages
        """
        lines = []
        for i, msg in enumerate(messages, start=1):
            content = self._content_to_str(msg.get("content", ""))
            role = msg.get("role", "")
            timestamp = msg.get("timestamp", "")

            # Format timestamp
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.isoformat(sep=" ", timespec="seconds")
                    elif isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        time_str = dt.isoformat(sep=" ", timespec="seconds")
                except (ValueError, AttributeError, TypeError):
                    pass

            if content:
                if time_str:
                    lines.append(f"[{i}] [{time_str}] {role}: {content}")
                else:
                    lines.append(f"[{i}] {role}: {content}")

        return "\n".join(lines)

    def _content_to_str(self, content: Any) -> str:
        """Normalize message content to plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: len(text) // 4.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _estimate_tokens_batch(self, messages: list[dict]) -> int:
        """Estimate total tokens for a batch of messages.

        Args:
            messages: List of message dicts

        Returns:
            Total estimated tokens
        """
        total = 0
        for msg in messages:
            content = self._content_to_str(msg.get("content", ""))
            role = msg.get("role", "")
            # Format: "role: content"
            text = f"{role}: {content}" if role else content
            total += self._estimate_tokens(text)
        return total

    def _merge_small_chunks(
        self,
        chunks: list[list[dict]],
    ) -> list[list[dict]]:
        """Merge chunks that are too small with adjacent chunks.

        Args:
            chunks: List of message chunks

        Returns:
            Merged chunks
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[list[dict]] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            current_tokens = self._estimate_tokens_batch(current)

            # If current chunk is too small and not the last, merge with next
            if (
                current_tokens < self._min_chunk_tokens
                and i + 1 < len(chunks)
            ):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged.append(current + next_chunk)
                i += 2  # Skip next chunk since we merged it
            else:
                merged.append(current)
                i += 1

        return merged


@dataclass
class BatchBoundaryResult:
    """Result from batch boundary detection."""
    boundaries: list[int] = None  # 1-indexed message numbers after which to split
    should_wait: bool = False

    def __post_init__(self):
        if self.boundaries is None:
            self.boundaries = []
