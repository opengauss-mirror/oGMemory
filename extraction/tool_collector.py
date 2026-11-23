# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tool usage statistics collector for extraction pipeline.

Collects tool call metadata from messages, aggregates statistics per tool,
and formats them for injection into the extraction prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUsageSummary:
    """Aggregated usage statistics for a single tool."""

    tool_name: str
    call_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    total_duration_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.call_count if self.call_count > 0 else 0.0

    def to_extraction_text(self) -> str:
        """Generate a single-line summary for injection into extraction prompt."""
        return (
            f"Tool '{self.tool_name}': {self.call_count} calls, "
            f"{self.success_rate:.0%} success "
            f"({self.success_count} ok, {self.fail_count} fail)"
        )


def parse_message_parts(content: Any) -> dict:
    """Parse message content, separating text from tool call metadata.

    Handles both plain strings and OpenClaw structured content
    (list of dicts with type, tool_name, tool_status, etc.).

    Args:
        content: Message content (str or list)

    Returns:
        {"text": str, "tool_calls": list[dict]}
    """
    if isinstance(content, str):
        return {"text": content, "tool_calls": []}

    if isinstance(content, list):
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    text_parts.append(block)
                continue

            block_type = block.get("type", "")

            if block_type == "tool_call" or "tool_name" in block:
                # Extract tool call metadata
                tool_calls.append({
                    "tool_name": block.get("tool_name", ""),
                    "tool_input": block.get("tool_input", {}),
                    "tool_output": block.get("tool_output", ""),
                    "tool_status": block.get("tool_status", ""),
                    "duration_ms": block.get("duration_ms", 0) or 0,
                    "prompt_tokens": block.get("prompt_tokens", 0) or 0,
                    "completion_tokens": block.get("completion_tokens", 0) or 0,
                })
            elif block_type == "text" or "text" in block:
                text_parts.append(block.get("text", ""))

        return {"text": " ".join(text_parts).strip(), "tool_calls": tool_calls}

    return {"text": str(content) if content else "", "tool_calls": []}


def format_tool_stats_text(stats: dict[str, dict]) -> str:
    """Format tool statistics into text for injection into extraction prompt.

    Args:
        stats: Aggregated tool usage stats dict

    Returns:
        Formatted string, or empty string if no stats
    """
    if hasattr(stats, "tool_stats"):
        stats = stats.tool_stats

    if not stats:
        return ""

    summaries = []
    for name, s in sorted(stats.items(), key=lambda x: x[1]["call_count"], reverse=True):
        summary = ToolUsageSummary(
            tool_name=name,
            call_count=s.get("call_count", 0),
            success_count=s.get("success_count", 0),
            fail_count=s.get("fail_count", 0),
            total_duration_ms=s.get("total_duration_ms", 0.0),
            total_prompt_tokens=s.get("total_prompt_tokens", 0),
            total_completion_tokens=s.get("total_completion_tokens", 0),
        )
        summaries.append(summary.to_extraction_text())

    header = "**Tool Usage Statistics (this session):**"
    lines = [f"- {s}" for s in summaries]
    return header + "\n" + "\n".join(lines)


def update_buffer_stats(
    buffer_stats: Any,
    tool_calls: list[dict],
    *,
    session_id: str = "",
    category: str = "tool",
) -> None:
    """Update SessionBuffer.tool_usage_stats with new tool calls.

    Mutates buffer_stats in place.

    Args:
        buffer_stats: SessionBuffer.tool_usage_stats dict
        tool_calls: List of parsed tool call dicts from parse_message_parts
    """
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        if not name:
            continue

        if hasattr(buffer_stats, "record_tool_call"):
            buffer_stats.record_tool_call(
                tool_name=name,
                category=tc.get("category", category),
                session_id=session_id,
                status=tc.get("tool_status", ""),
                duration_ms=tc.get("duration_ms", 0) or 0,
                prompt_tokens=tc.get("prompt_tokens", 0) or 0,
                completion_tokens=tc.get("completion_tokens", 0) or 0,
                llm_call_id=tc.get("llm_call_id"),
            )
            continue

        if name not in buffer_stats:
            buffer_stats[name] = {
                "call_count": 0,
                "success_count": 0,
                "fail_count": 0,
                "total_duration_ms": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            }

        s = buffer_stats[name]
        s["call_count"] += 1
        status = tc.get("tool_status", "")
        if status == "completed" or status == "success":
            s["success_count"] += 1
        else:
            s["fail_count"] += 1
        s["total_duration_ms"] += tc.get("duration_ms", 0) or 0
        s["total_prompt_tokens"] += tc.get("prompt_tokens", 0) or 0
        s["total_completion_tokens"] += tc.get("completion_tokens", 0) or 0
