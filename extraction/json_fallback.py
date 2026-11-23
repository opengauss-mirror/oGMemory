# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Structured JSON fallback for providers with unreliable tool calling."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtractionOperation(BaseModel):
    """One extraction operation encoded without a native tool call."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    input: dict[str, Any]


class ExtractionEnvelope(BaseModel):
    """Uniform envelope returned by the JSON fallback."""

    model_config = ConfigDict(extra="forbid")

    operations: list[ExtractionOperation]


def build_extraction_envelope_schema(tools: list[dict]) -> dict:
    """Build the response schema, constraining operation names when possible."""
    schema = ExtractionEnvelope.model_json_schema()
    operation_schema = schema.get("$defs", {}).get("ExtractionOperation", {})
    tool_schema = operation_schema.get("properties", {}).get("tool")
    tool_names = [tool.get("name", "") for tool in tools if tool.get("name")]
    if isinstance(tool_schema, dict) and tool_names:
        tool_schema["enum"] = tool_names
    return schema


def build_extraction_fallback_prompt(prompt: str, tools: list[dict]) -> str:
    """Ask for the same extraction as a strict JSON operation envelope."""
    tool_catalog = [
        {
            "tool": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {}),
        }
        for tool in tools
    ]
    return (
        f"{prompt}\n\n"
        "Native tool calling was unavailable. Perform the same extraction by "
        "returning one strict JSON object with this shape:\n"
        '{"operations":[{"tool":"extract_event","input":{...}}]}\n'
        "Use only a tool listed below and make each input satisfy that tool's "
        "schema. If there is no memory worth extracting, return "
        '{"operations":[]}. Do not return markdown or explanatory text.\n\n'
        "Available extraction operations:\n"
        f"{json.dumps(tool_catalog, ensure_ascii=False, separators=(',', ':'))}"
    )
