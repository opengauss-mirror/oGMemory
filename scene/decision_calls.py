"""Bounded, schema-validated model calls shared by scene decision stages."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from core.errors import ToolCallProtocolError
from core.interfaces import LLM
from scene.decision_contracts import DecisionLimits, SceneDecisionError, SceneModelOutputError

_Output = TypeVar("_Output", bound=BaseModel)


@dataclass
class CallBudget:
    """One batch's shared model-call allowance, including correction attempts."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> None:
        if self.remaining <= 0:
            raise SceneDecisionError("scene model call budget exhausted")
        self.used += 1


class SceneModelCaller:
    """Validate structured output with at most one budgeted correction."""

    def __init__(self, llm: LLM, limits: DecisionLimits | None = None) -> None:
        self.llm = llm
        self.limits = limits or DecisionLimits()
        self.limits.validate()

    def budgeted_prompt(self, prefix: str, payload: dict[str, Any], *, reserved_chars: int = 0) -> str:
        """Budget evidence narratives and candidate hints with complete source references."""
        capacity = self.limits.max_prompt_chars - reserved_chars
        prompt = prefix + self.dump(payload)
        if len(prompt) <= capacity:
            return prompt
        value = deepcopy(payload)
        packet = value.get("evidence", value)
        cards = [*packet.get("memories", []), *packet.get("provenance_context", []),
                 *packet.get("existing_scenes", []),
                 *(card for result in packet.get("tool_evidence", []) for card in result.get("results", []))]
        texts = [(card, key, card[key]) for card in cards if isinstance(card, dict)
                 for key in ("text", "description") if isinstance(card.get(key), str)]
        cap = self.limits.max_node_chars
        floor = min(64, cap)
        while True:
            for card, key, original in texts:
                if len(original) > cap:
                    card[key] = original[:cap]
                    card[f"{key}_truncated"] = True
                    card[f"original_{key}_chars"] = len(original)
            prompt = prefix + self.dump(value)
            if len(prompt) <= capacity or cap == floor:
                break
            cap = max(floor, cap // 2)
        omitted = 0
        while len(prompt) > capacity and packet.get("candidate_groups"):
            packet["candidate_groups"].pop()
            omitted += 1
            packet["omitted_candidate_groups"] = omitted
            prompt = prefix + self.dump(value)
        if len(prompt) > capacity:
            for key in ("memories", "provenance_context"):
                if packet.get(key):
                    packet[key] = self._card_table(packet[key])
            packet["card_table_format"] = (
                "Each memories/provenance_context table has columns and rows. "
                "Each row [i,values] maps columns[i] to values for one complete card, in original order. "
                "All URI and source strings are literal; output full URIs."
            )
            compact = prefix + self.dump(value)
            if len(compact) < len(prompt):
                prompt = compact
            if len(prompt) > capacity:
                shared = prefix + self.dump(self._reference_catalog(value))
                if len(shared) < len(prompt):
                    return shared
        return prompt

    @property
    def correction_reserve(self) -> int:
        return min(2_048, self.limits.max_prompt_chars // 8)

    @staticmethod
    def _card_table(cards: list[dict[str, Any]]) -> dict[str, Any]:
        columns, rows, shapes = [], [], {}
        for card in cards:
            keys = tuple(sorted(card))
            if keys not in shapes:
                shapes[keys] = len(columns)
                columns.append(keys)
            rows.append([shapes[keys], [card[key] for key in keys]])
        return {"columns": columns, "rows": rows}

    @staticmethod
    def _reference_catalog(value: dict[str, Any]) -> dict[str, Any]:
        if "reference_values" in value or "reference_format" in value:
            return value
        counts: Counter[str] = Counter()

        def collect(item):
            if isinstance(item, dict):
                return set(item) not in ({"$scene_ref"}, {"$scene_refs"}) and all(collect(child) for child in item.values())
            if isinstance(item, (list, tuple)):
                return all(collect(child) for child in item)
            if isinstance(item, str) and len(item) >= 80 and item.startswith(("ctx://", "prov:")):
                counts[item] += 1
            return True

        if not collect(value):
            return value
        references = [item for item, count in counts.items() if count > 1]
        if not references:
            return value
        indices = {item: index for index, item in enumerate(references)}

        def encode(item):
            if isinstance(item, str) and item in indices:
                return {"$scene_ref": indices[item]}
            if isinstance(item, dict):
                return {key: encode(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                if len(item) > 1 and all(isinstance(child, str) and child in indices for child in item):
                    return {"$scene_refs": [indices[child] for child in item]}
                return [encode(child) for child in item]
            return item

        encoded = encode(value)
        encoded["reference_values"] = references
        encoded["reference_format"] = (
            "Each object with only $scene_ref:i represents reference_values[i]; $scene_refs:[i,...] represents "
            "the ordered list of those complete strings, including duplicates. Indices are zero-based. "
            "Resolve references before reading the proposal, cards or sources; output full literal URIs and source refs."
        )
        return encoded

    def call(
        self, stage: str, prompt: str, model: type[_Output], budget: CallBudget,
        *, request: Callable[[], Any] | None = None, reserved_calls: int = 0,
        validator: Callable[[_Output], None] | None = None,
    ) -> _Output:
        schema = model.model_json_schema()
        current_prompt = prompt
        for attempt in range(2):
            if len(current_prompt) > self.limits.max_prompt_chars:
                raise SceneDecisionError(f"scene {stage} prompt exceeds capacity")
            budget.consume()
            try:
                output = (request() if request is not None and attempt == 0
                          else self.llm.complete_json(current_prompt, schema=schema))
                if isinstance(output, str):
                    output = self._parse_json(output, schema)
                validated = model.model_validate(output)
                if validator is not None:
                    validator(validated)
                return validated
            except (ValidationError, ValueError, TypeError, ToolCallProtocolError) as exc:
                failure = exc if isinstance(exc, SceneModelOutputError) else self._output_error(stage, exc, schema)
                correction = "\nOUTPUT CORRECTION:\n" + str(failure) + (
                    "\nReturn a complete JSON object matching the schema using the supplied evidence."
                )
                if (attempt or budget.remaining <= reserved_calls
                        or len(prompt) + len(correction) > self.limits.max_prompt_chars):
                    raise failure from None
                current_prompt = prompt + correction
        raise AssertionError("bounded scene validation loop completed")

    @staticmethod
    def _output_error(stage: str, exc: Exception, schema: dict[str, Any]) -> SceneModelOutputError:
        fields = set(schema.get("properties", {}))
        for definition in schema.get("$defs", {}).values():
            fields.update(definition.get("properties", {}))
        if isinstance(exc, ValidationError):
            errors = [{
                "field": ".".join(str(part) if isinstance(part, int) or part in fields else "[field]"
                                  for part in error["loc"]) or "$",
                "type": error["type"],
            } for error in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
        else:
            errors = [{"field": "$", "type": "json_invalid"}]
        return SceneModelOutputError(stage, errors)

    @staticmethod
    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _parse_json(content: str, schema: dict[str, Any]) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object for {schema.get('title', 'scene output')}")
        return value
