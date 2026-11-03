"""Scene preflight requests, scoped evidence tools and deterministic evidence floor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from scene.config import SceneConfig
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import PreflightOutput, SceneEvidenceReader, SceneModelOutputError
from scene.decision_prompts import PREFLIGHT_SYSTEM
from scene.decision_sources import independent_sources, source_refs


class ScenePreflight:
    """Acquire bounded evidence before decision and quality review."""

    def __init__(
        self, caller: SceneModelCaller, config: SceneConfig | None = None,
        *, evidence_reader: SceneEvidenceReader | None = None,
    ) -> None:
        self.caller = caller
        self.config = config or SceneConfig()
        self.config.validate()
        self.evidence_reader = evidence_reader

    def prompt(self, packet: dict[str, Any]) -> tuple[str, str, str]:
        """Use the same bounded evidence for text and native tool requests."""
        schema = PreflightOutput.model_json_schema()
        system = PREFLIGHT_SYSTEM + "\nOUTPUT SCHEMA:\n" + self.caller.dump(schema)
        if self.evidence_reader is None:
            system += "\nEvidence mode: packet-only. Base readiness on the packet; choose WAIT for unresolved gaps."
        prefix = system + "\nPACKET:\n"
        prompt = self.caller.budgeted_prompt(prefix, packet, reserved_chars=self.caller.correction_reserve)
        return system, prompt, prompt[len(prefix):]

    def run(self, packet: dict[str, Any], budget: CallBudget) -> PreflightOutput:
        """Attach tool evidence; apply enforce_gate after canonical evidence expansion."""
        schema = PreflightOutput.model_json_schema()
        tools = self._tool_schemas() if self.evidence_reader is not None else []
        system, prompt, wire_packet = self.prompt(packet)
        complete_with_tools = getattr(self.caller.llm, "complete_with_tools_messages", None)
        tool_calls = []

        def request():
            nonlocal tool_calls
            if self.evidence_reader is None or not callable(complete_with_tools):
                return self.caller.llm.complete_json(prompt, schema=schema)
            try:
                result = complete_with_tools(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": wire_packet},
                    ], tools=tools, tool_choice="auto",
                )
            except (AttributeError, NotImplementedError):
                return self.caller.llm.complete_json(prompt, schema=schema)
            if isinstance(result, tuple) and len(result) == 2:
                calls, content = result
            elif isinstance(result, list):
                calls, content = result, ""
            elif isinstance(result, str):
                calls, content = [], result
            else:
                return result
            if calls:
                tool_calls = self._validated_calls(calls, tools)
                return PreflightOutput(status="READY", reason="requested evidence acquired")
            return content

        proposed = self.caller.call("preflight", prompt, PreflightOutput, budget,
                                    request=request, reserved_calls=1)
        if tool_calls and self.evidence_reader is not None:
            packet["tool_evidence"] = self._execute_tools(packet, tool_calls)
        return proposed

    def enforce_gate(self, packet: dict[str, Any], proposed: PreflightOutput) -> PreflightOutput:
        """Apply the configured evidence floor to the model's readiness decision."""
        floor = self._floor(packet)
        return floor if floor.status == "WAIT" else proposed

    def _floor(self, packet: dict[str, Any]) -> PreflightOutput:
        cards = packet["memories"]
        evidence_cards = [*cards, *(item for result in packet.get("tool_evidence", [])
                                   for item in result.get("results", []) if isinstance(item, dict))]
        source_refs, resolved_memories = set(), set()
        for card in evidence_cards:
            refs = self._source_refs(card)
            source_refs.update(refs)
            uri = card.get("uri")
            if isinstance(uri, str) and uri:
                identity = self.independent_sources([uri])
                if self.independent_sources(refs) - identity:
                    resolved_memories.update(key for key in identity if key[0] == "memory")
        # Count the known roots of provenance chains, including multi-hop evidence.
        sources = self.independent_sources(source_refs) - resolved_memories
        offered = packet["offered_member_uris"]
        if (len(cards) < self.config.min_evidence_nodes
                or len(sources) < self.config.min_independent_sources):
            return PreflightOutput(
                status="WAIT", evidence_uris=offered,
                missing=["evidence_nodes", "independent_sources"],
                reason="evidence coverage below configured gate",
            )
        return PreflightOutput(status="READY", evidence_uris=offered)

    _source_refs = staticmethod(source_refs)
    independent_sources = staticmethod(independent_sources)

    @staticmethod
    def _tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": "search_related_nodes",
                "description": "Search authorized related memory nodes.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
            },
            {
                "name": "search_scene_cards",
                "description": "Search authorized existing scene summaries.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
            },
            {
                "name": "read_source_context",
                "description": "Read bounded source evidence for supplied source refs.",
                "input_schema": {"type": "object", "properties": {"source_refs": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer"}}, "required": ["source_refs"]},
            },
        ]

    def _validated_calls(self, calls: Any, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(calls, list):
            raise SceneModelOutputError("preflight", [{"field": "tool_calls", "type": "array_required"}])
        schemas = {tool["name"]: tool["input_schema"] for tool in tools}
        validated = []
        for call in calls[:self.config.max_react_rounds]:
            name = (call.get("tool") or call.get("name")) if isinstance(call, dict) else None
            if not isinstance(name, str) or name not in schemas:
                raise SceneModelOutputError("preflight", [{"field": "tool_calls", "type": "tool_required"}])
            args = call.get("input")
            schema = schemas[name]
            try:
                Draft202012Validator(schema).validate(args)
            except ValidationError as exc:
                path = [str(part) if isinstance(part, int) or part in schema["properties"] else "[field]"
                        for part in exc.path]
                raise SceneModelOutputError("preflight", [{
                    "field": ".".join([name, "input", *path]), "type": exc.validator,
                }]) from None
            validated.append({"tool": name, "input": args})
        return validated

    def _execute_tools(self, packet: dict[str, Any], calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        reader = self.evidence_reader
        assert reader is not None
        output: list[dict[str, Any]] = []
        for call in calls:
            name, args = call["tool"], call["input"]
            limit = min(max(int(args.get("limit", 8)), 1), 16)
            if name == "search_related_nodes":
                value = reader.search_related_nodes(packet["account_id"], packet["owner_space"], args["query"][:500], limit)
            elif name == "search_scene_cards":
                value = reader.search_scene_cards(packet["account_id"], packet["owner_space"], args["query"][:500], limit)
            elif name == "read_source_context":
                refs = args["source_refs"][:16]
                value = reader.read_source_context(packet["account_id"], packet["owner_space"], refs, limit)
            else:
                continue
            output.append({"tool": name, "results": list(value)[:limit]})
        return output
