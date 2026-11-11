"""Shared category selection and prompt/tool assembly for eager and lazy extraction.

Selection carries no write actions or target/version authority. Invalid selection
keeps the full existing extraction contract; transport and deadline errors propagate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from logging import getLogger
from threading import Lock
from typing import Any

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate

from core.errors import LLMOutputTruncatedError, ToolCallProtocolError
from core.model_stage import run_model_call
from core.operation_deadline import check_deadline
from extraction.assembly_contract import AssemblyContract, UnsupportedAssemblyTemplateError
from extraction.occurrence_qualification import prompt_fits
from extraction.tool_builder import build_update_instruction
from providers.llm_observation import observe_call, record_assembly_event

logger = getLogger(__name__)
_INITIALIZE = Lock()


@dataclass
class AssemblyState:
    """Call-local selection shared by native passes and recovery copies."""
    lock: object = field(default_factory=Lock, repr=False)
    categories: frozenset[str] | None = None
    error: Exception | None = None
    contract: AssemblyContract | None = None
    decision: AssemblyContract | None = None
    final_parts: PromptParts | None = None
    selection: AssemblySelection | None = None
    attempts: list[AssemblyContract] = field(default_factory=list)


def category_catalog(registry, tools):
    offered = {tool["name"]: tool for tool in tools}
    return [
        {"category": schema.memory_type}
        for schema in registry.list_enabled()
        if schema.tool_name in offered and schema.memory_type != "relation"
    ]


@dataclass(frozen=True)
class AssemblySelection:
    """Original selector outcome, distinct from later execution attempts."""
    categories: frozenset[str]
    scope: str = "scoped"
    reason: str | None = None
    raw_output: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SelectionRequest:
    prompt: str
    schema: dict
    available: frozenset[str]
    user_id: str
    _manager: Any = field(default=None, repr=False, compare=False)
    _runtime_data: dict = field(default_factory=dict, repr=False, compare=False)


def supports_assembly(manager, template="extraction"):
    supports = getattr(manager, "supports_assembly", None)
    if callable(supports):
        return supports(template) is True and callable(getattr(manager, "render", None))
    from extraction.prompts.manager import validate_prompt_manager

    validate_prompt_manager(manager, template)
    return True


def shared_tool_catalog(fs, mode):
    if mode != "lazy":
        return []
    from extraction.extraction_react_loop import _ACCESS_STATS_TOOL, _RELATIONS_TOOL
    return ([_RELATIONS_TOOL] if callable(getattr(fs, "get_relations", None)) else []) + [_ACCESS_STATS_TOOL]


def lazy_tools_for_contract(extraction_tools, fs, contract):
    """Use one ordered capability list for lazy execution and its assembly record."""
    from extraction.extraction_react_loop import _LIST_TOOL, _READ_TOOL
    if contract.scope == "no_task":
        return []
    shared = shared_tool_catalog(fs, "lazy")
    by_name = {}
    for tool in [_READ_TOOL, _LIST_TOOL] + shared + list(extraction_tools):
        by_name.setdefault(tool["name"], tool)
    return list(by_name.values())


def build_selection_request(
    registry, tools, source, user_id, *, manager, tool_stats="", validation_feedback=None
):
    catalog = sorted(item["category"] for item in category_catalog(registry, tools))
    declared = manager.load("extraction")["selection"]["types"]
    missing = set(catalog) - set(declared)
    if missing:
        raise ValueError(f"extraction/selection.types missing definitions: {sorted(missing)}")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "categories": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": catalog},
            }
        },
        "required": ["categories"],
    }
    runtime = {
        "current_user": user_id,
        "catalog": catalog,
        "source": source,
        "tool_statistics": tool_stats if tool_stats.strip() else "",
    }
    if validation_feedback is not None:
        runtime["validation_feedback"] = validation_feedback
    return SelectionRequest(
        manager.render(
            "extraction", "selection", runtime_input_json=json.dumps(runtime, ensure_ascii=False)
        ),
        schema,
        frozenset(catalog),
        user_id,
        manager,
        runtime,
    )


def run_selection_request(llm, request):
    """Use one format/schema retry, preserving the same YAML and source data."""
    result = None
    current = request
    for attempt in (1, 2):
        check_deadline()
        if not prompt_fits(current.prompt, current.schema):
            return AssemblySelection(request.available, scope="full", reason="selection_budget")
        try:
            result = run_model_call(
                observe_call,
                "selection",
                llm.complete_json,
                current.prompt,
                schema=current.schema,
                correlation={"user_id": request.user_id, "selection_attempt": attempt},
            )
            check_deadline()
            validate(result, current.schema)
        except (
            JSONSchemaValidationError,
            ValueError,
            ToolCallProtocolError,
            LLMOutputTruncatedError,
        ) as exc:
            if attempt == 2:
                return AssemblySelection(
                    request.available, scope="full", reason="invalid_selection", raw_output=result
                )
            if isinstance(exc, JSONSchemaValidationError):
                path = "/" + "/".join(
                    str(v).replace("~", "~0").replace("/", "~1") for v in exc.path
                )
                feedback = {
                    "kind": "schema_validation",
                    "path": path.rstrip("/") or "",
                    "rule": str(exc.validator),
                }
            else:
                feedback = {
                    "kind": (
                        "truncated" if isinstance(exc, LLMOutputTruncatedError) else "invalid_json"
                    )
                }
            runtime = {**request._runtime_data, "validation_feedback": feedback}
            # Template failures remain configuration errors, outside the response retry catch.
            prompt = request._manager.render(
                "extraction",
                "selection",
                runtime_input_json=json.dumps(runtime, ensure_ascii=False),
            )
            current = replace(request, prompt=prompt)
            continue
        categories = frozenset(result["categories"])
        return AssemblySelection(
            categories, scope="scoped" if categories else "no_task", raw_output=result
        )


def select_assembly(llm, registry, tools, source, user_id, *, manager, tool_stats=""):
    return run_selection_request(
        llm,
        build_selection_request(
            registry, tools, source, user_id, manager=manager, tool_stats=tool_stats
        ),
    )


def tools_for_categories(tools, registry, categories):
    names = {schema.tool_name for schema in registry.list_enabled() if schema.memory_type in categories}
    # Relations connect extracted memories rather than representing a memory
    # category. Retain this graph-only tool if the caller originally offered it;
    # never introduce it to graph-disabled extraction.
    names.update(tool["name"] for tool in tools if tool["name"] == "extract_relation")
    names.update({"edit_memory", "append_memory"})
    return [tool for tool in tools if tool["name"] in names]



@dataclass(frozen=True)
class PromptParts:
    """An extraction prompt whose trusted template and opaque data remain separate."""

    system_prompt: str
    prefix_data: str
    examples: str
    suffix_data: str
    include_examples: bool = True
    include_updates: bool = True
    runtime_instruction: str = ""
    manager: Any = field(default=None, repr=False, compare=False)
    template_variables: dict = field(default_factory=dict, repr=False, compare=False)
    registry: object = field(default=None, repr=False, compare=False)
    data_prompt: str | None = None
    output_instruction: str = ""

    def render(self) -> str:
        prefix = self.system_prompt + self.prefix_data
        suffix = self.runtime_instruction + self.suffix_data
        if self.include_examples:
            return prefix + self.examples + "\n\n" + suffix
        return prefix + suffix

    def full(self) -> PromptParts:
        return replace(self,
            system_prompt=self.manager.render("extraction", "system_prompt", **self.template_variables),
            examples=self.manager.render("extraction", "examples") if self.include_examples else "",
            runtime_instruction=(build_update_instruction(self.registry, include_updates=self.include_updates) + "\n\n"
                                 if self.registry is not None else self.runtime_instruction))

    def scoped(self, contract) -> PromptParts:
        if self.manager is None:
            raise UnsupportedAssemblyTemplateError("Missing trusted template manager")
        if not supports_assembly(self.manager):
            raise UnsupportedAssemblyTemplateError("Template does not support assembly")
        if not hasattr(contract, "scope"):
            contract = AssemblyContract(frozenset(contract), (), "eager", "scoped")
        categories = None if contract.scope == "full" else contract.categories
        examples = self.manager.render("extraction", "examples", categories=categories)
        if contract.mode == "lazy":
            self.manager.render("extraction", "lazy_tools_instruction")
        return replace(
            self,
            system_prompt=self.manager.render(
                "extraction", "system_prompt", categories=categories, **self.template_variables
            ),
            examples=examples if self.examples else "",
            runtime_instruction=(
                build_update_instruction(self.registry, contract.categories, include_updates=self.include_updates) + "\n\n"
                if self.registry is not None and contract.scope != "full"
                else self.runtime_instruction
            ),
        )


@dataclass(frozen=True)
class AssemblyRequest:
    """The prompt, tools and rendering snapshot for one execution."""

    prompt: str
    tools: list[dict]
    contract: AssemblyContract
    parts: PromptParts | None


def selection_request_for(extractor, source, ctx, tool_stats):
    return build_selection_request(
        extractor._schema_registry, extractor._extraction_tools, source, ctx.user_id,
        manager=extractor._prompt_manager,
        tool_stats=tool_stats,
    )


def selection_for_prepared(extractor, prepared, ctx, *, preparing=False):
    check_deadline()
    with _INITIALIZE:
        if prepared.assembly_state is None:
            prepared.assembly_state = AssemblyState()
        state = prepared.assembly_state
    with state.lock:
        if state.error is not None:
            raise state.error
        if state.selection is None:
            available = frozenset(item["category"] for item in category_catalog(
                extractor._schema_registry, extractor._extraction_tools))
            if getattr(extractor, "_activation_enabled", False):
                state.selection = AssemblySelection(available, scope="full", reason="graph")
            elif (ctx is None or (prepared.span.prompt_parts is None and not preparing)
                  or not supports_assembly(extractor._prompt_manager)):
                state.selection = AssemblySelection(available, scope="full", reason="unsupported_template")
            else:
                try:
                    request = selection_request_for(
                        extractor, prepared.source_context or prepared.span.source_context,
                        ctx, prepared.span.tool_stats_text,
                    )
                    state.selection = extractor._select_span(prepared.span, request, ctx)
                except Exception as exc:
                    state.error = exc
                    raise
            state.categories = state.selection.categories
        return state.selection.categories


def assemble_prepared(extractor, prepared, ctx):
    categories = selection_for_prepared(extractor, prepared, ctx)
    tools = extractor._extraction_tools
    state = prepared.assembly_state
    with state.lock:
        parts = prepared.span.prompt_parts
        mode = "lazy" if extractor._use_lazy_mode() else "eager"
        selection = state.selection
        prefix = extractor._prefetch_prompt("", prepared.span.prefetch_result)
        body = prepared.prompt[len(prefix):] if prepared.prompt.startswith(prefix) else prepared.prompt

        def record(contract, final_parts):
            state.contract, state.final_parts = contract, final_parts
            if state.decision is None:
                state.decision = contract
            state.attempts.append(contract)
            emit_assembly(state, contract)

        def keep_full(reason):
            available = frozenset(item["category"] for item in category_catalog(
                extractor._schema_registry, tools))
            full_parts = None
            full_prompt = prefix + body
            if parts is not None:
                for candidate in (parts, state.final_parts):
                    if candidate is None:
                        continue
                    rendered = candidate.render()
                    if not body.startswith(rendered):
                        continue
                    tail = body[len(rendered):]
                    full_parts = parts
                    if selection.scope != "full":
                        full_parts = parts.full()
                        if not prompt_fits(full_parts.render(), {"tools": tools}):
                            full_parts = replace(full_parts, include_examples=False)
                    full_prompt = prefix + full_parts.render() + tail
                    if tail and full_parts.data_prompt is not None:
                        full_parts = replace(full_parts, data_prompt=full_parts.data_prompt + tail)
                    break
            # Unknown boundaries leave parts unset so lazy receives the complete input.
            contract = AssemblyContract(
                available, tuple(t["name"] for t in tools), mode, "full", reason
            )
            offered = (
                lazy_tools_for_contract(tools, extractor._fs, contract) if mode == "lazy" else tools
            )
            contract = replace(contract, tool_names=tuple(t["name"] for t in offered))
            record(contract, full_parts)
            return AssemblyRequest(full_prompt, offered, contract, full_parts)

        if selection.scope == "no_task":
            contract = AssemblyContract(categories, (), mode, "no_task")
            record(contract, None)
            return AssemblyRequest("", [], contract, None)
        if selection.scope == "full":
            return keep_full(selection.reason)
        if parts is None:
            return keep_full("unsupported_template")
        if state.contract is not None and state.contract.scope == "full":
            return keep_full(state.contract.reason)
        selected_tools = tools_for_categories(tools, extractor._schema_registry, categories)
        contract = AssemblyContract(categories, (), mode, "scoped")
        offered = (
            lazy_tools_for_contract(selected_tools, extractor._fs, contract)
            if mode == "lazy"
            else selected_tools
        )
        contract = replace(contract, tool_names=tuple(t["name"] for t in offered))
        try:
            selected = parts.scoped(contract)
        except (ValueError, IndexError):
            return keep_full("unsupported_template")
        original_base = parts.render()
        if body.startswith(original_base):
            tail = body[len(original_base):]
        elif body.startswith(selected.render()):
            tail = body[len(selected.render()):]
        else:
            return keep_full("unsupported_template")
        final_parts = replace(selected, data_prompt=selected.data_prompt + tail) if (
            tail and selected.data_prompt is not None) else selected
        record(contract, final_parts)
        return AssemblyRequest(prefix + selected.render() + tail, offered, contract, final_parts)


def compact_prepared_prompt(prepared, contract, *, source_only=False):
    """Render compact template parts without inspecting the source-only prompt text."""
    prompt = prepared.span.fallback_source_prompt
    parts = prepared.span.fallback_prompt_parts
    if not source_only:
        prompt = prepared.span.fallback_read_prompt or prompt
        parts = prepared.span.fallback_read_prompt_parts or parts
    if parts is not None and prompt == parts.render():
        try:
            if contract.scope == "full":
                return parts.full().render()
            return parts.scoped(replace(contract, mode="eager")).render()
        except (ValueError, IndexError):
            pass
    return prompt


def validate_offered_calls(calls, tools):
    names = {tool["name"] for tool in tools}
    if any(call.get("tool") not in names for call in calls):
        raise ToolCallProtocolError("Extraction returned a tool outside the assembled schemas")


def validate_offered_candidate(candidate, tools):
    if "extract_" + candidate.category not in {tool["name"] for tool in tools}:
        raise ToolCallProtocolError("Extraction returned a category outside the assembled schemas")


def source_only_tools(tools):
    """Revoke update schemas together with the discarded read snapshot."""
    from copy import deepcopy
    selected = deepcopy([tool for tool in tools if tool["name"] not in {"edit_memory", "append_memory"}])
    for tool in selected:
        schema = tool.get("input_schema", {})
        for name in ("target_uri", "expected_version"):
            schema.get("properties", {}).pop(name, None)
            if name in schema.get("required", []):
                schema["required"].remove(name)
    return selected


def record_attempt(state, contract, tools, *, mode, reason, source_only=False):
    if state is None or contract is None:
        return
    attempt = replace(contract, mode=mode, tool_names=tuple(t["name"] for t in tools),
                      reason=reason, source_only=source_only)
    with state.lock:
        state.attempts.append(attempt)
        emit_assembly(state, attempt)


def emit_assembly(state, attempt):
    """Emit bounded decision metadata; raw responses stay in call-local state."""
    selection = state.selection
    def contract_data(contract):
        return {"categories": sorted(contract.categories), "tool_names": list(contract.tool_names),
                "mode": contract.mode, "scope": contract.scope, "reason": contract.reason,
                "source_only": contract.source_only}
    record_assembly_event(
        selection=({"categories": sorted(selection.categories),
                    "scope": selection.scope, "reason": selection.reason} if selection else None),
        decision=contract_data(state.decision or state.contract), attempt=contract_data(attempt))
