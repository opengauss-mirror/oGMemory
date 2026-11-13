"""Demand selection is strict, local, and cannot authorize hidden tool dispatch."""
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from extraction.prompt_assembly import assemble_prepared
from extraction.prompts import PromptManager
from extraction.tools import Extractor, _Span

CTX = RequestContext("acct", "user", "agent", "session", "trace")


class PublicManager(PromptManager):
    def render(self, template_name, section, *, categories=None, **variables):
        if section in {"system_prompt", "examples"}:
            return (
                "full rules" if categories is None else f"{section}:" + ",".join(sorted(categories))
            )
        return super().render(template_name, section, categories=categories, **variables)


def prepared_task(result, *, mode="eager"):
    llm = Mock()
    llm.complete_json.return_value = result
    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    extractor = Extractor(
        llm,
        mode=mode,
        prompt_manager=PublicManager(),
        fs=Mock(),
        schema_registry=registry,
        uri_resolver=URIResolver(registry),
    )
    prepared = extractor._prepare_span(
        _Span(0, 0),
        [{"role": "user", "content": "Project Beacon uses SQL."}],
        "English",
        None,
        "",
        "",
        CTX,
    )
    return extractor, prepared


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_valid_empty_stops_before_native_extraction(mode):
    extractor, prepared = prepared_task({"categories": []}, mode=mode)
    assert extractor._structure_prepared_span(prepared, CTX) == []
    extractor._llm.complete_with_tools.assert_not_called()
    extractor._llm.complete_with_tools_messages.assert_not_called()
    assert prepared.assembly_state.contract.scope == "no_task"
    assert prepared.assembly_state.contract.mode == mode
    assert extractor._use_lazy_mode() == (mode == "lazy")


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"categories": [], "shared_tools": ["read"]},
        {"categories": ["missing"]},
        {"categories": [], "extra": True},
        {"categories": ["entity", "entity"]},
    ],
)
def test_invalid_selection_retries_then_keeps_full(result):
    extractor, prepared = prepared_task(result)
    request = assemble_prepared(extractor, prepared, CTX)
    prompt, tools = request.prompt, request.tools
    assert prompt == prepared.prompt
    assert tools is extractor._extraction_tools
    assert prepared.assembly_state.contract.scope == "full"
    assert prepared.assembly_state.selection.raw_output == result
    assert extractor._llm.complete_json.call_count == 2


def test_legacy_manager_is_rejected_at_injection():
    manager = type("Legacy", (), {"render": lambda *args, **kwargs: "legacy"})()
    with pytest.raises(ValueError, match="load"):
        Extractor(Mock(), prompt_manager=manager)


def test_lazy_empty_does_not_read_or_call_native():
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.schemas.registry import SchemaRegistry
    llm, fs, registry = Mock(), Mock(), SchemaRegistry()
    llm.complete_json.return_value = {"categories": []}
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PublicManager())
    result = loop.run("No lasting facts.", CTX)
    assert result.completed and result.candidates == []
    llm.complete_with_tools_messages.assert_not_called()
    fs.read_node.assert_not_called()


def test_lazy_unoffered_query_is_rejected_before_dispatch():
    from core.errors import ToolCallProtocolError
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.schemas.registry import SchemaRegistry
    llm, fs, registry = Mock(), Mock(), SchemaRegistry()
    llm.complete_json.return_value = {"categories": ["entity"]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PublicManager())
    loop.run("Project Beacon uses SQL.", CTX)
    with pytest.raises(ToolCallProtocolError, match="outside"):
        loop._execute_tool("extract_profile", {"uri": "context://old"}, CTX)
    fs.read_node.assert_not_called()


def test_lazy_offers_default_queries_and_category_tools():
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.schemas.registry import SchemaRegistry
    llm, fs, registry = Mock(), Mock(), SchemaRegistry()
    llm.complete_json.return_value = {"categories": ["event"]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PublicManager())
    loop.run("I attended a workshop yesterday.", CTX)
    names = {tool["name"] for tool in llm.complete_with_tools_messages.call_args.kwargs["tools"]}
    assert names == {
        "read",
        "list",
        "get_relations",
        "get_access_stats",
        "extract_event",
        "edit_memory",
        "append_memory",
    }


@pytest.mark.parametrize("categories", [None, frozenset({"future"})], ids=["selected", "explicit"])
def test_direct_lazy_accepts_declared_type_without_exclusive_blocks(categories):
    from dataclasses import replace

    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prompts import PromptManager
    from extraction.schemas.registry import SchemaRegistry
    from extraction.tool_builder import build_update_instruction

    llm, fs, registry = Mock(), Mock(), SchemaRegistry()
    registry.register(replace(registry.get("event"), memory_type="future"))
    llm.complete_json.return_value = {"categories": ["future"]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    manager = PromptManager()
    manager.load("extraction")["selection"]["types"]["future"] = "Custom event type"
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), manager)
    source = "I attended a workshop yesterday."

    assert loop.run(source, CTX, categories=categories).completed

    assert llm.complete_json.call_count == (1 if categories is None else 0)
    llm.complete_with_tools_messages.assert_called_once()
    call = llm.complete_with_tools_messages.call_args.kwargs
    assert call["messages"][0]["content"] == "\n\n".join(
        [
            manager.render("extraction", "system_prompt", categories={"future"}),
            manager.render("extraction", "identity_block", user_id=CTX.user_id),
            manager.render("extraction", "lazy_tools_instruction"),
            manager.render("extraction", "examples", categories={"future"}),
            manager.render("extraction", "output_instruction"),
            build_update_instruction(registry, frozenset({"future"})),
        ]
    )
    assert {"role": "user", "content": f"## Conversation\n\n{source}"} in call["messages"]
    names = tuple(tool["name"] for tool in call["tools"])
    assert {"extract_future", "get_relations", "get_access_stats"} <= set(names)
    assert loop._assembly_contract.scope == "scoped"
    assert loop._assembly_contract.tool_names == names


def test_supported_preparation_renders_selected_components_without_full_body():
    class SelectedOnlyManager(PublicManager):
        def render(self, template_name, section, **variables):
            assert (
                section not in {"system_prompt", "examples"}
                or variables.get("categories") is not None
            ), "full body rendered before selection"
            return super().render(template_name, section, **variables)
    llm = Mock()
    llm.complete_json.return_value = {"categories": ["entity"]}
    extractor = Extractor(llm, prompt_manager=SelectedOnlyManager())
    prepared = extractor._prepare_span(_Span(0, 0), [{"role": "user", "content": "Project Beacon uses SQL."}], "English", None, "", "", CTX)
    request = assemble_prepared(extractor, prepared, CTX)
    assert "system_prompt:entity" in request.prompt
    assert {tool["name"] for tool in request.tools} == {"extract_entity", "edit_memory", "append_memory"}
    assert prepared.assembly_state.selection.categories == frozenset({"entity"})
    assert prepared.assembly_state.contract.scope == "scoped"
    assert prepared.assembly_state.attempts


def test_missing_registered_type_definition_is_configuration_error():
    from dataclasses import replace

    from extraction.schemas.registry import SchemaRegistry
    registry = SchemaRegistry()
    registry.register(replace(registry.get("event"), memory_type="future"))
    llm = Mock()
    extractor = Extractor(llm, schema_registry=registry)
    with pytest.raises(ValueError, match="missing definitions"):
        extractor._prepare_span(
            _Span(0, 0),
            [{"role": "user", "content": "Project uses SQL."}],
            "English",
            None,
            "",
            "",
            CTX,
        )
    llm.complete_json.assert_not_called()


def test_experiment_capture_records_selection_and_executed_assembly():
    from providers.llm_observation import capture_llm_events
    with capture_llm_events() as events:
        extractor, prepared = prepared_task({"categories": ["entity"]})
        assemble_prepared(extractor, prepared, CTX)
        # The ordinary extraction wrapper carries span identity for nested collectors.
        extractor._llm.complete_with_tools.return_value = []
        from core.errors import ToolCallProtocolError
        with pytest.raises(ToolCallProtocolError):
            extractor._structure_prepared_span(prepared, CTX)
    assembly = [event for event in events if event["event"] == "assembly"]
    assert assembly[-1]["selection"]["categories"] == ["entity"]
    assert assembly[-1]["decision"]["scope"] == "scoped"
    assert assembly[-1]["span_id"] == "0:0"


def test_source_only_attempt_does_not_overwrite_immutable_assembly_decision():
    from core.errors import LLMOutputTruncatedError
    from providers.llm_observation import capture_llm_events, correlate_extraction_calls

    extractor, prepared = prepared_task({"categories": ["entity"]})
    assemble_prepared(extractor, prepared, CTX)
    original_decision = prepared.assembly_state.contract
    extractor._llm.complete_json.return_value = {"operations": []}
    with capture_llm_events() as events:
        correlate_extraction_calls(extractor._structure_span_json_fallback)(prepared, CTX, None, errors=[LLMOutputTruncatedError("length")])
    assert prepared.assembly_state.decision is original_decision
    assert not prepared.assembly_state.decision.source_only
    assert "edit_memory" in prepared.assembly_state.decision.tool_names
    assert prepared.assembly_state.attempts[-1].source_only
    assert prepared.assembly_state.attempts[-1].tool_names == ("extract_entity",)
    assert prepared.assembly_state.attempts[-1].categories == frozenset({"entity"})
    assert prepared.span.prefetch_result is None
    schema = extractor._llm.complete_json.call_args.kwargs["schema"]
    assert schema["$defs"]["ExtractionOperation"]["properties"]["tool"]["enum"] == ["extract_entity"]
    event = [event for event in events if event["event"] == "assembly"][-1]
    assert not event["decision"]["source_only"] and event["attempt"]["source_only"]
