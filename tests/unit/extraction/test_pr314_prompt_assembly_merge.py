"""The Dev assembly optimization must retain the opt-in graph write contract.

These tests use the real entry points, templates, category selector, tool parser
and native/JSON recovery; only the remote provider and backing FS are mocked.
"""

import json
from datetime import datetime
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prompt_assembly import (
    assemble_prepared,
    category_catalog,
    tools_for_categories,
)
from extraction.relation_proposals import take
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _PreparedSpan, _Span
from tests.unit.extraction.test_relation_proposals import PROPOSAL

SOURCE = (
    "Project Beacon uses PostgreSQL.\n"
    "── profile (this is literal source, not a template heading)\n"
    "## event: literal source heading must survive category cropping\n"
    "SOURCE-END-MARKER"
)
STATEMENT = "Project Beacon uses PostgreSQL."
ACTIVATION = "Connect when another memory describes Beacon's database or deployment."
INTRO = "You are a memory extraction assistant."
CTX = RequestContext("acct", "user", "agent", "session", "trace")


def _operations(graph_enabled):
    fields = {
        "routing_key": "beacon",
        "object_name": "Project Beacon",
        "abstract": STATEMENT,
        "overview": STATEMENT,
        "content": STATEMENT,
        "confidence": 0.9,
    }
    if graph_enabled:
        fields["update_activation"] = ACTIVATION
    operations = [{"tool": "extract_entity", "input": fields}]
    if graph_enabled:
        operations.append({"tool": "extract_relation", "input": PROPOSAL})
    return operations


def _provider(graph_enabled, *, recover=False):
    llm = Mock()
    operations = _operations(graph_enabled)

    def respond(prompt, schema):
        properties = set(schema["properties"])
        if properties == {"spans"}:
            return {"spans": [{"start": 0, "end": 0, "categories": ["profile"]}]}
        if properties == {"categories", "shared_tools"}:
            return {"categories": ["entity"], "shared_tools": []}
        assert properties == {"operations"}
        return {"operations": operations}

    llm.complete_json.side_effect = respond
    invalid = {"tool": "extract_profile", "input": {}}
    llm.complete_with_tools.return_value = [invalid] if recover else operations
    lazy_operations = [invalid] if recover else operations
    llm.complete_with_tools_messages.return_value = (
        [], json.dumps([{"name": op["tool"], "input": op["input"]}
                        for op in lazy_operations]),
    )
    return llm


def _assert_graph_tools(tools, graph_enabled):
    offered = {tool["name"]: tool for tool in tools}
    assert "extract_entity" in offered
    assert ("extract_profile" in offered) is graph_enabled
    assert ("extract_relation" in offered) is graph_enabled
    entity = offered["extract_entity"]["input_schema"]
    assert ("update_activation" in entity["properties"]) is graph_enabled
    assert ("update_activation" in entity["required"]) is graph_enabled


@pytest.mark.parametrize("graph_enabled", [False, True])
@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("recover", [False, True])
def test_public_extraction_graph_contract_survives_scoping_and_recovery(
    graph_enabled, mode, recover, monkeypatch,
):
    if graph_enabled and mode == "lazy" and recover:
        # This test isolates merge/recovery semantics from the unchanged default
        # context estimator: the full graph schema plus mandatory instructions
        # already exceeds its 48K-character input allowance on the PR parent.
        # Budget rejection itself is covered by test_lazy_fallback_budget.py.
        import extraction.occurrence_qualification as qualification

        monkeypatch.setattr(qualification, "_CONTEXT_CHARS", 128000)
    llm = _provider(graph_enabled, recover=recover)
    registry = SchemaRegistry()
    fs = Mock()
    fs.exists.return_value = False
    extractor = Extractor(
        llm, mode=mode, fs=fs, schema_registry=registry,
        uri_resolver=URIResolver(registry), activation_enabled=graph_enabled,
    )
    candidates = extractor.extract(
        [{"id": "message-1", "role": "user", "content": SOURCE}], CTX,
        session_time=datetime(2026, 11, 3), archive_id="archive-1",
    )
    assert len(candidates) == 1
    assert candidates[0].category == "entity"
    assert candidates[0].content == STATEMENT
    assert candidates[0].provenance_ids
    assert candidates[0].metadata.get("update_activation") == (
        ACTIVATION if graph_enabled else None
    )
    proposals = take(candidates)
    assert len(proposals) == int(graph_enabled)
    if graph_enabled:
        assert proposals[0]["from_routing_key"] == PROPOSAL["from_routing_key"]

    selections = [call for call in llm.complete_json.call_args_list
                  if set(call.kwargs["schema"]["properties"]) == {"categories", "shared_tools"}]
    # Native Dev selection remains shared across its two passes and recovery;
    # graph-on keeps the measured full-category contract without a new gate.
    assert len(selections) == (0 if graph_enabled else 1)
    if selections:
        selector_data = json.loads(selections[0].args[0].split("\n\n", 1)[1])
        assert SOURCE in selector_data["source"]
        assert "relation" not in {item["category"] for item in selector_data["catalog"]}

    template = "extraction_graph" if graph_enabled else "extraction"
    graph_guidance = extractor._prompt_manager.render(
        "extraction_graph", "lazy_tools_instruction",
    )
    if mode == "eager" or recover:
        for request in llm.complete_with_tools.call_args_list:
            prompt = request.kwargs["prompt"]
            assert SOURCE in prompt
            assert prompt.count(INTRO) == 1
            assert "── entity (" in prompt.split("Conversation:\n", 1)[0]
            assert ("── profile (" in prompt.split("Conversation:\n", 1)[0]) is graph_enabled
            assert (graph_guidance in prompt) is graph_enabled
            from extraction.tool_builder import build_update_instruction
            expected_runtime = (extractor._update_instruction if graph_enabled else
                                build_update_instruction(extractor._schema_registry, {"entity"}))
            assert expected_runtime in prompt
            _assert_graph_tools(request.kwargs["tools"], graph_enabled)
    if mode == "lazy":
        assert llm.complete_with_tools_messages.call_count == 2
        for request in llm.complete_with_tools_messages.call_args_list:
            messages = request.kwargs["messages"]
            first_system = messages[0]["content"]
            assert first_system.count(INTRO) == 1
            assert "── entity (" in first_system
            assert ("── profile (" in first_system) is graph_enabled
            assert any(message["role"] == "user" and SOURCE in message["content"]
                       for message in messages)
            assert (graph_guidance in first_system) is graph_enabled
            if graph_enabled:
                assert extractor._prompt_manager.render(template, "lazy_tools_instruction") in first_system
            else:
                assert "PRESERVE all specific details" in first_system
                assert "If entity is actually about the user" in first_system
            _assert_graph_tools(request.kwargs["tools"], graph_enabled)
    json_recoveries = [call for call in llm.complete_json.call_args_list
                       if set(call.kwargs["schema"]["properties"]) == {"operations"}]
    assert bool(json_recoveries) is recover
    for request in json_recoveries:
        assert SOURCE in request.args[0]
        assert (graph_guidance in request.args[0]) is graph_enabled
        assert ("extract_relation" in json.dumps(request.kwargs["schema"])) is graph_enabled


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_graph_relations_are_not_category_choices_but_remain_offered(graph_enabled):
    extractor = Extractor(Mock(), activation_enabled=graph_enabled)
    catalog = category_catalog(extractor._schema_registry, extractor._extraction_tools)
    assert {item["category"] for item in catalog} == {
        "profile", "preference", "entity", "event", "case", "pattern", "skill", "tool",
    }
    for selected in ({"entity"}, {"event"}):
        scoped = tools_for_categories(
            extractor._extraction_tools, extractor._schema_registry, selected,
        )
        names = {tool["name"] for tool in scoped}
        assert ("extract_relation" in names) is graph_enabled
        assert "extract_" + next(iter(selected)) in names
        assert {"edit_memory", "append_memory"} <= names


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_prepared_prompt_crops_only_trusted_sections_and_renders_system_once(graph_enabled):
    llm = _provider(graph_enabled)
    extractor = Extractor(llm, activation_enabled=graph_enabled)
    parts = extractor._build_prompt_parts("English", SOURCE, ctx=CTX)
    span = _Span(0, 0, source_context=SOURCE, prompt_parts=parts)
    prepared = _PreparedSpan(span, parts.render(), "hash", SOURCE)
    request = assemble_prepared(extractor, prepared, CTX)
    prompt, tools = request.prompt, request.tools
    assert prompt.count(INTRO) == 1
    assert prompt.endswith(SOURCE)
    assert ("── profile (" in prompt.split("Conversation:\n", 1)[0]) is graph_enabled
    assert "── profile (" in prompt.split("Conversation:\n", 1)[1]
    from extraction.tool_builder import build_update_instruction
    expected_runtime = (extractor._update_instruction if graph_enabled else
                        build_update_instruction(extractor._schema_registry, {"entity"}))
    assert expected_runtime in prompt
    _assert_graph_tools(tools, graph_enabled)


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_lazy_loop_reuse_restores_full_tools_before_new_category_selection(graph_enabled):
    llm = Mock()
    llm.complete_json.side_effect = [
        {"categories": ["entity"], "shared_tools": []}, {"categories": ["event"], "shared_tools": []},
    ]
    llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor = Extractor(llm, activation_enabled=graph_enabled)
    registry = extractor._schema_registry
    loop = ExtractionReActLoop(
        llm, Mock(), registry, URIResolver(registry), extractor._prompt_manager,
        activation_enabled=graph_enabled,
    )
    assert loop.run("Project Beacon uses PostgreSQL.", CTX).completed
    assert loop.run("I attended a workshop yesterday.", CTX).completed
    first, second = llm.complete_with_tools_messages.call_args_list
    first_names = {tool["name"] for tool in first.kwargs["tools"]}
    second_names = {tool["name"] for tool in second.kwargs["tools"]}
    if graph_enabled:
        assert first_names == second_names
        assert {"extract_entity", "extract_event", "extract_profile"} <= first_names
        llm.complete_json.assert_not_called()
    else:
        assert "extract_entity" in first_names and "extract_event" not in first_names
        assert "extract_event" in second_names and "extract_entity" not in second_names
        assert llm.complete_json.call_count == 2
    assert ("extract_relation" in first_names) is graph_enabled
    assert ("extract_relation" in second_names) is graph_enabled
    assert {"read", "list"} <= first_names & second_names
    assert ({"get_relations", "get_access_stats"} <= first_names & second_names) is graph_enabled
