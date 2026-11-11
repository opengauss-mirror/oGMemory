"""Shared assembly narrows prompts and tools, never precommits write operations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.errors import LLMCallError, LLMOutputTruncatedError, ToolCallProtocolError
from core.models import RequestContext
from extraction.prefetch import PrefetchResult
from extraction.prompt_assembly import assemble_prepared
from extraction.tools import Extractor, _PreparedSpan, _Span
from providers.llm.mock_llm import MockLLM


@pytest.fixture
def task():
    extractor = Extractor(MockLLM())
    extractor._llm.complete_json = Mock(return_value={"categories": ["entity"]})
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    source = "user: Project Beacon uses PostgreSQL."
    reads = PrefetchResult(messages=["EXISTING MEMORY SNAPSHOT"], read_versions={"old-uri": 4})
    span = _Span(0, 0, source_context=source, prefetch_result=reads)
    span.prompt_parts = extractor._build_prompt_parts("English", source, ctx=ctx)
    prompt = extractor._prefetch_prompt(span.prompt_parts.render(), reads)
    return extractor, _PreparedSpan(span, prompt, "sha", source), ctx


def test_selection_contract_contains_only_categories(task):
    extractor, prepared, ctx = task
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    request = extractor._llm.complete_json.call_args
    schema = request.kwargs["schema"]
    assert set(schema["properties"]) == {"categories"}
    assert schema["additionalProperties"] is False
    assert "Project Beacon uses PostgreSQL" in request.args[0]
    assert "EXISTING MEMORY SNAPSHOT" not in request.args[0]
    assert "── entity (" in prompt
    assert "── profile (" not in prompt
    assert "── event (" not in prompt
    assert {t["name"] for t in tools} == {"extract_entity", "edit_memory", "append_memory"}


def test_assembly_preserves_source_and_all_existing_read_authority(task):
    extractor, prepared, ctx = task
    original_reads = prepared.span.prefetch_result
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, _ = request.prompt, request.tools
    assert prepared.source_context in prompt
    assert "EXISTING MEMORY SNAPSHOT" in prompt
    assert prepared.span.prefetch_result is original_reads
    assert prepared.span.prefetch_result.read_versions == {"old-uri": 4}
    assert "Extraction plan" not in prompt


def test_shared_selection_survives_recovery_copies_and_concurrent_passes(task):
    extractor, prepared, ctx = task
    # Initialize shared state on the original before dataclasses.replace in recovery.
    assemble_prepared(extractor, prepared, ctx)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: assemble_prepared(extractor, item, ctx),
                                [prepared, replace(prepared)]))
    assert results[0] == results[1]
    extractor._llm.complete_json.assert_called_once()


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"categories": ["missing"]},
        {"categories": ["entity"], "target_uri": "invented"},
        {"operations": []},
    ],
)
def test_invalid_selection_keeps_all_available_tools_instead_of_skipping(task, result):
    extractor, prepared, ctx = task
    extractor._llm.complete_json.return_value = result
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    assert prompt == prepared.prompt
    assert tools == extractor._extraction_tools


def test_truncated_selection_falls_back_without_losing_facts(task):
    extractor, prepared, ctx = task
    extractor._llm.complete_json.side_effect = LLMOutputTruncatedError("length")
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    assert prepared.source_context in prompt
    assert tools == extractor._extraction_tools


def test_provider_error_is_shared_and_propagates(task):
    extractor, prepared, ctx = task
    failure = LLMCallError("offline")
    extractor._llm.complete_json.side_effect = failure
    for _ in range(2):
        with pytest.raises(LLMCallError) as caught:
            assemble_prepared(extractor, prepared, ctx)
        assert caught.value is failure
    extractor._llm.complete_json.assert_called_once()


def test_add_only_selection_retains_fixed_incremental_tools(task):
    extractor, prepared, ctx = task
    extractor._llm.complete_json.return_value = {"categories": ["event"]}
    request = assemble_prepared(extractor, prepared, ctx)
    assert {t["name"] for t in request.tools} == {"extract_event", "edit_memory", "append_memory"}
    assert "── event (" in request.prompt and "── profile (" not in request.prompt
    assert "FACT" in request.prompt


def test_lazy_run_uses_the_same_selected_rules_and_preserves_read_tools(task):
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prompts import PromptManager
    from extraction.schemas.registry import SchemaRegistry

    extractor, prepared, ctx = task
    llm = extractor._llm
    llm.complete_json.return_value = {"categories": ["entity"]}
    llm.complete_with_tools_messages = Mock(return_value=([], "[]"))
    registry = SchemaRegistry()
    loop = ExtractionReActLoop(llm, Mock(), registry, URIResolver(registry), PromptManager())
    assert loop.run(prepared.source_context, ctx).completed
    request = llm.complete_with_tools_messages.call_args
    tools = request.kwargs.get("tools", request.args[1] if len(request.args) > 1 else [])
    messages = request.kwargs.get("messages", request.args[0] if request.args else [])
    assert {t["name"] for t in tools} == {"read", "list", "get_relations", "get_access_stats",
                                         "extract_entity", "edit_memory", "append_memory"}
    assert "── entity (" in messages[0]["content"]
    assert "── profile (" not in messages[0]["content"]
    assert "── event (" not in messages[0]["content"]
    assert "If entity is actually about the user" in messages[0]["content"]
    assert "PRESERVE all specific details" in messages[0]["content"]
    llm.complete_json.assert_called_once()


def test_lazy_final_json_cannot_emit_an_unselected_category(task):
    import json

    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prompts import PromptManager
    from extraction.schemas.registry import SchemaRegistry

    extractor, prepared, ctx = task
    llm = extractor._llm
    call = {"name": "extract_profile", "input": {"routing_key": "identity", "abstract": "Name",
             "overview": "Name", "content": "Name", "confidence": .9}}
    llm.complete_with_tools_messages = Mock(return_value=([], json.dumps([call])))
    registry = SchemaRegistry()
    loop = ExtractionReActLoop(llm, Mock(), registry, URIResolver(registry), PromptManager())
    with pytest.raises(ToolCallProtocolError):
        loop.run(prepared.source_context, ctx)


@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("recover", [False, True])
def test_public_extraction_selects_once_and_keeps_recovery_scoped(mode, recover):
    """Exercise the real entry point, parser, two native passes and JSON recovery."""
    import json
    from datetime import datetime

    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry

    statement = "Project Beacon uses PostgreSQL."
    operation = {"tool": "extract_entity", "input": {
        "routing_key": "beacon", "object_name": "Project Beacon", "abstract": statement,
        "overview": statement, "content": statement, "confidence": .9,
    }}
    llm = Mock()
    def respond(prompt, schema):
        properties = set(schema["properties"])
        if properties == {"spans"}:
            # Phase 1 is advisory; shared selection independently reads the complete source.
            return {"spans": [{"start": 0, "end": 0, "categories": ["profile"]}]}
        if properties == {"categories"}:
            return {"categories": ["entity"]}
        assert properties == {"operations"}
        return {"operations": [operation]}
    llm.complete_json.side_effect = respond
    llm.complete_with_tools.return_value = ([{"tool": "extract_profile", "input": {}}]
                                            if recover else [operation])
    lazy_operation = {"name": operation["tool"], "input": operation["input"]}
    llm.complete_with_tools_messages.return_value = ([], json.dumps([lazy_operation]))
    registry = SchemaRegistry()
    fs = Mock()
    fs.exists.return_value = False
    extractor = Extractor(llm, mode=mode, fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry))
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    result = extractor.extract([{"id": "m1", "role": "user", "content": statement}], ctx,
                               session_time=datetime(2026, 10, 28), archive_id="archive-1")
    assert len(result) == 1 and result[0].content == statement
    assert result[0].provenance_ids
    selections = [
        call
        for call in llm.complete_json.call_args_list
        if set(call.kwargs["schema"]["properties"]) == {"categories"}
    ]
    assert len(selections) == 1
    if mode == "eager":
        assert llm.complete_with_tools.call_count == 2
        for request in llm.complete_with_tools.call_args_list:
            assert "── profile (" not in request.kwargs["prompt"]
            assert {t["name"] for t in request.kwargs["tools"]} == {
                "extract_entity", "edit_memory", "append_memory"}
        assert llm.complete_json.call_count == (3 if recover else 2)
    else:
        assert llm.complete_with_tools_messages.call_count == 2
        assert llm.complete_json.call_count == 2


def test_lazy_reusing_a_loop_does_not_leak_previous_selection(task):
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prompts import PromptManager

    extractor, _, ctx = task
    llm = extractor._llm
    llm.complete_json.side_effect = [{"categories": ["entity"]}, {"categories": ["event"]}]
    llm.complete_with_tools_messages = Mock(return_value=([], "[]"))
    registry = extractor._schema_registry
    loop = ExtractionReActLoop(llm, Mock(), registry, URIResolver(registry), PromptManager())
    loop.run("Project Beacon uses PostgreSQL.", ctx)
    loop.run("I attended a workshop yesterday.", ctx)
    first, second = llm.complete_with_tools_messages.call_args_list
    assert "extract_entity" in {t["name"] for t in first.kwargs["tools"]}
    assert "extract_entity" not in {t["name"] for t in second.kwargs["tools"]}
    assert "extract_event" in {t["name"] for t in second.kwargs["tools"]}
    assert llm.complete_json.call_count == 2


def test_separate_spans_do_not_mutate_shared_tools(task):
    extractor, prepared, ctx = task
    source = "user: I attended a workshop yesterday."
    other_span = _Span(1, 1, source_context=source,
                       prompt_parts=extractor._build_prompt_parts("English", source, ctx=ctx))
    other = _PreparedSpan(other_span, other_span.prompt_parts.render(), "other", source)
    import json
    extractor._llm.complete_json.side_effect = lambda prompt, schema: {
        "categories": [
            (
                "entity"
                if "Beacon" in json.loads(prompt.split("Runtime input (JSON):\n", 1)[1])["source"]
                else "event"
            )
        ]
    }
    full_tools = list(extractor._extraction_tools)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda span: assemble_prepared(extractor, span, ctx), [prepared, other]))
    assert "extract_entity" in {t["name"] for t in results[0].tools}
    assert "extract_entity" not in {t["name"] for t in results[1].tools}
    assert extractor._extraction_tools == full_tools


def test_custom_prompt_shape_keeps_full_assembly_and_source(task):
    extractor, prepared, ctx = task
    prepared.prompt = "CUSTOM RULES\n" + prepared.source_context
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    prefix = extractor._prefetch_prompt("", prepared.span.prefetch_result)
    assert prompt == prefix + prepared.prompt
    assert prepared.source_context in prompt
    assert "EXISTING MEMORY SNAPSHOT" in prefix
    assert tools is extractor._extraction_tools


def test_model_can_choose_an_update_after_selection_without_a_preplanned_target(task):
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.extraction.test_memory_update_flow import (
        ExtractionProvider,
        operation,
        setup_flow,
    )

    original = memory("beverage", category="preference", content="Prefers tea.", who="user")
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("preference", original)
    llm = ExtractionProvider(operation("preference", "beverage", "Prefers green tea.", uri))
    llm.complete_json = Mock(return_value={"categories": ["preference"]})
    from core.uri_resolver import URIResolver
    extractor = Extractor(llm, fs=fs, vector_index=index, embedder=embedder,
                          schema_registry=registry, uri_resolver=URIResolver(registry))
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
                                       [{"role": "user", "content": "I prefer green tea."}],
                                       "English", None, "", "", ctx)
    candidate, = extractor._structure_prepared_span(prepared, ctx)
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert candidate.content == "Prefers green tea."
    assert prepared.span.prefetch_result.read_versions == {uri: 1}


def test_generic_edit_tool_cannot_write_a_category_outside_the_selection():
    from core.uri_resolver import URIResolver
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prompts import PromptManager
    from tests.unit.extraction.test_incremental_edit_flow import eager_flow, edit

    ctx, fs, registry, uri, llm, extractor, _ = eager_flow()
    llm.complete_json = Mock(return_value={"categories": ["entity"]})
    llm.complete_with_tools_messages = Mock(return_value=([], "[]"))
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager())
    loop.run("The project uses PostgreSQL.", ctx)
    loop._read_nodes = {uri: fs.read_node(uri, ctx)}
    with pytest.raises(ToolCallProtocolError, match="category outside"):
        loop._to_candidate("edit_memory", edit(uri)["input"])


def test_time_anchored_source_with_tool_stats_is_preserved_and_can_be_scoped(task):
    from datetime import datetime
    extractor, _, ctx = task
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", datetime(2026, 10, 28), "", "Some unrelated tool statistics", ctx)
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    assert "Project Beacon uses PostgreSQL." in prompt
    assert "2026-10-28" in prompt
    assert "── profile (" not in prompt
    assert {t["name"] for t in tools} == {"extract_entity", "edit_memory", "append_memory"}


def test_custom_template_without_a_common_rule_boundary_uses_category_labels(task):
    extractor, _, ctx = task
    render = extractor._prompt_manager.render
    extractor._prompt_manager.render = lambda name, section, **kw: (
        render(name, section, **kw).replace("====================================================================", "CUSTOM SEPARATOR")
        if section == "system_prompt" else render(name, section, **kw)
    )
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, "", "", ctx)
    request = assemble_prepared(extractor, prepared, ctx)
    prompt, tools = request.prompt, request.tools
    assert prompt == prepared.prompt
    assert {tool["name"] for tool in tools} == {"extract_entity", "edit_memory", "append_memory"}


def test_raw_provider_exception_is_shared_without_repeating_selection(task):
    extractor, prepared, ctx = task
    failure = RuntimeError("provider rejected request")
    extractor._llm.complete_json.side_effect = failure
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            assemble_prepared(extractor, prepared, ctx)
        assert caught.value is failure
    extractor._llm.complete_json.assert_called_once()
