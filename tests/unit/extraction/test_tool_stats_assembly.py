"""Session tool statistics must not lose their extraction tool during assembly."""

from unittest.mock import Mock

import pytest

from core.errors import LLMOutputTruncatedError
from core.models import RequestContext
from extraction.prompt_assembly import assemble_prepared
from extraction.tool_collector import format_tool_stats_text
from extraction.tools import Extractor, _Span

STATS = format_tool_stats_text({"web_search": {"call_count": 3, "success_count": 0, "fail_count": 3}})


def provider():
    llm = Mock()
    llm.complete_json.side_effect = lambda prompt, schema: (
        {"categories": ["preference"]} if set(schema["properties"]) == {"categories"}
        else {"operations": []}
    )
    return llm


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_attached_tool_statistics_keep_the_tool_schema_and_rules(mode):
    extractor = Extractor(provider(), mode=mode)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        [{"role": "user", "content": "I prefer tea."}], "English", None, "", STATS, ctx)

    prompt, tools = assemble_prepared(extractor, prepared, ctx)

    assert STATS in prompt
    assert STATS not in prepared.source_context
    assert "extract_tool" in {tool["name"] for tool in tools}
    assert "extract_preference" in {tool["name"] for tool in tools}
    assert "── tool (" in prompt
    assert "── entity (" not in prompt
    extractor._llm.complete_json.assert_called_once()


@pytest.mark.parametrize("truncated", [False, True])
def test_json_recovery_reuses_the_tool_category_for_attached_statistics(truncated):
    extractor = Extractor(provider())
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        [{"role": "user", "content": "I prefer tea."}], "English", None, "", STATS, ctx)
    assemble_prepared(extractor, prepared, ctx)

    extractor._structure_span_json_fallback(prepared, ctx, None,
        errors=[LLMOutputTruncatedError("length")] if truncated else None)

    request = extractor._llm.complete_json.call_args
    names = request.kwargs["schema"]["$defs"]["ExtractionOperation"]["properties"]["tool"]["enum"]
    assert "extract_tool" in names
    assert STATS in request.args[0]
    assert extractor._llm.complete_json.call_count == 2  # One selection, one recovery.


@pytest.mark.parametrize("stats", ["", "   ", STATS])
def test_non_attached_statistics_do_not_expand_early_span_categories(stats):
    extractor = Extractor(provider())
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    messages = [{"role": "user", "content": "I prefer tea."}] + [
        {"role": "assistant", "content": "Acknowledged."} for _ in range(9)]
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        messages, "English", None, "", stats, ctx)

    prompt, tools = assemble_prepared(extractor, prepared, ctx)

    assert STATS not in prompt
    assert "extract_tool" not in {tool["name"] for tool in tools}


def test_disabled_tool_schema_is_not_reenabled_for_statistics():
    from dataclasses import replace
    from extraction.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    registry.register(replace(registry.get("tool"), enabled=False))
    extractor = Extractor(provider(), schema_registry=registry)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        [{"role": "user", "content": "I prefer tea."}], "English", None, "", STATS, ctx)

    _, tools = assemble_prepared(extractor, prepared, ctx)

    assert "extract_tool" not in {tool["name"] for tool in tools}


@pytest.mark.parametrize("stats", ["", "   ", None])
def test_empty_tail_statistics_do_not_expand_categories(stats):
    extractor = Extractor(provider())
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        [{"role": "user", "content": "I prefer tea."}], "English", None, "", stats, ctx)
    _, tools = assemble_prepared(extractor, prepared, ctx)
    assert "extract_tool" not in {tool["name"] for tool in tools}


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_structured_extraction_can_emit_a_memory_from_statistics_alone(mode):
    import json
    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry

    llm = provider()
    operation = {"tool": "extract_tool", "input": {
        "routing_key": "web_search", "abstract": "web_search failed three times.",
        "overview": "web_search: 3 calls, all failed.",
        "content": "web_search failed on all three calls in this session.", "confidence": .9,
    }}
    llm.complete_with_tools.return_value = [operation]
    llm.complete_with_tools_messages.return_value = ([], json.dumps([
        {"name": operation["tool"], "input": operation["input"]}]))
    fs = Mock()
    fs.exists.return_value = False
    registry = SchemaRegistry()
    extractor = Extractor(llm, mode=mode, fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry))
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["preference"]),
        [{"role": "user", "content": "I prefer tea."}], "English", None, "", STATS, ctx)

    candidate, = extractor._structure_prepared_span(prepared, ctx)

    assert candidate.category == "tool"
    assert candidate.owner_scope == "agent"
    assert candidate.content == operation["input"]["content"]
    llm.complete_json.assert_called_once()
