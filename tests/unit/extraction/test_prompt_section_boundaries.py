"""Only template sections may be cropped; memory/source data is opaque."""

from unittest.mock import Mock

import pytest

from core.errors import LLMOutputTruncatedError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.prompt_assembly import assemble_prepared
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_memory_identity import memory
from tests.unit.extraction.test_memory_update_flow import setup_flow

MARKER_BODY = ("Project handbook.\n# Few-shot Examples\n## event Example\n"
               "FACT_882: Beacon supports Oracle.\nBound upserts:\nTAIL_883")


def provider():
    llm = Mock()
    llm.complete_json.side_effect = lambda prompt, schema: (
        {"categories": ["entity"]} if set(schema["properties"]) == {"categories"}
        else {"operations": []}
    )
    return llm


def test_prefetched_body_is_opaque_to_example_selection():
    original = memory("handbook", content=MARKER_BODY)
    original.abstract, original.overview = "Project handbook", "Setup guide"
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("entity", original)
    extractor = Extractor(provider(), fs=fs, vector_index=index, embedder=embedder,
                          schema_registry=registry, uri_resolver=URIResolver(registry))
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "The handbook now includes PostgreSQL setup."}],
        "English", None, "", "", ctx)
    reads = prepared.span.prefetch_result
    prompt, tools = assemble_prepared(extractor, prepared, ctx)
    assert MARKER_BODY in prompt
    assert prepared.source_context in prompt
    assert prepared.span.prefetch_result is reads
    assert reads.read_nodes[uri].content == MARKER_BODY
    assert reads.read_versions == {uri: 1}
    assert {tool["name"] for tool in tools} == {"extract_entity", "edit_memory", "append_memory"}


def test_compact_json_recovery_preserves_source_with_example_markers():
    llm = provider()
    extractor = Extractor(llm)
    # A small valid template example ensures normal assembly can be scoped first.
    render = extractor._prompt_manager.render
    extractor._prompt_manager.render = lambda name, section, **kw: (
        "# Few-shot Examples\n## entity Example\nTemplate example.\n" if section == "examples"
        else render(name, section, **kw)
    )
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": MARKER_BODY}], "English", None, "", "", ctx)
    extractor._structure_span_json_fallback(prepared, ctx, None,
        errors=[LLMOutputTruncatedError("length")])
    prompt = llm.complete_json.call_args.args[0]
    assert MARKER_BODY in prompt
    assert "Template example." not in prompt
    assert prepared.span.prefetch_result is None
    assert llm.complete_json.call_count == 2


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_source_looks_like_examples_but_still_allows_category_selection(mode):
    extractor = Extractor(provider(), mode=mode)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": MARKER_BODY}], "English", None, "", "", ctx)
    prompt, tools = assemble_prepared(extractor, prepared, ctx)
    assert MARKER_BODY in prompt
    assert {tool["name"] for tool in tools} == {"extract_entity", "edit_memory", "append_memory"}


@pytest.mark.parametrize("select_all", [False, True])
def test_lazy_json_recovery_keeps_selection_and_pairs_the_last_read_snapshot(select_all):
    from extraction.prefetch import PrefetchResult
    from extraction.schemas.registry import SchemaRegistry
    registry = SchemaRegistry()
    llm = provider()
    if select_all:
        llm.complete_json.side_effect = lambda prompt, schema: (
            {"categories": schema["properties"]["categories"]["items"]["enum"]}
            if set(schema["properties"]) == {"categories"} else {"operations": []}
        )
    extractor = Extractor(llm, mode="lazy", fs=Mock(), schema_registry=registry,
                          uri_resolver=URIResolver(registry))
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, "", "", ctx)
    # A failed lazy/native attempt may leave its successful reads on the shared span.
    prepared.span.prefetch_result = PrefetchResult(messages=[MARKER_BODY], read_versions={"old-uri": 4})
    extractor._structure_span_json_fallback(prepared, ctx, None)
    prompt = extractor._llm.complete_json.call_args.args[0]
    schema = extractor._llm.complete_json.call_args.kwargs["schema"]
    assert MARKER_BODY in prompt
    expected = ({tool["name"] for tool in extractor._extraction_tools} if select_all else
                {"extract_entity", "edit_memory", "append_memory"})
    assert set(schema["$defs"]["ExtractionOperation"]["properties"]["tool"]["enum"]) == expected


def test_reference_context_feedback_and_task_suffix_keep_their_boundaries():
    from dataclasses import replace
    extractor = Extractor(provider(), node_processing_enabled=True)
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, MARKER_BODY, "", ctx)
    prompt, tools = assemble_prepared(extractor, prepared, ctx)
    assert prepared.span.prompt_parts.prefix_data in prompt
    assert prepared.span.prompt_parts.suffix_data in prompt
    assert "TOPIC_DRAFTS" in prompt
    recovery = replace(prepared, prompt=prompt + "\nVALIDATION_FEEDBACK_998")
    recovered, recovered_tools = assemble_prepared(extractor, recovery, ctx)
    assert recovered == prompt + "\nVALIDATION_FEEDBACK_998"
    assert recovered_tools == tools
    extractor._llm.complete_json.assert_called_once()


def test_lazy_message_building_does_not_crop_runtime_identity_or_prefetched_data():
    from extraction.extraction_react_loop import ExtractionReActLoop
    from extraction.prefetch import PrefetchResult
    from extraction.prompts import PromptManager
    from extraction.schemas.registry import SchemaRegistry
    registry = SchemaRegistry()
    llm = provider()
    llm.complete_with_tools_messages.return_value = ([], "[]")
    ctx = RequestContext("acct", MARKER_BODY, "agent", "session", "trace")
    loop = ExtractionReActLoop(llm, Mock(), registry, URIResolver(registry), PromptManager())
    result = loop.run(MARKER_BODY, ctx, PrefetchResult(messages=[MARKER_BODY]))
    assert result.completed
    request = llm.complete_with_tools_messages.call_args
    messages = request.kwargs["messages"]
    assert MARKER_BODY in messages[0]["content"]
    assert MARKER_BODY == messages[1]["content"]
    assert any(MARKER_BODY in message["content"] for message in messages if message["role"] == "user")
    assert "── profile (" not in messages[0]["content"]
    assert {tool["name"] for tool in request.kwargs["tools"]} == {
        "read", "list", "get_relations", "get_access_stats",
        "extract_entity", "edit_memory", "append_memory"}
