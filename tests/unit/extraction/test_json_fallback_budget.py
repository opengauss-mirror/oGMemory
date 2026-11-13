"""JSON extraction recovery budgets complete source and read snapshots together."""

import json
from unittest.mock import Mock

import pytest

from core.errors import LLMOutputTruncatedError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.occurrence_qualification import prompt_fits
from extraction.tools import Extractor, _NativeSpanOutcome, _Span
from tests.unit.commit.test_profile_merge import profile
from tests.unit.extraction.test_memory_update_flow import operation, setup_flow

SOURCE = "I now work as an engineer. My reference is R-742."


def prepare(*, old_chars=64000, bound=False, mode="eager", source=SOURCE, examples=None):
    old = "Historical fact. " * (old_chars // 17)
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("profile", profile("identity", old))
    llm = Mock()
    llm.complete_with_tools.return_value = [{"tool": "extract_profile", "input": {}}]
    llm.complete_with_tools_messages.return_value = ([{"name": "extract_profile", "input": {}}], "")
    fields = operation("profile", "identity", old + SOURCE if bound else SOURCE, uri)["input"]
    fields.update(abstract="Reader occupation.", overview="Works as an engineer.")
    if not bound:
        fields.pop("target_uri")
        fields.pop("expected_version")
    llm.complete_json.return_value = {"operations": [{"tool": "extract_profile", "input": fields}]}
    llm.complete_json.side_effect = lambda prompt, schema: (
        {"categories": ["profile"]}
        if set(schema["properties"]) == {"categories"}
        else llm.complete_json.return_value
    )
    extractor = Extractor(llm, mode=mode, schema_registry=registry, fs=fs,
                          vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    if examples is not None:
        render = extractor._prompt_manager.render
        extractor._prompt_manager.render = lambda name, section, **kwargs: (
            examples if name == "extraction" and section == "examples" else render(name, section, **kwargs)
        )
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=["profile"]), [{"role": "user", "content": source}],
        "English", None, "", "", ctx,
    )
    return extractor, llm, prepared, ctx, uri


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_complete_source_recovery_with_large_existing_memory(mode):
    extractor, llm, prepared, ctx, _ = prepare(mode=mode)
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "archive")

    prompt = llm.complete_json.call_args.args[0]
    assert candidate.content == SOURCE and SOURCE in prompt
    assert prompt_fits(prompt, llm.complete_json.call_args.kwargs["schema"])
    assert candidate.target_uri == "" and prepared.span.prefetch_result is None
    assert llm.complete_json.call_count == 2


@pytest.mark.parametrize("old_chars", [34, 500])
def test_complete_read_snapshot_authorizes_fallback_update(old_chars):
    extractor, llm, prepared, ctx, uri = prepare(old_chars=old_chars, bound=True)
    snapshot = prepared.span.prefetch_result.read_nodes[uri]
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "archive")

    prompt = llm.complete_json.call_args.args[0]
    assert snapshot.content in prompt and SOURCE in prompt
    assert prompt_fits(prompt, llm.complete_json.call_args.kwargs["schema"])
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert prepared.span.prefetch_result.read_versions == {uri: 1}
    assert candidate.content == snapshot.content + SOURCE
    assert llm.complete_json.call_count == 2


def test_summary_repair_fallback_uses_its_final_read_snapshot():
    extractor, llm, prepared, ctx, _ = prepare()
    operation = llm.complete_json.return_value["operations"][0]
    llm.complete_with_tools.return_value = [{
        "tool": operation["tool"], "input": {**operation["input"], "abstract": "x" * 201},
    }]
    llm.complete_json.side_effect = [
        {"repairs": [{"operation_index": 0, "abstract": "x" * 201}]},
        {"operations": [operation]},
    ]
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "archive")
    assert candidate.content == SOURCE
    assert prepared.span.prefetch_result is None and prepared.span.successful_read_versions == [{}]
    assert llm.complete_json.call_count == 3


def test_source_equal_to_a_template_example_is_preserved():
    source = "Example: The repair flag is --safe-mode.\n" + SOURCE
    extractor, llm, prepared, ctx, _ = prepare(source=source, examples=source)
    extractor._structure_prepared_span_dual(prepared, ctx, "archive")
    assert prepared.source_context in llm.complete_json.call_args.args[0]


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_truncated_native_output_recovers_complete_source_facts(mode):
    extractor, llm, prepared, ctx, _ = prepare(old_chars=500, mode=mode)
    prepared.span.message_ids = ["message-1"]
    llm.complete_with_tools.side_effect = LLMOutputTruncatedError("finish_reason=length")
    llm.complete_with_tools_messages.side_effect = LLMOutputTruncatedError("finish_reason=length")

    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "archive")

    assert candidate.content == SOURCE
    assert candidate.provenance_ids
    assert (candidate.target_uri, candidate.expected_version) == ("", None)
    assert SOURCE in llm.complete_json.call_args.args[0]
    assert prepared.span.prefetch_result is None
    assert llm.complete_json.call_count == 2


@pytest.mark.parametrize("bound", [False, True], ids=["source-facts", "read-snapshot"])
def test_normal_json_fallback_preserves_complete_facts_within_budget(bound):
    source = 'I now work as an engineer.\nMy reference is "R-742 中文".'
    old = "Name: Nora. Residence: Utrecht."
    ctx, fs, registry, uri, embedder, index, _ = setup_flow("profile", profile("identity", old))
    fields = operation("profile", "identity", old + "\n" + source if bound else source, uri)["input"]
    fields.update(abstract="Reader occupation.", overview="Works as an engineer.")
    if not bound:
        fields.pop("target_uri")
        fields.pop("expected_version")
    llm = Mock()
    llm.complete_json.return_value = {"operations": [{"tool": "extract_profile", "input": fields}]}
    llm.complete_json.side_effect = lambda prompt, schema: (
        {"categories": ["profile"]}
        if "categories" in schema["properties"]
        else llm.complete_json.return_value
    )
    extractor = Extractor(llm, mode="eager", schema_registry=registry, fs=fs,
                          vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=["profile"], message_ids=["source-1"]),
        [{"role": "user", "content": source, "id": "source-1"}], "English", None, "", "", ctx,
    )
    candidate, = extractor._structure_span_json_fallback(prepared, ctx, "archive")

    prompt = llm.complete_json.call_args.args[0]
    schema = llm.complete_json.call_args.kwargs["schema"]
    assert source in prompt and old in prompt
    assert len(json.dumps(prompt, ensure_ascii=False)) > len(prompt)
    assert prompt_fits(prompt, schema)
    assert candidate.content == fields["content"]
    assert candidate.provenance_ids == ["prov:1:archive:archive:source-1"]
    assert (candidate.target_uri, candidate.expected_version) == ((uri, 1) if bound else ("", None))
    assert prepared.span.prefetch_result.read_versions == {uri: 1}
    assert [
        set(call.kwargs["schema"]["properties"]) for call in llm.complete_json.call_args_list
    ] == [
        {"categories"},
        {"operations"},
    ]


def test_topic_draft_mode_reaches_both_fallback_prompts_and_preserves_source_facts():
    source = "Nora owns a silver necklace. Its heart charm was a gift from her grandmother."
    ctx = RequestContext("source-test", "nora", "agent", "necklace", "trace")
    fields = {"routing_key": "necklace", "object_name": "Nora's necklace", "scope_ref": "",
              "abstract": "Nora's silver necklace.", "content": source, "confidence": .95}
    llm = Mock()
    llm.complete_json.return_value = {"operations": [{"tool": "extract_entity", "input": fields}]}
    llm.complete_json.side_effect = lambda prompt, schema: (
        {"categories": ["entity"]}
        if "categories" in schema["properties"]
        else llm.complete_json.return_value
    )
    extractor = Extractor(llm, node_processing_enabled=True, dual_pass_enabled=False)
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=["entity"], message_ids=["gift"]),
        [{"id": "gift", "role": "user", "content": source}], "English", None, "", "", ctx,
    )
    for prompt in (prepared.prompt, prepared.span.fallback_read_prompt, prepared.span.fallback_source_prompt):
        assert prepared.source_context in prompt and source in prompt
        assert prompt.count("TOPIC_DRAFTS:") == 1
        assert "content lists every supported atomic source fact once" in prompt
    candidate, = extractor._structure_span_json_fallback(prepared, ctx, "archive")
    candidate, = extractor._finalize_span_outcome(
        _NativeSpanOutcome(prepared, candidates=[candidate]), ctx, "archive",
    )
    assert candidate.content == source and candidate.object_name == fields["object_name"]
    assert candidate.provenance_ids == ["prov:1:archive:archive:gift"]
    assert candidate.source_contexts == [prepared.source_context]
    assert prompt_fits(
        llm.complete_json.call_args.args[0], llm.complete_json.call_args.kwargs["schema"]
    )
    assert [
        set(call.kwargs["schema"]["properties"]) for call in llm.complete_json.call_args_list
    ] == [
        {"categories"},
        {"operations"},
    ]
