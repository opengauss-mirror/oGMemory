"""JSON extraction recovery budgets complete source and read snapshots together."""

from unittest.mock import Mock

import pytest

from core.uri_resolver import URIResolver
from core.errors import LLMOutputTruncatedError
from extraction.occurrence_qualification import prompt_fits
from extraction.tools import Extractor, _Span
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
    assert llm.complete_json.call_count == 1


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
    assert llm.complete_json.call_count == 1


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
    assert llm.complete_json.call_count == 2


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
    assert llm.complete_json.call_count == 1
