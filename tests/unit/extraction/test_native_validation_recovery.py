"""Output constraint failures must recover within the existing span budget."""

import json
from copy import deepcopy
from unittest.mock import Mock

import jsonschema
import pytest

from core.errors import ExtractionIncompleteError, LLMOutputTruncatedError, ToolCallProtocolError
from core.models import ContextNode, RequestContext
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prefetch import PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _PreparedSpan, _Span


def ctx():
    return RequestContext(account_id="acme", user_id="alice", agent_id="bot", session_id="s1", trace_id="t1")


def read_snapshot(uri, version):
    return ContextNode(uri=uri, context_type="MEMORY", category="entity",
                       owner_space=ctx().user_space_name(), level=0,
                       abstract="Atlas", overview="Atlas is red.", content="Atlas is red.",
                       metadata={"version": version, "scope_ref": "Project A"})


def operation(key="coffee", **changes):
    return {"tool": "extract_preference", "input": {
        "routing_key": key, "abstract": "Alice likes coffee", "overview": "Coffee preference",
        "content": "Alice likes coffee and drinks it after lunch.", "confidence": 0.9,
        **changes,
    }}


def prepared():
    return _PreparedSpan(_Span(0, 0, message_ids=["m1"]), "Alice likes coffee after lunch.", "a" * 64)


@pytest.mark.parametrize("invalid_input", [{"raw": "{broken json"}, {}], ids=["raw", "missing-fields"])
@pytest.mark.parametrize("summary_first", [False, True])
def test_protocol_batch_recovers_all_operations_through_json(invalid_input, summary_first):
    invalid = {"tool": "extract_preference", "input": invalid_input}
    native = [operation("tea", abstract="a" * 201), invalid] if summary_first else [invalid]
    recovered = [operation("tea"), operation("coffee")]
    llm = Mock()
    llm.complete_with_tools.return_value = native
    llm.complete_json.return_value = {"operations": recovered}

    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive-1")

    assert [item.routing_key for item in result] == ["tea", "coffee"]
    assert all(item.provenance_ids for item in result)
    assert llm.complete_with_tools.call_count == 2
    llm.complete_json.assert_called_once()
    assert prepared().prompt in llm.complete_json.call_args.args[0]


@pytest.mark.parametrize("field,value", [("abstract", "a" * 201), ("overview", "o" * 5001)],
                         ids=["abstract", "overview"])
def test_native_constraint_failure_uses_one_feedback_fallback(field, value):
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(**{field: value})]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, field: "Coffee preference"}]}

    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive-1")

    assert len(result) == 1
    assert result[0].content.startswith(operation()["input"]["content"])
    assert value in result[0].content
    assert result[0].provenance_ids
    assert llm.complete_with_tools.call_count == 2
    llm.complete_json.assert_called_once()
    assert field in llm.complete_json.call_args.args[0]
    assert "fields_to_summarize" in llm.complete_json.call_args.args[0]


@pytest.mark.parametrize("field,limit", [("abstract", 200), ("overview", 5000)])
def test_summary_repair_uses_only_the_original_field_not_shared_source(field, limit):
    original_summary = "FIELD_LOCAL_FACT " * (limit // 10 + 1)
    other_field = "overview" if field == "abstract" else "abstract"
    call = operation(**{field: original_summary, other_field: "OTHER_FIELD_ONLY_FACT",
                        "content": "BODY_ONLY_FACT"})
    legal = operation("legal", content="OTHER_OPERATION_ONLY_FACT")
    llm = Mock()
    llm.complete_with_tools.return_value = [legal, call]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 1, field: "FIELD_LOCAL_FACT"}]}
    span = prepared()
    span.prompt = "SHARED_SOURCE_ONLY_FACT. EXISTING_MEMORY_ONLY_FACT."

    result = Extractor(llm)._structure_prepared_span_dual(span, ctx(), None)

    repair_prompt = llm.complete_json.call_args.args[0]
    assert original_summary in repair_prompt
    for unrelated in ("SHARED_SOURCE_ONLY_FACT", "EXISTING_MEMORY_ONLY_FACT", "OTHER_FIELD_ONLY_FACT",
                      "BODY_ONLY_FACT", "OTHER_OPERATION_ONLY_FACT"):
        assert unrelated not in repair_prompt
    assert result[0].content == legal["input"]["content"]
    assert getattr(result[1], other_field) == "OTHER_FIELD_ONLY_FACT"
    assert result[1].content.startswith("BODY_ONLY_FACT")
    assert original_summary in result[1].content
    llm.complete_json.assert_called_once()


@pytest.mark.parametrize("oversized_fields", [
    [("abstract",)], [("overview",)], [("abstract", "overview")],
    [("abstract",), ("overview",)],
], ids=["abstract", "overview", "both", "different-operations"])
def test_repair_schema_only_allows_requested_fields_for_each_operation(oversized_fields):
    calls = [operation(f"preference-{index}", **{
        field: ("a" * 201 if field == "abstract" else "o" * 5001)
        for field in fields
    }) for index, fields in enumerate(oversized_fields)]
    llm = Mock()
    llm.complete_with_tools.return_value = calls

    def respond(prompt, schema):
        valid = {"repairs": [{"operation_index": index,
                               **{field: "Coffee preference" for field in fields}}
                              for index, fields in enumerate(oversized_fields)]}
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(valid)
        for index, fields in enumerate(oversized_fields):
            for field in fields:
                missing = deepcopy(valid)
                del missing["repairs"][index][field]
                assert not validator.is_valid(missing)
            for field in {"abstract", "overview"} - set(fields):
                extra = deepcopy(valid)
                extra["repairs"][index][field] = "Unrequested summary"
                assert not validator.is_valid(extra)
        if len(oversized_fields) > 1:
            duplicate = deepcopy(valid)
            duplicate["repairs"][1] = deepcopy(duplicate["repairs"][0])
            duplicate["repairs"][1][oversized_fields[0][0]] = "Another valid summary"
            assert not validator.is_valid(duplicate)
        return valid

    llm.complete_json.side_effect = respond
    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), None)
    assert len(result) == len(calls)
    llm.complete_json.assert_called_once()


def test_invalid_batch_is_atomic_and_a_complete_native_attempt_can_win():
    llm = Mock()
    llm.complete_with_tools.side_effect = [
        [operation("tea"), operation("bad", abstract="a" * 201)], [operation("coffee")],
    ]

    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), None)

    assert [item.routing_key for item in result] == ["coffee"]
    llm.complete_json.assert_not_called()


def test_invalid_fallback_is_not_retried_or_truncated():
    invalid = operation(overview="o" * 5001)
    original = deepcopy(invalid)
    llm = Mock()
    llm.complete_with_tools.return_value = [invalid]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, "overview": "o" * 5001}]}

    with pytest.raises(ToolCallProtocolError, match="overview"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)

    llm.complete_with_tools.assert_not_called()
    llm.complete_json.assert_called_once()
    assert invalid == original


def test_native_constraint_repair_cannot_discard_the_whole_batch():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="a" * 201)]
    llm.complete_json.return_value = {"repairs": []}

    with pytest.raises(ToolCallProtocolError):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)

    llm.complete_json.assert_called_once()


def test_finish_length_recovers_source_through_json():
    llm = Mock()
    llm.complete_with_tools.side_effect = LLMOutputTruncatedError("finish_reason=length")
    llm.complete_json.return_value = {"operations": [operation()]}

    candidate, = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive")

    assert candidate.content == operation()["input"]["content"]
    assert candidate.provenance_ids
    assert llm.complete_json.call_count == 1


def test_lazy_bad_native_call_returns_error_and_cannot_be_recorded():
    loop = ExtractionReActLoop(llm=Mock(), fs=Mock(), registry=SchemaRegistry(),
                              uri_resolver=Mock(), prompt_manager=Mock())
    call = operation(abstract="a" * 201)

    result = loop._execute_tool(call["tool"], call["input"], ctx())

    assert "error" in result
    assert "abstract" in result["error"]
    assert result.get("status") != "recorded"
    with pytest.raises(ToolCallProtocolError):
        loop._post_tool_call_safety_check([call], [], [], ctx())


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2]])
def test_repair_requires_all_failed_operation_indices_once(indices):
    llm = Mock()
    llm.complete_with_tools.return_value = [operation("tea", abstract="a" * 201),
                                            operation("coffee", abstract="b" * 201)]
    llm.complete_json.return_value = {"repairs": [
        {"operation_index": index, "abstract": "Coffee preference"} for index in indices
    ]}

    with pytest.raises(ToolCallProtocolError, match="every operation index exactly once"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)

    llm.complete_json.assert_called_once()


def test_repair_keeps_legal_operations_and_cannot_change_source_identity():
    legal = operation("tea")
    invalid = operation("coffee", abstract="a" * 201, who="Alice", scope_ref="Project A")
    original = deepcopy([legal, invalid])
    llm = Mock()
    llm.complete_with_tools.return_value = [legal, invalid]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 1, "abstract": "Coffee preference"}]}

    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), None)

    assert [item.routing_key for item in result] == ["tea", "coffee"]
    assert result[0].content == legal["input"]["content"]
    assert result[1].scope_ref == "Project A"
    assert result[1].who == "Alice"
    assert [legal, invalid] == original


def test_repair_may_not_change_non_summary_fields():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="a" * 201)]
    llm.complete_json.return_value = {"repairs": [{
        "operation_index": 0, "abstract": "Coffee preference", "scope_ref": "Project B",
    }]}

    with pytest.raises(ToolCallProtocolError, match="invalid abstract/overview fields"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)


def test_repair_does_not_overflow_or_truncate_content():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="a" * 201, content="c" * 100000)]

    with pytest.raises(ToolCallProtocolError, match="preserve complete content"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)

    llm.complete_json.assert_not_called()


def test_unrepairable_identity_is_a_hard_failure():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="a" * 201, routing_key="/")]

    with pytest.raises(ToolCallProtocolError, match="routing_key"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), None)

    llm.complete_json.assert_not_called()


def test_repair_preserves_explicit_read_target_and_version():
    uri = "ctx://acme/users/alice/memories/entities/atlas"
    llm = Mock()
    llm.complete_with_tools.return_value = [{"tool": "extract_entity", "input": {
        "routing_key": "atlas", "object_name": "Atlas", "scope_ref": "Project A",
        "target_uri": uri, "expected_version": 2,
        "abstract": "Atlas instrument details. " * 15,
        "overview": "Atlas has a red case.", "content": "Atlas has a red case.", "confidence": .9,
    }}]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, "abstract": "Atlas: red case"}]}
    span = prepared()
    span.span.prefetch_result = PrefetchResult(read_versions={uri: 2}, read_nodes={uri: read_snapshot(uri, 2)})

    candidate, = Extractor(llm)._structure_prepared_span_dual(span, ctx(), "archive-1")

    assert (candidate.target_uri, candidate.expected_version) == (uri, 2)
    assert (candidate.owner_scope, candidate.object_name, candidate.scope_ref) == ("user", "Atlas", "Project A")
    assert candidate.provenance_ids


def test_repair_cannot_authorize_an_unread_target():
    llm = Mock()
    llm.complete_with_tools.return_value = [{"tool": "extract_entity", "input": {
        "routing_key": "atlas", "object_name": "Atlas", "scope_ref": "Project A",
        "target_uri": "ctx://acme/users/alice/memories/entities/atlas", "expected_version": 2,
        "abstract": "Atlas instrument details. " * 15,
        "overview": "Atlas has a red case.", "content": "Atlas has a red case.", "confidence": .9,
    }}]

    with pytest.raises(ToolCallProtocolError, match="successful read"):
        Extractor(llm)._repair_native_candidate_batch(prepared(), llm.complete_with_tools.return_value, ctx(), "archive-1")

    llm.complete_json.assert_not_called()


def test_repair_refuses_an_oversized_complete_prompt():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="original field " * 5000)]
    span = prepared()

    with pytest.raises(ExtractionIncompleteError, match="complete context budget"):
        Extractor(llm)._structure_prepared_span_dual(span, ctx(), None)

    llm.complete_json.assert_not_called()


def test_lazy_validation_restarts_the_whole_span_with_error_feedback():
    llm = Mock()
    llm.complete_with_tools_messages.return_value = ([operation(abstract="a" * 201)], "")
    llm.complete_with_tools.return_value = [operation()]
    extractor = Extractor(llm, fs=Mock(), schema_registry=SchemaRegistry(),
                          uri_resolver=Mock(), mode="lazy")
    extractor._prefetcher = None

    candidate, = extractor._structure_prepared_span_dual(prepared(), ctx(), "archive-1")

    assert candidate.content == operation()["input"]["content"]
    assert llm.complete_with_tools_messages.call_count == 2
    assert llm.complete_with_tools.call_count == 2
    llm.complete_json.assert_not_called()
    for call in llm.complete_with_tools.call_args_list:
        assert prepared().prompt in call.kwargs["prompt"]
        assert "abstract" in call.kwargs["prompt"]
        assert "exceeds maximum" in call.kwargs["prompt"]


@pytest.mark.parametrize("field,value", [("abstract", "a" * 219), ("overview", "o" * 5001)])
def test_lazy_final_json_repairs_summary_with_original_read_authority(field, value):
    uri = "ctx://acme/users/alice/memories/entities/atlas"
    fields = {"routing_key": "atlas", "object_name": "Atlas", "scope_ref": "Project A",
              "target_uri": uri, "expected_version": 2, "abstract": "Atlas",
              "overview": "Atlas is red.", "content": "Atlas is red.", "confidence": .9,
              field: value}
    llm = Mock()
    llm.complete_with_tools_messages.side_effect = [
        ([{"name": "read", "input": {"uri": uri}}], ""),
        ([], json.dumps([{"name": "extract_entity", "input": fields}])),
    ]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, field: "Atlas is red."}]}
    fs = Mock()
    fs.read_node.return_value = read_snapshot(uri, 2)
    extractor = Extractor(llm, fs=fs, schema_registry=SchemaRegistry(),
                          uri_resolver=Mock(), mode="lazy", dual_pass_enabled=False)
    extractor._prefetcher = None
    span = prepared()

    candidate, = extractor._structure_prepared_span_dual(span, ctx(), "archive-1")

    assert (candidate.target_uri, candidate.expected_version) == (uri, 2)
    assert getattr(candidate, field) == "Atlas is red."
    assert candidate.content.startswith(fields["content"])
    assert value in candidate.content
    assert candidate.provenance_ids
    assert span.span.prefetch_result.read_versions == {uri: 2}
    assert llm.complete_with_tools_messages.call_count == 2
    llm.complete_json.assert_called_once()


def test_programming_errors_do_not_enter_output_repair():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation()]
    extractor = Extractor(llm)
    extractor._parse_tool_call_fn = Mock(side_effect=RuntimeError("parser bug"))

    with pytest.raises(RuntimeError, match="parser bug"):
        extractor._structure_prepared_span_dual(prepared(), ctx(), None)

    llm.complete_json.assert_not_called()


def test_early_validation_keeps_existing_empty_text_representation():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="", overview="", content="")]

    candidate, = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), None)

    assert (candidate.abstract, candidate.overview, candidate.content) == ("", "", "")
    llm.complete_json.assert_not_called()


def test_lazy_eager_repair_keeps_the_failed_batch_read_snapshot():
    uri = "ctx://acme/users/alice/memories/entities/atlas"
    llm = Mock()
    llm.complete_with_tools_messages.return_value = ([operation(abstract="a" * 201)], "")
    llm.complete_with_tools.return_value = [{"tool": "extract_entity", "input": {
        "routing_key": "atlas", "object_name": "Atlas", "scope_ref": "Project A",
        "target_uri": uri, "expected_version": 2,
        "abstract": "a" * 201, "overview": "Atlas is red.", "content": "Atlas is red.", "confidence": .9,
    }}]
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, "abstract": "Atlas: red case"}]}
    extractor = Extractor(llm, fs=Mock(), schema_registry=SchemaRegistry(),
                          uri_resolver=Mock(), mode="lazy")
    extractor._prefetcher = None

    def supply_snapshot(span, prompt, context):
        span.prefetch_result = PrefetchResult(read_uris={uri}, read_versions={uri: 2},
                                             messages=["Atlas v2 complete body"], read_nodes={uri: read_snapshot(uri, 2)})
        return prompt + " Atlas v2 complete body"

    extractor._with_prefetch_context = supply_snapshot
    span = prepared()

    candidate, = extractor._structure_prepared_span_dual(span, ctx(), "archive-1")

    assert (candidate.target_uri, candidate.expected_version) == (uri, 2)
    assert span.span.prefetch_result.read_versions == {uri: 2}
    assert "Atlas v2 complete body" not in llm.complete_json.call_args.args[0]
    llm.complete_json.assert_called_once()


def test_failed_attempt_snapshot_isolated_when_the_other_attempt_reads_a_new_version():
    uri = "ctx://acme/users/alice/memories/entities/atlas"
    shared_snapshot = PrefetchResult(read_uris={uri})
    versions = iter((2, 3))
    llm = Mock()
    llm.complete_with_tools_messages.return_value = ([operation(abstract="a" * 201)], "")

    def native_output(**kwargs):
        version = shared_snapshot.read_versions[uri]
        return [{"tool": "extract_entity", "input": {
            "routing_key": "atlas", "object_name": "Atlas", "scope_ref": "Project A",
            "target_uri": uri, "expected_version": version, "abstract": "a" * 201,
            "overview": f"Atlas snapshot version {version}.",
            "content": f"Atlas snapshot version {version}.", "confidence": .9,
        }}]

    llm.complete_with_tools.side_effect = native_output
    llm.complete_json.return_value = {"repairs": [{"operation_index": 0, "abstract": "Atlas snapshot"}]}
    extractor = Extractor(llm, fs=Mock(), schema_registry=SchemaRegistry(),
                          uri_resolver=Mock(), mode="lazy")
    extractor._prefetcher = None

    def supply_snapshot(span, prompt, context):
        version = next(versions)
        shared_snapshot.read_versions[uri] = version
        shared_snapshot.read_nodes[uri] = ContextNode(
            uri=uri, context_type="MEMORY", category="entity", owner_space="user:alice", level=0,
            abstract="Atlas", overview=f"snapshot {version}", content=f"snapshot {version}",
            metadata={"version": version, "scope_ref": "Project A"})
        span.prefetch_result = shared_snapshot
        return prompt + f" Atlas snapshot version {version}."

    extractor._with_prefetch_context = supply_snapshot
    span = prepared()

    candidate, = extractor._structure_prepared_span_dual(span, ctx(), None)

    assert candidate.expected_version == 2
    assert candidate.content.startswith("Atlas snapshot version 2.")
    assert span.span.prefetch_result.read_versions == {uri: 2}
    assert span.span.prefetch_result.read_nodes[uri].metadata["version"] == 2
    assert shared_snapshot.read_versions == {uri: 3}
    assert "Atlas snapshot version 2." not in llm.complete_json.call_args.args[0]
    assert "Atlas snapshot version 3." not in llm.complete_json.call_args.args[0]
    llm.complete_json.assert_called_once()
