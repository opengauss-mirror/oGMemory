import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from extraction.tools import Extractor
from providers.llm.openai_llm import OpenAILLM
from tests.unit.extraction.test_json_fallback_budget import SOURCE, prepare
from tests.unit.extraction.test_native_validation_recovery import ctx, operation, prepared


@pytest.mark.parametrize("repair_response", ["missing_envelope", "truncated"])
def test_provider_summary_response_recovers_through_source_fallback(repair_response):
    llm = OpenAILLM(api_key="local-recovery-test", base_url="http://127.0.0.1:1/v1")
    stages = []

    def respond(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        finish_reason = "stop"
        tool_calls = []
        if kwargs.get("tools"):
            stages.append("native")
            tool_calls = [SimpleNamespace(function=SimpleNamespace(
                name="extract_preference", arguments=json.dumps(operation(abstract="a" * 201)["input"]),
            ))]
            content = None
            finish_reason = "tool_calls"
        elif prompt.startswith("Repair only the oversized summary fields"):
            stages.append("summary_repair")
            content = json.dumps({"abstract": "Alice likes coffee"})
            if repair_response == "truncated":
                finish_reason = "length"
        else:
            stages.append("source_fallback")
            content = json.dumps({"operations": [operation()]})
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason, message=SimpleNamespace(content=content, tool_calls=tool_calls),
        )])

    llm._create_completion = Mock(side_effect=respond)
    try:
        candidate, = Extractor(llm, dual_pass_enabled=False)._structure_prepared_span_dual(
            prepared(), ctx(), "archive",
        )
    finally:
        llm._client.close()

    assert candidate.content == operation()["input"]["content"]
    assert candidate.abstract == operation()["input"]["abstract"]
    assert candidate.provenance_ids
    assert stages == ["native", "summary_repair", "source_fallback"]


@pytest.mark.parametrize("field,value", [("abstract", "a" * 201), ("overview", "o" * 5001)])
def test_fallback_summary_repair_preserves_read_snapshot_and_details(field, value):
    extractor, llm, span, context, uri = prepare(old_chars=500, bound=True)
    span.span.message_ids = ["message-1"]
    extraction = llm.complete_json.return_value["operations"][0]
    body = extraction["input"]["content"]
    extraction["input"][field] = value
    llm.complete_json.side_effect = [
        {"operations": [extraction]},
        {"repairs": [{"operation_index": 0, field: "Reader works as an engineer."}]},
    ]

    candidate, = extractor._structure_prepared_span_dual(span, context, "archive")

    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert candidate.content == body + f"\n\nOriginal {field} detail:\n" + value
    assert SOURCE in candidate.content
    assert getattr(candidate, field) == "Reader works as an engineer."
    assert span.span.prefetch_result.read_versions == {uri: 1}
    assert len(candidate.provenance_ids) == 1
    assert llm.complete_json.call_count == 2
