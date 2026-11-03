from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.errors import ExtractionIncompleteError
from extraction.event_reconciliation import reconcile_event_drafts
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_event_occurrence_reconciliation import proposal
from tests.unit.extraction.test_source_occurrence_reconciliation import CTX, TIME, messages, writer
from tests.unit.extraction.test_source_summary_recovery import SummaryLLM


@pytest.mark.parametrize("field", ["when", "who", "where"])
@pytest.mark.parametrize("summary", [False, True])
def test_empty_optional_context_recovered_then_independently_verified(field, summary):
    response = proposal([[0], [1]])
    response["groups"][0][field] = ""
    if summary:
        response["groups"][0]["abstract"] = "A supported inspection detail. " * 9
    original = deepcopy(response)
    replies = [response]
    if summary:
        replies.append({"repairs": [{"group_index": 0, "abstract": "Atlas inspection"}]})
    replies.append({"valid": True})
    llm = Mock()
    llm.complete_json.side_effect = replies
    result = reconcile_event_drafts(
        [event(), event(content="Other detail")], "Original source", llm
    )
    assert len(result) == 2
    assert getattr(result[0], field) is None
    assert response == original
    assert llm.complete_json.call_args.kwargs["schema"]["required"] == ["valid"]
    if summary:
        assert original["groups"][0]["abstract"] in result[0].content


def test_empty_context_does_not_bypass_fact_rejection():
    response = proposal([[0], [1]])
    response["groups"][0]["where"] = ""
    llm = Mock()
    llm.complete_json.side_effect = [response, {"valid": False}, {"repair": None}]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts([event(), event(content="Other")], "source", llm)
    assert llm.complete_json.call_count == 3


@pytest.mark.parametrize(
    "bad", [{"members": [0, 0]}, {"where": " "}, {"where": 4}, {"content": ""}, {"title": ""}]
)
def test_other_malformed_fields_still_rejected_before_verifier(bad):
    response = proposal([[0, 1]])
    response["groups"][0]["when"] = ""
    response["groups"][0].update(bad)
    llm = Mock()
    llm.complete_json.return_value = response
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts([event(), event(content="Other")], "source", llm)
    assert llm.complete_json.call_count == 1


@pytest.mark.parametrize("verify", [True, False])
def test_source_partition_empty_context_and_summary_keep_fact_gate(verify):
    llm = SummaryLLM(singleton=True, verify=verify, corrupt=lambda group: group.update(where=""))
    api, fs = writer(llm)
    if not verify:
        with pytest.raises(ExtractionIncompleteError):
            api.commit_session(
                messages(), CTX, session_time=TIME, archive_id="source-empty-context"
            )
        assert fs.nodes == {}
    else:
        result = api.commit_session(
            messages(), CTX, session_time=TIME, archive_id="source-empty-context"
        )
        assert result["writes_completed"] == 2 and result["writes_failed"] == 0
    assert llm.repair_calls == 1
    assert llm.returned == llm.original
