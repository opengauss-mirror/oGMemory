"""Bounded summary recovery must retain source identity and fact verification."""

from copy import deepcopy

import pytest

from core.errors import ExtractionIncompleteError
from tests.unit.extraction.test_source_occurrence_reconciliation import (
    CTX,
    TIME,
    SourceLLM,
    messages,
    writer,
)

LONG_SUMMARY = "Nora inspected Atlas once, measuring pressure and observing its seal. " * 4


class SummaryLLM(SourceLLM):
    def __init__(self, *, singleton=False, verify=True, corrupt=None, repair=None):
        super().__init__(groups=[[0], [1]] if singleton else None, verify=verify)
        self.corrupt, self.repair = corrupt, repair
        self.repair_calls = 0
        self.original = None

    def complete_json(self, prompt, schema):
        if set(schema["properties"]) == {"repairs"}:
            self.repair_calls += 1
            return self.repair if self.repair is not None else {
                "repairs": [{"group_index": 0, "abstract": "Atlas inspection"}]}
        response = super().complete_json(prompt, schema)
        if set(schema["properties"]) == {"groups"}:
            response["groups"][0]["abstract"] = LONG_SUMMARY
            if self.corrupt:
                self.corrupt(response["groups"][0])
            self.original = deepcopy(response)
            self.returned = response
        return response


@pytest.mark.parametrize("singleton", [False, True])
def test_source_summary_recovery_preserves_text_and_public_write(singleton):
    llm = SummaryLLM(singleton=singleton)
    api, fs = writer(llm)
    result = api.commit_session(messages(), CTX, session_time=TIME, archive_id="summary-source")
    assert result["writes_completed"] == (2 if singleton else 1)
    assert result["writes_failed"] == 0
    assert llm.repair_calls == 1
    assert any(LONG_SUMMARY in node.content for node in fs.nodes.values())
    assert llm.returned == llm.original
    assert [call[0] for call in llm.json_calls] == [{"spans"}, {"groups"}, {"valid"}]


@pytest.mark.parametrize("singleton", [False, True])
def test_summary_recovery_never_bypasses_fact_rejection(singleton):
    llm = SummaryLLM(singleton=singleton, verify=False)
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="summary-source")
    assert llm.repair_calls == 1 and fs.nodes == {}
    assert [call[0] for call in llm.json_calls][-2:] == [{"valid"}, {"repair"}]


@pytest.mark.parametrize("corrupt", [
    lambda group: group.update(members=[0, 0]),
    lambda group: group.update(members=[0, 4]),
    lambda group: group.update(title="x" * 201),
    lambda group: group.update(extra_identity="unsupported"),
    lambda group: group["identity_evidence"][0].update(quote="not in the user message"),
    lambda group: group["identity_evidence"][0].update(source_index=99),
])
def test_summary_recovery_spends_nothing_on_invalid_identity_or_other_fields(corrupt):
    llm = SummaryLLM(corrupt=corrupt)
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="summary-source")
    assert llm.repair_calls == 0 and fs.nodes == {}


@pytest.mark.parametrize("repair", [
    {"repairs": []},
    {"repairs": [{"group_index": 0, "abstract": "x" * 201}]},
    {"repairs": [{"group_index": 0, "abstract": "ok", "members": [0]}]},
])
def test_summary_repair_is_bounded_and_cannot_change_membership(repair):
    llm = SummaryLLM(repair=repair)
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id="summary-source")
    assert llm.repair_calls == 1 and fs.nodes == {}
