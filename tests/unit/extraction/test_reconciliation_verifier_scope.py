"""Verify the changed members without making unrelated singletons merge obligations."""
import json
from copy import deepcopy
from unittest.mock import Mock

from extraction.event_reconciliation import reconcile_event_drafts
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_event_occurrence_reconciliation import proposal
from tests.unit.extraction.test_source_occurrence_reconciliation import (
    SourceLLM,
    extract,
    messages,
    writer,
)


def test_event_verification_keeps_relevant_facts_and_original_write_indices():
    drafts = [event(content='Unrelated receipt'), event(content='Pressure 5 bar'),
              event(content='Unrelated holiday'), event(content='Seal leak')]
    response = proposal([[0], [1, 3], [2]])
    before = deepcopy(response)
    llm = Mock()
    llm.complete_json.side_effect = [response, {'valid': True}]
    result = reconcile_event_drafts(drafts, 'Original complete source', llm)
    text = llm.complete_json.call_args_list[1].args[0]
    payload = json.loads(text[text.index('{"source":'):])
    assert [d['content'] for d in payload['member_drafts']] == ['Pressure 5 bar', 'Seal leak']
    assert 'members' not in payload['proposed_result']
    assert payload['source'] == 'Original complete source'
    assert result[0] is drafts[0] and result[2] is drafts[2]
    assert len(result) == 3 and response == before


def test_source_verification_reindexes_members_without_changing_source_evidence():
    # Four candidates across two sources; one untouched singleton on either side.
    llm = SourceLLM(groups=[[0], [1, 2], [3]], outputs_per_run=(2, 2))
    api, _ = writer(llm)
    candidates = extract(api, messages())
    verification = next(call for call in llm.json_calls if call[0] == {'valid'})
    payload = json.loads(verification[1].split('\n\n', 1)[1])
    assert len(candidates) == 3
    assert len(payload['member_drafts']) == 2
    assert [draft['source_index'] for draft in payload['member_drafts']] == [0, 1]
    assert 'members' not in payload['proposed_result']
    assert [item['member'] for item in payload['proposed_result']['identity_evidence']] == [0, 1]
    assert len(payload['sources']) == 2
