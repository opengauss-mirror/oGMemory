"""A rejected proposal is never written; a bounded correction needs a fresh verdict."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.errors import ExtractionIncompleteError
from extraction.event_reconciliation import reconcile_event_drafts
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_event_occurrence_reconciliation import proposal


def setup_recovery(repair_mutation=None, final=True):
    drafts = [event(content='Pressure 5 bar'), event(content='Seal leak')]
    partition = proposal([[0, 1]])
    fields = {k: v for k, v in partition['groups'][0].items() if k != 'members'}
    fields['content'] = 'Pressure 5 bar; seal leak. No technician was mentioned.'
    if repair_mutation:
        repair_mutation(fields)
    llm = Mock()
    llm.complete_json.side_effect = [partition, {'valid': False}, {'repair': fields}, {'valid': final}]
    return drafts, partition, fields, llm


def test_rejected_event_can_recover_only_after_fresh_fact_check():
    drafts, partition, fields, llm = setup_recovery()
    original = deepcopy(partition)
    result = reconcile_event_drafts(drafts, 'USER: Pressure 5 bar and seal leak.', llm)
    assert len(result) == 1 and result[0].content == fields['content']
    assert llm.complete_json.call_count == 4 and partition == original
    assert drafts[0].content == 'Pressure 5 bar'
    prompt = llm.complete_json.call_args_list[-1].args[0]
    assert fields['content'] in prompt


@pytest.mark.parametrize('final', [False, None, 'true', 1])
def test_a_second_rejection_never_accepts_or_retries_again(final):
    drafts, _, _, llm = setup_recovery(final=final)
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 4


@pytest.mark.parametrize('mutation', [
    lambda fields: fields.update(members=[1]),
    lambda fields: fields.update(identity_evidence=[]),
    lambda fields: fields.update(abstract='x' * 201),
    lambda fields: fields.update(who=3),
    lambda fields: fields.pop('content'),
])
def test_fact_repair_cannot_edit_identity_or_bypass_shape(mutation):
    drafts, _, _, llm = setup_recovery(mutation)
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 3


def test_non_boolean_initial_verdict_is_not_a_repairable_rejection():
    drafts, partition, _, llm = setup_recovery()
    llm.complete_json.side_effect = [partition, {'valid': 'false'}]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 2


def test_all_groups_must_pass_before_any_replacement_is_returned():
    drafts = [event(content=f'Original {i}') for i in range(4)]
    original = deepcopy(drafts)
    llm = Mock()
    llm.complete_json.side_effect = [proposal([[0, 1], [2, 3]]), {'valid': True},
                                     {'valid': False}, {'repair': None}]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert drafts == original and llm.complete_json.call_count == 4


def test_repair_budget_is_checked_before_spending_again(monkeypatch):
    from extraction import event_reconciliation as module
    drafts, _, _, llm = setup_recovery()
    budget = iter([True, False])
    monkeypatch.setattr(module, 'prompt_fits', lambda *_: next(budget))
    with pytest.raises(ExtractionIncompleteError, match='complete context budget'):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 2


def test_corrected_content_budget_is_checked_before_final_verdict(monkeypatch):
    from extraction import event_reconciliation as module
    drafts, _, _, llm = setup_recovery()
    budget = iter([True, True, False])
    monkeypatch.setattr(module, 'prompt_fits', lambda *_: next(budget))
    with pytest.raises(ExtractionIncompleteError, match='complete context budget'):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 3


def test_source_repair_preserves_each_members_evidence_and_source_indices():
    from tests.unit.extraction.test_source_occurrence_reconciliation import (
        SourceLLM,
        extract,
        messages,
        writer,
    )

    class RepairLLM(SourceLLM):
        def complete_json(self, prompt, schema):
            properties = set(schema['properties'])
            if properties == {'valid'}:
                self.json_calls.append((properties, prompt, schema))
                return {'valid': len([c for c in self.json_calls if c[0] == {'valid'}]) == 2}
            if properties == {'repair'}:
                self.json_calls.append((properties, prompt, schema))
                import json
                result = json.loads(prompt.split('\n\n', 1)[1])['proposed_result']
                corrected = {key: value for key, value in result.items() if key != 'identity_evidence'}
                corrected['title'] = 'Nora inspects Atlas'
                return {'repair': corrected}
            return super().complete_json(prompt, schema)

    llm = RepairLLM()
    api, _ = writer(llm)
    result = extract(api, messages())
    import json
    checks = [json.loads(c[1].split('\n\n', 1)[1]) for c in llm.json_calls if c[0] == {'valid'}]
    assert len(result) == 1 and len(checks) == 2
    assert checks[0]['sources'] == checks[1]['sources']
    assert checks[0]['member_drafts'] == checks[1]['member_drafts']
    evidence = checks[1]['proposed_result']['identity_evidence']
    assert evidence == checks[0]['proposed_result']['identity_evidence']
    assert [e['source_index'] for e in evidence] == [0, 1]


def test_unchanged_repair_does_not_turn_a_rejection_into_a_second_vote():
    drafts = [event(content='Pressure 5 bar'), event(content='Seal leak')]
    partition = proposal([[0, 1]])
    fields = {key: value for key, value in partition['groups'][0].items() if key != 'members'}
    llm = Mock()
    llm.complete_json.side_effect = [partition, {'valid': False}, {'repair': fields}, {'valid': True}]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts(drafts, 'USER: complete source.', llm)
    assert llm.complete_json.call_count == 3


def test_fact_correction_keeps_rejected_original_as_reference_not_written_fact():
    drafts, partition, fields, llm = setup_recovery()
    appendix = '\n\n[Original abstract]\nAn unsupported technician repaired Atlas.'
    partition['groups'][0]['content'] += appendix
    before = deepcopy(partition)
    result = reconcile_event_drafts(drafts, 'USER: Pressure 5 bar; seal leak.', llm)
    assert result[0].content == fields['content']
    assert 'Original abstract' in llm.complete_json.call_args_list[-2].args[0]
    assert 'Original abstract' not in llm.complete_json.call_args_list[-1].args[0]
    assert partition == before  # The rejected proposal is retained, never mutated.
