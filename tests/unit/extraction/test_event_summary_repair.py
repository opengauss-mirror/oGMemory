from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.errors import ExtractionIncompleteError
from extraction.event_reconciliation import reconcile_event_drafts
from extraction.occurrence_qualification import prompt_fits
from tests.unit.extraction.test_event_draft_boundaries import event
from tests.unit.extraction.test_event_occurrence_reconciliation import proposal


@pytest.mark.parametrize('singleton', [False, True])
def test_summary_repair_preserves_complete_original_and_requires_fact_check(singleton):
    response = proposal([[0], [1]] if singleton else [[0, 1]])
    original_summary = 'Supported detailed description ' * 9
    response['groups'][0]['abstract'] = original_summary
    before = deepcopy(response)
    llm = Mock()
    llm.complete_json.side_effect = [response, {'repairs':[{'group_index':0, 'abstract':'Atlas inspected'}]}, {'valid':True}]
    drafts = [event(), event(content='Gauge: 5 bar.')]
    result = reconcile_event_drafts(drafts, 'Nora inspected Atlas. Gauge: 5 bar.', llm)
    assert original_summary in result[0].content
    assert len(result) == (2 if singleton else 1)
    assert response == before
    assert llm.complete_json.call_count == 3


@pytest.mark.parametrize('repair', [{'repairs':[]}, {'repairs':[{'group_index':True, 'abstract':'ok'}]},
    {'repairs':[{'group_index':0, 'abstract':'ok', 'members':[0]}]},
    {'repairs':[{'group_index':0, 'abstract':''}]},
    {'repairs':[{'group_index':0, 'abstract':'x'*201}]}])
def test_invalid_repair_cannot_become_success(repair):
    response = proposal([[0, 1]])
    response['groups'][0]['abstract'] = 'x'*201
    llm = Mock()
    llm.complete_json.side_effect = [response, repair]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts([event(), event(content='Gauge: 5 bar.')], 'source', llm)
    assert llm.complete_json.call_count == 2


def test_repaired_summary_still_rejects_false_fact_verifier():
    response = proposal([[0, 1]])
    response['groups'][0]['abstract'] = 'x'*201
    llm = Mock()
    llm.complete_json.side_effect = [response, {'repairs':[{'group_index':0, 'abstract':'ok'}]}, {'valid':False}, {'repair':None}]
    with pytest.raises(ExtractionIncompleteError):
        reconcile_event_drafts([event(), event(content='Gauge: 5 bar.')], 'source', llm)


@pytest.mark.parametrize('repair_needed', [False, True])
def test_complete_repaired_content_is_budgeted_before_fact_verification(repair_needed):
    body1 = 'Nora measured pressure at Atlas. ' * 180
    body2 = 'Nora observed the Atlas seal. ' * 200
    source = ('Nora inspected Atlas once. ' * 180) + body1 + body2
    drafts = [event(content=body1), event(content=body2)]
    before = deepcopy(drafts)
    response = proposal([[0, 1]])
    response['groups'][0]['content'] = body1 + body2
    replies = [response]
    if repair_needed:
        response['groups'][0]['overview'] = body1
        replies.append({'repairs': [{'group_index': 0, 'overview': 'Nora inspected Atlas.'}]})
    replies.append({'valid': True})
    llm = Mock()
    llm.complete_json.side_effect = replies
    if repair_needed:
        with pytest.raises(ExtractionIncompleteError, match='complete context budget'):
            reconcile_event_drafts(drafts, source, llm)
    else:
        result = reconcile_event_drafts(drafts, source, llm)
        assert body1 + body2 == result[0].content
    assert drafts == before
    assert llm.complete_json.call_count == 2
    assert all(prompt_fits(call.args[0], call.kwargs['schema'])
               for call in llm.complete_json.call_args_list)
