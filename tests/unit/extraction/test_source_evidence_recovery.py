import json
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


class EvidenceLLM(SourceLLM):
    def __init__(self, *, repair=True, verify=True):
        super().__init__(verify=verify)
        self.repair, self.repair_calls = repair, 0

    def complete_json(self, prompt, schema):
        if set(schema['properties']) == {'repairs'}:
            self.repair_calls += 1
            payload = json.loads(prompt.split('\n\n', 1)[1])
            repairs = []
            for index in payload['repair_group_indices']:
                group = payload['groups'][index]
                evidence = deepcopy(group['identity_evidence'])
                for item in evidence:
                    original = payload['sources'][item['source_index']]['selected_user_messages'][0]
                    item['quote'] = original['content'] if self.repair else 'invented quote'
                repairs.append({'group_index': index, 'identity_evidence': evidence})
            return {'repairs': repairs}
        response = super().complete_json(prompt, schema)
        if set(schema['properties']) == {'groups'}:
            response['groups'][0]['identity_evidence'][0]['quote'] = 'Two excerpts joined instead of a continuous original quote'
        return response


def test_invalid_quote_has_one_bounded_recovery_before_public_write():
    llm = EvidenceLLM()
    api, fs = writer(llm)
    result = api.commit_session(messages(), CTX, session_time=TIME, archive_id='recovery-source')
    assert llm.repair_calls == 1
    assert result['writes_completed'] == 1 and result['writes_failed'] == 0
    assert len(fs.nodes) == 1


@pytest.mark.parametrize('repair,verify', [(False, True), (True, False)])
def test_invalid_repair_or_independent_verifier_still_rejects_without_writes(repair, verify):
    llm = EvidenceLLM(repair=repair, verify=verify)
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id='recovery-source')
    assert llm.repair_calls == 1 and fs.nodes == {}


@pytest.mark.parametrize('corrupt', [
    lambda response: {'repairs': []},
    lambda response: {'repairs': response['repairs'] * 2},
    lambda response: {'repairs': [{**response['repairs'][0], 'group_index': True}]},
    lambda response: {'repairs': [{**response['repairs'][0], 'group_index': 99}]},
    lambda response: {'repairs': [{**response['repairs'][0], 'content': 'new facts'}]},
    lambda response: {'repairs': [{**response['repairs'][0], 'identity_evidence': []}]},
])
def test_corrupt_quote_repair_does_not_write_or_retry_forever(corrupt):
    llm = EvidenceLLM()
    original = llm.complete_json

    def complete(prompt, schema):
        result = original(prompt, schema)
        return corrupt(result) if set(schema['properties']) == {'repairs'} else result

    llm.complete_json = complete
    api, fs = writer(llm)
    with pytest.raises(ExtractionIncompleteError):
        api.commit_session(messages(), CTX, session_time=TIME, archive_id='recovery-source')
    assert llm.repair_calls == 1 and fs.nodes == {}
