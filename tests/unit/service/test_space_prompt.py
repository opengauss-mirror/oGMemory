"""Evidence budgets and topic identity in team-memory generation."""
import json
import time

import pytest

from server.space_privacy import redact_text
from server.space_rules import validate_rule
from service.space_lifecycle import generate_plans


class CaptureModel:
    def complete_json(self, prompt, schema):
        self.prompt = prompt
        self.payload = json.loads(prompt.split('\n', 1)[1])
        return {'operations': [{'action': 'ARCHIVE', 'item_id': item['item_id']}
                               for item in self.payload['existing']]}


def test_long_email_candidate_redaction():
    text = '_' * 30000
    started = time.perf_counter()
    result = redact_text(text)
    elapsed = time.perf_counter() - started
    print(json.dumps({'request': {'character': '_', 'count': len(text)},
                      'response': {'characters': len(result), 'seconds': elapsed}}))
    assert result == text
    assert elapsed < 0.5
    assert redact_text('Contact first_last+tag@example.org.') == 'Contact [邮箱].'


def test_existing_topics_reach_model_redacted():
    items = [{'item_id': identity, 'metadata': {'topic': topic}, 'sources': []}
             for identity, topic in [('pool', 'Pool alice@example.org'), ('index', 'Index tuning')]]
    snapshot = {'rule': {'spec': validate_rule({'description': 'Reuse experience', 'memory_types': ['case']})},
                'sources': [{'uri': 'source:1', 'content': 'Pool and index advice'}], 'items': items}
    model = CaptureModel()
    result = generate_plans(model, snapshot)
    print(json.dumps({'request': snapshot, 'model_request': model.payload, 'response': result}))
    assert [item.get('topic') for item in model.payload['existing']] == ['Pool [邮箱]', 'Index tuning']


@pytest.mark.parametrize('pattern', ['\x00', '\"\\\n\t\x00中文'])
def test_escaped_evidence_fits_prompt_budget(pattern):
    snapshot = {'rule': {'spec': validate_rule({'description': 'Reuse experience', 'memory_types': ['case']})},
                'sources': [{'uri': f'source:{i}', 'content': pattern * 4000}
                            for i in range(20)], 'items': []}
    model = CaptureModel()
    try:
        result = generate_plans(model, snapshot)
    except ValueError as error:
        print(json.dumps({'request': {'sources': 20, 'content_pattern': pattern, 'repetitions': 4000},
                          'response': {'error': str(error)}}))
        raise
    print(json.dumps({'request': {'sources': 20, 'content_pattern': pattern, 'repetitions': 4000},
                      'response': result, 'prompt_bytes': len(model.prompt.encode()),
                      'evidence_bytes': [len(json.dumps(s, ensure_ascii=False).encode()) for s in model.payload['sources']]}))
    assert len(model.prompt.encode()) <= 32000
    assert all(source['content'] for source in model.payload['sources'])


def test_topics_share_budget_with_evidence():
    snapshot = {'rule': {'spec': validate_rule({'description': 'Rule ' * 3000, 'memory_types': ['case']})},
                'sources': [{'uri': 'source:1', 'content': 'Advice ' * 1000}],
                'items': [{'item_id': str(i), 'metadata': {'topic': '主题' * 256}, 'sources': []}
                          for i in range(100)]}
    model = CaptureModel()
    assert len(generate_plans(model, snapshot)) == 100
    assert len(model.prompt.encode()) <= 32000
    assert all(item['topic'] for item in model.payload['existing'])
    assert model.payload['sources'][0]['content']
