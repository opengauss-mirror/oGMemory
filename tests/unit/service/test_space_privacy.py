"""Custom privacy phrases retain coverage across built-in redaction matches."""
import json

import pytest

from server.space_privacy import redact_text
from server.space_rules import validate_rule
from service.space_lifecycle import generate_plans


@pytest.mark.parametrize('text,terms,expected', [
    ('Alice <alice@example.org>', ['Alice <alice@example.org>'], '[已遮蔽]'),
    ('Alice [邮箱]', ['Alice [邮箱]'], '[已遮蔽]'),
    ('[邮箱] Alice [手机号]', ['邮箱', 'Alice [手机号]'], '[邮箱] [已遮蔽]'),
    ('Alice 13812345678', ['Alice 13812345678'], '[已遮蔽]'),
    ('Alice 110101199001011234', ['Alice 110101199001011234'], '[已遮蔽]'),
    ('alice@example.org', ['alice@'], '[邮箱]'),
    ('Alice <alice@example.org> Smith', ['Alice <alice@example.org>', 'alice@example.org> Smith'], '[已遮蔽]'),
    ('ALICE <ALICE@example.org>', ['Alice <alice@example.org>'], '[已遮蔽]'),
    ('[邮箱] alice@example.org 13812345678 Alice', ['邮箱', 'Alice'], '[邮箱] [邮箱] [手机号] [已遮蔽]'),
])
def test_custom_phrases_cover_sensitive_spans(text, terms, expected):
    policy = {'redact_terms': terms}
    result = redact_text(text, policy)
    print(json.dumps({'request': {'text': text, 'policy': policy}, 'response': result}, ensure_ascii=False))
    assert result == expected
    assert redact_text(result, policy) == result


@pytest.mark.parametrize('description', ['Summarize experience', 'Summarize Alice <alice@example.org>'])
def test_generation_redacts_source_topic_and_output(description):
    phrase = 'Alice <alice@example.org>'
    spec = validate_rule({'description': description, 'memory_types': ['case'],
                          'privacy_policy': {'redact_terms': [phrase]}})
    snapshot = {'rule': {'spec': spec}, 'sources': [{'uri': 'source:1', 'content': phrase}],
                'items': [{'item_id': 'memory-1', 'metadata': {'topic': phrase}, 'sources': []}]}

    class Model:
        def complete_json(self, prompt, schema):
            self.payload = json.loads(prompt.split('\n', 1)[1])
            return {'operations': [{'action': 'UPDATE', 'item_id': 'memory-1', 'topic': phrase,
                                    'abstract': phrase, 'content': phrase, 'quality_score': 0.9,
                                    'source_uris': [entry['uri'] for entry in self.payload['sources']]}]}

    model = Model()
    result = generate_plans(model, snapshot)
    print(json.dumps({'request': snapshot, 'model_request': model.payload, 'response': result}, ensure_ascii=False))
    assert phrase not in json.dumps(model.payload, ensure_ascii=False)
    assert model.payload['sources'][0]['content'] == '[已遮蔽]'
    assert model.payload['existing'][0]['topic'] == '[已遮蔽]'
    assert {key: result[0][key] for key in ('topic', 'abstract', 'content')} == dict.fromkeys(
        ('topic', 'abstract', 'content'), '[已遮蔽]')


def test_generation_uses_anonymous_source_references():
    uris = ['ctx://tenant/users/alice@example.org/memories/profile',
            'ctx://tenant/users/13812345678/memories/profile']
    spec = validate_rule({'description': 'Summarize experience', 'memory_types': ['case']})
    snapshot = {'rule': {'spec': spec},
                'sources': [{'uri': uri, 'content': 'Pool recovery'} for uri in uris],
                'items': [{'item_id': 'memory-1', 'sources': [{'source_uri': uri} for uri in uris]}]}

    class Model:
        def complete_json(self, prompt, schema):
            self.payload = json.loads(prompt.split('\n', 1)[1])
            refs = [entry['uri'] for entry in self.payload['sources']]
            return {'operations': [{'action': 'UPDATE', 'item_id': 'memory-1', 'topic': 'Pool',
                                    'abstract': 'Recovery', 'content': 'Inspect waits', 'quality_score': 0.9,
                                    'source_uris': refs}]}

    model = Model()
    result = generate_plans(model, snapshot)
    print(json.dumps({'request': snapshot, 'model_request': model.payload, 'response': result}, ensure_ascii=False))
    assert [entry['uri'] for entry in model.payload['sources']] == ['source:0', 'source:1']
    assert model.payload['existing'][0]['source_uris'] == ['source:0', 'source:1']
    assert result[0]['source_uris'] == sorted(uris)


def test_keep_applies_current_privacy_policy():
    term = 'Northwind Research'
    source = {'uri': 'ctx://tenant/users/alice/memories/case/pool', 'content': f'{term}: inspect pool waits',
              'grant_rev': 1, 'content_rev': 2, 'sha256': 'source-hash', 'epoch': 'source-epoch'}
    contribution = {'source_uri': source['uri'], 'grant_rev': 1, 'source_content_rev': 2,
                    'source_hash': 'source-hash', 'source_epoch': 'source-epoch'}
    spec = validate_rule({'description': 'Summarize recovery steps', 'memory_types': ['case']})
    snapshot = {'rule': {'spec': spec}, 'sources': [source], 'items': []}

    class Model:
        def complete_json(self, prompt, schema):
            self.payload = json.loads(prompt.split('\n', 1)[1])
            if self.payload['existing']:
                operations = [{'action': 'KEEP', 'item_id': entry['item_id']} for entry in self.payload['existing']]
            else:
                operations = [{'action': 'ADD', 'source_uris': [self.payload['sources'][0]['uri']],
                               'quality_score': 0.9, 'topic': 'Pool recovery', 'abstract': 'Inspect waits',
                               'content': self.payload['sources'][0]['content']}]
            self.response = {'operations': operations}
            return self.response

    model = Model()
    [old] = generate_plans(model, snapshot)
    assert old['content'] == source['content']
    snapshot['items'] = [
        dict(old, item_id='old', rule_rev=1, sources=[contribution]),
        dict(old, item_id='clean', rule_rev=1, content='Inspect pool waits', sources=[contribution]),
    ]
    snapshot['rule'] = {'rule_rev': 2, 'spec': validate_rule({**spec, 'privacy_policy': {'redact_terms': [term]}})}
    result = generate_plans(model, snapshot)
    print(json.dumps({'request': snapshot, 'model_request': model.payload,
                      'model_response': model.response, 'response': result}, ensure_ascii=False))
    assert result == [
        dict(old, action='UPDATE', item_id='old', content='[已遮蔽]: inspect pool waits'),
        {'action': 'KEEP', 'item_id': 'clean'},
    ]
