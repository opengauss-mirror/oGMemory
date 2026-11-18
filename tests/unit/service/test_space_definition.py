"""Free-text team rules generate privacy-filtered reusable memory."""

import json
from copy import deepcopy

from server.space_privacy import redact_item, redact_text
from server.space_rules import digest, memory_definition, validate_rule
from service.space_lifecycle import generate_plans


def definition():
    return {'memory_types': ['case'], 'description': '总结连接池排查经验，自行组织适用场景、排查过程和结论。'}


def snapshot(spec, content='长事务占用连接。释放连接后检查等待队列，完成两次验证。'):
    return {'rule': {'spec': spec}, 'sources': [
        {'uri': 'ctx://acme/agents/ops/memories/case/pool', 'content': content}], 'items': []}


class PlanModel:
    def __init__(self, content, abstract='长事务导致的连接池等待', topic='connection-pool'):
        self.content, self.abstract, self.topic = content, abstract, topic
        self.calls = []

    def complete_json(self, prompt, schema):
        payload = json.loads(prompt.split('\n', 1)[1])
        self.calls.append((payload, deepcopy(schema)))
        return {'operations': [{
            'action': 'ADD', 'topic': self.topic, 'quality_score': 0.95,
            'abstract': self.abstract, 'content': self.content,
            'source_uris': [source['uri'] for source in payload['sources']],
        }]}


def test_text_rule_normalizes_privacy_and_system_defaults():
    requested = {**definition(), 'privacy_policy': {'redact_terms': [' 张三 ', '客户甲', '张三']}}
    normalized = validate_rule(requested)
    assert memory_definition(normalized) == {
        'description': requested['description'],
        'privacy_policy': {'redact_terms': ['客户甲', '张三']}}
    assert set(normalized) == {'memory_types', 'output_kind', 'description', 'privacy_policy', 'merge_policy',
                               'trigger_policy', 'maintenance_policy', 'quality_policy', 'output_acl'}
    assert normalized['output_kind'] == 'experience'
    assert normalized['merge_policy'] == 'by_topic'
    assert normalized['quality_policy'] == {'min_sources': 1, 'min_score': 0.8}
    assert normalized['output_acl'] == {'mode': 'inherit', 'readers': []}


def test_historical_definition_reads_and_saves_as_text_without_changing_snapshot():
    legacy = {'source_subjects': ['agent:ops'], 'memory_types': ['case'],
              'instructions': '保存排查步骤', 'output_kind': 'experience',
              'output_schema': {'fields': [{'name': 'steps', 'type': 'string_array', 'required': True}]}}
    before = deepcopy(legacy)
    original_hash = digest(legacy)
    expected = {'description': '保存排查步骤', 'privacy_policy': {'redact_terms': []}}
    assert memory_definition(legacy) == expected
    saved = validate_rule(legacy)
    assert memory_definition(saved) == expected
    assert saved['source_subjects'] == ['agent:ops']
    assert legacy == before
    assert digest(legacy) == original_hash


def test_generation_preserves_free_text_structure():
    spec = validate_rule(definition())
    content = '## 排查过程\n先观察等待队列，再定位长事务。\n\n释放连接后，等待恢复正常。'
    model = PlanModel(content)
    plans = generate_plans(model, snapshot(spec))
    payload, schema = model.calls[0]
    assert payload['rule'] == {'description': spec['description'], 'output_kind': 'experience',
                              'merge_policy': spec['merge_policy'], 'quality_policy': spec['quality_policy']}
    output = schema['properties']['operations']['items']['properties']
    assert output['content'] == {'type': 'string'}
    assert plans[0]['content'] == content
    assert plans[0]['metadata'] == {'privacy_redacted': True}


def test_generation_redacts_evidence_and_published_text():
    spec = validate_rule({**definition(), 'privacy_policy': {'redact_terms': ['张三', '客户甲']}})
    content = '客户甲的张三报告连接池等待，邮箱zhang@example.com，手机号13812345678，身份证110101199003074518。释放长事务后恢复。'
    model = PlanModel(content, abstract='客户甲张三的连接池恢复', topic='客户甲-张三-connection-pool')
    plan = generate_plans(model, snapshot(spec, content))[0]
    expected = '[已遮蔽]的[已遮蔽]报告连接池等待，邮箱[邮箱]，手机号[手机号]，身份证[身份证号]。释放长事务后恢复。'
    assert model.calls[0][0]['sources'][0]['content'] == expected
    assert plan['content'] == expected
    assert plan['abstract'] == '[已遮蔽][已遮蔽]的连接池恢复'
    assert plan['topic'] == '[已遮蔽]-[已遮蔽]-connection-pool'
    assert plan['source_uris'] == ['ctx://acme/agents/ops/memories/case/pool']


def test_historical_snapshot_generates_current_text_content():
    spec = validate_rule(definition())
    legacy = {**spec, 'output_kind': 'skill', 'fields': [{'name': 'steps', 'type': 'list'}]}
    before = digest(legacy)
    model = PlanModel('释放长事务占用的连接，检查等待队列。')
    plan = generate_plans(model, snapshot(legacy))[0]
    assert plan['content'] == '释放长事务占用的连接，检查等待队列。'
    assert model.calls[0][0]['rule']['output_kind'] == 'experience'
    assert digest(legacy) == before


def test_read_redaction_preserves_routing_and_versions():
    policy = {'redact_terms': ['张三']}
    original = {'uri': 'ctx://acme/spaces/ops/memories/experience/id', 'version': 3,
                'content': '张三联系138 1234 5678', 'abstract': '邮件zhang@example.com',
                'metadata': {'fields': {'客户': '张三'}, 'topic': '张三故障', 'agfs_content': {'path': '/stored/item'}}}
    result = redact_item(original, policy)
    assert result['content'] == '[已遮蔽]联系[手机号]'
    assert result['abstract'] == '邮件[邮箱]'
    assert result['metadata']['fields'] == {'客户': '[已遮蔽]'}
    assert result['metadata']['agfs_content'] == original['metadata']['agfs_content']
    assert (result['uri'], result['version']) == (original['uri'], 3)
    assert original['metadata']['fields'] == {'客户': '张三'}
    assert redact_item(result, policy) == result


def test_redaction_supports_country_prefixes_and_literal_terms():
    policy = {'redact_terms': ['客户(甲)', 'ACME', '已遮蔽']}
    content = '客户(甲)与acme：+86 138-1234-5678；[已遮蔽]。业务恢复。'
    expected = '[已遮蔽]与[已遮蔽]：[手机号]；[已遮蔽]。业务恢复。'
    assert redact_text(content, policy) == expected
    assert redact_text(expected, policy) == expected
