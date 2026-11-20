from contextlib import ExitStack
from copy import deepcopy

import pytest

from service.migration.engine import inventory
from service.migration.merge import MergePlan, PolicyConflictResolver
from service.migration.package import PackageError
from tests.unit.service.migration.test_engine import archive, node


def inventories(tmp_path, left, right):
    stack = ExitStack()
    target = stack.enter_context(inventory(tmp_path / 'target.db', lambda k: left.get(k, [])))
    source = stack.enter_context(inventory(tmp_path / 'source.db', lambda k: right.get(k, [])))
    return stack, target, source


def test_new_and_identical_nodes_preserve_owners(tmp_path):
    existing, new = node(), node()
    new['uri'] = new['uri'].replace('alice', 'bob')
    new['owner_space'] = 'user:bob'
    stack, target, source = inventories(tmp_path, {'nodes': [existing]}, {'nodes': [existing, new]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, lambda *_: pytest.fail('no conflict'))
        try:
            assert plan.counts == {'inserted': 1, 'merged': 0, 'skipped': 1}
            assert list(plan.changes('nodes')) == [new]
            assert len(list(plan.inventory.records('nodes'))) == 2
        finally:
            plan.close()


def test_archive_conflict_rejects_without_target_changes(tmp_path):
    old, incoming = archive(), archive()
    incoming['messages'] = [{'content': 'different'}]
    stack, target, source = inventories(tmp_path, {'archives': [old]}, {'archives': [incoming]})
    with stack:
        before = target.summary()
        with pytest.raises(PackageError, match='conflicting archives'):
            MergePlan(tmp_path / 'plan.db', target, source, None)
        assert target.summary() == before


def test_real_append_policy_remaps_relations_and_is_idempotent(tmp_path):
    old = node()
    old.update(category='event', uri='ctx://acme/users/alice/memories/events/meeting')
    incoming = deepcopy(old)
    incoming['content'] = 'A different meeting'
    other = node()
    relation = dict(from_uri=incoming['uri'], to_uri=other['uri'], relation_type='RELATED_TO', weight=1.0, reason='related')
    stack, target, source = inventories(tmp_path, {'nodes': [old, other]},
                                         {'nodes': [incoming, other], 'relations': [relation]})
    with stack:
        resolver = PolicyConflictResolver(target, llm=None)
        plan = MergePlan(tmp_path / 'plan.db', target, source, resolver)
        try:
            changed = list(plan.changes('nodes'))
            assert len(changed) == 1
            assert changed[0]['uri'] != old['uri']
            assert list(plan.changes('relations'))[0]['from_uri'] == changed[0]['uri']
            assert len(list(plan.inventory.records('nodes'))) == 3
            receipts = dict(plan.receipts())
            refreshed = {kind: list(source.records(kind)) for kind in ('nodes', 'archives', 'relations')}
            for record in refreshed['nodes']:
                record['version'] += 1
                record['updated_at'] = '2026-11-18T00:00:00+00:00'
            new_source = stack.enter_context(inventory(tmp_path / 'refreshed.db', lambda k: refreshed[k]))
            repeated = MergePlan(tmp_path / 'repeat.db', plan.inventory, new_source,
                                 lambda *_: pytest.fail('receipt must avoid policy'), applied=receipts.get)
            try:
                assert not list(repeated.changes('nodes'))
                assert not list(repeated.changes('relations'))
            finally:
                repeated.close()
        finally:
            plan.close()


def test_real_topic_policy_merges_content_and_preserves_metadata(tmp_path):
    old, incoming = node(), node()
    old['metadata']['shared_source_epoch'] = 'target-incarnation'
    incoming['metadata']['shared_source_epoch'] = 'source-incarnation'
    incoming['content'] = '另一条偏好'
    # ArchiveBuilder uses its established concatenation fallback without an LLM.
    stack, target, source = inventories(tmp_path, {'nodes': [old]}, {'nodes': [incoming]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, llm=None))
        try:
            merged = list(plan.changes('nodes'))[0]
            assert old['content'] in merged['content']
            assert incoming['content'] in merged['content']
            assert merged['metadata']['custom'] is True
            assert merged['metadata']['shared_source_epoch'] == 'target-incarnation'
            assert merged['version'] == old['version'] + 1
        finally:
            plan.close()


def test_event_storage_revision_is_not_a_content_conflict(tmp_path):
    old = node()
    old.update(category='event', uri='ctx://acme/users/alice/memories/events/meeting')
    incoming = {**old, 'version': old['version'] + 8,
                'updated_at': '2026-11-18T00:00:00+00:00'}
    stack, target, source = inventories(tmp_path, {'nodes': [old]}, {'nodes': [incoming]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, llm=None))
        try:
            assert plan.counts['skipped'] == 1
            assert not list(plan.changes('nodes'))
            assert list(plan.inventory.records('nodes')) == [old]
        finally:
            plan.close()


def test_receipt_is_stable_across_source_revisions_but_keeps_business_time():
    from service.migration.merge import fingerprint

    original = node()
    revised = {**original, 'version': 123, 'updated_at': '2026-11-18T00:00:00+00:00'}
    assert fingerprint(original) == fingerprint(revised)
    revised['metadata'] = {**original['metadata'], 'when': '2026-11-18'}
    assert fingerprint(original) != fingerprint(revised)


def test_merge_keeps_both_target_and_remapped_source_provenance(tmp_path):
    from core.provenance_resolver import ProvenanceResolver

    old_event = node()
    old_event.update(category='event', uri='ctx://acme/users/alice/memories/events/meeting')
    new_event = {**old_event, 'content': 'Different occurrence'}
    old_topic = node()
    pid = ProvenanceResolver.build_id('memory', old_event['uri'])
    old_topic['metadata']['provenance_ids'] = [pid]
    new_topic = {**old_topic, 'content': 'Different preference'}
    stack, target, source = inventories(tmp_path, {'nodes': [old_event, old_topic]},
                                         {'nodes': [new_event, new_topic]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, llm=None))
        try:
            changed = list(plan.changes('nodes'))
            event = next(r for r in changed if r['category'] == 'event')
            topic = next(r for r in changed if r['category'] == 'preference')
            assert set(topic['metadata']['provenance_ids']) == {
                pid, ProvenanceResolver.build_id('memory', event['uri'])}
        finally:
            plan.close()


def test_common_directory_is_rebuilt_from_both_child_sets(tmp_path):
    parent = node()
    parent.update(uri='ctx://acme/users/alice/memories/preferences/', category='preferences', content='',
                  abstract='Original summary', overview='Old directory')
    incoming_parent = {**parent, 'abstract': 'Source summary', 'overview': 'Source directory'}
    old, new = node(), node()
    new.update(uri=new['uri'].replace('/editor', '/keyboard'), abstract='Keyboard preference')
    stack, target, source = inventories(tmp_path, {'nodes': [parent, old]},
                                         {'nodes': [incoming_parent, new]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, llm=None))
        try:
            directory = next(r for r in plan.changes('nodes') if r['uri'].endswith('/'))
            assert directory['metadata']['child_count'] == 2
            assert old['abstract'] in directory['overview']
            assert new['abstract'] in directory['overview']
            assert len(list(plan.inventory.records('nodes'))) == 3
        finally:
            plan.close()


def test_directory_counts_children_without_abstract_beyond_summary_limit(tmp_path):
    parent = node()
    parent.update(uri='ctx://acme/users/alice/memories/preferences/', category='preferences', content='')
    children = [{**node(), 'uri': parent['uri'] + f'item-{index:03}',
                 'abstract': '' if index == 50 else f'Preference {index}'} for index in range(51)]
    stack, target, source = inventories(tmp_path, {}, {'nodes': [parent, *children]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, llm=None))
        try:
            directory = next(r for r in plan.changes('nodes') if r['uri'].endswith('/'))
            assert directory['metadata']['child_count'] == 51
            assert directory['metadata']['categories'] == ['preference']
        finally:
            plan.close()


def test_public_write_identity_history_survives_migration_conflict(tmp_path):
    from dataclasses import replace

    from core.models import RequestContext
    from service.api import MemoryWriteAPI
    from service.migration.records import normalize_node
    from tests.unit.commit.test_memory_identity import MemoryFS, memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    ctx = RequestContext('acme', 'alice', 'main', 'producer', 'producer')
    records = []
    for content in ('Use SQL explanations.', 'Keep database backups.'):
        fs = MemoryFS()
        writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
        result = writer.write_memory(replace(memory('editor', category='preference', content=content),
                                             provenance_ids=[]), ctx)
        produced = fs.read_node(result['target_uri'], ctx)
        record = node()
        record.update({key: getattr(produced, key) for key in
                       ('uri', 'owner_space', 'context_type', 'category', 'level',
                        'content', 'abstract', 'overview', 'metadata')})
        records.append(normalize_node(record))
    stack, target, source = inventories(tmp_path, {'nodes': [records[0]]}, {'nodes': [records[1]]})
    with stack:
        plan = MergePlan(tmp_path / 'merged.db', target, source, PolicyConflictResolver(target, None))
        try:
            merged = list(plan.changes('nodes'))[0]
            assert all(record['content'] in merged['content'] for record in records)
            observations = merged['metadata']['memory_observations']
            assert len(observations) == 2
            assert set(observations) == {v for r in records for v in r['metadata']['memory_observations']}
        finally:
            plan.close()


@pytest.mark.parametrize('field', ['memory_observation_sources', 'occurrence_replay_sources'])
@pytest.mark.parametrize('same_observation', [False, True])
def test_target_evidence_keeps_original_event_after_source_remapping(tmp_path, field, same_observation):
    from core.provenance_resolver import ProvenanceResolver

    old_event = node()
    old_event.update(category='event', uri='ctx://acme/users/alice/memories/events/meeting')
    new_event = {**old_event, 'content': 'Different occurrence'}
    old_topic = node()
    pid = ProvenanceResolver.build_id('memory', old_event['uri'])
    old_topic['metadata'].update(provenance_ids=[pid], memory_observations=['old'])
    old_topic['metadata'][field] = {'old': [pid]}
    new_topic = deepcopy(old_topic)
    new_topic['content'] = 'Different preference'
    observation = 'old' if same_observation else 'new'
    new_topic['metadata']['memory_observations'] = [observation]
    new_topic['metadata'][field] = {observation: [pid]}
    stack, target, source = inventories(tmp_path, {'nodes': [old_event, old_topic]},
                                       {'nodes': [new_event, new_topic]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, PolicyConflictResolver(target, None))
        try:
            changed = list(plan.changes('nodes'))
            event = next(r for r in changed if r['category'] == 'event')
            topic = next(r for r in changed if r['category'] == 'preference')
            remapped = ProvenanceResolver.build_id('memory', event['uri'])
            sources = topic['metadata'][field]
            assert sources[observation] == ([pid, remapped] if same_observation else [remapped])
            assert pid in sources['old']
            receipts = dict(plan.receipts())
            repeated = MergePlan(tmp_path / 'repeat.db', plan.inventory, source,
                                 lambda *a: pytest.fail('must not repeat merge'), applied=receipts.get)
            try:
                assert repeated.inventory.summary() == plan.inventory.summary()
            finally:
                repeated.close()
        finally:
            plan.close()


def test_archive_storage_timestamp_difference_is_unchanged(tmp_path):
    old, incoming = archive(), archive()
    incoming['created_at'] = '2026-02-01T00:00:00+00:00'
    stack, target, source = inventories(tmp_path, {'archives': [old]}, {'archives': [incoming]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, None)
        try:
            assert plan.counts['skipped'] == 1
            assert list(plan.changes('archives')) == []
            assert list(plan.inventory.records('archives'))[0]['created_at'] == old['created_at']
        finally:
            plan.close()


def test_archive_custom_metadata_is_not_a_memory_reference(tmp_path):
    incoming = archive()
    incoming['metadata']['operator_note'] = {'values': ['prov:manual', 'ctx://acme/users/alice/memories/events/1']}
    stack, target, source = inventories(tmp_path, {}, {'archives': [incoming]})
    with stack:
        plan = MergePlan(tmp_path / 'plan.db', target, source, lambda *_: pytest.fail('no conflict'))
        try:
            assert list(plan.inventory.records('archives'))[0]['metadata'] == incoming['metadata']
        finally:
            plan.close()
