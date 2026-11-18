"""Recovery and pagination through real team repository workflows."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import count
from threading import Event
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, Role
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server import space_job_repository
from server.api_keys import APIKeyManager
from server.space_rules import SpaceError
from server.sql_control_plane_store import SQLControlPlaneStore
from tests.integration.test_space_job_review import publish, snapshot
from tests.integration.test_space_repository_review import pool as pool
from tests.integration.test_space_repository_review import team as team


def test_admin_restores_space_after_owner_deletion(team, pool):
    repo, _, alice = team
    manager = APIKeyManager(SQLControlPlaneStore(pool=pool))
    manager.register_user(alice.account_id, 'admin', 'admin')
    admin = replace(alice, user_id='admin', role=Role.ADMIN)
    repo.create_space(alice, {'space_id': 'orphan', 'name': 'Owner recovery'})
    deleted = manager.delete_user(alice.account_id, 'alice')
    print('request', {'method': 'delete_user', 'user_id': 'alice'}, 'response', {'deleted': deleted})
    assert deleted
    members = {'members': [{'principal': 'user:alice', 'role': 'manager'},
                           {'principal': 'user:bob', 'role': 'manager'}]}
    print('request', {'method': 'set_members', 'space_id': 'orphan', 'body': members})
    try:
        updated = repo.set_members(admin, 'orphan', members, expected_rev=1)
    except SpaceError as error:
        print('response', {'status': error.status, 'code': error.code})
        raise
    print('response', updated)
    transferred = repo.update_space(admin, 'orphan', {'owner_user_id': 'bob'},
                                    expected_rev=updated['authz_rev'])
    print('request', {'method': 'update_space', 'owner_user_id': 'bob'}, 'response', transferred)
    final = repo.set_members(admin, 'orphan', {'members': [members['members'][1]]},
                             expected_rev=transferred['authz_rev'])
    space = repo.list_spaces(replace(alice, user_id='bob'), 'orphan')['space']
    print('request', {'method': 'list_spaces', 'user_id': 'bob'},
          'response', {'owner_user_id': space['owner_user_id'], 'members': final['members']})
    assert space['owner_user_id'] == 'bob'
    assert [member['principal'] for member in space['members']] == ['user:bob']


@pytest.mark.parametrize(('change', 'changed_count'), [
    ('source', 1), ('source', 3), ('acl', 1), ('acl', 3),
    ('privacy', 0), ('other_acl', 0), ('overlap', 0), ('overlap_inspect', 0),
])
def test_page_continues_after_reauthorization(team, pool, monkeypatch, change, changed_count):
    repo, bridge, alice = team
    repo.put_source(alice, 'ops', 'source', {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
    repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': {
        'description': 'Summarize each case', 'memory_types': ['case'],
        'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=0)
    overlap = change.startswith('overlap')
    prefix = 'ABC123' if overlap else 'Step'
    sources = {}
    for number in range(4):
        node = ContextNode(f'ctx://{alice.account_id}/agents/ops/memories/case/{number}',
                           'MEMORY', 'case', 2, 'agent:ops', f'Case {number}', '', f'{prefix} {number}')
        bridge.write_node(node, replace(alice, agent_id='ops'))
        sources[node.content] = node
    job, data = snapshot(repo, alice.account_id)
    generated = publish(repo, job, data, lambda payload: [
        {'action': 'ADD', 'topic': source['uri'], 'abstract': 'Operations',
         'content': source['content'], 'quality_score': 0.9, 'source_uris': [source['uri']]}
        for source in payload['sources']])
    assert len(generated['items']) == 4
    outbox = SQLOutboxStore(pool=pool, fs=bridge)
    worker = OutboxWorker(InMemoryVectorIndex(8), MockEmbedder(8), fs=bridge)
    assert worker.run_once(outbox, [alice.account_id])['succeeded'] == 4
    if overlap:
        result = repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': {
            'description': 'Summarize each case', 'memory_types': ['case'],
            'privacy_policy': {'redact_terms': ['ABC']},
            'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=1)
        print('request', {'method': 'save_rule', 'redact_terms': ['ABC']}, 'response', result)
    bob = replace(alice, user_id='alice' if change == 'overlap_inspect' else 'bob')
    loaded, proceed = Event(), Event()
    scanned = []
    load = bridge.load_content

    def paused_load(rows):
        hydrated = load(rows)
        scanned.extend(hydrated)
        loaded.set()
        assert proceed.wait(20)
        return hydrated

    with monkeypatch.context() as patch, ThreadPoolExecutor(1) as executor:
        patch.setattr(bridge, 'load_content', paused_load)
        pending = executor.submit(repo.list_items, bob, 'ops', limit=2)
        try:
            assert loaded.wait(20)
            if change == 'privacy' or overlap:
                terms = ['ABC', 'ABC123'] if overlap else ['Step']
                result = repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': {
                    'description': 'Summarize each case', 'memory_types': ['case'],
                    'privacy_policy': {'redact_terms': terms},
                    'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=2 if overlap else 1)
                print('request', {'method': 'save_rule', 'redact_terms': terms}, 'response', result)
            elif change == 'other_acl':
                unseen = next(item for item in generated['items']
                              if item['item_id'] not in {row['item_id'] for row in scanned})
                revision = repo.permissions(alice, 'ops')['authz_rev']
                result = repo.set_acl(alice, 'ops', {'item_id': unseen['item_id'],
                    'acl': {'mode': 'restricted', 'readers': ['user:alice']}}, expected_rev=revision)
                print('request', {'method': 'set_acl', 'unscanned_item': unseen['item_id']}, 'response', result)
            for item in scanned[:changed_count]:
                if change == 'source':
                    node = sources[item['content']]
                    bridge.write_node(replace(node, content=node.content + ' revised'), replace(alice, agent_id='ops'))
                else:
                    revision = repo.permissions(alice, 'ops')['authz_rev']
                    repo.set_acl(alice, 'ops', {'item_id': item['item_id'],
                                 'acl': {'mode': 'restricted', 'readers': ['user:alice']}}, expected_rev=revision)
            print('request', {'during_agfs_read': change, 'changed_items':
                              [item['item_id'] for item in scanned[:changed_count]]})
        finally:
            proceed.set()
        first = pending.result(timeout=20)
    expected = [item['item_id'] for item in repo.visible_items(bob, ['ops'], hydrate=False)]
    print('request', {'method': 'list_items', 'limit': 2}, 'response', first,
          'currently_readable', expected)
    if change == 'privacy' or overlap:
        assert [item['content'].split()[0] for item in first['items']] == ['[已遮蔽]'] * 2
    assert first['next_cursor']
    print('request', {'method': 'list_items', 'limit': 2, 'cursor': first['next_cursor']})
    try:
        second = repo.list_items(bob, 'ops', limit=2, cursor=first['next_cursor'])
    except SpaceError as error:
        print('response', {'status': error.status, 'code': error.code})
        raise
    print('request', {'method': 'list_items', 'cursor': first['next_cursor']}, 'response', second)
    assert [item['item_id'] for item in first['items'] + second['items']] == expected
    assert second['next_cursor'] is None
    if change == 'privacy' or overlap:
        assert [item['content'].split()[0] for item in second['items']] == ['[已遮蔽]'] * 2


@pytest.mark.parametrize('operation', ['transfer', 'archive', 'source'])
def test_role_change_applies_to_pending_space_management(team, pool, operation):
    repo, _, alice = team
    manager = APIKeyManager(SQLControlPlaneStore(pool=pool))
    repo.set_members(alice, 'ops', {'members': [
        {'principal': 'user:alice', 'role': 'manager'},
        {'principal': 'user:bob', 'role': 'manager'}, {'principal': 'agent:ops'}]}, expected_rev=2)
    assert manager.set_role(alice.account_id, 'bob', 'admin')
    identity = manager.resolve_user_key('bob', alice.account_id)
    pending_ctx = replace(alice, user_id=identity['user_id'], role=Role(identity['role'].upper()))
    assert manager.set_role(alice.account_id, 'bob', 'user')
    print('request', {'method': 'set_role', 'user_id': 'bob', 'role': 'user'},
          'response', manager.resolve_user_key('bob', alice.account_id))
    revision = repo.permissions(alice, 'ops')['authz_rev']
    payload = {'owner_user_id': 'bob'} if operation == 'transfer' else {'state': 'ARCHIVED'}
    def invoke(ctx):
        if operation == 'source':
            return repo.put_source(ctx, 'ops', 'source',
                {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
        return repo.update_space(ctx, 'ops', payload, expected_rev=revision)
    permissions = repo.permissions(pending_ctx, 'ops')
    members = repo.list_spaces(pending_ctx, 'ops')['space']['members']
    candidates = repo.list_principals(pending_ctx, kind='user')['items']
    flags = {'is_admin': permissions['is_admin'],
             'agent_can_contribute': next(m['can_contribute'] for m in members if m['principal'] == 'agent:ops'),
             'alice_can_contribute': next(m['can_contribute'] for m in candidates if m['id'] == 'alice')}
    try:
        response = {'status': 200, 'body': invoke(pending_ctx)}
    except SpaceError as error:
        response = {'status': error.status, 'code': error.code}
    print('request', {'operation': operation, 'authenticated_role': pending_ctx.role.value,
                      'current_role': 'user', 'payload': payload if operation != 'source' else
                      {'principal': 'agent:ops', 'memory_types': ['case']}},
          'response', response, 'capabilities', flags)
    code = 'SOURCE_OWNER_REQUIRED' if operation == 'source' else 'SPACE_OWNER_REQUIRED'
    assert response == {'status': 403, 'code': code}
    assert flags == {'is_admin': False, 'agent_can_contribute': False, 'alice_can_contribute': False}
    result = invoke(alice)
    print('request', {'operation': operation, 'user_id': 'alice'}, 'response', result)
    assert result


@pytest.mark.parametrize(('hydrate', 'revoke_pages'), [(True, 0), (False, 0), (True, 1), (True, 2)])
@pytest.mark.parametrize(('spaces', 'query'), [
    (('ops', 'support'), None), (('ops', 'support'), 'pool waits'), (('ops',), 'pool waits'),
])
def test_visible_pages_follow_item_order(team, pool, monkeypatch, hydrate, revoke_pages, spaces, query):
    repo, bridge, alice = team
    source = ContextNode(f'ctx://{alice.account_id}/agents/ops/memories/case/shared',
                         'MEMORY', 'case', 2, 'agent:ops', 'Shared case', '', 'Resolve pool waits.')
    bridge.write_node(source, replace(alice, agent_id='ops'))
    expected, published = [], []
    seed = uuid4().int & ((1 << 120) - 1)
    for space in spaces:
        if space == 'support':
            repo.create_space(alice, {'space_id': space, 'name': 'Support'})
            repo.set_members(alice, space, {'members': [
                {'principal': 'user:alice', 'role': 'manager'},
                {'principal': 'user:bob'}, {'principal': 'agent:ops'}]}, expected_rev=1)
        repo.put_source(alice, space, 'source', {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
        repo.save_rule(alice, space, 'notes', {'name': 'Notes', 'spec': {
            'description': 'Share the case', 'memory_types': ['case'],
            'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=0)
    job, data = snapshot(repo, alice.account_id)
    high, low = 'Resolve pool waits.', 'Resolve pool connection waits.'
    for _ in spaces:
        space = job['space_id']
        texts = [high, low, 'Resolve pool waits safely.', 'Diagnose disk errors.'] if len(spaces) == 1 else [high if space == 'ops' else low]
        ids = count(seed | (0xf << 124) if space == 'ops' else seed, -1)
        with monkeypatch.context() as patch:
            patch.setattr(space_job_repository, 'uuid', SimpleNamespace(uuid4=lambda: UUID(int=next(ids), version=4)))
            result = publish(repo, job, data, lambda payload: [
                {'action': 'ADD', 'topic': body, 'abstract': 'Operations',
                 'content': body, 'quality_score': 0.9,
                 'source_uris': [payload['sources'][0]['uri']]}
                for body in texts])
        print('request', {'method': 'publish', 'space_id': space, 'contents': texts}, 'response', result)
        for item, body in zip(result['items'], texts, strict=True):
            published.append(item)
            if query is None or body != 'Diagnose disk errors.':
                expected.append((space, item['item_id']))
        job = repo.claim_job(alice.account_id)
        if job:
            data = repo.job_snapshot(job)
    worker = OutboxWorker(InMemoryVectorIndex(8), MockEmbedder(8), fs=bridge)
    assert worker.run_once(SQLOutboxStore(pool=pool, fs=bridge), [alice.account_id])['succeeded'] == len(published)
    bob = replace(alice, user_id='bob')
    contents = {row['content']: row['item_id'] for row in repo.visible_items(bob, list(spaces))}
    assert contents[high] > contents[low]
    seen, scanned, revoked, after = [], [], [], ''
    load = bridge.load_content

    def revoke(row):
        payload = {'item_id': row['item_id'], 'acl': {'mode': 'restricted', 'readers': ['user:alice']}}
        result = repo.set_acl(alice, row['space_id'], payload,
                              expected_rev=repo.permissions(alice, row['space_id'])['authz_rev'])
        print('concurrent request', {'method': 'set_acl', 'space': row['space_id'], **payload}, 'response', result)
        revoked.append((row['space_id'], row['item_id']))

    def load_then_revoke(rows):
        hydrated = load(rows)
        scanned.extend((row['space_id'], row['item_id']) for row in hydrated)
        if hydrated and len(revoked) < revoke_pages:
            executor.submit(revoke, hydrated[0]).result(timeout=20)
        return hydrated

    with ThreadPoolExecutor(1) as executor, monkeypatch.context() as patch:
        patch.setattr(bridge, 'load_content', load_then_revoke)
        for _ in range(len(expected) + 1):
            rows = repo.visible_items(bob, list(spaces), query=query, limit=1, after=after, hydrate=hydrate)
            print('request', {'method': 'visible_items', 'spaces': spaces, 'query': query,
                              'limit': 1, 'after': after, 'hydrate': hydrate, 'revoke_pages': revoke_pages},
                  'response', [{'space_id': row['space_id'], 'item_id': row['item_id'],
                                'content': row.get('content')} for row in rows])
            seen.extend((row['space_id'], row['item_id']) for row in rows)
            if not rows:
                break
            after = rows[-1]['item_id']
    expected.sort(key=lambda item: item[1])
    remaining = [item for item in expected if item not in revoked]
    print('response summary', {'seen': seen, 'remaining': remaining, 'scanned': scanned})
    assert seen == remaining
    if hydrate:
        assert scanned == expected


@pytest.mark.parametrize('change', ['acl', 'pause', 'source', 'archive', 'privacy', 'overlap'])
def test_direct_read_reauthorizes_loaded_content(team, pool, monkeypatch, change):
    repo, bridge, alice = team
    repo.put_source(alice, 'ops', 'source', {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
    spec = {'description': 'Summarize each case', 'memory_types': ['case'],
            'trigger_policy': {'min_interval_seconds': 0}}
    repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': spec}, expected_rev=0)
    source = ContextNode(f'ctx://{alice.account_id}/agents/ops/memories/case/pool',
                         'MEMORY', 'case', 2, 'agent:ops', 'Operations', '', 'ABC123 pool advice')
    bridge.write_node(source, replace(alice, agent_id='ops'))
    job, data = snapshot(repo, alice.account_id)
    [item] = publish(repo, job, data, lambda p: [
        {'action': 'ADD', 'topic': 'pool', 'abstract': 'Operations', 'content': source.content,
         'quality_score': 0.9, 'source_uris': [p['sources'][0]['uri']]}])['items']
    outbox = SQLOutboxStore(pool=pool, fs=bridge)
    worker = OutboxWorker(InMemoryVectorIndex(8), MockEmbedder(8), fs=bridge)
    assert worker.run_once(outbox, [alice.account_id])['succeeded'] == 1
    if change == 'overlap':
        repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': {
            **spec, 'privacy_policy': {'redact_terms': ['ABC']}}}, expected_rev=1)
    bob = replace(alice, user_id='bob')
    loaded, proceed = Event(), Event()
    load = bridge.load_content

    def paused_load(rows):
        hydrated = load(rows)
        if not loaded.is_set():
            loaded.set()
            assert proceed.wait(20)
        return hydrated

    with monkeypatch.context() as patch, ThreadPoolExecutor(1) as executor:
        patch.setattr(bridge, 'load_content', paused_load)
        print('request', {'method': 'visible_items', 'user': 'bob', 'spaces': ['ops']})
        pending = executor.submit(repo.visible_items, bob, ['ops'])
        try:
            assert loaded.wait(20)
            if change == 'acl':
                request = {'method': 'set_acl', 'item_id': item['item_id'],
                           'acl': {'mode': 'restricted', 'readers': ['user:alice']}}
                result = repo.set_acl(alice, 'ops', request,
                                      expected_rev=repo.permissions(alice, 'ops')['authz_rev'])
            elif change == 'pause':
                request = {'method': 'put_source', 'principal': 'agent:ops',
                           'memory_types': ['case'], 'state': 'PAUSED'}
                result = repo.put_source(alice, 'ops', 'source', request, expected_rev=1)
            elif change == 'source':
                request = {'method': 'write_node', 'uri': source.uri, 'content': 'Revised pool advice'}
                bridge.write_node(replace(source, content=request['content']), replace(alice, agent_id='ops'))
                result = {'content': bridge.read_node(source.uri, replace(alice, agent_id='ops')).content}
                assert result['content'] == request['content']
            elif change == 'archive':
                request = {'method': 'complete_job', 'action': 'ARCHIVE', 'item_id': item['item_id']}
                repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': spec}, expected_rev=1)
                job, data = snapshot(repo, alice.account_id)
                result = publish(repo, job, data, lambda p: [request])
                assert result['items'] == [{'item_id': item['item_id'], 'state': 'ARCHIVED'}]
            else:
                terms = ['ABC', 'ABC123'] if change == 'overlap' else ['ABC123']
                request = {'method': 'save_rule', 'privacy_policy': {'redact_terms': terms}}
                result = repo.save_rule(alice, 'ops', 'notes', {'name': 'Operations', 'spec': {
                    **spec, 'privacy_policy': request['privacy_policy']}}, expected_rev=2 if change == 'overlap' else 1)
            print('concurrent request', request, 'response', result)
        finally:
            proceed.set()
        rows = pending.result(timeout=20)
    current = repo.visible_items(bob, ['ops'])
    print('response', [{'item_id': row['item_id'], 'content': row['content']} for row in rows],
          'current', [{'item_id': row['item_id'], 'content': row['content']} for row in current])
    assert rows == current
    if change in ('privacy', 'overlap'):
        assert [row['content'] for row in rows] == ['[已遮蔽] pool advice']
