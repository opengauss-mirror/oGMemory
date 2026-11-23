"""Shared-space control plane business workflows against a dedicated SQL database."""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

import pytest

from core.models import RequestContext, Role
from fs.sql_adapter.pool import SharedConnectionPool
from server.api_keys import APIKeyManager
from server.space_repository import SpaceRepository
from server.space_rules import digest, validate_rule
from server.sql_control_plane_store import SQLControlPlaneStore


@pytest.fixture()
def spaces():
    dsn = os.environ.get('OGMEM_SPACE_TEST_DSN')
    if not dsn:
        pytest.skip('OGMEM_SPACE_TEST_DSN selects an isolated SQL test database')
    pool = SharedConnectionPool(dsn, pool_size=2, dialect='auto')
    store = SQLControlPlaneStore(pool=pool)
    repository = SpaceRepository(pool)
    keys = APIKeyManager(store)
    account = 'space-control-' + uuid4().hex[:12]
    keys.create_account(account, 'alice')
    keys.register_user(account, 'bob')
    ctx = RequestContext(account, 'alice', '', 'test', 'control-test', role=Role.ADMIN)
    yield repository, keys, store, ctx
    pool.close_all()


def spec():
    return {'memory_types': ['event'], 'description': 'Reusable troubleshooting steps.',
            'privacy_policy': {'redact_terms': ['Example Customer']}}


def test_normalized_space_names_and_tenant_unique_index(spaces):
    repository, keys, _, ctx = spaces
    created = repository.create_space(ctx, {'space_id': 'ops', 'name': '  Operations  '})
    assert created['space']['name'] == 'Operations'
    renamed = repository.update_space(ctx, 'ops', {'name': '  Resolution  '}, expected_rev=1)
    assert renamed['space']['name'] == 'Resolution'
    unchanged = repository.update_space(ctx, 'ops', {'name': 'Resolution'}, expected_rev=renamed['authz_rev'])
    assert unchanged['space']['name'] == 'Resolution'
    reclaimed = repository.create_space(ctx, {'space_id': 'new-ops', 'name': 'Operations'})
    assert reclaimed['space']['name'] == 'Operations'
    other_account = 'space-name-' + uuid4().hex[:12]
    keys.create_account(other_account, 'alice')
    other = replace(ctx, account_id=other_account)
    assert repository.create_space(other, {'space_id': 'ops', 'name': 'Resolution'})['space']['name'] == 'Resolution'
    with repository.transaction(ctx.account_id, write=False) as cur:
        cur.execute("""SELECT i.indisunique,pg_get_indexdef(i.indexrelid) AS definition
            FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
            WHERE c.relname='uq_spaces_account_name' AND pg_table_is_visible(c.oid)""")
        index = cur.fetchone()
        assert index['indisunique'] is True
        assert 'account_id' in index['definition'] and 'btrim' in index['definition']


def test_registered_members_sources_and_stable_rule(spaces):
    repository, _, _, ctx = spaces
    created = repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'}, request_id='create')
    replayed = repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'}, request_id='create')
    assert replayed == created
    member_set = {'members': [{'principal': 'user:alice', 'role': 'manager'},
                              {'principal': 'user:bob', 'role': 'member'}]}
    repository.set_members(ctx, 'ops', member_set, expected_rev=1)
    source = repository.put_source(ctx, 'ops', 'alice', {'principal': 'user:alice', 'memory_types': ['event']}, expected_rev=0)
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    bob_source = repository.put_source(bob, 'ops', 'bob', {'principal': 'user:bob', 'memory_types': ['event']}, expected_rev=0)
    rule = repository.save_rule(ctx, 'ops', 'reusable', {'name': 'Reusable', 'spec': spec()}, expected_rev=0)
    first = repository.list_rules(ctx, 'ops')['items'][0]
    membership = repository.set_members(ctx, 'ops', {'members': [member_set['members'][0]]}, expected_rev=bob_source['authz_rev'])
    current = repository.list_rules(ctx, 'ops')['items'][0]
    assert current['rule_rev'] == rule['rule_rev']
    assert current['spec_hash'] == first['spec_hash']
    assert current['spec'] == first['spec']
    revised = repository.save_rule(ctx, 'ops', 'reusable', {'name': 'Refined', 'spec': {**spec(), 'description': 'Resolution steps and verification.'}}, expected_rev=1)
    assert revised['rule_rev'] == 2
    historical = repository.list_rules(ctx, 'ops', rule_id='reusable', revision=1)['items'][0]
    assert historical['name'] == 'Reusable'
    assert historical['spec_hash'] == first['spec_hash']
    detail = repository.list_spaces(ctx, space_id='ops')['space']
    assert detail['authz_rev'] == membership['authz_rev']
    assert next(row for row in detail['sources'] if row['binding_id'] == 'bob')['state'] == 'PAUSED'
    assert source['grant_rev'] == 1
    assert repository.permissions(ctx, 'ops')['capabilities']['manage_space']


def test_identity_delta_preserves_parallel_registration(spaces):
    _, keys, store, ctx = spaces
    path = store.users_path(ctx.account_id)
    before = store.read_json(path, {'users': {}})
    other_store = SQLControlPlaneStore(pool=store._shared_pool) if hasattr(store, '_shared_pool') else SQLControlPlaneStore(connection_string=store._connection_string)
    other_keys = APIKeyManager(other_store)
    other_keys.register_user(ctx.account_id, 'carol')
    before['users']['bob']['role'] = 'admin'
    store.write_json(path, before)
    users = {user['user_id']: user for user in keys.get_users(ctx.account_id)}
    assert users['carol']['status'] == 'active'
    assert users['bob']['role'] == 'admin'
    keys.delete_user(ctx.account_id, 'carol')
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM cp_users WHERE account_id=%s AND user_id=%s', (ctx.account_id, 'carol'))
            assert cur.fetchone() is None
    store._return_connection(conn)


def test_registered_user_identity_restores_with_fresh_credentials(spaces):
    _, keys, _, ctx = spaces
    original = next(user for user in keys.get_users(ctx.account_id) if user['user_id'] == 'bob')
    assert keys.delete_user(ctx.account_id, 'bob')
    key = keys.register_user(ctx.account_id, 'bob', role='admin')
    restored = next(user for user in keys.get_users(ctx.account_id) if user['user_id'] == 'bob')
    assert restored['status'] == 'active'
    assert restored['key'] == key
    assert datetime.fromisoformat(restored['created_at']) > datetime.fromisoformat(original['created_at'])
    assert keys.resolve_user_key(key, ctx.account_id) == {
        'account_id': ctx.account_id, 'user_id': 'bob', 'role': 'admin',
    }


def test_principal_and_space_cursor_pagination(spaces):
    repository, _, _, ctx = spaces
    repository.create_space(ctx, {'space_id': 'a', 'name': 'A'})
    repository.create_space(ctx, {'space_id': 'b', 'name': 'B'})
    first = repository.list_spaces(ctx, limit=1)
    second = repository.list_spaces(ctx, limit=1, cursor=first['next_cursor'])
    assert [first['items'][0]['space_id'], second['items'][0]['space_id']] == ['a', 'b']
    first = repository.list_principals(ctx, limit=1, kind='user')
    second = repository.list_principals(ctx, limit=1, kind='user', cursor=first['next_cursor'])
    assert [first['items'][0]['principal'], second['items'][0]['principal']] == ['user:alice', 'user:bob']
    filtered = repository.list_principals(ctx, kind='user', q='bob')
    assert filtered['items'][0]['id'] == 'bob'
    assert repository.list_items(ctx, 'a', state='ACTIVE')['items'] == []
    assert repository.list_jobs(ctx, 'a', state='SUCCEEDED')['items'] == []
    assert len(repository.list_audit(ctx, 'a', action='space.create')['items']) == 1


def test_registered_directory_drives_team_membership(spaces):
    from server.audit import AuditService
    from server.tenant_admin import TenantAdminService

    repository, keys, store, ctx = spaces
    admin = TenantAdminService(keys, store, AuditService(store))
    keys.register_user(ctx.account_id, 'carol')
    admin.create_agent(ctx, ctx.account_id, 'ops-agent', 'bob')
    created = repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    first = repository.list_principals(ctx, space_id='ops', limit=2)
    second = repository.list_principals(ctx, space_id='ops', limit=2, cursor=first['next_cursor'])
    candidates = first['items'] + second['items']
    assert [row['principal'] for row in candidates] == ['agent:ops-agent', 'user:bob', 'user:carol']
    assert repository.list_principals(ctx, space_id='ops', kind='agent', q='OPS')['items'][0]['owner_user_id'] == 'bob'
    members = [{'principal': 'user:alice', 'role': 'manager'}] + [
        {'principal': row['principal'], 'role': 'member'} for row in candidates]
    saved = repository.set_members(ctx, 'ops', {'members': members}, expected_rev=created['authz_rev'])
    detail = repository.list_spaces(ctx, 'ops')['space']
    assert [row['principal'] for row in detail['members']] == ['agent:ops-agent', 'user:alice', 'user:bob', 'user:carol']
    members[2]['role'] = 'manager'
    updated = repository.set_members(ctx, 'ops', {'members': members}, expected_rev=saved['authz_rev'])
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    assert repository.permissions(bob, 'ops')['capabilities']['manage_members'] is True
    retained = [member for member in members if member['principal'] != 'user:carol']
    removed = repository.set_members(bob, 'ops', {'members': retained}, expected_rev=updated['authz_rev'])
    available = repository.list_principals(bob, space_id='ops', kind='user')['items']
    assert [row['principal'] for row in available] == ['user:carol']
    restored = retained + [{'principal': available[0]['principal'], 'role': 'member'}]
    saved = repository.set_members(bob, 'ops', {'members': restored}, expected_rev=removed['authz_rev'])
    assert len(repository.list_spaces(bob, 'ops')['space']['members']) == 4
    admin.update_agent(ctx, ctx.account_id, 'ops-agent', '')
    repository.set_members(bob, 'ops', {'members': retained}, expected_rev=saved['authz_rev'])
    members_after_unbinding = repository.list_spaces(bob, 'ops')['space']['members']
    assert [row['principal'] for row in members_after_unbinding] == ['agent:ops-agent', 'user:alice', 'user:bob']


def test_registered_identity_spelling_is_preserved_in_members_and_rule_acl(spaces):
    repository, keys, _, ctx = spaces
    readers = ['reader@example.org', '读者甲', 'ops:reader']
    for user_id in readers:
        keys.register_user(ctx.account_id, user_id)
    repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    selected = []
    for user_id in readers:
        candidates = repository.list_principals(ctx, space_id='ops', kind='user', q=user_id)['items']
        assert [row['id'] for row in candidates] == [user_id]
        selected.append(candidates[0]['principal'])
    members = [{'principal': 'user:alice', 'role': 'manager'}] + [
        {'principal': subject, 'role': 'member'} for subject in selected]
    repository.set_members(ctx, 'ops', {'members': members}, expected_rev=1)
    rule = {**spec(), 'output_acl': {'mode': 'restricted', 'readers': selected}}
    repository.save_rule(ctx, 'ops', 'reader-rule', {'name': 'Reusable', 'spec': rule}, expected_rev=0)
    saved = repository.list_rules(ctx, 'ops', rule_id='reader-rule')['items'][0]
    assert saved['spec']['output_acl']['readers'] == sorted(selected)
    for user_id in readers:
        reader = replace(ctx, user_id=user_id, role=Role.MEMBER)
        assert repository.permissions(reader, 'ops')['viewer_user_id'] == user_id


def test_rule_defaults_are_canonical_and_independent():
    original = spec()
    first = validate_rule(original)
    second = validate_rule({**original, 'memory_types': ['event', 'event']})
    assert digest(first) == digest(second)
    assert first['privacy_policy']['redact_terms'] == ['Example Customer']
    assert first['output_kind'] == 'experience'
    assert first['maintenance_policy']['retention_days'] == 180
    assert original == spec()

def test_repeated_startup_during_content_write(spaces):
    from concurrent.futures import ThreadPoolExecutor

    repository, _, _, ctx = spaces
    with repository.transaction(ctx.account_id) as cur:
        cur.execute('LOCK TABLE context_nodes IN ROW EXCLUSIVE MODE')
        with ThreadPoolExecutor(max_workers=1) as executor:
            started = executor.submit(SpaceRepository, repository.pool).result(timeout=3)
            assert started.pool is repository.pool


def test_team_managers_maintain_shared_definition_after_membership_change(spaces):
    repository, _, _, ctx = spaces
    repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    repository.set_members(ctx, 'ops', {'members': [{'principal': 'user:alice', 'role': 'manager'}, {'principal': 'user:bob', 'role': 'manager'}]}, expected_rev=1)
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    repository.save_rule(bob, 'ops', 'bob-rule', {'name': 'Bob rule', 'spec': spec()}, expected_rev=0)
    assert repository.list_rules(ctx, 'ops')['items'][0]['spec']['description'] == spec()['description']
    repository.set_members(ctx, 'ops', {'members': [{'principal': 'user:alice', 'role': 'manager'}]}, expected_rev=2)
    repository.save_rule(ctx, 'ops', 'bob-rule', {'name': 'Managed rule', 'spec': spec()}, expected_rev=1)
    rule = repository.list_rules(ctx, 'ops', rule_id='bob-rule')['items'][0]
    assert rule['name'] == 'Managed rule'
    assert rule['rule_rev'] == 2
    assert repository.permissions(ctx, 'ops')['capabilities']['manage_rules']


def test_space_owner_deletion_replays_and_releases_name(spaces):
    repository, _, _, ctx = spaces
    owner = replace(ctx, role=Role.MEMBER)
    repository.create_space(owner, {'space_id': 'ops', 'name': 'Operations'})
    repository.create_space(owner, {'space_id': 'remaining', 'name': 'Remaining'})
    assert repository.list_spaces(owner, space_id='ops')['space']['capabilities']['delete_space']
    assert next(row for row in repository.list_spaces(owner)['items'] if row['space_id'] == 'ops')['capabilities']['delete_space']
    deleted = repository.delete_space(owner, 'ops', expected_rev=1, request_id='delete-space')
    assert deleted == {'deleted': True, 'space_id': 'ops', 'authz_rev': 2}
    assert repository.delete_space(owner, 'ops', expected_rev=1, request_id='delete-space') == deleted
    assert [row['space_id'] for row in repository.list_spaces(owner)['items']] == ['remaining']
    recreated = repository.create_space(owner, {'space_id': 'ops-new', 'name': 'Operations'})
    assert recreated['space']['name'] == 'Operations'
    assert [row['space_id'] for row in repository.list_spaces(owner)['items']] == ['ops-new', 'remaining']
    with repository.transaction(ctx.account_id, write=False) as cur:
        cur.execute('SELECT state,deleted_at,authz_rev FROM spaces WHERE account_id=%s AND space_id=%s', (ctx.account_id, 'ops'))
        row = cur.fetchone()
        assert (row['state'], row['authz_rev']) == ('ARCHIVED', 2)
        assert row['deleted_at'] is not None


def test_tenant_administrator_deletes_member_owned_space(spaces):
    repository, _, _, ctx = spaces
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    repository.create_space(bob, {'space_id': 'bob-space', 'name': 'Bob space'})
    assert repository.permissions(ctx, 'bob-space')['capabilities']['delete_space']
    assert repository.delete_space(ctx, 'bob-space', expected_rev=1)['deleted']
    assert repository.list_spaces(ctx)['items'] == []


def test_rule_deletion_keeps_spec_history_and_remaining_rules(spaces):
    repository, _, _, ctx = spaces
    repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    repository.set_members(ctx, 'ops', {'members': [
        {'principal': 'user:alice', 'role': 'manager'}, {'principal': 'user:bob', 'role': 'manager'},
    ]}, expected_rev=1)
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    repository.save_rule(ctx, 'ops', 'retired', {'name': 'Retired', 'spec': spec()}, expected_rev=0)
    repository.save_rule(ctx, 'ops', 'remaining', {'name': 'Remaining', 'spec': spec()}, expected_rev=0)
    original = repository.list_rules(ctx, 'ops', rule_id='retired')['items'][0]
    deleted = repository.delete_rule(bob, 'ops', 'retired', expected_rev=1, request_id='delete-rule')
    assert deleted == {'deleted': True, 'rule_id': 'retired', 'rule_rev': 2}
    assert repository.delete_rule(bob, 'ops', 'retired', expected_rev=1, request_id='delete-rule') == deleted
    assert [row['rule_id'] for row in repository.list_rules(ctx, 'ops')['items']] == ['remaining']
    with repository.transaction(ctx.account_id, write=False) as cur:
        cur.execute('SELECT rule_rev,state,spec,spec_hash FROM space_rule_revisions WHERE account_id=%s AND space_id=%s AND rule_id=%s ORDER BY rule_rev', (ctx.account_id, 'ops', 'retired'))
        history = cur.fetchall()
        assert [(row['rule_rev'], row['state']) for row in history] == [(1, 'ACTIVE'), (2, 'ARCHIVED')]
        assert [row['spec'] for row in history] == [original['spec'], original['spec']]
        assert [row['spec_hash'] for row in history] == [original['spec_hash'], original['spec_hash']]
        cur.execute('SELECT deleted_at FROM space_rules WHERE account_id=%s AND space_id=%s AND rule_id=%s', (ctx.account_id, 'ops', 'retired'))
        assert cur.fetchone()['deleted_at'] is not None


def test_source_owner_and_manager_revoke_and_reauthorize_new_binding(spaces):
    repository, _, _, ctx = spaces
    repository.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    repository.set_members(ctx, 'ops', {'members': [
        {'principal': 'user:alice', 'role': 'manager'}, {'principal': 'user:bob', 'role': 'member'},
    ]}, expected_rev=1)
    repository.put_source(ctx, 'ops', 'alice', {'principal': 'user:alice', 'memory_types': ['event']}, expected_rev=0)
    bob = replace(ctx, user_id='bob', role=Role.MEMBER)
    payload = {'principal': 'user:bob', 'memory_types': ['event']}
    repository.put_source(bob, 'ops', 'bob-old', payload, expected_rev=0)
    own_source = next(row for row in repository.list_spaces(bob, space_id='ops')['space']['sources'] if row['binding_id'] == 'bob-old')
    assert own_source['can_delete']
    deleted = repository.delete_source(bob, 'ops', 'bob-old', expected_rev=1, request_id='delete-source')
    assert deleted['deleted'] and deleted['grant_rev'] == 2
    assert repository.delete_source(bob, 'ops', 'bob-old', expected_rev=1, request_id='delete-source') == deleted
    assert [row['binding_id'] for row in repository.list_spaces(ctx, space_id='ops')['space']['sources']] == ['alice']
    recreated = repository.put_source(bob, 'ops', 'bob-new', payload, expected_rev=0)
    assert recreated['grant_rev'] == 1
    assert [row['binding_id'] for row in repository.list_spaces(ctx, space_id='ops')['space']['sources']] == ['alice', 'bob-new']
    manager = replace(ctx, role=Role.MEMBER)
    source = next(row for row in repository.list_spaces(manager, space_id='ops')['space']['sources'] if row['binding_id'] == 'bob-new')
    assert source['can_delete']
    assert repository.delete_source(manager, 'ops', 'bob-new', expected_rev=1)['deleted']
    with repository.transaction(ctx.account_id, write=False) as cur:
        cur.execute('SELECT binding_id,state,grant_rev,deleted_at FROM space_source_bindings WHERE account_id=%s AND space_id=%s ORDER BY binding_id', (ctx.account_id, 'ops'))
        rows = cur.fetchall()
        assert [(row['binding_id'], row['state'], row['grant_rev']) for row in rows] == [('alice', 'ACTIVE', 1), ('bob-new', 'PAUSED', 2), ('bob-old', 'PAUSED', 2)]
        assert all(row['deleted_at'] is not None for row in rows[1:])
