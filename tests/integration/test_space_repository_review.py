"""Team repository workflows with real SQL and AGFS storage."""

import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, RequestContext
from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.space_repository import SpaceRepository
from server.space_rules import SpaceError
from tests.integration.test_space_job_review import memory, publish, snapshot
from tests.integration.test_space_job_review import pool as pool


@pytest.fixture
def team(pool):
    config = os.environ.get('OGMEM_SPACE_TEST_CONFIG')
    if not config:
        pytest.skip('OGMEM_SPACE_TEST_CONFIG requires an isolated AGFS server')
    settings = json.loads(Path(config).read_text())
    bridge = AGFSSpaceBridge(pool, AGFSClient(settings['agfs_url']))
    repo = SpaceRepository(pool, content_store=bridge)
    ctx = RequestContext('repo-review-' + uuid4().hex[:12], 'alice', '', '', 'repository-review')
    with repo.transaction(ctx.account_id) as cur:
        cur.execute('INSERT INTO cp_accounts(account_id) VALUES (%s)', (ctx.account_id,))
        for user in ('alice', 'bob'):
            cur.execute('INSERT INTO cp_users(account_id,user_id,key) VALUES (%s,%s,%s)',
                        (ctx.account_id, user, user))
        cur.execute("INSERT INTO cp_agents(account_id,agent_id,owner_user_id) VALUES (%s,'ops','alice')",
                    (ctx.account_id,))
    repo.create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
    repo.set_members(ctx, 'ops', {'members': [
        {'principal': 'user:alice', 'role': 'manager'}, {'principal': 'user:bob'},
        {'principal': 'agent:ops'}]}, expected_rev=1)
    try:
        yield repo, bridge, ctx
    finally:
        bridge.close(10)


def test_published_memory_becomes_readable_after_indexing(team, pool):
    repo, bridge, ctx = team
    repo.put_source(ctx, 'ops', 'agent-source',
                    {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
    repo.save_rule(ctx, 'ops', 'pool', {'name': 'Pool advice', 'spec': {
        'description': 'Summarize connection pool advice', 'memory_types': ['case'],
        'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=0)
    source = ContextNode(f'ctx://{ctx.account_id}/agents/ops/memories/case/pool',
                         'MEMORY', 'case', 2, 'agent:ops', 'Pool advice', '', 'Check long transactions.')
    bridge.write_node(source, replace(ctx, agent_id='ops'))
    job, data = snapshot(repo, ctx.account_id)
    result = publish(repo, job, data, lambda p: [memory(p, 'pool')])
    [item] = result['items']
    print('request', {'method': 'generate_plans/complete_job', 'source_uri': source.uri}, 'response', result)
    outbox = SQLOutboxStore(pool=pool, fs=bridge)
    worker = OutboxWorker(InMemoryVectorIndex(8), MockEmbedder(8), fs=bridge)
    stats = worker.run_once(outbox, [ctx.account_id], 'repository-review')
    print('request', {'method': 'OutboxWorker.run_once', 'account_id': ctx.account_id}, 'response', stats)
    assert stats['succeeded'] == 1
    bob = replace(ctx, user_id='bob')
    page = repo.list_items(bob, 'ops')
    permission = repo.permissions(bob, 'ops', item['item_id'])
    print('request', {'method': 'list_items/permissions', 'user': 'bob', 'item_id': item['item_id']},
          'response', {'page': page, 'permissions': permission})
    assert [row['item_id'] for row in page['items']] == [item['item_id']]
    assert page['items'][0]['content'] == source.content
    assert page['items'][0]['index_state'] == 'READY'
    assert permission['can_read'] is True


def test_source_binding_conflict_is_recoverable(team):
    repo, _, ctx = team
    payload = {'principal': 'agent:ops', 'memory_types': ['case']}
    first = repo.put_source(ctx, 'ops', 'first', payload, expected_rev=0, request_id='first')
    print('request', {'method': 'put_source', 'binding_id': 'first', 'payload': payload}, 'response', first)
    try:
        repo.put_source(ctx, 'ops', 'second', payload, expected_rev=0, request_id='second')
    except Exception as exc:
        print('request', {'method': 'put_source', 'binding_id': 'second', 'payload': payload},
              'response', {'type': type(exc).__name__, 'status': getattr(exc, 'status', None),
                           'code': getattr(exc, 'code', None), 'message': str(exc)})
        assert isinstance(exc, SpaceError)
        assert (exc.status, exc.code) == (409, 'SOURCE_EXISTS')
    else:
        pytest.fail('Expected a source binding conflict')
    replay = repo.put_source(ctx, 'ops', 'first', payload, expected_rev=0, request_id='first')
    assert replay == first
    repo.delete_source(ctx, 'ops', 'first', expected_rev=first['grant_rev'])
    replacement = repo.put_source(ctx, 'ops', 'second', payload, expected_rev=0, request_id='second')
    print('request', {'method': 'delete_source/put_source', 'binding_id': 'second'}, 'response', replacement)
    assert replacement['binding_id'] == 'second'
