"""Service activation and retrieval against isolated SQL and AGFS stores."""

import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from flask import Flask

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, RequestContext
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from scripts.shared_space_agfs_qa import QAMemoryService
from server.mcp_tools import search_shared_memory
from server.space_management import create_shared_search_blueprint
from tests.integration.test_space_job_review import memory, publish, snapshot
from tests.integration.test_space_job_review import pool as pool
from tests.integration.test_space_repository_review import team as team


def configuration(**overrides):
    path = os.environ.get('OGMEM_SPACE_TEST_CONFIG')
    if not path:
        pytest.skip('OGMEM_SPACE_TEST_CONFIG requires isolated SQL and AGFS stores')
    env = json.loads(Path(path).read_text())
    values = dict(storage_backend='agfs', shared_spaces_enabled=False,
                  sql_connection_string=env['dsn'], sql_dialect='opengauss', sql_pool_size=4,
                  agfs_base_url=env['agfs_url'], agfs_mount_prefix='/local',
                  role_control_enabled=True, root_api_key=uuid4().hex, index_workers=0,
                  session_state_bridge_enabled=False, provider='mock', embedding_provider='mock',
                  vector_db_type='opengauss', opengauss_connection_string=env['dsn'],
                  opengauss_dimension=64, opengauss_table_name='mr385_vectors',
                  opengauss_pool_size=2)
    return OgMemConfig(**(values | overrides))


def seed_personal(service, ctx, kind):
    if kind == 'identity':
        service.get_key_manager().create_account(ctx.account_id, 'admin')
        return service.get_key_manager().register_user(ctx.account_id, ctx.user_id)
    if kind == 'memory':
        node = ContextNode(f'ctx://{ctx.account_id}/users/alice/memories/case/pool',
                           'MEMORY', 'case', 2, 'user:alice', 'Pool advice', '', 'Check pool limits.')
        service._get_context_fs().write_node(node, ctx)
        return node.uri
    result = service._get_archive_store().write_archive(
        'session1', 'Pool conversation', 'Pool advice', [{'role': 'user', 'content': 'Check pool limits.'}],
        ctx, archive_id='archive1')
    assert result.success
    return 'archive1'


def read_personal(service, ctx, kind, handle):
    if kind == 'identity':
        value = service.get_auth_service().resolve_identity(
            {'X-API-Key': handle, 'X-Account-Id': ctx.account_id})
        result = {'account_id': value.account_id, 'user_id': value.user_id}
    elif kind == 'memory':
        result = service._get_context_fs().read_node(handle, ctx).content
    else:
        result = service._get_archive_store().read_archive('session1', handle, ctx).messages
    print('request', {'operation': 'read_personal', 'kind': kind, 'account': ctx.account_id},
          'response', result)
    return result


@pytest.mark.parametrize('kind', ['identity', 'memory', 'session'])
def test_existing_agfs_requires_explicit_migration(kind):
    cfg = configuration(agfs_mount_prefix='/local/legacy-' + uuid4().hex)
    ctx = RequestContext('legacy-' + uuid4().hex[:8], 'alice', '', 'session1', 'mr385-legacy')
    old = QAMemoryService(cfg)
    try:
        handle = seed_personal(old, ctx, kind)
        before = read_personal(old, ctx, kind, handle)
    finally:
        old.shutdown(10)
    with pytest.raises(ValueError, match='migrate and verify accounts, keys, memories and sessions') as error:
        attempted = QAMemoryService(replace(cfg, shared_spaces_enabled=True))
        attempted.shutdown(10)
    print('request', {'shared_spaces_enabled': True, 'existing_data': kind}, 'response', str(error.value))
    reopened = QAMemoryService(cfg)
    try:
        assert read_personal(reopened, ctx, kind, handle) == before
    finally:
        reopened.shutdown(10)


def test_shared_agfs_restart_preserves_registered_identity_memory_and_session():
    cfg = configuration(shared_spaces_enabled=True, agfs_mount_prefix='/local/shared-' + uuid4().hex)
    ctx = RequestContext('fresh-' + uuid4().hex[:8], 'alice', '', 'session1', 'mr385-fresh')
    service = QAMemoryService(cfg)
    try:
        handles = {kind: seed_personal(service, ctx, kind) for kind in ('identity', 'memory', 'session')}
        before = {kind: read_personal(service, ctx, kind, handle) for kind, handle in handles.items()}
    finally:
        service.shutdown(10)
    reopened = QAMemoryService(cfg)
    try:
        after = {kind: read_personal(reopened, ctx, kind, handle) for kind, handle in handles.items()}
        assert after == before
    finally:
        reopened.shutdown(10)


@pytest.mark.parametrize('backend', ['milvus', 'opengauss'])
def test_shared_search_initializes_dependencies_for_requested_mode(team, backend):
    if backend == 'milvus':
        pytest.importorskip('pymilvus')
    repo, bridge, ctx = team
    repo.put_source(ctx, 'ops', 'agent-source', {'principal': 'agent:ops', 'memory_types': ['case']}, expected_rev=0)
    repo.save_rule(ctx, 'ops', 'pool', {'name': 'Pool advice', 'spec': {
        'description': 'Summarize pool advice', 'memory_types': ['case'],
        'trigger_policy': {'min_interval_seconds': 0}}}, expected_rev=0)
    node = ContextNode(f'ctx://{ctx.account_id}/agents/ops/memories/case/pool',
                       'MEMORY', 'case', 2, 'agent:ops', 'Pool advice', '', 'Check pool limits.')
    bridge.write_node(node, replace(ctx, agent_id='ops'))
    job, data = snapshot(repo, ctx.account_id)
    [item] = publish(repo, job, data, lambda p: [memory(p, 'pool')])['items']
    cfg = configuration(shared_spaces_enabled=True, vector_db_type=backend,
                        milvus_uri='http://127.0.0.1:19999', milvus_timeout=1)
    service = QAMemoryService(cfg)
    try:
        index = service._get_shared_vector_index() if backend == 'opengauss' else InMemoryVectorIndex(64)
        stats = OutboxWorker(index, MockEmbedder(64), fs=bridge).run_once(
            SQLOutboxStore(pool=repo.pool, fs=bridge), [ctx.account_id], 'mr385')
        assert stats['succeeded'] == 1
        app = Flask(__name__)
        app.register_blueprint(create_shared_search_blueprint(lambda: service, lambda e: ({'error': str(e)}, 400)))
        headers = {'X-API-Key': 'alice', 'X-Account-Id': ctx.account_id}
        modes = ['keyword', 'semantic', 'hybrid', 'keyword'] if backend == 'opengauss' else ['keyword']
        for mode in modes:
            body = {'query': 'pool', 'space_ids': ['ops'], 'mode': mode}
            response = app.test_client().post('/api/spaces/search', json=body, headers=headers)
            print('request', {'method': 'POST', 'path': '/api/spaces/search', 'backend': backend, **body},
                  'response', {'status': response.status_code, 'body': response.get_json()})
            assert response.status_code == 200
            assert [row['item_id'] for row in response.get_json()['items']] == [item['item_id']]
            result = search_shared_memory(service, **body, transport_headers=headers, allow_stdio_env_auth=False)
            print('request', {'tool': 'search_shared_memory', 'backend': backend, **body}, 'response', result)
            assert [row['item_id'] for row in result['items']] == [item['item_id']]
            assert result['items'][0]['content'] == node.content
    finally:
        service.shutdown(10)
