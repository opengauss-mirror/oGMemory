"""Registered-user workflows through the management gateway and SQL storage."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from werkzeug.test import Client
from werkzeug.wrappers import Response

from core.models import ContextNode, Role
from providers.unified_config import OgMemConfig
from server.auth import ResolvedIdentity
from server.memory_service import MemoryService


@pytest.fixture
def spaces_api(monkeypatch):
    dsn = os.environ.get("OGMEM_SPACE_TEST_DSN")
    if not dsn:
        pytest.skip("Set OGMEM_SPACE_TEST_DSN to an isolated SQL database")
    account = "http-" + uuid4().hex[:16]
    cfg = OgMemConfig(
        storage_backend="sql", sql_connection_string=dsn, sql_pool_size=8, sql_dialect="auto",
        vector_db_type="memory", embedding_provider="mock",
        role_control_enabled=True, root_api_key=uuid4().hex,
        account_id=account, session_state_bridge_enabled=False,
    )
    service = MemoryService(cfg)
    manager = service.get_key_manager()
    manager.create_account(account, "admin")
    alice_key = manager.register_user(account, "alice")
    manager.register_user(account, "bob")
    cfg.shared_spaces_enabled = True
    repository = service.get_space_repository()
    import server.app as public
    import server.management_app as gateway

    monkeypatch.setattr(public, "_service", service)
    monkeypatch.setattr(gateway, "_get_service", lambda: service)
    client = Client(gateway.create_app(), Response)
    headers = {"X-API-Key": alice_key}
    actor = service.build_context({}, identity=ResolvedIdentity(Role.MEMBER, account, "alice"))
    yield client, headers, service, repository, actor
    service.shutdown()


def mutate(client, headers, method, path, body, revision=None, key=None):
    headers = {**headers, "Idempotency-Key": key or uuid4().hex}
    if revision is not None:
        headers["If-None-Match" if revision == 0 else "If-Match"] = "*" if revision == 0 else f'"{revision}"'
    response = client.open(path, method=method, headers=headers, json=body)
    assert response.status_code in (200, 201), response.get_json()
    return response.get_json()


def test_frontend_management_rule_configuration_stays_stable(spaces_api):
    client, headers, service, repo, actor = spaces_api
    page = client.get("/memory-viewer/")
    assert page.status_code == 200
    candidates = client.get("/management/spaces/principals", headers=headers).get_json()
    assert {"user:alice", "user:bob"} <= {row["principal"] for row in candidates["items"]}
    created = mutate(client, headers, "POST", "/management/spaces", {"space_id": "ops", "name": "运维"}, key="create")
    replay = mutate(client, headers, "POST", "/management/spaces", {"space_id": "ops", "name": "运维"}, key="create")
    assert replay["authz_rev"] == created["authz_rev"]
    members = [{"principal": "user:alice", "role": "manager"}, {"principal": "user:bob", "role": "member"}]
    mutate(client, headers, "PUT", "/management/spaces/ops/members", {"members": members}, created["authz_rev"])
    source = mutate(client, headers, "PUT", "/management/spaces/ops/sources/alice", {"principal": "user:alice", "memory_types": ["event"], "state": "ACTIVE"}, 0)
    rule = {"name": "排障归纳", "state": "ACTIVE", "spec": {"memory_types": ["event"], "description": "排障步骤与验证方法", "output_kind": "experience"}}
    saved = mutate(client, headers, "PUT", "/management/spaces/ops/rules/diagnosis", rule, 0)
    mutate(client, headers, "PATCH", "/management/spaces/ops", {"name": "数据库运维"}, source["authz_rev"])
    current = client.get("/management/spaces/ops/rules?rule_id=diagnosis", headers=headers).get_json()["items"][0]
    assert current["rule_rev"] == saved["rule_rev"]
    assert current["spec_hash"] == saved["spec_hash"]
    assert current["spec"]["description"] == rule["spec"]["description"]
    permissions = client.get("/management/spaces/ops/permissions", headers=headers).get_json()
    assert permissions["viewer_user_id"] == "alice"
    assert permissions["capabilities"]["manage_members"] is True
    audit = client.get("/management/spaces/ops/audit", headers=headers).get_json()
    assert len(audit["items"]) >= 5


def test_source_write_and_event_are_committed_together(spaces_api):
    _, _, service, repo, actor = spaces_api
    uri = f"ctx://{actor.account_id}/users/alice/memories/events/pool"
    node = ContextNode(uri, "MEMORY", "event", 2, "user:alice", "排障", "连接池排障", "检查连接释放，再观察等待队列。")
    service._get_context_fs().write_node(node, actor)
    with repo.transaction(actor.account_id, write=False) as cur:
        cur.execute("SELECT n.content,e.payload,e.status FROM context_nodes n JOIN outbox_events e ON e.uri=n.uri AND e.account_id=n.account_id WHERE n.uri=%s AND e.event_type='SOURCE_CHANGED'", (uri,))
        row = cur.fetchone()
    assert row["content"] == node.content
    assert row["payload"]["source_uri"] == uri
    assert row["status"] == "SOURCE_PENDING"
    node.content = "检查连接释放，关闭长事务，再确认等待队列恢复。"
    fs = service._get_context_fs()
    fs.write_node(node, actor)
    fs.write_node(node, actor)
    current = fs.read_node(uri, actor)
    assert current.content == node.content
    assert current.metadata["version"] == 2


def test_registered_candidates_flow_through_management_api(spaces_api):
    client, headers, service, _, actor = spaces_api
    service.get_key_manager().register_user(actor.account_id, 'carol')
    base = '/management/spaces'
    created = mutate(client, headers, 'POST', base, {'space_id': 'ops', 'name': 'Operations'})
    first = client.get(base + '/principals', headers=headers,
                       query_string={'space_id': 'ops', 'kind': 'user', 'q': 'o', 'limit': 1}).get_json()
    second = client.get(base + '/principals', headers=headers,
                        query_string={'space_id': 'ops', 'kind': 'user', 'q': 'o', 'limit': 1,
                                      'cursor': first['next_cursor']}).get_json()
    assert [first['items'][0]['principal'], second['items'][0]['principal']] == ['user:bob', 'user:carol']
    members = [{'principal': 'user:alice', 'role': 'manager'},
               {'principal': first['items'][0]['principal'], 'role': 'member'}]
    saved = mutate(client, headers, 'PUT', base + '/ops/members', {'members': members}, created['authz_rev'])
    remaining = client.get(base + '/principals', headers=headers,
                           query_string={'space_id': 'ops', 'kind': 'user', 'q': 'o'}).get_json()
    assert [row['principal'] for row in remaining['items']] == ['user:carol']
    current = client.get(base, headers=headers, query_string={'space_id': 'ops'}).get_json()['space']
    assert current['authz_rev'] == saved['authz_rev']
    assert [row['principal'] for row in current['members']] == ['user:alice', 'user:bob']


def test_http_and_mcp_share_authenticated_retrieval(spaces_api):
    client, headers, service, _, actor = spaces_api
    import anyio

    from server.mcp_server import create_mcp_server
    from server.mcp_tools import search_shared_memory

    mcp = create_mcp_server(service=service)
    tool = next(tool for tool in anyio.run(mcp.list_tools) if tool.name == "search_shared_memory")
    assert {"query", "space_ids"} <= set(tool.input_schema["required"])
    assert tool.input_schema['properties']['mode']['default'] == 'hybrid'
    assert {'limit', 'cursor'} <= set(tool.input_schema['properties'])
    mutate(client, headers, "POST", "/management/spaces", {"space_id": "ops", "name": "运维"})
    response = client.post("/api/spaces/search", headers=headers, json={"query": "排障", "space_ids": ["ops"]})
    assert response.status_code == 200, response.get_json()
    result = search_shared_memory(service, query="排障", space_ids=["ops"], transport_headers=headers, allow_stdio_env_auth=False)
    assert result["items"] == response.get_json()["items"]
    paged = client.post('/api/spaces/search', headers=headers, json={
        'query': '排障', 'space_ids': ['ops'], 'mode': 'keyword', 'top_k': 100, 'limit': 10})
    assert paged.status_code == 200
    result = search_shared_memory(service, query='排障', space_ids=['ops'], mode='keyword',
                                  top_k=100, limit=10, transport_headers=headers, allow_stdio_env_auth=False)
    assert (result['mode'], result['total'], result['next_cursor']) == (
        paged.get_json()['mode'], paged.get_json()['total'], paged.get_json()['next_cursor'])


def test_management_deletions_replay_and_refresh_lists(spaces_api):
    client, headers, _, _, _ = spaces_api
    base = '/management/spaces'
    mutate(client, headers, 'POST', base, {'space_id': 'ops', 'name': 'Operations'})
    mutate(client, headers, 'POST', base, {'space_id': 'remaining', 'name': 'Remaining'})
    source_payload = {'principal': 'user:alice', 'memory_types': ['event']}
    source = mutate(client, headers, 'PUT', base + '/ops/sources/alice-old', source_payload, 0)
    rule_payload = {'name': 'Resolution', 'spec': {'description': 'Verified resolution steps', 'memory_types': ['event']}}
    mutate(client, headers, 'PUT', base + '/ops/rules/retired', rule_payload, 0)
    mutate(client, headers, 'PUT', base + '/ops/rules/remaining', rule_payload, 0)

    deleted_rule = mutate(client, headers, 'DELETE', base + '/ops/rules/retired', {}, 1, key='delete-rule')
    replay = mutate(client, headers, 'DELETE', base + '/ops/rules/retired', {}, 1, key='delete-rule')
    assert (deleted_rule['deleted'], deleted_rule['rule_id'], deleted_rule['rule_rev']) == (True, 'retired', 2)
    assert (replay['deleted'], replay['rule_rev']) == (True, deleted_rule['rule_rev'])
    rules = client.get(base + '/ops/rules', headers=headers).get_json()['items']
    assert [row['rule_id'] for row in rules] == ['remaining']

    deleted_source = mutate(client, headers, 'DELETE', base + '/ops/sources/alice-old', {}, source['grant_rev'], key='delete-source')
    replay = mutate(client, headers, 'DELETE', base + '/ops/sources/alice-old', {}, source['grant_rev'], key='delete-source')
    assert (deleted_source['deleted'], deleted_source['grant_rev']) == (True, 2)
    assert replay['authz_rev'] == deleted_source['authz_rev']
    renewed = mutate(client, headers, 'PUT', base + '/ops/sources/alice-new', source_payload, 0)
    detail = client.get(base + '?space_id=ops', headers=headers).get_json()['space']
    assert [row['binding_id'] for row in detail['sources']] == ['alice-new']
    assert detail['sources'][0]['can_delete']
    assert detail['capabilities']['delete_space']

    deleted_space = mutate(client, headers, 'DELETE', base + '/ops', {}, renewed['authz_rev'], key='delete-space')
    replay = mutate(client, headers, 'DELETE', base + '/ops', {}, renewed['authz_rev'], key='delete-space')
    assert (deleted_space['deleted'], deleted_space['space_id']) == (True, 'ops')
    assert replay['authz_rev'] == deleted_space['authz_rev']
    assert [row['space_id'] for row in client.get(base, headers=headers).get_json()['items']] == ['remaining']
    mutate(client, headers, 'POST', base, {'space_id': 'ops-new', 'name': 'Operations'})
    assert [row['space_id'] for row in client.get(base, headers=headers).get_json()['items']] == ['ops-new', 'remaining']
