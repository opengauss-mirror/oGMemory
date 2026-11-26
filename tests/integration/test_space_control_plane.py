"""Control-plane writes preserve team-space references on real SQL storage."""
import json
import os
import threading
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")
import server.app as app_module
from fs.sql_adapter.dialect import resolve_sql_dialect
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService
from server.memory_service import MemoryService
from server.sql_control_plane_store import SQLControlPlaneStore
from server.tenant_admin import TenantAdminService


@pytest.fixture
def registry_connection():
    dsn = os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("Set TEST_OPENGAUSS_CONNECTION_STRING or TEST_SQL_CONNECTION_STRING")
    conn = psycopg2.connect(dsn)
    schema = 'mr375_' + uuid.uuid4().hex[:12]
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE SCHEMA ' + schema)
    cur.execute('SET search_path TO ' + schema)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.autocommit = True
        cur.execute('DROP SCHEMA ' + schema + ' CASCADE')
        conn.close()


@pytest.fixture
def control_store(registry_connection):
    conn = registry_connection
    cur = conn.cursor()
    dialect = resolve_sql_dialect('opengauss' if os.environ.get('TEST_OPENGAUSS_CONNECTION_STRING') else 'postgresql')
    base = Path('fs/sql_adapter/db_schema.sql').read_text(encoding='utf-8')
    control_plane = base[base.index('CREATE TABLE IF NOT EXISTS cp_accounts'):base.index('-- Protected workspace-sync registry')]
    cur.execute(dialect.prepare_schema(control_plane))
    cur.execute('CREATE TABLE context_nodes(account_id varchar(128), uri varchar(512), abstract text, content text, owner_space text)')
    ddl = Path('server/space_schema.sql').read_text(encoding='utf-8')
    # Exercise the exact new control-plane references and indexes.
    cur.execute(dialect.prepare_schema(ddl[:ddl.index('CREATE TABLE IF NOT EXISTS space_rules')]))
    conn.autocommit = False
    store = SQLControlPlaneStore.__new__(SQLControlPlaneStore)
    store._get_connection = lambda: conn
    store._return_connection = lambda connection: None
    store._snapshots = threading.local()
    store._dialect = dialect
    return store


def test_registry_writes_with_space_references(registry_connection, control_store):
    conn = registry_connection
    cur = conn.cursor()
    store = control_store
    manager = APIKeyManager(store)
    manager.create_account('tenant', 'alice')
    store.write_json(store.agents_path('tenant'), {'agents': {'bot': {'owner_user_id': 'alice', 'created_at': '2026-11-14T00:00:00Z'}}})
    cur.execute("INSERT INTO spaces(account_id,space_id,name,owner_user_id) VALUES ('tenant','team','Team','alice')")
    cur.execute("INSERT INTO space_members(account_id,space_id,principal,user_id) VALUES ('tenant','team','user:alice','alice')")
    cur.execute("INSERT INTO space_members(account_id,space_id,principal,agent_id) VALUES ('tenant','team','agent:bot','bot')")
    cur.execute("UPDATE cp_accounts SET authz_rev=7 WHERE account_id='tenant'")
    cur.execute("UPDATE cp_agents SET status='disabled' WHERE account_id='tenant' AND agent_id='bot'")
    conn.commit()
    for request, call in [
        ({'method': 'create_account', 'account_id': 'second', 'admin_user_id': 'admin'}, lambda: manager.create_account('second', 'admin')),
        ({'method': 'register_user', 'account_id': 'tenant', 'user_id': 'bob'}, lambda: manager.register_user('tenant', 'bob')),
        ({'method': 'set_role', 'account_id': 'tenant', 'user_id': 'alice', 'role': 'root'}, lambda: manager.set_role('tenant', 'alice', 'root')),
        ({'method': 'write_agents', 'account_id': 'tenant', 'agent_ids': ['helper']}, lambda: store.write_json(store.agents_path('tenant'), {'agents': {'helper': {'owner_user_id': 'alice', 'created_at': '2026-11-14T00:00:00Z'}}})),
    ]:
        try:
            call()
            response = {'success': True}
        except Exception as error:
            response = {'error': type(error).__name__, 'detail': str(error).splitlines()[0]}
        print(json.dumps({'request': request, 'response': response}))
        assert response == {'success': True}
    cur.execute("SELECT authz_rev FROM cp_accounts WHERE account_id='tenant'")
    assert cur.fetchone()[0] == 10
    assert {row['account_id'] for row in manager.get_accounts()} == {'tenant', 'second'}
    assert {row['user_id']: row['role'] for row in manager.get_users('tenant')} == {'alice': 'root', 'bob': 'user'}
    cur.execute("SELECT agent_id FROM cp_agents WHERE account_id='tenant' ORDER BY agent_id")
    assert cur.fetchall() == [('bot',), ('helper',)]
    cur.execute("SELECT status FROM cp_agents WHERE account_id='tenant' AND agent_id='bot'")
    assert cur.fetchone()[0] == 'disabled'
    assert manager.set_role('tenant', 'bob', 'admin')
    assert manager.delete_user('tenant', 'bob')
    assert [u['user_id'] for u in manager.get_users('tenant')] == ['alice']
    print(json.dumps({'account_authz_rev': 10, 'deleted_user': 'bob', 'remaining_users': ['alice']}))


@pytest.mark.parametrize('reference', ['member', 'owner', 'grantor'])
def test_admin_deletes_referenced_user(monkeypatch, registry_connection, control_store, reference):
    manager = APIKeyManager(control_store)
    admin_key = manager.create_account('tenant', 'alice')
    user_key = manager.register_user('tenant', 'bob', 'admin')
    cfg = OgMemConfig(role_control_enabled=True, root_api_key='root-key', admin_api_keys=[], account_id='tenant')
    service = MemoryService(config=cfg)
    service._auth = AuthService(cfg, manager)
    service._tenant_admin = TenantAdminService(manager, control_store, AuditService(control_store))
    monkeypatch.setattr(app_module, '_service', service)
    client = app_module.app.test_client()
    admin_headers = {'X-API-Key': admin_key, 'X-Account-ID': 'tenant'}
    deleted_headers = {'X-API-Key': user_key, 'X-Account-ID': 'tenant'}
    users_url = '/api/v1/admin/accounts/tenant/users'
    conn = registry_connection
    with conn.cursor() as cur:
        cur.execute("INSERT INTO spaces(account_id,space_id,name,owner_user_id) VALUES ('tenant','team','Team',%s)",
                    ('bob' if reference == 'owner' else 'alice',))
        if reference == 'member':
            cur.execute("INSERT INTO space_members(account_id,space_id,principal,user_id) VALUES ('tenant','team','user:bob','bob')")
        elif reference == 'grantor':
            cur.execute("""INSERT INTO space_source_bindings(account_id,space_id,binding_id,principal,authorized_by)
                           VALUES ('tenant','team','source','user:alice','bob')""")
    conn.commit()

    assert client.get(users_url, headers=deleted_headers).status_code == 200
    response = client.delete(f'{users_url}/bob', headers=admin_headers)
    revoked = client.get(users_url, headers=deleted_headers)
    listed = client.get(users_url, headers=admin_headers)
    listed_users = [row['user_id'] for row in listed.get_json()['users']]
    print(json.dumps({'reference': reference, 'request': f'DELETE {users_url}/bob',
                      'response': {'status': response.status_code, 'body': response.get_json()},
                      'old_key_status': revoked.status_code, 'listed_users': listed_users}))
    assert (response.status_code, response.get_json(), revoked.status_code, listed_users) == (
        200, {'deleted': True}, 401, ['alice'])
    assert client.delete(f'{users_url}/bob', headers=admin_headers).get_json() == {'deleted': False}
    assert client.post(users_url, json={'user_id': 'bob'}, headers=admin_headers).status_code == 400
    created = client.post(users_url, json={'user_id': 'carol'}, headers=admin_headers)
    assert created.status_code == 200 and created.get_json()['user_id'] == 'carol'
    with conn.cursor() as cur:
        cur.execute("SELECT status, key FROM cp_users WHERE account_id='tenant' AND user_id='bob'")
        assert cur.fetchone() == ('deleted', 'revoked')
