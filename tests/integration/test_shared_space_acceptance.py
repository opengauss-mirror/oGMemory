"""Team memory publication and permissions through normal member workflows."""

from uuid import uuid4

from werkzeug.test import Client
from werkzeug.wrappers import Response

from tests.integration.test_shared_space_agfs import (
    await_item,
    configure_space,
    write_source,
)
from tests.integration.test_shared_space_agfs import (
    deployment as deployment,
)
from tests.integration.test_shared_space_lifecycle import (
    run_rule,
)
from tests.integration.test_shared_space_lifecycle import (
    shared_space as shared_space,
)
from tests.integration.test_shared_space_lifecycle import (
    space_pool as space_pool,
)


def test_current_content_pages_follow_source_revisions(shared_space):
    repo, alice, source, worker, _, _ = shared_space
    run_rule(repo, alice, worker)
    first = repo.list_items(alice, 'ops', state='ACTIVE', limit=1)
    assert len(first['items']) == 1
    assert first['items'][0]['content']
    with repo.transaction(alice.account_id) as cur:
        cur.execute('UPDATE context_nodes SET content=%s,version=version+1 WHERE account_id=%s AND uri=%s',
                    ('Inspect pool waits and confirm restored capacity.', alice.account_id, source))
    pending = repo.list_items(alice, 'ops', state='ACTIVE', limit=1)
    assert pending == {'items': [], 'next_cursor': None}
    run_rule(repo, alice, worker)
    current = repo.list_items(alice, 'ops', state='ACTIVE', limit=1)
    assert [row['content'] for row in current['items']] == ['Inspect pool waits and confirm restored capacity.']
    assert current['next_cursor'] is None


def test_http_acl_current_revision_and_source_restore(deployment, monkeypatch):
    service, account = deployment['service'], deployment['account']
    repo, alice = configure_space(service, account)
    import server.app as public
    import server.management_app as gateway

    monkeypatch.setattr(public, '_service', service)
    monkeypatch.setattr(gateway, '_get_service', lambda: service)
    client = Client(gateway.create_app(), Response)
    credentials = deployment['credentials']

    def request(method, path, *, user='alice', body=None, revision=None, status=200):
        headers = {'X-API-Key': credentials[user + '_key']}
        if method in ('PUT', 'POST') and path.startswith('/management/'):
            headers['Idempotency-Key'] = uuid4().hex
        if revision is not None:
            headers['If-Match'] = f'"{revision}"'
        response = client.open(path, method=method, json=body, headers=headers)
        data = response.get_json()
        assert response.status_code == status, data
        return data

    def revision():
        return request('GET', '/management/spaces?space_id=ops')['space']['authz_rev']

    def visible_ids(user):
        result = []
        for mode in ('keyword', 'semantic', 'hybrid'):
            data = request('POST', '/api/spaces/search', user=user, body={
                'query': 'connection pool', 'space_ids': ['ops'], 'mode': mode,
            })
            result.append([item['item_id'] for item in data['items']])
        return result

    source_a = 'Connection pool procedure A: inspect waits and restore capacity.'
    source_b = 'Connection pool procedure B: check leaks and verify recovery.'
    write_source(service, account, source_a)
    first = await_item(repo, alice, source_a)
    opened_revision = revision()
    assert visible_ids('bob') == [[first['item_id']]] * 3

    write_source(service, account, source_b)
    current = await_item(repo, alice, source_b)
    write_source(service, account, source_a)
    restored_a = await_item(repo, alice, source_a)
    assert restored_a['item_id'] not in {first['item_id'], current['item_id']}
    assert visible_ids('bob') == [[restored_a['item_id']]] * 3
    write_source(service, account, source_b)
    restored_b = await_item(repo, alice, source_b)
    assert restored_b['item_id'] not in {first['item_id'], current['item_id'], restored_a['item_id']}
    current = restored_b
    conflict = request('PUT', '/management/spaces/ops/acl', revision=opened_revision, status=412,
                       body={'item_id': first['item_id'], 'mode': 'restricted', 'readers': ['user:alice']})
    assert conflict['error']['code'] == 'ITEM_CHANGED'
    page = request('GET', '/management/spaces/ops/items?state=ACTIVE&limit=1')
    assert [item['item_id'] for item in page['items']] == [current['item_id']]
    request('PUT', '/management/spaces/ops/acl', revision=revision(), body={
        'item_id': current['item_id'], 'mode': 'restricted', 'readers': ['user:alice'],
    })
    assert visible_ids('alice') == [[current['item_id']]] * 3
    assert visible_ids('bob') == [[], [], []]
    permissions = request('GET', f"/management/spaces/ops/permissions?item_id={current['item_id']}", user='bob')
    assert permissions['can_read'] is False

    request('PUT', '/management/spaces/ops/acl', revision=revision(), body={
        'item_id': current['item_id'], 'mode': 'restricted', 'readers': ['user:alice', 'user:bob'],
    })
    assert visible_ids('bob') == [[current['item_id']]] * 3
    write_source(service, account, source_a)
    restored = await_item(repo, alice, source_a)
    assert restored['item_id'] not in {first['item_id'], current['item_id']}
    assert restored['acl_mode'] == 'restricted'
    assert restored['acl_origin'] == 'manual'
    assert visible_ids('bob') == [[restored['item_id']]] * 3

    remaining = [
        {'principal': 'user:alice', 'role': 'manager'},
        {'principal': 'agent:ops-agent', 'role': 'member'},
    ]
    request('PUT', '/management/spaces/ops/members', revision=revision(), body={'members': remaining})
    assert visible_ids('bob') == [[], [], []]
    request('PUT', '/management/spaces/ops/members', revision=revision(), body={
        'members': [*remaining, {'principal': 'user:bob', 'role': 'member'}],
    })
    request('PUT', '/management/spaces/ops/acl', revision=revision(), body={
        'item_id': restored['item_id'], 'mode': 'inherit', 'readers': [],
    })
    assert visible_ids('bob') == [[restored['item_id']]] * 3
