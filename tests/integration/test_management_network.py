"""HTTP network policy across the public and management gateway applications."""

import json
from threading import Thread
from uuid import uuid4

import pytest
import requests
from werkzeug.serving import make_server

import server.app as public
from core.models import RequestContext
from scripts.shared_space_agfs_qa import QAMemoryService
from server.management_app import create_app
from tests.integration.test_shared_space_service_review import configuration


@pytest.mark.parametrize('allowlist,trust_proxy,trusted_proxies,forwarded,allowed', [
    ([], False, [], {}, True),
    (['127.0.0.0/8'], False, [], {}, True),
    (['203.0.113.0/24'], False, [], {}, False),
    (['203.0.113.7'], True, ['127.0.0.1'], {'X-Forwarded-For': '203.0.113.7'}, True),
    (['203.0.113.7'], True, ['127.0.0.1'], {'X-Forwarded-For': '203.0.113.8'}, False),
    (['203.0.113.7'], True, ['10.0.0.0/8'], {'X-Forwarded-For': '203.0.113.7'}, False),
    (['203.0.113.7'], True, ['127.0.0.1'], {'X-Real-IP': '203.0.113.7'}, True),
    (['203.0.113.7'], False, [], {'X-Forwarded-For': '203.0.113.7'}, False),
], ids=['open', 'direct-cidr', 'direct-excluded', 'trusted-forwarded',
        'trusted-excluded', 'untrusted-forwarded', 'trusted-real-ip', 'direct-forwarded'])
def test_gateway_applies_configured_network_policy(monkeypatch, allowlist, trust_proxy,
                                                    trusted_proxies, forwarded, allowed):
    cfg = configuration(shared_spaces_enabled=True,
                        agfs_mount_prefix='/local/network-' + uuid4().hex,
                        http_ip_allowlist=allowlist, http_ip_allowlist_trust_proxy=trust_proxy,
                        http_trusted_proxies=trusted_proxies)
    service = QAMemoryService(cfg)
    server = None
    thread = None
    try:
        account = 'network-' + uuid4().hex[:12]
        service.get_key_manager().create_account(account, 'admin')
        key = service.get_key_manager().register_user(account, 'alice')
        ctx = RequestContext(account, 'alice', '', '', 'network-policy')
        service.get_space_repository().create_space(ctx, {'space_id': 'ops', 'name': 'Operations'})
        monkeypatch.setattr(public, '_service', service)
        server = make_server('127.0.0.1', 0, create_app())
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        calls = [
            ('POST', '/api/spaces/search', {'query': 'pool', 'space_ids': ['ops'], 'mode': 'keyword'}),
            ('GET', '/management/spaces', None),
            ('POST', '/management/spaces', {'space_id': 'created', 'name': 'Created over HTTP'}),
        ]
        responses = []
        with requests.Session() as client:
            client.trust_env = False
            for method, path, body in calls:
                response = client.request(method, url + path, json=body, timeout=10,
                    headers={'X-API-Key': key, 'X-Account-Id': account,
                             'Idempotency-Key': uuid4().hex, **forwarded})
                result = response.json()
                print(json.dumps({'request': {'method': method, 'path': path, 'json': body,
                                  'remote_addr': '127.0.0.1', 'forwarded_headers': forwarded,
                                  'http_ip_allowlist': allowlist, 'trust_proxy': trust_proxy,
                                  'trusted_proxies': trusted_proxies, 'identity': 'registered Alice key'},
                                  'response': {'status': response.status_code, 'body': result}},
                                 ensure_ascii=False), flush=True)
                responses.append(response.status_code)
            listed = service.get_space_repository().list_spaces(ctx)
            print('stored spaces', [space['space_id'] for space in listed['items']], flush=True)
        assert responses == ([200, 200, 201] if allowed else [403, 403, 403])
        assert {space['space_id'] for space in listed['items']} == ({'ops', 'created'} if allowed else {'ops'})
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(10)
        service.shutdown(10)
