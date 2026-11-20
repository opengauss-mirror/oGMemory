import time
import uuid

import pytest
from flask import Flask

from server.migration_api import create_migration_blueprint
from service.migration.admission import business_wsgi
from tests.unit.service.migration.test_merge_recovery import package

pytest_plugins = ['tests.unit.service.migration.test_instance']


@pytest.fixture
def client(instance):
    service, migration, files = instance
    service._cfg.root_api_key = 'test-root-key'
    service._cfg.role_control_enabled = True
    app = Flask(__name__)
    app.register_blueprint(create_migration_blueprint(lambda: service))
    app.add_url_rule('/business', view_func=lambda: 'business')
    app.wsgi_app = business_wsgi(app.wsgi_app, lambda: service)
    return app.test_client()


HEADERS = {'X-API-Key': 'test-root-key'}


def wait(client, identity):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get('/api/v1/migration/jobs/' + identity, headers=HEADERS)
        assert response.status_code == 200, response.json
        job = response.json
        if job['status'] != 'running':
            return job
        time.sleep(.02)
    pytest.fail('job did not finish')


def submit(client, kind, **params):
    body = dict(kind=kind, request_id=str(uuid.uuid4()), **params)
    response = client.post('/api/v1/migration/jobs', json=body, headers=HEADERS)
    assert response.status_code == 202, response.json
    return wait(client, response.json['job_id'])


def test_root_api_upload_preflight_import_and_snapshot_download(client, instance, tmp_path):
    service, migration, files = instance
    assert client.get('/api/v1/migration/status').status_code == 401
    assert client.get('/api/v1/migration/status', headers={'X-API-Key': 'wrong'}).status_code == 403
    content = package(tmp_path).read_bytes()
    uploaded = client.post('/api/v1/migration/uploads', data=content, content_type='application/octet-stream', headers=HEADERS)
    assert uploaded.status_code == 201, uploaded.json
    artifact = uploaded.json['artifact_id']
    preview = submit(client, 'preflight', artifact_id=artifact)
    assert preview['status'] == 'succeeded', preview
    result = submit(client, 'import', artifact_id=artifact, preflight_id=preview['id'], maintenance_confirmed=True)
    assert result['status'] == 'succeeded', result
    assert result['result']['outcome'] == 'succeeded'
    assert result['result']['data']['nodes']['count'] == 2
    assert set(result['maintenance']) == {'id', 'mode', 'phase', 'modified'}
    retried = client.post('/api/v1/migration/jobs', json=dict(
        kind='import', request_id=result['id'], artifact_id=artifact,
        preflight_id=preview['id'], maintenance_confirmed=True), headers=HEADERS)
    assert retried.status_code == 202
    assert retried.json['job_id'] == result['id']
    assert service._get_shared_vector_index().count() == 4
    exported = submit(client, 'export')
    assert exported['status'] == 'succeeded', exported
    url = '/api/v1/migration/jobs/' + exported['id'] + '/download'
    assert client.get(url).status_code == 401
    downloaded = client.get(url, headers=HEADERS)
    assert downloaded.status_code == 200
    assert downloaded.data[:2] == b'\x1f\x8b'


def test_management_remains_available_while_business_is_blocked(client, instance):
    service, migration, files = instance
    gate = service._admission.store
    gate.enter('maintenance', mode='import', resources={})
    try:
        assert client.get('/business').status_code == 503
        assert client.get('/api/v1/migration/status', headers=HEADERS).status_code == 200
    finally:
        from service.migration.participants import wait_participants
        wait_participants(gate, 'maintenance', 'drain', timeout=2)
        gate.request_refresh('maintenance')
        wait_participants(gate, 'maintenance', 'refresh', timeout=2)
        gate.reopen('maintenance')


@pytest.mark.parametrize('known_length', [True, False])
def test_control_body_bounded_before_json_parse(client, monkeypatch, known_length):
    import io
    original = client.application.json.loads
    def bounded(raw, *a, **kw):
        assert len(raw) <= 16 * 1024
        return original(raw, *a, **kw)
    monkeypatch.setattr(client.application.json, 'loads', bounded)
    raw = b'{"kind":"export","extra":"' + b'x' * 65536 + b'"}'
    options = {'data': raw} if known_length else {'environ_overrides': {
        'wsgi.input': io.BytesIO(raw), 'wsgi.input_terminated': True, 'CONTENT_LENGTH': ''}}
    response = client.post('/api/v1/migration/jobs', content_type='application/json', headers=HEADERS, **options)
    assert response.status_code == 413


@pytest.mark.parametrize('raw', [b'null', b'[]', b'{', b'[' * 2000])
def test_malformed_control_json_rejected(client, raw):
    assert client.post('/api/v1/migration/jobs', data=raw, content_type='application/json', headers=HEADERS).status_code == 400


def test_retired_api_and_standalone_cli_options_not_available(client):
    from cli.main import build_parser
    assert client.get('/api/migration/status', headers=HEADERS).status_code == 404
    with pytest.raises(SystemExit):
        build_parser().parse_args(['migration', '--maintenance-confirmed'])


@pytest.mark.parametrize('known_length', [True, False])
def test_upload_with_read_only_flask_limit(client, monkeypatch, known_length):
    import io

    from flask import Request
    # Flask 3.0 exposes this property without a setter.
    monkeypatch.setattr(Request, 'max_content_length', property(lambda self: 16))
    monkeypatch.setattr('server.migration_api.MAX_UPLOAD_BYTES', 32)
    for size, expected in ((24, 201), (33, 413)):
        raw = b'x' * size
        options = {'data': raw} if known_length else {'environ_overrides': {
            'wsgi.input': io.BytesIO(raw), 'wsgi.input_terminated': True, 'CONTENT_LENGTH': ''}}
        response = client.post('/api/v1/migration/uploads', content_type='application/octet-stream',
                               headers=HEADERS, **options)
        assert response.status_code == expected, response.json


def test_status_reads_gate_after_interrupted_export_cleanup(client, instance):
    _, migration, _ = instance
    migration.gate.enter('interrupted-export', mode='export', resources={})
    response = client.get('/api/v1/migration/status', headers=HEADERS)
    assert response.status_code == 200
    assert response.json['maintenance'] is None
    assert migration.gate.status() is None


@pytest.mark.parametrize('fields', [
    {'kind': []}, {'kind': {}}, {'request_id': None}, {'request_id': 7},
    {'kind': 'preflight', 'artifact_id': None},
    {'kind': 'import', 'maintenance_confirmed': True, 'preflight_id': []},
    {'kind': 'recover', 'recover_task': None},
])
def test_invalid_control_fields_return_bad_request(client, fields):
    body = {'kind': 'export', 'request_id': str(uuid.uuid4()), **fields}
    response = client.post('/api/v1/migration/jobs', json=body, headers=HEADERS)
    assert response.status_code == 400
