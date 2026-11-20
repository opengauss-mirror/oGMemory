"""CLI uses the same ROOT API and persistent jobs as the viewer."""
import hashlib
import json
import uuid

import httpx
import pytest

from cli.commands.migration import MigrationClient, execute, run
from cli.main import build_parser


def arguments(*args):
    return build_parser().parse_args(['migration', *args])


def test_import_requires_confirmation_before_network():
    with pytest.raises(ValueError, match='--yes'):
        execute(arguments('import', '--preflight-id', str(uuid.uuid4())), None)


def test_import_reuses_preflight_artifact_and_request_identity():
    preview_id, request_id = str(uuid.uuid4()), str(uuid.uuid4())
    calls = []
    def handle(request):
        calls.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json={'id': preview_id, 'kind': 'preflight', 'status': 'succeeded', 'artifact_id': 'fixed-package'})
        body = json.loads(request.content)
        assert body == {'kind': 'import', 'request_id': request_id, 'preflight_id': preview_id,
                        'artifact_id': 'fixed-package', 'maintenance_confirmed': True}
        return httpx.Response(202, json={'job_id': request_id})
    with httpx.Client(base_url='http://test/api/v1/migration/', transport=httpx.MockTransport(handle)) as client:
        result = execute(arguments('--request-id', request_id, '--no-wait', 'import', '--preflight-id', preview_id, '--yes'), MigrationClient(client))
    assert result['job_id'] == request_id
    assert len(calls) == 2


def test_upload_streams_package_and_submits_preflight(tmp_path):
    path = tmp_path / 'source.gz'
    path.write_bytes(b'abc' * 500000)
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith('/uploads'):
            assert request.headers['content-type'] == 'application/octet-stream'
            assert request.content == path.read_bytes()
            return httpx.Response(201, json={'artifact_id': 'uploaded'})
        assert json.loads(request.content)['artifact_id'] == 'uploaded'
        return httpx.Response(202, json={'job_id': 'task'})
    with httpx.Client(base_url='http://test/api/v1/migration/', transport=httpx.MockTransport(handle)) as client:
        execute(arguments('--no-wait', 'preflight', str(path)), MigrationClient(client))
    assert calls == ['/api/v1/migration/uploads', '/api/v1/migration/jobs']


@pytest.mark.parametrize('payload', [b'complete', b'truncated', b'complete-extra'])
def test_download_publishes_only_verified_bytes(tmp_path, payload):
    job = {'id': str(uuid.uuid4()), 'kind': 'export', 'status': 'succeeded', 'result': {
        'archive_bytes': 8, 'archive_sha256': hashlib.sha256(b'complete').hexdigest()}}
    path = tmp_path / 'export.gz'
    with httpx.Client(base_url='http://test/', transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))) as client:
        api = MigrationClient(client)
        if payload == b'complete':
            api.download(job, path)
            assert path.read_bytes() == payload
            with pytest.raises(FileExistsError):
                api.download(job, path)
            assert path.read_bytes() == payload
        else:
            with pytest.raises(ValueError):
                api.download(job, path)
            assert not path.exists()
    assert not list(tmp_path.glob('.ogmem-download-*'))


def test_lost_submission_reports_identity(capsys):
    identity = str(uuid.uuid4())
    def fail(request):
        raise httpx.ReadTimeout('response lost')
    with httpx.Client(base_url='http://test/', transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(httpx.ReadTimeout):
            MigrationClient(client).submit('export', identity)
    assert identity in capsys.readouterr().err


def test_wait_timeout_does_not_resubmit_or_cancel():
    identity = str(uuid.uuid4())
    calls = []
    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json={'status': 'running', 'phase': 'backing_up'})
    with httpx.Client(base_url='http://test/', transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TimeoutError, match=identity):
            MigrationClient(client).wait(identity, 0)
    assert calls == ['GET']


def test_root_key_required_and_provider_url_not_used(monkeypatch, capsys):
    monkeypatch.delenv('OGMEM_ROOT_API_KEY', raising=False)
    monkeypatch.setenv('OGMEM_BASE_URL', 'https://llm.invalid')
    assert run(arguments('status')) == 1
    assert 'OGMEM_ROOT_API_KEY' in capsys.readouterr().err


pytest_plugins = ['tests.unit.server.test_migration_api']


def test_cli_live_service_roundtrip_and_status_download(client, instance, tmp_path):
    from service.migration.package import open_package
    from tests.unit.service.migration.test_merge_recovery import package
    source = package(tmp_path)
    with httpx.Client(base_url='http://test/api/v1/migration/', headers={'X-API-Key': 'test-root-key'},
                      transport=httpx.WSGITransport(app=client.application)) as transport:
        api = MigrationClient(transport)
        preview = execute(arguments('preflight', str(source)), api)
        assert preview['status'] == 'succeeded'
        imported = execute(arguments('import', '--preflight-id', preview['id'], '--yes'), api)
        assert imported['status'] == 'succeeded'
        assert imported['result']['data']['nodes']['count'] == 2
        output = tmp_path / 'export.gz'
        exported = execute(arguments('export', '--output', str(output)), api)
        assert exported['status'] == 'succeeded'
        second = tmp_path / 'download.gz'
        execute(arguments('status', exported['id'], '--output', str(second)), api)
        assert second.read_bytes() == output.read_bytes()
        with open_package(second) as archive:
            assert len(list(archive.records('nodes'))) == 2
        assert instance[0]._get_shared_vector_index().count() == 4


def test_cli_recovers_interrupted_real_instance(client, instance, tmp_path, monkeypatch):
    from service.migration.jobs import InstanceJobs
    from service.migration.rebuild import Rebuilder
    from tests.unit.service.migration.test_merge_recovery import package
    service, migration, _ = instance
    queue = InstanceJobs(migration)
    with package(tmp_path).open('rb') as source:
        artifact = queue.upload(source)['artifact_id']
    original = str(uuid.uuid4())
    queue._save({'id': original, 'kind': 'import', 'artifact_id': artifact, 'status': 'running', 'created_at': 0})
    verify = Rebuilder.verify
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(Rebuilder, 'verify', interrupt)
    with pytest.raises(KeyboardInterrupt):
        migration.merge(queue.artifact(artifact), task=original)
    monkeypatch.setattr(Rebuilder, 'verify', verify)
    with httpx.Client(base_url='http://test/api/v1/migration/', headers={'X-API-Key': 'test-root-key'},
                      transport=httpx.WSGITransport(app=client.application)) as transport:
        result = execute(arguments('recover', original, '--yes'), MigrationClient(transport))
    assert result['status'] == 'succeeded'
    assert result['result']['outcome'] == 'restored'
    assert service._get_shared_vector_index().count() == 0


def test_failed_export_preserves_error_report_without_downloading(tmp_path):
    identity = str(uuid.uuid4())
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.method == 'POST':
            return httpx.Response(202, json={'job_id': identity})
        return httpx.Response(200, json={'id': identity, 'kind': 'export', 'status': 'failed',
                                       'phase': 'snapshotting', 'error': 'source unavailable'})
    output = tmp_path / 'failed.gz'
    with httpx.Client(base_url='http://test/', transport=httpx.MockTransport(handle)) as client:
        job = execute(arguments('export', '--output', str(output)), MigrationClient(client))
    assert job['error'] == 'source unavailable'
    assert not output.exists()
    assert not any(path.endswith('/download') for path in calls)


def test_preflight_retry_reuses_task_and_rejects_changed_package(client, tmp_path):
    from tests.unit.service.migration.test_merge_recovery import package
    source = package(tmp_path)
    request_id = str(uuid.uuid4())
    args = arguments('--request-id', request_id, 'preflight', str(source))
    with httpx.Client(base_url='http://test/api/v1/migration/', headers={'X-API-Key': 'test-root-key'},
                      transport=httpx.WSGITransport(app=client.application)) as transport:
        api = MigrationClient(transport)
        first = execute(args, api)
        assert first['status'] == 'succeeded'
        second = execute(args, api)
        assert second['id'] == first['id']
        assert second['artifact_id'] == first['artifact_id']
        assert len(api.request('GET', 'status')['jobs']) == 1
        source.write_bytes(b'another package')
        with pytest.raises(ValueError, match='package differs'):
            execute(args, api)
        assert len(api.request('GET', 'status')['jobs']) == 1
