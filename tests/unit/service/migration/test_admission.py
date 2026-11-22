import asyncio
from contextlib import contextmanager

import pytest

from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceError, MaintenanceStore


@pytest.fixture
def admission(tmp_path):
    return Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))


def test_export_rejects_upgrade_without_leaking_reader(admission):
    with admission.scope():
        admission.store.enter('export', mode='export', resources={})
        with pytest.raises(MaintenanceError):
            with admission.backend(write=True):
                pytest.fail('backend must not be called')
    admission.store.wait_drained('export', 0)
    admission.store.reopen('export')
    admission.store.enter('import', mode='import', resources={})
    admission.store.wait_drained('import', 0)


def test_unknown_write_completion_keeps_gate_closed(admission):
    with pytest.raises(TimeoutError):
        with admission.scope():
            with admission.backend(write=True):
                raise TimeoutError('response lost')
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)


def test_native_rejection_preserves_prior_uncertain_write(admission):
    from fs.agfs_adapter.client import NativeAGFSHTTPError

    with admission.scope():
        with pytest.raises(TimeoutError):
            with admission.backend(write=True):
                raise TimeoutError('response lost')
        with pytest.raises(NativeAGFSHTTPError):
            with admission.backend(write=True):
                raise NativeAGFSHTTPError('already exists', status_code=409)
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)


def test_rejected_business_parameters_do_not_leak_operation(admission):
    with pytest.raises(ValueError):
        with admission.scope():
            raise ValueError('invalid input, backend not called')
    admission.store.enter('import', mode='import', resources={})
    admission.store.wait_drained('import', 0)


def test_generator_holds_admission_until_closed(admission):
    @admission.wrap
    def stream():
        yield 'first'
        yield 'last'
    iterator = stream()
    assert next(iterator) == 'first'
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)
    iterator.close()
    admission.store.wait_drained('import', 0)


def test_context_manager_preserves_value_and_lifetime(admission):
    @admission.wrap
    @contextmanager
    def locked():
        yield 'lease'
    with locked() as lease:
        assert lease == 'lease'
        admission.store.enter('import', mode='import', resources={})
        with pytest.raises(TimeoutError):
            admission.store.wait_drained('import', 0)
    admission.store.wait_drained('import', 0)


@pytest.mark.asyncio
async def test_async_child_has_independent_lifetime(admission):
    started, release = asyncio.Event(), asyncio.Event()

    @admission.wrap
    async def child():
        with admission.backend(write=True):
            started.set()
            await release.wait()

    with admission.scope():
        task = asyncio.create_task(child())
        await started.wait()
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)
    release.set()
    await task
    admission.store.wait_drained('import', 0)


@pytest.mark.asyncio
async def test_delayed_child_cannot_bypass_gate(admission):
    release = asyncio.Event()

    async def child():
        await release.wait()
        with admission.scope():
            pytest.fail('must not enter maintenance')

    with admission.scope():
        task = asyncio.create_task(child())
    admission.store.enter('import', mode='import', resources={})
    release.set()
    with pytest.raises(MaintenanceError):
        await task
    admission.store.wait_drained('import', 0)


def test_actual_adapter_remains_guarded_after_service_returns(admission):
    class Storage:
        def read_node(self):
            return 'data'
        def write_node(self):
            return 'written'
    storage = admission.guard(Storage())
    assert isinstance(storage, Storage)
    assert storage.read_node() == 'data'
    admission.store.enter('export', mode='export', resources={})
    assert storage.read_node() == 'data'
    with pytest.raises(MaintenanceError):
        storage.write_node()
    admission.store.reopen('export')
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        storage.read_node()
    admission.store.wait_drained('import', 0)


def test_agfs_read_remains_available_during_snapshot(admission, tmp_path):
    from pyagfs.exceptions import AGFSClientError

    from core.models import RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from service.migration.storage import AGFSStorage
    from tests.unit.service.migration.test_engine import config, node
    from tests.migration_fixtures import LocalClient

    class AdapterClient(LocalClient):
        def read(self, path, offset=0, size=-1):
            try:
                return super().read(path, offset=offset, size=size)
            except FileNotFoundError as exc:
                raise AGFSClientError(str(exc)) from exc

    client = AdapterClient(tmp_path / 'files')
    client.path('/local').mkdir(parents=True)
    AGFSStorage(config(), client).restore('nodes', [node()])
    fs = admission.guard(AGFSContextFS(client, mount_prefix='/local'))
    ctx = RequestContext('acme', 'alice', '', '', 'test')
    admission.store.enter('export', mode='export', resources={})
    assert fs.read_node(node()['uri'], ctx).content == node()['content']
    with pytest.raises(MaintenanceError):
        fs._recover_acknowledging_delete('unused', 'unused', {})
    admission.store.wait_drained('export', 0)


def test_http_stream_is_drained_and_control_page_stays_available(admission):
    from types import SimpleNamespace

    from service.migration.admission import business_wsgi

    def application(environ, start):
        start('200 OK', [])
        yield b'first'
        yield b'last'
    response_status = []
    serve = business_wsgi(application, lambda: SimpleNamespace(_admission=admission))
    def start(status, headers):
        response_status.append(status)
    response = serve({'PATH_INFO': '/api/v1/memories'}, start)
    assert next(response) == b'first'
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)
    rejected = b''.join(serve({'PATH_INFO': '/api/v1/memories'}, start))
    assert b'migration_maintenance' in rejected
    assert response_status[-1].startswith('503')
    assert b''.join(serve({'PATH_INFO': '/api/v1/migration/status'}, start)) == b'firstlast'
    response.close()
    admission.store.wait_drained('import', 0)


@pytest.mark.asyncio
async def test_mcp_request_uses_same_maintenance_gate(admission):
    from types import SimpleNamespace

    from server.mcp_server import _CaptureMCPRequestContext

    middleware = _CaptureMCPRequestContext(SimpleNamespace(_admission=admission))
    async def call_next(ctx):
        return 'memory'
    assert await middleware(None, call_next) == 'memory'
    admission.store.enter('import', mode='import', resources={})
    from mcp.shared.exceptions import MCPError
    with pytest.raises(MCPError, match='migration_maintenance') as rejected:
        await middleware(None, call_next)
    assert rejected.value.data == {'status': 503, 'retryable': True}


def test_trace_runtime_repair_continues_during_export(admission):
    from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
    from tests.unit.fs.test_agfs_trace_store import _MemoryAGFS

    client = _MemoryAGFS()
    trace = admission.guard(AGFSTraceEventStore(client=client))
    admission.store.enter('export', mode='export', resources={})
    assert trace.list_pending_skill_accounts() == []
    admission.store.wait_drained('export', 0)
    admission.store.reopen('export')
    assert trace.list_pending_skill_accounts() == []
    assert '/local/.ogmemory/trace-store/work-accounts-index.json' in client.files
    # Once repaired the same read needs no mutation and remains available.
    admission.store.enter('next', mode='export', resources={})
    before = dict(client.files)
    assert trace.list_pending_skill_accounts() == []
    assert client.files == before


def test_background_handoff_is_reserved_before_child_starts(admission, monkeypatch):
    from types import SimpleNamespace

    calls = []
    def delayed_thread(*, target, name, daemon):
        calls.append(target)
        return SimpleNamespace(start=lambda: None, ident=None)
    monkeypatch.setattr('service.migration.admission.threading.Thread', delayed_thread)
    admission.launch(lambda: None)
    admission.store.enter('import', mode='import', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('import', 0)
    calls.pop()()
    admission.store.wait_drained('import', 0)


def test_thread_start_failure_releases_handoff(admission, monkeypatch):
    from types import SimpleNamespace

    def fail():
        raise RuntimeError('cannot start thread')
    monkeypatch.setattr('service.migration.admission.threading.Thread',
                        lambda **_: SimpleNamespace(start=fail, ident=None))
    with pytest.raises(RuntimeError, match='cannot start'):
        admission.launch(lambda: None)
    admission.store.enter('import', mode='import', resources={})
    admission.store.wait_drained('import', 0)


def test_wsgi_garbage_collection_closes_in_original_context(admission):
    import gc
    from types import SimpleNamespace

    from service.migration.admission import business_wsgi

    def app(environ, start_response):
        start_response('200 OK', [])
        yield b'first'
        yield b'last'
    wrapped = business_wsgi(app, lambda: SimpleNamespace(_admission=admission))
    response = wrapped({'PATH_INFO': '/business'}, lambda *a: None)
    assert next(response) == b'first'
    admission.store.enter('import', mode='import', resources={})
    del response
    gc.collect()
    admission.store.wait_drained('import', 0)


@pytest.mark.parametrize('status', [403, 404, 409, 500, 502, 504])
@pytest.mark.parametrize('method, verb', [('mkdir', 'post'), ('rm', 'delete')])
def test_http_agfs_rejection_admission(admission, monkeypatch, status, method, verb):
    import requests
    from pyagfs import AGFSClient
    from pyagfs.exceptions import AGFSClientError

    client = admission.guard(AGFSClient())
    response = requests.Response()
    response.status_code = status
    response._content = b'{"error":"filesystem rejection"}'
    monkeypatch.setattr(client.session, verb, lambda *args, **kwargs: response)
    with pytest.raises(AGFSClientError):
        getattr(client, method)('/existing')
    admission.store.enter('export', mode='export', resources={})
    if status in {403, 404, 409}:
        admission.store.wait_drained('export', 0)
    else:
        with pytest.raises(TimeoutError):
            admission.store.wait_drained('export', 0)


def test_http_agfs_cat_is_read_during_export(admission, monkeypatch):
    import requests
    from pyagfs import AGFSClient

    client = admission.guard(AGFSClient())
    response = requests.Response()
    response.status_code = 200
    response._content = b'hello'
    monkeypatch.setattr(client.session, 'get', lambda *args, **kwargs: response)
    admission.store.enter('export', mode='export', resources={})
    assert client.cat('/file') == b'hello'
    admission.store.wait_drained('export', 0)


@pytest.mark.parametrize('first', ['timeout', 502])
@pytest.mark.parametrize('later', [200, 409])
def test_http_write_does_not_retry_unknown_completion(admission, monkeypatch, first, later):
    import requests
    from pyagfs import AGFSClient
    from pyagfs.exceptions import AGFSClientError

    client = admission.guard(AGFSClient())
    calls = []
    def put(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1 and first == 'timeout':
            raise requests.exceptions.ReadTimeout('unknown completion')
        response = requests.Response()
        response.status_code = first if len(calls) == 1 else later
        response._content = b'{"error":"rejected","message":"OK"}'
        return response
    monkeypatch.setattr(client.session, 'put', put)
    with pytest.raises(AGFSClientError):
        client.write('/file', b'payload', max_retries=3)
    assert len(calls) == 1
    admission.store.enter('export', mode='export', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('export', 0)


@pytest.mark.parametrize('confirmed', [False, True])
def test_sql_conflict_releases_only_confirmed_rollback(admission, confirmed):
    from core.errors import ConcurrentModificationError
    exc = ConcurrentModificationError('ctx://a/node', 1, 2)
    if confirmed:
        exc.sql_rollback_confirmed = True
    with pytest.raises(ConcurrentModificationError):
        with admission.backend(write=True):
            raise exc
    admission.store.enter('export', mode='export', resources={})
    if confirmed:
        admission.store.wait_drained('export', 0)
    else:
        with pytest.raises(TimeoutError):
            admission.store.wait_drained('export', 0)


@pytest.mark.parametrize('method', ['delete_node', 'archive_node', 'move_node'])
@pytest.mark.parametrize('rollback_fails', [False, True])
def test_sql_missing_node_settles_only_after_rollback(admission, method, rollback_fails):
    from unittest.mock import MagicMock

    from core.errors import NodeNotFoundError
    from core.models import RequestContext, Role
    from fs.sql_adapter.sql_context_fs import SQLContextFS

    fs = SQLContextFS.__new__(SQLContextFS)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 0
    cur.fetchone.return_value = None
    fs._get_connection = lambda: conn
    fs._return_connection = lambda connection: None
    if rollback_fails:
        conn.rollback.side_effect = ConnectionError('rollback unknown')
    call = admission.wrap(getattr(fs, method), write=True, backend=True)
    uri = 'ctx://acme/users/alice/memories/preferences/missing'
    ctx = RequestContext('acme', 'alice', '', 's', 'test', Role.ROOT)
    args = (uri, uri + '-moved', ctx) if method == 'move_node' else (uri, ctx)
    with pytest.raises(ConnectionError if rollback_fails else NodeNotFoundError):
        call(*args)
    conn.rollback.assert_called_once()
    with admission.store._transaction() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0] == int(rollback_fails)


def test_settled_sql_rejection_does_not_clear_prior_unknown_write(admission):
    from core.errors import NodeNotFoundError
    with admission.scope(write=True):
        with pytest.raises(TimeoutError), admission.backend(write=True):
            raise TimeoutError('write completion unknown')
        exc = NodeNotFoundError('ctx://acme/missing')
        exc.sql_rollback_confirmed = True
        with pytest.raises(NodeNotFoundError), admission.backend(write=True):
            raise exc
    admission.store.enter('export', mode='export', resources={})
    with pytest.raises(TimeoutError):
        admission.store.wait_drained('export', 0)


@pytest.mark.parametrize('method', ['delete_node', 'archive_node', 'move_node'])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_sql_access_rejection_precedes_connection(admission, method, prior_unknown):
    from core.errors import AccessDeniedError
    from core.models import RequestContext, Role
    from fs.sql_adapter.sql_context_fs import SQLContextFS

    fs = SQLContextFS.__new__(SQLContextFS)
    fs._get_connection = lambda: pytest.fail('validation must precede SQL access')
    ctx = RequestContext('acme', 'alice', '', 's', 'test', Role.ROOT)
    call = admission.wrap(getattr(fs, method), write=True, backend=True)
    uri = 'ctx://acme/users/alice/memories/preferences/item'
    args = (uri, 'invalid', ctx) if method == 'move_node' else ('invalid', ctx)
    with admission.scope(write=True):
        if prior_unknown:
            with pytest.raises(TimeoutError), admission.backend(write=True):
                raise TimeoutError('unknown completion')
        with pytest.raises(AccessDeniedError):
            call(*args)
    with admission.store._transaction() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0] == int(prior_unknown)
