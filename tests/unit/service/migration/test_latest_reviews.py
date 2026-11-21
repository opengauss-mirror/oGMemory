"""Review regressions at session, AGFS rejection and snapshot boundaries."""
import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.errors import ConcurrentModificationError, NodeNotFoundError
from core.models import ContextNode, RequestContext, Role
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceStore
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


class RejectionClient(_AtomicMemoryAGFSClient):
    def _request(self, method, *args, **kwargs):
        from pyagfs.exceptions import AGFSClientError
        from requests import Response
        from requests.exceptions import HTTPError
        try:
            return method(*args, **kwargs)
        except AGFSClientError as exc:
            response = Response()
            response.status_code = 409 if 'exist' in str(exc) else 404
            try:
                raise HTTPError(response=response)
            except HTTPError:
                raise AGFSClientError(str(exc))

    def mv(self, *args, **kwargs):
        return self._request(super().mv, *args, **kwargs)

    def rm(self, *args, **kwargs):
        return self._request(super().rm, *args, **kwargs)


@pytest.mark.parametrize('operation', ['create', 'move', 'delete'])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_settled_agfs_rejection_drains(tmp_path, operation, prior_unknown):
    client = RejectionClient()
    fs = AGFSContextFS(client=client, mount_prefix='/local')
    ctx = RequestContext('acme', 'alice', 'agent', 's', 'review', Role.ROOT)
    node = ContextNode(uri='ctx://acme/users/alice/memories/preferences/editor',
                       context_type='MEMORY', category='preferences', owner_space='user:alice',
                       level=2, abstract='original', overview='original', content='original', metadata={'version': 1})
    if operation == 'create':
        fs.write_node(node, ctx)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    admission.guard(fs)
    with admission.scope():
        if prior_unknown:
            with pytest.raises(TimeoutError), admission.backend(write=True):
                raise TimeoutError("previous write remains unknown")
        with pytest.raises((ConcurrentModificationError, NodeNotFoundError)):
            if operation == 'create':
                fs.write_node(replace(node, metadata={'_expected_version': 0}), ctx)
            elif operation == 'move':
                fs.move_node(node.uri, node.uri + '-other', ctx)
            else:
                fs.delete_node(node.uri, ctx)
    gate.enter('export', mode='export', resources={})
    if prior_unknown:
        with pytest.raises(TimeoutError):
            gate.wait_drained('export', 0)
    else:
        gate.wait_drained('export', 0)


def test_old_session_checkpoint_cannot_overwrite_newer_state(tmp_path, monkeypatch):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle
    from session.session_manager import SessionManager
    from tests.integration.conftest import InMemoryContextFS

    fs = InMemoryContextFS()
    ctx = RequestContext('acme', 'alice', 'agent', 's', 'review')
    old = SessionManager(get_context_fs=lambda: fs)
    old.add_message('s', 'user', 'old message', ctx)
    assert old.save_session_state('s', ctx)
    newer = SessionManager(get_context_fs=lambda: fs)
    newer.add_message('s', 'user', 'new message', ctx)
    assert newer.save_session_state('s', ctx)
    before = next(iter(fs._store.values())).content
    service = MemoryService(OgMemConfig(skill_sedimentation_enabled=False))
    service._session_mgr = old
    monkeypatch.setattr(service, '_session_mutation_scope', lambda *a, **kw: nullcontext())
    try:
        ServiceLifecycle(service)._save_sessions()
        assert next(iter(fs._store.values())).content == before
    finally:
        service._session_mgr = None
        service.shutdown()


def test_warm_archive_getter_is_available_during_export(tmp_path, monkeypatch):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.unit.service.migration.test_tree_backup import LocalClient

    client = LocalClient(tmp_path / 'files')
    client.path('/local').mkdir(parents=True)
    monkeypatch.setattr('server.memory_service.AGFSClient', lambda **kw: client)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    service = MemoryService(OgMemConfig(storage_backend='agfs', skill_sedimentation_enabled=False),
                            admission=Admission(gate))
    original = service._get_archive_store()
    assert original is not None
    gate.enter('export', mode='export', resources={})
    try:
        assert service._get_archive_store() is original
    finally:
        gate.reopen('export')
        service.shutdown()




@pytest.mark.parametrize('failure', ['cleanup', 'transport'])
def test_unsettled_create_conflict_keeps_permit(tmp_path, monkeypatch, failure):
    from pyagfs.exceptions import AGFSClientError

    client = RejectionClient()
    fs = AGFSContextFS(client=client, mount_prefix='/local')
    ctx = RequestContext('acme', 'alice', 'agent', 's', 'review', Role.ROOT)
    node = ContextNode(uri='ctx://acme/users/alice/memories/preferences/editor',
                       context_type='MEMORY', category='preferences', owner_space='user:alice',
                       level=2, abstract='original', overview='original', content='original', metadata={})
    fs.write_node(node, ctx)
    if failure == 'cleanup':
        monkeypatch.setattr(fs, '_remove_tree_if_present', lambda path: False)
    else:
        def timeout(*args):
            raise AGFSClientError('request timeout')
        monkeypatch.setattr(client, 'mv', timeout)
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    Admission(gate).guard(fs)
    with pytest.raises(ConcurrentModificationError):
        fs.write_node(replace(node, metadata={'_expected_version': 0}), ctx)
    gate.enter('export', mode='export', resources={})
    with pytest.raises(TimeoutError):
        gate.wait_drained('export', 0)


@pytest.mark.parametrize('status,error,endpoint,expected', [
    (500, 'failed to rename: rename /staging /target: file exists', 'rename', False),
    (500, 'failed to rename: rename /staging /target: file exists', 'files', False),
    (500, 'Internal server error', 'rename', False),
    (502, 'failed to rename: rename /staging /target: file exists', 'rename', False),
])
def test_legacy_rename_rejection_requires_exact_http_evidence(status, error, endpoint, expected):
    from pyagfs.exceptions import AGFSClientError
    from requests import Request, Response
    from requests.exceptions import HTTPError

    from fs.agfs_adapter.client import confirmed_rejection

    response = Response()
    response.status_code = status
    response.request = Request('POST', 'http://agfs/' + endpoint).prepare()
    response._content = json.dumps({'error': error}).encode()
    try:
        try:
            raise HTTPError(response=response)
        except HTTPError:
            raise AGFSClientError(error)
    except AGFSClientError as exc:
        assert confirmed_rejection(exc) is expected
