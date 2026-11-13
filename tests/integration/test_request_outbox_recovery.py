"""Durable request projection across recovery, compaction and SQL ordering."""

import json
import threading
from dataclasses import asdict
from uuid import uuid4
from unittest.mock import Mock

import pytest
from pyagfs import AGFSClient
from commit.outbox_store import OutboxStore
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from providers.unified_config import OgMemConfig
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.llm.mock_llm import MockLLM
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from session.archive_store import SessionArchiveStore
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_request_outbox_events import request_context
from tests.integration.test_agfs_node_snapshot_reads import native_agfs
from tests.integration.test_sql_outbox_claim_ownership import sql_runtime, publish
from index.outbox_worker import OutboxWorker


def emit(**data):
    print('EVIDENCE ' + json.dumps(data, default=str, ensure_ascii=False))


@pytest.fixture
def runtime(native_agfs, monkeypatch):
    client = AGFSClient(native_agfs[0])
    client.session.trust_env = False
    fs = AGFSContextFS(client, mount_prefix='/local')
    store = OutboxStore(client, fs, mount_prefix='/local')
    ctx = request_context('review350-' + uuid4().hex[:12])
    index = InMemoryVectorIndex(16)
    services = []

    def service(directory=False):
        llm = MockLLM()
        writer = MemoryWriteAPI(fs, llm, store, directory_summary_enabled=directory)
        writer._pipeline.extract = Mock(return_value=[memory('receiver', content='The receiver was repaired.')])
        svc = MemoryService(OgMemConfig(session_state_bridge_enabled=False))
        svc._cfg.after_turn_threshold = 1
        svc._cfg.directory_summary_enabled = directory
        svc._cfg.scene_enabled = False
        svc._cfg.chunking_enabled = False
        svc._write_api = writer
        monkeypatch.setattr(svc, 'get_llm', lambda: llm)
        monkeypatch.setattr(svc, '_get_context_fs', lambda: fs)
        monkeypatch.setattr(svc, '_get_outbox_store', lambda context_fs: store)
        monkeypatch.setattr(svc, '_get_archive_store', lambda: SessionArchiveStore(fs))
        monkeypatch.setattr(svc, '_get_shared_projection_index', lambda: index)
        monkeypatch.setattr(svc, '_get_shared_embedder', lambda: MockEmbedder(16))
        monkeypatch.setattr('extraction.summary_generator.SessionSummaryGenerator.generate',
                            lambda *a, **kw: {'facts_and_details': 'The receiver was repaired.'})
        mgr = svc.get_session_manager()
        monkeypatch.setattr(mgr, '_compress', lambda *a, **kw: ('The receiver was repaired.', 'Receiver repair.'))
        services.append(svc)
        return svc, mgr

    yield ctx, fs, store, index, service
    for svc in services:
        svc.shutdown()
    client.session.close()


def request(ctx):
    return dict(sessionId=ctx.session_id, clientRequestId='receiver-repair', wait=True,
                messages=[dict(role='user', content='The receiver was repaired.')], _ctx=ctx)


def record_case(case, req, response, store, ctx, index):
    pending = [(uri, asdict(event)) for uri, event in store.list_pending(ctx.account_id)]
    emit(case=case, request={k: v for k, v in req.items() if k != '_ctx'},
         response=response, pending=pending, indexed=[(r.uri, r.text) for r in index._records.values()])
    return pending


@pytest.mark.parametrize('background', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('checkpoint_saved', [False, True])
def test_resumed_request_projects_checkpoint_events(runtime, monkeypatch, legacy, checkpoint_saved, background):
    ctx, fs, store, index, make = runtime
    svc, mgr = make()
    original = mgr.save_session_state

    def stop_after_accepted_write(session_id, context):
        batch = mgr.get_or_create(session_id).runtime_extensions.get(EXTENSION_KEY, {}).get('batch', {})
        if any(op.get('status') == 'complete' for slot in batch.get('slots', {}).values()
               for op in slot['operations'].values()):
            if checkpoint_saved:
                original(session_id, context)
            raise SystemExit('process interrupted at candidate completion checkpoint')
        return original(session_id, context)

    monkeypatch.setattr(mgr, 'save_session_state', stop_after_accepted_write)
    req = request(ctx)
    if background:
        req.pop('clientRequestId')
        req['messages'][0]['id'] = 'message-one'
    try:
        svc.after_turn(req)
    except SystemExit as exc:
        record_case('checkpoint-before', req, str(exc), store, ctx, index)
    if legacy:
        batch = mgr.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]['batch']
        batch.pop('outbox_tracking_version')
        batch.pop('outbox_events')
        original(ctx.session_id, ctx)
    restarted, _ = make()
    if background:
        req.pop('wait')
        req.pop('clientRequestId', None)
    result = restarted.after_turn(req)
    if background:
        for thread in threading.enumerate():
            if thread.name == f'extract-{ctx.session_id[:8]}':
                thread.join(timeout=30)
                assert not thread.is_alive()
    pending = record_case('checkpoint-resume-legacy-' + str(legacy), req, result, store, ctx, index)
    assert result['status'] == ('processing' if background else 'completed')
    assert pending == []
    assert any('/memories/' in r.uri for r in index._records.values())


@pytest.mark.parametrize('legacy', [False, True])
def test_separate_prepare_token_projects_accepted_memory(runtime, legacy):
    ctx, fs, store, index, make = runtime
    svc, mgr = make()
    mgr.add_message(ctx.session_id, 'user', 'The receiver was repaired.', ctx, message_id='message-one')
    mgr.save_session_state(ctx.session_id, ctx)
    prepared = svc.prepare_compaction(dict(sessionId=ctx.session_id, _ctx=ctx))
    if legacy:
        batch = mgr.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]['batch']
        batch.pop('outbox_tracking_version')
        batch.pop('outbox_events')
        mgr.save_session_state(ctx.session_id, ctx)
    emit(case='prepare', request={'sessionId': ctx.session_id},
         response={k: prepared.get(k) for k in ['ok', 'prepareToken', 'archive_id']})
    restarted, _ = make()
    req = dict(sessionId=ctx.session_id, _ctx=ctx, prepared=True, prepareToken=prepared['prepareToken'])
    result = restarted.compact(req)
    pending = record_case('compact-token', req, result, store, ctx, index)
    assert result['compacted']
    assert pending == []
    assert any('/memories/' in r.uri for r in index._records.values())


def test_directory_enabled_after_turn_projects_directory(runtime):
    ctx, fs, store, index, make = runtime
    svc, _ = make(directory=True)
    req = request(ctx)
    result = svc.after_turn(req)
    pending = record_case('directory-enabled', req, result, store, ctx, index)
    assert result['status'] == 'completed'
    assert pending == []
    assert any(r.uri.endswith('/entities/') for r in index._records.values())


def test_sql_request_projects_same_uri_predecessor(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    older, _ = publish(store, ctx, 'receiver', 'The receiver needs repair.')
    requested, record = publish(store, ctx, 'receiver', 'The receiver was repaired.')
    references = [(requested.uri, requested.event_id)]
    index = InMemoryVectorIndex(16)
    result = OutboxWorker(index, MockEmbedder(16)).run_once(
        store, [ctx.account_id], pending_events=store.list_pending_events(references))
    emit(case='sql-predecessor', request=asdict(requested), references=references,
         response=result, predecessor=asdict(store.get_event(ctx.account_id, older.event_id)),
         requested=asdict(store.get_event(ctx.account_id, requested.event_id)),
         indexed=[(r.uri, r.text) for r in index._records.values()])
    assert store.get_event(ctx.account_id, requested.event_id).status == 'DONE'
    assert index._records[record.id].text == 'The receiver was repaired.'


def test_agfs_backlog_preserves_latest_memory(runtime):
    from dataclasses import replace
    from tests.unit.commit.test_request_outbox_events import memory_node
    ctx, fs, store, index, make = runtime
    node = memory_node(ctx)
    fs.write_node(node, ctx)
    older = store.register_write(fs.read_node(node.uri, ctx), ctx)
    current = fs.read_node(node.uri, ctx)
    revised = replace(current, content='Receiver repaired and tested.',
                      overview='Receiver repaired and tested.', abstract='Receiver repaired and tested.',
                      metadata={**current.metadata, 'expected_version': 1})
    fs.write_node(revised, ctx)
    latest = store.register_write(fs.read_node(node.uri, ctx), ctx)
    worker = OutboxWorker(index, MockEmbedder(16), fs=fs, directory_summary_enabled=False)
    first = worker.run_once(store, [ctx.account_id],
                            pending_events=store.list_pending_events([(latest.uri, latest.event_id)]))
    before = [(r.uri, r.text) for r in index._records.values()]
    second = worker.run_once(store, [ctx.account_id])
    emit(case='agfs-backlog', request={'older': asdict(older), 'latest': asdict(latest)},
         response={'targeted': first, 'poll': second}, before=before,
         after=[(r.uri, r.text) for r in index._records.values()])
    assert all(r.text == 'Receiver repaired and tested.' for r in index._records.values())
    assert store.list_pending(ctx.account_id) == []
