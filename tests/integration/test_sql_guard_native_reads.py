"""Real SQL and Extractor; deterministic native protocol, no provider calls."""

import json
import os
import threading
from collections import Counter
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from commit.candidate_pipeline import CandidatePipeline
from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, operation_digest
from core.models import CandidateMemory, ContextNode, RequestContext, SeedHit
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.session_mutation import active_session_guard
from fs.sql_adapter.sql_context_fs import SQLContextFS
from providers.token_tracker import _active_usage_scope, request_usage_scope
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from session.session_state import SessionState
from session.sql_archive_store import SQLSessionArchiveStore
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex
from tests.unit.extraction.test_source_occurrence_reconciliation import TIME, SourceLLM, messages
from tests.unit.server.test_extraction_batch_recovery import NoModel


class NativeProtocolLLM:
    """Two real spans, each with the normal two native attempts and tool replies."""

    def __init__(self, uris, mode):
        self.uris, self.mode = uris, mode
        self.rounds, self.tool_replies, self.temperatures = [], [], []
        self.lock = threading.Lock()

    def complete_json(self, prompt, schema):
        assert 'spans' in schema['properties'], 'unexpected JSON fallback/model call'
        return {'spans': [
            {'start': 0, 'end': 0, 'categories': ['entity'], 'reason': 'alpha calibration'},
            {'start': 9, 'end': 9, 'categories': ['entity'], 'reason': 'beta calibration'},
        ]}

    def operation(self, text):
        key = 'alpha' if 'ALPHA_UPDATE' in text else 'beta'
        return {'name': 'edit_memory', 'input': {
            'target_uri': self.uris[key], 'expected_version': 1,
            'search': 'uncalibrated', 'replacement': 'calibrated',
            'abstract': f'Device {key} is calibrated.',
            'overview': f'Device {key} on the test bench is calibrated.',
            'confidence': .9,
        }}

    def record(self, phase):
        guard = active_session_guard()
        with self.lock:
            self.rounds.append({'phase': phase, 'thread': threading.get_ident(),
                                'owner_thread': guard.thread_id if guard else None,
                                'usage': _active_usage_scope.get()})

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        assert self.mode == 'lazy'
        self.record('lazy')
        conversation = next(message['content'] for message in messages if message['role'] == 'user')
        operation = self.operation(conversation)
        target = operation['input']['target_uri']
        replies = [json.loads(message['content']) for message in messages if message['role'] == 'tool']
        if not replies:
            return [
                {'id': 'read-body', 'name': 'read', 'input': {'uri': target}},
                {'id': 'list-catalog', 'name': 'list', 'input': {'uri': target.rsplit('/', 1)[0]}},
                {'id': 'read-relations', 'name': 'get_relations', 'input': {'uri': target}},
                {'id': 'read-stats', 'name': 'get_access_stats', 'input': {'uri': target}},
            ], ''
        if len(replies) == 4:
            with self.lock:
                self.tool_replies.append(replies)
            assert all('error' not in reply for reply in replies), replies
            assert replies[0]['content'].endswith('uncalibrated.')
            assert replies[0]['metadata']['version'] == 1
            return [dict(operation, id='edit-read-snapshot')], ''
        assert replies[-1]['status'] == 'recorded'
        return [], '[]'

    def complete_with_tools(self, *, prompt, tools, temperature=None, **kwargs):
        assert self.mode == 'eager', 'lazy native reads failed and fell back to eager'
        self.record('eager')
        with self.lock:
            self.temperatures.append(temperature)
        operation = self.operation(prompt)
        assert self.uris['alpha'] in prompt and self.uris['beta'] in prompt
        return [{'tool': operation['name'], 'input': operation['input']}]

    def complete(self, *args, **kwargs):
        raise AssertionError('unexpected summary/identity model call')


@pytest.fixture
def native_sql(monkeypatch):
    dsn = os.environ.get('OGMEM_IDENTITY_TEST_DSN')
    if not dsn:
        pytest.skip('set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database')
    if os.environ.get('OGMEM_SQL_NATIVE_LEGACY'):
        monkeypatch.setattr(SQLContextFS, 'requires_thread_bound_access', lambda self: False, raising=False)
        # Reproduce only the inner copied-context span bug, using the actual
        # serial pipeline/Extractor rather than replacing extraction results.
        monkeypatch.setattr(CandidatePipeline, '_extract_parallel', CandidatePipeline._extract_serial)
    pool = SharedConnectionPool(dsn, pool_size=1)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    archive = SQLSessionArchiveStore(pool=pool)
    ctx = RequestContext('native-sql-' + uuid4().hex, 'reader', 'agent', 'session', 'trace')
    directory = f'ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities'
    uris = {}
    for key in ('alpha', 'beta'):
        uri = f'{directory}/{key}.md'
        fs.write_node(ContextNode(
            uri=uri, context_type='MEMORY', category='entity', level=2,
            owner_space=ctx.user_space_name(), abstract=f'Device {key} is uncalibrated.',
            overview=f'Device {key} on the test bench is uncalibrated.',
            content=f'Device {key} is uncalibrated.',
            metadata={'routing_key': key, 'object_name': f'Device {key}', 'scope_ref': 'Test bench',
                      'owner_scope': 'user', 'confidence': .9},
        ), ctx)
        uris[key] = uri
    yield fs, outbox, archive, ctx, uris
    pool.close_all()


def source_messages():
    messages = [{'id': f'source-{index}', 'role': 'user', 'content': f'Neutral filler {index}.'}
                for index in range(10)]
    messages[0]['content'] = 'ALPHA_UPDATE: Device alpha was calibrated on the test bench.'
    messages[9]['content'] = 'BETA_UPDATE: Device beta was calibrated on the test bench.'
    return messages


@pytest.mark.parametrize('mode', ['lazy', 'eager'])
@pytest.mark.parametrize('entry', ['commit_session', 'after_turn', 'after_turn_background'])
def test_guarded_real_native_multispan_reads_preserve_owner_and_tool_protocol(native_sql, mode, entry):
    fs, outbox, archive, ctx, uris = native_sql
    llm = NativeProtocolLLM(uris, mode)
    registry = SchemaRegistry()
    index, embedder = FakeVectorIndex(), FakeEmbedder()
    index.hits = [SeedHit(uri=uri, category='entity', abstract=f'Device {key}', score=.95, level=2)
                  for key, uri in uris.items()]
    writer = MemoryWriteAPI(fs, llm, outbox, schema_registry=registry,
                            uri_resolver=URIResolver(registry), vector_index=index, embedder=embedder,
                            extraction_mode=mode, directory_summary_enabled=False)
    assert writer._pipeline._extractors[0].__class__.__name__ == 'Extractor'
    if entry == 'commit_session':
        with request_usage_scope(ctx), fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
            result = writer.commit_session(source_messages(), ctx, wait=True, archive_id='native-source')
            assert guard.poison_reason is None
            guard.ensure_owned()
        assert result['writes_completed'] == 2 and result['writes_failed'] == 0
    else:
        service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False, after_turn_threshold=1))
        service._write_api = writer
        service._generate_and_write_summary = Mock()
        service.drain_outbox_sync = Mock(return_value={})
        manager = service.get_session_manager()
        manager._get_archive_store = lambda: archive
        manager._compress = Mock(return_value=('Device calibrations', 'Device calibrations'))
        request = {'sessionId': ctx.session_id, 'messages': source_messages(), '_ctx': ctx}
        if entry == 'after_turn':
            request.update(clientRequestId='native-request', wait=True)
        result = service.after_turn(request)
        if entry == 'after_turn':
            assert result['status'] == 'completed', result
        else:
            assert result['status'] == 'processing', result
            assert manager.get_or_create(ctx.session_id).extraction_done_event.wait(10)
        assert len(archive.list_archives(ctx.session_id, ctx)) == 1
        assert not manager.get_or_create(ctx.session_id).messages
    expected_usage = (ctx.account_id, ctx.agent_space_name())
    assert llm.rounds and all(round['usage'] == expected_usage for round in llm.rounds)
    assert all(round['thread'] == round['owner_thread'] for round in llm.rounds)
    if mode == 'lazy':
        assert len(llm.tool_replies) == 4  # Two spans × two complete native read/edit runs.
        assert len(llm.rounds) == 12
    else:
        assert Counter(llm.temperatures) == Counter({None: 2, 0: 2})
    for key, uri in uris.items():
        node = fs.read_node(uri, ctx)
        assert node.content == f'Device {key} is calibrated.'
        assert node.metadata['version'] == 2
    assert len([event for uri, event in outbox.list_pending(ctx.account_id) if '/entities/' in uri]) == 2


class GuardedSourceProtocolLLM(SourceLLM):
    """Actual source reconciliation with every protocol round on the owner."""

    def __init__(self):
        super().__init__()
        self.owner_rounds = []

    def record_owner(self, phase):
        guard = active_session_guard()
        assert guard is not None
        guard.check_active()
        assert guard.borrowed == 0
        with guard.connection.cursor() as cursor:
            cursor.execute('SELECT pg_backend_pid()')
            pid = cursor.fetchone()[0]
        guard.connection.rollback()
        self.owner_rounds.append((phase, threading.get_ident(), guard.thread_id, pid,
                                  _active_usage_scope.get()))

    def complete_json(self, prompt, schema):
        phase = next(iter(schema['properties']))
        self.record_owner(phase)
        if phase != 'spans':
            payload = json.loads(prompt.split('\n\n', 1)[1])
            drafts = payload['member_drafts'] if phase == 'valid' else payload['drafts']
            assert all(not draft['draft'].get('source_occurrence_id')
                       and not draft['draft'].get('write_operation_id')
                       for draft in drafts)
        return super().complete_json(prompt, schema)

    def complete_with_tools(self, **kwargs):
        self.record_owner('native')
        return super().complete_with_tools(**kwargs)


def source_writer(fs, outbox, llm):
    return MemoryWriteAPI(fs, llm, outbox, vector_index=FakeVectorIndex(),
                          embedder=FakeEmbedder(), directory_summary_enabled=False,
                          extraction_mode='eager', source_occurrence_reconciliation_enabled=True)


def assert_source_protocol_owner(llm, ctx):
    assert all(thread == owner and usage == (ctx.account_id, ctx.agent_space_name())
               for _, thread, owner, _, usage in llm.owner_rounds)
    assert len({pid for _, _, _, pid, _ in llm.owner_rounds}) == 1


def test_guarded_eager_source_multispan_keeps_stable_archive_namespace(native_sql):
    fs, outbox, _, ctx, _ = native_sql
    llm = GuardedSourceProtocolLLM()
    writer = source_writer(fs, outbox, llm)
    assert writer._pipeline._extractors[0].__class__.__name__ == 'Extractor'
    with request_usage_scope(ctx), fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        def extract(archive_id):
            return writer._pipeline.extract(messages(), ctx, session_time=TIME,
                                            archive_id=archive_id, parallel=False)
        first, second, another = extract('source-archive'), extract('source-archive'), extract('another-archive')
        assert len(first) == len(second) == len(another) == 1
        assert first[0].source_occurrence_id == second[0].source_occurrence_id
        assert first[0].source_occurrence_id != another[0].source_occurrence_id
        assert '5 bar' in first[0].content and 'seal leak' in first[0].content
        assert not first[0].write_operation_id
        guard.ensure_owned()
    assert_source_protocol_owner(llm, ctx)
    assert Counter(phase for phase, *_ in llm.owner_rounds) == Counter(
        {'spans': 3, 'native': 12, 'groups': 3, 'valid': 3})
    assert Counter(temperature for _, temperature in llm.native_calls) == Counter({None: 6, 0: 6})


def test_public_guarded_source_receipt_restart_does_not_reextract_or_duplicate(native_sql):
    fs, outbox, archive, ctx, _ = native_sql
    llm = GuardedSourceProtocolLLM()
    writer = source_writer(fs, outbox, llm)
    storage_rounds = []

    def trace_connections(stores, label):
        for name, store in stores.items():
            original_get = store._get_connection

            def get_owned(original_get=original_get, name=name):
                conn = original_get()
                guard = active_session_guard()
                if guard is not None:
                    assert conn is guard.connection
                    guard.check_context(ctx)
                    storage_rounds.append((label, name, conn.get_backend_pid(), threading.get_ident()))
                return conn

            store._get_connection = get_owned

    trace_connections({'fs': fs, 'outbox': outbox, 'archive': archive}, 'initial')

    def service_for(writer, archive):
        service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False,
                                                   after_turn_threshold=1,
                                                   source_occurrence_reconciliation_enabled=True))
        service._write_api = writer
        service._generate_and_write_summary = Mock()
        service.drain_outbox_sync = Mock(return_value={})
        manager = service.get_session_manager()
        manager._get_archive_store = lambda: archive
        manager._compress = Mock(return_value=('Atlas inspection', 'Atlas inspection'))
        return service, manager

    service, manager = service_for(writer, archive)
    original = fs.write_node_with_outbox
    original_read = fs.read_node
    unread_receipts = set()

    def read_after_connection_loss(uri, request):
        if uri in unread_receipts:
            unread_receipts.remove(uri)
            raise ConnectionError('receipt unavailable after connection loss')
        return original_read(uri, request)

    def commit_then_lose_ack(node, request, outbox_event=None):
        original(node, request, outbox_event)
        if node.category == 'event':
            unread_receipts.add(node.uri)
            raise RuntimeError('deterministic acknowledgement loss after the real SQL commit')

    fs.write_node_with_outbox = commit_then_lose_ack
    fs.read_node = read_after_connection_loss
    # after_turn intentionally omits assistant messages from extraction. Keep
    # the two actual user sources separated in its public extraction input.
    public_messages = [{**message, 'role': 'user', 'created_at': TIME.isoformat(),
                        'content': f'Neutral filler {index}.' if message['role'] == 'assistant'
                        else message['content']}
                       for index, message in enumerate(messages())]
    request = {'sessionId': ctx.session_id, 'clientRequestId': 'source-request',
               'messages': public_messages, 'wait': True, '_ctx': ctx}
    result = service.after_turn(request)
    assert result['status'] == 'needs_confirmation', result
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]['batch']
    slot = batch['slots']['chunk:0']
    accepted = slot['candidates'][0]
    assert len(slot['candidates']) == 1 and slot['operations']['0']['status'] == 'writing'
    assert len(accepted['source_occurrence_id']) == 64 and len(accepted['write_operation_id']) == 64
    directory = writer._writer._identity_resolver.uris.get_directory_uri('event', ctx)
    event_uri, = fs.list_children(directory, ctx)
    before = fs.read_node(event_uri, ctx)
    assert before.owner_space == ctx.user_space_name()
    assert before.metadata[RECEIPTS_KEY] == {accepted['write_operation_id']: slot['candidate_digests'][0]}
    assert before.metadata['provenance_ids'] == accepted['provenance_ids']
    assert '5 bar' in before.content and 'seal leak' in before.content
    assert len([entry for uri, entry in outbox.list_pending(ctx.account_id) if '/events/' in uri]) == 1
    assert_source_protocol_owner(llm, ctx)

    # An independent service and new SQL pool/connection must load the accepted
    # candidates and their source partition IDs, with no Extractor/provider call.
    pool2 = SharedConnectionPool(os.environ['OGMEM_IDENTITY_TEST_DSN'], pool_size=1)
    try:
        fs2 = SQLContextFS(pool=pool2)
        outbox2 = SQLOutboxStore(pool=pool2, fs=fs2)
        archive2 = SQLSessionArchiveStore(pool=pool2)
        trace_connections({'fs': fs2, 'outbox': outbox2, 'archive': archive2}, 'restart')
        writer2 = source_writer(fs2, outbox2, NoModel())
        restarted, manager2 = service_for(writer2, archive2)
        assert writer2._pipeline._extractors[0].__class__.__name__ == 'Extractor'
        retried = restarted.after_turn({**request, '_ctx': replace(ctx, trace_id='source-retry')})
        assert retried['status'] == 'completed', retried
        assert fs2.list_children(directory, ctx) == [event_uri]
        assert fs2.read_node(event_uri, ctx) == before
        assert len([entry for uri, entry in outbox2.list_pending(ctx.account_id) if '/events/' in uri]) == 1
        assert len(archive2.list_archives(ctx.session_id, ctx)) == 1
        assert manager2.get_or_create(ctx.session_id).request_completed('source-request')
        assert EXTENSION_KEY not in manager2.get_or_create(ctx.session_id).runtime_extensions
        assert operation_digest(CandidateMemory(**accepted), ctx) == slot['candidate_digests'][0]
        for label in ('initial', 'restart'):
            rounds = [entry for entry in storage_rounds if entry[0] == label]
            assert rounds and len({(pid, thread) for _, _, pid, thread in rounds}) == 1
        assert {pid for label, _, pid, _ in storage_rounds if label == 'initial'}.isdisjoint(
            pid for label, _, pid, _ in storage_rounds if label == 'restart')
        assert {'fs', 'archive'} <= {name for label, name, _, _ in storage_rounds if label == 'restart'}
    finally:
        pool2.close_all()
