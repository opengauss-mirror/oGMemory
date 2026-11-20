"""Exercise shared AGFS hybrid retrieval with a real SQL catalog and Outbox."""

import os
import time
from uuid import uuid4

import pytest

from core.models import ContextNode, RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin
from index.outbox_worker import OutboxWorker
from providers.composite_index import create_projection_writer
from providers.unified_config import OgMemConfig
from retrieval.identity import canonical_uri
from retrieval.seed_retriever import RetrievalConfigError
from scripts.run_index_service import _create_storage
from server.memory_service import MemoryService


@pytest.fixture(params=['opengauss_native', 'sqlite_fts'])
def setup(request, tmp_path):
    dsn = os.environ.get('OGMEM_OPENGAUSS_SQL_DSN')
    agfs_url = os.environ.get('OGMEM_AGFS_TEST_URL')
    if not dsn or not agfs_url:
        pytest.skip('requires isolated openGauss and AGFS services')
    options = {'path': str(tmp_path / 'lexical.db')} if request.param == 'sqlite_fts' else {'connection_string': dsn}
    mount = os.environ.get('OGMEM_AGFS_TEST_MOUNT', '/local').rstrip('/')
    cfg = OgMemConfig(
        provider='mock', vector_db_type='memory', storage_backend='agfs',
        shared_spaces_enabled=True, role_control_enabled=True, root_api_key=uuid4().hex,
        sql_connection_string=dsn, sql_dialect='opengauss', agfs_transport='http',
        agfs_base_url=agfs_url, agfs_mount_prefix=f'{mount}/shared-lexical-{uuid4().hex}',
        index_workers=0,
        retrieval_hybrid_enabled=True, retrieval_expand_hierarchy=False,
        retrieval_lexical_provider=request.param, retrieval_lexical_options=options,
        directory_summary_enabled=False,
    )
    ctx = RequestContext('sharedlex-' + uuid4().hex[:8], 'alice', '', 'test', 'pr401')
    service = MemoryService(cfg)
    try:
        yield service, ctx, service._get_context_fs(), service.get_read_api(start_outbox_worker=False)
    finally:
        service.shutdown()


def store(fs, ctx, name, content):
    node = ContextNode(f'ctx://{ctx.account_id}/users/alice/memories/entities/{name}',
                       'MEMORY', 'entity', 2, 'user:alice', 'General abstract', '', content)
    fs.write_node(node, ctx)
    return fs.read_node(node.uri, ctx)


def search(api, ctx, text):
    try:
        result = api.search_memory(text, ctx, top_k=1, context_type='MEMORY', score_threshold=0)
        response = {'hits': [canonical_uri(hit.uri) for hit in result.hits], 'partial': result.partial}
    except RetrievalConfigError as exc:
        response = {'error': str(exc)}
    print({'request': {'query': text, 'top_k': 1, 'account_id': ctx.account_id}, 'response': response}, flush=True)
    return response


def wait_for_history(api, ctx, history):
    expires = time.monotonic() + 10
    while time.monotonic() < expires:
        result = search(api, ctx, 'historicmarker')
        if result == {'hits': [history.uri], 'partial': False}:
            return
        time.sleep(.5)
    pytest.fail(f'historical body did not become searchable: {result}')


def test_shared_historical_body(setup):
    service, ctx, fs, api = setup
    history = store(fs, ctx, 'old', 'historicmarker')
    pool = service._get_shared_sql_pool()
    conn = pool.get_connection()
    try:
        with conn, conn.cursor() as cur:
            PoolAdapterMixin.bind_tenant(conn, ctx.account_id)
            cur.execute("UPDATE context_nodes SET content='' WHERE account_id=%s AND uri=%s",
                        (ctx.account_id, history.uri))
            assert cur.rowcount == 1
    finally:
        pool.return_connection(conn)
    assert fs.read_node(history.uri, ctx).content == 'historicmarker'
    wait_for_history(api, ctx, history)


def test_shared_sql_outbox_gates_ready_index(setup):
    service, ctx, fs, api = setup
    wait_for_history(api, ctx, store(fs, ctx, 'old', 'historicmarker'))
    new = store(fs, ctx, 'new', 'newbodymarker')
    outbox = service._get_outbox_store(fs)
    event = outbox.register_write(new, ctx)
    print({'request': {'event_id': event.event_id}, 'response': outbox.get_event_status(new.uri, event.event_id)}, flush=True)
    assert search(api, ctx, 'historicmarker') == {'hits': [], 'partial': True}
    worker_fs, worker_outbox, pool = _create_storage(service._cfg)
    projection = create_projection_writer(service._cfg, service._get_shared_vector_index())
    try:
        worker = OutboxWorker(projection, service._get_shared_embedder(),
                              fs=worker_fs, directory_summary_enabled=False)
        stats = worker.run_once(worker_outbox, [ctx.account_id])
        print({'request': {'operation': 'worker.run_once'}, 'response': stats}, flush=True)
        assert stats['succeeded'] == 1
        assert outbox.get_event_status(new.uri, event.event_id) == 'DONE'
    finally:
        projection.close()
        worker_fs.close()
        pool.close_all()
    assert search(api, ctx, 'newbodymarker') == {'hits': [new.uri], 'partial': False}
