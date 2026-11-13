"""Durable projection receipts on an isolated AGFS server."""
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from commit.outbox_store import OutboxStore
from core.models import ContextNode, OutboxEvent, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS


@pytest.fixture
def storage():
    address = os.environ.get('OGMEM_AGFS_TEST_URL')
    if not address:
        pytest.skip('Set OGMEM_AGFS_TEST_URL to the isolated AGFS server')
    client = AGFSClient(address)
    fs = AGFSContextFS(client, mount_prefix='/local')
    ctx = RequestContext('receipt-' + uuid4().hex, 'alice', 'main', 'normal-session', 'receipt-test')
    return client, fs, ctx, OutboxStore(client, fs)


def test_new_account_recovery_initializes_empty(storage):
    _, fs, ctx, _ = storage
    rows, cursor = fs.scan_session_recovery(ctx.account_id)
    assert rows == []
    assert cursor is None


def test_completed_projection_receipt_survives_new_store(storage):
    client, fs, ctx, store = storage
    uri = f'ctx://{ctx.account_id}/users/alice/memories/entities/atlas'
    node = ContextNode(uri, 'MEMORY', 'entity', 'L2', 'user:alice',
                       abstract='Atlas release', overview='Atlas uses uv.', content='Atlas releases at 21:00.')
    fs.write_node(node, ctx)
    event = OutboxEvent(uuid4().hex, 'UPSERT_CONTEXT', uri, {'version': 1}, 'PENDING')
    store._write_event(event, uri)
    assert store.get_event_status(uri, event.event_id) == 'PENDING'
    with store.claim_processing_event(event, uri, 'positive-worker') as claimed:
        assert claimed is True
        store.mark_done(event, uri)
    restored = OutboxStore(client, fs)
    assert restored.get_event_status(uri, event.event_id) == 'DONE'
    assert restored.count_unfinished(ctx.account_id) == {'pending': 0, 'processing': 0, 'total': 0}
    assert fs.read_node(uri, ctx).content == node.content


def test_successive_projection_receipts_preserve_both_versions(storage):
    client, fs, ctx, store = storage
    uri = f'ctx://{ctx.account_id}/users/alice/memories/entities/atlas'
    fs.write_node(ContextNode(uri, 'MEMORY', 'entity', 'L2', 'user:alice', 'Atlas', 'Atlas project',
                              'Atlas uses uv.'), ctx)
    events = [OutboxEvent(uuid4().hex, 'UPSERT_CONTEXT', uri, {'version': version}, 'PENDING')
              for version in (1, 2)]
    for event in events:
        store._write_event(event, uri)
        with store.claim_processing_event(event, uri, 'positive-worker') as claimed:
            assert claimed is True
            store.mark_done(event, uri)
    restored = OutboxStore(client, fs)
    assert [restored.get_event_status(uri, event.event_id) for event in events] == ['DONE', 'DONE']
    assert restored.list_pending(ctx.account_id) == []


def test_concurrent_registration_and_completed_replay_share_one_receipt(storage):
    client, fs, ctx, store = storage
    uri = f'ctx://{ctx.account_id}/users/alice/memories/entities/atlas'
    fs.write_node(ContextNode(uri, 'MEMORY', 'entity', 'L2', 'user:alice', 'Atlas', 'Atlas project',
                              'Atlas uses uv.'), ctx)
    event = OutboxEvent(uuid4().hex, 'UPSERT_CONTEXT', uri, {'version': 1}, 'PENDING')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: store.register_once(event, ctx), range(2)))
    pending = store.list_pending(ctx.account_id)
    assert [item.event_id for _, item in pending] == [event.event_id]
    with store.claim_processing_event(event, uri, 'positive-worker') as claimed:
        assert claimed is True
        store.mark_done(event, uri)
    restored = OutboxStore(client, fs)
    restored.register_once(event, ctx)
    assert restored.get_event_status(uri, event.event_id) == 'DONE'
    assert restored.list_pending(ctx.account_id) == []
