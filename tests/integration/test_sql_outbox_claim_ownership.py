"""Real SQL claims serialize normal projection, handoff, and request scopes."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from commit.sql_outbox_store import SQLOutboxStore
from core.models import IndexRecord, OutboxEvent, RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from index.outbox_worker import OutboxWorker, create_upsert_event
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex


@pytest.fixture
def sql_runtime():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("Set OGMEM_IDENTITY_TEST_DSN to the isolated PostgreSQL database")
    import psycopg2
    from psycopg2 import sql

    namespace = "outbox_review_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    dsn = psycopg2.extensions.make_dsn(dsn, options=f"-c search_path={namespace}")
    pools = []

    def store():
        pool = SharedConnectionPool(dsn, pool_size=2)
        pools.append(pool)
        return SQLOutboxStore(pool=pool)

    ctx = RequestContext("outbox-owner-" + uuid4().hex, "reader", "agent", "session", "trace")
    try:
        yield ctx, store
    finally:
        for pool in pools:
            pool.close_all()
        with control.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))
        control.close()


def publish(store, ctx, name, text):
    uri = f"ctx://{ctx.account_id}/users/reader/memories/entities/{name}"
    record = IndexRecord(IndexRecord.generate_id(uri, 0), uri, 0, text,
                         {"account_id": ctx.account_id, "owner_space": "user:reader"}, {})
    event = create_upsert_event(uri, [record])
    store._insert_event(event, ctx.account_id)
    return event, record


def stats():
    return dict(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0)


def test_request_projection_completes_successive_sql_references(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    existing, old_record = publish(store, ctx, "existing", "Earlier request")

    @capture_outbox_events
    def request():
        first, _ = publish(store, ctx, "requested", "Current request first version")
        event, record = publish(store, ctx, "requested", "Current request latest version")
        return first, event, record, current_outbox_events(ctx.account_id)

    first_version, requested, new_record, references = request()
    assert references == [(first_version.uri, first_version.event_id), (requested.uri, requested.event_id)]
    index = InMemoryVectorIndex(dimension=16)
    worker = OutboxWorker(index, MockEmbedder(16))
    pending = store.list_pending_events(references)
    first = worker.run_once(store, [ctx.account_id], pending_events=pending)
    assert first["succeeded"] == 2
    assert store.get_event_status(first_version.uri, first_version.event_id) == "DONE"
    assert store.get_event_status(requested.uri, requested.event_id) == "DONE"
    assert store.get_event_status(existing.uri, existing.event_id) == "PENDING"
    assert set(index._records) == {new_record.id}
    assert index._records[new_record.id].text == "Current request latest version"
    worker.run_once(store, [ctx.account_id], pending_events=[(existing.uri, existing)])
    assert set(index._records) == {old_record.id, new_record.id}
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_request_projection_reports_owned_predecessor_then_completes(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    earlier, _ = publish(store, ctx, "receiver", "Receiver before repair")
    claimed = store.claim_batch("earlier-owner", event_ids=[earlier.event_id])
    requested, record = publish(store, ctx, "receiver", "Receiver repaired")
    references = [(requested.uri, requested.event_id)]
    index = InMemoryVectorIndex(dimension=16)
    worker = OutboxWorker(index, MockEmbedder(16))
    waiting = worker.run_once(store, [ctx.account_id],
                              pending_events=store.list_pending_events(references))
    assert waiting["pending"] == 2
    assert waiting["deferred"] is True
    worker._process_claimed_events(claimed, store, stats(), "earlier-owner")
    completed = worker.run_once(store, [ctx.account_id],
                                pending_events=store.list_pending_events(references))
    assert completed["succeeded"] == 1
    assert store.get_event(ctx.account_id, requested.event_id).status == "DONE"
    assert index._records[record.id].text == "Receiver repaired"


@pytest.mark.parametrize("next_worker", ["batch-owner", "next-owner"])
def test_slow_batch_continues_after_delete_and_rebuild(sql_runtime, next_worker):
    ctx, make = sql_runtime
    original, successor = make(), make()
    first, first_record = publish(original, ctx, "long-document", "Long document")
    old_write, old_record = publish(original, ctx, "receiver", "Receiver before rebuild")
    index = InMemoryVectorIndex(dimension=16)
    normal_worker = OutboxWorker(index, MockEmbedder(16))
    normal_worker.run_once(original, [ctx.account_id], pending_events=[(old_write.uri, old_write)])
    deletion = original.register_delete(old_write.uri, ctx)
    lease_seconds = 0.03
    claimed = original.claim_batch("batch-owner", timeout_seconds=lease_seconds,
                                   event_ids=[first.event_id, deletion.event_id])
    assert [event.event_id for _, event in claimed] == [first.event_id, deletion.event_id]
    embedding_started, continue_embedding = threading.Event(), threading.Event()

    class DocumentEmbedder(MockEmbedder):
        def embed_texts(self, texts):
            embedding_started.set()
            assert continue_embedding.wait(10)
            return super().embed_texts(texts)

    slow_worker = OutboxWorker(index, DocumentEmbedder(16))
    original_stats = stats()
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(slow_worker._process_claimed_events, claimed, original,
                                  original_stats, "batch-owner")
        try:
            assert embedding_started.wait(5)
            time.sleep(lease_seconds * 2)
            continued = successor.claim_batch(next_worker, timeout_seconds=lease_seconds,
                                              event_ids=[deletion.event_id])
            assert len(continued) == 1
            delete_stats = stats()
            normal_worker._process_claimed_events(continued, successor, delete_stats, next_worker)
            assert delete_stats["succeeded"] == 1
            rebuilt, rebuilt_record = publish(successor, ctx, "receiver", "Receiver rebuilt and tested")
            completed = normal_worker.run_once(successor, [ctx.account_id], worker_id=next_worker,
                                               pending_events=[(rebuilt.uri, rebuilt)])
            assert completed["succeeded"] == 1
        finally:
            continue_embedding.set()
        running.result(timeout=10)
    assert original_stats["succeeded"] == 1
    assert original_stats["skipped"] == 1
    assert original_stats["failed"] == 0
    assert set(index._records) == {first_record.id, rebuilt_record.id}
    assert index._records[old_record.id].text == "Receiver rebuilt and tested"
    assert original.get_event_status(deletion.uri, deletion.event_id) == "DONE"
    assert original.get_event_status(rebuilt.uri, rebuilt.event_id) == "DONE"
    assert original.count_unfinished(ctx.account_id)["total"] == 0


def test_agent_claim_keeps_checkpoint_and_completion_with_same_ticket(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    uri = f"ctx://{ctx.account_id}/agents/agent/memories/agent_extraction/job"
    event = OutboxEvent(uuid4().hex, "EXTRACT_AGENT_MEMORY", uri,
                        {"context": vars(ctx), "cursor": 0}, "PENDING")
    store._insert_event(event, ctx.account_id)
    claimed = store.claim_batch("agent-owner", event_ids=[event.event_id])[0][1]
    with store.claim_agent_event(claimed, "agent-owner") as acquired:
        assert acquired
        claimed.payload["cursor"] = 1
        store.save_payload(claimed)
        assert store.get_event(ctx.account_id, event.event_id).payload["cursor"] == 1
        store.mark_done(claimed, uri)
    assert store.get_event_status(uri, event.event_id) == "DONE"


def test_individual_claim_release_and_completion_share_uri_lock(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    event, _ = publish(store, ctx, "individual", "Individual projection")
    with store.claim_processing_event(event, event.uri, "first-owner") as acquired:
        assert acquired
        with store.processing_lock(event.uri):
            store.ensure_processing_owned(event, event.uri)
        store.release(event, event.uri)
    with store.claim_processing_event(event, event.uri, "next-owner") as acquired:
        assert acquired
        store.mark_done(event, event.uri)
    assert store.get_event_status(event.uri, event.event_id) == "DONE"
