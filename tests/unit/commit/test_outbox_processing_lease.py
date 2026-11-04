"""Concurrent AGFS workers preserve complete projections under renewable URI leases."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from threading import Barrier, Event

from commit.outbox_store import OutboxStore
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from tests.unit.commit.test_request_outbox_events import make_runtime, memory_node, request_context


def publish(fs, store, ctx, key="receiver"):
    node = memory_node(ctx, key)
    fs.write_node(node, ctx)
    return store.register_write(fs.read_node(node.uri, ctx), ctx)


def test_shared_uri_workers_preserve_completed_projection():
    client, fs, store = make_runtime()
    ctx = request_context()
    event = publish(fs, store, ctx)
    pending = [(event.uri, deepcopy(event))]
    entered, finish = Event(), Event()

    class CoordinatedEmbedder(MockEmbedder):
        calls = 0

        def embed_texts(self, texts):
            self.calls += 1
            entered.set()
            assert finish.wait(3)
            return super().embed_texts(texts)

    embedder = CoordinatedEmbedder(16)
    index = InMemoryVectorIndex(dimension=16)
    first = OutboxWorker(index, embedder, fs=fs)
    second = OutboxWorker(index, embedder, fs=fs)
    restarted = OutboxStore(client, AGFSContextFS(client))
    store.list_pending = lambda account: deepcopy(pending)
    restarted.list_pending = lambda account: deepcopy(pending)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(first.run_once, store, [ctx.account_id], "first")
        try:
            assert entered.wait(3)
            overlapping = second.run_once(restarted, [ctx.account_id], "second")
            assert overlapping["skipped"] == 1
            assert store.count_unfinished(ctx.account_id) == {"pending": 0, "processing": 1, "total": 1}
            hidden_locks = {path.split(".resource-mutation-lock", 1)[0] + ".resource-mutation-lock"
                            for path in client.files if ".outbox-processing-" in path}
            assert hidden_locks
            assert all(store._is_atomic_resource_staging_directory(path.rsplit("/", 1)[-1])
                       for path in hidden_locks)
        finally:
            finish.set()
        assert running.result()["succeeded"] == 1
    replay = second.run_once(restarted, [ctx.account_id], "second")
    assert replay["skipped"] == 1
    expected = build_index_records(fs.read_node(event.uri, ctx))
    assert set(index._records) == {record.id for record in expected}
    assert embedder.calls == 1
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_distinct_uri_leases_complete_in_parallel():
    client, fs, store = make_runtime()
    ctx = request_context()
    events = [publish(fs, store, ctx, key) for key in ("receiver", "telescope")]
    ready = Barrier(2)

    def process(event):
        independent = OutboxStore(client, AGFSContextFS(client))
        with independent.claim_processing_event(event, event.uri, event.event_id) as acquired:
            assert acquired
            ready.wait(timeout=3)
            independent.ensure_processing_owned(event, event.uri)
            independent.mark_done(event, event.uri)
        return event.event_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(process, events)) == {event.event_id for event in events}
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_projection_guard_renews_and_publishes_from_owned_directory(monkeypatch):
    from fs.agfs_adapter import agfs_context_fs

    monkeypatch.setattr(agfs_context_fs, "_RESOURCE_MUTATION_LOCK_RENEW_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(agfs_context_fs, "_RESOURCE_MUTATION_LOCK_STALE_AFTER", timedelta(seconds=0.1))
    client, fs, store = make_runtime()
    ctx = request_context()
    event = publish(fs, store, ctx)
    renewed = Event()
    writes, moves = [], []
    write, move = client.write, client.mv

    def record_write(path, data):
        result = write(path, data)
        writes.append((str(path), data))
        if sum("/.lease-" in value for value, _ in writes) >= 2:
            renewed.set()
        return result

    def record_move(source, destination):
        result = move(source, destination)
        moves.append((str(source), str(destination)))
        return result

    monkeypatch.setattr(client, "write", record_write)
    monkeypatch.setattr(client, "mv", record_move)
    with store.claim_processing_event(event, event.uri, "renewing") as acquired:
        assert acquired and renewed.wait(3)
        store.ensure_processing_owned(event, event.uri)
        store.increment_retry(event, event.uri)
        event_path = store._uri_to_agfs_path(event.uri) + ".outbox/" + event.event_id + ".json"
        assert json.loads(client.read(event_path))["retry_count"] == 1
        publications = [(source, target) for source, target in moves
                        if target.endswith((".processing", "/" + event.event_id + ".json"))]
        assert len(publications) == 2
        assert all(source.startswith(event._outbox_processing_lease.outbox_directory + "/.outbox-")
                   for source, _ in publications)
        store.mark_done(event, event.uri)
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_named_resource_is_scanned_and_projected():
    from tests.unit.fs.test_agfs_context_fs import _atomic_code_stack, _atomic_code_context, _atomic_code_node

    client, fs, store = _atomic_code_stack()
    client.api_base = "memory://named-resource"
    ctx = _atomic_code_context()
    node = replace(_atomic_code_node(), uri=f"ctx://{ctx.account_id}/resources/.outbox-processing-report.resource-mutation-lock-notes")
    event = store.build_write_event(node)
    fs.write_node_with_outbox(node, ctx, event)
    assert fs.read_node(node.uri, ctx).content == node.content
    pending = store.list_pending(ctx.account_id)
    assert [(uri, value.event_id) for uri, value in pending] == [(node.uri, event.event_id)]
    assert store.count_unfinished(ctx.account_id) == {"pending": 1, "processing": 0, "total": 1}
    index = InMemoryVectorIndex(dimension=16)
    stats = OutboxWorker(index, MockEmbedder(16), fs=fs).run_once(store, [ctx.account_id])
    assert stats["succeeded"] == 1
    expected = {record["id"]: record for record in event.payload["records"]}
    assert set(index._records) == set(expected)
    for record_id, actual in index._records.items():
        assert (actual.uri, actual.level, actual.text) == tuple(
            expected[record_id][field] for field in ("uri", "level", "text"))
    assert store.count_unfinished(ctx.account_id)["total"] == 0
