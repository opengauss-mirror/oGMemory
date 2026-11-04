"""Executable outbox ownership review cases."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from uuid import uuid4

import pytest

from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker, WorkerConfig
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from tests.integration.test_agfs_node_snapshot_reads import _runtime
from tests.integration.test_sql_outbox_claim_ownership import publish
from tests.unit.commit.test_request_outbox_events import memory_node

pytest_plugins = [
    "tests.integration.test_sql_outbox_claim_ownership",
    "tests.integration.test_agfs_node_snapshot_reads",
]


def emit(kind, data):
    print(kind + " " + json.dumps(data, default=str, ensure_ascii=False))


def test_sql_active_claim_completes_after_queue_timeout(sql_runtime):
    ctx, make = sql_runtime
    store, successor = make(), make()
    event, record = publish(store, ctx, "receiver", "Receiver repaired.")
    emit("REQUEST publish", vars(event))
    with store.claim_processing_event(event, event.uri, "owner") as owned:
        assert owned
        independent, _ = publish(store, ctx, "other", "Second receiver ready.")
        with ThreadPoolExecutor(1) as pool:
            rows = pool.submit(successor.claim_batch, "successor", 1, 0).result()
        emit("RESPONSE independent claim", [row[1].event_id for row in rows])
        assert [row[1].event_id for row in rows] == [independent.event_id]
        claimed = [row[1].event_id for row in rows if row[1].event_id == event.event_id]
        emit("RESPONSE concurrent claim", {"reclaimed_active_event": claimed})
        store.mark_done(event, event.uri)
    actual = store.get_event(ctx.account_id, event.event_id).status
    emit("RESPONSE completion", {"status": actual})
    assert claimed == [] and actual == "DONE"


def test_sql_direct_claim_round_trip(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    event, _ = publish(store, ctx, "release", "Receiver ready.")
    emit("REQUEST try_acquire", vars(event))
    assert store.try_acquire(event, event.uri, "direct")
    store.release(event, event.uri)
    emit("RESPONSE release", {"status": store.get_event(ctx.account_id, event.event_id).status})
    assert store.get_event(ctx.account_id, event.event_id).status == "PENDING"
    assert store.try_acquire(event, event.uri, "direct")
    store.mark_done(event, event.uri)
    emit(
        "RESPONSE direct completion",
        {"status": store.get_event(ctx.account_id, event.event_id).status},
    )
    completed = store.get_event(ctx.account_id, event.event_id)
    emit("RESPONSE stored receipt", {"status": completed.status, "payload": completed.payload})
    assert (completed.status, completed.payload) == ("DONE", {"filters": {"owner_space": ctx.user_space_name()}})


def test_delete_resource_scrubs_legacy_done_receipt(sql_runtime):
    from fs.sql_adapter.sql_context_fs import SQLContextFS

    ctx, make = sql_runtime
    store = make()
    fs = SQLContextFS(pool=store._shared_pool)
    node = memory_node(ctx, "legacy", "Receiver ready.")
    fs.write_node(node, ctx)
    event = store.build_write_event(node)
    event.status = "DONE"
    store._insert_event(event, ctx.account_id)
    emit("REQUEST delete", {"uri": node.uri, "legacy_payload": event.payload})
    assert fs.delete_nodes_with_outbox([node.uri], ctx, [store.build_delete_event(node.uri)]) == 1
    receipt = store.get_event(ctx.account_id, event.event_id)
    emit("RESPONSE historical receipt", {"status": receipt.status, "payload": receipt.payload})
    assert (receipt.status, receipt.payload) == ("DONE", {"task_owner_spaces": [ctx.user_space_name()]})


def test_agfs_backlog_uses_bounded_processing_scopes(native_agfs):
    client, fs, ctx = _runtime(native_agfs[0], "batch-" + uuid4().hex)
    store = OutboxStore(client, fs)
    for i in range(9):
        node = memory_node(ctx, f"receiver-{i}")
        fs.write_node(node, ctx)
        store.register_write(fs.read_node(node.uri, ctx), ctx)
    active = peak = 0
    claim = store.claim_processing_event

    @contextmanager
    def measured(*args):
        nonlocal active, peak
        with claim(*args) as owned:
            if owned:
                active += 1
                peak = max(active, peak)
            try:
                yield owned
            finally:
                if owned:
                    active -= 1

    store.claim_processing_event = measured
    config = WorkerConfig()
    config.processing_batch_size = 3
    emit("REQUEST drain", {"nodes": 9, "processing_batch_size": 3})
    result = OutboxWorker(InMemoryVectorIndex(8), MockEmbedder(8), config, fs=fs).run_once(
        store, [ctx.account_id]
    )
    emit("RESPONSE drain", {"stats": result, "peak_leases": peak, "remaining_leases": active})
    assert result["succeeded"] == 9 and peak <= 3 and active == 0


@pytest.fixture
def projection_index(tmp_path):
    backend = os.environ.get("OGMEM_TEST_RACE_BACKEND", "memory")
    name = "race_" + uuid4().hex
    if backend == "chroma":
        from providers.vector_index.chroma_index import ChromaVectorIndex

        yield ChromaVectorIndex(name, str(tmp_path / "index"), 8)
    elif backend == "opengauss":
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        index = OpenGaussVectorIndex(os.environ["OGMEM_TEST_OPENGAUSS_DSN"], 8, table_name=name)
        try:
            yield index
        finally:
            index.close()
    elif backend == "milvus":
        from providers.vector_index.milvus_index import MilvusVectorIndex

        index = MilvusVectorIndex(
            uri=os.environ["OGMEM_TEST_MILVUS_URI"], collection_name=name, dimension=8
        )
        try:
            yield index
        finally:
            index._client.drop_collection(collection_name=name)
            index.close()
    else:
        assert backend == "memory"
        yield InMemoryVectorIndex(8)


@pytest.mark.parametrize("operation", ["upsert", "rollback", "bm25_rollback"])
def test_late_projection_keeps_latest_published_text(native_agfs, monkeypatch, projection_index, operation):
    import time
    from datetime import datetime, timedelta
    from threading import Event

    from commit import outbox_store as outbox_module
    from fs.agfs_adapter import agfs_context_fs

    monkeypatch.setattr(agfs_context_fs, "_RESOURCE_MUTATION_LOCK_RENEW_INTERVAL_SECONDS", 60)
    monkeypatch.setattr(
        agfs_context_fs, "_RESOURCE_MUTATION_LOCK_STALE_AFTER", timedelta(seconds=0.1)
    )
    client, fs, ctx = _runtime(native_agfs[0], "late-" + uuid4().hex)
    store = OutboxStore(client, fs)
    other_client, other_fs, _ = _runtime(native_agfs[0], ctx.account_id)
    successor = OutboxStore(other_client, other_fs)
    entered, resume = Event(), Event()
    index = projection_index

    def pause_old(texts):
        if any("Monday" in text for text in texts) and not entered.is_set():
            entered.set()
            assert resume.wait(45)

    from retrieval.bm25_index import Bm25Index
    from tests.integration.test_code_resource_sql_lifecycle import _node as code_node

    bm25 = Bm25Index() if operation == "bm25_rollback" else None
    if operation != "upsert":
        target = bm25 if bm25 is not None else index
        write, delete = index.upsert, target.delete

        def move_after_upsert(records):
            write(records)
            if any("Monday" in record.text for record in records):
                emit("REQUEST move", {"from": node.uri, "to": node.uri + "-moved"})
                fs.move_node(node.uri, node.uri + "-moved", ctx)

        def delayed_delete(ids):
            emit("REQUEST rollback delete", ids)
            entered.set()
            assert resume.wait(45)
            delete(ids)

        monkeypatch.setattr(index, "upsert", move_after_upsert)
        monkeypatch.setattr(target, "delete", delayed_delete)
    elif os.environ.get("OGMEM_TEST_RACE_BACKEND") == "milvus":
        send = index._client.upsert

        def delayed_rpc(**request):
            pause_old(row["text"] for row in request["data"])
            return send(**request)

        monkeypatch.setattr(index._client, "upsert", delayed_rpc)
    else:
        write = index.upsert

        def delayed_upsert(records):
            pause_old(record.text for record in records)
            write(records)

        monkeypatch.setattr(index, "upsert", delayed_upsert)
    def release_node(text):
        if bm25 is None:
            return memory_node(ctx, "release", text)
        node = code_node(ctx.account_id, "release")
        node.content = node.abstract = node.overview = text
        return node

    node = release_node("Atlas releases Monday.")
    fs.write_node(node, ctx)
    old = store.register_write(fs.read_node(node.uri, ctx), ctx)
    emit("REQUEST first projection", vars(old))
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(
            OutboxWorker(index, MockEmbedder(8), fs=fs, bm25_index=bm25).run_once, store, [ctx.account_id], "old"
        )
        try:
            assert entered.wait(10)
            if operation != "upsert":
                fs.delete_node(node.uri + "-moved", ctx)
                node = release_node("Atlas releases Friday.")
            else:
                node = fs.read_node(node.uri, ctx)
                node.metadata["_expected_version"] = node.metadata["version"]
            node.content = node.abstract = node.overview = "Atlas releases Friday."
            fs.write_node(node, ctx)
            latest = successor.register_write(fs.read_node(node.uri, ctx), ctx)
            emit("REQUEST successor projection", vars(latest))

            class ExpiredClaimClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return datetime.now(tz) + timedelta(seconds=301)

            monkeypatch.setattr(outbox_module, "datetime", ExpiredClaimClock)
            time.sleep(0.2)
            newer = OutboxWorker(index, MockEmbedder(8), fs=other_fs, bm25_index=bm25).run_once(
                successor, [ctx.account_id], "new"
            )
            emit("RESPONSE successor projection", newer)
            successor_texts = [
                index.get_record(row["id"]).text for row in latest.payload["records"]
            ]
            emit("RESPONSE before late write", successor_texts)
            assert successor_texts and all("Friday" in text for text in successor_texts)
        finally:
            resume.set()
        try:
            late_result = running.result()
        except RuntimeError as exc:
            late_result = {"error": str(exc)}
        emit("RESPONSE late projection", late_result)
    saved = [index.get_record(record["id"]) for record in latest.payload["records"]]
    texts = [record.text if record else None for record in saved]
    remaining = successor.count_unfinished(ctx.account_id)
    emit("RESPONSE final state", {"texts": texts, "unfinished": remaining})
    if bm25 is not None:
        emit("RESPONSE lexical count", bm25.doc_count)
        assert bm25.doc_count == len(texts)
    assert texts == successor_texts
    assert remaining["total"] == 0
