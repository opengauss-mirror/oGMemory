"""Resume accepted writes across source and projection publication boundaries."""

import json
from dataclasses import asdict, replace

import pytest

from commit.outbox_store import OutboxStore
from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import PROJECTION_RECEIPTS_KEY, RECEIPTS_KEY, operation_digest
from core.models import CandidateMemory
from fs.sql_adapter.sql_context_fs import SQLContextFS
from index.outbox_worker import OutboxWorker
from index.index_record_builder import build_index_records
from providers.embedder.mock_embedder import MockEmbedder
from providers.llm import MockLLM
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.api import MemoryWriteAPI
from tests.integration.test_agfs_projection_receipts import storage
from tests.integration.test_sql_outbox_claim_ownership import sql_runtime


def accepted(category, operation_id, provenance=True):
    return CandidateMemory(
        category=category, owner_scope="agent" if category == "case" else "user",
        routing_key="atlas_release", abstract="Atlas releases on Friday.",
        overview="Alice confirmed the Atlas release on Friday.",
        content="Alice confirmed the Atlas release on Friday.", confidence=.9,
        provenance_ids=["prov:1:archive:release:message-one"] if provenance else [],
        write_operation_id=operation_id,
    )


def write(fs, store, candidate, ctx):
    api = MemoryWriteAPI(fs, MockLLM(), store, directory_summary_enabled=False)
    api._writer._require_index_event = True
    request = {"method": "MemoryWriteAPI.write_memory", "candidate": asdict(candidate),
               "context": asdict(ctx)}
    try:
        result = api.write_memory(CandidateMemory(**request["candidate"]), ctx)
    except Exception as exc:
        result = {"error": type(exc).__name__, "message": str(exc)}
    print(json.dumps({"request": request, "response": result}, ensure_ascii=False))
    return result


def project(fs, store, ctx, uri, event_id):
    index = InMemoryVectorIndex(16)
    stats = OutboxWorker(index, MockEmbedder(16), fs=fs).run_once(store, [ctx.account_id])
    status = store.get_event_status(uri, event_id)
    print(json.dumps({"worker": stats, "event_id": event_id, "status": status}))
    assert status == "DONE"
    assert stats["processed"] == 1
    for expected in build_index_records(fs.read_node(uri, ctx)):
        assert index.get_record(expected.id).text == expected.text


@pytest.mark.parametrize("category", ["entity", "event", "case"])
@pytest.mark.parametrize("prefix", ["receipt-v2:", "legacy:"])
def test_restart_completes_projection_for_committed_source(storage, monkeypatch, category, prefix):
    client, fs, ctx, store = storage
    candidate = accepted(category, prefix + "publish")

    def unavailable(event, request_ctx):
        raise ConnectionError("projection publication temporarily unavailable")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(store, "register_once", unavailable)
        write(fs, store, candidate, ctx)
    directory = MemoryWriteAPI(fs, MockLLM())._writer._identity_resolver.uris.get_directory_uri(category, ctx)
    uri = fs.list_children(directory, ctx)[0]
    source = fs.read_node(uri, ctx)
    event_id = source.metadata[PROJECTION_RECEIPTS_KEY][candidate.write_operation_id]["event_id"]
    restored = OutboxStore(client, fs)
    result = write(fs, restored, candidate, replace(ctx, trace_id="restart"))
    status = restored.get_event_status(uri, event_id)
    print(json.dumps({"source": uri, "event_id": event_id, "status": status,
                      "pending": [event.event_id for _, event in restored.list_pending(ctx.account_id)]}))
    assert result["merged_fields"]["status"] == "operation_committed"
    assert status == "PENDING"
    assert fs.read_node(uri, ctx) == source
    project(fs, restored, ctx, uri, event_id)
    write(fs, OutboxStore(client, fs), candidate, ctx)
    assert restored.get_event_status(uri, event_id) == "DONE"
    assert restored.count_unfinished(ctx.account_id)["total"] == 0


@pytest.fixture(params=["agfs", "sql"])
def resume_storage(request):
    if request.param == "agfs":
        client, fs, ctx, store = request.getfixturevalue("storage")
        return fs, ctx, store, lambda: OutboxStore(client, fs)
    ctx, make_store = request.getfixturevalue("sql_runtime")
    store = make_store()
    fs = SQLContextFS(pool=store._shared_pool)
    return fs, ctx, SQLOutboxStore(pool=store._shared_pool, fs=fs), lambda: SQLOutboxStore(pool=store._shared_pool, fs=fs)


@pytest.mark.parametrize("category,provenance", [
    ("session_archive", False), ("session_summary", False),
    ("event", False), ("case", False), ("event", True), ("case", True),
])
def test_restart_completes_first_write_at_bound_target(resume_storage, monkeypatch, category, provenance):
    fs, ctx, store, restart = resume_storage
    candidate = accepted(category, "receipt-v2:source", provenance)
    persist = fs.write_node
    targets = []

    def interrupted_source(node, request_ctx, *args):
        if node.category == category:
            targets.append(node.uri)
            raise ConnectionError("source publication temporarily unavailable")
        return persist(node, request_ctx)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(fs, "write_node", interrupted_source)
        if isinstance(fs, SQLContextFS):
            interrupted.setattr(fs, "write_node_with_outbox", interrupted_source)
        write(fs, store, candidate, ctx)
    assert len(targets) == 1
    result = write(fs, restart(), candidate, replace(ctx, trace_id="restart"))
    assert result["target_uri"] == targets[0]
    source = fs.read_node(targets[0], ctx)
    assert source.content == candidate.content
    assert source.metadata["version"] == 1
    assert source.metadata[RECEIPTS_KEY][candidate.write_operation_id] == operation_digest(candidate, ctx)
    event_id = source.metadata[PROJECTION_RECEIPTS_KEY][candidate.write_operation_id]["event_id"]
    project(fs, store, ctx, source.uri, event_id)
    replay = write(fs, restart(), candidate, ctx)
    assert replay["target_uri"] == source.uri
    assert replay["merged_fields"]["status"] == "operation_committed"


def test_plain_write_publishes_while_previous_embedding_runs(storage):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic

    from commit.outbox_scope import capture_outbox_events, current_outbox_events

    _, fs, ctx, store = storage
    first = accepted("entity", "")
    created = write(fs, store, first, ctx)
    uri = created["target_uri"]
    reached, release = Event(), Event()

    class SlowEmbedding(MockEmbedder):
        def embed_texts(self, texts):
            reached.set()
            assert release.wait(30)
            return super().embed_texts(texts)

    index = InMemoryVectorIndex(16)
    worker = OutboxWorker(index, SlowEmbedding(16), fs=fs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        processing = pool.submit(worker.run_once, store, [ctx.account_id])
        try:
            assert reached.wait(10)
            updated = replace(first, target_uri=uri, expected_version=1,
                              provenance_ids=["prov:1:archive:release:message-two"],
                              content="Atlas releases on Monday.",
                              overview="Atlas releases on Monday.", abstract="Atlas releases on Monday.")
            api = MemoryWriteAPI(fs, MockLLM(), store, directory_summary_enabled=False)
            start = monotonic()

            @capture_outbox_events
            def publish_update():
                result = api.write_memory(updated, ctx)
                return result, current_outbox_events(ctx.account_id)

            result, references = publish_update()
            pending = [event.event_id for _, event in store.list_pending(ctx.account_id)]
            print(json.dumps({"request": asdict(updated), "response": result,
                              "elapsed": monotonic()-start, "pending_during_embedding": pending,
                              "references": references}))
            assert len(references) == 1
            assert store.get_event_status(*references[0]) == "PENDING"
        finally:
            release.set()
        processing.result(timeout=10)
    OutboxWorker(index, MockEmbedder(16), fs=fs).run_once(store, [ctx.account_id])
    source = fs.read_node(uri, ctx)
    rows = [index.get_record(record.id).text for record in build_index_records(source)]
    print(json.dumps({"source": source.content, "indexed": rows}))
    assert source.content == "Atlas releases on Monday."
    for expected in build_index_records(source):
        assert index.get_record(expected.id).text == expected.text
