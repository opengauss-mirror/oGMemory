"""Directory derivations retain revocation through generation, indexing and reads."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, local
from unittest.mock import Mock

import pytest

from commit.deterministic import deterministic_batch
from commit.outbox_store import OutboxStore
from core.errors import NodeNotFoundError
from core.models import ContextNode, IndexRecord, OutboxEvent
from index.directory_event_handler import DirectoryEventHandler
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker, create_upsert_event
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from service.evidence_directory import schedule_directory_rebuilds
from service.evidence_lifecycle import EvidenceLifecycle
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import frozen_candidates

ROOT = "ctx://account/users/alice/memories/"
DIR = ROOT + "facts/"


def write_fact(setup, ctx=CTX):
    _, api, store, _ = setup
    candidate = next(c for c in frozen_candidates(store, [message("I visited Paris.")], ctx) if c.category == "fact")
    with deterministic_batch():
        return api._writer.write_candidate(candidate, ctx).target_uri


def components(fs):
    llm, embedder = Mock(), Mock()
    llm.complete_json.return_value = {"abstract": "Paris visit", "overview": "Visited Paris."}
    embedder.embed_texts.side_effect = lambda texts: [[1., 0.] for _ in texts]
    index = InMemoryVectorIndex(2)
    return llm, embedder, index, DirectoryEventHandler(fs, llm, embedder, index)


def event(uri=DIR):
    return OutboxEvent("summary", "UPSERT_DIRECTORY", uri, {}, "PENDING")


@pytest.fixture
def built_directory(setup):
    write_fact(setup)
    llm, embedder, index, handler = components(setup[0])
    assert handler.process_directory_event(event(), CTX).success
    return llm, embedder, index, handler


def test_multilevel_summaries_inherit_sources_and_hide_immediately_on_revocation(setup, built_directory):
    fs, _, store, _ = setup
    _, _, index, handler = built_directory
    result = handler.process_directory_event(event(ROOT), CTX)
    assert result.success and ROOT in result.written_dir_uris
    reader = ContextReader(fs)
    for uri in (DIR, ROOT):
        assert reader.read(uri, CTX).metadata.get("evidence_source_refs")
    store.revoke_session(CTX)
    for uri in (DIR, ROOT):
        with pytest.raises(NodeNotFoundError):
            reader.read(uri, CTX)
        with pytest.raises(NodeNotFoundError):
            reader.read_authorized(uri, CTX)
    assert index._records  # Visibility cannot depend on asynchronous index cleanup.


@pytest.mark.parametrize("legacy", [False, True])
def test_delete_scrubs_all_ancestors_and_queues_index_cleanup(setup, built_directory, legacy):
    fs, _, _, _ = setup
    llm, embedder, index, handler = built_directory
    handler.process_directory_event(event(ROOT), CTX)
    if legacy:
        for uri in (DIR, ROOT):
            node = fs.read_node(uri, CTX)
            fs.write_node(replace(node, metadata={"node_type": "directory"}), CTX)
    outbox = Mock()
    lifecycle = EvidenceLifecycle(fs, outbox)
    lifecycle.delete_session(CTX)
    cleanup = [call.args[0] for call in outbox.register_once.call_args_list
               if call.args[0].payload.get("evidence_directory_cleanup")]
    assert {item.uri for item in cleanup} == {DIR, ROOT}
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm)
    for item in cleanup:
        assert worker.process_event(item, CTX).success
    assert not any(record.uri in (DIR, ROOT) for record in index._records.values())
    for uri in (DIR, ROOT):
        node = fs.read_node(uri, CTX)
        assert not node.abstract and not node.overview and not node.content
        with pytest.raises(NodeNotFoundError):
            ContextReader(fs).read(uri, CTX)
    assert lifecycle.delete_session(CTX)["memories_cleaned"] == 0


@pytest.mark.parametrize("stage", ["llm", "embedding"])
def test_inflight_directory_work_cannot_republish_revoked_text(setup, stage):
    fs, _, _, _ = setup
    write_fact(setup)
    llm, embedder, index, handler = components(fs)
    entered, release = Event(), Event()

    def paused(*args):
        entered.set()
        assert release.wait(10)
        return ({"abstract": "Paris visit", "overview": "Visited Paris."} if stage == "llm"
                else [[1., 0.] for _ in args[0]])

    (llm.complete_json if stage == "llm" else embedder.embed_texts).side_effect = paused
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(handler.process_directory_event, event(), CTX)
        try:
            assert entered.wait(10)
            EvidenceLifecycle(fs).delete_session(CTX)
        finally:
            release.set()
        pending.result(timeout=10)
    assert not index._records
    with pytest.raises(NodeNotFoundError):
        ContextReader(fs).read(DIR, CTX)


def test_late_cleanup_does_not_remove_a_rebuilt_ordinary_directory(setup, built_directory):
    fs, _, _, _ = setup
    llm, embedder, index, handler = built_directory
    outbox = Mock()
    EvidenceLifecycle(fs, outbox).delete_session(CTX)
    ordinary = ContextNode(DIR + "ordinary", "MEMORY", "fact", 2, CTX.user_space_name(),
                           "Rome trip", "Rome", "Visited Rome.", {})
    fs.write_node(ordinary, CTX)
    llm.complete_json.return_value = {"abstract": "Rome trip", "overview": "Visited Rome."}
    assert handler.process_directory_event(event(), CTX).success
    assert ContextReader(fs).read(DIR, CTX).abstract == "Rome trip"
    cleanup = [call.args[0] for call in outbox.register_once.call_args_list
               if call.args[0].payload.get("evidence_directory_cleanup") and call.args[0].uri == DIR]
    assert cleanup
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm)
    assert worker.process_event(cleanup[0], CTX).success
    assert any(record.uri == DIR for record in index._records.values())


def test_cross_session_directory_keeps_request_identity_and_revocation_of_either_source(setup, monkeypatch):
    from fs.sql_adapter.session_mutation import SQLSessionGuard

    fs, _, store, _ = setup
    write_fact(setup)
    other = replace(CTX, session_id="other")
    write_fact(setup, other)
    request = replace(CTX, session_id="worker")
    state = local()
    acquire = fs.memory_identity_lock

    def owned_lock(self, uri, ctx):
        if not hasattr(state, "guard"):
            state.guard = SQLSessionGuard(Mock(closed=False), "dbname=test", request, 1)
        state.guard.check_context(ctx)
        return acquire(uri, ctx)

    monkeypatch.setattr(type(fs), "memory_identity_lock", owned_lock)
    _, _, _, handler = components(fs)
    assert handler.process_directory_event(event(), request).success
    refs = fs.read_node(DIR, request).metadata.get("evidence_source_refs", [])
    assert {ref["session_id"] for ref in refs} == {CTX.session_id, "other"}
    monkeypatch.undo()
    store.revoke_session(other)
    with pytest.raises(NodeNotFoundError):
        ContextReader(fs).read(DIR, request)


def test_parent_cannot_launder_a_revoked_legacy_directory_summary(setup, built_directory):
    fs, _, store, _ = setup
    _, _, _, handler = built_directory
    node = fs.read_node(DIR, CTX)
    fs.write_node(replace(node, metadata={"node_type": "directory"}), CTX)
    # Upgrading a cached summary must recover lineage without another child LLM call.
    assert handler.process_directory_event(event(ROOT), CTX).success
    assert fs.read_node(ROOT, CTX).metadata["evidence_source_refs"]
    store.revoke_session(CTX)
    assert handler.process_directory_event(event(ROOT), CTX).success
    assert "Paris" not in ContextReader(fs).read(ROOT, CTX).abstract


def test_dag_propagates_lineage_from_child_to_parent(setup, monkeypatch):
    fs, _, store, _ = setup
    write_fact(setup)
    children = fs.list_children
    monkeypatch.setattr(fs, "list_children", lambda uri, ctx: [DIR] if uri == ROOT else children(uri, ctx))
    _, _, _, handler = components(fs)
    result = handler.process_directory_event(event(ROOT), CTX)
    assert set(result.written_dir_uris) == {DIR, ROOT}
    assert fs.read_node(ROOT, CTX).metadata["evidence_source_refs"]
    store.revoke_session(CTX)
    with pytest.raises(NodeNotFoundError):
        ContextReader(fs).read(ROOT, CTX)


def test_directory_cleanup_resumes_after_outbox_failure(setup, built_directory):
    fs, _, store, _ = setup
    outbox = Mock()
    outbox.register_once.side_effect = ConnectionError("outbox unavailable")
    lifecycle = EvidenceLifecycle(fs, outbox)
    with pytest.raises(ConnectionError, match="outbox"):
        lifecycle.delete_session(CTX)
    first = outbox.register_once.call_args.args[0]
    assert store.manifest(CTX)[0]["status"] == "revoked"
    with pytest.raises(NodeNotFoundError):
        ContextReader(fs).read(DIR, CTX)
    outbox.register_once.side_effect = None
    lifecycle.delete_session(CTX)
    assert outbox.register_once.call_args_list[1].args[0].event_id == first.event_id
    assert DIR in store.manifest(CTX)[0]["directory_cleanup_completed"]


@pytest.mark.parametrize("revoked", [False, True])
@pytest.mark.parametrize("lexical_only", [False, True])
def test_generic_directory_index_replay_also_checks_sources(setup, built_directory, revoked, lexical_only):
    from dataclasses import asdict

    fs, _, store, _ = setup
    llm, embedder, _, _ = built_directory
    node = fs.read_node(DIR, CTX)
    old = OutboxEvent("old", "UPSERT_CONTEXT", DIR,
                     {"records": [asdict(record) for record in build_index_records(node)]}, "PENDING")
    if revoked:
        store.revoke_session(CTX)
    index = Mock()
    embedder.reset_mock()
    result = OutboxWorker(index, None if lexical_only else embedder, fs=fs, llm=llm).process_event(old, CTX)
    assert result.success
    assert index.upsert.called is not revoked
    if lexical_only:
        embedder.embed_texts.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_search_hides_revoked_directory_even_with_stale_legacy_index(setup, built_directory, legacy):
    from core.models import RetrievalConfig, RetrievedBlock, SeedResult, TypedQuery
    from retrieval.pipeline import RetrievalPipeline

    fs, _, store, _ = setup
    planner, seed = Mock(), Mock()
    planner.plan.return_value = [TypedQuery("Paris", "MEMORY", ["fact"])]
    seed.search.return_value = SeedResult([], [])
    pipeline = RetrievalPipeline(planner, seed, None, Mock(), RetrievalConfig(expand_hierarchy=False),
                                 context_reader=ContextReader(fs))
    pipeline._run_assembly = lambda *a, **kw: [RetrievedBlock(
        uri=DIR, level_hit="L0", abstract="stale Paris visit",
        metadata={"node_type": "directory"} if legacy else {"evidence_derived": True})]
    assert pipeline.run("Paris", CTX).hits[0].abstract == "Paris visit"
    store.revoke_session(CTX)
    assert pipeline.run("Paris", CTX).hits == []


def test_cleanup_rebuild_retains_ordinary_and_other_session_memories(setup):
    fs, _, _, _ = setup
    write_fact(setup)
    write_fact(setup, replace(CTX, session_id="other"))
    fs.write_node(ContextNode(DIR + "ordinary", "MEMORY", "fact", 2, CTX.user_space_name(),
                              "Rome trip", "Rome", "Visited Rome.", {}), CTX)
    llm, embedder, index, handler = components(fs)

    def summary(prompt, schema):
        text = " ".join(word for word in ("Paris", "Rome") if word in prompt)
        return {"abstract": text, "overview": text}

    llm.complete_json.side_effect = summary
    handler.process_directory_event(event(), CTX)
    handler.process_directory_event(event(ROOT), CTX)
    outbox = Mock()
    EvidenceLifecycle(fs, outbox).delete_session(CTX)
    events = [call.args[0] for call in outbox.register_once.call_args_list]
    rebuilds = [item for item in events if item.event_type == "UPSERT_DIRECTORY"]
    assert len(rebuilds) == 1 and rebuilds[0].uri == ROOT
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm, directory_summary_enabled=True)
    assert worker.process_event(rebuilds[0], CTX).success
    for uri in (DIR, ROOT):
        current = ContextReader(fs).read(uri, CTX)
        assert "Rome" in current.abstract and "Paris" in current.abstract
        assert {ref["session_id"] for ref in current.metadata["evidence_source_refs"]} == {"other"}
    # Archive events can arrive after the rebuilt projection.
    for item in events:
        if item.payload.get("evidence_directory_cleanup"):
            assert worker.process_event(item, CTX).success
    assert any(record.uri == ROOT for record in index._records.values())


# Real AGFS Outbox leases must never wait while a source lock is held.


def assert_directory_registration_keeps_lock_order(store, reader, outbox, directory, ctx, operation, monkeypatch):
    if operation == "rebuild":
        EvidenceLifecycle(store.fs).delete_session(ctx)
    held, registering = Event(), Event()
    register = outbox.register_once

    def signal_registration(pending, request):
        if pending.uri == directory:
            registering.set()
        return register(pending, request)

    monkeypatch.setattr(outbox, "register_once", signal_registration)

    def worker():
        with outbox.processing_lock(directory):
            held.set()
            assert registering.wait(15)
            with reader.session_lock(ctx):
                assert store.manifest(ctx)[0]["status"] == "revoked"

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(worker)
        assert held.wait(10)
        if operation == "invalidate":
            EvidenceLifecycle(store.fs, outbox).delete_session(ctx)
        else:
            schedule_directory_rebuilds(store, ctx, outbox)
        pending.result(timeout=15)
    assert list(outbox.list_pending(ctx.account_id))


@pytest.mark.parametrize("operation", ["invalidate", "rebuild"])
def test_directory_event_registration_releases_source_locks_before_waiting_for_worker(
    setup, built_directory, monkeypatch, operation,
):
    fs, _, store, _ = setup
    fs._client.api_base = "http://test-agfs"
    outbox = OutboxStore(fs._client, fs, mount_prefix="/local")
    assert_directory_registration_keeps_lock_order(store, store, outbox, DIR, CTX, operation, monkeypatch)


def test_rebuild_registration_retries_and_preserves_concurrent_work(setup, built_directory):
    fs, _, store, _ = setup
    EvidenceLifecycle(fs).delete_session(CTX)
    before = store.manifest(CTX)[0]["directory_rebuild_pending"]
    assert before
    with pytest.raises(OSError):
        schedule_directory_rebuilds(store, CTX, Mock(register_once=Mock(side_effect=OSError("unavailable"))))
    assert store.manifest(CTX)[0]["directory_rebuild_pending"] == before
    new_directory = DIR + "new/"

    def register(pending, ctx):
        with store.session_lock(ctx):
            manifest, version = store.manifest(ctx)
            manifest["directory_rebuild_pending"].append(new_directory)
            store._write(store.session_uri(ctx), manifest, version, ctx)

    schedule_directory_rebuilds(store, CTX, Mock(register_once=register))
    assert store.manifest(CTX)[0]["directory_rebuild_pending"] == [new_directory]
    schedule_directory_rebuilds(store, CTX, Mock())
    assert not store.manifest(CTX)[0]["directory_rebuild_pending"]


# AGFS batch replay must use the same source authority as single projection.


@pytest.mark.parametrize("revoke_at", ["before", "embedding", "legacy_payload", "invalidated"])
def test_agfs_run_once_rejects_revoked_directory_and_preserves_ordinary_batch(setup, monkeypatch, revoke_at):
    fs, _, store, _ = setup
    fs._client.api_base = "http://test-agfs"
    write_fact(setup)
    llm, embedder, index, handler = components(fs)
    handler.process_directory_event(event(), CTX)
    index._records.clear()
    records = build_index_records(fs.read_node(DIR, CTX))
    old = create_upsert_event(DIR, records)
    if revoke_at == "legacy_payload":
        for record in old.payload["records"]:
            record["metadata"].pop("evidence_derived", None)
    ordinary = create_upsert_event("ctx://account/resources/ordinary", [IndexRecord(
        "ordinary", "ctx://account/resources/ordinary", 0, "Ordinary resource",
        {"account_id": CTX.account_id, "owner_space": "resource:account"})])
    outbox = OutboxStore(fs._client, fs, mount_prefix="/local")
    for pending in (ordinary, old):
        outbox.register_once(pending, CTX)
    if revoke_at == "invalidated":
        EvidenceLifecycle(fs).delete_session(CTX)

        def reject_empty(texts):
            assert all(text.strip() for text in texts), "provider rejects empty embedding input"
            return [[1., 0.] for _ in texts]

        embedder.embed_texts.side_effect = reject_empty
    elif revoke_at != "embedding":
        store.revoke_session(CTX)
    else:
        def revoke(texts):
            if any("Paris" in text for text in texts):
                store.revoke_session(CTX)
            return [[1., 0.] for _ in texts]
        embedder.embed_texts.side_effect = revoke
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm)
    batches = Mock(wraps=worker._process_upsert_batch)
    monkeypatch.setattr(worker, "_process_upsert_batch", batches)
    stats = worker.run_once(outbox, [CTX.account_id], pending_events=[(item.uri, item) for item in (ordinary, old)])
    assert stats["succeeded"] == 2 and not stats["failed"]
    assert {record.uri for record in index._records.values()} == {ordinary.uri}
    assert batches.call_count == 1
    assert [item[0].uri for item in batches.call_args.args[0]] == [ordinary.uri]


def test_batch_final_publication_checks_source_even_if_called_directly(setup, built_directory):
    fs, _, store, _ = setup
    llm, embedder, index, _ = built_directory
    records = build_index_records(fs.read_node(DIR, CTX))
    old = create_upsert_event(DIR, records)
    index._records.clear()

    def revoke(texts):
        store.revoke_session(CTX)
        return [[1., 0.] for _ in texts]

    embedder.embed_texts.side_effect = revoke
    stats = {"processed": 0, "succeeded": 0, "failed": 0}
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm)
    worker._process_upsert_batch([(old, DIR, records)], Mock(), stats)
    assert stats["succeeded"] == 1 and not stats["failed"]
    assert not index._records


def test_agfs_refreshes_leaf_source_marker_before_batch_routing(setup, monkeypatch):
    fs, _, store, _ = setup
    fs._client.api_base = "http://test-agfs"
    uri = write_fact(setup)
    records = build_index_records(fs.read_node(uri, CTX))
    old = create_upsert_event(uri, records)
    for record in old.payload["records"]:
        record["metadata"].pop("extraction_method", None)
    outbox = OutboxStore(fs._client, fs, mount_prefix="/local")
    outbox.register_once(old, CTX)
    store.revoke_session(CTX)
    llm, embedder, index, _ = components(fs)
    worker = OutboxWorker(index, embedder, fs=fs, llm=llm)
    batches = Mock(wraps=worker._process_upsert_batch)
    monkeypatch.setattr(worker, "_process_upsert_batch", batches)
    stats = worker.run_once(outbox, [CTX.account_id], pending_events=[(uri, old)])
    assert stats["succeeded"] == 1 and not stats["failed"]
    assert not index._records
    batches.assert_not_called()
