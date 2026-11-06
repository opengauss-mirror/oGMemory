"""Scoped AGFS publication retains the worker and after_turn completion paths."""

import json
from dataclasses import replace
from unittest.mock import Mock

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from session.archive_store import SessionArchiveStore
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_request_outbox_events import make_runtime, memory_node, request_context
from tests.unit.server.test_extraction_batch_recovery import NoModel


def test_targeted_worker_projects_current_version_under_processing_lease():
    client, fs, store = make_runtime()
    ctx = request_context()
    notifications = []
    active_locks = []

    class CheckedIndex(InMemoryVectorIndex):
        def upsert(self, records):
            locks = [json.loads(value) for path, value in client.files.items()
                     if path.endswith(".processing")]
            assert all(lock["worker_id"] == "request-worker" for lock in locks)
            assert len(locks) >= 1
            active_locks.append(len(locks))
            super().upsert(records)

    index = CheckedIndex(dimension=16)
    worker = OutboxWorker(index, MockEmbedder(16), fs=fs, directory_summary_enabled=False,
                          memory_projection_handler=lambda event, request: notifications.append(
                              (event.payload["_memory_projection_version"], request.account_id)))

    @capture_outbox_events
    def write_versions():
        node = memory_node(ctx)
        fs.write_node(node, ctx)
        store.register_write(fs.read_node(node.uri, ctx), ctx)
        current = fs.read_node(node.uri, ctx)
        revised = replace(current, content="The receiver was repaired and tested.",
                          overview="The receiver was repaired and tested.",
                          abstract="The receiver was repaired and tested.",
                          metadata={**current.metadata, "expected_version": 1})
        fs.write_node(revised, ctx)
        store.register_write(fs.read_node(node.uri, ctx), ctx)
        return node.uri, current_outbox_events(ctx.account_id)

    uri, references = write_versions()
    client.list_paths.clear()
    stats = worker.run_once(store, [ctx.account_id], worker_id="request-worker",
                            pending_events=store.list_pending_events(references))
    assert stats == {"processed": 2, "succeeded": 2, "failed": 0, "moved_to_dlq": 0, "skipped": 0}
    assert active_locks == [2, 1]
    assert notifications == [(2, ctx.account_id), (2, ctx.account_id)]
    assert client.list_paths == []
    expected = build_index_records(fs.read_node(uri, ctx))
    assert set(index._records) == {record.id for record in expected}
    for record in expected:
        actual = index._records[record.id]
        assert (actual.uri, actual.level, actual.text) == (record.uri, record.level, record.text)
        assert len(actual.metadata["_embedding"]) == 16
    assert store.count_unfinished(ctx.account_id) == {"pending": 0, "processing": 0, "total": 0}


def test_unscoped_worker_completes_preexisting_persisted_events():
    _, fs, store = make_runtime()
    ctx = request_context()
    node = memory_node(ctx)
    fs.write_node(node, ctx)
    store.register_write(fs.read_node(node.uri, ctx), ctx)
    index = InMemoryVectorIndex(dimension=16)
    stats = OutboxWorker(index, MockEmbedder(16), fs=fs).run_once(store, [ctx.account_id])
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0
    assert len(index._records) == len(build_index_records(fs.read_node(node.uri, ctx)))
    assert store.count_unfinished(ctx.account_id)["total"] == 0


def test_agfs_after_turn_drains_its_persisted_memory_events(monkeypatch):
    client, fs, store = make_runtime()
    ctx = request_context()
    index = InMemoryVectorIndex(dimension=16)
    writer = MemoryWriteAPI(fs, NoModel(), store, directory_summary_enabled=False)
    writer._pipeline.extract = Mock(return_value=[memory("receiver", content="The receiver was repaired.")])
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._cfg.after_turn_threshold = 1
    service._cfg.directory_summary_enabled = False
    service._cfg.scene_enabled = False
    service._cfg.chunking_enabled = False
    service._write_api = writer
    monkeypatch.setattr(service, "get_llm", lambda: NoModel())
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda context_fs: store)
    monkeypatch.setattr(service, "_get_archive_store", lambda: SessionArchiveStore(fs))
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: index)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: MockEmbedder(16))
    monkeypatch.setattr("extraction.summary_generator.SessionSummaryGenerator.generate",
                        lambda *args, **kwargs: {"facts_and_details": "The receiver was repaired."})
    manager = service.get_session_manager()
    monkeypatch.setattr(manager, "_compress", lambda *args, **kwargs: (
        "The receiver was repaired.", "Receiver repair."))
    references_seen = []
    direct_read = store.list_pending_events
    full_scan = Mock(wraps=store.list_pending)
    monkeypatch.setattr(store, "list_pending", full_scan)

    def observe_references(references):
        references_seen.append(list(references))
        return direct_read(references)

    monkeypatch.setattr(store, "list_pending_events", observe_references)
    request = {"sessionId": ctx.session_id, "clientRequestId": "request-one", "wait": True,
               "messages": [{"role": "user", "content": "The receiver was repaired."}], "_ctx": ctx}
    response = service.after_turn(request)
    assert response["status"] == "completed", response
    assert response["commit"]["archived"] is True
    assert response["drain"]["failed"] == 0
    assert response["drain"]["succeeded"] == 3
    assert len(references_seen) == 1 and len(references_seen[0]) == 3
    assert full_scan.call_count == 0
    assert {index._records[record_id].filters["category"] for record_id in index._records} == {
        "entity", "session_archive", "session_summary"}
    assert len(index._records) == 6
    assert manager.get_or_create(ctx.session_id).request_completed("request-one")
    assert store.count_unfinished(ctx.account_id)["total"] == 0
    assert current_outbox_events() is None
