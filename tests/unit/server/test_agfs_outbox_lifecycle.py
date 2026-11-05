"""Normal AGFS publication across worker stop and restart boundaries."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from time import monotonic
from unittest.mock import Mock

from commit.outbox_store import OutboxStore
from core.lexical import HybridLexicalIndex
from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from index.index_record_builder import build_index_records
from providers.composite_index import CompositeProjectionWriter
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from tests.unit.server.test_agent_memory import _MemoryAGFS


CTX = RequestContext("outbox-lifecycle", "ben", "openclaw", "atlas", "request")


class CompletedOutbox(OutboxStore):
    def __init__(self, client, fs):
        super().__init__(client, fs)
        self.completed = Event()

    def mark_done(self, event, node_uri):
        super().mark_done(event, node_uri)
        self.completed.set()


def runtime(monkeypatch, *, embedder=None):
    client = _MemoryAGFS()
    client.api_base = "memory://outbox-lifecycle"
    fs = AGFSContextFS(client)
    store = CompletedOutbox(client, fs)
    service = MemoryService(config=OgMemConfig(
        provider="mock", openai_api_key="", vector_db_type="memory", storage_backend="agfs",
        account_id=CTX.account_id, directory_summary_enabled=False,
    ))
    service._embedder = embedder or MockEmbedder(16)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda context_fs: store)
    monkeypatch.setattr(service._provider_cfg, "create_vector_index",
                        lambda: InMemoryVectorIndex(16))
    return service, fs, store


def publish_atlas(fs, store, *, revision=1):
    uri = f"ctx://{CTX.account_id}/users/ben/memories/preferences/atlas_release"
    text = "Ben prefers Atlas releases on Friday at 19:00 using npm."
    if revision == 1:
        node = ContextNode(uri, "MEMORY", "preference", 0, CTX.user_space_name(),
                           text, text, text, {"who": "ben", "scope_ref": "Atlas"})
    else:
        previous = fs.read_node(uri, CTX)
        text += " Run npm test before each release."
        node = replace(previous, abstract=text, overview=text, content=text,
                       metadata={**previous.metadata,
                                 "_expected_version": previous.metadata["version"]})
    fs.write_node(node, CTX)
    saved = fs.read_node(uri, CTX)
    store.completed.clear()
    return saved, store.register_write(saved, CTX)


def assert_published(store, index, node, event):
    assert store.completed.wait(3)
    assert store.list_pending(CTX.account_id) == []
    expected = build_index_records(node)
    assert {record.id for record in expected} == set(index._records)
    assert [(index._records[record.id].level, index._records[record.id].text)
            for record in expected] == [(record.level, record.text) for record in expected]
    assert store.count_unfinished(CTX.account_id)["total"] == 0


def test_completed_worker_stops_and_restarts_after_parallel_start(monkeypatch):
    service, fs, store = runtime(monkeypatch)
    node, event = publish_atlas(fs, store)
    ready = Barrier(2)

    def start():
        ready.wait(timeout=3)
        service._start_outbox_worker()
        return service._outbox_thread

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = [future.result(timeout=3)
                             for future in [pool.submit(start), pool.submit(start)]]
        assert first is second
        assert first.is_alive()
        assert_published(store, service._vector_index, node, event)
        started = monotonic()
        service.shutdown(join_timeout=1)
        assert monotonic() - started < 1
        assert first.is_alive() is False
        assert service._outbox_thread is None

        node, event = publish_atlas(fs, store, revision=2)
        service._start_outbox_worker()
        restarted = service._outbox_thread
        assert restarted is not first
        assert restarted.is_alive()
        assert_published(store, service._vector_index, node, event)
        service.shutdown(join_timeout=1)
        assert restarted.is_alive() is False
        assert service._outbox_thread is None
    finally:
        service.shutdown(join_timeout=3)


def test_shutdown_finishes_embedding_and_lazy_initialization(monkeypatch):
    entered = Event()
    lazy_initialized = Event()

    class FinishingEmbedder(MockEmbedder):
        def embed_texts(self, texts):
            entered.set()
            assert service._outbox_stop.wait(3)
            assert service.get_llm() is service._llm
            lazy_initialized.set()
            return super().embed_texts(texts)

    service, fs, store = runtime(monkeypatch, embedder=FinishingEmbedder(16))
    node, event = publish_atlas(fs, store)
    try:
        service._start_outbox_worker()
        worker = service._outbox_thread
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(service.shutdown, join_timeout=1).result(timeout=3)
        assert lazy_initialized.is_set()
        assert_published(store, service._vector_index, node, event)
        assert worker.is_alive() is False
        assert service._outbox_thread is None
    finally:
        service.shutdown(join_timeout=3)


def test_parallel_shutdown_serializes_close_and_restarts_new_projection(monkeypatch):
    service, fs, store = runtime(monkeypatch)
    vectors = [InMemoryVectorIndex(16), InMemoryVectorIndex(16)]
    lexicals = [Mock(spec=HybridLexicalIndex), Mock(spec=HybridLexicalIndex)]
    for vector, lexical in zip(vectors, lexicals):
        vector.close = Mock(return_value=True)
        lexical.close.return_value = True
    generations = iter(zip(vectors, lexicals))

    def create_projection():
        vector, lexical = next(generations)
        service._vector_index, service._lexical_index = vector, lexical
        service._projection_index = CompositeProjectionWriter(vector, lexical)
        return service._projection_index

    projection_factory = Mock(side_effect=create_projection)
    monkeypatch.setattr(service, "_get_shared_projection_index", projection_factory)
    close_entered, finish_close = Event(), Event()

    def close_lexical(timeout):
        close_entered.set()
        assert finish_close.wait(3)
        return True

    lexicals[0].close.side_effect = close_lexical
    node, event = publish_atlas(fs, store)
    ready = Barrier(2)

    def shutdown():
        ready.wait(timeout=3)
        service.shutdown(join_timeout=1)

    try:
        service._start_outbox_worker()
        original_worker, original_projection = service._outbox_thread, service._projection_index
        assert_published(store, vectors[0], node, event)
        with ThreadPoolExecutor(max_workers=2) as pool:
            stops = [pool.submit(shutdown), pool.submit(shutdown)]
            try:
                assert close_entered.wait(3)
                assert original_worker.is_alive() is False
                assert service._outbox_stop.is_set()
                service._start_outbox_worker()
                assert service._outbox_thread is original_worker
                assert service._projection_index is original_projection
                assert projection_factory.call_count == 1
            finally:
                finish_close.set()
            for future in stops:
                future.result(timeout=3)
        vectors[0].close.assert_called_once_with()
        lexicals[0].close.assert_called_once()
        assert service._outbox_thread is service._projection_index is None

        node, event = publish_atlas(fs, store, revision=2)
        service._start_outbox_worker()
        restarted = service._outbox_thread
        assert restarted is not original_worker
        assert service._projection_index is not original_projection
        assert service._projection_index.vector_index is vectors[1]
        assert service._projection_index.lexical_index is lexicals[1]
        assert projection_factory.call_count == 2
        assert_published(store, vectors[1], node, event)
        service.shutdown(join_timeout=1)
        assert restarted.is_alive() is False
        vectors[1].close.assert_called_once_with()
        lexicals[1].close.assert_called_once()
    finally:
        finish_close.set()
        service.shutdown(join_timeout=3)
