"""Production scene wiring with controlled storage and local model providers."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from index.index_record_builder import build_index_records
from index.outbox_worker import create_upsert_event
from providers.embedder.mock_embedder import MockEmbedder
from providers.llm.mock_llm import MockLLM
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from scene.adapters import CanonicalSceneAdapter
from scene.config import InMemorySceneConfigProvider
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import SceneDecisionResult
from scene.jobs import InMemorySceneJobStore, SceneJobStatus
from scene.manager import SceneMemoryManager
from scene.models import SceneSnapshot
from server.memory_service import MemoryService


class Repository:
    def __init__(self):
        self.snapshots = {}
        self.lexical_revisions = {}

    def pending_projections(self):
        for key, snapshot in tuple(self.snapshots.items()):
            if self.lexical_revisions.get(key, -1) < snapshot.revision:
                yield snapshot

    def publish_projection(self, snapshot, publish):
        key = snapshot.account_id, snapshot.owner_space
        if self.load_snapshot(*key).revision == snapshot.revision and self.lexical_revisions.get(key, -1) < snapshot.revision:
            publish(snapshot)
            self.lexical_revisions[key] = snapshot.revision
        return self.lexical_revisions.get(key, -1) == snapshot.revision

    def load_snapshot(self, account_id, owner_space):
        return self.snapshots.get((account_id, owner_space), SceneSnapshot(account_id, owner_space, 0))

    def compare_and_swap_checked(self, expected, replacement, check):
        return check(None) and self.compare_and_swap(expected, replacement)

    def compare_and_swap(self, expected, replacement):
        key = (expected.account_id, expected.owner_space)
        if self.load_snapshot(*key).revision != expected.revision:
            return False
        self.snapshots[key] = replacement
        return True


class CanonicalStore:
    def __init__(self):
        self.nodes = {}

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return deepcopy(self.nodes[uri])


class Outbox:
    supports_batch_claim = True

    def __init__(self, event, *, replay_after_enqueue=False):
        self.events = {event.event_id: event}
        self.replay_after_enqueue = replay_after_enqueue
        self.acknowledged = []

    def claim_batch(self, **kwargs):
        return [(event.uri, event) for event in self.events.values()]

    def is_event_current(self, event_id, node_uri):
        return event_id in self.events

    def mark_done(self, event, node_uri):
        if self.replay_after_enqueue:
            self.replay_after_enqueue = False
            raise RuntimeError("outbox acknowledgement interrupted")
        self.acknowledged.append(event.event_id)
        self.events.pop(event.event_id)

    def increment_retry(self, event, node_uri, next_retry_at):
        event.retry_count += 1

    def release(self, event, node_uri):
        pass


@pytest.fixture
def runtime(monkeypatch):
    config = OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        sql_connection_string="host=scene-runtime-test dbname=scene", scene_enabled=True,
        scene_options={"analysis_trigger_tokens": 256, "query_threshold": 0.0},
        retrieval_expand_hierarchy=False,
    )
    service = MemoryService(config=config)
    pool, fs, repository = object(), CanonicalStore(), Repository()
    vector, embedder, llm = InMemoryVectorIndex(dimension=8), MockEmbedder(dimension=8), MockLLM()
    factories = {
        "repository": Mock(return_value=repository),
        "jobs": Mock(side_effect=lambda pool: InMemorySceneJobStore()),
        "config": Mock(side_effect=lambda pool, default: InMemorySceneConfigProvider(default)),
    }
    monkeypatch.setattr("scene.sql_store.SQLSceneRepository", factories["repository"])
    monkeypatch.setattr("scene.sql_store.SQLSceneJobStore", factories["jobs"])
    monkeypatch.setattr("scene.sql_store.SQLSceneConfigProvider", factories["config"])
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_shared_vector_index", lambda: vector)
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: vector)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: embedder)
    monkeypatch.setattr(service, "get_llm", lambda: llm)
    monkeypatch.setattr(service, "_get_archive_store", lambda: None)
    monkeypatch.setattr(service, "_get_relation_store", lambda: None)
    starts = []
    original_start = SceneMemoryManager.start

    def start(manager):
        starts.append(manager)
        return original_start(manager, poll_seconds=.01)

    monkeypatch.setattr(SceneMemoryManager, "start", start)
    ctx = RequestContext("acme", "alice", "helper", "session", "trace")
    yield SimpleNamespace(service=service, pool=pool, fs=fs, repository=repository, vector=vector,
                          embedder=embedder, factories=factories, starts=starts, ctx=ctx)
    service.shutdown(join_timeout=2)


def _node(version=7):
    return ContextNode("ctx://acme/users/alice/memories/events/release", "MEMORY", "event", 2,
                       "user:alice", "Release planning", "Coordinate release checks.",
                       "Release on Monday after review.", {"version": version})


def _index(runtime, monkeypatch, *, replay=False):
    current = _node()
    runtime.fs.nodes[current.uri] = current
    event = create_upsert_event(current.uri, build_index_records(_node(version=1)))
    outbox = Outbox(event, replay_after_enqueue=replay)
    monkeypatch.setattr(runtime.service, "_get_outbox_store", lambda fs: outbox)
    first = runtime.service.drain_outbox_sync(account_id="acme")
    if replay:
        assert first["failed"] == 1
        assert runtime.service.drain_outbox_sync(account_id="acme")["succeeded"] == 1
    else:
        assert first["succeeded"] == 1
    return current, event, outbox


def test_enabled_runtime_initializes_shared_components_once(runtime):
    with ThreadPoolExecutor(max_workers=8) as callers:
        services = list(callers.map(lambda _: runtime.service.get_scene_service(), range(16)))
    assert len({id(service) for service in services}) == 1
    manager = runtime.service._scene_manager
    assert isinstance(manager, SceneMemoryManager)
    assert isinstance(manager.engine, SceneDecisionEngine)
    assert isinstance(manager.scene_store, CanonicalSceneAdapter)
    assert manager.scene_store is runtime.service._scene_source
    assert manager.scene_store.repository is runtime.repository
    assert manager.scene_store.fs is runtime.fs
    assert manager.scene_store.vector_index is runtime.vector
    assert manager.scene_store.embedder is runtime.embedder
    assert manager.engine.config.analysis_trigger_tokens == 256
    assert services[0].config_provider is manager._config_provider
    assert runtime.starts == [manager]
    for factory in runtime.factories.values():
        factory.assert_called_once()
        assert factory.call_args.args == (runtime.pool,)


def test_outbox_callback_projects_latest_version_and_replays_notice_once(runtime, monkeypatch):
    node, event, outbox = _index(runtime, monkeypatch, replay=True)
    assert event.payload["_memory_projection_version"] == 7
    assert outbox.acknowledged == [event.event_id]
    manager = runtime.service._scene_manager
    status = manager.jobs.summary("acme", "user:alice")
    assert status["pending_nodes"] == 1
    assert status["pending_tokens"] == (len(node.content) + 3) // 4
    batch = manager.scene_store.load_input("acme", "user:alice", node_uris=[node.uri])
    assert batch.nodes[0].metadata["version"] == 7
    assert batch.nodes[0].content == node.content


def test_read_api_reuses_scene_runtime_and_returns_canonical_members(runtime, monkeypatch):
    node, _, _ = _index(runtime, monkeypatch)
    scenes = runtime.service.get_scene_service()
    created = scenes.save(runtime.ctx, {
        "revision": 0, "title": node.abstract, "description": node.overview, "member_uris": [node.uri],
    })
    read_api = runtime.service.get_read_api(start_outbox_worker=False)
    assert read_api is runtime.service.get_read_api(start_outbox_worker=False)
    assert read_api._pipeline._scene_stage.source is runtime.service._scene_source
    assert read_api._pipeline._scene_stage.ranker is runtime.service._scene_ranker
    result = read_api.search_memory(node.abstract, runtime.ctx, top_k=5, fill_content_for_top_k=5)
    assert [hit.uri for hit in result.hits] == [node.uri]
    hit = result.hits[0]
    assert hit.content_excerpt == node.content
    assert hit.metadata["scene_match"]
    assert created["scene"]["scene_id"] in hit.metadata["scene_ids"]


def test_shutdown_retains_dependencies_until_current_scene_job_completes(runtime, monkeypatch):
    runtime.service.get_scene_service()
    manager = runtime.service._scene_manager
    entered, release = Event(), Event()

    class BlockingEngine:
        config = manager.engine.config

        def for_config(self, config):
            return self

        def refresh(self, store, account_id, owner_space, *, node_uris):
            entered.set()
            release.wait(3)
            return SceneDecisionResult(SceneSnapshot(account_id, owner_space, 0), "WAIT")

    manager.worker.engine = BlockingEngine()
    job = manager.after_memory_write("acme", "user:alice", ["pending-evidence"], token_count=1, force=True)
    closed = []
    runtime.service._extraction_retry = SimpleNamespace(close=lambda timeout: closed.append("extraction"))
    runtime.service._outbox_listener = SimpleNamespace(stop=lambda: closed.append("outbox"))
    runtime.service._projection_index = SimpleNamespace(close=lambda **kwargs: closed.append("projection") or True)
    runtime.service._lexical_index = object()
    monkeypatch.setattr(runtime.service, "stop_skill_sedimentation", lambda **kwargs: closed.append("skills"))
    try:
        assert entered.wait(1)
        runtime.service.shutdown(join_timeout=.01)
        assert manager.running
        assert closed == []
    finally:
        release.set()
        assert manager.stop(timeout=2)
    runtime.service.shutdown(join_timeout=1)
    assert manager.jobs.get(job.job_id).status == SceneJobStatus.WAITING
    assert closed == ["extraction", "skills", "outbox", "projection"]


def test_yaml_scene_settings_enable_runtime(monkeypatch):
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda path: {
        "scene": {"enabled": True, "options": {"analysis_trigger_tokens": 256}},
    })
    config = OgMemConfig.load("scene-test.yaml")
    assert config.scene_enabled is True
    assert config.scene_generation_enabled is True
    assert config.scene_retrieval_enabled is True
    assert config.scene_options == {"analysis_trigger_tokens": 256}


def test_live_configuration_applies_to_enqueue_and_request_snapshot(runtime):
    scenes = runtime.service.get_scene_service()
    current = scenes.configuration(runtime.ctx)
    updated = scenes.update_configuration(runtime.ctx, {
        "revision": current["revision"],
        "values": {"analysis_trigger_tokens": 4096, "query_threshold": .9},
    })
    manager = runtime.service._scene_manager
    manager.after_memory_write("acme", "user:alice", ["next-memory"], token_count=1024)
    state = manager.jobs.summary("acme", "user:alice")
    assert state["pending_tokens"] == 1024
    assert updated["values"]["analysis_trigger_tokens"] == 4096
    assert manager._effective_config("acme", "user:alice").query_threshold == .9
    from core.models import TypedQuery
    pinned = runtime.service._scene_source.for_request(
        runtime.ctx, [TypedQuery("release", "MEMORY", [])],
    )
    assert pinned.config.query_threshold == .9


def test_scene_notice_retries_through_outbox_until_durably_enqueued(runtime, monkeypatch):
    runtime.service.get_scene_service()
    manager = runtime.service._scene_manager
    enqueue = manager.after_memory_write
    attempts = []

    def recoverable_enqueue(*args, **kwargs):
        attempts.append(kwargs["notice_id"])
        if len(attempts) == 1:
            raise RuntimeError("queue transaction interrupted")
        return enqueue(*args, **kwargs)

    monkeypatch.setattr(manager, "after_memory_write", recoverable_enqueue)
    node = _node()
    runtime.fs.nodes[node.uri] = node
    event = create_upsert_event(node.uri, build_index_records(node))
    outbox = Outbox(event)
    monkeypatch.setattr(runtime.service, "_get_outbox_store", lambda fs: outbox)
    first = runtime.service.drain_outbox_sync(account_id="acme")
    assert first["failed"] == 1
    second = runtime.service.drain_outbox_sync(account_id="acme")
    assert second["succeeded"] == 1
    assert attempts == [event.event_id, event.event_id]
    assert outbox.acknowledged == [event.event_id]
    assert manager.jobs.summary("acme", "user:alice")["pending_nodes"] == 1


def test_indexing_converges_after_canonical_change(runtime, monkeypatch):
    node = _node()
    runtime.fs.nodes[node.uri] = node
    event = create_upsert_event(node.uri, build_index_records(node))
    outbox = Outbox(event)
    monkeypatch.setattr(runtime.service, "_get_outbox_store", lambda fs: outbox)
    original = runtime.vector.upsert
    calls = []

    def concurrent_change(records):
        original(records)
        calls.append(1)
        if len(calls) == 1:
            runtime.fs.nodes[node.uri] = _node(version=8)

    monkeypatch.setattr(runtime.vector, "upsert", concurrent_change)
    assert runtime.service.drain_outbox_sync(account_id="acme")["failed"] == 1
    assert runtime.service.drain_outbox_sync(account_id="acme")["succeeded"] == 1
    batch = runtime.service._scene_source.load_input("acme", "user:alice", node_uris=[node.uri])
    assert batch.nodes[0].metadata["version"] == 8
    assert outbox.acknowledged == [event.event_id]


def test_wait_scenes_flushes_then_finishes_retained_waiting_window(runtime, monkeypatch):
    runtime.service.get_scene_service()
    manager = runtime.service._scene_manager

    class WaitingEngine:
        config = manager.engine.config

        def for_config(self, config):
            return self

        def refresh(self, store, account_id, owner_space, *, node_uris):
            return SceneDecisionResult(SceneSnapshot(account_id, owner_space, 0), "WAIT",
                                       pending_member_uris=tuple(node_uris))

    manager.worker.engine = WaitingEngine()
    manager.after_memory_write("acme", "user:alice", ["pending-evidence"], token_count=1)
    monkeypatch.setattr(runtime.service, "_wait_until_session_idle", lambda params: {
        "idle": True, "reason": "idle", "extraction_in_progress": False,
    })
    result = runtime.service.wait_until_idle({
        "sessionId": "session", "_ctx": runtime.ctx, "waitScenes": True, "timeoutSeconds": 2,
    })
    assert result["idle"] is True
    state = result["scenes"]["user:alice"]
    assert state["last_job"]["status"] == "WAITING"
    assert state["retained_nodes"] == 1


def test_lexical_configuration_hot_enable_reuses_existing_scene_projection(runtime, monkeypatch):
    from core.models import TypedQuery
    from tests.unit.scene.test_candidate_index import MemoryIndex

    lexical = MemoryIndex()
    factory = Mock(return_value=lexical)
    monkeypatch.setattr("providers.lexical_index.factory.create_lexical_index", factory)
    node, _, _ = _index(runtime, monkeypatch)
    scenes = runtime.service.get_scene_service()
    created = scenes.save(runtime.ctx, {
        "revision": 0, "title": node.abstract, "description": node.overview, "member_uris": [node.uri],
    })
    query = TypedQuery(node.abstract, "MEMORY", [])
    for enabled in (False, True, False, True):
        revision = scenes.configuration(runtime.ctx)["revision"]
        scenes.update_configuration(runtime.ctx, {
            "revision": revision, "values": {"lexical_enabled": enabled, "query_threshold": 0.0},
        })
        hits = runtime.service._scene_source.search(
            node.abstract, runtime.ctx, limit=5, typed_query=query,
        )
        hit = next(item for item in hits if item.scene_id == created["scene"]["scene_id"])
        assert ("lexical" in hit.routes) == enabled
    assert factory.call_count == 1
    assert lexical.events.count("publish") == 1
    runtime.service._scene_manager.process_next("projection-check")
    assert runtime.repository.lexical_revisions["acme", "user:alice"] == 1


@pytest.mark.parametrize("generation,retrieval", [(True, True), (True, False), (False, True), (False, False)])
def test_scene_modes_preserve_members_and_query_selection(runtime, monkeypatch, generation, retrieval):
    runtime.service._cfg.scene_generation_enabled = generation
    runtime.service._cfg.scene_retrieval_enabled = retrieval
    node, _, _ = _index(runtime, monkeypatch)
    scenes = runtime.service.get_scene_service()
    scenes.save(runtime.ctx, {
        "revision": 0, "title": node.abstract, "description": node.overview,
        "member_uris": [node.uri],
    })
    before = runtime.repository.load_snapshot("acme", "user:alice")
    api = runtime.service.get_read_api(start_outbox_worker=False)
    assert runtime.service._scene_manager.running is generation
    assert len(runtime.starts) == int(generation)
    from retrieval.identity import canonical_uri
    for selection in (None, True, False, None):
        result = api.search_memory(node.content, runtime.ctx, top_k=5, fill_content_for_top_k=5,
                                   score_threshold=0.0, include_debug=True, use_scene=selection)
        assert [canonical_uri(hit.uri) for hit in result.hits] == [node.uri]
        expected = retrieval if selection is None else selection
        assert bool(result.hits[0].metadata.get("scene_match")) is expected
    assert runtime.repository.load_snapshot("acme", "user:alice") == before
    status = runtime.service._scene_manager.jobs.summary("acme", "user:alice")
    assert status["pending_nodes"] == int(generation)


@pytest.mark.parametrize("generation,retrieval", [(True, False), (False, True), (True, True)])
def test_yaml_and_environment_scene_modes(monkeypatch, generation, retrieval):
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda path: {
        "scene": {"enabled": True, "generation_enabled": generation, "retrieval_enabled": retrieval},
    })
    config = OgMemConfig.load("scene-test.yaml")
    assert (config.scene_generation_enabled, config.scene_retrieval_enabled) == (generation, retrieval)
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda path: {"scene": {"enabled": True}})
    monkeypatch.setenv("OGMEM_SCENE_GENERATION_ENABLED", str(not generation).lower())
    monkeypatch.setenv("OGMEM_SCENE_RETRIEVAL_ENABLED", str(not retrieval).lower())
    config = OgMemConfig.load("scene-test.yaml")
    assert (config.scene_generation_enabled, config.scene_retrieval_enabled) == (not generation, not retrieval)
