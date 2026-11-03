"""Scene publication and evidence-window recovery acceptance."""

from dataclasses import replace

import pytest

from fs.sql_adapter import SharedConnectionPool, SQLContextFS
from scene.adapters import CanonicalSceneAdapter
from scene.candidate_index import LazySceneIndex, SceneIndexRefresh
from scene.decision import SceneDecisionEngine
from scene.jobs import InMemorySceneJobStore, SceneWorker
from scene.service import SceneService
from scene.sql_store import SQLSceneConfigProvider, SQLSceneJobStore, SQLSceneRepository
from tests.integration.test_scene_sql_persistence import _scene, _write
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_adapters import Embedder, IndexedVectors
from tests.unit.scene.test_candidate_index import MemoryIndex
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import EvidenceModel, source_case


@pytest.mark.parametrize("manual", [False, True])
def test_single_connection_publishes_checked_evidence(scene_sql, manual):
    original_pool, _, _, _, _, ctx = scene_sql
    pool = SharedConnectionPool(original_pool.connection_string, pool_size=1,
                                dialect=original_pool.dialect)
    acquire = pool.get_connection
    pool.get_connection = lambda: acquire(timeout=.25)
    try:
        fs, repository = SQLContextFS(pool=pool), SQLSceneRepository(pool)
        configuration = SQLSceneConfigProvider(pool)
        memory = _write(fs, ctx, "single-slot")
        adapter = CanonicalSceneAdapter(repository, fs, IndexedVectors([memory]), Embedder())
        if manual:
            service = SceneService(repository, adapter, None, configuration, Embedder())
            result = service.save(ctx, {"revision": 0, "title": "Release", "description": "Plan release",
                                        "member_uris": [memory.uri]})
            assert result["revision"] == 1
        else:
            batch = adapter.load_input(ctx.account_id, "user:alice", node_uris=[memory.uri])
            published = replace(batch.snapshot, revision=1, scenes=(_scene("release", batch.nodes[0]),))
            assert adapter.compare_and_swap(batch.snapshot, published, evidence_nodes=batch.nodes)
        assert repository.load_snapshot(ctx.account_id, "user:alice").revision == 1
        fs.archive_node(memory.uri, ctx)
        refreshed = adapter.load_input(ctx.account_id, "user:alice", node_uris=[memory.uri])
        assert refreshed.snapshot.scenes[0].status == "ARCHIVED"
        assert refreshed.unavailable_uris == (memory.uri,)
    finally:
        pool.close_all()


@pytest.mark.parametrize("durable", [False, True])
def test_expanded_wait_window_survives_new_evidence_and_restart(request, durable):
    pool = request.getfixturevalue("scene_sql")[0] if durable else None
    source, nodes, _, proposal = source_case()
    source.search_related_nodes = lambda *args: [source.node_card(n) for n in nodes[3:]]
    wait = proposal.model_copy(update={"action": "WAIT", "draft": None, "reason": "Await next review"})
    engine = SceneDecisionEngine(EvidenceModel(wait), evidence_reader=source)
    jobs = SQLSceneJobStore(pool) if durable else InMemorySceneJobStore()
    jobs.enqueue("acct", "user:alice", [n.uri for n in nodes[:3]], token_count=3)
    outcome = SceneWorker(jobs, engine, source).process_next("worker")
    assert outcome.decision.action == "WAIT"
    assert set(outcome.decision.pending_member_uris) == {n.uri for n in nodes}
    if durable:
        jobs = SQLSceneJobStore(pool)
    jobs.enqueue("acct", "user:alice", [nodes[0].uri], token_count=1)
    resumed = jobs.claim("resumed")
    assert set(resumed.node_uris) == {n.uri for n in nodes}
    engine = SceneDecisionEngine(JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"}))
    assert SceneWorker(jobs, engine, source).process(resumed).job.status == "SUCCEEDED"


@pytest.mark.parametrize("action", ["WAIT", "CREATE"])
def test_consumed_missing_evidence_leaves_live_window(action):
    source, nodes, _, proposal = source_case()
    missing = nodes[0].uri.rsplit("/", 1)[0] + "/deleted"
    model = (JSONModel({"status": "WAIT"}) if action == "WAIT" else
             JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"}))
    jobs = InMemorySceneJobStore()
    jobs.enqueue("acct", "user:alice", [missing, *(n.uri for n in nodes)], token_count=6)
    outcome = SceneWorker(jobs, SceneDecisionEngine(model), source).process_next("worker")
    assert outcome.decision.action == action
    expected = {n.uri for n in nodes} if action == "WAIT" else {nodes[-1].uri}
    assert set(jobs._scope("acct", "user:alice")["retained"]) == expected


def test_budgeted_window_preserves_unexamined_evidence():
    source, nodes, _, _ = source_case()
    source.max_input_nodes = 3
    source.fs.nodes[nodes[1].uri] = replace(nodes[1], metadata={"status": "ARCHIVED"})
    batch = source.load_input("acct", "user:alice", node_uris=[n.uri for n in nodes])
    assert batch.unavailable_uris == (nodes[1].uri,)
    result = SceneDecisionEngine(JSONModel()).refresh(
        source, "acct", "user:alice", node_uris=[n.uri for n in nodes])
    assert set(result.pending_member_uris) == {n.uri for n in nodes if n != nodes[1]}


def test_delayed_reader_preserves_newer_lexical_projection(scene_sql):
    pool, fs, repository, _, _, ctx = scene_sql
    node = _write(fs, ctx, "projection")
    original = repository.load_snapshot(ctx.account_id, "user:alice")
    old = replace(original, revision=1, scenes=(_scene("release", node),))
    assert repository.compare_and_swap(original, old)
    index = MemoryIndex()
    delayed = LazySceneIndex(lambda: index, projection_queue=repository)
    current = LazySceneIndex(lambda: index, projection_queue=SQLSceneRepository(pool))
    newer = replace(old, revision=2, scenes=(replace(old.scenes[0], version=2),))
    assert repository.compare_and_swap(old, newer)
    SceneIndexRefresh(SQLSceneRepository(pool), current)()
    delayed.search("Release", old, 5)
    assert current.search("Release", newer, 5) == ["release"]
