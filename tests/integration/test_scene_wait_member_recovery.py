"""Updated scene members survive WAIT, restart and bounded candidate selection."""

from dataclasses import replace

import pytest

from core.models import RequestContext, TypedQuery
from scene.adapters import CanonicalSceneAdapter
from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.jobs import InMemorySceneJobStore, SceneWorker
from scene.models import SceneSnapshot
from scene.sql_store import SQLSceneJobStore
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_adapters import Embedder, IndexedVectors, adapter
from tests.unit.scene.test_decision_evidence import node, record
from tests.unit.scene.test_decision_preflight import JSONModel


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("wait_stage", ["preflight", "proposal", "quality"])
def test_updated_member_waits_then_publishes_after_fresh_evidence(request, durable, wait_stage):
    config = SceneConfig(max_candidate_scenes=1)
    nodes = [node(i) for i in range(5)]
    nodes = [replace(item, metadata={**item.metadata, "version": 1}) for item in nodes]
    scenes = [record("first", nodes[:2]), record("z-other", nodes[2:])]
    scenes = tuple(replace(scene, cluster=replace(scene.cluster, members=tuple(
        replace(member, node_version=1) for member in scene.cluster.members))) for scene in scenes)
    ctx = RequestContext("acct", "alice", "", "session", "trace")
    if durable:
        pool, fs, repository, jobs, _, _ = request.getfixturevalue("scene_sql")
        for item in nodes:
            fs.write_node(item, ctx)
        initial = repository.load_snapshot("acct", "user:alice")
        assert repository.compare_and_swap(initial, replace(initial, revision=1, scenes=scenes))
        source = CanonicalSceneAdapter(repository, fs, IndexedVectors(nodes), Embedder(), config=config)
    else:
        source = adapter(nodes, config=config)
        source.repository.snapshot = SceneSnapshot("acct", "user:alice", 1, scenes)
        jobs = InMemorySceneJobStore()
    updated = replace(nodes[0], content="Reviewed Atlas release milestone", metadata={**nodes[0].metadata, "version": 2})
    source.fs.write_node(updated, ctx)
    submitted = [nodes[i].uri for i in (0, 2, 3)]
    proposal = {
        "action": "UPDATE", "target_scene_ids": ["first"],
        "draft": {"title": "Atlas release", "description": "Reviewed release milestones",
                  "member_uris": [item.uri for item in nodes[:4]]},
        "evidence_uris": [item.uri for item in nodes[:4]], "unassigned_member_uris": [],
    }
    if wait_stage == "preflight":
        replies = [{"status": "WAIT"}]
    elif wait_stage == "proposal":
        replies = [{"status": "READY"}, {"action": "WAIT"}]
    else:
        replies = [{"status": "READY"}, proposal, {"decision": "DEFER"}]
    jobs.enqueue("acct", "user:alice", submitted, token_count=3)
    outcome = SceneWorker(jobs, SceneDecisionEngine(JSONModel(*replies), config), source).process_next("first-worker")
    assert outcome.job.status == "WAITING"
    assert set(outcome.decision.pending_member_uris) == set(submitted)
    if durable:
        jobs = SQLSceneJobStore(pool)
    jobs.enqueue("acct", "user:alice", [nodes[4].uri], token_count=1)
    resumed = jobs.claim("restarted-worker")
    assert set(resumed.node_uris) == {*submitted, nodes[4].uri}
    batch = source.load_input("acct", "user:alice", node_uris=resumed.node_uris)
    packet = SceneDecisionEngine(JSONModel(), config)._packet(batch.snapshot, list(batch.nodes), batch.embeddings)
    assert [card["scene_id"] for card in packet["existing_scenes"]] == ["first"]
    engine = SceneDecisionEngine(JSONModel({"status": "READY"}, proposal, {"decision": "ACCEPT"}), config)
    completed = SceneWorker(jobs, engine, source).process(resumed)
    assert completed.job.status == "SUCCEEDED"
    current = source.repository.load_snapshot("acct", "user:alice")
    target = next(scene for scene in current.scenes if scene.scene_id == "first")
    assert target.version == 2
    assert {member.context_uri: member.node_version for member in target.cluster.members}[updated.uri] == 2
    query = TypedQuery("Atlas release", "MEMORY", [])
    hits = source.search(query.text, ctx, limit=5, typed_query=query)
    assert updated.uri in {hit.uri for hit in source.resolve_members(hits, ctx, limit=10, typed_query=query)}
