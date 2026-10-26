"""Publication preserves pending members and revalidates source-only evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.models import RequestContext, TypedQuery
from scene.adapters import CanonicalSceneAdapter
from scene.decision import SceneDecisionEngine
from scene.decision_prompts import DECISION_PROMPT
from scene.jobs import InMemorySceneJobStore, SceneWorker
from scene.sql_store import SQLSceneJobStore
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_adapters import Embedder, IndexedVectors, adapter, scene
from tests.unit.scene.test_decision_evidence import node
from tests.unit.scene.test_decision_preflight import JSONModel


@pytest.fixture(params=[False, True], ids=["memory", "sql"])
def publication(request):
    nodes = [node(index) for index in range(6)]
    nodes = [replace(item, metadata={**item.metadata, "version": 1}) for item in nodes]
    ctx = RequestContext("acct", "alice", "", "session", "trace")
    if request.param:
        pool, fs, repository, jobs, _, _ = request.getfixturevalue("scene_sql")
        for item in nodes:
            fs.write_node(item, ctx)
        source = CanonicalSceneAdapter(repository, fs, IndexedVectors(nodes), Embedder())
    else:
        source = adapter(nodes)
        jobs = InMemorySceneJobStore()

    def restart():
        if request.param:
            return SQLSceneJobStore(pool)
        return jobs

    return SimpleNamespace(source=source, jobs=jobs, nodes=nodes, ctx=ctx, restart=restart)


def proposal(nodes, target=None):
    uris = [item.uri for item in nodes]
    return {"action": "UPDATE" if target else "CREATE", "target_scene_ids": [target] if target else [],
            "draft": {"title": "Release review", "description": "Approved release milestones", "member_uris": uris},
            "evidence_uris": uris, "unassigned_member_uris": []}


@pytest.mark.parametrize("shared_member", [False, True])
def test_updated_member_remains_pending_until_every_scene_is_current(publication, shared_member):
    source, jobs, nodes, ctx = publication.source, publication.jobs, publication.nodes, publication.ctx
    second_members = nodes[3:] + (nodes[:1] if shared_member else [])
    initial = source.repository.load_snapshot("acct", "user:alice")
    assert source.repository.compare_and_swap(initial, replace(initial, revision=1, scenes=(
        scene("a", nodes[:3]), scene("b", second_members))))
    updated = replace(nodes[0], content="Approved revised milestone", metadata={**nodes[0].metadata, "version": 2})
    source.fs.write_node(updated, ctx)
    jobs.enqueue("acct", "user:alice", [item.uri for item in nodes], token_count=6)
    engine = SceneDecisionEngine(JSONModel({"status": "READY"}, proposal(second_members, "b"), {"decision": "ACCEPT"}))
    completed = SceneWorker(jobs, engine, source).process_next("publish-b")
    assert completed.job.status == "SUCCEEDED"
    assert completed.decision.pending_member_uris == (updated.uri,)
    jobs = publication.restart()
    jobs.enqueue("acct", "user:alice", [nodes[1].uri], token_count=1)
    resumed = jobs.claim("publish-a")
    assert updated.uri in resumed.node_uris
    engine = SceneDecisionEngine(JSONModel({"status": "READY"}, proposal(nodes[:3], "a"), {"decision": "ACCEPT"}))
    completed = SceneWorker(jobs, engine, source).process(resumed)
    assert completed.job.status == "SUCCEEDED"
    assert completed.decision.pending_member_uris == ()
    current = source.repository.load_snapshot("acct", "user:alice")
    assert all(member.node_version == 2 for item in current.scenes for member in item.cluster.members
               if member.context_uri == updated.uri)
    query = TypedQuery("Release review", "MEMORY", [])
    hits = [hit for hit in source.search(query.text, ctx, limit=5, typed_query=query) if hit.scene_id == "a"]
    assert updated.uri in {hit.uri for hit in source.resolve_members(hits, ctx, limit=10, typed_query=query)}


class SourceModel(JSONModel):
    def __init__(self, draft, reference, on_review=lambda: None):
        super().__init__(draft, {"decision": "ACCEPT"})
        self.reference, self.on_review = reference, on_review

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        return [{"tool": "read_source_context", "input": {"source_refs": [self.reference]}}], ""

    def complete_json(self, prompt, schema):
        result = super().complete_json(prompt, schema)
        if schema["title"] == "QualityGateOutput":
            self.on_review()
        return result


@pytest.mark.parametrize("change", ["update", "archive"])
@pytest.mark.parametrize("timing", ["after_read", "quality"])
def test_source_read_recovers_from_change_then_publishes_current_evidence(publication, change, timing):
    source, jobs, nodes, ctx = publication.source, publication.jobs, publication.nodes, publication.ctx
    external = nodes[-1]
    source.vector_index.vectors.pop(external.uri)
    read = source.read_source_context

    def mutate():
        if change == "archive" and hasattr(source.fs, "archive_node"):
            source.fs.archive_node(external.uri, ctx)
        else:
            source.fs.write_node(replace(external, content="Revised source", metadata={
                **external.metadata, "version": 2, "status": "ARCHIVED" if change == "archive" else "ACTIVE",
            }), ctx)

    def read_then_change(*args):
        cards = read(*args)
        if timing == "after_read":
            mutate()
        return cards

    source.read_source_context = read_then_change
    draft = proposal(nodes[:3])
    model = SourceModel(draft, external.uri, mutate if timing == "quality" else lambda: None)
    jobs.enqueue("acct", "user:alice", [item.uri for item in nodes[:3]], token_count=3)
    outcome = SceneWorker(jobs, SceneDecisionEngine(model, evidence_reader=source), source,
                          retry_delay_seconds=0).process_next("read-old-source")
    assert outcome.job.status == "PENDING"
    assert outcome.job.failure_count == 1
    source.fs.write_node(replace(external, content="Current approved source", metadata={
        **external.metadata, "version": 3, "status": "ACTIVE",
    }), ctx)
    source.read_source_context = read
    jobs = publication.restart()
    resumed = jobs.claim("read-current-source")
    model = SourceModel(draft, external.uri)
    completed = SceneWorker(jobs, SceneDecisionEngine(model, evidence_reader=source), source).process(resumed)
    assert completed.job.status == "SUCCEEDED"
    current = source.repository.load_snapshot("acct", "user:alice")
    assert current.revision == 1
    assert {member.context_uri for member in current.scenes[0].cluster.members} == set(draft["draft"]["member_uris"])
    packet = json.loads(model.requests[0][0].removeprefix(DECISION_PROMPT))
    assert {card["uri"] for card in packet["memories"]} == set(draft["draft"]["member_uris"])
    assert packet["tool_evidence"][0]["results"][0]["text"] == "Current approved source"
