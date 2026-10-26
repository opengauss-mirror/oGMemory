"""Deletion cleanup and search evidence recover through the scene worker."""

import json
from dataclasses import replace

import pytest

from core.models import SeedHit
from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import DecisionLimits
from scene.decision_prompts import DECISION_PROMPT
from scene.jobs import SceneWorker
from scene.service import SceneService
from tests.integration.test_scene_publication_evidence_recovery import proposal
from tests.integration.test_scene_publication_evidence_recovery import publication as publication
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_adapters import scene
from tests.unit.scene.test_decision_preflight import JSONModel


def change_memory(source, node, ctx, change):
    if change == "delete":
        if hasattr(source.fs, "delete_node"):
            source.fs.delete_node(node.uri, ctx)
        else:
            source.fs.nodes.pop(node.uri)
    elif change == "archive" and hasattr(source.fs, "archive_node"):
        source.fs.archive_node(node.uri, ctx)
    else:
        source.fs.write_node(replace(node, content="Changed release evidence", metadata={
            **node.metadata, "version": 2, "status": "ARCHIVED" if change == "archive" else "ACTIVE",
            "source_refs": ["changed-source"],
        }), ctx)


@pytest.mark.parametrize("change", ["delete", "archive"])
@pytest.mark.parametrize("candidate_limit", [1, 8])
def test_removed_member_cleans_all_scenes_before_fresh_publication(publication, change, candidate_limit):
    source, jobs, nodes, ctx = publication.source, publication.jobs, publication.nodes, publication.ctx
    source.config = SceneConfig(max_candidate_scenes=candidate_limit)
    affected = tuple(scene(f"old-{number}", nodes[:3]) for number in range(candidate_limit + 1))
    keep = scene("unrelated", nodes[3:])
    original = source.repository.load_snapshot("acct", "user:alice")
    assert source.repository.compare_and_swap(original, replace(original, revision=1, scenes=(*affected, keep)))
    change_memory(source, nodes[0], ctx, change)
    jobs.enqueue("acct", "user:alice", [nodes[0].uri], force=True)
    outcome = SceneWorker(jobs, SceneDecisionEngine(JSONModel(), source.config), source).process_next("cleanup")
    current = source.repository.load_snapshot("acct", "user:alice")
    assert [(item.status, item.version) for item in current.scenes if item.scene_id.startswith("old-")] == [
        ("ARCHIVED", 2)] * len(affected)
    service = SceneService(source.repository, source, None, None, None)
    assert [item["scene_id"] for item in service.list(ctx)["items"]] == [keep.scene_id]
    assert outcome.job.status == "WAITING"
    jobs = publication.restart()
    assert jobs.summary("acct", "user:alice")["retained_nodes"] == 0
    source.fs.write_node(replace(nodes[0], metadata={**nodes[0].metadata, "version": 3}), ctx)
    jobs.enqueue("acct", "user:alice", [item.uri for item in nodes[:3]], force=True)
    model = JSONModel({"status": "READY"}, proposal(nodes[:3]), {"decision": "ACCEPT"})
    completed = SceneWorker(jobs, SceneDecisionEngine(model, source.config), source).process_next("fresh")
    assert completed.job.status == "SUCCEEDED"
    assert service.list(ctx)["total"] == 2
    published = next(item for item in completed.decision.snapshot.scenes
                     if item.status == "ACTIVE" and item.scene_id != keep.scene_id)
    assert {member.context_uri: member.node_version for member in published.cluster.members}[nodes[0].uri] == 3


class SearchModel(JSONModel):
    def __init__(self, draft):
        super().__init__(draft, {"decision": "ACCEPT"})

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        return [{"tool": "search_related_nodes", "input": {"query": "release"}}], ""


@pytest.mark.parametrize("change", ["update", "delete", "archive"])
@pytest.mark.parametrize("window", ["expanded", "bounded"])
def test_search_card_change_recovers_with_current_evidence(publication, change, window):
    source, jobs, nodes, ctx = publication.source, publication.jobs, publication.nodes, publication.ctx
    external = nodes[-1]
    source.vector_index.hits = [SeedHit(external.uri, .9)]
    search = source.search_related_nodes

    def search_then_change(*args):
        cards = search(*args)
        change_memory(source, external, ctx, change)
        return cards

    source.search_related_nodes = search_then_change
    limits = DecisionLimits(max_input_nodes=3 if window == "bounded" else 128)
    draft = proposal(nodes[:3])
    if window == "expanded" and change == "update":
        draft["unassigned_member_uris"] = [external.uri]
    jobs.enqueue("acct", "user:alice", [item.uri for item in nodes[:3]], force=True)
    model = SearchModel(draft)
    engine = SceneDecisionEngine(model, limits=limits, evidence_reader=source)
    outcome = SceneWorker(jobs, engine, source, retry_delay_seconds=0).process_next("search-old")
    assert outcome.job.status == "PENDING"
    assert outcome.job.failure_count == 1
    restored = replace(external, content="Current reviewed milestone", metadata={
        **external.metadata, "version": 3, "status": "ACTIVE", "source_refs": ["reviewed-source"],
    })
    source.fs.write_node(restored, ctx)
    source.search_related_nodes = search
    jobs = publication.restart()
    draft["unassigned_member_uris"] = [external.uri] if window == "expanded" else []
    model = SearchModel(draft)
    engine = SceneDecisionEngine(model, limits=limits, evidence_reader=source)
    completed = SceneWorker(jobs, engine, source).process_next("search-current")
    assert completed.job.status == "SUCCEEDED"
    packet = json.loads(model.requests[0][0].removeprefix(DECISION_PROMPT))
    card = packet["tool_evidence"][0]["results"][0]
    assert card["text"].endswith(restored.content)
    assert card["source_refs"] == ["reviewed-source"]
    current = source.repository.load_snapshot("acct", "user:alice")
    assert current.revision == 1
    assert {member.context_uri for member in current.scenes[0].cluster.members} == set(draft["draft"]["member_uris"])
