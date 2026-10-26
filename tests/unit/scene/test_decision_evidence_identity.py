"""Executable supporting-evidence and event-identity recovery scenarios."""

import json
from dataclasses import replace

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.decision_application import SceneDecisionApplier
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_proposal import SceneProposalStage
from scene.models import SceneSnapshot
from tests.unit.scene.test_adapters import adapter
from tests.unit.scene.test_decision_application import application_case
from tests.unit.scene.test_decision_preflight import JSONModel


def subset_case(action, indirect):
    snapshot, nodes, embeddings, decision, _ = application_case(action)
    archive_a = ProvenanceResolver.build_id("archive", "planning", ["m1"])
    archive_b = ProvenanceResolver.build_id("archive", "delivery", ["m2"])
    nodes[-1].metadata["source_refs"] = [archive_a]
    for item in nodes[:3]:
        item.metadata["source_refs"] = [nodes[-1].uri if indirect else archive_a]
    nodes[3].metadata["source_refs"] = [archive_b]
    packet = SceneEvidenceBuilder().build(snapshot, nodes, embeddings)
    initial = decision.model_copy(update={"evidence_uris": [item.uri for item in nodes[:3]]})
    corrected = decision.model_copy(update={"evidence_uris": [nodes[index].uri for index in (0, 1, 3)]})
    return snapshot, nodes, embeddings, packet, initial, corrected


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("entry", ["gate", "correction"])
def test_supporting_subset_recovers_with_independent_evidence(action, indirect, entry):
    snapshot, nodes, embeddings, packet, initial, corrected = subset_case(action, indirect)
    model = JSONModel(initial.model_dump(), corrected.model_dump(), {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    if entry == "gate":
        assert stage.deterministic_quality_gate(initial, packet).decision == "DEFER"
        recovered = corrected
        stage.validate(recovered, snapshot, packet)
    else:
        recovered = stage.decide(snapshot, packet, budget)
        assert recovered.evidence_uris == corrected.evidence_uris
        assert "OUTPUT CORRECTION" in model.requests[1][0]
        assert stage.review(recovered, packet, budget).decision == "ACCEPT"
        assert budget.used == 3
    assert stage.deterministic_quality_gate(recovered, packet).decision == "ACCEPT"
    result = SceneDecisionApplier().apply(snapshot, nodes, embeddings, recovered)
    assert result.revision == snapshot.revision + 1
    assert {member.context_uri for member in result.scenes[0].cluster.members} == set(recovered.draft.member_uris)


@pytest.mark.parametrize("kind", ["boundary", "nested", "title"])
def test_shared_members_publish_distinct_events_and_replay_each_identity(kind):
    _, nodes, _, decision, _ = application_case()
    for item in nodes:
        item.content = "Atlas launch milestones completed. A separate service outage was repaired."
    source = adapter(nodes)
    source.repository.snapshot = SceneSnapshot("acct", "user:alice", 0)
    drafts = []
    for activity in ("launch", "outage recovery"):
        boundary = {"subject": "Atlas", "activity": activity}
        if kind == "nested":
            boundary = {"thread": boundary}
        elif kind == "title":
            boundary = {}
        drafts.append(decision.model_copy(update={"draft": decision.draft.model_copy(update={
            "title": "Atlas " + activity if kind == "title" else "Atlas work",
            "description": "Confirmed " + activity, "boundary": boundary,
        })}))
    applier = SceneDecisionApplier()
    saved_ids = []
    for proposal in drafts:
        batch = source.load_input("acct", "user:alice", node_uris=[item.uri for item in nodes])
        packet = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
        model = JSONModel(proposal.model_dump(), {"decision": "ACCEPT"})
        stage = SceneProposalStage(SceneModelCaller(model))
        budget = CallBudget(3)
        accepted = stage.decide(batch.snapshot, packet, budget)
        assert stage.deterministic_quality_gate(accepted, packet).decision == "ACCEPT"
        assert stage.review(accepted, packet, budget).decision == "ACCEPT"
        replacement = applier.apply(batch.snapshot, list(batch.nodes), batch.embeddings, accepted)
        assert source.compare_and_swap(batch.snapshot, replacement)
        saved_ids.append(next(scene.scene_id for scene in replacement.scenes
                              if scene.cluster.summary == accepted.draft.description))
    assert len(set(saved_ids)) == 2
    batch = source.load_input("acct", "user:alice", node_uris=[item.uri for item in nodes])
    for proposal in drafts:
        assert applier.apply(batch.snapshot, list(batch.nodes), batch.embeddings, proposal) is batch.snapshot
    first = next(scene for scene in batch.snapshot.scenes if scene.scene_id == saved_ids[0])
    second = next(scene for scene in batch.snapshot.scenes if scene.scene_id == saved_ids[1])
    update = drafts[0].model_copy(update={"action": "UPDATE", "target_scene_ids": [first.scene_id],
                                        "unassigned_member_uris": [], "draft": drafts[0].draft.model_copy(update={
                                            "description": "Launch follow-up complete",
                                        })})
    updated = applier.apply(batch.snapshot, list(batch.nodes), batch.embeddings, update)
    assert next(scene for scene in updated.scenes if scene.scene_id == first.scene_id).version == first.version + 1
    assert next(scene for scene in updated.scenes if scene.scene_id == second.scene_id) == second
    archived = replace(updated, revision=updated.revision + 1, scenes=tuple(
        replace(scene, status="ARCHIVED", version=scene.version + 1) if scene.scene_id == second.scene_id else scene
        for scene in updated.scenes
    ))
    restored = applier.apply(archived, list(batch.nodes), batch.embeddings, drafts[1])
    assert len(restored.scenes) == 2
    revived = next(scene for scene in restored.scenes if scene.scene_id == second.scene_id)
    assert (revived.status, revived.version) == ("ACTIVE", second.version + 2)


@pytest.mark.parametrize("nested_list", [False, True])
def test_nested_boundary_order_replays_same_scene(nested_list):
    snapshot, nodes, embeddings, decision, _ = application_case()
    original = {"name": "发布", "scope": {"region": "east", "phase": 1}}
    reordered = {"scope": {"phase": 1, "region": "east"}, "name": "发布"}
    if nested_list:
        original, reordered = [original], [reordered]
    proposal = decision.model_copy(update={"draft": decision.draft.model_copy(update={"boundary": {"thread": original}})})
    replay = proposal.model_copy(update={"draft": proposal.draft.model_copy(update={"boundary": {"thread": reordered}})})
    applier = SceneDecisionApplier()
    first = applier.apply(snapshot, nodes, embeddings, proposal)
    assert applier.apply(first, nodes, embeddings, replay) is first
    assert json.loads(dict(first.scenes[0].cluster.boundary)["thread"]) == original
    assert applier.apply(snapshot, nodes, embeddings, replay) == first


@pytest.mark.parametrize("via_tool", [False, True])
def test_supporting_subset_uses_reachable_provenance_context(via_tool):
    snapshot, nodes, embeddings, packet, initial, _ = subset_case("CREATE", False)
    root_b = ProvenanceResolver.build_id("archive", "delivery", ["m2"])
    if via_tool:
        packet["tool_evidence"] = [{"tool": "read_source_context", "results": [
            {"uri": nodes[0].uri, "source_refs": [root_b]},
        ]}]
    else:
        packet["memories"][0]["source_refs"].append(nodes[3].uri)
    model = JSONModel(initial.model_dump(), {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    accepted = stage.decide(snapshot, packet, budget)
    assert stage.deterministic_quality_gate(accepted, packet).decision == "ACCEPT"
    assert stage.review(accepted, packet, budget).decision == "ACCEPT"
    assert budget.used == 2
    assert SceneDecisionApplier().apply(snapshot, nodes, embeddings, accepted).scenes[0].version == 1
