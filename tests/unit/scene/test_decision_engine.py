"""Snapshot decision composition, checkpoints and bounded batch recovery."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import DecisionLimits, SceneCallBudgetError
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from tests.unit.scene.test_decision_application import application_case
from tests.unit.scene.test_decision_evidence import node, vectors
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import source_case


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_three_stage_decision_materializes_scene_and_preserves_inputs(action):
    snapshot, nodes, embeddings, proposal, _ = application_case(action)
    original = deepcopy((snapshot, nodes, embeddings, proposal))
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    result = SceneDecisionEngine(model).decide_with_result(snapshot, nodes, embeddings)

    assert result.action == action
    assert result.calls == 3
    assert result.pending_member_uris == (nodes[-1].uri,)
    assert result.snapshot.revision == snapshot.revision + 1
    scene = result.snapshot.scenes[0]
    assert scene.version == (1 if action == "CREATE" else 2)
    assert scene.cluster.label == proposal.draft.title
    assert scene.cluster.summary == proposal.draft.description
    assert [member.context_uri for member in scene.cluster.members] == proposal.draft.member_uris
    assert (snapshot, nodes, embeddings, proposal) == original
    assert [schema["title"] for _, schema in model.requests] == [
        "PreflightOutput", "SceneDecisionOutput", "QualityGateOutput",
    ]


def test_two_call_mode_materializes_with_deterministic_evidence_gate():
    snapshot, nodes, embeddings, proposal, _ = application_case()
    model = JSONModel({"status": "READY"}, proposal.model_dump())
    engine = SceneDecisionEngine(model, SceneConfig(max_llm_calls_per_batch=2))
    result = engine.decide_with_result(snapshot, nodes, embeddings)

    assert result.action == "CREATE"
    assert result.calls == len(model.requests) == 2
    assert result.snapshot.scenes[0].cluster.label == proposal.draft.title
    assert result.pending_member_uris == (nodes[-1].uri,)


@pytest.mark.parametrize("edited", [False, True])
def test_quality_adjustment_updates_title_and_preserves_effective_description(edited):
    snapshot, nodes, embeddings, proposal, _ = application_case("UPDATE")
    existing = snapshot.scenes[0]
    if edited:
        existing = replace(existing, description_edited=True,
                           cluster=replace(existing.cluster, summary="Reviewed by the owner"))
        snapshot = replace(snapshot, scenes=(existing,))
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {
        "decision": "ADJUST", "title": "Atlas launch", "description": "Verified launch milestones",
    })
    result = SceneDecisionEngine(model).decide_with_result(snapshot, nodes, embeddings)

    assert result.action == "UPDATE"
    assert result.calls == 3
    scene = result.snapshot.scenes[0]
    assert scene.scene_id == existing.scene_id
    assert scene.cluster.label == "Atlas launch"
    assert scene.cluster.summary == ("Reviewed by the owner" if edited else "Verified launch milestones")
    assert scene.description_edited == edited


@pytest.mark.parametrize("initial", ["empty", "small", "shared_source"])
def test_completed_evidence_resumes_the_same_engine(initial):
    snapshot, nodes, embeddings, proposal, _ = application_case()
    offered = [] if initial == "empty" else deepcopy(nodes[:2] if initial == "small" else nodes)
    first_outputs = []
    if initial == "shared_source":
        for item in offered:
            item.metadata["source_refs"] = ["planning"]
        first_outputs = [{"status": "READY"}]
    model = JSONModel(*first_outputs, {"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)
    waiting = engine.decide_with_result(snapshot, offered, vectors(offered))

    assert waiting.action == "WAIT"
    assert waiting.snapshot is snapshot
    assert waiting.calls == len(first_outputs)
    assert waiting.pending_member_uris == tuple(sorted(item.uri for item in offered))
    completed = engine.decide_with_result(snapshot, nodes, embeddings)
    assert completed.action == "CREATE"
    assert completed.calls == 3
    assert completed.snapshot.revision == snapshot.revision + 1
    assert completed.pending_member_uris == (nodes[-1].uri,)


@pytest.mark.parametrize("stage", ["preflight", "proposal", "quality", "evidence_floor"])
def test_wait_checkpoint_retains_ordered_evidence_then_completes(stage):
    snapshot, nodes, embeddings, proposal, _ = application_case()
    offered = [nodes[2].uri, nodes[0].uri, nodes[2].uri]
    reason = "Awaiting the release review"
    if stage == "preflight":
        first = [{"status": "WAIT", "evidence_uris": offered, "reason": reason}]
    elif stage == "proposal":
        first = [{"status": "READY"}, {
            "action": "WAIT", "unassigned_member_uris": offered, "reason": reason,
        }]
    elif stage == "quality":
        first = [{"status": "READY"}, proposal.model_dump(), {"decision": "DEFER", "reason": reason}]
    else:
        first = [{"status": "READY"}, {**proposal.model_dump(), "conflicts": [reason]}]
        reason = "proposal does not meet deterministic evidence floor"
    model = JSONModel(*first, {"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)
    waiting = engine.decide_with_result(snapshot, list(reversed(nodes)), embeddings)

    assert waiting.action == "WAIT"
    assert waiting.snapshot is snapshot
    assert waiting.calls == len(first)
    assert waiting.reason == reason
    assert waiting.pending_member_uris == tuple(sorted(item.uri for item in nodes))
    completed = engine.decide_with_result(snapshot, nodes, embeddings)
    assert completed.action == "CREATE"
    assert completed.calls == 3
    assert completed.pending_member_uris == (nodes[-1].uri,)


def test_replay_preserves_checkpoint_and_unique_unassigned_evidence():
    snapshot, nodes, _, proposal, _ = application_case()
    nodes.append(node(5))
    embeddings = vectors(nodes)
    payload = proposal.model_dump()
    payload["unassigned_member_uris"] = [nodes[5].uri, nodes[4].uri, nodes[5].uri]
    outputs = [{"status": "READY"}, payload, {"decision": "ACCEPT"}]
    model = JSONModel(*outputs, *deepcopy(outputs))
    engine = SceneDecisionEngine(model)
    created = engine.decide_with_result(snapshot, nodes, embeddings)
    replayed = engine.decide_with_result(created.snapshot, list(reversed(nodes)), embeddings)

    assert created.action == "CREATE"
    assert replayed.action == "WAIT"
    assert replayed.snapshot is created.snapshot
    assert created.calls == replayed.calls == 3
    assert created.pending_member_uris == replayed.pending_member_uris == (nodes[4].uri, nodes[5].uri)
    assert len(model.requests) == 6


@pytest.mark.parametrize("stage", ["preflight", "proposal"])
def test_corrected_batch_recovers_with_fresh_budget_and_quality_review(stage):
    snapshot, nodes, embeddings, proposal, _ = application_case()
    if stage == "preflight":
        first = [{"status": "READ"}, {"status": "READY"}, proposal.model_dump()]
        correction_index = 1
    else:
        first = [{"status": "READY"}, {**proposal.model_dump(), "unassigned_member_uris": []},
                 proposal.model_dump()]
        correction_index = 2
    model = JSONModel(*first, {"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)

    with pytest.raises(SceneCallBudgetError) as interrupted:
        engine.decide_with_result(snapshot, nodes, embeddings)
    diagnostic = json.loads(str(interrupted.value))
    assert diagnostic["stage"] == "quality"
    assert diagnostic["code"] == "scene_call_budget_exhausted"
    assert len(model.requests) == 3
    correction = model.requests[correction_index][0]
    assert "OUTPUT CORRECTION" in correction
    assert f'"stage":"{stage if stage == "preflight" else "decision"}"' in correction

    completed = engine.decide_with_result(snapshot, nodes, embeddings)
    assert completed.action == "CREATE"
    assert completed.calls == 3
    assert completed.snapshot.revision == snapshot.revision + 1
    assert completed.pending_member_uris == (nodes[-1].uri,)
    assert [schema["title"] for _, schema in model.requests[3:]] == [
        "PreflightOutput", "SceneDecisionOutput", "QualityGateOutput",
    ]


@pytest.mark.parametrize("action", ["CREATE", "UPDATE"])
@pytest.mark.parametrize("stage", ["preflight", "decision", "quality"])
def test_four_call_budget_publishes_corrected_and_reviewed_canonical_scene(action, stage):
    source, nodes, _, proposal = source_case(action)
    original = deepcopy((nodes, proposal))
    outputs = [{"status": "READY"}, proposal.model_dump(),
               {"decision": "ADJUST", "title": "Reviewed Atlas release", "description": "Verified scene evidence"}]
    position, initial = {
        "preflight": (0, {"status": "READ"}),
        "decision": (1, {**proposal.model_dump(), "unassigned_member_uris": []}),
        "quality": (2, {"decision": "ACCPET"}),
    }[stage]
    outputs.insert(position, initial)
    model = JSONModel(*outputs)
    engine = SceneDecisionEngine(model, SceneConfig(max_llm_calls_per_batch=4), evidence_reader=source)
    result = engine.refresh(source, "acct", "user:alice", node_uris=[node.uri for node in nodes])

    assert result.action == action
    assert result.calls == len(model.requests) == 4
    assert result.snapshot.scenes == source.repository.snapshot.scenes
    assert result.snapshot.revision == source.repository.snapshot.revision == 1
    assert result.snapshot.scenes[0].cluster.label == "Reviewed Atlas release"
    assert result.snapshot.scenes[0].cluster.summary == "Verified scene evidence"
    assert result.snapshot.scenes[0].version == (1 if action == "CREATE" else 2)
    assert result.pending_member_uris == (nodes[-1].uri,)
    assert model.requests[-1][1]["title"] == "QualityGateOutput"
    assert "OUTPUT CORRECTION" in model.requests[position + 1][0]
    assert (nodes, proposal) == original


def test_configured_engines_share_model_and_limits_and_reset_each_batch_budget():
    snapshot, nodes, embeddings, proposal, _ = application_case()
    ready, accepted = {"status": "READY"}, {"decision": "ACCEPT"}
    model = JSONModel(ready, proposal.model_dump(), ready, proposal.model_dump(), accepted,
                      ready, proposal.model_dump())
    limits = DecisionLimits(max_node_chars=32)
    original_config = SceneConfig()
    engine = SceneDecisionEngine(model, original_config, limits=limits)
    configuration = replace(original_config, max_llm_calls_per_batch=2, max_members_per_scene=4)
    configured = engine.for_config(configuration)

    assert configured.llm is engine.llm is model
    assert configured.limits is engine.limits is limits
    assert engine.config is original_config
    assert configured.config is configuration
    first = configured.decide_with_result(snapshot, nodes, embeddings)
    second = engine.decide_with_result(snapshot, nodes, embeddings)
    third = configured.decide(snapshot, nodes, embeddings)
    assert (first.calls, second.calls, len(model.requests)) == (2, 3, 7)
    assert first.snapshot == second.snapshot == third
    for index, member_limit in ((1, 4), (3, 64), (6, 4)):
        packet = json.loads(model.requests[index][0].removeprefix(DECISION_PROMPT))
        assert packet["limits"]["max_members"] == member_limit
        assert all(len(card["text"]) == 32 for card in packet["memories"])


def test_quality_review_receives_parent_provenance_from_the_canonical_window():
    snapshot, nodes, embeddings, proposal, _ = application_case()
    parent = nodes[3]
    parent.metadata["source_refs"] = [
        ProvenanceResolver.build_id("archive", name) for name in ("planning", "review")
    ]
    for item in nodes[:3]:
        item.metadata["source_refs"] = [parent.uri]
    payload = proposal.model_dump()
    payload["draft"]["member_uris"] = [item.uri for item in nodes[:3]]
    payload["evidence_uris"] = list(payload["draft"]["member_uris"])
    payload["unassigned_member_uris"] = [item.uri for item in nodes[3:]]
    before = deepcopy(nodes)
    model = JSONModel({"status": "READY"}, payload, {"decision": "ACCEPT"})
    result = SceneDecisionEngine(model).decide_with_result(snapshot, nodes, embeddings)

    assert result.action == "CREATE"
    assert result.calls == 3
    evidence = json.loads(model.requests[2][0].removeprefix(QUALITY_PROMPT))["evidence"]
    assert [card["uri"] for card in evidence["memories"]] == payload["draft"]["member_uris"]
    assert [card["uri"] for card in evidence["provenance_context"]] == [parent.uri]
    assert evidence["provenance_context"][0]["source_refs"] == parent.metadata["source_refs"]
    assert result.pending_member_uris == tuple(item.uri for item in nodes[3:])
    assert nodes == before
