"""Selected evidence recovers to the configured scene publication floor."""

from copy import deepcopy
from dataclasses import replace

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import PreflightOutput, SceneDecisionOutput
from scene.decision_preflight import ScenePreflight
from scene.decision_proposal import SceneProposalStage
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_decision_proposal import case


def select(proposal, packet, count):
    result = deepcopy(proposal)
    members = [card["uri"] for card in packet["memories"][:count]]
    result["draft"]["member_uris"] = members
    result["evidence_uris"] = members
    result["unassigned_member_uris"] = [uri for uri in packet["offered_member_uris"] if uri not in members]
    return result


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
@pytest.mark.parametrize("shortfall", ["nodes", "sources"])
def test_selected_evidence_correction_recovers_to_quality_review(action, shortfall):
    snapshot, packet, proposal = case(action)
    config = replace(SceneConfig(), min_evidence_nodes=5) if shortfall == "nodes" else SceneConfig()
    for index, card in enumerate(packet["memories"]):
        source = "planning" if index < 4 else "review"
        card["source_refs"] = [ProvenanceResolver.build_id("archive", source, [f"m{index}"])]
    initial, corrected = select(proposal, packet, 4), select(proposal, packet, 5)
    if shortfall == "nodes":
        packet["memories"][3]["source_refs"] = [ProvenanceResolver.build_id("archive", "review")]
    before = deepcopy(packet)
    model = JSONModel(initial, corrected, {"decision": "ACCEPT"})
    caller = SceneModelCaller(model)
    assert ScenePreflight(caller, config).enforce_gate(packet, PreflightOutput(status="READY")).status == "READY"
    stage = SceneProposalStage(caller, config)
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**corrected)
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 3
    assert packet == before
    diagnostic = "minimum_evidence_nodes_required" if shortfall == "nodes" else "independent_sources_required"
    assert diagnostic in model.requests[1][0]


@pytest.mark.parametrize("count", [2, 3])
def test_default_evidence_floor_recovers_to_wait_with_preflight_budget(count):
    snapshot, packet, proposal = case()
    for index, card in enumerate(packet["memories"]):
        source = "planning" if index < 3 else "review"
        card["source_refs"] = [ProvenanceResolver.build_id("archive", source, [f"m{index}"])]
    model = JSONModel({"status": "READY"}, select(proposal, packet, count),
                      {"action": "WAIT", "reason": "Await independent evidence"})
    caller = SceneModelCaller(model)
    budget = CallBudget(3)
    preflight = ScenePreflight(caller)
    assert preflight.enforce_gate(packet, preflight.run(packet, budget)).status == "READY"
    result = SceneProposalStage(caller).decide(snapshot, packet, budget)
    assert result.action == "WAIT"
    assert result.reason == "Await independent evidence"
    assert budget.used == 3


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_selected_sources_follow_provenance_then_recover_with_independent_member(depth, reverse):
    snapshot, packet, proposal = case()
    archive = "ctx://acct/sessions/planning/history/source"
    refs = [archive, ProvenanceResolver.build_id("archive", "source", ["m1"]),
            ProvenanceResolver.build_id("archive", archive, ["m2"])]
    context = []
    for index in range(depth):
        uri = f"ctx://acct/users/alice/memories/event/origin-{index}"
        context.append({"uri": uri, "source_refs": refs})
        refs = [uri, ProvenanceResolver.build_id("memory", uri), uri]
    for index, card in enumerate(packet["memories"][:3]):
        card["source_refs"] = [refs[index]]
    packet["memories"][3]["source_refs"] = [ProvenanceResolver.build_id("archive", "review")]
    context.append({"uri": "ctx://acct/sessions/other/history/unrelated", "source_refs": ["unrelated-source"]})
    packet["tool_evidence"] = [{"results": list(reversed(context)) if reverse else context}]
    initial, corrected = select(proposal, packet, 3), select(proposal, packet, 4)
    model = JSONModel(initial, corrected, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**corrected)
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 3


def test_custom_source_floor_uses_selected_members_after_correction():
    snapshot, packet, proposal = case()
    for index, card in enumerate(packet["memories"]):
        card["source_refs"] = [ProvenanceResolver.build_id("archive", f"source-{index // 2}")]
    initial, corrected = select(proposal, packet, 4), select(proposal, packet, 5)
    model = JSONModel(initial, corrected, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model), replace(SceneConfig(), min_independent_sources=3))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**corrected)
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 3


def test_configured_two_member_floor_accepts_supported_scene():
    snapshot, packet, proposal = case()
    proposal = select(proposal, packet, 2)
    model = JSONModel(proposal, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model), replace(SceneConfig(), min_evidence_nodes=2))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**proposal)
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 2


@pytest.mark.parametrize("location", ["memories", "tool_evidence"])
def test_shared_parent_resolves_to_two_scoped_original_sources(location):
    snapshot, packet, proposal = case()
    proposal = select(proposal, packet, 3)
    parent = packet["memories"][4]
    parent["source_refs"] = [f"ctx://acct/sessions/{session}/history/source" for session in ("planning", "review")]
    for card in packet["memories"][:3]:
        card["source_refs"] = [ProvenanceResolver.build_id("memory", parent["uri"])]
    if location == "tool_evidence":
        packet["tool_evidence"] = [{"results": [deepcopy(parent)]}]
        parent["source_refs"] = []
    model = JSONModel(proposal, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**proposal)
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 2
