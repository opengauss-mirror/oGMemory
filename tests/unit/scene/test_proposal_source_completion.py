"""Completed canonical sources and WAIT outcomes preserve stage contracts."""

from copy import deepcopy

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.decision_application import SceneDecisionApplier
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import PreflightOutput
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_preflight import ScenePreflight
from scene.decision_proposal import SceneProposalStage
from scene.decision_sources import supporting_sources
from tests.unit.scene.test_adapters import adapter
from tests.unit.scene.test_decision_evidence import node, vectors
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_decision_proposal import case
from tests.unit.scene.test_proposal_review_context import EvidenceReviewModel


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
@pytest.mark.parametrize("source_kind", ["archive", "parent_memory"])
def test_tool_completed_member_sources_reach_quality_review(action, source_kind):
    snapshot, _, proposal = case(action)
    nodes = [node(index) for index in range(5)]
    planning = [ProvenanceResolver.build_id("archive", "planning", [f"m{index}"]) for index in range(4)]
    review_source = ProvenanceResolver.build_id("archive", "review")
    for item in nodes:
        item.metadata["source_refs"] = list(planning)
    extra = review_source
    if source_kind == "parent_memory":
        nodes[-1].metadata["source_refs"] = [review_source]
        extra = ProvenanceResolver.build_id("memory", nodes[-1].uri)
    nodes[0].metadata["source_refs"].append(extra)
    packet = SceneEvidenceBuilder().build(snapshot, nodes, vectors(nodes))
    source = adapter(nodes)
    cards = source.read_source_context(snapshot.account_id, snapshot.owner_space, [nodes[0].uri], 1)
    assert packet["memories"][0]["source_refs"] == planning
    assert cards[0]["source_refs"] == [*planning, extra]
    assert source.fs.reads == [nodes[0].uri]
    packet["tool_evidence"] = [{"tool": "read_source_context", "results": cards}]
    before = deepcopy(packet)
    model = EvidenceReviewModel(proposal)
    caller = SceneModelCaller(model)
    assert ScenePreflight(caller).enforce_gate(packet, PreflightOutput(status="READY")).status == "READY"
    stage = SceneProposalStage(caller)
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"
    assert stage.review(decision, packet, budget).decision == "ACCEPT"
    assert decision.draft.member_uris == proposal["draft"]["member_uris"]
    assert model.evidence["tool_evidence"] == packet["tool_evidence"]
    selected = model.evidence["memories"]
    context = [*selected, *model.evidence["provenance_context"], *cards]
    assert supporting_sources(selected, context) == {("archive", "planning"), ("archive", "review")}
    assert packet == before
    assert budget.used == 2
    result = SceneDecisionApplier().apply(snapshot, nodes, vectors(nodes), decision)
    member = next(member for member in result.scenes[0].cluster.members if member.context_uri == nodes[0].uri)
    assert member.source_refs == (*planning, extra)


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_wait_after_preflight_or_correction_preserves_budget(with_draft, correction):
    snapshot, packet, proposal = case()
    wait = {"action": "WAIT", "reason": "Await a confirmed launch date"}
    if with_draft:
        wait["draft"] = deepcopy(proposal["draft"])
    outputs = [{"status": "READY"}]
    if correction:
        outputs.append({**proposal, "target_scene_ids": ["refresh-candidate"]})
    model = JSONModel(*outputs, wait, {"decision": "ACCEPT"})
    caller = SceneModelCaller(model)
    budget = CallBudget(3)
    preflight = ScenePreflight(caller)
    assert preflight.enforce_gate(packet, preflight.run(packet, budget)).status == "READY"
    stage = SceneProposalStage(caller)
    decision = stage.decide(snapshot, packet, budget)
    assert decision.action == "WAIT"
    assert decision.reason == wait["reason"]
    before = decision.model_dump()
    assert stage.review(decision, packet, budget).decision == "DEFER"
    assert decision.model_dump() == before
    assert budget.used == len(model.requests) == 2 + int(correction)
