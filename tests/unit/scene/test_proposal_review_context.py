"""Quality requests retain the evidence used to validate selected members."""

import json
from copy import deepcopy

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_prompts import QUALITY_PROMPT
from scene.decision_proposal import SceneProposalStage
from scene.decision_sources import supporting_sources
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_decision_proposal import case
from tests.unit.scene.test_proposal_evidence_floor import select


class EvidenceReviewModel:
    def __init__(self, proposal):
        self.proposal = proposal
        self.evidence = None

    def complete_json(self, prompt, schema):
        if schema["title"] == "SceneDecisionOutput":
            return self.proposal
        self.evidence = json.loads(prompt.removeprefix(QUALITY_PROMPT))["evidence"]
        context = [*self.evidence["memories"], *self.evidence.get("provenance_context", []),
                   *(card for result in self.evidence.get("tool_evidence", []) for card in result["results"])]
        sources = supporting_sources(self.evidence["memories"], context)
        return {"decision": "ACCEPT" if len(sources) >= 2 else "DEFER"}


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("reference", ["uri", "provenance"])
@pytest.mark.parametrize("location", ["memories", "mixed"])
def test_quality_request_retains_transitive_parent_cards(depth, reference, location):
    snapshot, packet, proposal = case()
    selected = packet["memories"][:3]
    parents = packet["memories"][3:3 + depth]
    for index, card in enumerate(parents):
        card["text"] = f"Original release evidence at hop {index}"
        card["source_refs"] = ([parents[index + 1]["uri"]] if index + 1 < depth else
                               [ProvenanceResolver.build_id("archive", name) for name in ("planning", "review")])
    ref = parents[0]["uri"]
    for card in selected:
        card["source_refs"] = [ref if reference == "uri" else ProvenanceResolver.build_id("memory", ref)]
    canonical_parents = deepcopy(parents)
    if location == "mixed" and depth == 2:
        packet["tool_evidence"] = [{"results": [deepcopy(parents[-1])]}]
        parents[-1]["source_refs"] = []
        canonical_parents = deepcopy(parents)
    proposal = select(proposal, packet, 3)
    original = deepcopy(packet)
    model = EvidenceReviewModel(proposal)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    review = stage.review(decision, packet, budget)
    assert review.decision == "ACCEPT"
    assert model.evidence["memories"] == selected
    assert model.evidence["provenance_context"] == canonical_parents
    assert decision.draft.member_uris == proposal["draft"]["member_uris"]
    assert packet == original
    assert budget.used == 2


@pytest.mark.parametrize("location", ["memories", "tool_evidence"])
def test_quality_resolves_shared_cyclic_parents_to_original_sources(location):
    snapshot, packet, proposal = case()
    parents = deepcopy(packet["memories"][3:])
    for index, parent in enumerate(parents):
        parent["source_refs"] = [parents[1 - index]["uri"],
                                 ProvenanceResolver.build_id("archive", f"source-{index}")]
    for card in packet["memories"][:3]:
        card["source_refs"] = [parents[0]["uri"]]
    if location == "memories":
        packet["memories"][3:] = parents
    else:
        packet["memories"] = packet["memories"][:3]
        packet["offered_member_uris"] = [card["uri"] for card in packet["memories"]]
        packet["tool_evidence"] = [{"results": parents}]
    proposal = select(proposal, packet, 3)
    original = deepcopy(packet)
    model = EvidenceReviewModel(proposal)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    assert stage.review(decision, packet, budget).decision == "ACCEPT"
    assert model.evidence["provenance_context"] == (parents if location == "memories" else [])
    assert model.evidence.get("tool_evidence", []) == packet.get("tool_evidence", [])
    assert packet == original
    assert budget.used == 2


def test_quality_correction_retains_provenance_context_with_shared_budget():
    snapshot, packet, proposal = case()
    parent = packet["memories"][-1]
    parent["source_refs"] = [ProvenanceResolver.build_id("archive", name) for name in ("planning", "review")]
    for card in packet["memories"][:3]:
        card["source_refs"] = [parent["uri"]]
    proposal = select(proposal, packet, 3)
    model = JSONModel(proposal, {"decision": "ACCPET"}, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    assert stage.review(decision, packet, budget).decision == "ACCEPT"
    for prompt, _ in model.requests[1:]:
        payload = json.loads(prompt.removeprefix(QUALITY_PROMPT).split("\nOUTPUT CORRECTION:")[0])
        assert payload["evidence"]["provenance_context"] == [parent]
        assert payload["evidence"]["memories"] == packet["memories"][:3]
    assert budget.used == 3
