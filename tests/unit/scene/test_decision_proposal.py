"""Proposal correction and quality review through the shared model caller."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scene.config import SceneConfig
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import SceneDecisionOutput
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from scene.decision_proposal import SceneProposalStage
from scene.models import SceneSnapshot
from tests.unit.scene.test_decision_evidence import node, record, vectors
from tests.unit.scene.test_decision_preflight import JSONModel


def case(action="CREATE"):
    nodes = [node(index) for index in range(5)]
    scenes = () if action == "CREATE" else (record("first", nodes[:2]),)
    if action == "MERGE":
        scenes += (record("second", nodes[2:4]),)
    snapshot = SceneSnapshot("acct", "user:alice", 2, scenes)
    packet = SceneEvidenceBuilder().build(snapshot, nodes, vectors(nodes))
    members = [item.uri for item in nodes[:4]]
    proposal = {"action": action, "target_scene_ids": [item.scene_id for item in scenes],
                "draft": {"title": "Atlas release", "description": "Release progress and results",
                          "member_uris": members, "time_start": "2026-10-20", "time_end": "2026-10-22"},
                "evidence_uris": members, "unassigned_member_uris": [nodes[-1].uri]}
    return snapshot, packet, proposal


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_proposal_and_review_share_budget_and_keep_source_packet(action):
    snapshot, packet, proposal = case(action)
    before = deepcopy(packet)
    model = JSONModel(proposal, {"decision": "ACCEPT", "reason": "Evidence supports this thread"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    review = stage.review(result, packet, budget)
    assert result.action == action
    assert review.decision == "ACCEPT"
    assert budget.used == 2
    assert packet == before
    assert json.loads(model.requests[0][0].removeprefix(DECISION_PROMPT)) == packet
    payload = json.loads(model.requests[1][0].removeprefix(QUALITY_PROMPT))
    assert payload["decision"] == result.model_dump()
    assert [card["uri"] for card in payload["evidence"]["memories"]] == proposal["draft"]["member_uris"]


@pytest.mark.parametrize("field,value", [
    ("target_scene_ids", ["unoffered"]), ("target_scene_ids", ["first", "first"]),
    ("evidence_uris", ["unknown"]), ("unassigned_member_uris", []),
    ("unassigned_member_uris", ["unknown"]), ("draft", None),
])
def test_corrected_proposal_restores_complete_assignments(field, value):
    snapshot, packet, valid = case("UPDATE")
    initial = {**valid, field: value}
    model = JSONModel(initial, valid, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result == SceneDecisionOutput(**valid)
    assert "OUTPUT CORRECTION" in model.requests[1][0]
    assert stage.review(result, packet, budget).decision == "ACCEPT"
    assert budget.used == 3


@pytest.mark.parametrize("field,value", [
    ("member_uris", ["unknown", "other"]),
    ("member_uris", ["ctx://acct/users/alice/memories/event/0"] * 3),
    ("member_uris", ["ctx://acct/users/alice/memories/event/2", "ctx://acct/users/alice/memories/event/3"]),
    ("time_start", "next Thursday"), ("time_end", "2026-10-19"),
])
def test_corrected_draft_restores_members_and_chronology(field, value):
    snapshot, packet, valid = case("UPDATE")
    initial = deepcopy(valid)
    initial["draft"][field] = value
    model = JSONModel(initial, valid)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    assert stage.decide(snapshot, packet, budget) == SceneDecisionOutput(**valid)
    assert budget.used == 2
    assert "OUTPUT CORRECTION" in model.requests[1][0]


@pytest.mark.parametrize("start,end", [("2026-10-20", "2026-10-20T08:00:00+08:00"),
                                       ("2026-10-20T09:00:00+08:00", "2026-10-20T02:00:00+00:00")])
def test_equivalent_timezone_dates_accept_chronological_proposal(start, end):
    snapshot, packet, proposal = case()
    proposal["draft"].update(time_start=start, time_end=end)
    stage = SceneProposalStage(SceneModelCaller(JSONModel(proposal)))
    assert stage.decide(snapshot, packet, CallBudget(3)).draft.time_end == end


@pytest.mark.parametrize("review", [{"decision": "ACCEPT"}, {"decision": "DEFER", "reason": "Need evidence"},
                                  {"decision": "ADJUST", "title": "Atlas launch", "description": "Confirmed launch progress"}])
def test_quality_outcomes_are_returned_to_orchestrator(review):
    snapshot, packet, proposal = case()
    model = JSONModel(proposal, review)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    before = decision.model_dump()
    result = stage.review(decision, packet, budget)
    assert result.decision == review["decision"]
    assert result.title == review.get("title")
    assert decision.model_dump() == before


def test_quality_schema_correction_uses_remaining_shared_budget():
    snapshot, packet, proposal = case()
    model = JSONModel(proposal, {"decision": "ACCPET"}, {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    assert stage.review(decision, packet, budget).decision == "ACCEPT"
    assert budget.used == 3


def test_wait_proposal_preserves_reason_and_budget():
    snapshot, packet, _ = case()
    model = JSONModel({"action": "WAIT", "reason": "Awaiting launch decision"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    proposal = stage.decide(snapshot, packet, budget)
    assert proposal.reason == "Awaiting launch decision"
    assert stage.review(proposal, packet, budget).decision == "DEFER"
    assert budget.used == 1


def test_member_capacity_correction_retains_unassigned_evidence():
    snapshot, packet, valid = case()
    initial = deepcopy(valid)
    initial["draft"]["member_uris"] += initial["unassigned_member_uris"]
    initial["unassigned_member_uris"] = []
    model = JSONModel(initial, valid)
    stage = SceneProposalStage(SceneModelCaller(model), replace(SceneConfig(), max_members_per_scene=4))
    budget = CallBudget(3)
    assert stage.decide(snapshot, packet, budget) == SceneDecisionOutput(**valid)
    assert budget.used == 2


def test_merge_target_correction_uses_two_distinct_scenes():
    snapshot, packet, valid = case("MERGE")
    model = JSONModel({**valid, "target_scene_ids": ["first", "first"]}, valid)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    result = stage.decide(snapshot, packet, budget)
    assert result.target_scene_ids == ["first", "second"]
    assert budget.used == 2
    assert "unique_targets_required" in model.requests[1][0]


@pytest.mark.parametrize("change", [{"status": "ARCHIVED"}, {"origin": "manual"}])
def test_stale_candidate_recovers_to_wait_then_current_snapshot_update(change):
    snapshot, packet, valid = case("UPDATE")
    stale_snapshot = replace(snapshot, scenes=(replace(snapshot.scenes[0], **change),))
    model = JSONModel(valid, {"action": "WAIT", "reason": "Refresh candidates"}, valid)
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    assert stage.decide(stale_snapshot, packet, budget).reason == "Refresh candidates"
    assert stage.decide(snapshot, packet, budget).action == "UPDATE"
    assert budget.used == 3
