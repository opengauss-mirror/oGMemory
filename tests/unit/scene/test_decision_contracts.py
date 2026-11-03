"""Executable contracts for the preflight, decision and quality stages."""

import json
from dataclasses import asdict, replace

import pytest

from scene.decision_contracts import (
    DecisionLimits,
    PreflightOutput,
    QualityGateOutput,
    SceneCallBudgetError,
    SceneDecisionError,
    SceneDecisionOutput,
    SceneDecisionResult,
    SceneDraft,
    SceneModelOutputError,
    SceneRetryableDecisionError,
)
from scene.decision_prompts import (
    DECISION_PROMPT,
    PREFLIGHT_SYSTEM,
    QUALITY_PROMPT,
    SCENE_BOUNDARY_RULES,
)
from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot

URIS = [f"ctx://acme/users/alice/memories/event/{i}" for i in range(3)]


def draft_payload():
    return {
        "title": " Atlas release planning ",
        "description": " The team planned milestones for the Atlas release. ",
        "member_uris": URIS,
        "boundary": {"subject": "Atlas", "activity": "release planning", "time_scope": "2026-10"},
        "time_start": None,
        "time_end": None,
    }


@pytest.mark.parametrize("status", ["READY", "WAIT"])
def test_preflight_round_trip_preserves_evidence_and_recovery_state(status):
    payload = {
        "status": status,
        "anchors": ["Atlas", "release planning"],
        "evidence_uris": URIS,
        "candidate_scene_ids": ["release"],
        "missing": ["release date"] if status == "WAIT" else [],
        "reason": "Await dated source" if status == "WAIT" else "Three grounded sources",
    }
    output = PreflightOutput.model_validate_json(json.dumps(payload))
    assert output.model_dump() == payload
    assert PreflightOutput.model_validate_json(output.model_dump_json()) == output


@pytest.mark.parametrize("action,targets", [
    ("CREATE", []), ("UPDATE", ["release"]), ("MERGE", ["planning", "milestones"]),
])
def test_decision_round_trip_preserves_identity_membership_and_unassigned_evidence(action, targets):
    output = SceneDecisionOutput.model_validate({
        "action": action,
        "target_scene_ids": targets,
        "draft": draft_payload(),
        "evidence_uris": URIS[:2],
        "unassigned_member_uris": ["ctx://acme/users/alice/memories/event/outage"],
        "conflicts": [],
        "reason": "The sources track one release planning activity.",
    })
    restored = SceneDecisionOutput.model_validate_json(output.model_dump_json())
    assert restored == output
    assert restored.action == action
    assert restored.target_scene_ids == targets
    assert restored.draft.member_uris == URIS
    assert restored.draft.title == "Atlas release planning"
    assert restored.draft.description == "The team planned milestones for the Atlas release."
    assert restored.evidence_uris == URIS[:2]
    assert restored.unassigned_member_uris == ["ctx://acme/users/alice/memories/event/outage"]
    assert restored.conflicts == []


def test_wait_decision_retains_pending_evidence_for_later_analysis():
    output = SceneDecisionOutput.model_validate_json(json.dumps({
        "action": "WAIT", "unassigned_member_uris": URIS,
        "reason": "Await an independent source for the activity.",
    }))
    assert output.model_dump() == {
        "action": "WAIT", "target_scene_ids": [], "draft": None, "evidence_uris": [],
        "unassigned_member_uris": URIS, "conflicts": [],
        "reason": "Await an independent source for the activity.",
    }


@pytest.mark.parametrize("start,end", [(None, None), ("2026-10-21", "2026-10-22")])
def test_draft_preserves_supported_date_precision(start, end):
    output = SceneDraft.model_validate({**draft_payload(), "time_start": start, "time_end": end})
    restored = SceneDraft.model_validate_json(output.model_dump_json())
    assert (restored.time_start, restored.time_end) == (start, end)
    assert restored.boundary["time_scope"] == "2026-10"
    schema = SceneDraft.model_json_schema()
    assert "use null" in schema["properties"]["time_start"]["description"]


@pytest.mark.parametrize("decision", ["ACCEPT", "ADJUST", "DEFER"])
@pytest.mark.parametrize("reason", ["Supported", "证据支持同一活动。" * 160])
def test_quality_gate_preserves_decision_with_bounded_diagnostics(decision, reason):
    output = QualityGateOutput.model_validate_json(json.dumps({
        "decision": decision, "title": "Atlas release planning",
        "description": "The team planned release milestones.", "reason": f"  {reason}  ",
    }))
    assert output.model_dump() == {
        "decision": decision, "title": "Atlas release planning",
        "description": "The team planned release milestones.", "reason": reason[:1000],
    }
    assert QualityGateOutput.model_validate_json(output.model_dump_json()) == output


@pytest.mark.parametrize("model,payload,field,value", [
    (PreflightOutput, {"status": "READY"}, "anchors", "release"),
    (SceneDecisionOutput, {"action": "WAIT"}, "unassigned_member_uris", URIS[0]),
    (SceneDraft, draft_payload(), "boundary", ("goal", "release")),
])
def test_model_defaults_and_payloads_are_owned_by_each_decision(model, payload, field, value):
    first, second = model.model_validate(payload), model.model_validate(payload)
    container = getattr(first, field)
    if isinstance(container, dict):
        container[value[0]] = value[1]
    else:
        container.append(value)
    assert second == model.model_validate(payload)
    assert first != second


@pytest.mark.parametrize("size", [1, 128, 256])
def test_decision_limits_accept_configured_positive_bounds(size):
    limits = DecisionLimits(size, size * 500, size * 4, size * 20)
    limits.validate()
    assert asdict(limits) == {
        "max_input_nodes": size, "max_prompt_chars": size * 500,
        "max_node_chars": size * 4, "max_description_chars": size * 20,
    }


def test_result_preserves_snapshot_revision_and_pending_work():
    member = SceneMember(URIS[0], 1.0, node_version=7, source_refs=("archive-1", "archive-2"))
    cluster = SceneCluster(
        "user:alice", (member,), (1.0, 0.0), 1.0, "Atlas release",
        "Planning milestones", "source-v7", boundary=(("activity", "release planning"),),
    )
    snapshot = SceneSnapshot("acme", "user:alice", 3, (SceneRecord("release", cluster),))
    result = SceneDecisionResult(snapshot, "CREATE", (URIS[2],), 3, "Reviewed")
    updated = replace(result, pending_member_uris=(URIS[1], URIS[2]))
    assert updated.snapshot is snapshot
    assert updated.snapshot.scenes[0].cluster.members == (member,)
    assert (updated.action, updated.calls, updated.reason) == ("CREATE", 3, "Reviewed")
    assert result.pending_member_uris == (URIS[2],)
    assert updated.pending_member_uris == (URIS[1], URIS[2])


@pytest.mark.parametrize("stage", ["preflight", "decision", "quality"])
def test_retry_diagnostics_carry_stage_and_field_codes(stage):
    error = SceneModelOutputError(stage, [{"field": "draft.title", "type": "string_too_short"}])
    assert isinstance(error, SceneRetryableDecisionError)
    assert isinstance(error, SceneDecisionError)
    assert error.stage == stage
    assert error.errors == ({"field": "draft.title", "type": "string_too_short"},)
    assert json.loads(str(error)) == {
        "code": "scene_model_output_invalid", "stage": stage,
        "errors": [{"field": "draft.title", "type": "string_too_short"}],
    }
    budget = SceneCallBudgetError(stage)
    assert json.loads(str(budget)) == {
        "code": "scene_call_budget_exhausted", "stage": stage,
        "errors": [{"field": "max_llm_calls_per_batch", "type": "budget_exhausted"}],
    }


@pytest.mark.parametrize("prompt", [PREFLIGHT_SYSTEM, DECISION_PROMPT, QUALITY_PROMPT])
def test_three_stages_share_event_boundary_and_passage_evidence_rules(prompt):
    assert prompt.count(SCENE_BOUNDARY_RULES) == 1
    assert "an event/goal connection establishes membership" in prompt
    assert "identify the passage that supports the chosen" in prompt
    assert "Every descriptive claim" in prompt


def test_stage_prompts_frame_packet_decisions_and_quality_adjustments():
    packet = {"offered_member_uris": URIS, "memories": [], "existing_scenes": []}
    decision_prompt = DECISION_PROMPT + json.dumps(packet)
    quality_prompt = QUALITY_PROMPT + json.dumps({"decision": {"action": "WAIT"}, "evidence": packet})
    assert json.loads(decision_prompt.split("\nEVIDENCE PACKET:\n", 1)[1]) == packet
    assert json.loads(quality_prompt.split("\nPROPOSAL:\n", 1)[1])["evidence"] == packet
    assert "concrete continuity or identity evidence" in decision_prompt
    assert "membership and identity are sound" in quality_prompt
