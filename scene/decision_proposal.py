"""Scoped scene proposals and bounded model quality review."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from scene.config import SceneConfig
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import QualityGateOutput, SceneDecisionOutput, SceneModelOutputError
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from scene.decision_sources import independent_sources, resolve_sources, supporting_sources
from scene.models import SceneSnapshot


class SceneProposalStage:
    """Validate model proposals against a canonical evidence packet before review."""

    def __init__(self, caller: SceneModelCaller, config: SceneConfig | None = None) -> None:
        self.caller = caller
        self.config = config or SceneConfig()
        self.config.validate()

    def decide(self, snapshot: SceneSnapshot, packet: dict[str, Any], budget: CallBudget) -> SceneDecisionOutput:
        return self.caller.call(
            "decision", DECISION_PROMPT + self.caller.dump(packet), SceneDecisionOutput, budget,
            validator=lambda proposal: self.validate(proposal, snapshot, packet),
        )

    def review(self, decision: SceneDecisionOutput, packet: dict[str, Any], budget: CallBudget) -> QualityGateOutput:
        if decision.action == "WAIT":
            return QualityGateOutput(decision="DEFER", reason=decision.reason or "awaiting scene evidence")
        if decision.draft is None:
            return QualityGateOutput(decision="DEFER", reason="missing scene draft")
        members = set(decision.draft.member_uris)
        selected = [card for card in packet["memories"] if card["uri"] in members]
        _, reached = resolve_sources(selected, self._evidence_cards(packet))
        provenance = [card for card in packet["memories"] if card["uri"] not in members
                      and independent_sources([card["uri"]]) & reached]
        evidence = {**packet, "memories": selected, "provenance_context": provenance}
        prompt = QUALITY_PROMPT + self.caller.dump({"decision": decision.model_dump(), "evidence": evidence})
        return self.caller.call("quality", prompt, QualityGateOutput, budget)

    def deterministic_quality_gate(
        self, decision: SceneDecisionOutput, packet: dict[str, Any],
    ) -> QualityGateOutput:
        """Enforce the selected evidence floor before applying a reviewed proposal."""
        if decision.action == "WAIT":
            return QualityGateOutput(decision="DEFER", reason=decision.reason or "awaiting scene evidence")
        if decision.draft is None:
            return QualityGateOutput(decision="DEFER", reason="missing scene draft")
        members = set(decision.draft.member_uris)
        evidence = set(decision.evidence_uris)
        selected = [card for card in packet["memories"] if card["uri"] in members]
        supporting = [card for card in selected if card["uri"] in evidence]
        if (decision.conflicts or len({card["uri"] for card in selected}) != len(members)
                or len(members) < self.config.min_evidence_nodes
                or not evidence <= members or len(evidence) < self.config.min_evidence_nodes
                or len(supporting_sources(supporting, self._evidence_cards(packet))) < self.config.min_independent_sources):
            return QualityGateOutput(decision="DEFER", reason="proposal does not meet deterministic evidence floor")
        return QualityGateOutput(decision="ACCEPT", reason="deterministic evidence gate passed")

    @staticmethod
    def _evidence_cards(packet: dict[str, Any]) -> list[dict[str, Any]]:
        return [*packet["memories"], *(card for result in packet.get("tool_evidence", [])
                                     for card in result.get("results", []) if isinstance(card, dict))]

    def validate(self, decision: SceneDecisionOutput, snapshot: SceneSnapshot, packet: dict[str, Any]) -> None:
        """Check references and complete assignment; semantic review is a separate stage."""
        if decision.action == "WAIT":
            return
        errors = []

        def require(condition: bool, field: str, error_type: str) -> None:
            if not condition:
                errors.append({"field": field, "type": error_type})

        existing = {record["scene_id"] for record in packet["existing_scenes"]}
        eligible = {record.scene_id for record in snapshot.scenes
                    if record.status == "ACTIVE" and record.origin == "automatic"}
        offered = set(packet["offered_member_uris"])
        all_uris = {card["uri"] for card in packet["memories"]}
        targets = decision.target_scene_ids
        require(len(set(targets)) == len(targets), "target_scene_ids", "unique_targets_required")
        if decision.action == "CREATE":
            require(not targets, "target_scene_ids", "empty_for_create")
        elif decision.action == "UPDATE":
            require(len(targets) == 1, "target_scene_ids", "one_target_for_update")
        else:
            require(len(targets) >= 2, "target_scene_ids", "multiple_targets_for_merge")
        require(set(targets) <= existing & eligible, "target_scene_ids", "candidate_scene_reference_required")
        if decision.draft is None:
            errors.append({"field": "draft", "type": "draft_required"})
            raise SceneModelOutputError("decision", errors)
        times = {}
        for field in ("time_start", "time_end"):
            value = getattr(decision.draft, field)
            if value:
                try:
                    parsed = datetime.fromisoformat(value)
                    times[field] = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
                except ValueError:
                    errors.append({"field": f"draft.{field}", "type": "datetime_parsing"})
        if len(times) == 2:
            require(times["time_start"] <= times["time_end"], "draft.time_end", "chronological_range_required")
        members = set(decision.draft.member_uris)
        require(len(members) == len(decision.draft.member_uris), "draft.member_uris", "unique_members_required")
        require(len(members) <= self.config.max_members_per_scene, "draft.member_uris", "member_limit_exceeded")
        require(members <= all_uris, "draft.member_uris", "provided_memory_reference_required")
        selected = [card for card in packet["memories"] if card["uri"] in members]
        require(len(selected) >= self.config.min_evidence_nodes,
                "draft.member_uris", "minimum_evidence_nodes_required")
        evidence = set(decision.evidence_uris)
        supporting = [card for card in selected if card["uri"] in evidence]
        require(len(supporting_sources(supporting, self._evidence_cards(packet))) >= self.config.min_independent_sources,
                "evidence_uris", "independent_sources_required")
        require(evidence <= members, "evidence_uris", "member_reference_required")
        source_members = {member.context_uri for record in snapshot.scenes if record.scene_id in targets
                          for member in record.cluster.members}
        require(source_members <= members, "draft.member_uris", "source_members_required")
        unassigned = set(decision.unassigned_member_uris)
        require(unassigned <= offered, "unassigned_member_uris", "offered_memory_reference_required")
        require(not unassigned & members, "unassigned_member_uris", "member_assignment_must_be_disjoint")
        require(offered <= members | unassigned, "unassigned_member_uris", "offered_memories_must_be_assigned")
        if errors:
            raise SceneModelOutputError("decision", errors)
