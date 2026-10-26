"""Compose bounded scene decisions over a canonical evidence snapshot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from core.interfaces import LLM
from core.models import ContextNode
from scene.config import SceneConfig
from scene.decision_application import SceneDecisionApplier
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import (
    DecisionLimits,
    SceneCallBudgetError,
    SceneDecisionError,
    SceneDecisionResult,
    SceneEvidenceReader,
)
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_preflight import ScenePreflight
from scene.decision_proposal import SceneProposalStage
from scene.decision_sources import node_source_refs
from scene.models import SceneSnapshot
from scene.store import SceneDecisionStore


class SceneDecisionEngine:
    """Run preflight, proposal, quality review and versioned snapshot projection."""

    def __init__(
        self, llm: LLM, config: SceneConfig | None = None, *, limits: DecisionLimits | None = None,
        evidence_reader: SceneEvidenceReader | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SceneConfig()
        self.limits = limits or DecisionLimits()
        self.evidence_reader = evidence_reader
        self._evidence = SceneEvidenceBuilder(self.config, self.limits)
        caller = SceneModelCaller(llm, self.limits)
        self._preflight = ScenePreflight(caller, self.config, evidence_reader=evidence_reader)
        self._proposal = SceneProposalStage(caller, self.config)
        self._applier = SceneDecisionApplier(self.config, self.limits)

    def for_config(self, config: SceneConfig) -> SceneDecisionEngine:
        """Bind another configuration while sharing the model and input limits."""
        return SceneDecisionEngine(self.llm, config, limits=self.limits, evidence_reader=self.evidence_reader)

    def decide(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode], embeddings: dict[str, list[float]],
    ) -> SceneSnapshot:
        return self.decide_with_result(snapshot, nodes, embeddings).snapshot

    def decide_with_result(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode], embeddings: dict[str, list[float]],
    ) -> SceneDecisionResult:
        """Return a detached projection and bounded diagnostics for the next batch."""
        return self._decide_with_result(snapshot, list(nodes), dict(embeddings))

    def _packet(self, snapshot, nodes, embeddings):
        packet = self._evidence.build(snapshot, nodes, embeddings)
        canonical = {node.uri: node for node in nodes}
        for card in packet["memories"]:
            card["source_refs"] = list(node_source_refs(canonical[card["uri"]]))
        return packet

    def _decide_with_result(self, snapshot, nodes, embeddings, *, source_fingerprints=None):
        self._evidence.validate_input(snapshot, nodes, embeddings)
        if not nodes:
            return SceneDecisionResult(snapshot, "WAIT", reason="empty input")
        known = {node.uri for node in nodes}
        if len(nodes) < self.config.min_evidence_nodes:
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(known)),
                reason="evidence count below configured threshold",
            )

        packet = self._packet(snapshot, nodes, embeddings)
        budget = CallBudget(self.config.max_llm_calls_per_batch)
        preflight = self._preflight.run(packet, budget)
        if source_fingerprints is not None:
            source_fingerprints.extend(
                (card["uri"], card["node_fingerprint"])
                for result in packet.get("tool_evidence", [])
                if result["tool"] in {"read_source_context", "search_related_nodes"}
                for card in result.get("results", []) if isinstance(card, dict)
                and isinstance(card.get("uri"), str) and isinstance(card.get("node_fingerprint"), str)
            )
        packet = self._expand_evidence(snapshot, nodes, embeddings, packet)
        known = {node.uri for node in nodes}
        offered = set(packet["offered_member_uris"])
        preflight = self._preflight.enforce_gate(packet, preflight)
        if preflight.status == "WAIT":
            return SceneDecisionResult(
                snapshot, "WAIT", calls=budget.used,
                pending_member_uris=tuple(sorted(offered | (set(preflight.evidence_uris) & known))),
                reason=preflight.reason or "; ".join(preflight.missing),
            )

        decision = self._proposal.decide(snapshot, packet, budget)
        if decision.action == "WAIT":
            return SceneDecisionResult(
                snapshot, "WAIT", calls=budget.used,
                pending_member_uris=tuple(sorted(offered | (set(decision.unassigned_member_uris) & known))),
                reason=decision.reason or "; ".join(decision.conflicts),
            )

        floor = self._proposal.deterministic_quality_gate(decision, packet)
        if floor.decision == "DEFER":
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(offered)),
                calls=budget.used, reason=floor.reason,
            )
        if self.config.max_llm_calls_per_batch == 3:
            if budget.remaining == 0:
                raise SceneCallBudgetError("quality")
            gate = self._proposal.review(decision, packet, budget)
        else:
            gate = floor
        if gate.decision == "DEFER":
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(offered)),
                calls=budget.used, reason=gate.reason,
            )
        if gate.decision == "ADJUST" and decision.draft is not None:
            decision = decision.model_copy(update={
                "draft": decision.draft.model_copy(update={
                    "title": gate.title or decision.draft.title,
                    "description": gate.description or decision.draft.description,
                }),
            })
        replacement = self._applier.apply(snapshot, nodes, embeddings, decision)
        pending = tuple(sorted(set(decision.unassigned_member_uris)))
        if replacement == snapshot:
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=pending,
                calls=budget.used, reason="no projection change",
            )
        return SceneDecisionResult(
            replacement, decision.action, pending_member_uris=pending, calls=budget.used,
        )

    def _expand_evidence(self, snapshot, nodes, embeddings, packet):
        loader = getattr(self.evidence_reader, "load_nodes", None)
        tool_evidence = packet.get("tool_evidence", [])
        if not callable(loader) or not tool_evidence:
            return packet
        requested, selected_ids = [], set()
        for result in tool_evidence:
            for item in result.get("results", []):
                if not isinstance(item, dict):
                    continue
                if result["tool"] == "search_related_nodes" and isinstance(item.get("uri"), str):
                    requested.append(item["uri"])
                if result["tool"] == "search_scene_cards" and isinstance(item.get("scene_id"), str):
                    selected_ids.add(item["scene_id"])
        for record in snapshot.scenes:
            if record.scene_id in selected_ids:
                requested.extend(member.context_uri for member in record.cluster.members)
        known = {node.uri for node in nodes}
        requested = list(dict.fromkeys(uri for uri in requested if uri not in known))
        requested = requested[:max(0, self.limits.max_input_nodes - len(nodes))]
        if requested:
            extra, vectors = loader(snapshot.account_id, snapshot.owner_space, requested)
            extra = [node for node in extra if node.uri in requested and node.uri in vectors]
            vectors = {node.uri: vectors[node.uri] for node in extra}
            self._evidence.validate_input(snapshot, nodes + extra, {**embeddings, **vectors})
            nodes.extend(extra)
            embeddings.update(vectors)
        expanded = self._packet(snapshot, nodes, embeddings)
        expanded["tool_evidence"] = tool_evidence
        return expanded

    def refresh(
        self, store: SceneDecisionStore, account_id: str, owner_space: str,
        *, node_uris: Sequence[str] | None = None,
    ) -> SceneDecisionResult:
        """Publish one decision with the complete canonical evidence guarded by CAS."""
        batch = store.load_input(account_id, owner_space, node_uris=node_uris)
        if (batch.snapshot.account_id, batch.snapshot.owner_space) != (account_id, owner_space):
            raise SceneDecisionError("scene store returned a different scope")
        nodes = list(batch.nodes)
        source_fingerprints = []
        result = self._decide_with_result(
            batch.snapshot, nodes, dict(batch.embeddings), source_fingerprints=source_fingerprints,
        )
        pending = set(node_uris or ()) - set(batch.unavailable_uris)
        if result.action != "WAIT":
            versions = {node.uri: int(node.metadata.get("version", node.metadata.get("node_version", 0)))
                        for node in nodes if node.uri in pending}
            stale = {member.context_uri for scene in result.snapshot.scenes if scene.status == "ACTIVE"
                     for member in scene.cluster.members if member.context_uri in versions
                     and member.node_version != versions[member.context_uri]}
            pending.difference_update(versions.keys() - stale)
        result = replace(result, pending_member_uris=tuple(sorted(set(result.pending_member_uris) | pending)))
        if result.snapshot is not batch.snapshot and not store.compare_and_swap(
            batch.snapshot, result.snapshot, evidence_nodes=nodes, source_fingerprints=source_fingerprints,
        ):
            raise RuntimeError("scene snapshot or evidence changed during decision; refresh again")
        return result
