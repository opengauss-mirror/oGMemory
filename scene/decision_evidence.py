"""Build scoped evidence packets for scene-model decisions."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from core.models import ContextNode
from retrieval.owner_scope import canonical_owner_space
from scene.config import SceneConfig
from scene.decision_contracts import DecisionLimits, SceneDecisionError
from scene.decision_sources import node_source_refs
from scene.discovery import CandidateConfig, generate_scene_candidates
from scene.models import SceneRecord, SceneSnapshot


class SceneEvidenceBuilder:
    """Assemble memory cards and candidate hints from one scoped input window."""

    def __init__(
        self, config: SceneConfig | None = None, limits: DecisionLimits | None = None,
    ) -> None:
        self.config = config or SceneConfig()
        self.limits = limits or DecisionLimits()
        self.config.validate()
        self.limits.validate()

    def build(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode],
        embeddings: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Return bounded cards; the caller budgets the complete stage prompt."""
        self.validate_input(snapshot, nodes, embeddings)
        nodes = [replace(node, owner_space=canonical_owner_space(node.owner_space)) for node in nodes]
        candidates = generate_scene_candidates(
            nodes, embeddings,
            CandidateConfig(
                neighbor_k=self.config.neighbor_k,
                similarity_threshold=self.config.similarity_threshold,
                strong_pair_threshold=self.config.strong_pair_threshold,
                min_candidate_size=self.config.min_evidence_nodes,
                max_candidate_size=self.config.max_members_per_scene,
                min_quality=self.config.min_quality,
                max_owner_nodes=max(self.limits.max_input_nodes, self.config.max_members_per_scene),
            ),
        )
        known = {
            member.context_uri
            for record in snapshot.scenes if record.status == "ACTIVE"
            for member in record.cluster.members
        }
        cards = [self.node_card(node) for node in sorted(nodes, key=lambda value: value.uri)]
        prompt_budget_chars = min(
            self.limits.max_prompt_chars, self.config.analysis_input_budget_tokens * 4,
        )
        per_card_chars = max(64, prompt_budget_chars // max(len(cards) * 2, 1))
        for card in cards:
            card["text"] = card["text"][:per_card_chars]
        candidate_cards = [
            {
                "basis": candidate.basis,
                "member_uris": [
                    member.context_uri
                    for member in candidate.members[:self.config.max_members_per_scene]
                ],
                "quality": candidate.quality,
                "source_archive_ids": list(candidate.source_archive_ids),
            }
            for candidate in candidates[:self.config.max_candidate_scenes]
        ]
        existing = [
            {
                "scene_id": record.scene_id,
                "title": record.cluster.label,
                "description": record.cluster.summary[:self.limits.max_description_chars],
                "member_uris": [
                    member.context_uri
                    for member in record.cluster.members
                ],
                "quality": record.cluster.quality,
            }
            for record in self._eligible_scenes(snapshot, nodes)
        ]
        while existing and len(json.dumps(
            existing, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )) > self.config.candidate_card_budget_tokens * 4:
            existing.pop()
        return {
            "account_id": snapshot.account_id,
            "owner_space": canonical_owner_space(snapshot.owner_space),
            "limits": {
                "max_members": self.config.max_members_per_scene,
                "max_candidates": self.config.max_candidate_scenes,
                "min_evidence_nodes": self.config.min_evidence_nodes,
                "min_independent_sources": self.config.min_independent_sources,
            },
            "offered_member_uris": sorted({card["uri"] for card in cards} - known),
            "memories": cards,
            "existing_scenes": existing,
            "candidate_groups": candidate_cards,
        }

    def node_card(self, node: ContextNode) -> dict[str, Any]:
        """Retain bounded node text, deduplicated source references and timestamps."""
        metadata = node.metadata or {}
        text = "\n".join(
            value.strip()
            for value in (node.abstract, node.overview, node.content)
            if isinstance(value, str) and value.strip()
        )
        return {
            "uri": node.uri,
            "context_type": node.context_type,
            "category": node.category,
            "owner_space": canonical_owner_space(node.owner_space),
            "text": text[:self.limits.max_node_chars],
            "source_refs": list(node_source_refs(node)[:4]),
            "time": {
                key: metadata[key]
                for key in ("created_at", "updated_at", "event_time", "session_date", "timestamp", "when")
                if key in metadata
            },
        }

    def validate_input(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode],
        embeddings: dict[str, list[float]],
    ) -> None:
        """Check the scope and bounds shared by packet building and application."""
        if not snapshot.account_id or not snapshot.owner_space or snapshot.revision < 0:
            raise SceneDecisionError("scene snapshot requires a scope and nonnegative revision")
        if len(nodes) > self.limits.max_input_nodes:
            raise SceneDecisionError("scene decision input exceeds max_input_nodes")
        if sum(record.status == "ACTIVE" for record in snapshot.scenes) > self.config.max_active_scenes_per_space:
            raise SceneDecisionError("active scene count exceeds configured capacity")
        owner_space = canonical_owner_space(snapshot.owner_space)
        seen: set[str] = set()
        for node in nodes:
            uri = urlsplit(node.uri)
            if node.uri in seen:
                raise SceneDecisionError("scene decision requires unique node URIs")
            seen.add(node.uri)
            if (canonical_owner_space(node.owner_space) != owner_space
                    or uri.scheme != "ctx" or uri.netloc != snapshot.account_id):
                raise SceneDecisionError("scene decision node scope mismatch")
            if not embeddings.get(node.uri):
                raise SceneDecisionError(f"scene decision requires an embedding for {node.uri}")
        known_ids = {record.scene_id for record in snapshot.scenes}
        if len(known_ids) != len(snapshot.scenes):
            raise SceneDecisionError("scene snapshot requires unique scene IDs")
        for record in snapshot.scenes:
            if canonical_owner_space(record.cluster.owner_space) != owner_space:
                raise SceneDecisionError("scene snapshot owner mismatch")

    def _eligible_scenes(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode],
    ) -> list[SceneRecord]:
        available = {node.uri for node in nodes}
        return [
            record for record in snapshot.scenes
            if record.status == "ACTIVE" and record.origin == "automatic"
            and 0 < len(record.cluster.members) <= self.config.max_members_per_scene
            and {member.context_uri for member in record.cluster.members} <= available
        ][:self.config.max_candidate_scenes]
