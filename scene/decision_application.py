"""Apply validated scene decisions to detached, versioned snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from core.models import ContextNode
from retrieval.owner_scope import canonical_owner_space
from scene.config import SceneConfig
from scene.decision_contracts import DecisionLimits, SceneDecisionError, SceneDecisionOutput
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_sources import node_source_refs
from scene.discovery import _candidate, _validated_vectors
from scene.models import SceneCluster, SceneRecord, SceneSnapshot


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class SceneDecisionApplier:
    """Materialize a proposal after shape validation and quality acceptance.

    The caller applies any approved title/description adjustment first. The returned
    snapshot retains evidence guards carried by snapshot subclasses for store CAS.
    """

    def __init__(self, config: SceneConfig | None = None, limits: DecisionLimits | None = None) -> None:
        self.evidence = SceneEvidenceBuilder(config, limits)
        self.config = self.evidence.config

    def apply(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode],
        embeddings: dict[str, list[float]], decision: SceneDecisionOutput,
    ) -> SceneSnapshot:
        if decision.action == "WAIT":
            return snapshot
        self.evidence.validate_input(snapshot, nodes, embeddings)
        draft = decision.draft
        if draft is None:
            raise SceneDecisionError("scene application requires a validated draft")
        nodes = [replace(node, owner_space=canonical_owner_space(node.owner_space))
                 for node in sorted(nodes, key=lambda item: item.uri)]
        by_uri = {node.uri: index for index, node in enumerate(nodes)}
        members = set(draft.member_uris)
        if (not members <= by_uri.keys() or len(members) != len(draft.member_uris)
                or not self.config.min_evidence_nodes <= len(members) <= self.config.max_members_per_scene):
            raise SceneDecisionError("scene application requires bounded canonical members")
        result = {record.scene_id: record for record in snapshot.scenes}
        targets = decision.target_scene_ids
        if decision.action == "CREATE":
            valid_targets = not targets
        else:
            valid_targets = (len(targets) == len(set(targets))
                             and (len(targets) == 1 if decision.action == "UPDATE" else len(targets) >= 2)
                             and all(target in result and result[target].status == "ACTIVE"
                                     and result[target].origin == "automatic" for target in targets))
        if not valid_targets:
            raise SceneDecisionError("scene application requires eligible targets")
        retained = {member.context_uri for target in targets for member in result[target].cluster.members}
        if not retained <= members:
            raise SceneDecisionError("scene application requires all target members")

        ordered_members = sorted(members)
        candidate = _candidate(nodes, tuple(by_uri[uri] for uri in ordered_members),
                               _validated_vectors(nodes, embeddings), basis="llm")
        fingerprint = _digest({"members": ordered_members, "title": draft.title,
                               "description": draft.description, "boundary": draft.boundary})
        projected_members = []
        for member in candidate.members:
            node = nodes[by_uri[member.context_uri]]
            metadata = node.metadata or {}
            projected_members.append(replace(
                member, node_version=int(metadata.get("version", metadata.get("node_version", 0))),
                source_refs=node_source_refs(node),
            ))
        cluster = SceneCluster(
            owner_space=candidate.owner_space,
            members=tuple(projected_members),
            centroid=candidate.centroid, quality=candidate.quality,
            label=draft.title, summary=draft.description, source_fingerprint=fingerprint,
            basis="llm", source_archive_ids=candidate.source_archive_ids,
            boundary=tuple(sorted((key, _canonical_json(value) if isinstance(value, (dict, list)) else str(value))
                                  for key, value in draft.boundary.items())),
            time_start=draft.time_start, time_end=draft.time_end,
        )
        scene_version = 1
        previous = None
        if targets:
            scene_id = min(targets)
            previous = result[scene_id]
            scene_version = max(result[target].version for target in targets) + 1
            for target in targets:
                result.pop(target)
        else:
            event = {"boundary": draft.boundary} if draft.boundary else {"title": draft.title}
            scene_id = _digest([snapshot.account_id, canonical_owner_space(snapshot.owner_space), ordered_members, event])[:32]
            if scene_id in result and result[scene_id].origin == "automatic":
                existing = result[scene_id]
                if existing.status == "ARCHIVED":
                    previous = result.pop(scene_id)
                    scene_version = previous.version + 1
                elif existing.status == "ACTIVE":
                    replay = replace(cluster, summary=existing.cluster.summary) if existing.description_edited else cluster
                    if existing.cluster == replay:
                        return snapshot
            if scene_id in result:
                raise SceneDecisionError("scene identity collision")
        description_edited = previous is not None and previous.description_edited
        if description_edited:
            cluster = replace(cluster, summary=previous.cluster.summary)
        result[scene_id] = SceneRecord(scene_id, cluster, version=scene_version, description_edited=description_edited)
        if sum(record.status == "ACTIVE" for record in result.values()) > self.config.max_active_scenes_per_space:
            raise SceneDecisionError("scene result exceeds configured active capacity")
        return replace(snapshot, revision=snapshot.revision + 1, scenes=tuple(result[key] for key in sorted(result)))
