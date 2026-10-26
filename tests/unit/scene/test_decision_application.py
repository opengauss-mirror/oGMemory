"""Executable scene lifecycle and evidence-gate recovery contracts."""

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import permutations

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision_application import SceneDecisionApplier
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import SceneDecisionError, SceneDecisionOutput
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_proposal import SceneProposalStage
from scene.models import SceneSnapshot
from tests.unit.scene.test_adapters import adapter
from tests.unit.scene.test_decision_evidence import node, record, vectors
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_decision_proposal import case


def application_case(action="CREATE"):
    snapshot, packet, proposal = case(action)
    nodes = [node(index) for index in range(5)]
    decision = SceneDecisionOutput(**proposal)
    return snapshot, nodes, vectors(nodes), decision, packet


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_accepted_proposal_materializes_versioned_scene(action):
    snapshot, nodes, embeddings, decision, packet = application_case(action)
    model = JSONModel(decision.model_dump(), {"decision": "ACCEPT"})
    stage = SceneProposalStage(SceneModelCaller(model))
    budget = CallBudget(3)
    decision = stage.decide(snapshot, packet, budget)
    assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"
    assert stage.review(decision, packet, budget).decision == "ACCEPT"
    result = SceneDecisionApplier().apply(snapshot, nodes, embeddings, decision)
    assert result.revision == snapshot.revision + 1
    assert (result.account_id, result.owner_space) == ("acct", "user:alice")
    assert len(result.scenes) == 1
    scene = result.scenes[0]
    assert scene.version == (1 if action == "CREATE" else 2)
    assert (scene.status, scene.origin, scene.cluster.basis) == ("ACTIVE", "automatic", "llm")
    assert scene.cluster.label == decision.draft.title
    assert scene.cluster.summary == decision.draft.description
    assert [member.context_uri for member in scene.cluster.members] == decision.draft.member_uris
    assert scene.cluster.centroid == (1.0, 0.0)
    assert scene.cluster.quality == 1.0
    assert scene.cluster.time_start == "2026-10-20"
    assert scene.cluster.time_end == "2026-10-22"
    assert budget.used == 2


@pytest.mark.parametrize("action", ["UPDATE", "MERGE"])
def test_replacement_keeps_survivor_identity_edited_summary_and_other_scenes(action):
    snapshot, nodes, embeddings, decision, _ = application_case(action)
    first = replace(snapshot.scenes[0], version=4, description_edited=True,
                    cluster=replace(snapshot.scenes[0].cluster, summary="Curated release description"))
    other_targets = tuple(replace(scene, version=7) for scene in snapshot.scenes[1:])
    untouched = record("unrelated", [nodes[-1]], origin="manual", version=9)
    snapshot = replace(snapshot, scenes=(untouched, first, *other_targets))
    decision = decision.model_copy(update={"target_scene_ids": list(reversed(decision.target_scene_ids))})
    before = deepcopy((snapshot, nodes, embeddings, decision))
    result = SceneDecisionApplier().apply(snapshot, nodes, embeddings, decision)
    survivor, remaining = result.scenes
    assert survivor.scene_id == "first"
    assert survivor.version == (5 if action == "UPDATE" else 8)
    assert survivor.description_edited is True
    assert survivor.cluster.summary == "Curated release description"
    assert remaining is untouched
    assert (snapshot, nodes, embeddings, decision) == before


@pytest.mark.parametrize("edited", [False, True])
def test_archived_scene_reactivates_with_stable_identity_and_replay(edited):
    snapshot, nodes, embeddings, decision, _ = application_case()
    applier = SceneDecisionApplier()
    created = applier.apply(snapshot, nodes, embeddings, decision)
    original = created.scenes[0]
    archived = replace(original, status="ARCHIVED", version=6, description_edited=edited,
                       cluster=replace(original.cluster, summary="Curated" if edited else original.cluster.summary))
    checkpoint = replace(created, revision=7, scenes=(archived,))
    restored = applier.apply(checkpoint, nodes, embeddings, decision)
    assert restored.revision == 8
    scene = restored.scenes[0]
    assert (scene.scene_id, scene.version, scene.status) == (original.scene_id, 7, "ACTIVE")
    assert scene.description_edited == edited
    assert scene.cluster.summary == ("Curated" if edited else decision.draft.description)
    assert applier.apply(restored, nodes, embeddings, decision) is restored


def test_create_identity_and_projection_are_stable_across_order_and_owner_aliases():
    snapshot, nodes, embeddings, decision, _ = application_case()
    embeddings.update({item.uri: [1.0, index * 0.13] for index, item in enumerate(nodes)})
    applier = SceneDecisionApplier()
    expected = applier.apply(snapshot, nodes, embeddings, decision)
    for ordered in permutations(nodes[:3]):
        inputs = [*ordered, *nodes[3:]]
        inputs[0] = replace(inputs[0], owner_space="user_alice")
        draft = decision.draft.model_copy(update={"member_uris": list(reversed(decision.draft.member_uris))})
        result = applier.apply(replace(snapshot, owner_space="user_alice"), inputs, embeddings,
                               decision.model_copy(update={"draft": draft}))
        assert result.scenes == expected.scenes
        assert result.scenes[0].cluster.owner_space == "user:alice"


@pytest.mark.parametrize("version_key", ["version", "node_version"])
def test_cluster_keeps_canonical_versions_sources_and_grounded_boundary(version_key):
    snapshot, nodes, embeddings, decision, _ = application_case()
    archive_ref = ProvenanceResolver.build_id("archive", "release-session", ["message-1"])
    for index, item in enumerate(nodes):
        item.metadata = {version_key: index + 10, "source_refs": [f"source-{index}"],
                         "provenance_ids": [archive_ref]}
    decision = decision.model_copy(update={"draft": decision.draft.model_copy(update={
        "boundary": {"subject": "Atlas", "activity": "release", "milestone": 2},
    })})
    cluster = SceneDecisionApplier().apply(snapshot, nodes, embeddings, decision).scenes[0].cluster
    assert [member.node_version for member in cluster.members] == [10, 11, 12, 13]
    assert cluster.members[0].source_refs == (archive_ref, "source-0")
    assert cluster.source_archive_ids == ("release-session",)
    assert cluster.boundary == (("activity", "release"), ("milestone", "2"), ("subject", "Atlas"))
    assert sum(member.role == "anchor" for member in cluster.members) == 1


@pytest.mark.parametrize("change", ["time", "version", "source", "vector"])
def test_changed_create_projection_recovers_through_explicit_update(change):
    snapshot, nodes, embeddings, decision, _ = application_case()
    applier = SceneDecisionApplier()
    created = applier.apply(snapshot, nodes, embeddings, decision)
    if change == "time":
        decision = decision.model_copy(update={"draft": decision.draft.model_copy(update={"time_end": "2026-10-23"})})
    elif change == "version":
        nodes[0].metadata["version"] = 2
    elif change == "source":
        nodes[0].metadata["source_refs"] = ["new-source"]
    else:
        embeddings[nodes[0].uri] = [1.0, 0.5]
    with pytest.raises(SceneDecisionError, match="identity collision"):
        applier.apply(created, nodes, embeddings, decision)
    updated = decision.model_copy(update={"action": "UPDATE", "target_scene_ids": [created.scenes[0].scene_id]})
    result = applier.apply(created, nodes, embeddings, updated)
    assert result.revision == created.revision + 1
    assert result.scenes[0].version == 2
    assert result.scenes[0].scene_id == created.scenes[0].scene_id
    if change == "time":
        assert result.scenes[0].cluster.time_end == "2026-10-23"
    elif change == "version":
        assert result.scenes[0].cluster.members[0].node_version == 2
    elif change == "source":
        assert result.scenes[0].cluster.members[0].source_refs == ("new-source",)
    else:
        assert result.scenes[0].cluster.centroid[1] > 0


def test_capacity_recovery_applies_merge_and_preserves_snapshot_guards():
    @dataclass(frozen=True)
    class GuardedSnapshot(SceneSnapshot):
        evidence_fingerprints: tuple = ()

    snapshot, nodes, embeddings, merge, _ = application_case("MERGE")
    snapshot = GuardedSnapshot(snapshot.account_id, snapshot.owner_space, snapshot.revision,
                               snapshot.scenes, ((nodes[0].uri, "version-1"),))
    applier = SceneDecisionApplier(SceneConfig(max_active_scenes_per_space=2))
    create = merge.model_copy(update={"action": "CREATE", "target_scene_ids": []})
    with pytest.raises(SceneDecisionError, match="active capacity"):
        applier.apply(snapshot, nodes, embeddings, create)
    result = applier.apply(snapshot, nodes, embeddings, merge)
    assert len(result.scenes) == 1
    assert result.revision == snapshot.revision + 1
    assert type(result) is GuardedSnapshot
    assert result.evidence_fingerprints == snapshot.evidence_fingerprints


def test_wait_preserves_checkpoint_then_applies_completed_evidence():
    snapshot, nodes, embeddings, decision, packet = application_case()
    applier = SceneDecisionApplier()
    stage = SceneProposalStage(SceneModelCaller(JSONModel()))
    waiting = decision.model_copy(update={"action": "WAIT", "reason": "awaiting independent evidence"})
    assert applier.apply(snapshot, nodes, embeddings, waiting) is snapshot
    assert stage.deterministic_quality_gate(waiting, packet).reason == waiting.reason
    assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"
    assert applier.apply(snapshot, nodes, embeddings, decision).revision == snapshot.revision + 1


@pytest.mark.parametrize("missing", ["draft", "members", "evidence", "sources", "conflict"])
def test_quality_gate_accepts_completed_selected_evidence(missing):
    snapshot, nodes, embeddings, decision, packet = application_case()
    stage = SceneProposalStage(SceneModelCaller(JSONModel()))
    initial, incomplete = decision, deepcopy(packet)
    if missing == "draft":
        initial = decision.model_copy(update={"draft": None})
    elif missing == "members":
        initial = decision.model_copy(update={"draft": decision.draft.model_copy(update={"member_uris": [nodes[0].uri] * 3})})
    elif missing == "evidence":
        initial = decision.model_copy(update={"evidence_uris": [nodes[0].uri] * 3})
    elif missing == "sources":
        for card in incomplete["memories"][:4]:
            card["source_refs"] = ["one-source"]
    else:
        initial = decision.model_copy(update={"conflicts": ["event date unresolved"]})
    assert stage.deterministic_quality_gate(initial, incomplete).decision == "DEFER"
    assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"
    assert SceneDecisionApplier().apply(snapshot, nodes, embeddings, decision).scenes[0].version == 1


def test_quality_gate_resolves_parent_provenance_completed_by_tools():
    _, nodes, _, decision, packet = application_case()
    parent = packet["memories"][-1]
    parent["source_refs"] = [ProvenanceResolver.build_id("archive", "session-a", ["message-1"])]
    for card in packet["memories"][:4]:
        card["source_refs"] = [parent["uri"]]
    stage = SceneProposalStage(SceneModelCaller(JSONModel()))
    assert stage.deterministic_quality_gate(decision, packet).decision == "DEFER"
    packet["tool_evidence"] = [{"tool": "read_source_context", "results": [
        {"uri": parent["uri"], "source_refs": [ProvenanceResolver.build_id("archive", "session-b", ["message-2"])]},
    ]}]
    assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"


def test_quality_gate_uses_effective_configuration_for_completed_window():
    snapshot, nodes, embeddings, decision, packet = application_case()
    config = SceneConfig(min_evidence_nodes=5, min_independent_sources=5)
    stage = SceneProposalStage(SceneModelCaller(JSONModel()), config)
    assert stage.deterministic_quality_gate(decision, packet).decision == "DEFER"
    members = [item.uri for item in nodes]
    completed = decision.model_copy(update={"draft": decision.draft.model_copy(update={"member_uris": members}),
                                          "evidence_uris": members, "unassigned_member_uris": []})
    packet = SceneEvidenceBuilder(config).build(snapshot, nodes, embeddings)
    assert stage.deterministic_quality_gate(completed, packet).decision == "ACCEPT"
    assert len(SceneDecisionApplier(config).apply(snapshot, nodes, embeddings, completed).scenes[0].cluster.members) == 5


@pytest.mark.parametrize("source_changed", [False, True])
def test_reviewed_projection_publishes_with_current_canonical_evidence(source_changed):
    _, nodes, _, decision, _ = application_case()
    source = adapter(nodes)
    source.repository.snapshot = SceneSnapshot("acct", "user:alice", 0)
    model = JSONModel({"decision": "ADJUST", "title": "Atlas launch", "description": "Verified launch milestones"})
    stage = SceneProposalStage(SceneModelCaller(model))
    applier = SceneDecisionApplier()
    published = False
    for attempt in range(2):
        batch = source.load_input("acct", "user:alice", node_uris=[item.uri for item in nodes])
        packet = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
        stage.validate(decision, batch.snapshot, packet)
        assert stage.deterministic_quality_gate(decision, packet).decision == "ACCEPT"
        if attempt == 0:
            gate = stage.review(decision, packet, CallBudget(3))
            decision = decision.model_copy(update={"draft": decision.draft.model_copy(update={
                "title": gate.title, "description": gate.description,
            })})
        replacement = applier.apply(batch.snapshot, list(batch.nodes), batch.embeddings, decision)
        if source_changed and attempt == 0:
            source.fs.nodes[nodes[0].uri].metadata["version"] = 2
        published = source.compare_and_swap(batch.snapshot, replacement)
        if published:
            break
    assert published
    saved = source.repository.load_snapshot("acct", "user:alice")
    assert saved.revision == 1
    assert saved.scenes[0].cluster.label == "Atlas launch"
    assert saved.scenes[0].cluster.summary == "Verified launch milestones"
    assert saved.scenes[0].cluster.members[0].node_version == (2 if source_changed else 0)
    assert saved.scenes[0].cluster.members[0].source_refs == ("source-0",)
    assert attempt == int(source_changed)
