"""Canonical source projection and explicit update recovery."""

from copy import deepcopy

import pytest

from scene.decision_application import SceneDecisionApplier
from scene.decision_contracts import SceneDecisionError
from scene.decision_evidence import SceneEvidenceBuilder
from tests.unit.scene.test_adapters import adapter
from tests.unit.scene.test_decision_application import application_case


@pytest.mark.parametrize("provenance", ["ref-0", ["ref-0"], ("ref-0",), {"ref-0"}])
def test_full_metadata_sources_publish_with_bounded_prompt_cards(provenance):
    snapshot, nodes, embeddings, decision, _ = application_case()
    expected = tuple(f"ref-{index}" for index in range(20))
    nodes[0].metadata = {
        "provenance_ids": provenance,
        "source_refs": [expected[0], "", "  ", *expected[1:-1], expected[-2]],
        "source_ref": expected[-1],
    }
    original = deepcopy(nodes[0].metadata)
    assert SceneEvidenceBuilder().node_card(nodes[0])["source_refs"] == list(expected[:4])
    applier = SceneDecisionApplier()
    result = applier.apply(snapshot, nodes, embeddings, decision)
    assert result.scenes[0].cluster.members[0].source_refs == expected
    assert applier.apply(result, nodes, embeddings, decision) is result
    assert nodes[0].metadata == original


@pytest.mark.parametrize("change", ["append", "replace", "remove"])
def test_tail_source_change_recovers_through_explicit_published_update(change):
    snapshot, nodes, _, decision, _ = application_case()
    refs = [f"evidence-{index}" for index in range(5)]
    nodes[0].metadata["source_refs"] = refs
    source = adapter(nodes)
    source.repository.snapshot = snapshot
    uris = [node.uri for node in nodes]
    batch = source.load_input("acct", "user:alice", node_uris=uris)
    applier = SceneDecisionApplier()
    created = applier.apply(batch.snapshot, list(batch.nodes), batch.embeddings, decision)
    assert source.compare_and_swap(batch.snapshot, created)
    current = list(refs)
    if change == "append":
        current.append("evidence-5")
    elif change == "replace":
        current[-1] = "evidence-5"
    else:
        current.pop()
    source.fs.nodes[nodes[0].uri].metadata["source_refs"] = current
    fresh = source.load_input("acct", "user:alice", node_uris=uris)
    with pytest.raises(SceneDecisionError, match="identity collision"):
        applier.apply(fresh.snapshot, list(fresh.nodes), fresh.embeddings, decision)
    update = decision.model_copy(update={
        "action": "UPDATE", "target_scene_ids": [created.scenes[0].scene_id],
    })
    result = applier.apply(fresh.snapshot, list(fresh.nodes), fresh.embeddings, update)
    assert source.compare_and_swap(fresh.snapshot, result)
    saved = source.repository.load_snapshot("acct", "user:alice")
    assert saved.scenes[0].cluster.members[0].source_refs == tuple(current)
    assert saved.scenes[0].scene_id == created.scenes[0].scene_id
    assert saved.scenes[0].version == created.scenes[0].version + 1
    assert saved.revision == created.revision + 1
    assert applier.apply(saved, list(fresh.nodes), fresh.embeddings, decision) is saved
