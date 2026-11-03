from copy import deepcopy
from dataclasses import replace

from core.models import ContextNode
from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot
from scene.store import SceneDecisionInput, SceneDecisionStore, SceneSnapshotRepository


class MemoryDecisionStore:
    def __init__(self, *inputs):
        self.inputs = {
            (batch.snapshot.account_id, batch.snapshot.owner_space): deepcopy(batch)
            for batch in inputs
        }

    def load_input(self, account_id, owner_space):
        return deepcopy(self.inputs[account_id, owner_space])

    def load_snapshot(self, account_id, owner_space):
        return self.load_input(account_id, owner_space).snapshot

    def compare_and_swap(self, expected, replacement):
        scope = (expected.account_id, expected.owner_space)
        if (replacement.account_id, replacement.owner_space) != scope:
            raise ValueError("scene scope mismatch")
        if replacement.revision != expected.revision + 1:
            raise ValueError("scene revision must advance by one")
        current = self.inputs[scope]
        if current.snapshot.revision != expected.revision:
            return False
        self.inputs[scope] = replace(current, snapshot=deepcopy(replacement))
        return True


def _input(account="account", owner="user:alice"):
    nodes = tuple(
        ContextNode(
            uri=f"ctx://{account}/memories/event/{i}",
            owner_space=owner,
            context_type="MEMORY",
            category="event",
            level=0,
            abstract=f"Conference preparation {i}",
            overview="",
            content="Conference preparation",
        )
        for i in range(2)
    )
    return SceneDecisionInput(
        SceneSnapshot(account, owner, 0), nodes, {n.uri: [1.0, 0.0] for n in nodes}
    )


def test_snapshot_contract_publishes_scene_identity_and_revision():
    store: SceneDecisionStore = MemoryDecisionStore(_input())
    assert isinstance(store, SceneDecisionStore)
    assert isinstance(store, SceneSnapshotRepository)
    batch = store.load_input("account", "user:alice")
    cluster = SceneCluster(
        owner_space=batch.snapshot.owner_space,
        members=tuple(SceneMember(n.uri, 1.0) for n in batch.nodes),
        centroid=(1.0, 0.0),
        quality=1.0,
        label="Conference preparation",
        summary="Preparing the conference materials.",
        source_fingerprint="evidence-revision-1",
    )
    replacement = replace(batch.snapshot, revision=1, scenes=(SceneRecord("scene-1", cluster),))
    assert store.compare_and_swap(batch.snapshot, replacement)
    published = store.load_input("account", "user:alice")
    assert published.snapshot == replacement
    assert published.nodes == batch.nodes
    assert published.embeddings == batch.embeddings
    assert batch.snapshot == SceneSnapshot("account", "user:alice", 0)


def test_snapshot_preserves_member_versions_sources_and_activity_boundary():
    store = MemoryDecisionStore(_input())
    snapshot = store.load_snapshot("account", "user:alice")
    member = SceneMember(
        "ctx://account/memories/event/0", 1.0,
        node_version=7, source_refs=("archive-1", "archive-2"),
    )
    cluster = SceneCluster(
        owner_space="user:alice", members=(member,), centroid=(1.0, 0.0),
        quality=1.0, label="Conference preparation", summary="Prepare the launch.",
        source_fingerprint="source-v7", boundary=(("goal", "conference launch"),),
        time_start="2026-10-21", time_end="2026-10-22",
    )
    record = SceneRecord(
        "scene-1", cluster, version=2, origin="manual", description_edited=True,
    )
    assert store.compare_and_swap(snapshot, replace(snapshot, revision=1, scenes=(record,)))
    saved = store.load_snapshot("account", "user:alice").scenes[0]
    assert saved == record
    assert saved.status == "ACTIVE"
    assert saved.cluster.members[0].node_version == 7
    assert saved.cluster.members[0].source_refs == ("archive-1", "archive-2")
    assert saved.cluster.boundary == (("goal", "conference launch"),)


def test_snapshot_inputs_are_detached_from_store_state():
    store: SceneDecisionStore = MemoryDecisionStore(_input())
    first = store.load_input("account", "user:alice")
    uri = first.nodes[0].uri
    first.nodes[0].abstract = "Updated locally"
    first.embeddings[uri][0] = 0.5
    current = store.load_input("account", "user:alice")
    assert current.nodes[0].abstract == "Conference preparation 0"
    assert current.embeddings[uri] == [1.0, 0.0]


def test_snapshot_scopes_advance_independently():
    batches = (_input(), _input(owner="user:bob"), _input(account="other"))
    store: SceneDecisionStore = MemoryDecisionStore(*batches)
    for batch in batches:
        replacement = replace(batch.snapshot, revision=1)
        assert store.compare_and_swap(batch.snapshot, replacement)
        assert (
            store.load_input(replacement.account_id, replacement.owner_space).snapshot
            == replacement
        )
