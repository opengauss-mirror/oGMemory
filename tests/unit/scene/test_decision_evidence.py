"""Executable evidence-window and card-budget contracts."""

import json
from copy import deepcopy
from dataclasses import replace
from itertools import permutations

import pytest

from core.models import ContextNode
from core.provenance_resolver import ProvenanceResolver
from scene.config import InMemorySceneConfigProvider, SceneConfig, SceneConfigCache
from scene.decision_contracts import DecisionLimits
from scene.decision_evidence import SceneEvidenceBuilder
from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot


def node(index, *, account="acct", owner="user:alice"):
    return ContextNode(
        uri=f"ctx://{account}/users/{owner.split(':')[1]}/memories/event/{index}",
        context_type="MEMORY", category="event", level=0, owner_space=owner,
        abstract=f"Atlas release milestone {index}", overview="Release planning",
        content=f"The team completed milestone {index}.",
        metadata={"source_refs": [f"source-{index}"], "event_time": "2026-10-22"},
    )


def vectors(nodes):
    return {item.uri: [1.0, 0.0] for item in nodes}


def record(scene_id, nodes, **overrides):
    cluster = SceneCluster(
        owner_space=nodes[0].owner_space,
        members=tuple(SceneMember(item.uri, 1.0) for item in nodes),
        centroid=(1.0, 0.0), quality=1.0, label="Atlas release",
        summary="The team is planning the Atlas release.", source_fingerprint=scene_id,
    )
    return SceneRecord(scene_id, cluster, **overrides)


def test_packet_contains_scoped_memories_candidates_and_evidence_floor():
    nodes = [node(index) for index in range(3)]
    packet = SceneEvidenceBuilder().build(SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes))
    assert packet["account_id"] == "acct"
    assert packet["owner_space"] == "user:alice"
    assert packet["limits"] == {
        "max_members": 64, "max_candidates": 8,
        "min_evidence_nodes": 3, "min_independent_sources": 2,
    }
    assert packet["offered_member_uris"] == [item.uri for item in nodes]
    assert packet["existing_scenes"] == []
    assert packet["candidate_groups"] == [{
        "basis": "semantic", "member_uris": [item.uri for item in nodes],
        "quality": 1.0, "source_archive_ids": [],
    }]
    assert packet["memories"][0] == {
        "uri": nodes[0].uri, "context_type": "MEMORY", "category": "event",
        "owner_space": "user:alice", "source_refs": ["source-0"],
        "text": "Atlas release milestone 0\nRelease planning\nThe team completed milestone 0.",
        "time": {"event_time": "2026-10-22"},
    }


@pytest.mark.parametrize("account,owner", [("acct", "user:alice"), ("other", "user:bob")])
def test_builder_uses_scope_from_each_input_snapshot(account, owner):
    nodes = [node(index, account=account, owner=owner) for index in range(3)]
    packet = SceneEvidenceBuilder().build(SceneSnapshot(account, owner, 4), nodes, vectors(nodes))
    assert (packet["account_id"], packet["owner_space"]) == (account, owner)
    assert {card["owner_space"] for card in packet["memories"]} == {owner}
    assert packet["candidate_groups"][0]["member_uris"] == [item.uri for item in nodes]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_small_windows_preserve_available_evidence(count):
    nodes = [node(index) for index in range(count)]
    packet = SceneEvidenceBuilder().build(SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes))
    assert packet["offered_member_uris"] == [item.uri for item in nodes]
    assert len(packet["memories"]) == count
    assert len(packet["candidate_groups"]) == int(count == 2)


def test_packet_is_stable_across_input_order_and_owns_its_cards():
    nodes = [node(index) for index in range(3)]
    snapshot = SceneSnapshot("acct", "user:alice", 2, (record("release", nodes),))
    embeddings = vectors(nodes)
    before = deepcopy((snapshot, nodes, embeddings))
    builder = SceneEvidenceBuilder()
    expected = builder.build(snapshot, nodes, embeddings)
    for ordered in permutations(nodes):
        assert builder.build(snapshot, list(ordered), embeddings) == expected
    expected["memories"][0]["source_refs"].append("local-note")
    assert (snapshot, nodes, embeddings) == before
    assert builder.build(snapshot, nodes, embeddings)["memories"][0]["source_refs"] == ["source-0"]


@pytest.mark.parametrize("refs", ["archive-1", ["archive-1"], ("archive-1",), {"archive-1"}])
def test_node_card_combines_source_formats_and_preserves_timestamp_precision(refs):
    item = node(0)
    item.metadata = {
        "provenance_ids": refs,
        "source_refs": ["archive-1", "", "archive-2", "archive-3"],
        "source_ref": "archive-4",
        "created_at": "2026-10", "when": "last summer",
    }
    card = SceneEvidenceBuilder().node_card(item)
    assert card["source_refs"] == ["archive-1", "archive-2", "archive-3", "archive-4"]
    assert card["time"] == {"created_at": "2026-10", "when": "last summer"}


def test_node_card_retains_first_four_distinct_source_refs():
    item = node(0)
    item.metadata = {"source_refs": [f"source-{index}" for index in range(6)]}
    assert SceneEvidenceBuilder().node_card(item)["source_refs"] == [f"source-{index}" for index in range(4)]


@pytest.mark.parametrize("max_node_chars,max_prompt_chars,expected", [(80, 4096, 80), (800, 600, 100)])
def test_packet_applies_node_and_shared_text_budgets(max_node_chars, max_prompt_chars, expected):
    nodes = [node(index) for index in range(3)]
    for item in nodes:
        item.abstract = "证据" * 1000
    limits = DecisionLimits(max_node_chars=max_node_chars, max_prompt_chars=max_prompt_chars)
    packet = SceneEvidenceBuilder(limits=limits).build(
        SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes),
    )
    assert [len(card["text"]) for card in packet["memories"]] == [expected] * 3
    assert packet["offered_member_uris"] == [item.uri for item in nodes]


def test_analysis_token_budget_shapes_card_text():
    nodes = [node(index) for index in range(4)]
    for item in nodes:
        item.content = "x" * 1000
    config = replace(SceneConfig(), analysis_input_budget_tokens=256, analysis_trigger_tokens=128)
    packet = SceneEvidenceBuilder(config).build(SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes))
    assert [len(card["text"]) for card in packet["memories"]] == [128] * 4


def test_existing_candidates_require_complete_active_automatic_evidence():
    nodes = [node(index) for index in range(6)]
    eligible = record("release", nodes[:2])
    snapshot = SceneSnapshot("acct", "user:alice", 7, (
        eligible,
        record("manual", nodes[2:3], origin="manual"),
        record("archived", nodes[3:4], status="ARCHIVED"),
        record("partial", [nodes[4], node(99)]),
        replace(eligible, scene_id="empty", cluster=replace(eligible.cluster, members=())),
    ))
    packet = SceneEvidenceBuilder().build(snapshot, nodes, vectors(nodes))
    assert packet["existing_scenes"] == [{
        "scene_id": "release", "title": eligible.cluster.label,
        "description": eligible.cluster.summary,
        "member_uris": [item.uri for item in nodes[:2]], "quality": 1.0,
    }]
    assert packet["offered_member_uris"] == [nodes[3].uri, nodes[5].uri]


def test_existing_scene_descriptions_follow_configured_character_limit():
    nodes = [node(index) for index in range(3)]
    existing = record("release", nodes)
    builder = SceneEvidenceBuilder(limits=DecisionLimits(max_description_chars=12))
    packet = builder.build(SceneSnapshot("acct", "user:alice", 0, (existing,)), nodes, vectors(nodes))
    assert packet["existing_scenes"][0]["description"] == existing.cluster.summary[:12]
    assert packet["existing_scenes"][0]["member_uris"] == [item.uri for item in nodes]


def test_existing_cards_fit_serialized_budget_with_stable_priority():
    nodes = [node(index) for index in range(3)]
    records = tuple(record(f"release-{index}", nodes) for index in range(3))
    snapshot = SceneSnapshot("acct", "user:alice", 0, records)
    baseline = SceneEvidenceBuilder().build(snapshot, nodes, vectors(nodes))
    first_size = len(json.dumps(baseline["existing_scenes"][:1], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    config = replace(SceneConfig(), candidate_card_budget_tokens=(first_size + 3) // 4)
    packet = SceneEvidenceBuilder(config).build(snapshot, nodes, vectors(nodes))
    assert packet["existing_scenes"] == baseline["existing_scenes"][:1]


def test_candidate_count_and_member_count_follow_configuration():
    nodes = [node(index) for index in range(9)]
    config = replace(SceneConfig(), max_candidate_scenes=2, max_members_per_scene=3)
    snapshot = SceneSnapshot("acct", "user:alice", 0, tuple(
        record(f"release-{index}", nodes[index:index + 3]) for index in (0, 3, 6)
    ))
    packet = SceneEvidenceBuilder(config).build(snapshot, nodes, vectors(nodes))
    assert len(packet["candidate_groups"]) == 2
    assert all(len(group["member_uris"]) == 3 for group in packet["candidate_groups"])
    assert [card["scene_id"] for card in packet["existing_scenes"]] == ["release-0", "release-3"]


def test_shared_archive_produces_an_episodic_candidate_with_original_sources():
    nodes = [node(index) for index in range(3)]
    source = ProvenanceResolver.build_id("archive", "release-meeting")
    for item in nodes:
        item.metadata = {"provenance_ids": [source]}
    embeddings = {nodes[0].uri: [1.0, 0.0], nodes[1].uri: [0.0, 1.0], nodes[2].uri: [-1.0, 0.0]}
    packet = SceneEvidenceBuilder().build(SceneSnapshot("acct", "user:alice", 0), nodes, embeddings)
    assert len(packet["candidate_groups"]) == 1
    assert packet["candidate_groups"][0]["basis"] == "episodic"
    assert packet["candidate_groups"][0]["source_archive_ids"] == ["release-meeting"]
    assert [card["source_refs"] for card in packet["memories"]] == [[source]] * 3


@pytest.mark.parametrize("member_limit", [3, 4])
@pytest.mark.parametrize("candidate_limit", [1, 2])
def test_hot_reload_selects_complete_scenes_and_restores_members_on_rollback(member_limit, candidate_limit):
    nodes = [node(index) for index in range(2 * member_limit + 2)]
    large = record("large", nodes[:member_limit + 1])
    fitting = record("fitting", nodes[member_limit + 1:-1])
    snapshot = SceneSnapshot("acct", "user:alice", 7, (large, fitting))
    original = deepcopy(snapshot)
    provider = InMemorySceneConfigProvider(SceneConfig(max_candidate_scenes=candidate_limit))
    cache = SceneConfigCache(provider)

    def build_current():
        config = cache.get("acct", "user:alice").config
        return SceneEvidenceBuilder(config).build(snapshot, nodes, vectors(nodes))

    initial = build_current()
    assert initial["existing_scenes"][0]["member_uris"] == [
        member.context_uri for member in large.cluster.members
    ]
    provider.publish("acct", "user:alice", 0, {"max_members_per_scene": member_limit})
    updated = build_current()
    assert [card["scene_id"] for card in updated["existing_scenes"]] == ["fitting"]
    assert updated["existing_scenes"][0]["member_uris"] == [
        member.context_uri for member in fitting.cluster.members
    ]
    assert updated["offered_member_uris"] == [nodes[-1].uri]
    assert updated["limits"]["max_members"] == member_limit
    assert snapshot == original

    provider.rollback("acct", "user:alice", 0)
    assert build_current() == initial


@pytest.mark.parametrize("owner", ["user:alice_team", "agent:helper_team"])
@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_adapter_evidence_groups_mixed_owner_aliases_and_preserves_publication(owner, legacy_snapshot):
    from tests.unit.scene.test_adapters import adapter, scene
    from tests.unit.scene.test_adapters import node as stored_node

    legacy = owner.replace(":", "_", 1)
    nodes = [stored_node(str(index), owner=owner) for index in range(4)]
    for item in nodes[::2]:
        item.owner_space = legacy
    existing = scene("release", nodes[:2])
    snapshot_owner, scene_owner = (legacy, owner) if legacy_snapshot else (owner, legacy)
    existing = replace(existing, cluster=replace(existing.cluster, owner_space=scene_owner))
    source = adapter(nodes)
    source.repository.snapshot = SceneSnapshot("acme", snapshot_owner, 7, (existing,))
    stored_before = deepcopy(source.fs.nodes)
    batch = source.load_input("acme", snapshot_owner, node_uris=[item.uri for item in nodes])
    batch_before = deepcopy(batch)

    packet = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)

    assert packet["owner_space"] == owner
    assert {card["owner_space"] for card in packet["memories"]} == {owner}
    assert [group["member_uris"] for group in packet["candidate_groups"]] == [[item.uri for item in nodes]]
    assert packet["existing_scenes"][0]["member_uris"] == [item.uri for item in nodes[:2]]
    assert packet["offered_member_uris"] == [item.uri for item in nodes[2:]]
    assert batch == batch_before
    assert source.fs.nodes == stored_before
    assert source.compare_and_swap(batch.snapshot, replace(batch.snapshot, revision=8)) is True
    assert source.repository.snapshot.owner_space == snapshot_owner
    assert source.repository.snapshot.scenes == (existing,)


@pytest.mark.parametrize("owner", ["user:alice_team", "agent:helper_team"])
def test_node_card_uses_canonical_owner_alias(owner):
    item = replace(node(0), owner_space=owner.replace(":", "_", 1))
    before = deepcopy(item)
    assert SceneEvidenceBuilder().node_card(item)["owner_space"] == owner
    assert item == before
