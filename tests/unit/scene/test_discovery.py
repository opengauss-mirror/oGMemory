from itertools import permutations

import pytest

from core.models import ContextNode
from core.provenance_resolver import ProvenanceResolver
from scene.discovery import (
    CandidateConfig,
    _nearest_neighbors,
    _pairwise_similarities,
    cosine,
    generate_scene_candidates,
)
from scene.models import SceneCandidate


def _node(index: int, owner: str = "user:alice") -> ContextNode:
    return ContextNode(
        uri=f"ctx://acct/users/alice/memories/event/{index}",
        context_type="MEMORY",
        category="event",
        level=0,
        owner_space=owner,
        abstract=f"trip planning detail {index}",
        overview="",
        content=f"travel plan {index}",
    )


def test_discovers_compact_candidate_and_rejects_singletons():
    nodes = [_node(index) for index in range(5)]
    embeddings = {
        nodes[0].uri: [1.0, 0.02],
        nodes[1].uri: [0.99, 0.03],
        nodes[2].uri: [0.98, 0.01],
        nodes[3].uri: [0.0, 1.0],
        nodes[4].uri: [-1.0, 0.0],
    }

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            neighbor_k=2,
            similarity_threshold=0.8,
            min_candidate_size=3,
            max_candidate_size=8,
            min_quality=0.8,
        ),
    )

    assert len(candidates) == 1
    assert {member.context_uri for member in candidates[0].members} == {
        nodes[0].uri,
        nodes[1].uri,
        nodes[2].uri,
    }
    assert candidates[0].quality > 0.99
    assert type(candidates[0]) is SceneCandidate
    assert candidates[0].label_hint == "trip / planning / detail"
    assert set(candidates[0].summary_hint.split(" | ")) == {node.abstract for node in nodes[:3]}


def test_candidate_fingerprint_is_deterministic():
    nodes = [_node(index) for index in range(3)]
    embeddings = {node.uri: [1.0, index / 100] for index, node in enumerate(nodes)}
    config = CandidateConfig(min_candidate_size=3, similarity_threshold=0.8)

    first = generate_scene_candidates(nodes, embeddings, config)
    second = generate_scene_candidates(list(reversed(nodes)), embeddings, config)

    assert first[0].source_fingerprint == second[0].source_fingerprint


@pytest.mark.parametrize("basis", ["semantic", "episodic", "entity_anchor"])
def test_tied_candidates_preserve_content_across_input_permutations(basis):
    nodes = [_node(index) for index in range(4)]
    for node, subject in zip(nodes, ["Alpha", "Bravo", "Charlie", "Delta"]):
        node.abstract = f"{subject} meets Morgan."
        node.content = ""
    if basis == "semantic":
        embeddings = {node.uri: [1.0, 0.0] for node in nodes}
    elif basis == "episodic":
        for node in nodes:
            node.metadata = {"provenance_ids": [ProvenanceResolver.build_id("archive", "meeting")]}
        embeddings = {node.uri: [float(i == j) for j in range(4)] for i, node in enumerate(nodes)}
    else:
        nodes[3].uri = "ctx://acct/users/alice/memories/entity/morgan"
        nodes[3].category = "entity"
        embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    expected = generate_scene_candidates(nodes, embeddings)

    assert len(expected) == 1
    assert expected[0].basis == basis
    assert expected[0].summary_hint == " | ".join(
        node.abstract for node in sorted(nodes, key=lambda node: node.uri)[:3]
    )
    anchor_uri = nodes[3].uri if basis == "entity_anchor" else min(node.uri for node in nodes)
    assert (
        next(member.context_uri for member in expected[0].members if member.role == "anchor")
        == anchor_uri
    )
    for ordered in permutations(nodes):
        actual = generate_scene_candidates(list(ordered), embeddings)
        assert actual == expected


def test_strong_duplicate_pair_is_a_candidate():
    nodes = [_node(index) for index in range(2)]
    embeddings = {nodes[0].uri: [1.0, 0.0], nodes[1].uri: [0.99, 0.01]}

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(min_candidate_size=3, strong_pair_threshold=0.95),
    )

    assert len(candidates) == 1
    assert len(candidates[0].members) == 2


def test_complete_link_blocks_semantic_chain_merge():
    nodes = [_node(index) for index in range(3)]
    embeddings = {
        nodes[0].uri: [1.0, 0.0],
        nodes[1].uri: [0.8, 0.6],
        nodes[2].uri: [0.28, 0.96],
    }

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            neighbor_k=2,
            similarity_threshold=0.7,
            strong_pair_threshold=0.95,
            min_candidate_size=3,
            min_quality=0.7,
        ),
    )

    assert candidates == []


def test_discovers_minimum_sized_episodic_candidate_from_shared_source():
    nodes = [_node(index) for index in range(3)]
    for node in nodes:
        node.metadata = {"provenance_ids": [f"prov:1:archive:source-7:msg-{node.uri[-1]}"]}
    embeddings = {
        nodes[0].uri: [1.0, 0.0, 0.0],
        nodes[1].uri: [0.0, 1.0, 0.0],
        nodes[2].uri: [0.0, 0.0, 1.0],
    }

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            min_candidate_size=3,
            similarity_threshold=0.95,
            strong_pair_threshold=0.99,
            min_quality=0.8,
        ),
    )

    assert len(candidates) == 1
    assert candidates[0].basis == "episodic"
    assert len(candidates[0].members) == 3
    assert candidates[0].source_archive_ids == ("source-7",)


def test_discovers_entity_anchor_candidate_from_explicit_mentions():
    melanie = ContextNode(
        uri="ctx://acct/users/alice/memories/entity/melanie",
        context_type="MEMORY",
        category="entity",
        level=0,
        owner_space="user:alice",
        abstract="Melanie is a parent and musician.",
        overview="",
        content="Melanie profile",
    )
    museum = ContextNode(
        uri="ctx://acct/users/alice/memories/event/museum_visit",
        context_type="MEMORY",
        category="event",
        level=0,
        owner_space="user:alice",
        abstract="Melanie took her children to a museum.",
        overview="",
        content="A dinosaur exhibit delighted the family.",
    )
    pottery = ContextNode(
        uri="ctx://acct/users/alice/memories/preference/pottery",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:alice",
        abstract="Melanie enjoys pottery with her children.",
        overview="",
        content="They make bowls together.",
    )
    noise = ContextNode(
        uri="ctx://acct/users/alice/memories/event/alice_trip",
        context_type="MEMORY",
        category="event",
        level=0,
        owner_space="user:alice",
        abstract="Alice visited a beach.",
        overview="",
        content="Alice traveled alone.",
    )
    nodes = [melanie, museum, pottery, noise]
    embeddings = {
        melanie.uri: [1.0, 0.0, 0.0],
        museum.uri: [0.0, 1.0, 0.0],
        pottery.uri: [0.0, 0.0, 1.0],
        noise.uri: [-1.0, 0.0, 0.0],
    }

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            min_candidate_size=3,
            similarity_threshold=0.99,
            strong_pair_threshold=0.99,
        ),
    )

    entity_candidates = [scene for scene in candidates if scene.basis == "entity_anchor"]
    assert len(entity_candidates) == 1
    assert {member.context_uri for member in entity_candidates[0].members} == {
        melanie.uri,
        museum.uri,
        pottery.uri,
    }
    assert (
        next(
            member for member in entity_candidates[0].members if member.context_uri == melanie.uri
        ).role
        == "anchor"
    )


def test_discovery_isolates_clusters_by_owner_space():
    nodes = [
        _node(0, "user:alice"),
        _node(1, "user:alice"),
        _node(2, "user:bob"),
        _node(3, "user:bob"),
    ]
    embeddings = {node.uri: [1.0, 0.01 * index] for index, node in enumerate(nodes)}

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(min_candidate_size=3, strong_pair_threshold=0.95),
    )

    assert len(candidates) == 2
    for scene in candidates:
        member_uris = {member.context_uri for member in scene.members}
        member_owners = {node.owner_space for node in nodes if node.uri in member_uris}
        assert member_owners == {scene.owner_space}


def test_discovery_rejects_mixed_embedding_dimensions():
    nodes = [_node(index) for index in range(3)]
    for node in nodes:
        node.metadata = {"provenance_ids": ["prov:1:archive:source-7:message"]}
    embeddings = {
        nodes[0].uri: [1.0, 0.0],
        nodes[1].uri: [0.0, 1.0, 0.0],
        nodes[2].uri: [0.0, 1.0],
    }

    with pytest.raises(ValueError, match=rf"{nodes[1].uri}.*expected 2, got 3"):
        generate_scene_candidates(nodes, embeddings)


def test_discovery_rejects_non_finite_embeddings():
    nodes = [_node(index) for index in range(2)]
    embeddings = {nodes[0].uri: [1.0, 0.0], nodes[1].uri: [float("nan"), 1.0]}

    with pytest.raises(ValueError, match=rf"{nodes[1].uri}.*finite numeric values"):
        generate_scene_candidates(nodes, embeddings)


@pytest.mark.parametrize("structured", [False, True])
def test_discovers_entity_candidate_from_continuous_chinese_mentions(structured):
    nodes = [_node(index) for index in range(3)]
    nodes[0].uri = "ctx://acct/users/alice/memories/entity/梅兰妮"
    nodes[0].category = "entity"
    nodes[0].abstract = "梅兰妮的个人资料"
    nodes[1].abstract = "梅兰妮带孩子参观博物馆"
    nodes[2].abstract = "周末梅兰妮和孩子一起制作陶器"
    if structured:
        for node in nodes[1:]:
            node.metadata = {"who": "梅兰妮"}
    embeddings = {node.uri: [float(i == j) for j in range(3)] for i, node in enumerate(nodes)}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == "entity_anchor"
    assert {member.context_uri for member in candidates[0].members} == {node.uri for node in nodes}
    assert (
        next(member for member in candidates[0].members if member.role == "anchor").context_uri
        == nodes[0].uri
    )


@pytest.mark.parametrize("max_candidate_size", [2, 3])
def test_episodic_candidate_preserves_all_shared_archives(max_candidate_size):
    nodes = [_node(index) for index in range(3)]
    for node in nodes:
        node.metadata = {
            "provenance_ids": [
                ProvenanceResolver.build_id("archive", "source-a", ["message"]),
                ProvenanceResolver.build_id("archive", "source-b", ["message"]),
            ]
        }
    embeddings = {node.uri: [float(i == j) for j in range(3)] for i, node in enumerate(nodes)}

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            max_candidate_size=max_candidate_size, min_candidate_size=min(3, max_candidate_size)
        ),
    )

    assert len(candidates) == (2 if max_candidate_size == 2 else 1)
    assert {m.context_uri for c in candidates for m in c.members} == {n.uri for n in nodes}
    assert all(c.basis == "episodic" and len(c.members) == max_candidate_size for c in candidates)
    assert all(c.source_archive_ids == ("source-a", "source-b") for c in candidates)


def test_candidate_summary_uses_three_available_abstracts():
    nodes = [_node(index) for index in range(6)]
    for node in nodes[:3]:
        node.abstract = "  "
    embeddings = {node.uri: [1.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].summary_hint == " | ".join(node.abstract for node in nodes[3:])


@pytest.mark.parametrize("basis", ["episodic", "entity_anchor"])
def test_zero_vectors_preserve_contextual_candidate_members(basis):
    nodes = [_node(index) for index in range(3)]
    if basis == "episodic":
        for node in nodes:
            node.metadata = {
                "provenance_ids": [ProvenanceResolver.build_id("archive", "shared", ["message"])]
            }
    else:
        nodes[0].uri = "ctx://acct/users/alice/memories/entity/melanie"
        nodes[0].category = "entity"
        for node in nodes:
            node.abstract = "Melanie plans a family trip."
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == basis
    assert {member.context_uri for member in candidates[0].members} == {node.uri for node in nodes}


@pytest.mark.parametrize("max_candidate_size", [1, 2, 3, 4])
def test_contextual_candidates_respect_member_capacity(max_candidate_size):
    nodes = [_node(index) for index in range(6)]
    nodes[0].uri = "ctx://acct/users/alice/memories/entity/melanie"
    nodes[0].category = "entity"
    for node in nodes[:4]:
        node.abstract = "Melanie plans a family trip."
    for node in nodes[4:]:
        node.metadata = {
            "provenance_ids": [ProvenanceResolver.build_id("archive", "shared", ["message"])]
        }
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(
        nodes,
        embeddings,
        CandidateConfig(
            max_candidate_size=max_candidate_size, min_candidate_size=min(3, max_candidate_size)
        ),
    )

    assert len(candidates) == (2 if max_candidate_size == 1 else 1) + int(max_candidate_size >= 3)
    assert all(len(scene.members) <= max_candidate_size for scene in candidates)
    episodic = next(scene for scene in candidates if scene.basis == "episodic")
    assert len(episodic.members) == min(2, max_candidate_size)


@pytest.mark.parametrize(
    ("slug", "mention"),
    [
        ("mary_jane", "Mary Jane"),
        ("open-gauss", "Open-Gauss"),
        ("open-gauss", "Open Gauss"),
        ("li_wei", "Li Wei"),
    ],
)
def test_multiword_entities_group_explicit_mentions(slug, mention):
    nodes = [_node(index) for index in range(3)]
    nodes[0].uri = f"ctx://acct/users/alice/memories/entity/{slug}"
    nodes[0].category = "entity"
    for node in nodes:
        node.abstract = f"{mention} participates in this event."
    embeddings = {node.uri: [float(i == j) for j in range(3)] for i, node in enumerate(nodes)}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == "entity_anchor"
    assert {member.context_uri for member in candidates[0].members} == {node.uri for node in nodes}
    assert (
        next(member for member in candidates[0].members if member.role == "anchor").context_uri
        == nodes[0].uri
    )


@pytest.mark.parametrize("neighbor_k", [1, 3, 32])
def test_streamed_neighbors_match_exact_ranking(neighbor_k):
    nodes = [_node(index) for index in range(8)]
    vectors = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.6), (0.0, 1.0)] * 2
    indices = {0, 1, 2, 4, 5, 7}

    nearest = _nearest_neighbors(
        nodes, _pairwise_similarities(vectors, indices), indices, neighbor_k
    )

    for index in indices:
        ranked = sorted(
            (other for other in indices if other != index),
            key=lambda other: (cosine(vectors[index], vectors[other]), nodes[other].uri),
            reverse=True,
        )[:neighbor_k]
        assert list(nearest[index]) == ranked
        assert nearest[index] == {other: cosine(vectors[index], vectors[other]) for other in ranked}


def test_each_owner_uses_its_full_input_budget():
    nodes = [_node(index, f"user:{index // 4}") for index in range(8)]
    embeddings = {node.uri: [1.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings, CandidateConfig(max_owner_nodes=4))

    assert len(candidates) == 2
    for candidate in candidates:
        assert {member.context_uri for member in candidate.members} == {
            node.uri for node in nodes if node.owner_space == candidate.owner_space
        }


@pytest.mark.parametrize("archive_id", ["session:archive", "folder/session 1", "会话/%3A"])
@pytest.mark.parametrize("basis", ["semantic", "episodic"])
def test_candidates_preserve_decoded_archive_identity(archive_id, basis):
    nodes = [_node(index) for index in range(3)]
    for node in nodes:
        node.metadata = {"provenance_ids": [ProvenanceResolver.build_id("archive", archive_id)]}
    embeddings = {
        node.uri: [1.0, 0.0, 0.0] if basis == "semantic" else [float(i == j) for j in range(3)]
        for i, node in enumerate(nodes)
    }

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == basis
    assert candidates[0].source_archive_ids == (archive_id,)


@pytest.mark.parametrize("count,neighbor_k,capacity", [(20, 8, 12), (32, 1, 8), (36, 3, 12)])
def test_dense_semantic_groups_cover_members_in_bounded_candidates(count, neighbor_k, capacity):
    nodes = [_node(index) for index in range(count)]
    embeddings = {node.uri: [1.0, 0.0] for node in nodes}
    config = CandidateConfig(neighbor_k=neighbor_k, max_candidate_size=capacity)

    candidates = generate_scene_candidates(nodes, embeddings, config)

    assert {member.context_uri for c in candidates for member in c.members} == {
        n.uri for n in nodes
    }
    assert sum(len(c.members) for c in candidates) == count
    assert all(3 <= len(c.members) <= capacity for c in candidates)
    assert all(c.basis == "semantic" and c.quality == 1.0 for c in candidates)
    reversed_candidates = generate_scene_candidates(list(reversed(nodes)), embeddings, config)
    assert [c.source_fingerprint for c in candidates] == [
        c.source_fingerprint for c in reversed_candidates
    ]


@pytest.mark.parametrize("basis", ["semantic", "episodic"])
@pytest.mark.parametrize("count", [13, 25])
def test_capacity_remainders_preserve_all_evidence(basis, count):
    nodes = [_node(i) for i in range(count)]
    for node in nodes:
        if basis == "episodic":
            node.metadata = {"provenance_ids": [ProvenanceResolver.build_id("archive", "shared")]}
    embeddings = {
        node.uri: [1.0, 0.0] if basis == "semantic" else [float(i == j) for j in range(count)]
        for i, node in enumerate(nodes)
    }
    candidates = generate_scene_candidates(nodes, embeddings)
    assert {m.context_uri for c in candidates for m in c.members} == {n.uri for n in nodes}
    assert all(c.basis == basis and 3 <= len(c.members) <= 12 for c in candidates)
    assert sum(len(c.members) for c in candidates) == count
    assert generate_scene_candidates(list(reversed(nodes)), embeddings) == candidates


@pytest.mark.parametrize("structured", [False, True])
def test_entity_uses_preserved_actor_name(structured):
    from commit.archive_builder import ArchiveBuilder
    from core.models import CandidateMemory, RequestContext, WritePlan
    from core.validation import validate_candidate
    from providers.llm.mock_llm import MockLLM

    name = "林晓"
    extracted = validate_candidate(
        CandidateMemory(
            category="entity",
            owner_scope="user",
            routing_key=name,
            abstract=f"{name}的个人资料",
            overview="",
            content=f"{name}是一名设计师。",
            confidence=1.0,
            who=name,
        )
    )
    assert extracted.routing_key == name
    nodes = [_node(i) for i in range(3)]
    nodes[0] = ArchiveBuilder(MockLLM()).build(
        extracted,
        WritePlan(
            action="create",
            target_uri=f"ctx://acct/users/alice/memories/entity/{extracted.routing_key}",
        ),
        RequestContext(
            account_id="acct",
            user_id="alice",
            agent_id="agent",
            session_id="session",
            trace_id="trace",
        ),
    )
    nodes[1].abstract = f"{name}参加设计展览。"
    nodes[2].abstract = f"昨天{name}提交了设计方案。"
    if structured:
        for node in nodes[1:]:
            node.metadata = {"who": name}
    embeddings = {node.uri: [float(i == j) for j in range(3)] for i, node in enumerate(nodes)}
    candidates = generate_scene_candidates(nodes, embeddings)
    assert len(candidates) == 1
    assert candidates[0].basis == "entity_anchor"
    assert {m.context_uri for m in candidates[0].members} == {n.uri for n in nodes}


@pytest.mark.parametrize("capacity,count", [(6, 7), (8, 17), (12, 13)])
def test_semantic_rebalancing_preserves_separate_topics(capacity, count):
    nodes = [_node(i) for i in range(count * 2)]
    embeddings = {node.uri: [float(i < count), float(i >= count)] for i, node in enumerate(nodes)}
    candidates = generate_scene_candidates(
        nodes, embeddings, CandidateConfig(max_candidate_size=capacity)
    )
    assert {m.context_uri for c in candidates for m in c.members} == {n.uri for n in nodes}
    assert sum(len(c.members) for c in candidates) == len(nodes)
    for candidate in candidates:
        assert 3 <= len(candidate.members) <= capacity
        assert candidate.quality == 1.0
        assert len({tuple(embeddings[m.context_uri]) for m in candidate.members}) == 1


@pytest.mark.parametrize("capacity,minimum,count", [(2, 2, 3), (3, 3, 4), (5, 5, 6), (8, 4, 17)])
def test_archive_groups_cover_evidence_with_configured_bounds(capacity, minimum, count):
    nodes = [_node(i) for i in range(count)]
    for node in nodes:
        node.metadata = {"provenance_ids": [ProvenanceResolver.build_id("archive", "shared")]}
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}
    candidates = generate_scene_candidates(
        nodes, embeddings, CandidateConfig(max_candidate_size=capacity, min_candidate_size=minimum)
    )
    assert {m.context_uri for c in candidates for m in c.members} == {n.uri for n in nodes}
    assert all(minimum <= len(c.members) <= capacity for c in candidates)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("quality", [0.0, 1.0])
def test_config_score_boundaries_produce_candidates(threshold, quality):
    nodes = [_node(i) for i in range(2)]
    embeddings = {node.uri: [1.0, 0.0] for node in nodes}
    config = CandidateConfig(
        neighbor_k=1,
        min_candidate_size=2,
        max_candidate_size=2,
        max_owner_nodes=2,
        entity_candidate_max_chunks=1,
        similarity_threshold=threshold,
        strong_pair_threshold=threshold,
        min_quality=quality,
    )
    candidates = generate_scene_candidates(nodes, embeddings, config)
    assert len(candidates) == 1
    assert {m.context_uri for m in candidates[0].members} == {node.uri for node in nodes}
    assert candidates[0].quality == 1.0


@pytest.mark.parametrize("basis", ["semantic", "episodic", "entity_anchor"])
@pytest.mark.parametrize(
    "owner_type,owner_space", [("users", "user:alice"), ("agents", "agent:alice")]
)
def test_accounts_have_independent_candidates_and_input_budgets(basis, owner_type, owner_space):
    nodes = []
    for account in ("acct-a", "acct-b"):
        for index in range(3):
            node = _node(index, owner_space)
            node.uri = f"ctx://{account}/{owner_type}/alice/memories/event/{index}"
            node.abstract = f"Melanie plans a trip for {account}."
            if basis == "episodic":
                node.metadata = {
                    "provenance_ids": [ProvenanceResolver.build_id("archive", "shared")]
                }
            elif basis == "entity_anchor" and index == 0:
                node.uri = f"ctx://{account}/{owner_type}/alice/memories/entity/melanie"
                node.category = "entity"
            nodes.append(node)
    embeddings = {node.uri: [float(basis == "semantic"), 0.0] for node in nodes}
    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 2
    for candidate, account in zip(candidates, ("acct-a", "acct-b")):
        assert candidate.account_id == account
        assert candidate.owner_space == owner_space
        assert candidate.basis == basis
        assert {member.context_uri for member in candidate.members} == {
            node.uri for node in nodes if node.uri.startswith(f"ctx://{account}/")
        }
    assert len({candidate.source_fingerprint for candidate in candidates}) == 2
    assert (
        generate_scene_candidates(
            list(reversed(nodes)), embeddings, CandidateConfig(max_owner_nodes=3)
        )
        == candidates
    )


@pytest.mark.parametrize("count,expected_sizes", [(5, [5]), (7, [5]), (9, [5, 5])])
def test_entity_candidates_apply_configured_minimum_to_each_chunk(count, expected_sizes):
    nodes = [_node(index) for index in range(count)]
    nodes[0].uri = "ctx://acct/users/alice/memories/entity/melanie"
    nodes[0].category = "entity"
    for node in nodes:
        node.abstract = "Melanie plans a trip."
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(
        nodes, embeddings, CandidateConfig(min_candidate_size=5, max_candidate_size=5)
    )

    assert sorted(len(candidate.members) for candidate in candidates) == expected_sizes
    assert all(candidate.basis == "entity_anchor" for candidate in candidates)
    assert all(
        next(member.context_uri for member in candidate.members if member.role == "anchor")
        == nodes[0].uri
        for candidate in candidates
    )


@pytest.mark.parametrize("neighbor_k", [1, 8])
def test_dense_owner_reuses_pair_scores_across_capacity_rounds(monkeypatch, neighbor_k):
    import scene.discovery as discovery

    count = 512
    nodes = [_node(index) for index in range(count)]
    embeddings = {node.uri: [1.0] * 8 for node in nodes}
    calls = 0

    def counted_cosine(left, right):
        nonlocal calls
        calls += 1
        return cosine(left, right)

    monkeypatch.setattr(discovery, "cosine", counted_cosine)
    candidates = generate_scene_candidates(
        nodes, embeddings, CandidateConfig(neighbor_k=neighbor_k)
    )

    assert {member.context_uri for candidate in candidates for member in candidate.members} == {
        node.uri for node in nodes
    }
    assert sum(len(candidate.members) for candidate in candidates) == count
    assert all(3 <= len(candidate.members) <= 12 for candidate in candidates)
    assert calls < count * count


def test_cached_neighbors_preserve_exact_ranking_for_active_subsets():
    nodes = [_node(index) for index in range(8)]
    vectors = [(1.0, 0.0), (0.8, 0.6), (0.8, 0.6), (0.0, 1.0)] * 2
    scores = _pairwise_similarities(vectors, set(range(len(nodes))))

    assert sum(row.itemsize * len(row) for row in scores.values()) == 8 * len(nodes) ** 2
    for indices in ({0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 5, 7}, {1, 2, 7}):
        nearest = _nearest_neighbors(nodes, scores, indices, 2)
        for index in indices:
            expected = sorted(
                (other for other in indices if other != index),
                key=lambda other: (cosine(vectors[index], vectors[other]), nodes[other].uri),
                reverse=True,
            )[:2]
            assert list(nearest[index]) == expected
            assert nearest[index] == {
                other: cosine(vectors[index], vectors[other]) for other in expected
            }


@pytest.mark.parametrize(
    "name_field", ["who", "attributed_speaker", "routing_key", "subject", "text", "continuous"]
)
def test_chinese_entities_group_evidence_by_complete_name(name_field):
    nodes = []
    expected_members = []
    for name in ("李雷", "李雷鸣"):
        group = [_node(index) for index in range(3)]
        for index, node in enumerate(group):
            category = "entity" if index == 0 else "event"
            node.uri = f"ctx://acct/users/alice/memories/{category}/{name}-{index}"
            node.category = category
            node.metadata = {"entity_name": name} if index == 0 else {}
            node.abstract = f"{name}参加会议"
            if index:
                if name_field == "text":
                    node.abstract = f"参会者：{name}，讨论项目。"
                elif name_field != "continuous":
                    node.metadata[name_field] = name
        nodes.extend(group)
        expected_members.append({node.uri for node in group})
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 2
    assert all(candidate.basis == "entity_anchor" for candidate in candidates)
    assert {
        frozenset(member.context_uri for member in candidate.members) for candidate in candidates
    } == {frozenset(members) for members in expected_members}


@pytest.mark.parametrize("name_field", ["who", "attributed_speaker", "routing_key", "subject"])
def test_entity_mentions_preserve_metadata_prose(name_field):
    nodes = [_node(index) for index in range(3)]
    nodes[0].uri = "ctx://acct/users/alice/memories/entity/melanie"
    nodes[0].category = "entity"
    for node in nodes[1:]:
        node.metadata = {name_field: "Melanie and her family"}
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == "entity_anchor"
    assert {member.context_uri for member in candidates[0].members} == {node.uri for node in nodes}


@pytest.mark.parametrize(
    "name,sentences",
    [
        ("李雷", ["昨天李雷提交了方案", "今天李雷参加会议"]),
        ("田中", ["昨日田中さんが提案しました", "今日田中さんが会議に参加しました"]),
        ("민수", ["어제민수가제안했습니다", "오늘민수가참석했습니다"]),
    ],
)
@pytest.mark.parametrize("field", ["abstract", "overview", "content"])
def test_continuous_cjk_mentions_cover_text_fields(name, sentences, field):
    nodes = [_node(index) for index in range(3)]
    nodes[0].uri = f"ctx://acct/users/alice/memories/entity/{name}"
    nodes[0].category = "entity"
    for node, sentence in zip(nodes[1:], sentences):
        setattr(node, field, sentence)
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert candidates[0].basis == "entity_anchor"
    assert {member.context_uri for member in candidates[0].members} == {node.uri for node in nodes}


def test_continuous_text_preserves_each_complete_name_in_shared_events():
    nodes = [_node(index) for index in range(4)]
    for node, name in zip(nodes[:2], ("李雷", "李雷鸣")):
        node.uri = f"ctx://acct/users/alice/memories/entity/{name}"
        node.category = "entity"
    for node in nodes[2:]:
        node.abstract = "昨天李雷鸣和李雷一起提交了方案"
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 2
    assert {
        frozenset(member.context_uri for member in candidate.members) for candidate in candidates
    } == {frozenset([anchor.uri, nodes[2].uri, nodes[3].uri]) for anchor in nodes[:2]}
    assert generate_scene_candidates(list(reversed(nodes)), embeddings) == candidates


def test_structured_long_name_keeps_short_name_evidence_group_precise():
    nodes = [_node(index) for index in range(5)]
    nodes[0].uri = "ctx://acct/users/alice/memories/entity/李雷"
    nodes[0].category = "entity"
    for node in nodes[1:3]:
        node.abstract = "昨天李雷提交了方案"
    for node in nodes[3:]:
        node.abstract = "今天李雷鸣参加会议"
        node.metadata = {"who": "李雷鸣"}
    embeddings = {node.uri: [0.0, 0.0] for node in nodes}

    candidates = generate_scene_candidates(nodes, embeddings)

    assert len(candidates) == 1
    assert {member.context_uri for member in candidates[0].members} == {
        node.uri for node in nodes[:3]
    }
