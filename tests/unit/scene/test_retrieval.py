import pytest

from core.models import SeedHit
from scene.config import SceneConfig
from scene.retrieval import SceneHit, SceneRanker


def test_scene_boosts_members_but_preserves_direct_evidence_ordering():
    direct = [
        SeedHit(uri="a", score=0.95, owner_space="user:a"),
        SeedHit(uri="b", score=0.94, owner_space="user:a"),
    ]
    result = SceneRanker().rank(
        direct,
        [SceneHit("s1", 0.90, "user:a", ("b",), "Atlas release", "release planning")],
        owner_spaces={"user:a"},
    )
    assert result.hits[0].uri == "b"
    assert result.hits[0].metadata["scene_ids"] == ("s1",)


@pytest.mark.parametrize("suffix", ["", "/", "/content.md", "/.abstract.md", "/.overview.md"])
def test_scene_member_boost_preserves_memory_projection_uri(suffix):
    uri = "ctx://acct/users/u1/memories/events/release"
    hit = SeedHit(uri=uri + suffix, score=0.8, owner_space="user:u1")
    result = SceneRanker().rank(
        [hit], [SceneHit("release-scene", 0.9, "user:u1", (uri,))],
        owner_spaces={"user:u1"},
    )
    boosted = result.hits[0]
    assert boosted.uri == hit.uri
    assert boosted.score == pytest.approx(0.836)
    assert boosted.metadata["scene_match"] is True
    assert boosted.metadata["scene_ids"] == ("release-scene",)


def test_scene_boost_keeps_independent_agent_memory_units():
    uri = "ctx://acct/agents/a1/memories/experiences/release"
    hits = [
        SeedHit(uri=uri + "/content.md", score=0.8, owner_space="agent:a1",
                metadata={"agent_memory_unit_key": unit})
        for unit in ("diagnosis", "recovery")
    ]
    result = SceneRanker().rank(
        hits, [SceneHit("release-scene", 0.9, "agent:a1", (uri,))],
        owner_spaces={"agent:a1"},
    )
    assert [hit.metadata["agent_memory_unit_key"] for hit in result.hits] == ["diagnosis", "recovery"]
    assert [hit.score for hit in result.hits] == pytest.approx([0.836, 0.836])
    assert all(hit.metadata["scene_match"] for hit in result.hits)


def test_unmatched_direct_hit_keeps_original_score():
    direct = [SeedHit(uri="a", score=0.42, owner_space="user:a")]
    result = SceneRanker().rank(
        direct,
        [SceneHit("s1", 0.90, "user:a", ("other",))],
        owner_spaces={"user:a"},
    )
    assert result.hits[0].score == 0.42
    assert result.hits[0].metadata == {}


def test_scene_ranking_scales_with_direct_rrf_scores():
    scenes = [SceneHit("s1", 0.9, "user:a", ("b",))]
    ranker = SceneRanker()
    large = ranker.rank([SeedHit("a", 0.95), SeedHit("b", 0.94)], scenes)
    small = ranker.rank([SeedHit("a", 0.0095), SeedHit("b", 0.0094)], scenes)
    assert [hit.uri for hit in small.hits] == [hit.uri for hit in large.hits]
    assert [hit.score * 100 for hit in small.hits] == pytest.approx([hit.score for hit in large.hits])


def test_scene_expansion_reserves_direct_evidence_on_rrf_scale():
    direct = [SeedHit("direct", 0.03),
              SeedHit("member", 0.95, metadata={"scene_expanded": True})]
    result = SceneRanker().rank(direct, [SceneHit("s1", 0.95, "user:a", ("member",))])
    assert result.hits[0].uri == "direct"
    assert result.hits[1].score < 0.03


@pytest.mark.parametrize("route", ["member", "lexical"])
def test_expansion_occupies_the_configured_context_budget(route):
    direct = [SeedHit(f"direct-{i}", 1 - i / 100) for i in range(20)]
    expanded = SeedHit("rollback", .01, metadata={"scene_expanded": True})
    hit = SceneHit("deployment", 1, "user:alice", ("rollback",),
                   query_threshold=0, routes=(route,))
    result = SceneRanker().rank([*direct, expanded], [hit], limit=10)
    assert len(result.hits) == 10
    assert [h.uri for h in result.hits[:9]] == [h.uri for h in direct[:9]]
    assert result.hits[9].uri == "rollback"
    assert result.hits[9].metadata["scene_ids"] == ("deployment",)


@pytest.mark.parametrize("owner,visible", [
    ("user:alice", "user_alice"), ("user_alice", "user:alice"),
    ("agent:bot", "agent_bot"), ("agent_bot", "agent:bot"),
])
def test_scene_owner_aliases_preserve_hit_fields(owner, visible):
    hit = SeedHit("member", 0.8, owner_space=owner, level=2, parent_uri="parent",
                  abstract="details", category="event", has_content=True,
                  metadata={"source": "canonical"})
    scene = SceneHit("scene", 0.9, owner, ("member",))
    ranked = SceneRanker().rank([hit], [scene], owner_spaces={visible}).hits[0]
    assert (ranked.level, ranked.parent_uri, ranked.abstract, ranked.category, ranked.has_content) == (
        2, "parent", "details", "event", True,
    )
    assert ranked.metadata["source"] == "canonical"
    assert ranked.metadata["scene_match"] is True
    assert hit.score == 0.8
    assert hit.metadata == {"source": "canonical"}


def test_custom_reserve_and_member_decay_distribute_evidence_slots():
    config = SceneConfig(direct_evidence_reserve=0.6, member_decay=0.5)
    direct = [SeedHit(f"direct-{i}", 1.0) for i in range(6)]
    expanded = [SeedHit(f"member-{i}", 0.1, metadata={"scene_expanded": True}) for i in range(3)]
    scene = SceneHit("scene", 0.5, "user:alice", tuple(h.uri for h in expanded),
                     query_threshold=0.5, routes=("member",))
    result = SceneRanker(config).rank([*direct, *expanded], [scene], limit=5)
    assert [h.uri for h in result.hits] == ["direct-0", "direct-1", "direct-2", "member-0", "member-1"]
    assert [h.metadata["scene_score"] for h in result.hits[3:]] == [0.25, 0.125]
