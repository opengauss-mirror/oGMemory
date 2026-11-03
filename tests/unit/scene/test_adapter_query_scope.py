"""Request configuration and typed candidate selection through the scene adapter."""

from dataclasses import replace

import pytest

from core.models import SeedHit, TypedQuery
from scene.adapters import _scope_context
from scene.candidate_index import LexicalSceneIndex
from scene.config import InMemorySceneConfigProvider, SceneConfig
from scene.models import SceneSnapshot
from scene.retrieval import SceneRanker
from tests.unit.scene.test_adapters import adapter, node, scene
from tests.unit.scene.test_candidate_index import MemoryIndex

CTX = _scope_context("acme", "user:alice")


@pytest.mark.parametrize("visible", [(), ("user:alice", "agent:helper"), ("user_alice", "agent_helper")])
def test_default_request_pins_user_and_agent_configurations(visible):
    user, agent = node("release"), node("tool", owner="agent:helper", category="tool")
    provider = InMemorySceneConfigProvider()
    source = adapter([user, agent], config_provider=provider)
    snapshots = {item.owner_space: SceneSnapshot("acme", item.owner_space, 0, (
        scene(item.category, [item], centroid=(0.8, 0.6)),
    )) for item in (user, agent)}
    source.repository.load_snapshot = lambda account, owner: snapshots[owner]
    ctx = replace(CTX, agent_id="helper", visible_owner_spaces=visible)
    request = source.for_request(ctx, [])
    provider.publish("acme", "agent:helper", 0, {"query_threshold": 0.9})
    hits = request.search("release", ctx, limit=2, query_vector=[1, 0])
    assert {hit.owner_space for hit in hits} == {"user:alice", "agent:helper"}
    next_hits = source.for_request(ctx, []).search("release", ctx, limit=2, query_vector=[1, 0])
    assert [hit.owner_space for hit in next_hits] == ["user:alice"]


@pytest.mark.parametrize("context_type", ["MEMORY", "SKILL", "RESOURCE"])
@pytest.mark.parametrize("entry", ["search", "resolve"])
def test_typed_query_ranks_matching_direct_and_scene_evidence(context_type, entry):
    item = node("release")
    source = adapter([item], [scene("release", [item])])
    query = TypedQuery("release", context_type, [])
    memory_scenes = source.search(query.text, CTX, limit=1, query_vector=[1, 0])
    hits = (source.search(query.text, CTX, limit=1, query_vector=[1, 0], typed_query=query)
            if entry == "search" else memory_scenes)
    members = source.resolve_members(hits, CTX, limit=1, query_vector=[1, 0], typed_query=query)
    direct = SeedHit("ctx://acme/direct", 1.0, context_type=context_type, owner_space="user:alice")
    ranked = SceneRanker().rank([direct, *members], hits, limit=2)
    expected = [direct.uri, item.uri] if context_type == "MEMORY" else [direct.uri]
    assert [hit.uri for hit in ranked.hits] == expected
    assert {hit.context_type for hit in ranked.hits} == {context_type}
    if entry == "search":
        assert len(ranked.matched_scenes) == int(context_type == "MEMORY")


@pytest.mark.parametrize("route", ["semantic", "lexical", "member"])
@pytest.mark.parametrize("filter_name", ["category", "directory", "target"])
def test_scene_limit_selects_matching_member_query(route, filter_name):
    preference, event = node("style", category="preference"), node("release")
    centroid = (1, 0) if route == "semantic" else (0, 1)
    scenes = [scene("a release", [preference], centroid=centroid),
              scene("b release", [event], centroid=centroid)]
    source = adapter([preference, event], scenes, config=replace(SceneConfig(), lexical_enabled=True))
    if route == "lexical":
        source.candidate_index = LexicalSceneIndex(MemoryIndex())
        source.candidate_index.publish(source.repository.snapshot)
    query = TypedQuery("release", "MEMORY", ["event"] if filter_name == "category" else [],
                       memory_directories=["events"] if filter_name == "directory" else None,
                       target_uri=event.uri if filter_name == "target" else None)
    direct = [SeedHit(item.uri, score, owner_space=item.owner_space)
              for item, score in ((preference, 2.0), (event, 1.0))]
    hits = source.search_with_evidence(
        query.text, CTX, limit=1, query_vector=[1, 0], typed_query=query,
        direct_hits=direct if route == "member" else [],
    )
    members = source.resolve_members(hits, CTX, limit=1, query_vector=[1, 0], typed_query=query)
    assert [member.uri for member in members] == [event.uri]
    assert [hit.scene_id for hit in hits] == ["b release"]
    assert hits[0].routes == (route,)
