"""Scene candidates recover with current canonical and indexed evidence."""

from dataclasses import replace

import pytest

from core.models import SeedHit
from index.index_record_builder import build_index_records
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from scene.adapters import _scope_context
from scene.candidate_index import LexicalSceneIndex
from scene.config import SceneConfig
from scene.retrieval import SceneRanker
from tests.unit.scene.test_adapters import adapter, node, scene
from tests.unit.scene.test_candidate_index import MemoryIndex

CTX = _scope_context("acme", "user:alice")


def index_node(index, item):
    record = build_index_records(item)[0]
    record.metadata["_embedding"] = [1.0, 0.0]
    index.upsert([record])


def test_ranking_uses_current_scene_members_and_recovers_refreshed_membership():
    first, second = node("a"), node("b")
    current = replace(first, abstract="Updated evidence", metadata={"version": 2})
    source = adapter([current, second], [scene("release", [first, second])])
    direct = [SeedHit(first.uri, 1.0, owner_space="user:alice"),
              SeedHit(second.uri, 0.99, owner_space="user:alice")]
    hits = source.search("release", CTX, limit=1, query_vector=[1, 0])
    ranked = SceneRanker().rank(direct, hits)
    assert [hit.uri for hit in ranked.hits] == [second.uri, first.uri]
    assert ranked.hits[1].score == direct[0].score
    assert ranked.hits[0].metadata["scene_match"] is True
    assert hits[0].member_uris == (second.uri,)
    assert hits[0].member_versions == ((second.uri, 1),)
    record = source.repository.snapshot.scenes[0]
    members = (replace(record.cluster.members[0], node_version=2), record.cluster.members[1])
    source.repository.snapshot = replace(source.repository.snapshot, scenes=(
        replace(record, cluster=replace(record.cluster, members=members)),
    ))
    refreshed = source.search("release", CTX, limit=1, query_vector=[1, 0])
    ranked = SceneRanker().rank(direct, refreshed)
    assert refreshed[0].member_versions == ((first.uri, 2), (second.uri, 1))
    assert all(hit.metadata["scene_match"] for hit in ranked.hits)


@pytest.mark.parametrize("route", ["semantic", "lexical", "member"])
@pytest.mark.parametrize("pending", ["missing", "outdated_text"])
def test_indexed_scene_fills_quota_and_pending_scene_recovers(route, pending):
    first, second = node("a"), node("b")
    centroid = (1, 0) if route == "semantic" else (0, 1)
    source = adapter([first, second], [scene("a release", [first], centroid=centroid),
                                      scene("b release", [second], centroid=centroid)],
                     config=replace(SceneConfig(), lexical_enabled=True))
    index = InMemoryVectorIndex(dimension=2)
    index_node(index, second)
    if pending == "outdated_text":
        index_node(index, replace(first, abstract="Older evidence"))
    source.vector_index = index
    if route == "lexical":
        source.candidate_index = LexicalSceneIndex(MemoryIndex())
        source.candidate_index.publish(source.repository.snapshot)
    direct = [SeedHit(first.uri, 1.0, owner_space="user:alice"),
              SeedHit(second.uri, 0.9, owner_space="user:alice")]

    def search():
        return source.search_with_evidence("release", CTX, limit=1, query_vector=[1, 0],
                                           direct_hits=direct if route == "member" else [])

    hits = search()
    members = source.resolve_members(hits, CTX, limit=1, query_vector=[1, 0])
    assert [member.uri for member in members] == [second.uri]
    assert hits[0].scene_id == "b release"
    assert hits[0].routes == (route,)
    index_node(index, first)
    recovered = search()
    assert [member.uri for member in source.resolve_members(recovered, CTX, limit=1)] == [first.uri]
    assert recovered[0].scene_id == "a release"
