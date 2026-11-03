from dataclasses import replace
from math import ceil
from types import SimpleNamespace

import pytest

from core.models import LeafHit, RetrievalConfig, SeedHit, SeedResult, TypedQuery
from index.index_record_builder import build_index_records
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.identity import canonical_uri
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.scene_stage import SceneRetrievalStage
from retrieval.seed_retriever import SeedRetriever
from retrieval.trace import RetrievalTrace
from scene.adapters import CanonicalSceneAdapter, _scope_context
from scene.config import InMemorySceneConfigProvider, SceneConfig
from scene.retrieval import SceneHit, SceneRanker
from tests.unit.scene.test_adapters import CanonicalFS, Embedder, Repository, node, scene


@pytest.fixture
def setup():
    direct, related = node("release"), node("follow-up")
    fs = CanonicalFS([direct, related])
    index = InMemoryVectorIndex(dimension=2)
    for item, vector in [(direct, [1.0, 0.0]), (related, [0.0, 1.0])]:
        records = build_index_records(item)
        for record in records:
            record.metadata["_embedding"] = vector
        index.upsert(records)
    embedder = Embedder()
    provider = InMemorySceneConfigProvider()
    source = CanonicalSceneAdapter(
        Repository((scene("release-plan", [direct, related], centroid=(0.0, 1.0)),)),
        fs, index, embedder, config_provider=provider,
    )
    config = RetrievalConfig(expand_hierarchy=False, hotness_alpha=0.0, default_score_threshold=0.75)
    pipeline = RetrievalPipeline(
        QueryPlanner(config), SeedRetriever(index, embedder, config), None,
        ResultRanker(config), config, context_reader=ContextReader(fs),
        scene_ranker=SceneRanker(), scene_source=source,
    )
    return SimpleNamespace(
        direct=direct, related=related, source=source, pipeline=pipeline,
        ctx=_scope_context("acme", "user:alice"), provider=provider,
    )


def search(setup, **kwargs):
    return setup.pipeline.run(
        "remember release", setup.ctx, top_k=10, context_type="MEMORY",
        fill_content_for_top_k=10, **kwargs,
    )


def evidence_uris(result):
    return {canonical_uri(hit.uri) for hit in result.hits}


def test_pipeline_adds_low_similarity_member_with_canonical_content(setup):
    result = search(setup)
    assert evidence_uris(result) == {setup.direct.uri, setup.related.uri}
    related = next(hit for hit in result.hits if hit.uri == setup.related.uri)
    assert related.metadata["scene_expanded"] is True
    assert related.metadata["scene_ids"] == ("release-plan",)
    assert related.content_excerpt == setup.related.content
    assert related.metadata["scene_member_similarity"] == 0.0
    assert setup.source.embedder.calls == ["remember release"]
    stage = next(stage for stage in result.trace.stages if stage.stage == "scene_ranking")
    assert (stage.input_count, stage.output_count) == (1, 2)


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_scene_switch_selects_expected_evidence(setup, enabled):
    result = search(setup, use_scene=enabled)
    expected = {setup.direct.uri, setup.related.uri} if enabled else {setup.direct.uri}
    assert evidence_uris(result) == expected


@pytest.mark.parametrize("failure", ["for_request", "search_with_evidence", "resolve_members"])
def test_scene_source_recovery_restores_expansion(setup, monkeypatch, failure):
    original = getattr(setup.source, failure)

    def unavailable(*args, **kwargs):
        raise RuntimeError("projection unavailable")

    monkeypatch.setattr(setup.source, failure, unavailable)
    degraded = search(setup)
    assert evidence_uris(degraded) == {setup.direct.uri}
    assert any("RuntimeError" in warning for warning in degraded.trace.warnings)
    monkeypatch.setattr(setup.source, failure, original)
    restored = search(setup)
    assert evidence_uris(restored) == {setup.direct.uri, setup.related.uri}
    assert restored.trace.warnings == []


def test_configuration_is_bound_once_and_next_request_uses_new_revision(setup, monkeypatch):
    bind = setup.source.for_request
    calls = []

    def counted(ctx, queries):
        calls.append(tuple(query.text for query in queries))
        return bind(ctx, queries)

    monkeypatch.setattr(setup.source, "for_request", counted)
    first = search(setup)
    config = setup.provider.get_effective("acme", "user:alice")
    setup.provider.publish("acme", "user:alice", config.revision, {"scene_boost": 0.9})
    second = search(setup)
    assert second.hits[0].score > first.hits[0].score
    assert calls == [("remember release",), ("remember release",)]


def test_canonical_refresh_restores_scene_member(setup):
    setup.source.fs.nodes[setup.related.uri].metadata["version"] = 2
    stale = search(setup)
    assert evidence_uris(stale) == {setup.direct.uri}
    snapshot = setup.source.repository.snapshot
    record = snapshot.scenes[0]
    members = tuple(replace(member, node_version=2) if member.context_uri == setup.related.uri
                    else member for member in record.cluster.members)
    setup.source.repository.snapshot = replace(
        snapshot, revision=1, scenes=(replace(record, cluster=replace(record.cluster, members=members)),)
    )
    fresh = search(setup)
    assert evidence_uris(fresh) == {setup.direct.uri, setup.related.uri}


@pytest.mark.parametrize("filters", [{"categories": ["event"]}, {"memory_directories": ["events"]}])
def test_scene_members_follow_pipeline_query_filters(setup, filters):
    result = search(setup, **filters)
    assert evidence_uris(result) == {setup.direct.uri, setup.related.uri}
    assert all(hit.category == "event" for hit in result.hits)


def test_empty_scene_result_keeps_direct_candidate_order_and_capacity():
    leaves = [LeafHit(f"ctx://acme/users/alice/memories/events/{i}", 1 - i / 10)
              for i in range(4)]
    source = SimpleNamespace(search=lambda *args, **kwargs: [])
    stage = SceneRetrievalStage(source, SceneRanker())
    result = stage.apply(leaves, TypedQuery("release", "MEMORY", [], top_k=1), [],
                         _scope_context("acme", "user:alice"), RetrievalTrace())
    assert result.hits is leaves


def test_mixed_queries_keep_route_local_limits_and_evidence(setup):
    memory_query = TypedQuery("release", "MEMORY", [], top_k=2)
    skill_query = TypedQuery("workflow", "SKILL", [], top_k=1)
    skill = SeedHit("ctx://acme/skills/release", 2.0, context_type="SKILL", owner_space="user:alice")
    setup.pipeline.planner = SimpleNamespace(plan=lambda *args, **kwargs: [memory_query, skill_query])
    seed = setup.pipeline.seed_retriever
    setup.pipeline.seed_retriever = SimpleNamespace(search=lambda query, ctx, **kwargs:
        SeedResult(initial_candidates=[skill]) if query is skill_query else seed.search(query, ctx, **kwargs))
    setup.pipeline._context_reader = None
    result = search(setup)
    assert evidence_uris(result) == {setup.direct.uri, setup.related.uri, skill.uri}
    assert sum(stage.stage == "scene_ranking" for stage in result.trace.stages) == 1


def test_hierarchy_metadata_and_content_projection_identity_survive_scene_boost(setup):
    leaf = LeafHit(setup.direct.uri + "/content.md", 0.8, depth=3,
                   hierarchy_path=["root", "events"], owner_space="user:alice")
    trace = RetrievalTrace()
    stage = setup.pipeline._scene_stage.bind(setup.ctx, [TypedQuery("release", "MEMORY", [])], trace)
    result = stage.apply([leaf], TypedQuery("release", "MEMORY", []), [1, 0], setup.ctx, trace).hits
    direct = next(hit for hit in result if hit.uri == leaf.uri)
    assert direct.depth == 3
    assert direct.hierarchy_path == ["root", "events"]
    assert direct.score > leaf.score
    assert len(result) == 2


def test_legacy_search_port_expands_members_with_alias_owner(setup):
    source = SimpleNamespace(
        search=lambda *args, **kwargs: [SceneHit(
            "plan", 0.95, "user_alice", (setup.direct.uri, setup.related.uri))],
        resolve_members=lambda *args, **kwargs: [SeedHit(
            setup.related.uri, 0.3, owner_space="user_alice", category="event")],
    )
    stage = SceneRetrievalStage(source, SceneRanker())
    result = stage.apply([LeafHit(setup.direct.uri, 0.8, owner_space="user:alice")],
                         TypedQuery("release", "MEMORY", []), [1, 0], setup.ctx, RetrievalTrace()).hits
    assert {hit.uri for hit in result} == {setup.direct.uri, setup.related.uri}


def test_bounded_selection_preserves_direct_reserve_and_expansion_quota(setup):
    uris = [f"ctx://acme/users/alice/memories/events/{i}" for i in range(12)]
    source = SimpleNamespace(
        search=lambda *args, **kwargs: [SceneHit(
            "plan", 1.0, "user:alice", tuple(uris), routes=("member",))],
        resolve_members=lambda *args, **kwargs: [
            SeedHit(uri, 1.0, owner_space="user:alice") for uri in uris[8:]],
    )
    config = replace(SceneConfig(), direct_evidence_reserve=0.5, max_members_per_scene=3)
    stage = SceneRetrievalStage(source, SceneRanker(config))
    leaves = [LeafHit(uri, 0.9 - i / 100, owner_space="user:alice") for i, uri in enumerate(uris[:8])]
    query = TypedQuery("release", "MEMORY", [], top_k=6)
    candidates = stage.apply(leaves, query, [1, 0], setup.ctx, RetrievalTrace())
    assert len(candidates.hits) == 11
    result = ResultRanker().assemble(
        query, candidates.hits, None, direct_evidence_reserve=candidates.direct_evidence_reserve)
    assert len(result) == 6
    assert sum(hit.metadata.get("scene_expanded", False) for hit in result) == 3
    assert {hit.uri for hit in result} == set(uris[:3] + uris[8:11])


def test_ranker_preserves_distinct_knowledge_units(setup):
    leaves = [LeafHit(setup.direct.uri, 0.8, owner_space="user:alice", metadata={"unit_key": key})
              for key in ("deploy", "rollback")]
    stage = setup.pipeline._scene_stage
    result = stage.apply(leaves, TypedQuery("release", "MEMORY", []), [1, 0],
                         setup.ctx, RetrievalTrace()).hits
    assert {hit.metadata["unit_key"] for hit in result if hit.uri == setup.direct.uri} == {"deploy", "rollback"}


def test_empty_memory_route_retains_skill_results(setup):
    queries = [TypedQuery("release", "MEMORY", []), TypedQuery("workflow", "SKILL", [])]
    skill = SeedHit("ctx://acme/skills/release", 0.9, context_type="SKILL")
    setup.pipeline.planner = SimpleNamespace(plan=lambda *args, **kwargs: queries)
    setup.pipeline.seed_retriever = SimpleNamespace(search=lambda query, ctx, **kwargs:
        SeedResult(initial_candidates=[skill]) if query.context_type == "SKILL" else SeedResult())
    setup.pipeline._context_reader = None
    setup.source.repository.snapshot = replace(setup.source.repository.snapshot, scenes=())
    result = search(setup)
    assert [hit.uri for hit in result.hits] == [skill.uri]


def test_live_candidates_fill_slots_after_canonical_removal(setup):
    current = node("current")
    setup.source.fs.write_node(current, setup.ctx)
    records = build_index_records(current)
    for record in records:
        record.metadata["_embedding"] = [0.9, 0.1]
    setup.source.vector_index.upsert(records)
    del setup.source.fs.nodes[setup.direct.uri]
    setup.source.repository.snapshot = replace(
        setup.source.repository.snapshot,
        scenes=(scene("active", [current, setup.related], centroid=(1.0, 0.0)),),
    )
    result = setup.pipeline.run("remember release", setup.ctx, top_k=1, context_type="MEMORY")
    assert evidence_uris(result) == {current.uri}


def test_multiple_memory_queries_bind_once_and_keep_categories(setup, monkeypatch):
    preference = node("style", category="preference")
    setup.source.fs.write_node(preference, setup.ctx)
    records = build_index_records(preference)
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    setup.source.vector_index.upsert(records)
    queries = [TypedQuery("events", "MEMORY", ["event"], top_k=2),
               TypedQuery("preferences", "MEMORY", ["preference"], top_k=1)]
    setup.pipeline.planner = SimpleNamespace(plan=lambda *args, **kwargs: queries)
    calls = []
    bind = setup.source.for_request

    def counted(ctx, bound_queries):
        calls.append(tuple(query.text for query in bound_queries))
        return bind(ctx, bound_queries)

    monkeypatch.setattr(setup.source, "for_request", counted)
    result = search(setup)
    assert evidence_uris(result) == {setup.direct.uri, setup.related.uri, preference.uri}
    assert calls == [("events", "preferences")]
    assert sum(hit.category == "event" for hit in result.hits) == 2
    assert sum(hit.category == "preference" for hit in result.hits) == 1


def test_duplicate_projections_leave_slots_for_distinct_evidence(setup):
    source = SimpleNamespace(search=lambda *args, **kwargs: [SceneHit(
        "plan", 1.0, "user:alice", (setup.direct.uri, setup.related.uri))])
    leaves = [LeafHit(setup.direct.uri, 0.9, owner_space="user:alice"),
              LeafHit(setup.direct.uri + "/content.md", 0.8, owner_space="user:alice"),
              LeafHit(setup.related.uri, 0.7, owner_space="user:alice")]
    result = SceneRetrievalStage(source, SceneRanker()).apply(
        leaves, TypedQuery("release", "MEMORY", [], top_k=2), [1, 0], setup.ctx, RetrievalTrace()).hits
    assert {canonical_uri(hit.uri) for hit in result} == {setup.direct.uri, setup.related.uri}


@pytest.mark.parametrize("strategy", ["semantic", "metadata_aware"])
@pytest.mark.parametrize("scene_enabled", [False, True])
def test_scene_candidate_pool_keeps_date_winner_until_final_ranking(setup, strategy, scene_enabled):
    config = replace(setup.pipeline.cfg, candidate_pool_enabled=True,
                     min_candidate_pool_size=4, result_ranker_strategy=strategy)
    setup.pipeline.cfg = config
    setup.pipeline.assembly = ResultRanker(config)
    if not scene_enabled:
        setup.pipeline._scene_stage = None
    setup.provider.publish("acme", "user:alice", 0, {"scene_boost": 0.0})
    setup.direct.metadata["when"] = "2026-10-01"
    setup.related.metadata["when"] = "2026-10-20"
    for item, vector in [(setup.direct, [0.90, 0.435889894]),
                         (setup.related, [0.89, 0.455960525])]:
        setup.source.fs.write_node(item, setup.ctx)
        records = build_index_records(item)
        for record in records:
            record.metadata["_embedding"] = vector
        setup.source.vector_index.upsert(records)
    setup.pipeline.seed_retriever = SeedRetriever(
        setup.source.vector_index, setup.source.embedder, config)
    result = setup.pipeline.run("release on 2026-10-20", setup.ctx, top_k=1, context_type="MEMORY")
    assert evidence_uris(result) == {setup.related.uri}
    assert result.hits[0].ranking_score > 1.0


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_member_budget_fills_new_evidence_after_direct_members(setup, budget):
    a, b, c, d = [node(name) for name in ("a", "b", "c", "d")]
    setup.source.repository.snapshot = replace(
        setup.source.repository.snapshot,
        scenes=(scene("first", [a, c]), scene("second", [b, d])),
    )
    for item, vector in [(a, [1, 0]), (b, [1, 0]), (c, [0, 1]), (d, [0, 1])]:
        setup.source.fs.write_node(item, setup.ctx)
        records = build_index_records(item)
        for record in records:
            record.metadata["_embedding"] = vector
        setup.source.vector_index.upsert(records)
    setup.provider.publish("acme", "user:alice", 0,
                           {"max_members_per_scene": budget, "min_evidence_nodes": 1})
    result = search(setup)
    expanded = {hit.uri for hit in result.hits if hit.metadata.get("scene_expanded")}
    assert expanded == set([c.uri, d.uri][:budget])


def test_final_scene_quota_uses_temporal_order_in_both_evidence_groups(setup):
    direct = [node("direct-first"), node("direct-date")]
    expanded = [node("expanded-first"), node("expanded-date")]
    source = SimpleNamespace(
        search=lambda *args, **kwargs: [SceneHit(
            "plan", 1.0, "user:alice", tuple(item.uri for item in direct + expanded), routes=("member",))],
        resolve_members=lambda *args, **kwargs: [
            SeedHit(item.uri, score, owner_space="user:alice", category="event", metadata=metadata)
            for item, score, metadata in [(expanded[0], 0.9, {}),
                                           (expanded[1], 0.89, {"when": "2026-10-20"})]],
    )
    stage = SceneRetrievalStage(source, SceneRanker(replace(
        SceneConfig(), scene_boost=0.0, direct_evidence_reserve=0.5)))
    leaves = [LeafHit(direct[0].uri, 0.9, owner_space="user:alice", category="event"),
              LeafHit(direct[1].uri, 0.89, owner_space="user:alice", category="event",
                      metadata={"when": "2026-10-20"})]
    query = TypedQuery("release on 2026-10-20", "MEMORY", [], top_k=2)
    candidates = stage.apply(leaves, query, [1, 0], setup.ctx, RetrievalTrace())
    result = ResultRanker().assemble(
        query, candidates.hits, None, direct_evidence_reserve=candidates.direct_evidence_reserve)
    assert [hit.uri for hit in result] == [direct[1].uri, expanded[1].uri]
    assert result[0].ranking_score > result[1].ranking_score


@pytest.mark.parametrize("global_rerank", [False, True])
@pytest.mark.parametrize("reserve", [0.5, 0.9])
@pytest.mark.parametrize("repeat_query", [False, True])
def test_multiple_queries_preserve_scene_quota_after_global_ranking(
    setup, global_rerank, reserve, repeat_query,
):
    config = replace(setup.pipeline.cfg, global_result_rerank_enabled=global_rerank)
    index = InMemoryVectorIndex(dimension=2)
    records, queries, expanded_uris = [], [], set()
    for category in ("event", "preference"):
        direct = [node(f"direct-{i}", category=category) for i in range(9)]
        expanded = [node(f"expanded-{i}", category=category) for i in range(6)]
        expanded_uris.update(item.uri for item in expanded)
        for item in direct + expanded:
            setup.source.fs.write_node(item, setup.ctx)
            vectors = build_index_records(item)
            for vector in vectors:
                vector.metadata["_embedding"] = [0, 1] if item.uri in expanded_uris else [1, 0]
            index.upsert(vectors)
        records.append(scene(category, direct + expanded, centroid=(0, 1)))
        queries.append(TypedQuery(category, "MEMORY", [category], top_k=10))
    if repeat_query:
        queries.insert(1, queries[0])
    setup.source.repository.snapshot = replace(setup.source.repository.snapshot, scenes=tuple(records))
    setup.source.vector_index = index
    setup.provider.publish("acme", "user:alice", 0, {"direct_evidence_reserve": reserve})
    setup.pipeline.cfg = config
    setup.pipeline.assembly = ResultRanker(config)
    setup.pipeline.seed_retriever = SeedRetriever(index, setup.source.embedder, config)
    setup.pipeline.planner = SimpleNamespace(plan=lambda *args, **kwargs: queries)

    result = search(setup)

    multiplier = 1 if global_rerank else 2
    assert len(result.hits) == len(evidence_uris(result)) == 10 * multiplier
    assert sum(bool(hit.metadata.get("scene_expanded")) for hit in result.hits) == (
        10 - ceil(10 * reserve)
    ) * multiplier
    for hit in result.hits:
        assert hit.content_excerpt == setup.source.fs.nodes[canonical_uri(hit.uri)].content
    if global_rerank:
        scores = [hit.ranking_score for hit in result.hits]
        assert scores == sorted(scores, reverse=True)


def test_concurrent_scene_queries_keep_request_selection(setup):
    from concurrent.futures import ThreadPoolExecutor

    selections = [True, False] * 12
    with ThreadPoolExecutor(max_workers=6) as callers:
        results = list(callers.map(lambda enabled: search(setup, use_scene=enabled), selections))
    for enabled, result in zip(selections, results):
        expected = {setup.direct.uri, setup.related.uri} if enabled else {setup.direct.uri}
        assert evidence_uris(result) == expected
    assert evidence_uris(search(setup)) == {setup.direct.uri, setup.related.uri}


def test_request_selection_overrides_retrieval_default(setup):
    setup.pipeline._scene_enabled = False
    assert evidence_uris(search(setup)) == {setup.direct.uri}
    assert evidence_uris(search(setup, use_scene=True)) == {setup.direct.uri, setup.related.uri}
    assert evidence_uris(search(setup)) == {setup.direct.uri}
