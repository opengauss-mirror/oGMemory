"""Native SQL publication, migration and component reuse for graph projections."""

from collections import Counter
from unittest.mock import Mock

import pytest

from core.models import RelationEdge
from index.edge_projection import EdgeProjectionWorker
from index.embedding_components import EmbeddingComponentCache
from index.outbox_worker import OutboxWorker
from providers.relation_store.projection_identity import FACTORIZED_REPRESENTATION_VERSION
from providers.relation_store.sql_projection import SQLEdgeProjectionStore
from providers.relation_store.sql_relation_store import SQLRelationStore
from tests.integration.test_graph_projection_source_refresh import _update_content
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def _profile(profile):
    return dict(profile, profile_id="test-factorized-full-v1",
                representation_version=FACTORIZED_REPRESENTATION_VERSION)


def _embedder():
    provider = Mock(model="test-embedding")
    provider.embed_texts.side_effect = lambda texts: [
        [0., 1.] if text.startswith("Relationship: ") else [1., 0.] for text in texts]
    return provider


def test_profile_migration_pages_reuse_full_target_across_worker_instances(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, old_profile = graph_db
    source, target, third = nodes
    _update_content(fs, ctx, target, "Complete reusable target fact. " * 400)
    store.upsert_edges([
        RelationEdge(source.uri, target.uri, "owns", 1., "first evidence"),
        RelationEdge(third.uri, target.uri, "supports", 1., "second evidence"),
    ], ctx)
    while projections.has_due_work(ctx.account_id, profile=old_profile):
        worker.run_once(outbox, [ctx.account_id])
    profile = _profile(old_profile)
    cache = EmbeddingComponentCache(max_entries=16)
    provider = _embedder()
    page_sizes = []
    for _ in range(2):
        current = SQLEdgeProjectionStore(pool=projections._shared_pool, component_cache=cache)
        page_sizes.append(current.reconcile_profile(ctx.account_id, profile, limit=1))
        assert current.has_due_work(ctx.account_id, profile=profile)
        restarted = OutboxWorker(index, provider, fs=fs, directory_summary_enabled=False,
                                edge_projection_components=(current, profile))
        result = restarted.run_once(outbox, [ctx.account_id])
        assert result["failed"] == 0
    page_sizes.append(current.reconcile_profile(ctx.account_id, profile, limit=1))
    assert page_sizes == [1, 1, 0]
    target_now = fs.read_node(target.uri, ctx)
    full = "\n".join((target_now.abstract, target_now.overview, target_now.content)).strip()
    actual = Counter(text for call in provider.embed_texts.call_args_list for text in call.args[0])
    assert actual == Counter([full, "Relationship: owns", "Relationship: supports"])
    expected = [3 / (10 ** .5), 1 / (10 ** .5)]
    for anchor in (source, third):
        row = store.get_traversal_neighbors(anchor.uri, ctx, embedding_profile=profile)[0]
        assert row["target_uri"] == target.uri
        assert row["edge_projection"]["vector"] == pytest.approx(expected)
        assert row["edge_projection"]["target_revision"] == 2
        assert row["edge_projection"]["representation_version"] == FACTORIZED_REPRESENTATION_VERSION
    assert _graph_sql(fs, ctx,
        "SELECT status,embedding_profile,representation_version,source_revision,target_revision "
        "FROM edge_projection_jobs WHERE account_id=%s ORDER BY edge_id", (ctx.account_id,)) == [
            ("COMPLETE", profile["profile_id"], FACTORIZED_REPRESENTATION_VERSION, 1, 2)] * 2


def test_factorized_source_refresh_and_target_revision_publish_current_vectors(graph_db):
    ctx, fs, index, _, projections, outbox, _, nodes, old_profile = graph_db
    source, target, _ = nodes
    profile = _profile(old_profile)
    cache = EmbeddingComponentCache(max_entries=16)
    store = SQLRelationStore(pool=projections._shared_pool,
        edge_projection_profile_id=profile["profile_id"],
        edge_projection_representation=FACTORIZED_REPRESENTATION_VERSION)
    current = SQLEdgeProjectionStore(pool=projections._shared_pool, component_cache=cache)
    provider = _embedder()
    worker = OutboxWorker(index, provider, fs=fs, directory_summary_enabled=False,
                         edge_projection_components=(current, profile))
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    worker.run_once(outbox, [ctx.account_id])
    initial = store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)[0]["edge_projection"]
    initial_inputs = [text for call in provider.embed_texts.call_args_list for text in call.args[0]]
    for revision in (2, 3):
        _update_content(fs, ctx, source, f"Source revision {revision}")
        row = store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)[0]
        assert row["edge_projection"] == dict(initial, source_revision=revision)
        assert _graph_sql(fs, ctx,
            "SELECT status,source_revision,target_revision,attempt_count FROM edge_projection_jobs "
            "WHERE account_id=%s", (ctx.account_id,)) == [("COMPLETE", revision, 1, 1)]
    assert [text for call in provider.embed_texts.call_args_list for text in call.args[0]] == initial_inputs
    _update_content(fs, ctx, target, "Target revision two has a new verified fact")
    worker.run_once(outbox, [ctx.account_id])
    row = store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)[0]
    assert (row["edge_projection"]["source_revision"], row["edge_projection"]["target_revision"]) == (3, 2)
    assert "new verified fact" in row["edge_projection"]["text"]
    counts = Counter(text for call in provider.embed_texts.call_args_list for text in call.args[0])
    assert counts["Relationship: owns"] == 1
    assert sum(counts.values()) == 3


def test_each_profile_claims_and_publishes_its_current_representation(graph_db):
    ctx, fs, _, legacy, projections, _, worker, nodes, old_profile = graph_db
    source, target, third = nodes
    profile = _profile(old_profile)
    current = SQLRelationStore(pool=projections._shared_pool,
        edge_projection_profile_id=profile["profile_id"],
        edge_projection_representation=FACTORIZED_REPRESENTATION_VERSION)
    legacy.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    current.upsert_edges([RelationEdge(source.uri, third.uri, "supports", 1., "evidence")], ctx)
    assert projections.has_due_work(ctx.account_id, profile=old_profile)
    assert projections.has_due_work(ctx.account_id, profile=profile)
    old_claims = projections.claim_pending(ctx.account_id, profile=old_profile)
    new_claims = projections.claim_pending(ctx.account_id, profile=profile)
    assert [edge.target_uri for edge in old_claims] == [target.uri]
    assert [edge.target_uri for edge in new_claims] == [third.uri]
    EdgeProjectionWorker(worker._embedder, projections, old_profile).project_accepted_sync(old_claims)
    EdgeProjectionWorker(_embedder(), projections, profile,
                         component_cache=EmbeddingComponentCache()).project_accepted_sync(new_claims)
    assert [r["target_uri"] for r in legacy.get_traversal_neighbors(
        source.uri, ctx, embedding_profile=old_profile)] == [target.uri]
    assert [r["target_uri"] for r in current.get_traversal_neighbors(
        source.uri, ctx, embedding_profile=profile)] == [third.uri]
    assert _graph_sql(fs, ctx,
        "SELECT status FROM edge_projection_jobs WHERE account_id=%s ORDER BY edge_id",
        (ctx.account_id,)) == [("COMPLETE",), ("COMPLETE",)]
