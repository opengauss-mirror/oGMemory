"""Real SQL chunk leases publish current graph revisions independently."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from secrets import randbits
from threading import BoundedSemaphore, Event

import psycopg2

from core.models import RelationEdge
from fs.sql_adapter.session_mutation import SQLSessionGuard, session_mutation_scope
from index.edge_projection import EdgeProjectionWorker
from providers.relation_store.projection_batch import graph_projection_batch
from tests.integration.test_graph_projection_source_refresh import _update_content
from tests.integration.test_semantic_context_graph import _graph_sql
from tests.integration.test_semantic_context_graph import graph_db as graph_db


@contextmanager
def _writer_batch(ctx):
    with session_mutation_scope(
        os.environ["GRAPH_TEST_DSN"], ctx, randbits(63) or 1,
        BoundedSemaphore(1), timeout_seconds=5,
        dialect=os.environ.get("GRAPH_TEST_DIALECT", "postgresql"),
    ) as guard:
        with graph_projection_batch(guard):
            yield guard


def _jobs(fs, ctx):
    return {
        row[0]: dict(zip(
            ("status", "target_revision", "attempt_count", "batch_key", "deadline", "created_at"),
            row[1:],
        ))
        for row in _graph_sql(
            fs, ctx,
            "SELECT target_uri,status,target_revision,attempt_count,mutation_batch_key,"
            "next_retry_at,created_at FROM edge_projection_jobs WHERE account_id=%s",
            (ctx.account_id,),
        )
    }


def _project_independently(projections, worker, profile, account_id):
    def project():
        assert projections.has_due_work(account_id)
        accepted = projections.claim_pending(account_id)
        EdgeProjectionWorker(worker._embedder, projections, profile).project_accepted_sync(accepted)
        return accepted

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(project).result(timeout=10)


def _projection(store, ctx, source, target, profile):
    return next(
        item["edge_projection"]
        for item in store.get_traversal_neighbors(source.uri, ctx, embedding_profile=profile)
        if item["target_uri"] == target.uri
    )


def test_chunk_projects_latest_target_once_while_ready_job_completes(graph_db, monkeypatch):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, ready = nodes
    batch_key = (1 << 62) + (1 << 31) + 27
    monkeypatch.setattr("providers.relation_store.projection_batch.randbits", lambda bits: batch_key)
    store.upsert_edges([RelationEdge(source.uri, ready.uri, "supports", 1., "evidence")], ctx)
    calls_before = worker._embedder.embed_texts.call_count

    with _writer_batch(ctx):
        store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
        for revision in (2, 3):
            _update_content(fs, ctx, target, f"Target delivery revision {revision}")
        accepted = _project_independently(projections, worker, profile, ctx.account_id)
        assert [item.target_uri for item in accepted] == [ready.uri]
        jobs = _jobs(fs, ctx)
        assert jobs[ready.uri]["status"] == "COMPLETE"
        assert jobs[target.uri]["status"] == "PENDING"
        assert jobs[target.uri]["target_revision"] == 3
        assert jobs[target.uri]["batch_key"] == batch_key
        _update_content(fs, ctx, target, "Target delivery revision 4")

    accepted = _project_independently(projections, worker, profile, ctx.account_id)
    assert [(item.target_uri, item.target_revision) for item in accepted] == [(target.uri, 4)]
    published = _projection(store, ctx, source, target, profile)
    assert published["target_revision"] == 4
    assert "Target delivery revision 4" in published["text"]
    assert published["vector"] == [1., 0.]
    assert [(row["status"], row["attempt_count"]) for row in _jobs(fs, ctx).values()] == [
        ("COMPLETE", 1), ("COMPLETE", 1),
    ]
    assert worker._embedder.embed_texts.call_count - calls_before == 2


def test_overlapping_chunks_release_their_jobs_independently(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, first, second = nodes
    entered = [Event(), Event()]
    release = [Event(), Event()]

    def write_chunk(position, target):
        writer_ctx = replace(ctx, session_id=f"chunk-{position}")
        with _writer_batch(writer_ctx):
            store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], writer_ctx)
            _update_content(fs, writer_ctx, target, f"Completed chunk {position}")
            entered[position].set()
            assert release[position].wait(20)
        return target.uri

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_chunk, i, target) for i, target in enumerate((first, second))]
        try:
            assert all(event.wait(10) for event in entered)
            jobs = _jobs(fs, ctx)
            assert len({row["batch_key"] for row in jobs.values()}) == 2
            assert all(row["batch_key"] > 0 for row in jobs.values())
            release[0].set()
            assert futures[0].result(timeout=10) == first.uri
            accepted = _project_independently(projections, worker, profile, ctx.account_id)
            assert [item.target_uri for item in accepted] == [first.uri]
            jobs = _jobs(fs, ctx)
            assert jobs[first.uri]["status"] == "COMPLETE"
            assert jobs[second.uri]["status"] == "PENDING"
            assert jobs[second.uri]["target_revision"] == 2
            release[1].set()
            assert futures[1].result(timeout=10) == second.uri
        finally:
            for event in release:
                event.set()
            for future in futures:
                future.result(timeout=10)

    accepted = _project_independently(projections, worker, profile, ctx.account_id)
    assert [item.target_uri for item in accepted] == [second.uri]
    assert [(row["status"], row["attempt_count"]) for row in _jobs(fs, ctx).values()] == [
        ("COMPLETE", 1), ("COMPLETE", 1),
    ]


def test_pending_deadline_survives_batch_switches_and_publishes_when_due(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    with _writer_batch(ctx):
        store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
        original = _jobs(fs, ctx)[target.uri]
        assert original["deadline"] - original["created_at"] == timedelta(seconds=30)
        _update_content(fs, ctx, target, "Batched revision 2")
        assert _jobs(fs, ctx)[target.uri]["deadline"] == original["deadline"]

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                _update_content, fs, replace(ctx, session_id="ordinary-writer"), target, "Ordinary revision 3",
            ).result(timeout=10)
        ordinary = _jobs(fs, ctx)[target.uri]
        assert ordinary["batch_key"] is None
        assert ordinary["deadline"] == original["deadline"]

        _update_content(fs, ctx, target, "Batched revision 4")
        resumed = _jobs(fs, ctx)[target.uri]
        assert resumed["batch_key"] == original["batch_key"]
        assert resumed["deadline"] == original["deadline"]
        _graph_sql(
            fs, ctx,
            "UPDATE edge_projection_jobs SET next_retry_at=NOW()-INTERVAL '1 second' "
            "WHERE account_id=%s AND target_uri=%s",
            (ctx.account_id, target.uri),
        )
        due = _jobs(fs, ctx)[target.uri]["deadline"]
        _update_content(fs, ctx, target, "Current revision 5 at original deadline")
        assert _jobs(fs, ctx)[target.uri]["deadline"] == due
        accepted = _project_independently(projections, worker, profile, ctx.account_id)
        assert [(item.target_uri, item.target_revision) for item in accepted] == [(target.uri, 5)]
        assert _jobs(fs, ctx)[target.uri]["status"] == "COMPLETE"

    published = _projection(store, ctx, source, target, profile)
    assert published["target_revision"] == 5
    assert "Current revision 5 at original deadline" in published["text"]


def test_closed_writer_connection_releases_pending_projection(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    dsn = os.environ["GRAPH_TEST_DSN"]
    raw = psycopg2.connect(dsn)
    key = randbits(63) or 1
    guard = SQLSessionGuard(raw, dsn, ctx, key, os.environ.get("GRAPH_TEST_DIALECT", "postgresql"))
    try:
        with raw.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            assert cursor.fetchone()[0]
        raw.commit()
        with graph_projection_batch(guard):
            fs.bind_tenant(raw, ctx.account_id)
            with raw.cursor() as cursor:
                cursor.execute(
                    "UPDATE context_nodes SET content=%s,version=version+1,updated_at=NOW() "
                    "WHERE account_id=%s AND uri=%s",
                    ("Committed before writer disconnect", ctx.account_id, target.uri),
                )
                assert cursor.rowcount == 1
            raw.commit()
            pending = _jobs(fs, ctx)[target.uri]
            assert (pending["status"], pending["target_revision"]) == ("PENDING", 2)
            assert pending["batch_key"] > 0
            raw.close()
            accepted = _project_independently(projections, worker, profile, ctx.account_id)
            assert [(item.target_uri, item.target_revision) for item in accepted] == [(target.uri, 2)]
    finally:
        raw.close()

    published = _projection(store, ctx, source, target, profile)
    assert published["target_revision"] == 2
    assert "Committed before writer disconnect" in published["text"]
    assert _jobs(fs, ctx)[target.uri]["status"] == "COMPLETE"


def test_latest_claim_stays_published_after_earlier_claim_finishes(graph_db):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, _ = nodes
    store.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
    earlier = projections.claim_pending(ctx.account_id)
    assert earlier[0].target_revision == 1
    with _writer_batch(ctx):
        _update_content(fs, ctx, target, "Target revision 2")
        _update_content(fs, ctx, target, "Target revision 3")
    latest = projections.claim_pending(ctx.account_id)
    assert latest[0].target_revision == 3
    projector = EdgeProjectionWorker(worker._embedder, projections, profile)
    projector.project_accepted_sync(latest)
    expected = _projection(store, ctx, source, target, profile)
    projector.project_accepted_sync(earlier)
    assert _projection(store, ctx, source, target, profile) == expected
    assert expected["target_revision"] == 3
    assert "Target revision 3" in expected["text"]
    final = _jobs(fs, ctx)[target.uri]
    assert (final["status"], final["target_revision"], final["attempt_count"]) == ("COMPLETE", 3, 2)
