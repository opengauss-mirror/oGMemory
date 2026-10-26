"""Scene persistence acceptance tests for the configured SQL dialect."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread
from time import sleep
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn
from psycopg2.extras import Json

from core.models import ContextNode, RequestContext
from fs.sql_adapter import SharedConnectionPool, SQLContextFS
from scene.adapters import CanonicalSceneAdapter
from scene.config import SceneConfigCache
from scene.jobs import SceneJobStatus, current_scene_job, job_to_dict
from scene.models import SceneCluster, SceneMember, SceneRecord
from scene.sql_store import SQLSceneConfigProvider, SQLSceneJobStore, SQLSceneRepository
from tests.unit.scene.test_adapters import Embedder, IndexedVectors


@pytest.fixture
def scene_sql():
    dsn = os.environ.get("OGMEM_SCENE_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_SCENE_SQL_DSN is required for SQL acceptance")
    dialect = os.environ.get("OGMEM_SCENE_SQL_DIALECT", "postgresql")
    namespace = "scene_test_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    pool = SharedConnectionPool(make_dsn(dsn, options=f"-c search_path={namespace}"), pool_size=6, dialect=dialect)
    try:
        fs = SQLContextFS(pool=pool)
        repository = SQLSceneRepository(pool)
        jobs = SQLSceneJobStore(pool)
        configuration = SQLSceneConfigProvider(pool)
        ctx = RequestContext(account_id="scene-test", user_id="alice", agent_id="agent",
                             session_id="session", trace_id="test")
        yield pool, fs, repository, jobs, configuration, ctx
    finally:
        pool.close_all()
        with control.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))
        control.close()


def _write(fs, ctx, name, version=1):
    node = ContextNode(uri=f"ctx://{ctx.account_id}/users/alice/memories/events/{name}",
                       context_type="MEMORY", category="event", owner_space="user:alice", level=4,
                       abstract=name, overview=name, content=name, metadata={"version": version})
    fs.write_node(node, ctx)
    return node


def _scene(scene_id, node):
    cluster = SceneCluster("user:alice", (SceneMember(node.uri, 1.0, node_version=node.metadata["version"]),),
                           (1.0, 0.0), .9, "Release preparation", "Review and release milestones.", "source")
    return SceneRecord(scene_id, cluster)


def test_scene_projection_queue_tracks_published_revisions(scene_sql):
    _, fs, repository, _, _, ctx = scene_sql
    memory = _write(fs, ctx, "projection")
    original = repository.load_snapshot(ctx.account_id, "user:alice")
    published = replace(original, revision=1, scenes=(_scene("planning", memory),))
    assert repository.compare_and_swap(original, published)
    for snapshot in repository.pending_projections():
        assert snapshot == published
        repository.publish_projection(snapshot, lambda current: None)
    assert list(repository.pending_projections()) == []
    updated = replace(published, revision=2, scenes=(replace(published.scenes[0], version=2),))
    assert repository.compare_and_swap(published, updated)
    assert list(repository.pending_projections()) == [updated]


def test_scene_snapshot_roundtrip_cas_history_and_source_refresh(scene_sql):
    pool, fs, repository, _, _, ctx = scene_sql
    first_node = _write(fs, ctx, "review")
    second_node = _write(fs, ctx, "release")
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    first = replace(initial, revision=1, scenes=(_scene("one", first_node), _scene("two", second_node)))
    assert repository.compare_and_swap(initial, first)
    assert SQLSceneRepository(pool).load_snapshot(ctx.account_id, "user:alice") == first
    assert repository.load_snapshot("other-account", "user:alice").revision == 0

    fs.archive_node(second_node.uri, ctx)
    edited = replace(first.scenes[0], version=2, description_edited=True,
                     cluster=replace(first.scenes[0].cluster, summary="A manually refined description."))
    second = replace(first, revision=2, scenes=(edited, first.scenes[1]))
    assert repository.compare_and_swap(first, second)
    assert repository.load_snapshot(ctx.account_id, "user:alice").scenes[0].description_edited

    _write(fs, ctx, "review", version=2)
    stale = replace(second, revision=3, scenes=(replace(edited, version=3), second.scenes[1]))
    assert repository.compare_and_swap(second, stale) is False
    refreshed = replace(stale, scenes=(replace(stale.scenes[0], cluster=replace(
        edited.cluster, members=(replace(edited.cluster.members[0], node_version=2),))), second.scenes[1]))
    assert repository.compare_and_swap(second, refreshed)
    archived = replace(refreshed, revision=4, scenes=tuple(replace(record, status="ARCHIVED")
                                                       for record in refreshed.scenes))
    assert repository.compare_and_swap(refreshed, archived)
    with repository._transaction(ctx.account_id) as cursor:
        cursor.execute("SELECT revision FROM scene_snapshot_history ORDER BY revision")
        assert [row[0] for row in cursor.fetchall()] == [1, 2, 3, 4]


def test_parallel_scene_publication_has_one_revision_winner(scene_sql):
    _, _, repository, _, _, ctx = scene_sql
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    target = replace(initial, revision=1)
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(lambda _: repository.compare_and_swap(initial, target), range(2)))
    assert sorted(outcomes) == [False, True]
    assert repository.load_snapshot(ctx.account_id, "user:alice").revision == 1


def test_durable_queue_drains_every_window_and_deduplicates_notices(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    nodes = [f"n{index:03}" for index in range(70)]
    jobs.enqueue(ctx.account_id, "user:alice", nodes, token_count=700, trigger_tokens=100,
                 max_pending_nodes=16, notice_id="canonical-write-1")
    jobs = SQLSceneJobStore(pool)
    jobs.enqueue(ctx.account_id, "user:alice", nodes, token_count=700, trigger_tokens=100,
                 max_pending_nodes=16, notice_id="canonical-write-1")
    processed = []
    while job := jobs.claim("worker"):
        processed.extend(job.node_uris)
        jobs.mark_succeeded(job)
    assert processed == nodes
    jobs.enqueue(ctx.account_id, "user:alice", nodes, token_count=700,
                 notice_id="canonical-write-1")
    assert jobs.summary(ctx.account_id, "user:alice")["pending_nodes"] == 0


def test_queue_accumulates_wait_evidence_and_flushes_fresh_tail(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    jobs.enqueue(ctx.account_id, "user:alice", ["old"], token_count=1, trigger_tokens=100)
    jobs.flush(ctx.account_id, "user:alice")
    first = jobs.claim("worker")
    jobs.mark_waiting(first, reason="More context is needed")
    jobs = SQLSceneJobStore(pool)
    jobs.enqueue(ctx.account_id, "user:alice", ["fresh"], token_count=3, trigger_tokens=100)
    summary = jobs.summary(ctx.account_id, "user:alice")
    assert summary["retained_nodes"] == 1
    assert summary["pending_tokens"] == 3
    jobs.flush(ctx.account_id, "user:alice")
    next_job = jobs.claim("worker")
    assert next_job.node_uris == ("fresh", "old")
    jobs.mark_succeeded(next_job, remaining_uris=["old"])
    assert jobs.summary(ctx.account_id, "user:alice")["retained_nodes"] == 1


def test_completed_projection_persists_remaining_work_across_restart(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    jobs.enqueue(ctx.account_id, "user:alice", ["n1", "n2", "n3", "n4", "n5", "n6"], token_count=4096)
    first = jobs.claim("worker")
    jobs.mark_succeeded(first, reason="CREATE", remaining_uris=["n4", "n5", "n6"])
    jobs = SQLSceneJobStore(pool)
    following = jobs.claim("restarted-worker")
    assert following.node_uris == ("n4", "n5", "n6")
    assert following.token_count == 0
    jobs.mark_waiting(following, reason="Additional independent context is required")
    jobs = SQLSceneJobStore(pool)
    summary = jobs.summary(ctx.account_id, "user:alice")
    assert (summary["pending_nodes"], summary["retained_nodes"]) == (0, 3)
    assert jobs.flush(ctx.account_id, "user:alice").status == SceneJobStatus.WAITING
    jobs.enqueue(ctx.account_id, "user:alice", ["n7"], token_count=4096)
    resumed = jobs.claim("worker")
    assert resumed.node_uris == ("n7", "n4", "n5", "n6")
    jobs.mark_succeeded(resumed, reason="UPDATE")
    assert jobs.summary(ctx.account_id, "user:alice")["pending_nodes"] == 0


def test_expired_worker_publication_is_fenced_by_recovered_lease(scene_sql):
    _, _, repository, jobs, _, ctx = scene_sql
    jobs.enqueue(ctx.account_id, "user:alice", ["n1"], token_count=1)
    old = jobs.claim("worker-1", lease_seconds=.02)
    sleep(.04)
    recovered = jobs.claim("worker-2", lease_seconds=60)
    assert recovered.job_id == old.job_id
    assert recovered.lease_token != old.lease_token
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    target = replace(initial, revision=1)
    token = current_scene_job.set(recovered)
    try:
        assert repository.compare_and_swap(initial, target)
    finally:
        current_scene_job.reset(token)
    jobs.mark_succeeded(recovered)
    assert jobs.get(recovered.job_id, account_id=ctx.account_id).status == SceneJobStatus.SUCCEEDED


def test_configuration_revisions_hot_reload_and_rollback(scene_sql):
    pool, _, _, _, provider, ctx = scene_sql
    cache = SceneConfigCache(provider)
    initial = cache.get(ctx.account_id, "user:alice")
    assert initial.revision == 0
    changed = provider.publish(ctx.account_id, "user:alice", 0, {"analysis_trigger_tokens": 200})
    assert changed.revision == 1
    assert cache.get(ctx.account_id, "user:alice").config.analysis_trigger_tokens == 200
    external = SQLSceneConfigProvider(pool)
    external.publish(ctx.account_id, "user:alice", 1, {"analysis_trigger_tokens": 300})
    assert cache.get(ctx.account_id, "user:alice").config.analysis_trigger_tokens == 300
    rolled_back = external.rollback(ctx.account_id, "user:alice", 1)
    assert rolled_back.revision == 3
    assert rolled_back.config.analysis_trigger_tokens == 200


def test_concurrent_notice_replay_and_claim_have_one_owner(scene_sql):
    _, _, _, jobs, _, ctx = scene_sql

    def enqueue(_):
        jobs.enqueue(ctx.account_id, "user:alice", ["n1", "n2"], token_count=200,
                     trigger_tokens=100, notice_id="same-write")

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(enqueue, range(4)))
        claims = list(workers.map(lambda index: jobs.claim(f"worker-{index}"), range(4)))
    claimed = [job for job in claims if job]
    assert len(claimed) == 1
    assert claimed[0].token_count == 200
    jobs.mark_succeeded(claimed[0])
    assert jobs.summary(ctx.account_id, "user:alice")["pending_nodes"] == 0


def test_failed_job_remains_visible_after_queue_recreation(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    jobs.enqueue(ctx.account_id, "user:alice", ["n1"], token_count=1)
    claimed = jobs.claim("worker")
    jobs.mark_failed(claimed, reason="provider rate limit (429)")
    summary = SQLSceneJobStore(pool).summary(ctx.account_id, "user:alice")
    assert summary["last_job"]["job_id"] == claimed.job_id
    assert summary["last_job"]["status"] == "FAILED"
    assert summary["last_job"]["reason"] == "provider rate limit (429)"
    assert summary["last_job"]["failure_count"] == 1


def test_job_failure_count_survives_storage_restart_and_lease_recovery(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    jobs.enqueue(ctx.account_id, "user:alice", ["n1"], token_count=1)
    first = jobs.claim("worker")
    retry = jobs.mark_retry(first, reason="RuntimeError", delay_seconds=0)
    restarted = SQLSceneJobStore(pool)
    assert restarted.get(first.job_id, account_id=ctx.account_id) == retry
    claimed = restarted.claim("restarted-worker", lease_seconds=.02)
    assert (claimed.attempt, claimed.failure_count) == (2, 1)
    sleep(.04)
    recovered = SQLSceneJobStore(pool).claim("recovered-worker")
    assert (recovered.attempt, recovered.failure_count) == (3, 1)
    completed = restarted.mark_succeeded(recovered)
    summary = SQLSceneJobStore(pool).summary(ctx.account_id, "user:alice")
    assert summary["last_job"] == job_to_dict(completed)
    assert summary["recent_jobs"][0]["failure_count"] == 1


def test_legacy_job_payload_acquires_independent_failure_counter(scene_sql):
    pool, _, _, jobs, _, ctx = scene_sql
    queued = jobs.enqueue(ctx.account_id, "user:alice", ["n1"], token_count=1)
    legacy = job_to_dict(queued)
    legacy.pop("failure_count")
    legacy["attempt"] = 4
    with jobs._transaction(ctx.account_id) as cursor:
        cursor.execute("UPDATE scene_jobs SET payload=%s WHERE account_id=%s AND job_id=%s",
                       (Json(legacy), ctx.account_id, queued.job_id))
    restarted = SQLSceneJobStore(pool)
    assert restarted.summary(ctx.account_id, "user:alice")["recent_jobs"][0]["failure_count"] == 0
    recovered = restarted.claim("recovered-worker")
    assert (recovered.attempt, recovered.failure_count) == (5, 0)
    retry = restarted.mark_retry(recovered, reason="RuntimeError", delay_seconds=0)
    assert retry.failure_count == 1
    final = SQLSceneJobStore(pool).claim("final-worker")
    assert restarted.mark_succeeded(final).failure_count == 1


def test_ready_accounts_share_worker_turns_during_backlog_drain(scene_sql):
    _, _, _, jobs, _, _ = scene_sql
    jobs.enqueue("account-a", "user:alice", ["a1", "a2", "a3"], token_count=3, max_pending_nodes=1)
    jobs.enqueue("account-b", "user:alice", ["b1"], token_count=1)
    first = jobs.claim("worker")
    assert first.account_id == "account-a"
    jobs.mark_succeeded(first)
    second = jobs.claim("worker")
    assert second.account_id == "account-b"
    jobs.mark_succeeded(second)
    remaining = []
    while job := jobs.claim("worker"):
        remaining.extend(job.node_uris)
        jobs.mark_succeeded(job)
    assert remaining == ["a2", "a3"]


def test_configuration_restores_default_and_delivers_after_listener_failure(scene_sql):
    _, _, _, _, provider, ctx = scene_sql
    received = []

    def unavailable(account, owner, snapshot):
        raise RuntimeError("listener temporarily unavailable")

    provider.subscribe(unavailable)
    provider.subscribe(lambda account, owner, snapshot: received.append(snapshot))
    provider.publish(ctx.account_id, "user:alice", 0, {"analysis_trigger_tokens": 200})
    restored = provider.rollback(ctx.account_id, "user:alice", 0)
    assert restored.config == provider.default
    assert [snapshot.revision for snapshot in received] == [1, 2]
    assert provider.get_effective(ctx.account_id, "user:alice") == restored


def test_checked_publication_serializes_canonical_write_then_refreshes(scene_sql):
    pool, fs, repository, _, _, ctx = scene_sql
    writer_fs = SQLContextFS(pool=pool)
    memory = _write(fs, ctx, "source")
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    first = replace(initial, revision=1, scenes=(_scene("planning", memory),))
    checked, connected = Event(), Event()
    errors, writer_pid = [], []
    acquire = writer_fs._get_connection

    def writer_connection():
        conn = acquire()
        with conn.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            writer_pid.append(cursor.fetchone()[0])
        connected.set()
        return conn

    writer_fs._get_connection = writer_connection

    def writer():
        try:
            assert checked.wait(10)
            _write(writer_fs, ctx, "source", version=2)
        except Exception as exc:
            errors.append(exc)

    def check(reader):
        assert reader(memory.uri, ctx).metadata["version"] == 1
        checked.set()
        assert connected.wait(10)
        with repository._transaction(ctx.account_id) as cur:
            for _ in range(1000):
                cur.execute("SELECT bool_or(NOT granted) FROM pg_locks WHERE pid=%s", (writer_pid[0],))
                row = cur.fetchone()
                if row and row[0]:
                    break
                cur.execute("SELECT pg_sleep(0.01)")
            assert row[0]
        return True

    worker = Thread(target=writer)
    worker.start()
    try:
        assert repository.compare_and_swap_checked(initial, first, check)
    finally:
        checked.set()
        worker.join(15)
    assert errors == []
    refreshed = fs.read_node(memory.uri, ctx)
    assert refreshed.metadata["version"] == 2
    second = replace(first, revision=2, scenes=(_scene("planning", refreshed),))
    assert repository.compare_and_swap_checked(first, second, lambda reader: True)
    assert repository.load_snapshot(ctx.account_id, "user:alice") == second


def test_canonical_adapter_publishes_current_fingerprints_across_source_refresh(scene_sql):
    _, fs, repository, _, _, ctx = scene_sql
    memory = _write(fs, ctx, "adapter-source")
    adapter = CanonicalSceneAdapter(repository, fs, IndexedVectors([memory]), Embedder())
    for version in (1, 2):
        if version == 2:
            _write(fs, ctx, "adapter-source", version=version)
        batch = adapter.load_input(ctx.account_id, "user:alice", node_uris=[memory.uri])
        record = _scene("planning", batch.nodes[0])
        published = replace(batch.snapshot, revision=batch.snapshot.revision + 1, scenes=(record,))
        assert adapter.compare_and_swap(batch.snapshot, published, evidence_nodes=batch.nodes)
        snapshot = repository.load_snapshot(ctx.account_id, "user:alice")
        assert snapshot.scenes[0].cluster.members[0].node_version == version
        assert repository.contributions(ctx.account_id, ["user:alice"], [memory.uri])[0][1] == memory.uri
