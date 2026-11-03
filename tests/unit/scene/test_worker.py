from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest

from providers.token_tracker import _active_usage_scope
from scene.config import InMemorySceneConfigProvider, SceneConfig, SceneConfigCache
from scene.decision_contracts import SceneDecisionResult, SceneModelOutputError
from scene.jobs import InMemorySceneJobStore, SceneJobStatus, SceneWorker, current_scene_job
from scene.models import SceneSnapshot


class Engine:
    config = SceneConfig(retry_limit=2)

    def __init__(self, action="CREATE"):
        self.action = action
        self.windows = []

    def refresh(self, store, account_id, owner_space, *, node_uris):
        self.windows.append(tuple(node_uris))
        return SceneDecisionResult(SceneSnapshot(account_id, owner_space, 1), self.action,
                                   pending_member_uris=tuple(node_uris) if self.action == "WAIT" else ())


def test_worker_waits_then_processes_accumulated_evidence():
    jobs = InMemorySceneJobStore()
    jobs.enqueue("acct", "user:a", ["n1"], token_count=1)
    engine = Engine("WAIT")
    worker = SceneWorker(jobs, engine, object())
    waiting = worker.process_next("worker")
    assert waiting.job.status == SceneJobStatus.WAITING
    jobs.enqueue("acct", "user:a", ["n2"], token_count=1)
    engine.action = "CREATE"
    done = worker.process_next("worker")
    assert done.job.status == SceneJobStatus.SUCCEEDED
    assert engine.windows == [("n1",), ("n2", "n1")]


class CapacityWindowEngine(Engine):
    def __init__(self, capacity=3):
        super().__init__("WAIT")
        self.capacity = capacity
        self.reviewed = []

    def refresh(self, store, account_id, owner_space, *, node_uris):
        self.windows.append(tuple(node_uris))
        selected = tuple(node_uris[:self.capacity])
        self.reviewed.extend(selected)
        return SceneDecisionResult(
            SceneSnapshot(account_id, owner_space, 1), self.action,
            pending_member_uris=tuple(node_uris) if self.action == "WAIT" else tuple(node_uris[self.capacity:]),
            reviewed_member_uris=selected, deferred_member_uris=tuple(node_uris[self.capacity:]),
        )


def exercise_capacity_wait_windows(jobs, reopen=lambda jobs: jobs):
    engine = CapacityWindowEngine()
    original = [f"n{index}" for index in range(6)]
    jobs.enqueue("acct", "user:a", original, token_count=100, max_pending_nodes=6)
    outcomes = []
    for round_number in range(12):
        jobs = reopen(jobs)
        outcome = SceneWorker(jobs, engine, object()).process_next("worker")
        if outcome is None:
            break
        outcomes.append(outcome)
        if round_number == 0:
            jobs.enqueue("acct", "user:a", ["n6", "n7"], token_count=1,
                         trigger_tokens=100, max_pending_nodes=6)
    assert len(outcomes) == 4
    assert all(outcome.job.status == SceneJobStatus.WAITING for outcome in outcomes)
    assert engine.windows[:2] == [tuple(original), ("n3", "n0", "n4", "n1", "n5", "n2")]
    assert outcomes[1].job.retained_node_uris == ("n0", "n1", "n2")
    assert set(engine.reviewed) == {*original, "n6", "n7"}
    summary = reopen(jobs).summary("acct", "user:a")
    assert (summary["pending_nodes"], summary["retained_nodes"]) == (0, 8)
    return outcomes


def test_capacity_wait_windows_review_all_fresh_evidence_and_settle():
    exercise_capacity_wait_windows(InMemorySceneJobStore())


def test_capacity_blocked_input_remains_available_for_a_larger_window():
    jobs = InMemorySceneJobStore()
    nodes = ("n1", "n2", "n3")
    jobs.enqueue("acct", "user:a", nodes, token_count=1)
    engine = CapacityWindowEngine(capacity=0)
    blocked = SceneWorker(jobs, engine, object()).process_next("worker")
    saved = jobs.get(blocked.job.job_id)
    assert saved.status == SceneJobStatus.FAILED
    assert saved.node_uris == nodes
    assert saved.reason == "scene capacity blocked: requested evidence exceeds the window"
    assert jobs.due_at("acct", "user:a") is None
    engine.capacity = 3
    engine.action = "CREATE"
    jobs.enqueue("acct", "user:a", saved.node_uris, token_count=1)
    completed = SceneWorker(jobs, engine, object()).process_next("worker")
    assert completed.job.status == SceneJobStatus.SUCCEEDED
    assert completed.decision.reviewed_member_uris == nodes


@pytest.mark.parametrize(("owner", "usage_owner"), [
    ("user:a", "user:a"), ("user_a", "user:a"),
    ("agent:a", "agent:a"), ("agent_a", "agent:a"),
])
def test_worker_attributes_usage_and_publication_lease(owner, usage_owner):
    observed = []

    class ScopedEngine(Engine):
        def refresh(self, store, account_id, owner_space, *, node_uris):
            observed.append((_active_usage_scope.get(), current_scene_job.get()))
            return super().refresh(store, account_id, owner_space, node_uris=node_uris)

    jobs = InMemorySceneJobStore()
    jobs.enqueue("acct", owner, ["n1"], token_count=1)
    completed = SceneWorker(jobs, ScopedEngine(), object()).process_next("worker")
    usage, claimed = observed[0]
    assert usage == ("acct", usage_owner)
    assert claimed.job_id == completed.job.job_id
    assert claimed.lease_token == completed.job.lease_token
    assert current_scene_job.get() is None
    assert _active_usage_scope.get() is None


@pytest.mark.parametrize("error", [
    RuntimeError("temporary provider failure"),
    SceneModelOutputError("preflight", ({"field": "status", "type": "literal_error"},)),
])
def test_worker_recovers_with_independent_failure_budget(error):
    if isinstance(error, SceneModelOutputError):
        error.__context__ = ValueError("model field contains 429")
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    jobs = InMemorySceneJobStore(clock=lambda: clock[0])
    jobs.enqueue("acct", "user:a", ["n1"], token_count=1)
    for index in range(3):
        claimed = jobs.claim(f"worker-{index}", lease_seconds=10)
        if index < 2:
            clock[0] += timedelta(seconds=11)

    class RecoveringEngine(Engine):
        def refresh(self, store, account_id, owner_space, *, node_uris):
            if not self.windows:
                self.windows.append(tuple(node_uris))
                raise error
            return super().refresh(store, account_id, owner_space, node_uris=node_uris)

    worker = SceneWorker(jobs, RecoveringEngine(), object(), retry_delay_seconds=5)
    retry = worker.process(claimed).job
    assert (retry.status, retry.attempt, retry.failure_count) == (SceneJobStatus.PENDING, 3, 1)
    assert retry.next_run_at == clock[0] + timedelta(seconds=5)
    clock[0] += timedelta(seconds=5)
    completed = worker.process_next("recovered-worker").job
    assert (completed.status, completed.attempt, completed.failure_count) == (SceneJobStatus.SUCCEEDED, 4, 1)
    assert completed.node_uris == claimed.node_uris


def test_worker_retains_rate_limit_history_before_explicit_recreation():
    class ProviderError(RuntimeError):
        status_code = 429

    class LimitedEngine(Engine):
        def refresh(self, store, account_id, owner_space, *, node_uris):
            raise ProviderError("rate limit")

    jobs = InMemorySceneJobStore()
    jobs.enqueue("acct", "user:a", ["n1"], token_count=1)
    worker = SceneWorker(jobs, LimitedEngine(), object())
    result = worker.process_next("worker")
    assert result.halted and worker.halted
    assert jobs.summary("acct", "user:a")["last_job"]["reason"] == "provider rate limit (429)"
    jobs.enqueue("acct", "user:a", ["n2"], token_count=1)
    completed = SceneWorker(jobs, Engine(), object()).process_next("recovered-worker")
    assert completed.job.status == SceneJobStatus.SUCCEEDED
    assert completed.job.node_uris == ("n2",)


def test_worker_renews_lease_during_analysis():
    renewed, release = Event(), Event()

    class Jobs(InMemorySceneJobStore):
        def renew(self, job, *, lease_seconds):
            result = super().renew(job, lease_seconds=lease_seconds)
            renewed.set()
            return result

    class SlowEngine(Engine):
        def refresh(self, store, account_id, owner_space, *, node_uris):
            assert release.wait(3)
            return super().refresh(store, account_id, owner_space, node_uris=node_uris)

    jobs = Jobs()
    jobs.enqueue("acct", "user:a", ["n1"], token_count=1)
    worker = SceneWorker(jobs, SlowEngine(), object(), lease_seconds=.3)
    results = []
    thread = Thread(target=lambda: results.append(worker.process_next("worker")))
    thread.start()
    try:
        assert renewed.wait(2)
    finally:
        release.set()
        thread.join(3)
    assert results[0].job.status == SceneJobStatus.SUCCEEDED


@pytest.mark.parametrize("cached", [False, True])
def test_worker_binds_fresh_owner_configuration_for_each_window(cached):
    provider = InMemorySceneConfigProvider()
    provider.publish("acct", "user:a", 0, {"min_evidence_nodes": 2})
    used = []

    class ConfiguredEngine(Engine):
        def for_config(self, config):
            used.append(config.min_evidence_nodes)
            return self

    jobs = InMemorySceneJobStore()
    engine = ConfiguredEngine()
    worker = SceneWorker(jobs, engine, object(), config_provider=SceneConfigCache(provider) if cached else provider)
    for count in (2, 4):
        if count == 4:
            provider.publish("acct", "user:a", 1, {"min_evidence_nodes": count})
        jobs.enqueue("acct", "user:a", [f"n{count}"], token_count=1)
        assert worker.process_next("worker").job.status == SceneJobStatus.SUCCEEDED
    assert used == [2, 4]


def exercise_mixed_capacity_checkpoint(jobs, reopen=lambda jobs: jobs):
    class MixedEngine(Engine):
        def refresh(self, store, account_id, owner_space, *, node_uris):
            self.windows.append(tuple(node_uris))
            return SceneDecisionResult(
                SceneSnapshot(account_id, owner_space, 0), "WAIT",
                pending_member_uris=tuple(node_uris),
                reviewed_member_uris=tuple(uri for uri in node_uris if uri == "normal"),
                deferred_member_uris=("oversized",),
            )

    jobs.enqueue("acct", "user:a", ["oversized", "normal"], token_count=1)
    engine = MixedEngine()
    first = SceneWorker(jobs, engine, object()).process_next("worker")
    assert first.job.status == SceneJobStatus.WAITING
    jobs = reopen(jobs)
    stopped = SceneWorker(jobs, engine, object()).process_next("worker")
    assert stopped.job.status == SceneJobStatus.FAILED
    assert stopped.job.failure_count == 1
    assert stopped.job.retained_node_uris == ("normal",)
    assert len(engine.windows) == 2
    jobs = reopen(jobs)
    assert jobs.get(stopped.job.job_id).node_uris == ("oversized", "normal")
    assert jobs.summary("acct", "user:a")["pending_nodes"] == 0
    jobs.enqueue("acct", "user:a", stopped.job.node_uris, token_count=1)
    resumed = SceneWorker(jobs, Engine(), object()).process_next("worker")
    assert resumed.job.status == SceneJobStatus.SUCCEEDED
    assert set(resumed.job.node_uris) == {"normal", "oversized"}


def test_mixed_capacity_checkpoint_recovers_after_window_expansion():
    exercise_mixed_capacity_checkpoint(InMemorySceneJobStore())


def exercise_retained_publication_continuation(jobs, reopen=lambda jobs: jobs):
    retained, fresh = ("a", "b", "c"), ("d", "e", "f")
    jobs.enqueue("acct", "user:a", retained, token_count=1)
    jobs.mark_waiting(jobs.claim("worker"))
    jobs = reopen(jobs)
    jobs.enqueue("acct", "user:a", fresh, token_count=1)
    claimed = jobs.claim("worker")
    assert claimed.retained_node_uris == retained
    jobs.mark_succeeded(claimed, remaining_uris=retained, deferred_uris=("c",))
    jobs = reopen(jobs)
    completed = SceneWorker(jobs, Engine(), object()).process_next("worker")
    assert completed.job.status == SceneJobStatus.SUCCEEDED
    assert set(completed.job.node_uris) == set(retained)
    summary = reopen(jobs).summary("acct", "user:a")
    assert (summary["pending_nodes"], summary["retained_nodes"]) == (0, 0)


def test_retained_publication_continues_remaining_evidence():
    exercise_retained_publication_continuation(InMemorySceneJobStore())
