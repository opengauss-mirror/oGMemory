from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from scene.jobs import InMemorySceneJobStore, SceneJobStatus, job_from_dict, job_to_dict


@pytest.mark.parametrize("supply", ["writes", "backlog"])
def test_claim_rotates_busy_accounts_and_owners(supply):
    store = InMemorySceneJobStore()
    scopes = [("acct", "user:a"), ("acct", "user:b"), ("other", "user:a")]
    for account, owner in scopes:
        nodes = [f"n{i}" for i in range(4)] if supply == "backlog" else ["n0"]
        store.enqueue(account, owner, nodes, token_count=len(nodes), max_pending_nodes=1)
    claimed_scopes = []
    for index in range(12):
        job = store.claim("worker")
        assert job is not None
        claimed_scopes.append((job.account_id, job.owner_space))
        if supply == "writes":
            store.enqueue(job.account_id, job.owner_space, [f"fresh-{index}"],
                          token_count=1, max_pending_nodes=1)
        store.mark_succeeded(job)
    assert claimed_scopes == scopes * 4


def test_enqueue_coalesces_until_trigger_and_claims_atomically():
    store = InMemorySceneJobStore()
    assert store.enqueue("acct", "user:a", ["n1"], token_count=3, trigger_tokens=10) is None
    job = store.enqueue("acct", "user:a", ["n2"], token_count=7, trigger_tokens=10)
    assert job is not None
    assert job.node_uris == ("n1", "n2")
    claimed = store.claim("worker-1")
    assert claimed is not None
    assert claimed.status == SceneJobStatus.RUNNING
    assert claimed.attempt == 1
    assert store.claim("worker-2") is None


@pytest.mark.parametrize("readiness", ["max_wait", "retry", "lease"])
def test_rotation_serves_other_scopes_then_recovers_due_work(readiness):
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = InMemorySceneJobStore(clock=lambda: clock[0])
    store.enqueue("acct", "user:due", ["due"], token_count=1,
                  trigger_tokens=2 if readiness == "max_wait" else 1, max_wait_seconds=10)
    original = None
    if readiness != "max_wait":
        original = store.claim("first-worker", lease_seconds=10)
        if readiness == "retry":
            store.mark_retry(original, reason="temporary failure", delay_seconds=10)
    for owner in ("user:busy", "user:ready"):
        store.enqueue("acct", owner, [owner], token_count=1)
    busy = store.claim("worker")
    assert busy.owner_space == "user:busy"
    store.enqueue("acct", "user:busy", ["next"], token_count=1)
    store.mark_succeeded(busy)
    ready = store.claim("worker")
    assert ready.owner_space == "user:ready"
    store.mark_succeeded(ready)
    clock[0] += timedelta(seconds=10)
    recovered = store.claim("worker")
    assert recovered.owner_space == "user:due"
    assert recovered.node_uris == ("due",)
    if original is not None:
        assert recovered.job_id == original.job_id
        assert recovered.attempt == 2
        assert recovered.lease_token != original.lease_token
    assert store.mark_succeeded(recovered).status == SceneJobStatus.SUCCEEDED
    following = store.claim("worker")
    assert following.node_uris == ("next",)
    store.mark_succeeded(following)
    assert store.claim("worker") is None
    store.enqueue("late", "user:new", ["fresh"], token_count=1)
    late = store.claim("worker")
    assert (late.account_id, late.owner_space, late.node_uris) == ("late", "user:new", ("fresh",))
    store.mark_succeeded(late)


def test_concurrent_workers_claim_distinct_scopes_across_rounds():
    store = InMemorySceneJobStore()
    scopes = [("acct", f"user:{i}") for i in range(8)]
    for account, owner in scopes:
        store.enqueue(account, owner, [f"n{i}" for i in range(5)],
                      token_count=5, max_pending_nodes=1)
    with ThreadPoolExecutor(max_workers=8) as workers:
        for _ in range(5):
            claims = list(workers.map(lambda i: store.claim(f"worker-{i}"), range(8)))
            assert {(job.account_id, job.owner_space) for job in claims} == set(scopes)
            assert len({job.job_id for job in claims}) == 8
            completed = list(workers.map(store.mark_succeeded, claims))
            assert all(job.status == SceneJobStatus.SUCCEEDED for job in completed)
    assert store.claim("worker") is None


def test_waiting_evidence_joins_next_threshold_window():
    store = InMemorySceneJobStore()
    job = store.enqueue("acct", "user:a", ["n1"], token_count=10, trigger_tokens=1)
    assert job is not None
    claimed = store.claim("worker")
    assert claimed is not None
    waiting = store.mark_waiting(claimed, reason="insufficient evidence")
    assert waiting.status == SceneJobStatus.WAITING
    accumulating = store.enqueue("acct", "user:a", ["n2"], token_count=40, trigger_tokens=100)
    assert accumulating.status == SceneJobStatus.WAITING
    assert store.summary("acct", "user:a")["pending_tokens"] == 40
    reopened = store.enqueue("acct", "user:a", ["n3"], token_count=60, trigger_tokens=100)
    assert reopened is not None
    assert reopened.status == SceneJobStatus.PENDING
    assert reopened.node_uris == ("n2", "n1", "n3")


def test_write_during_running_job_is_scheduled_after_current_refresh():
    store = InMemorySceneJobStore()
    first = store.enqueue("acct", "user:a", ["n1"], token_count=1, trigger_tokens=1)
    assert first is not None
    claimed = store.claim("worker")
    assert claimed is not None

    assert store.enqueue("acct", "user:a", ["n2"], token_count=100, trigger_tokens=100) == claimed
    completed = store.mark_succeeded(claimed)
    assert completed.status == SceneJobStatus.SUCCEEDED

    next_job = store.claim("worker-2")
    assert next_job is not None
    assert next_job.node_uris == ("n2",)


def test_write_during_running_wait_is_reopened_with_new_evidence():
    store = InMemorySceneJobStore()
    first = store.enqueue("acct", "user:a", ["n1"], token_count=1, trigger_tokens=1)
    assert first is not None
    claimed = store.claim("worker")
    assert claimed is not None
    assert store.enqueue("acct", "user:a", ["n2"], token_count=100, trigger_tokens=100) == claimed

    waiting = store.mark_waiting(claimed, reason="insufficient evidence")
    assert waiting.status == SceneJobStatus.WAITING
    reopened = store.claim("next-worker")
    assert reopened.status == SceneJobStatus.RUNNING
    assert reopened.node_uris == ("n2", "n1")


def test_backlog_drains_in_bounded_complete_windows():
    store = InMemorySceneJobStore()
    nodes = [f"node-{index:03}" for index in range(71)]
    store.enqueue("acct", "user:a", nodes, token_count=710, trigger_tokens=100, max_pending_nodes=16)
    processed = []
    while job := store.claim("worker"):
        assert len(job.node_uris) <= 16
        processed.extend(job.node_uris)
        store.mark_succeeded(job)
    assert processed == nodes
    assert store.summary("acct", "user:a")["pending_nodes"] == 0


def test_max_wait_releases_small_fresh_window():
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = InMemorySceneJobStore(clock=lambda: clock[0])
    store.enqueue("acct", "user:a", ["n1"], token_count=4, trigger_tokens=10, max_wait_seconds=60)
    clock[0] += timedelta(seconds=60)
    job = store.claim("worker")
    assert job.node_uris == ("n1",)


def test_lease_recovery_issues_fresh_fence_and_preserves_window():
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = InMemorySceneJobStore(clock=lambda: clock[0])
    store.enqueue("acct", "user:a", ["n1"], token_count=1)
    original = store.claim("worker-1", lease_seconds=10)
    clock[0] += timedelta(seconds=11)
    recovered = store.claim("worker-2", lease_seconds=10)
    assert recovered.job_id == original.job_id
    assert recovered.node_uris == original.node_uris
    assert recovered.lease_token != original.lease_token
    assert recovered.attempt == 2
    assert recovered.failure_count == 0
    renewed = store.renew(recovered, lease_seconds=30)
    assert renewed.lease_until == clock[0] + timedelta(seconds=30)
    assert store.mark_succeeded(renewed).status == SceneJobStatus.SUCCEEDED


def test_job_payload_roundtrip_preserves_failure_count_and_loads_legacy_claims():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1"], token_count=1)
    claimed = store.claim("worker")
    retry = store.mark_retry(claimed, reason="RuntimeError", delay_seconds=0)
    assert job_from_dict(job_to_dict(retry)) == retry
    legacy = job_to_dict(retry)
    legacy.pop("failure_count")
    legacy["attempt"] = 7
    restored = job_from_dict(legacy)
    assert (restored.attempt, restored.failure_count) == (7, 0)


def test_pending_evidence_continues_after_success():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1", "n2"], token_count=10)
    job = store.claim("worker")
    store.mark_succeeded(job, remaining_uris=["n2"])
    store.enqueue("acct", "user:a", ["n3"], token_count=10)
    next_job = store.claim("worker")
    assert next_job.node_uris == ("n3", "n2")


def test_capacity_deferred_tail_continues_after_success():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1", "n2", "n3", "n4"], token_count=100)
    first = store.claim("worker")
    store.mark_succeeded(first, reason="CREATE", remaining_uris=["n4"], deferred_uris=["n4"])
    following = store.claim("worker")
    assert following.node_uris == ("n4",)
    assert following.retained_node_uris == ()
    store.mark_waiting(following, reason="Additional evidence will complete this scene")
    summary = store.summary("acct", "user:a")
    assert (summary["pending_nodes"], summary["retained_nodes"]) == (0, 1)


def test_claimed_retained_provenance_roundtrips_and_fresh_write_advances():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["old"], token_count=1)
    store.mark_waiting(store.claim("worker"))
    store.enqueue("acct", "user:a", ["fresh"], token_count=1)
    claimed = store.claim("worker")
    assert claimed.retained_node_uris == ("old",)
    assert job_from_dict(job_to_dict(claimed)) == claimed
    store.enqueue("acct", "user:a", ["old"], token_count=1)
    store.mark_waiting(claimed, remaining_uris=claimed.node_uris, deferred_uris=["old"])
    refreshed = store.claim("worker")
    assert refreshed.node_uris[0] == "old"
    assert refreshed.retained_node_uris == ("fresh",)


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_successful_progress_continues_remaining_evidence_with_fresh_writes(action):
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1", "n2", "n3", "n4"], token_count=100,
                  trigger_tokens=100, max_pending_nodes=4)
    first = store.claim("worker")
    store.enqueue("acct", "user:a", ["n5"], token_count=1, trigger_tokens=100, max_pending_nodes=4)
    store.mark_succeeded(first, reason=action, remaining_uris=["n3", "n4"], min_evidence_nodes=2)
    summary = store.summary("acct", "user:a")
    assert (summary["pending_nodes"], summary["pending_tokens"], summary["retained_nodes"]) == (3, 1, 0)
    following = store.claim("worker")
    assert following.node_uris == ("n3", "n4", "n5")
    assert following.token_count == 1
    store.mark_succeeded(following)
    assert store.summary("acct", "user:a")["pending_nodes"] == 0


def test_unchanged_remaining_window_waits_for_new_evidence():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1", "n2", "n3"], token_count=100, trigger_tokens=100)
    first = store.claim("worker")
    store.mark_succeeded(first, remaining_uris=first.node_uris)
    summary = store.summary("acct", "user:a")
    assert (summary["pending_nodes"], summary["retained_nodes"], summary["active_job"]) == (0, 3, None)
    assert store.flush("acct", "user:a") is None
    store.enqueue("acct", "user:a", ["n4"], token_count=100, trigger_tokens=100)
    following = store.claim("worker")
    assert following.node_uris == ("n4", "n1", "n2", "n3")


def test_flush_releases_fresh_tail_and_retains_waiting_evidence():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1"], token_count=2, trigger_tokens=100)
    reopened = store.flush("acct", "user:a")
    assert reopened.status == SceneJobStatus.PENDING
    claimed = store.claim("worker")
    waiting = store.mark_waiting(claimed, reason="context accumulates")
    assert store.flush("acct", "user:a") == waiting
    assert store.summary("acct", "user:a")["retained_nodes"] == 1


def test_notice_replay_counts_each_canonical_write_once():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["n1"], token_count=3, trigger_tokens=10, notice_id="write-1")
    store.enqueue("acct", "user:a", ["n1"], token_count=3, trigger_tokens=10, notice_id="write-1")
    assert store.summary("acct", "user:a")["pending_tokens"] == 3
    store.enqueue("acct", "user:a", ["n2"], token_count=7, trigger_tokens=10, notice_id="write-2")
    job = store.claim("worker")
    assert job.node_uris == ("n1", "n2")
    store.mark_succeeded(job)
    store.enqueue("acct", "user:a", ["n1"], token_count=3, trigger_tokens=10, notice_id="write-1")
    assert store.summary("acct", "user:a")["pending_tokens"] == 0


def test_node_count_releases_window_and_rotates_retained_evidence():
    store = InMemorySceneJobStore()
    store.enqueue("acct", "user:a", ["old1", "old2"], token_count=2, max_pending_nodes=2)
    first = store.claim("worker")
    store.mark_waiting(first)
    store.enqueue("acct", "user:a", ["new1", "new2"], token_count=2,
                  trigger_tokens=100, max_pending_nodes=2)
    second = store.claim("worker")
    assert second.node_uris == ("new1", "old1")
    store.mark_succeeded(second)
    third = store.claim("worker")
    assert third.node_uris == ("new2", "old2")


def test_concurrent_notice_delivery_and_claim_complete_one_window():
    store = InMemorySceneJobStore()

    def deliver(_):
        store.enqueue("acct", "user:a", ["n1", "n2"], token_count=10,
                      trigger_tokens=10, notice_id="canonical-write")

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(deliver, range(4)))
        claims = list(workers.map(lambda i: store.claim(f"worker-{i}"), range(4)))
    claimed = [job for job in claims if job is not None]
    assert len(claimed) == 1
    assert claimed[0].token_count == 10
    assert claimed[0].node_uris == ("n1", "n2")
    completed = store.mark_succeeded(claimed[0])
    assert store.get(completed.job_id) == completed
    assert store.summary("acct", "user:a")["last_job"] == job_to_dict(completed)


def test_same_notice_is_processed_independently_per_account_and_owner():
    store = InMemorySceneJobStore()
    scopes = [("acct", "user:a"), ("acct", "user:b"), ("other", "user:a")]
    for account, owner in scopes:
        store.enqueue(account, owner, ["n1"], token_count=1, notice_id="write-1")
    completed_scopes = []
    while job := store.claim("worker"):
        completed_scopes.append((job.account_id, job.owner_space))
        assert job.node_uris == ("n1",)
        store.mark_succeeded(job)
    assert completed_scopes == scopes


def test_retry_deadline_and_lease_renewal_preserve_failure_history():
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = InMemorySceneJobStore(clock=lambda: clock[0])
    store.enqueue("acct", "user:a", ["n1"], token_count=1)
    claimed = store.claim("worker", lease_seconds=10)
    renewed = store.renew(claimed, lease_seconds=30)
    assert store.due_at("acct", "user:a") == renewed.lease_until
    retry = store.mark_retry(renewed, reason="temporary failure", delay_seconds=5)
    assert retry.failure_count == 1
    assert store.due_at("acct", "user:a") == clock[0] + timedelta(seconds=5)
    clock[0] += timedelta(seconds=5)
    recovered = store.claim("recovered-worker")
    assert recovered.job_id == claimed.job_id
    assert recovered.node_uris == claimed.node_uris
    assert recovered.attempt == 2
    completed = store.mark_succeeded(recovered)
    assert completed.status == SceneJobStatus.SUCCEEDED
    assert completed.failure_count == 1
