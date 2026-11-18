"""Additional tests for ContextWriter edge cases."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from commit.context_writer import ContextWriter
from core.errors import ConcurrentModificationError
from core.models import CandidateMemory, ContextNode, RequestContext, WritePlan
from providers.llm import MockLLM


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="u1",
        agent_id="a1",
        session_id="session",
        trace_id="trace",
    )


@pytest.fixture
def candidate() -> CandidateMemory:
    return CandidateMemory(
        category="preference",
        owner_scope="user",
        routing_key="coffee",
        abstract="likes coffee",
        overview="overview",
        content="content",
        confidence=0.9,
    )


@pytest.fixture
def node() -> ContextNode:
    return ContextNode(
        uri="ctx://acct-1/users/u1/memories/preferences/coffee",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:u1",
        abstract="likes coffee",
        overview="overview",
        content="content",
    )


def test_write_candidate_returns_skip_without_building(candidate, ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=WritePlan(action="skip", target_uri="ctx://skip"))
    archive_builder = Mock()
    outbox_store = Mock()
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=archive_builder,
        outbox_store=outbox_store,
    )

    plan = writer.write_candidate(candidate, ctx)

    assert plan.action == "skip"
    archive_builder.build.assert_not_called()
    fs.write_node.assert_not_called()
    outbox_store.register_write.assert_not_called()


def test_write_candidate_swallows_outbox_failures(candidate, node, ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    plan = WritePlan(action="create", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=plan)
    archive_builder = Mock(build=Mock(return_value=node))
    outbox_store = Mock(spec=['register_write', 'register_directory'])
    outbox_store.register_write.side_effect = RuntimeError("write outbox failed")
    outbox_store.register_directory.side_effect = RuntimeError("directory outbox failed")
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=archive_builder,
        outbox_store=outbox_store,
    )

    result = writer.write_candidate(candidate, ctx)

    assert result is plan
    fs.write_node.assert_called_once_with(node, ctx)
    outbox_store.register_write.assert_called_once_with(node, ctx)
    outbox_store.register_directory.assert_called_once_with(node, ctx)


def test_write_candidate_retries_after_concurrent_modification(monkeypatch, candidate, node, ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    fs.write_node.side_effect = [
        ConcurrentModificationError(node.uri, expected_version=1, actual_version=2),
        None,
    ]
    plan = WritePlan(action="create", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=plan)
    outbox_store = Mock(spec=['register_write', 'register_directory'])
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=Mock(build=Mock(return_value=node)),
        outbox_store=outbox_store,
    )
    sleep = Mock()
    monkeypatch.setattr("commit.context_writer.time.sleep", sleep)

    result = writer.write_candidate(candidate, ctx)

    assert result is plan
    assert fs.write_node.call_count == 2
    sleep.assert_called_once_with(0.1)


def test_write_candidate_raises_after_max_retries(monkeypatch, candidate, node, ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    # Exhaust the full retry budget (see write_candidate's max_retries).
    fs.write_node.side_effect = [
        ConcurrentModificationError(node.uri, expected_version=v, actual_version=v + 1)
        for v in range(1, 9)
    ]
    plan = WritePlan(action="merge", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=plan)
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=Mock(build=Mock(return_value=node)),
    )
    sleep = Mock()
    monkeypatch.setattr("commit.context_writer.time.sleep", sleep)

    with pytest.raises(ConcurrentModificationError):
        writer.write_candidate(candidate, ctx)

    assert fs.write_node.call_count == 8
    assert sleep.call_count == 7


def test_write_candidates_continues_when_one_candidate_fails(ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    writer = ContextWriter(fs, llm=MockLLM(), policy_router=policy_router)
    candidates = [
        CandidateMemory("profile", "user", "profile", "", "", "", 0.9),
        CandidateMemory("preference", "user", "coffee", "", "", "", 0.9),
        CandidateMemory("event", "user", "visit", "", "", "", 0.9),
    ]
    plans = [
        WritePlan(action="create", target_uri="ctx://one"),
        WritePlan(action="append", target_uri="ctx://three"),
    ]
    writer.write_candidate = Mock(side_effect=[plans[0], RuntimeError("boom"), plans[1]])

    result = writer.write_candidates(candidates, ctx)

    assert result == plans


def test_write_candidates_parallel_keeps_successful_results(ctx):
    from commit.policy_router import PolicyRouter
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    writer = ContextWriter(fs, llm=MockLLM(), policy_router=policy_router)
    candidates = [
        CandidateMemory("profile", "user", "profile", "", "", "", 0.9),
        CandidateMemory("preference", "user", "coffee", "", "", "", 0.9),
        CandidateMemory("event", "user", "visit", "", "", "", 0.9),
    ]

    def write_candidate(candidate, _ctx):
        if candidate.routing_key == "coffee":
            raise RuntimeError("boom")
        return WritePlan(action="create", target_uri=f"ctx://{candidate.routing_key}")

    writer.write_candidate = write_candidate

    result = writer.write_candidates_parallel(candidates, ctx, max_workers=2)

    assert {plan.target_uri for plan in result} == {"ctx://profile", "ctx://visit"}


def test_write_candidate_lock_wait_respects_request_deadline(candidate, node, ctx):
    """A waiter blocked on a hot node's per-URI lock must fail fast with
    the typed write-timeout once its request budget is gone, not pin its
    thread until the holder releases."""
    import threading

    from commit.policy_router import PolicyRouter
    from commit.write_receipts import WriteDeadlineExceededError
    from core.operation_deadline import operation_deadline
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    plan = WritePlan(action="create", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=plan)
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=Mock(build=Mock(return_value=node)),
        outbox_store=Mock(spec=["register_write", "register_directory"]),
    )

    # Another writer holds the per-URI lock (e.g. inside its LLM merge)
    # and releases only well after this request's budget is gone.
    lock = writer._lock_for_uri(plan.target_uri)
    lock.acquire()
    release = threading.Event()

    def holder():
        release.wait(5.0)
        lock.release()

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        with operation_deadline(0.2):
            with pytest.raises(WriteDeadlineExceededError, match="per-URI write lock"):
                writer.write_candidate(candidate, ctx)
        fs.write_node.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=5)


def test_write_candidate_rechecks_deadline_before_replan_under_lock(
    candidate, node, ctx, monkeypatch
):
    """After acquiring the per-URI lock, the remaining budget is re-checked
    before re-planning — it may have been consumed while waiting."""
    from commit.policy_router import PolicyRouter
    from commit.write_receipts import WriteDeadlineExceededError
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    plan = WritePlan(action="create", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    policy_router.plan = Mock(return_value=plan)
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=Mock(build=Mock(return_value=node)),
        outbox_store=Mock(spec=["register_write", "register_directory"]),
    )

    # The in-lock deadline re-check fails (budget consumed while waiting).
    # Detect "inside the lock" via lock ownership rather than call order —
    # there are several pre-lock check sites.
    import commit.context_writer as cw
    original = cw._check_prewrite_deadline
    writer_lock = writer._lock_for_uri(plan.target_uri)

    def failing_check():
        if writer_lock.locked():
            raise WriteDeadlineExceededError("budget consumed waiting for lock")
        original()

    monkeypatch.setattr(cw, "_check_prewrite_deadline", failing_check)

    with pytest.raises(WriteDeadlineExceededError, match="budget consumed"):
        writer.write_candidate(candidate, ctx)

    # Step-1 plan ran once (before the lock); the in-lock re-plan must
    # never have been reached.
    assert policy_router.plan.call_count == 1
    fs.write_node.assert_not_called()


def test_cas_retry_backoff_is_capped_to_request_budget(candidate, node, ctx):
    """Consecutive CAS conflicts must not sleep past the request deadline —
    the backoff is capped to the remaining budget and the deadline is
    re-checked right after sleeping."""
    import time as _time

    from commit.policy_router import PolicyRouter
    from commit.write_receipts import WriteDeadlineExceededError
    from core.errors import ConcurrentModificationError
    from core.operation_deadline import operation_deadline
    from extraction.schemas.registry import SchemaRegistry

    fs = Mock()
    plan = WritePlan(action="create", target_uri=node.uri)
    registry = SchemaRegistry()
    policy_router = PolicyRouter(fs, registry=registry)
    # Every attempt conflicts: the retry loop must exhaust the *budget*,
    # not the full 12.7s backoff schedule.
    policy_router.plan = Mock(
        side_effect=ConcurrentModificationError(node.uri, 1, 2)
    )
    writer = ContextWriter(
        fs,
        llm=MockLLM(),
        policy_router=policy_router,
        archive_builder=Mock(),
        outbox_store=Mock(),
    )

    start = _time.monotonic()
    with operation_deadline(1.0):
        with pytest.raises(WriteDeadlineExceededError):
            writer.write_candidate(candidate, ctx)
    # Uncapped, the schedule sleeps 0.1+0.2+0.4+0.8 = 1.5s before the
    # loop-top check fires; capped, the write unwinds at ~the deadline.
    assert _time.monotonic() - start < 1.25
