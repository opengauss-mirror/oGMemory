import threading
from types import SimpleNamespace

from core.models import RequestContext
from service.agent_turn_scheduler import AgentTurnQueue
from service.extraction_retry import AGENT_TURN_TASK, ExtractionRetryWorker


def _context(account, user, session):
    return RequestContext(account, user, "assistant", session, f"trace-{session}")


def test_ready_sessions_rotate_by_account_and_user():
    queue = AgentTurnQueue()
    for account, user, session, ready_at in [
        ("team", "alice", "alice-2", 102),
        ("team", "alice", "alice-1", 101),
        ("team", "bob", "bob-1", 103),
        ("partner", "carol", "carol-1", 104),
    ]:
        queue.wake(_context(account, user, session), ready_at, 0, 0)

    observed = []
    for _ in range(4):
        turn = queue.pop(0)
        observed.append(turn.ctx.session_id)
        queue.finish(turn, None, 0, 105)

    assert observed == ["alice-1", "carol-1", "bob-1", "alice-2"]


def test_due_retry_and_repeat_wake_follow_the_saved_attempt_time():
    queue = AgentTurnQueue()
    alice = _context("team", "alice", "backoff")
    bob = _context("team", "bob", "independent")
    queue.wake(alice, 100, 0, 0)
    queue.finish(queue.pop(0), 3, 0, 101)

    queue.wake(alice, 102, 1, 1)
    queue.wake(bob, 103, 1, 1)
    independent = queue.pop(1)
    assert independent.ctx == bob
    queue.finish(independent, None, 1, 104)

    assert queue.next_deadline() == 3
    assert queue.pop(3).ctx == alice


def test_dispatch_rotates_to_new_user_and_rechecks_active_session():
    worker = ExtractionRetryWorker(SimpleNamespace(_cfg=SimpleNamespace(agent_turn_workers=1)))
    first_started = threading.Event()
    release_first = threading.Event()
    all_started = threading.Event()
    calls = []
    first = _context("team", "alice", "alice-first")
    later = _context("team", "alice", "alice-later")
    quiet = _context("team", "bob", "bob-first")

    def run(ctx, task_id):
        assert task_id == AGENT_TURN_TASK
        calls.append(ctx.session_id)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(5)
        if len(calls) == 4:
            all_started.set()

    worker.run_once = run
    try:
        worker.wake(first, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": 10})
        assert first_started.wait(5)
        worker.wake(first, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": 10,
                            "agent_async": True})
        worker.wake(later, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": 20})
        worker.wake(quiet, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": 30})
        worker.wake(first, {"id": AGENT_TURN_TASK, "status": "queued", "ready_at_ms": 40})
        release_first.set()
        assert all_started.wait(5)
        assert calls == ["alice-first", "bob-first", "alice-later", "alice-first"]
    finally:
        release_first.set()
        worker.close(timeout=5)
