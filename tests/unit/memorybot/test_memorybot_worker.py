import threading
from types import SimpleNamespace

from memorybot.runtime import BotWorker


def test_worker_runs_configured_parallel_slots():
    stop, entered = threading.Event(), threading.Barrier(4)
    owners = []

    def run_once(owner):
        owners.append(owner)
        entered.wait(timeout=5)
        stop.set()
        return True

    worker = SimpleNamespace(
        config=SimpleNamespace(worker_concurrency=4, poll_seconds=0.01),
        run_once=run_once,
    )
    BotWorker.run_forever(worker, stop)
    assert len(set(owners)) == 4


def test_worker_runs_maintenance_and_continues_after_failure():
    from unittest.mock import Mock

    store = Mock()
    store.claim.return_value = None
    worker = BotWorker(store, Mock(), Mock())
    worker.maintenance = Mock()
    worker.maintenance.run_once.side_effect = RuntimeError("maintenance unavailable")

    assert worker.run_once() is False
    worker.maintenance.run_once.assert_called_once_with()
    store.claim.assert_called_once()
    assert not worker._maintenance_lock.locked()
