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
