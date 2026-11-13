"""Legacy polling follows completed work, retry waits and service stop."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def run_polling(monkeypatch, results, *, stop_after_polls=None, outbox_store=None):
    service = MemoryService(config=OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        sql_dialect="opengauss", sql_connection_string="host=localhost dbname=test",
        directory_summary_enabled=False,
    ))
    timeline = []
    outcomes = iter(results)
    polls = 0

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, seconds):
            timeline.append(("wait", seconds))
            self.stopped = polls >= len(results)
            return self.stopped

    stop = StopEvent()
    service._outbox_stop = stop

    class Worker:
        def run_once(self, *, outbox_store, account_ids):
            nonlocal polls
            polls += 1
            timeline.append("poll")
            if polls == stop_after_polls:
                stop.stopped = True
            return next(outcomes)

    targets = []

    class Thread:
        def __init__(self, *, target, daemon, name):
            targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(service, "_get_context_fs", lambda: object())
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs:
        outbox_store if outbox_store is not None else SimpleNamespace(supports_batch_claim=False))
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: Mock())
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: Mock())
    monkeypatch.setattr(service, "_graph_projection_components", lambda: (None, None))
    monkeypatch.setattr("index.outbox_worker.OutboxWorker", lambda **kwargs: Worker())
    monkeypatch.setattr("server.memory_service.threading.Thread", Thread)
    service._start_outbox_worker()
    targets[0]()
    assert stop.is_set()
    return timeline


@pytest.mark.parametrize("progress", [
    {"succeeded": 1},
    {"edge_projections": 3, "deferred": 4},
])
def test_completed_work_drains_until_idle(monkeypatch, progress):
    timeline = run_polling(monkeypatch, [progress, {}])
    assert timeline == ["poll", "poll", ("wait", 5)]


@pytest.mark.parametrize("waiting", [
    {"succeeded": 1, "failed": 1},
    {"edge_projections": 1, "edge_projection_failures": 1},
    {"succeeded": 1, "moved_to_dlq": 1},
    {"processed": 4, "deferred": 4},
    {"processed": 4, "skipped": 4},
])
def test_scheduled_wait_resumes_due_work(monkeypatch, waiting):
    timeline = run_polling(monkeypatch, [waiting, {"succeeded": 1}, {}])
    assert timeline == ["poll", ("wait", 5), "poll", "poll", ("wait", 5)]


def test_idle_wait_completes_service_stop(monkeypatch):
    assert run_polling(monkeypatch, [{}]) == ["poll", ("wait", 5)]


def test_completed_work_observes_service_stop(monkeypatch):
    timeline = run_polling(monkeypatch, [{"succeeded": 1}], stop_after_polls=1)
    assert timeline == ["poll"]


def test_future_pending_deadline_wakes_then_returns_to_idle(monkeypatch):
    accounts = []
    delays = iter([0.75, None])

    def future_pending_delay(account_ids):
        accounts.append(list(account_ids))
        return next(delays)

    store = SimpleNamespace(supports_batch_claim=False, future_pending_delay=future_pending_delay)
    timeline = run_polling(monkeypatch, [{}, {"succeeded": 1}, {}], outbox_store=store)
    assert timeline == ["poll", ("wait", 0.75), "poll", "poll", ("wait", 5)]
    assert accounts == [[OgMemConfig().account_id], [OgMemConfig().account_id]]


def test_long_future_deadline_keeps_regular_idle_sweep(monkeypatch):
    store = SimpleNamespace(supports_batch_claim=False, future_pending_delay=lambda accounts: 12.)
    assert run_polling(monkeypatch, [{}], outbox_store=store) == ["poll", ("wait", 5)]


def test_retry_backoff_then_future_deadline_and_idle(monkeypatch):
    delays = iter([0.5, None])
    store = SimpleNamespace(supports_batch_claim=False, future_pending_delay=lambda accounts: next(delays))
    timeline = run_polling(monkeypatch, [
        {"succeeded": 1, "failed": 1}, {}, {"succeeded": 1}, {},
    ], outbox_store=store)
    assert timeline == ["poll", ("wait", 5), "poll", ("wait", 0.5), "poll", "poll", ("wait", 5)]
