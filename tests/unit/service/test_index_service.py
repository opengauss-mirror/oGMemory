"""Tests for service.index_service."""

from __future__ import annotations

from unittest.mock import Mock

import service.index_service as index_service_module
from index.scheduler import SchedulerMetrics


class ImmediateThread:
    def __init__(self, target, args=(), name=None, daemon=None):
        self._target = target
        self._args = args
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True
        self._target(*self._args)

    def join(self):
        pass


class FakeScheduler:
    instances = []

    def __init__(self, worker, outbox_store, get_account_ids, worker_id, interval_seconds):
        self.worker = worker
        self.outbox_store = outbox_store
        self.get_account_ids = get_account_ids
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self.started = False
        self.stop_calls = []
        self.metrics = SchedulerMetrics(
            worker_id=worker_id,
            total_processed=3,
            total_succeeded=2,
            total_failed=1,
            total_dlq=1,
        )
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self, wait=True):
        self.stop_calls.append(wait)


def test_start_creates_worker_threads_and_schedulers(monkeypatch):
    FakeScheduler.instances = []
    worker_ctor = Mock(return_value=Mock())
    monkeypatch.setattr(index_service_module, "OutboxWorker", worker_ctor)
    monkeypatch.setattr(index_service_module, "OutboxScheduler", FakeScheduler)
    monkeypatch.setattr(index_service_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(index_service_module.uuid, "uuid4", Mock(side_effect=[Mock(hex="aaaabbbbccccdddd"), Mock(hex="1111222233334444")]))
    service = index_service_module.IndexService(
        outbox_store=Mock(),
        embedder=Mock(),
        vector_index=Mock(),
        get_account_ids=lambda: ["acct-1"],
        interval_seconds=12,
        worker_count=2,
    )

    service.start()

    assert service.is_running is True
    assert len(service.schedulers) == 2
    assert all(s.started for s in service.schedulers)
    assert len(service._threads) == 2
    worker_ctor.assert_called_once()


def test_start_is_idempotent(monkeypatch):
    FakeScheduler.instances = []
    monkeypatch.setattr(index_service_module, "OutboxWorker", Mock(return_value=Mock()))
    monkeypatch.setattr(index_service_module, "OutboxScheduler", FakeScheduler)
    monkeypatch.setattr(index_service_module.threading, "Thread", ImmediateThread)
    warning = Mock()
    monkeypatch.setattr(index_service_module.logger, "warning", warning)
    service = index_service_module.IndexService(
        outbox_store=Mock(),
        embedder=Mock(),
        vector_index=Mock(),
        get_account_ids=lambda: [],
        worker_count=1,
    )

    service.start()
    service.start()

    warning.assert_called_once()


def test_stop_clears_runtime_state(monkeypatch):
    FakeScheduler.instances = []
    monkeypatch.setattr(index_service_module, "OutboxWorker", Mock(return_value=Mock()))
    monkeypatch.setattr(index_service_module, "OutboxScheduler", FakeScheduler)
    monkeypatch.setattr(index_service_module.threading, "Thread", ImmediateThread)
    service = index_service_module.IndexService(
        outbox_store=Mock(),
        embedder=Mock(),
        vector_index=Mock(),
        get_account_ids=lambda: [],
        worker_count=2,
    )
    service.start()

    service.stop(wait=False)

    assert service.is_running is False
    assert service.schedulers == []
    assert service._threads == []
    for scheduler in FakeScheduler.instances:
        assert scheduler.stop_calls == [False]


def test_get_metrics_and_aggregated_stats(monkeypatch):
    FakeScheduler.instances = []
    monkeypatch.setattr(index_service_module, "OutboxWorker", Mock(return_value=Mock()))
    monkeypatch.setattr(index_service_module, "OutboxScheduler", FakeScheduler)
    monkeypatch.setattr(index_service_module.threading, "Thread", ImmediateThread)
    service = index_service_module.IndexService(
        outbox_store=Mock(),
        embedder=Mock(),
        vector_index=Mock(),
        get_account_ids=lambda: [],
        worker_count=2,
    )
    service.start()

    metrics = service.get_metrics()
    stats = service.get_aggregated_stats()

    assert len(metrics) == 2
    assert stats == {
        "worker_count": 2,
        "total_processed": 6,
        "total_succeeded": 4,
        "total_failed": 2,
        "total_dlq": 2,
        "is_running": True,
    }


def test_singleton_helpers_and_signal_handlers(monkeypatch):
    handlers = {}
    exits = []
    monkeypatch.setattr(index_service_module.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(index_service_module.sys, "exit", lambda code: exits.append(code))
    service = Mock()

    created = index_service_module.init_index_service(
        outbox_store=Mock(),
        embedder=Mock(),
        vector_index=Mock(),
        get_account_ids=lambda: ["acct-1"],
        worker_count=1,
    )
    assert index_service_module.get_index_service() is created

    index_service_module.setup_signal_handlers(service)
    handlers[index_service_module.signal.SIGINT](2, None)

    service.stop.assert_called_once_with(wait=True)
    assert exits == [0]


def test_partial_thread_startup_can_be_stopped(monkeypatch):
    class FailingThread(ImmediateThread):
        calls = 0
        def start(self):
            type(self).calls += 1
            if self.calls == 2:
                raise RuntimeError('second thread failed')
            super().start()
    monkeypatch.setattr(index_service_module.threading, 'Thread', FailingThread)
    monkeypatch.setattr(index_service_module, 'OutboxScheduler', FakeScheduler)
    monkeypatch.setattr(index_service_module, 'OutboxWorker', Mock())
    service = index_service_module.IndexService(Mock(), Mock(), Mock(), lambda: ['a'], worker_count=2)
    import pytest
    with pytest.raises(RuntimeError, match='second thread'):
        service.start()
    first = service.schedulers[0]
    service.stop(wait=True)
    assert first.stop_calls == [True]
    assert not service.is_running and service.schedulers == []
