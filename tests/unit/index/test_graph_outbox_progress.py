"""Projection wakeups survive dependency waits and acknowledge completion."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from core.models import OutboxEvent
from index.outbox_worker import OutboxWorker


@pytest.mark.parametrize("batch_claim", [True, False])
def test_projection_wait_preserves_retry_budget_until_restart_completion(monkeypatch, batch_claim):
    now = [datetime(2026, 11, 6, tzinfo=UTC)]

    class Clock:
        @staticmethod
        def now(tz):
            return now[0]

    monkeypatch.setattr("index.outbox_worker.datetime", Clock)
    event = OutboxEvent(
        "projection", "PROJECT_GRAPH", "ctx://acme/users/alice/memories/entities/item",
        {"edge_id": "edge"}, "PENDING", retry_count=2,
    )

    class Queue:
        supports_batch_claim = batch_claim

        def list_pending(self, account):
            due = not event.next_retry_at or datetime.fromisoformat(event.next_retry_at) <= now[0]
            return [(event.uri, event)] if event.status == "PENDING" and due else []

        def try_acquire(self, item, uri, worker_id):
            item.status = "PROCESSING"
            return True

        def claim_batch(self, worker_id, *, exclude_event_types=None):
            ready = self.list_pending("acme")
            for uri, item in ready:
                self.try_acquire(item, uri, worker_id)
            return ready

        def defer(self, item, uri, when):
            item.status = "PENDING"
            item.next_retry_at = when.isoformat()

        def mark_done(self, item, uri):
            item.status = "DONE"

    queue = Queue()
    projections = Mock()
    projections.drain_sync.return_value = 0
    projections.unfinished.return_value = True

    def worker():
        return OutboxWorker(Mock(), Mock(), edge_projection_components=(projections, {}))

    first = worker()
    for _ in range(first._config.max_retry + 2):
        stats = first.run_once(queue, ["acme"], worker_id="first")
        assert stats["deferred"] == 1
        assert (event.status, event.retry_count) == ("PENDING", 2)
        assert datetime.fromisoformat(event.next_retry_at) > now[0]
        now[0] += timedelta(seconds=1)

    projections.unfinished.return_value = False
    completed = worker().run_once(queue, ["acme"], worker_id="restarted")
    assert completed["succeeded"] == 1
    assert (event.status, event.retry_count) == ("DONE", 2)
    assert event.payload == {"edge_id": "edge"}
