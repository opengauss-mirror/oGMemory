"""Graph-disabled workers pause projection wakeups without consuming retries."""
from unittest.mock import Mock

from core.models import OutboxEvent
from index.outbox_worker import OutboxWorker


URI = "ctx://paused-graph/users/alice/memories/entities/item"


class OrderedOutbox:
    """Small ordered-queue fixture; the real SQL ordering is covered separately."""
    supports_batch_claim = True

    def __init__(self):
        self.events = [
            OutboxEvent("graph", "PROJECT_GRAPH", URI, {"edge_id": "edge"}, "PENDING"),
            OutboxEvent("ordinary", "UPSERT_CONTEXT", URI, {"records": []}, "PENDING"),
        ]
        self.retries = self.dlq = 0

    def claim_batch(self, worker_id, *, exclude_event_types=None):
        excluded = set(exclude_event_types or ())
        ready = []
        for index, event in enumerate(self.events):
            if event.status != "PENDING" or event.event_type in excluded:
                continue
            if any(previous.status in {"PENDING", "PROCESSING"}
                   and not (previous.event_type == "PROJECT_GRAPH" and previous.event_type in excluded)
                   for previous in self.events[:index]):
                continue
            event.status = "PROCESSING"
            ready.append((event.uri, event))
        return ready

    def mark_done(self, event, uri):
        event.status = "DONE"

    def release(self, event, uri):
        event.status = "PENDING"

    def increment_retry(self, event, uri, when):
        self.retries += 1
        event.retry_count += 1

    def move_to_dlq(self, event, uri):
        self.dlq += 1
        event.status = "FAILED"


def test_disabled_graph_preserves_wakeup_and_indexes_later_same_uri_event():
    queue = OrderedOutbox()
    disabled = OutboxWorker(Mock(), Mock())
    for _ in range(disabled._config.max_retry + 2):
        stats = disabled.run_once(queue, [], worker_id="disabled")
        assert stats["failed"] == stats["moved_to_dlq"] == 0
    wakeup, ordinary = queue.events
    assert (wakeup.status, wakeup.retry_count) == ("PENDING", 0)
    assert ordinary.status == "DONE"
    assert (queue.retries, queue.dlq) == (0, 0)

    # A process restart with graph enabled rediscovers the durable wakeup even
    # with no account event or account discovery result to prime its drain.
    projections = Mock()
    projections.drain_sync.return_value = 1
    projections.unfinished.return_value = False
    enabled = OutboxWorker(Mock(), Mock(), edge_projection_components=(projections, {"profile_id": "test"}))
    enabled.run_once(queue, [], worker_id="enabled")
    assert wakeup.status == "DONE"
    assert projections.drain_sync.call_count >= 1
    assert projections.drain_sync.call_args.args[0] == "paused-graph"


def test_disabled_graph_skips_agfs_event_before_acquiring_it():
    queue = Mock()
    event = OutboxEvent("graph", "PROJECT_GRAPH", URI, {"edge_id": "edge"}, "PENDING", retry_count=4)
    queue.list_pending.return_value = [(URI, event)]
    worker = OutboxWorker(Mock(), Mock())
    stats = worker.run_once(queue, ["paused-graph"])
    assert stats["processed"] == stats["failed"] == stats["moved_to_dlq"] == 0
    assert stats["skipped"] == 1
    queue.try_acquire.assert_not_called()
    queue.increment_retry.assert_not_called()
    queue.move_to_dlq.assert_not_called()
    assert event.retry_count == 4


def test_legacy_batch_claim_that_ignores_exclusions_releases_without_retry():
    queue = Mock()
    queue.supports_batch_claim = True
    event = OutboxEvent("graph", "PROJECT_GRAPH", URI, {"edge_id": "edge"}, "PROCESSING", retry_count=4)
    queue.claim_batch.return_value = [(URI, event)]
    stats = OutboxWorker(Mock(), Mock()).run_once(queue, [])
    assert stats["failed"] == stats["moved_to_dlq"] == 0
    queue.release.assert_called_once_with(event, URI)
    queue.increment_retry.assert_not_called()
    queue.mark_done.assert_not_called()
    assert event.retry_count == 4
