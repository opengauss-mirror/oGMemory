"""Unit tests for OutboxWorker.

Tests verify the happy path and retry/DLQ behavior.
Uses mock VectorIndex and mock Embedder for isolated testing.
"""

from unittest.mock import ANY, MagicMock, Mock, patch

import pytest

from core.errors import NodeNotFoundError
from core.models import IndexRecord, OutboxEvent, RequestContext
from index.outbox_worker import (
    OutboxWorker,
    WorkerConfig,
    WorkerResult,
    create_upsert_event,
)
from providers.token_tracker import TokenTracker


# Mock VectorIndex for testing
class MockVectorIndex:
    """In-memory vector index for testing."""

    def __init__(self) -> None:
        self.records: dict[str, IndexRecord] = {}
        self.deleted_ids: list[str] = []

    def upsert(self, records: list[IndexRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            if id_ in self.records:
                del self.records[id_]
            self.deleted_ids.append(id_)

    def search(self, query):
        return []  # Not used in these tests


# Mock Embedder for testing
class MockEmbedder:
    """Mock embedder that returns fixed vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        # Return unique vectors per text
        return [[float(i) for i in range(self._dimension)] for _ in texts]


class FailingMockEmbedder(MockEmbedder):
    """Mock embedder that always fails."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding failed!")


class FailingMockVectorIndex(MockVectorIndex):
    """Mock vector index that always fails."""

    def upsert(self, records: list[IndexRecord]) -> None:
        raise RuntimeError("Vector index down!")


def test_instance_can_disable_class_batch_claim_capability():
    class Store:
        supports_batch_claim = True

        def __init__(self):
            self.supports_batch_claim = False

    assert OutboxWorker._supports_batch_claim(Store()) is False


class TestCreateUpsertEvent:
    """Tests for create_upsert_event utility."""

    def test_creates_valid_event(self):
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        assert event.event_type == "UPSERT_CONTEXT"
        assert event.uri == "ctx://test/users/u1/memories/profile"
        assert event.status == "PENDING"
        assert event.retry_count == 0
        assert "records" in event.payload

    def test_serializes_records_correctly(self):
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
                metadata={"level": "abstract"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        payload_records = event.payload["records"]
        assert len(payload_records) == 1
        assert payload_records[0]["id"] == "rec1"
        assert payload_records[0]["level"] == 0
        assert payload_records[0]["filters"]["account_id"] == "test"


class TestOutboxWorkerProcessEvent:
    """Tests for OutboxWorker.process_event happy path."""

    def test_successful_processing(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 1
        assert result.event_id == event.event_id

    def test_multiple_records_upserted(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id=f"rec{i}",
                uri="ctx://test/users/u1/memories/profile",
                level=i,
                text=f"text{i}",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
            for i in range(3)
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 3
        assert len(vector_index.records) == 3

    def test_embedder_called_once_per_batch(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        worker.process_event(event)

        assert embedder.call_count == 1

    def test_event_with_no_records_succeeds(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Create event with empty records list
        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 0

    def test_records_stored_in_vector_index(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        worker.process_event(event)

        assert "rec1" in vector_index.records
        stored_record = vector_index.records["rec1"]
        assert stored_record.uri == "ctx://test/users/u1/memories/profile"
        # Embedding should be added to metadata
        assert "_embedding" in stored_record.metadata


class TestOutboxWorkerFailureCases:
    """Tests for OutboxWorker failure handling."""

    def test_embedding_failure_returns_error(self):
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is False
        assert "Embedding failed" in result.error_message
        assert result.records_upserted == 0

    def test_vector_index_failure_returns_error(self):
        vector_index = FailingMockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is False
        assert "Vector index down" in result.error_message


class TestOutboxWorkerValidation:
    """Tests for OutboxWorker input validation."""

    def test_missing_records_field_raises_error(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={},  # Missing "records"
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing 'records'" in result.error_message

    def test_records_not_list_raises_error(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": "not-a-list"},
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "not a list" in result.error_message

    def test_record_missing_account_id_raises_error(self):
        """CRITICAL: Missing account_id causes silent leakage - must reject."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={
                "records": [
                    {
                        "id": "rec1",
                        "uri": "ctx://test/users/u1/memories/profile",
                        "level": 0,
                        "text": "abstract",
                        "filters": {"owner_space": "user:u1"},  # Missing account_id
                    }
                ]
            },
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing account_id" in result.error_message

    def test_record_missing_owner_space_raises_error(self):
        """CRITICAL: Missing owner_space causes silent leakage - must reject."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={
                "records": [
                    {
                        "id": "rec1",
                        "uri": "ctx://test/users/u1/memories/profile",
                        "level": 0,
                        "text": "abstract",
                        "filters": {"account_id": "test"},  # Missing owner_space
                    }
                ]
            },
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing owner_space" in result.error_message


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_backoff_increases_exponentially(self):
        config = WorkerConfig(base_delay_ms=1000, max_delay_ms=60000)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        delay_0 = worker.calculate_backoff(0)
        delay_1 = worker.calculate_backoff(1)
        delay_2 = worker.calculate_backoff(2)

        assert delay_0.total_seconds() == 1.0
        assert delay_1.total_seconds() == 2.0
        assert delay_2.total_seconds() == 4.0

    def test_backoff_capped_at_max_delay(self):
        config = WorkerConfig(base_delay_ms=1000, max_delay_ms=3000)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        # With base=1000, max=3000, retry 3 would give 8000ms but should cap at 3000
        delay_3 = worker.calculate_backoff(3)

        assert delay_3.total_seconds() == 3.0


class TestShouldRetry:
    """Tests for retry logic."""

    def test_success_should_not_retry(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        result = WorkerResult(
            event_id="test-event",
            success=True,
            records_upserted=0,
        )

        assert worker.should_retry(event, result) is False

    def test_failure_within_retry_limit_should_retry(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=2,
        )

        result = WorkerResult(
            event_id="test-event",
            success=False,
            records_upserted=0,
            error_message="Temporary error",
        )

        assert worker.should_retry(event, result) is True

    def test_failure_exceeding_retry_limit_should_not_retry(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=5,  # At max retry
        )

        result = WorkerResult(
            event_id="test-event",
            success=False,
            records_upserted=0,
            error_message="Permanent error",
        )

        assert worker.should_retry(event, result) is False

    def test_max_retry_event_returns_dlq_result(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=5,
        )

        result = worker.process_event(event)

        assert result.success is False
        assert result.moved_to_dlq is True
        assert "Max retry exceeded" in result.error_message

    def test_cross_tenant_account_id_mismatch_rejected(self):
        """Test that records with mismatched account_id are rejected (security fix C-3)."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Event URI has account "test" but record has account "other-account"
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://other-account/users/u1/memories/profile",  # Different account
                level=0,
                text="abstract",
                filters={"account_id": "other-account", "owner_space": "user:u1"},
            )
        ]

        # Event URI is for "test" account
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        # Should fail due to account mismatch
        assert result.success is False
        assert "does not match" in result.error_message
        assert "cross-tenant injection" in result.error_message


class TestOutboxWorkerRunOnce:
    """Tests for OutboxWorker.run_once() polling entry point."""

    def test_run_once_happy_path(self):
        """Test run_once processes pending events successfully."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store with explicit configuration
        outbox_store = Mock()
        outbox_store.try_acquire.return_value = True  # Explicitly set

        # Create pending event
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"], worker_id="worker-1")

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 1
        assert stats["failed"] == 0
        assert stats["moved_to_dlq"] == 0

        # Verify event was marked done
        outbox_store.mark_done.assert_called_once_with(event, event.uri)

        # Verify try_acquire was called with worker_id
        outbox_store.try_acquire.assert_called_once_with(event, event.uri, "worker-1")

    def test_run_once_with_failures(self):
        """Test run_once handles failures correctly."""
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()  # Will fail
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()

        # Create pending event
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        event.retry_count = 0
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"])

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 1  # Failed but under max retry
        assert stats["moved_to_dlq"] == 0

        # Verify event retry was incremented with a scheduled retry time
        outbox_store.increment_retry.assert_called_once_with(event, event.uri, ANY)

    def test_run_once_moves_to_dlq_after_max_retry(self):
        """Test run_once moves events to DLQ after max retry."""
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config=WorkerConfig(max_retry=3))

        # Mock outbox store
        outbox_store = Mock()

        # Create pending event at max retry
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        event.retry_count = 3  # At max retry
        event.payload["failure_reason"] = "embedding unavailable"
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"])

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 0
        assert stats["moved_to_dlq"] == 1

        # Verify event was moved to DLQ
        outbox_store.move_to_dlq.assert_called_once()
        assert event.payload["failure_reason"] == "Embedding failed!"

    def test_run_once_handles_multiple_accounts(self):
        """Test run_once processes events from multiple accounts."""
        vector_index = MockVectorIndex()

        class TrackedEmbedder(MockEmbedder):
            def __init__(self):
                super().__init__()
                self.token_tracker = TokenTracker(model="embedding-test")

            def embed_texts(self, texts):
                embeddings = super().embed_texts(texts)
                self.token_tracker.record_embed(len(texts) * 10)
                return embeddings

        embedder = TrackedEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()

        # Create events for two accounts
        def make_side_effect(account_id):
            records = [
                IndexRecord(
                    id=f"rec_{account_id}",
                    uri=f"ctx://{account_id}/users/u1/memories/profile",
                    level=0,
                    text="abstract",
                    filters={"account_id": account_id, "owner_space": "user:u1"},
                )
            ]
            event = create_upsert_event(f"ctx://{account_id}/users/u1/memories/profile", records)
            return [(event.uri, event)]

        outbox_store.list_pending.side_effect = make_side_effect

        # Run polling cycle for multiple accounts
        stats = worker.run_once(outbox_store, ["account1", "account2"])

        # Verify both accounts were processed
        assert stats["processed"] == 2
        assert stats["succeeded"] == 2
        assert outbox_store.list_pending.call_count == 2
        assert embedder.call_count == 1
        assert embedder.token_tracker.scoped_snapshot("account1").embed_tokens == 10
        assert embedder.token_tracker.scoped_snapshot("account2").embed_tokens == 10

    def test_run_once_skips_acquired_events(self):
        """Test run_once skips events already locked by another worker."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store with try_acquire returning False (already locked)
        outbox_store = Mock()
        outbox_store.try_acquire.return_value = False

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle with worker_id
        stats = worker.run_once(outbox_store, ["test"], worker_id="worker-1")

        # Verify event was skipped
        assert stats["processed"] == 0
        assert stats["skipped"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 0

        # Verify try_acquire was called with worker_id
        outbox_store.try_acquire.assert_called_once_with(event, event.uri, "worker-1")

        # Verify process_event was NOT called (event was locked)
        # The only calls should be list_pending and try_acquire
        assert outbox_store.mark_done.call_count == 0

    def test_run_once_releases_lock_on_exception(self):
        """Test run_once releases lock when processing raises exception."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()
        outbox_store.try_acquire.return_value = True

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Make mark_done raise an exception
        outbox_store.mark_done.side_effect = RuntimeError("Database error")

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"], worker_id="worker-1")

        # Verify stats
        assert stats["processed"] == 1
        assert stats["failed"] == 1

        # Verify release was called despite exception (this is the test's purpose)
        outbox_store.release.assert_called_once_with(event, event.uri)

        # Verify mark_done was NOT called (it raised exception)
        # Actually mark_done WAS called but raised exception, so call count should be 1
        # The important thing is that release was still called afterwards


class TestPostUpsertLivenessCheck:
    """Tests for post-upsert tombstone detection and rollback."""

    def _make_event(self, **overrides):
        defaults = dict(
            event_id="evt-tomb-1",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={
                "records": [
                    {
                        "id": "rec1",
                        "uri": "ctx://test/users/u1/memories/profile",
                        "level": 0,
                        "text": "abstract",
                        "filters": {
                            "account_id": "test",
                            "owner_space": "user:u1",
                        },
                    }
                ]
            },
            status="PENDING",
            retry_count=0,
        )
        defaults.update(overrides)
        return OutboxEvent(**defaults)

    @staticmethod
    def _make_liveness_aware_store(**kwargs):
        """Create a Mock outbox_store where is_event_current is a real
        class-level method so _supports_liveness_check detects it."""
        is_event_current_mock = kwargs.pop("is_event_current", Mock(return_value=True))

        class MockOutboxStore:
            supports_batch_claim = False  # Explicitly disable batch claims

            @classmethod
            def is_event_current(cls, event_id):
                return is_event_current_mock(event_id)

            def __init__(self):
                self.try_acquire = kwargs.pop("try_acquire", Mock(return_value=True))
                self.list_pending = kwargs.pop("list_pending", Mock(return_value=[]))
                self.mark_done = kwargs.pop("mark_done", Mock())
                self.increment_retry = kwargs.pop("increment_retry", Mock())
                self.release = kwargs.pop("release", Mock())
                self.move_to_dlq = kwargs.pop("move_to_dlq", Mock())

        return MockOutboxStore()

    def test_sql_batch_processing_holds_uri_lock_through_ack(self):
        vector_index = MockVectorIndex()
        worker = OutboxWorker(vector_index, MockEmbedder())
        event = self._make_event()

        class MockBatchClaimStore:
            supports_batch_claim = True

            def __init__(self):
                self.locked = False
                self.lock_entries = 0

            def processing_lock(self, node_uri):
                store = self

                class Lock:
                    def __enter__(self):
                        assert node_uri == event.uri
                        store.locked = True
                        store.lock_entries += 1

                    def __exit__(self, *_args):
                        store.locked = False

                return Lock()

            def is_event_current(self, event_id):
                assert self.locked
                assert event_id == event.event_id
                return True

            def claim_batch(self, worker_id="default", *, exclude_event_types=None):
                return [(event.uri, event)]

            def mark_done(self, claimed_event, node_uri):
                assert self.locked
                assert claimed_event is event
                assert node_uri == event.uri

            def increment_retry(self, event, node_uri, next_retry_at):
                raise AssertionError("successful event must not retry")

            def move_to_dlq(self, event, node_uri):
                raise AssertionError("successful event must not enter DLQ")

        outbox_store = MockBatchClaimStore()

        stats = worker.run_once(outbox_store, ["test"])

        assert stats["succeeded"] == 1
        assert outbox_store.lock_entries == 1
        assert outbox_store.locked is False

    def test_sql_batch_processing_retries_processing_lock_timeout(self):
        worker = OutboxWorker(MockVectorIndex(), MockEmbedder())
        event = self._make_event()

        class MockBatchClaimStore:
            supports_batch_claim = True

            def __init__(self):
                self.retry_calls = 0
                self.release_calls = 0

            def claim_batch(self, worker_id="default", *, exclude_event_types=None):
                return [(event.uri, event)]

            def processing_lock(self, _node_uri):
                class Lock:
                    def __enter__(self):
                        raise TimeoutError("lock busy")

                    def __exit__(self, *_args):
                        raise AssertionError("unacquired lock must not exit")

                return Lock()

            def increment_retry(self, claimed_event, node_uri, next_retry_at):
                assert claimed_event is event
                assert node_uri == event.uri
                assert next_retry_at is not None
                self.retry_calls += 1

            def release(self, claimed_event, node_uri):
                assert claimed_event is event
                assert node_uri == event.uri
                self.release_calls += 1

            def move_to_dlq(self, _event, _node_uri):
                raise AssertionError("first lock timeout must not enter DLQ")

        outbox_store = MockBatchClaimStore()

        stats = worker.run_once(outbox_store, ["test"])

        assert stats == {
            "processed": 1,
            "succeeded": 0,
            "failed": 1,
            "moved_to_dlq": 0,
            "skipped": 0,
        }
        assert outbox_store.retry_calls == 1
        assert outbox_store.release_calls == 1

    def test_liveness_db_error_triggers_failure_not_ack(self):
        """Post-upsert liveness check DB error should cause event to fail/retry,
        not silently succeed. (P1 fix: re-raise instead of current=True)"""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = self._make_event()

        # Need batch claim path for liveness check
        class MockBatchClaimStore:
            supports_batch_claim = True

            @classmethod
            def is_event_current(cls, event_id):
                # Pre-check: DB error (treated as proceed), Post-upsert: DB error (re-raises)
                if not hasattr(cls, '_call_count'):
                    cls._call_count = 0
                cls._call_count += 1
                if cls._call_count == 1:
                    # Pre-check DB error - treated as proceed
                    raise RuntimeError("DB down pre-check")
                else:
                    # Post-upsert DB error - re-raises
                    raise RuntimeError("DB down post-check")

            def claim_batch(self, worker_id="default", *, exclude_event_types=None):
                return [(event.uri, event)]

            def mark_done(self, event, node_uri):
                pass

            def increment_retry(self, event, node_uri, next_retry_at):
                pass

            def release(self, event, node_uri):
                pass

            def move_to_dlq(self, event, node_uri):
                pass

        outbox_store = MockBatchClaimStore()

        stats = worker.run_once(outbox_store, ["test"])

        # The post-upsert DB error causes the event to fail
        assert stats["succeeded"] == 0
        assert stats["failed"] == 1

    def test_tombstoned_event_vector_rollback(self):
        """When event is tombstoned after upsert, vector records should be deleted.
        Pre-check returns True (event exists), post-check returns False (tombstoned)."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = self._make_event()

        # For this test to work properly, we need to simulate the batch claim path
        # which has the liveness check logic. Let's create a store with claim_batch support.
        class MockBatchClaimStore:
            supports_batch_claim = True

            @classmethod
            def is_event_current(cls, event_id):
                # First call (pre-check): True → process. Second (post-upsert): False → rollback.
                if not hasattr(cls, '_call_count'):
                    cls._call_count = 0
                cls._call_count += 1
                return cls._call_count == 1  # True first time, False second time

            def claim_batch(self, worker_id="default", *, exclude_event_types=None):
                return [(event.uri, event)]

            def mark_done(self, event, node_uri):
                pass

            def increment_retry(self, event, node_uri, next_retry_at):
                pass

            def release(self, event, node_uri):
                pass

            def move_to_dlq(self, event, node_uri):
                pass

        outbox_store = MockBatchClaimStore()

        stats = worker.run_once(outbox_store, ["test"])

        # Vector records should have been deleted during rollback
        assert "rec1" in vector_index.deleted_ids
        # Event should be treated as success (rolled back, not retried)
        assert stats["succeeded"] == 1

    def test_tombstoned_event_skips_processing(self):
        """Pre-check: if event is already tombstoned before processing, skip it."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = self._make_event()

        # Need batch claim path for liveness check in pre-processing
        class MockBatchClaimStore:
            supports_batch_claim = True

            @classmethod
            def is_event_current(cls, event_id):
                # Pre-check returns False — event already gone
                return False

            def claim_batch(self, worker_id="default", *, exclude_event_types=None):
                return [(event.uri, event)]

            def mark_done(self, event, node_uri):
                pass

            def increment_retry(self, event, node_uri, next_retry_at):
                pass

            def release(self, event, node_uri):
                pass

            def move_to_dlq(self, event, node_uri):
                pass

        outbox_store = MockBatchClaimStore()

        stats = worker.run_once(outbox_store, ["test"])

        assert stats["skipped"] == 1
        assert stats["succeeded"] == 0
        # No vector records should have been upserted
        assert len(vector_index.records) == 0

    def test_liveness_check_passes_normally(self):
        """When liveness check returns True, event is processed and acked normally."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = self._make_event()
        outbox_store = self._make_liveness_aware_store(
            list_pending=Mock(return_value=[(event.uri, event)]),
            is_event_current=Mock(return_value=True),
        )

        stats = worker.run_once(outbox_store, ["test"])

        assert stats["succeeded"] == 1
        assert "rec1" not in vector_index.deleted_ids


class TestRollbackStaleUpsert:
    """Tests for _rollback_stale_upsert covering vector + directory node cleanup."""

    def _make_ctx(self):
        return RequestContext(
            account_id="test",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
        )

    def test_rollback_deletes_vector_records_from_payload(self):
        """Rollback deletes vector IDs stored in payload.records."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Pre-populate vector index
        vector_index.upsert([
            IndexRecord(
                id="r0",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ])
        assert "r0" in vector_index.records

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": [{"id": "r0"}]},
            status="PENDING",
        )

        worker._rollback_stale_upsert(event, Mock(), self._make_ctx())

        assert "r0" in vector_index.deleted_ids

    def test_rollback_deletes_stale_directory_nodes(self):
        """Rollback deletes directory summary nodes written during processing."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        mock_fs = Mock()
        worker = OutboxWorker(vector_index, embedder, fs=mock_fs)

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://test/users/u1/memories/",
            payload={
                "records": [{"id": "r0"}],
                "written_dir_uris": [
                    "ctx://test/users/u1/memories/",
                    "ctx://test/users/u1/memories/preferences/",
                ],
            },
            status="PENDING",
        )

        outbox_store = Mock()
        worker._rollback_stale_upsert(event, outbox_store, self._make_ctx())

        # Both directory nodes should have been deleted
        assert mock_fs.delete_node.call_count == 2
        mock_fs.delete_node.assert_any_call(
            "ctx://test/users/u1/memories/", self._make_ctx()
        )
        mock_fs.delete_node.assert_any_call(
            "ctx://test/users/u1/memories/preferences/", self._make_ctx()
        )
        # No cleanup events needed — all deletes succeeded
        outbox_store.register_delete.assert_not_called()

    def test_rollback_swallows_node_not_found(self):
        """NodeNotFoundError (node already moved) is silently skipped."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        mock_fs = Mock()
        mock_fs.delete_node.side_effect = NodeNotFoundError("ctx://test/users/u1/memories/")
        worker = OutboxWorker(vector_index, embedder, fs=mock_fs)

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://test/users/u1/memories/",
            payload={
                "records": [{"id": "r0"}],
                "written_dir_uris": ["ctx://test/users/u1/memories/"],
            },
            status="PENDING",
        )

        outbox_store = Mock()
        # Should not raise
        worker._rollback_stale_upsert(event, outbox_store, self._make_ctx())

        # No cleanup event needed — node already gone
        outbox_store.register_delete.assert_not_called()

    def test_rollback_emits_cleanup_event_on_transient_failure(self):
        """When delete_node fails with a transient error, a DELETE_CONTEXT
        event is emitted so the normal worker loop eventually cleans it up."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        mock_fs = Mock()
        mock_fs.delete_node.side_effect = RuntimeError("FS temporarily down")
        worker = OutboxWorker(vector_index, embedder, fs=mock_fs)

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://test/users/u1/memories/",
            payload={
                "records": [{"id": "r0"}],
                "written_dir_uris": ["ctx://test/users/u1/memories/"],
            },
            status="PENDING",
        )

        outbox_store = Mock()
        outbox_store.register_delete = Mock()
        worker._rollback_stale_upsert(event, outbox_store, self._make_ctx())

        # A cleanup DELETE event should have been emitted
        outbox_store.register_delete.assert_called_once_with(
            "ctx://test/users/u1/memories/", self._make_ctx()
        )

    def test_rollback_handles_register_delete_failure_gracefully(self):
        """If emitting the cleanup event itself fails, log but don't crash."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        mock_fs = Mock()
        mock_fs.delete_node.side_effect = RuntimeError("FS down")
        worker = OutboxWorker(vector_index, embedder, fs=mock_fs)

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://test/users/u1/memories/",
            payload={
                "records": [{"id": "r0"}],
                "written_dir_uris": ["ctx://test/users/u1/memories/"],
            },
            status="PENDING",
        )

        outbox_store = Mock()
        outbox_store.register_delete = Mock(
            side_effect=RuntimeError("Outbox also down")
        )
        # Should not raise — both FS and outbox are down, but we survive
        worker._rollback_stale_upsert(event, outbox_store, self._make_ctx())

    def test_rollback_noop_when_no_records_and_no_dirs(self):
        """Rollback is a no-op when payload has no records or dir URIs."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="evt-1",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={},
            status="PENDING",
        )

        worker._rollback_stale_upsert(event, Mock(), self._make_ctx())

        assert vector_index.deleted_ids == []


class TestEnsureTableRaisesOnFailure:
    """Tests for SQLOutboxStore._ensure_table raising on DDL failure."""

    def test_ensure_table_raises_on_ddl_error(self):
        """_ensure_table must raise on DDL failure, not swallow it silently."""
        from psycopg2.extensions import TRANSACTION_STATUS_IDLE

        from fs.sql_adapter.schema import _SCHEMA_SQL

        with patch("commit.sql_outbox_store.psycopg2"), \
             patch("commit.sql_outbox_store.Json", lambda x: x), \
             patch("commit.sql_outbox_store._HAS_PSYCOPG2", True):
            mock_conn = MagicMock()
            mock_conn.get_transaction_status.return_value = TRANSACTION_STATUS_IDLE
            mock_conn.autocommit = False
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.fetchone.return_value = None

            def execute(statement):
                if statement == _SCHEMA_SQL:
                    raise RuntimeError("DDL failed")

            mock_cursor.execute.side_effect = execute

            from commit.sql_outbox_store import SQLOutboxStore

            store = SQLOutboxStore.__new__(SQLOutboxStore)
            store._connection_string = "host=localhost"
            store._fs = MagicMock()
            store._pool_size = 5
            store._pool = []
            store._get_connection = lambda: mock_conn
            store._return_connection = lambda c: None

            with pytest.raises(RuntimeError, match="DDL failed"):
                store._ensure_table()

            mock_conn.rollback.assert_called_once()
            assert any(call.args[0] == _SCHEMA_SQL for call in mock_cursor.execute.call_args_list)
