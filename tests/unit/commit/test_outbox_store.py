"""Unit tests for OutboxStore."""

import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from core.models import ContextNode, IndexRecord, OutboxEvent, RequestContext
from core.enums import EventType
from commit.outbox_store import OutboxStore


class TestOutboxStoreInit:
    """Verify OutboxStore initialization."""

    def test_init_with_defaults(self):
        """init should accept default mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs)

        assert store._client is mock_client
        assert store._fs is mock_fs
        assert store._mount_prefix == "/local"

    def test_init_with_custom_mount_prefix(self):
        """init should accept custom mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs, mount_prefix="/custom")

        assert store._mount_prefix == "/custom"

    def test_init_trailing_slash_removed(self):
        """init should remove trailing slash from mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs, mount_prefix="/custom/")

        assert store._mount_prefix == "/custom"

    def test_init_with_path_traversal_raises_error(self):
        """init should reject mount_prefix with path traversal patterns."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="../../../etc")

    def test_init_with_double_dot_raises_error(self):
        """init should reject mount_prefix containing '..'."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="/local/../test")

    def test_init_with_relative_path_raises_error(self):
        """init should reject relative mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="local")


class TestUriToAgfsPath:
    """Verify _uri_to_agfs_path method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/local")

    def test_uri_to_path_with_default_prefix(self):
        """_uri_to_agfs_path should prepend default mount_prefix."""
        uri = "ctx://test-account/users/user-123/memories/profile"
        result = self.store._uri_to_agfs_path(uri)

        assert result.startswith("/local/")
        assert "accounts/test-account/users/user-123/memories/profile" in result

    def test_uri_to_path_with_custom_prefix(self):
        """_uri_to_agfs_path should use custom mount_prefix."""
        store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/custom")
        uri = "ctx://test-account/users/user-123/memories/profile"
        result = store._uri_to_agfs_path(uri)

        assert result.startswith("/custom/")
        assert "accounts/test-account/users/user-123/memories/profile" in result


class TestRegisterWrite:
    """Verify register_write method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/local")

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        self.node = ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Test profile",
            overview="Overview",
            content="Full content",
        )

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_builds_index_records(self, mock_build):
        """register_write should build IndexRecords from ContextNode."""
        mock_build.return_value = []

        self.store.register_write(self.node, self.ctx)

        mock_build.assert_called_once_with(self.node)

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_creates_upsert_event(self, mock_build):
        """register_write should create UPSERT_CONTEXT event."""
        from core.models import IndexRecord
        mock_build.return_value = [
            IndexRecord(
                id="test-id",
                uri=self.node.uri,
                level=0,
                text="abstract",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile"},
            )
        ]

        event = self.store.register_write(self.node, self.ctx)

        assert event.event_type == EventType.UPSERT_CONTEXT.value
        assert event.uri == self.node.uri
        assert event.status == "PENDING"
        assert "records" in event.payload

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_includes_all_record_fields(self, mock_build):
        """register_write should include all required IndexRecord fields in payload."""
        from core.models import IndexRecord
        mock_build.return_value = [
            IndexRecord(
                id="test-id-0",
                uri=self.node.uri,
                level=0,
                text="abstract text",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile", "has_content": True},
            ),
            IndexRecord(
                id="test-id-1",
                uri=self.node.uri,
                level=1,
                text="overview text",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile", "has_content": True},
            ),
        ]

        event = self.store.register_write(self.node, self.ctx)

        records = event.payload["records"]
        assert len(records) == 2

        # Verify all required fields are present
        for record in records:
            assert "id" in record
            assert "uri" in record
            assert "level" in record
            assert "text" in record
            assert "filters" in record
            assert "metadata" in record

        # Verify filters contain account_id and owner_space
        assert records[0]["filters"]["account_id"] == "test-account"
        assert records[0]["filters"]["owner_space"] == "user_space"

    @patch('index.index_record_builder.build_index_records')
    @patch('commit.outbox_store.uuid.uuid4')
    def test_register_write_generates_unique_event_id(self, mock_uuid, mock_build):
        """register_write should generate unique event_id."""
        mock_uuid.return_value = "test-uuid-123"
        mock_build.return_value = []

        event = self.store.register_write(self.node, self.ctx)

        assert event.event_id == "test-uuid-123"

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_calls_write_event(self, mock_build):
        """register_write should call _write_event to persist event."""
        mock_build.return_value = []
        self.store._write_event = Mock()

        self.store.register_write(self.node, self.ctx)

        self.store._write_event.assert_called_once()


class TestRegisterDelete:
    """Verify register_delete method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_register_delete_creates_delete_event(self):
        """register_delete should create DELETE_CONTEXT event."""
        uri = "ctx://test-account/users/user-123/memories/profile"

        event = self.store.register_delete(uri, self.ctx)

        assert event.event_type == EventType.DELETE_CONTEXT.value
        assert event.uri == uri
        assert event.status == "PENDING"
        assert event.payload["ids_to_delete"] == [
            IndexRecord.generate_id(uri, 0),
            IndexRecord.generate_id(uri, 1),
            IndexRecord.generate_id(uri + "/content.md", 2),
        ]


class TestRegisterRelation:
    """Verify register_relation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_register_relation_with_edges(self):
        """register_relation should create UPSERT_RELATION event with edges."""
        from core.models import RelationEdge

        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/preferences/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="User preference",
            )
        ]

        event = self.store.register_relation(edges, self.ctx)

        assert event.event_type == EventType.UPSERT_RELATION.value
        assert "edges" in event.payload
        assert len(event.payload["edges"]) == 1

        edge_data = event.payload["edges"][0]
        assert edge_data["from_uri"] == edges[0].from_uri
        assert edge_data["to_uri"] == edges[0].to_uri
        assert edge_data["relation_type"] == edges[0].relation_type
        assert edge_data["weight"] == edges[0].weight

    def test_register_relation_with_empty_edges(self):
        """register_relation should handle empty edge list."""
        event = self.store.register_relation([], self.ctx)

        assert event.event_type == EventType.UPSERT_RELATION.value
        assert event.payload["edges"] == []


class TestPathTraversalProtection:
    """Tests for path traversal protection (security fix C-4)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

    def test_write_event_rejects_double_dot_in_uri(self):
        """Test that _write_event rejects URIs with '..' path traversal."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/../admin/memories/profile",  # Path traversal
            payload={"records": []},
            status="PENDING",
        )

        # parse_uri rejects invalid URIs before our custom check
        with pytest.raises(ValueError, match="Invalid node_uri"):
            self.store._write_event(event, event.uri)

    def test_scan_outbox_skips_traversal_paths(self):
        """Test that _scan_outbox_recursive skips paths with '..'."""
        results = []

        # Mock client that returns a path with '..'
        mock_entries = [{"name": "safe_dir", "is_dir": True}]
        self.mock_client.ls.return_value = mock_entries

        # Scan with path containing '..' should skip it
        self.store._scan_outbox_recursive("/local/accounts/test/../admin/", results)

        # Should not attempt to list the directory (no ls call on child path)
        assert results == []

    def test_write_event_accepts_valid_uri(self):
        """Test that _write_event accepts valid URIs."""
        from core.models import OutboxEvent
        self.mock_client.write.return_value = None
        self.mock_client.mkdir.return_value = None

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        # Should not raise
        try:
            self.store._write_event(event, event.uri)
        except ValueError:
            pytest.fail("Valid URI should not raise ValueError")


class TestTryAcquireAndRelease:
    """Tests for try_acquire and release methods (concurrent lock protection)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

        self.event = OutboxEvent(
            event_id="test-event-123",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )
        self.node_uri = self.event.uri

    def test_try_acquire_returns_true_when_not_locked(self):
        """try_acquire should return True when no .processing file exists."""
        # Mock client.read raises FileNotFoundError (no lock file)
        self.mock_client.read.side_effect = FileNotFoundError("Not found")
        self.mock_client.write.return_value = None

        result = self.store.try_acquire(self.event, self.node_uri, "worker-1")

        assert result is True
        # Verify lock file was written
        self.mock_client.write.assert_called_once()
        written_path = self.mock_client.write.call_args[0][0]
        assert written_path.endswith(".processing")

    def test_try_acquire_returns_false_when_locked(self):
        """try_acquire should return False when .processing file exists and is valid."""
        # Mock existing lock file with recent timestamp
        lock_data = json.dumps({
            "worker_id": "worker-2",
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        })
        self.mock_client.read.return_value = lock_data.encode("utf-8")

        result = self.store.try_acquire(self.event, self.node_uri, "worker-1")

        assert result is False
        # Should not write new lock
        self.mock_client.write.assert_not_called()

    def test_try_acquire_respects_timeout(self):
        """try_acquire should return True when lock file is expired (>300s old)."""
        # Mock existing lock file with old timestamp (>300 seconds ago)
        from datetime import timedelta
        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        lock_data = json.dumps({
            "worker_id": "worker-2",
            "acquired_at": old_time.isoformat(),
        })
        self.mock_client.read.return_value = lock_data.encode("utf-8")
        self.mock_client.write.return_value = None

        result = self.store.try_acquire(self.event, self.node_uri, "worker-1", timeout_seconds=300)

        assert result is True
        # Verify new lock file was written
        self.mock_client.write.assert_called_once()

    def test_release_removes_processing_file(self):
        """release should remove the .processing file."""
        self.mock_client.rm.return_value = None

        # Should not raise even if file doesn't exist
        self.store.release(self.event, self.node_uri)

        # Verify rm was called
        self.mock_client.rm.assert_called_once()
        called_path = self.mock_client.rm.call_args[0][0]
        assert called_path.endswith(".processing")

    def test_list_pending_skips_locked_events(self):
        """list_pending should skip events with active .processing locks."""
        results = []

        event_json = json.dumps({
            "event_id": "test-event-123",
            "event_type": "UPSERT_CONTEXT",
            "uri": "ctx://test-account/users/user-123/memories/profile",
            "payload": {"records": []},
            "status": "PENDING",
        })

        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "test-event-123.json", "is_dir": False}],
        ]

        # Mock active lock file (recent timestamp)
        lock_data = json.dumps({
            "worker_id": "worker-1",
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        })

        def mock_read_side_effect(path):
            if path.endswith(".json"):
                return event_json.encode("utf-8")
            elif path.endswith(".processing"):
                return lock_data.encode("utf-8")
            raise FileNotFoundError(path)

        self.mock_client.read.side_effect = mock_read_side_effect

        self.store._scan_outbox_recursive("/local/accounts/test-account/", results)

        # Event with active lock should be skipped
        assert len(results) == 0


class TestTryAcquire:
    """Tests for try_acquire lock mechanism."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

    def test_try_acquire_returns_false_when_locked(self):
        """Test that try_acquire returns False when lock already exists."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        recent_time = datetime.now(timezone.utc).isoformat()
        lock_data = json.dumps({"worker_id": "other-worker", "acquired_at": recent_time})
        self.mock_client.read.return_value = lock_data.encode("utf-8")

        result = self.store.try_acquire(event, event.uri, "worker-1")

        assert result is False

    def test_try_acquire_respects_timeout(self):
        """Test that try_acquire allows acquisition after lock timeout."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        lock_data = json.dumps({"worker_id": "old-worker", "acquired_at": old_time})
        self.mock_client.read.return_value = lock_data.encode("utf-8")

        result = self.store.try_acquire(event, event.uri, "worker-1", timeout_seconds=300)

        assert result is True
        self.mock_client.write.assert_called_once()

    def test_try_acquire_succeeds_when_no_lock(self):
        """Test that try_acquire succeeds when no lock exists."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        self.mock_client.read.side_effect = FileNotFoundError("No lock file")

        result = self.store.try_acquire(event, event.uri, "worker-1")

        assert result is True
        self.mock_client.write.assert_called_once()

    def test_try_acquire_creates_lock_with_worker_id(self):
        """Test that try_acquire creates lock file with worker_id."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        self.mock_client.read.side_effect = FileNotFoundError("No lock file")

        self.store.try_acquire(event, event.uri, "worker-123")

        call_args = self.mock_client.write.call_args
        lock_content = json.loads(call_args[0][1].decode("utf-8"))
        assert lock_content["worker_id"] == "worker-123"
        assert "acquired_at" in lock_content


class TestRelease:
    """Tests for release lock mechanism."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

    def test_release_removes_processing_file(self):
        """Test that release removes the .processing file."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        self.store.release(event, event.uri)

        self.mock_client.rm.assert_called_once()

    def test_release_ignores_missing_file(self):
        """Test that release silently ignores if lock file doesn't exist."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        self.mock_client.rm.side_effect = FileNotFoundError("No lock file")

        self.store.release(event, event.uri)

        self.mock_client.rm.assert_called_once()


class TestListPendingLockFiltering:
    """Tests for list_pending lock filtering behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

    def test_list_pending_skips_locked_events(self):
        """Test that list_pending skips events with active locks."""
        from core.models import OutboxEvent

        event = OutboxEvent(
            event_id="locked-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test-account/users/user-123/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        recent_time = datetime.now(timezone.utc).isoformat()
        lock_data = json.dumps({"worker_id": "other-worker", "acquired_at": recent_time})

        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [
                {"name": "locked-event.json", "is_dir": False},
                {"name": "locked-event.processing", "is_dir": False},
            ],
        ]
        self.mock_client.read.side_effect = lambda path: (
            lock_data.encode("utf-8") if ".processing" in path
            else json.dumps({
                "event_id": "locked-event",
                "event_type": "UPSERT_CONTEXT",
                "uri": "ctx://test-account/users/user-123/memories/profile",
                "payload": {"records": []},
                "status": "PENDING",
                "retry_count": 0,
                "created_at": recent_time,
            }).encode("utf-8")
        )

        results = self.store.list_pending("test-account")

        assert len(results) == 0

    def test_list_pending_includes_unlocked_events(self):
        """Test that list_pending includes events without locks."""
        from core.models import OutboxEvent

        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "unlocked-event.json", "is_dir": False}],
        ]
        self.mock_client.read.return_value = json.dumps({
            "event_id": "unlocked-event",
            "event_type": "UPSERT_CONTEXT",
            "uri": "ctx://test-account/users/user-123/memories/profile",
            "payload": {"records": []},
            "status": "PENDING",
            "retry_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")

        results = self.store.list_pending("test-account")

        assert len(results) == 1
        assert results[0][1].event_id == "unlocked-event"

    def test_list_pending_includes_events_with_expired_locks(self):
        """Test that list_pending includes events with expired locks."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        recent_time = datetime.now(timezone.utc).isoformat()
        lock_data = json.dumps({"worker_id": "old-worker", "acquired_at": old_time})

        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [
                {"name": "expired-lock-event.json", "is_dir": False},
                {"name": "expired-lock-event.processing", "is_dir": False},
            ],
        ]
        self.mock_client.read.side_effect = lambda path: (
            lock_data.encode("utf-8") if ".processing" in path
            else json.dumps({
                "event_id": "expired-lock-event",
                "event_type": "UPSERT_CONTEXT",
                "uri": "ctx://test-account/users/user-123/memories/profile",
                "payload": {"records": []},
                "status": "PENDING",
                "retry_count": 0,
                "created_at": recent_time,
            }).encode("utf-8")
        )

        results = self.store.list_pending("test-account")

        assert len(results) == 1
        assert results[0][1].event_id == "expired-lock-event"


class TestCountUnfinished:
    """Tests for the fail-closed AGFS freshness counter."""

    def setup_method(self):
        self.mock_client = Mock()
        self.store = OutboxStore(self.mock_client, Mock())

    @staticmethod
    def _event(event_id, *, status="PENDING", next_retry_at=None):
        return json.dumps(
            {
                "event_id": event_id,
                "event_type": "UPSERT_CONTEXT",
                "uri": "ctx://test-account/users/user-123/memories/profile",
                "payload": {"records": []},
                "status": status,
                "retry_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "next_retry_at": next_retry_at,
            }
        ).encode("utf-8")

    def test_count_unfinished_includes_locked_and_deferred_events(self):
        self.mock_client.ls.side_effect = [
            [{"name": "resources", "is_dir": True}],
            [{"name": ".outbox", "is_dir": True}],
            [
                {"name": "pending.json", "is_dir": False},
                {"name": "active.json", "is_dir": False},
                {"name": "active.processing", "is_dir": False},
                {"name": "deferred.json", "is_dir": False},
                {"name": "dlq", "is_dir": True},
            ],
        ]
        payloads = {
            "pending.json": self._event("pending"),
            "active.json": self._event("active"),
            "deferred.json": self._event(
                "deferred",
                next_retry_at="2030-01-01T00:00:00+00:00",
            ),
        }
        self.mock_client.read.side_effect = (
            lambda path: payloads[path.rsplit("/", 1)[-1]]
        )

        assert self.store.count_unfinished("test-account") == {
            "pending": 2,
            "processing": 1,
            "total": 3,
        }

    @pytest.mark.parametrize("lock_name", [
        ".state.json.resource-mutation-lock",
        ".state.json.resource-mutation-lock-" + "a" * 32,
        ".profile.update-lock",
        ".profile.update-lock-" + "b" * 32,
    ])
    def test_count_unfinished_during_metadata_lock_publication(self, lock_name):
        directories = {
            "/local/accounts/test-account/": [
                {"name": lock_name, "is_dir": True},
                {"name": "profile", "is_dir": True},
            ],
            "/local/accounts/test-account/profile/": [
                {"name": ".outbox", "is_dir": True},
            ],
            "/local/accounts/test-account/profile/.outbox/": [
                {"name": "live.json", "is_dir": False},
            ],
        }

        def list_directory(path):
            if path not in directories:
                raise FileNotFoundError(f"no such directory: {path}")
            return directories[path]

        self.mock_client.ls.side_effect = list_directory
        self.mock_client.read.return_value = self._event("live")
        assert self.store.count_unfinished("test-account") == {
            "pending": 1,
            "processing": 0,
            "total": 1,
        }

    def test_count_unfinished_includes_owner_named_like_metadata_lock(self):
        self.mock_client.ls.side_effect = [
            [{"name": "users", "is_dir": True}],
            [{"name": ".alice.update-lock", "is_dir": True}],
            [{"name": "memories", "is_dir": True}],
            [{"name": "profile", "is_dir": True}],
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "live.json", "is_dir": False}],
        ]
        self.mock_client.read.return_value = self._event("live")

        assert self.store.count_unfinished("test-account") == {
            "pending": 1,
            "processing": 0,
            "total": 1,
        }

    def test_count_unfinished_during_event_completion(self):
        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "completed.json", "is_dir": False}],
        ]
        self.mock_client.read.side_effect = FileNotFoundError("event not found")

        assert self.store.count_unfinished("test-account") == {
            "pending": 0,
            "processing": 0,
            "total": 0,
        }

    def test_count_unfinished_skips_dlq_entries(self):
        self.mock_client.ls.side_effect = [
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "dlq", "is_dir": True}],
        ]

        assert self.store.count_unfinished("test-account") == {
            "pending": 0,
            "processing": 0,
            "total": 0,
        }
        self.mock_client.read.assert_not_called()

    def test_count_unfinished_scans_a_live_resource_directory_named_dlq(self):
        self.mock_client.ls.side_effect = [
            [{"name": "dlq", "is_dir": True}],
            [{"name": ".outbox", "is_dir": True}],
            [{"name": "live.json", "is_dir": False}],
        ]
        self.mock_client.read.return_value = self._event("live")

        assert self.store.count_unfinished("test-account") == {
            "pending": 1,
            "processing": 0,
            "total": 1,
        }

    def test_count_unfinished_rejects_unsafe_account_before_scanning(self):
        with pytest.raises(ValueError, match="account_id is invalid"):
            self.store.count_unfinished("../other-account")

        self.mock_client.ls.assert_not_called()

    def test_count_unfinished_raises_when_a_nested_scan_fails(self):
        self.mock_client.ls.side_effect = [
            [{"name": "resources", "is_dir": True}],
            RuntimeError("AGFS unavailable"),
        ]

        with pytest.raises(RuntimeError, match="AGFS unavailable"):
            self.store.count_unfinished("test-account")
