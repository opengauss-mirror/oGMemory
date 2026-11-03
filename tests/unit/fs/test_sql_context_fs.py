"""Unit tests for SQLContextFS.

Tests use mocked psycopg2 connections to verify behavior without
requiring a running PostgreSQL instance.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

import fs.sql_adapter.sql_context_fs as sql_context_fs
from core.enums import ContextType, EventType
from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    NodeBrokenError,
    NodeNotFoundError,
)
from core.models import ContextNode, OutboxEvent, RelationEdge, RequestContext
from core.resource_identity import RootBindingError, compute_workspace_root_id
from fs.sql_adapter.sql_context_fs import _resource_mutation_lock_key, parse_uri

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ctx(account_id="acme", user_id="alice", session_id="s1"):
    return RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id="main",
        session_id=session_id,
        trace_id="t1",
    )


def _profile_node(content="Profile content", abstract="Short abstract"):
    return ContextNode(
        uri="ctx://acme/users/alice/memories/profile",
        context_type="MEMORY",
        category="profile",
        level=0,
        owner_space="user:alice",
        abstract=abstract,
        overview="## Overview\nDetailed overview",
        content=content,
        metadata={"tags": ["test"]},
    )


def _code_resource_node(slug="chunk-a"):
    return ContextNode(
        uri=f"ctx://acme/resources/{slug}",
        context_type="RESOURCE",
        category="code",
        level=0,
        owner_space="resource:acme",
        abstract="parse config",
        overview="parse_config(path)",
        content="def parse_config(path):\n    return path\n",
        metadata={"resource_type": "code"},
    )


def _write_event(node: ContextNode) -> OutboxEvent:
    return OutboxEvent(
        event_id="write-event",
        event_type=EventType.UPSERT_CONTEXT.value,
        uri=node.uri,
        payload={"records": []},
        status="PENDING",
    )


def _sql_calls(cursor) -> list[str]:
    return [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]


def _pref_node(slug="coffee"):
    return ContextNode(
        uri=f"ctx://acme/users/alice/memories/preferences/{slug}",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:alice",
        abstract="Likes coffee",
        overview="## Coffee preferences",
        content="Prefers dark roast",
        metadata={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_managed_skill_uri():
    assert parse_uri("ctx://acme/agents/main/skills/imported/demo") == {
        "account": "acme",
        "owner_type": "agents",
        "owner_id": "main",
        "category": "skills/imported",
        "slug": "demo",
    }


def test_parse_legacy_generated_skill_uri():
    assert parse_uri("ctx://acme/agents/main/skills/demo") == {
        "account": "acme",
        "owner_type": "agents",
        "owner_id": "main",
        "category": "skills",
        "slug": "demo",
    }


@pytest.mark.parametrize(
    "slug",
    [".", "..", "%2F", "%5c", "%2E%2e", "foo%252fbar", "foo+bar"],
)
def test_parse_resource_uri_rejects_unsafe_slug(slug):
    with pytest.raises(ValueError):
        parse_uri(f"ctx://acme/resources/{slug}")


@pytest.mark.parametrize("slug", ["a" * 32, "dataset", "repo-name_01", "repo.v1"])
def test_parse_resource_uri_preserves_safe_slug(slug):
    assert parse_uri(f"ctx://acme/resources/{slug}")["slug"] == slug


def test_unsafe_resource_uri_is_rejected_before_database_access(mock_sql_fs):
    fs, conn, cursor = mock_sql_fs
    node = _code_resource_node(slug="foo%2F..%2F_sensitive")

    with pytest.raises(AccessDeniedError, match="Invalid URI format"):
        fs.write_node(node, _ctx())

    assert fs.exists(node.uri, _ctx()) is False
    cursor.execute.assert_not_called()
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()


def test_tool_output_mutation_lock_uses_a_shared_session_advisory_lock(mock_sql_fs):
    fs, conn, cursor = mock_sql_fs
    uri = "ctx://acme/agents/main/memories/tool_outputs"

    with patch.object(
        sql_context_fs.time,
        "monotonic",
        side_effect=[0.0, 0.0, 0.0, 0.0],
    ):
        with fs.tool_output_mutation_lock(uri, _ctx()):
            pass

    sql = _sql_calls(cursor)
    assert any("pg_try_advisory_lock" in statement for statement in sql)
    assert any("pg_advisory_unlock" in statement for statement in sql)
    assert not any(statement == "SELECT pg_advisory_lock(%s)" for statement in sql)
    sql_context_fs.psycopg2.connect.assert_called_once_with(
        fs._connection_string,
        connect_timeout=sql_context_fs._TOOL_OUTPUT_LOCK_CONNECT_TIMEOUT_SECONDS,
    )
    conn.close.assert_called_once()


def test_memory_identity_lock_holds_a_session_advisory_lock(mock_sql_fs):
    fs, conn, cursor = mock_sql_fs
    directory = "ctx://acme/users/alice/memories/preferences"

    with fs.memory_identity_lock(directory, _ctx()):
        assert any("pg_try_advisory_lock" in sql for sql in _sql_calls(cursor))

    assert any("pg_advisory_unlock" in sql for sql in _sql_calls(cursor))
    conn.close.assert_called_once()


def test_tool_output_mutation_lock_times_out_without_unlocking_unowned_lock(
    mock_sql_fs,
):
    fs, conn, cursor = mock_sql_fs
    uri = "ctx://acme/agents/main/memories/tool_outputs"
    cursor.fetchone.return_value = (False,)
    slots = MagicMock()
    slots.acquire.return_value = True

    with (
        patch.object(sql_context_fs, "_TOOL_OUTPUT_LOCK_CONNECTION_SLOTS", slots),
        patch.object(
            sql_context_fs.time,
            "monotonic",
            side_effect=[0.0, 0.0, 0.0, 0.0, 6.0],
        ),
        patch.object(sql_context_fs.time, "sleep"),
    ):
        with pytest.raises(
            TimeoutError,
            match="SQL tool-output mutation lock",
        ):
            with fs.tool_output_mutation_lock(uri, _ctx()):
                raise AssertionError("lock body must not run")

    assert cursor.execute.call_count == 2  # tenant binding + try-lock
    assert _sql_calls(cursor)[-1] == "SELECT pg_try_advisory_lock(%s)"
    assert not any("pg_advisory_unlock" in statement for statement in _sql_calls(cursor))
    conn.close.assert_called_once()
    slots.release.assert_called_once()
    sql_context_fs.psycopg2.connect.assert_called_once_with(
        fs._connection_string,
        connect_timeout=sql_context_fs._TOOL_OUTPUT_LOCK_CONNECT_TIMEOUT_SECONDS,
    )


def test_tool_output_mutation_lock_bounds_extra_connection_slots(mock_sql_fs):
    fs, conn, cursor = mock_sql_fs
    uri = "ctx://acme/agents/main/memories/tool_outputs"
    slots = MagicMock()
    slots.acquire.return_value = False

    with patch.object(sql_context_fs, "_TOOL_OUTPUT_LOCK_CONNECTION_SLOTS", slots):
        with pytest.raises(
            TimeoutError,
            match="SQL tool-output lock connection",
        ):
            with fs.tool_output_mutation_lock(uri, _ctx()):
                raise AssertionError("lock body must not run")

    slots.acquire.assert_called_once()
    slots.release.assert_not_called()
    sql_context_fs.psycopg2.connect.assert_not_called()
    cursor.execute.assert_not_called()


@pytest.mark.parametrize("remaining, expected", [(None, 120.0), (3.0, 3.0)])
def test_guarded_memory_identity_wait_uses_caller_deadline(mock_sql_fs, remaining, expected):
    from contextlib import nullcontext

    fs, _, _ = mock_sql_fs
    guard = MagicMock()
    guard.wait_for_identity_lock = True
    guard.identity_lock.return_value = nullcontext()
    directory = "ctx://acme/users/alice/memories/preferences"

    with (
        patch("fs.sql_adapter.session_mutation.active_session_guard", return_value=guard),
        patch("core.operation_deadline.remaining_seconds", return_value=remaining),
        fs.memory_identity_lock(directory, _ctx()),
    ):
        pass

    guard.identity_lock.assert_called_once_with(
        _resource_mutation_lock_key("acme", directory), _ctx(), expected,
    )


@pytest.fixture
def mock_sql_fs():
    """Create SQLContextFS with mocked psycopg2."""

    # Mock Json to just return the value (identity wrapper)
    def mock_json(x):
        return x

    with (
        patch("fs.sql_adapter.sql_context_fs.psycopg2") as mock_pg,
        patch("fs.sql_adapter.sql_context_fs.Json", mock_json),
    ):
        mock_conn = MagicMock()
        mock_conn.closed = 0
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value = mock_conn

        # Import after patching
        from fs.sql_adapter.sql_context_fs import SQLContextFS

        fs = SQLContextFS.__new__(SQLContextFS)
        fs._connection_string = "host=localhost dbname=test"
        fs._pool_size = 5
        fs._pool = []

        # Override pool methods to use mock
        fs._get_connection = lambda: mock_conn
        fs._return_connection = lambda c: None

        yield fs, mock_conn, mock_cursor


def test_inventory_counts_agent_tools_and_experiences(mock_sql_fs):
    fs, _, cursor = mock_sql_fs
    tools = "ctx://acme/agents/main/memories/tools"
    experiences = "ctx://acme/agents/main/memories/experiences"
    cursor.fetchall.return_value = [(tools, "tool", 3), (experiences, "experience", 2)]

    assert fs.count_memory_children(_ctx()) == {tools: 3, experiences: 2}
    query = next(call for call in cursor.execute.call_args_list if "GROUP BY parent" in call.args[0])
    assert {"tool", "experience"} <= set(query.args[1][1])


class TestSQLContextFSWriteNode:
    def test_write_new_node(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        ctx = _ctx()

        fs.write_node(node, ctx)

        conn.commit.assert_called_once()
        # Verify INSERT was executed (cursor.execute called)
        assert cursor.execute.called
        sql_calls = _sql_calls(cursor)
        assert not any("pg_advisory_xact_lock" in sql for sql in sql_calls)
        assert not any(sql.startswith("SELECT 1 FROM outbox_events") for sql in sql_calls)

    def test_write_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        ctx = _ctx(account_id="other")  # Different account

        with pytest.raises(AccessDeniedError):
            fs.write_node(node, ctx)

    def test_write_with_relations(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        edge = RelationEdge(
            from_uri="ctx://acme/users/alice/memories/profile",
            to_uri="ctx://acme/users/alice/memories/preferences/coffee",
            relation_type="related_to",
            weight=0.8,
            reason="test",
        )
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="abs",
            overview="ov",
            content="cont",
            metadata={"_relations": [edge]},
        )
        ctx = _ctx()

        fs.write_node(node, ctx)
        conn.commit.assert_called_once()

    def test_write_optimistic_lock_failure(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        node.metadata["expected_version"] = 3
        ctx = _ctx()

        # Simulate UPDATE matching 0 rows (version mismatch)
        cursor.rowcount = 0
        # After rowcount==0, code does SELECT to get actual version
        cursor.fetchone.return_value = (5,)

        with pytest.raises(ConcurrentModificationError):
            fs.write_node(node, ctx)
        conn.rollback.assert_called_once()

    def test_write_optimistic_lock_create_new(self, mock_sql_fs):
        """Optimistic lock with expected_version=None should use INSERT ON CONFLICT (create)."""
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        ctx = _ctx()

        # No expected_version → INSERT ON CONFLICT path
        fs.write_node(node, ctx)
        conn.commit.assert_called_once()

    def test_write_create_only_rejects_existing_uri(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        node.metadata["_expected_version"] = 0
        cursor.rowcount = 0
        cursor.fetchone.return_value = (4,)

        with pytest.raises(ConcurrentModificationError):
            fs.write_node(node, _ctx())

        assert any(
            "ON CONFLICT (uri) DO NOTHING" in call.args[0] for call in cursor.execute.call_args_list
        )
        conn.rollback.assert_called_once()

    def test_public_expected_version_zero_uses_legacy_update(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        node.metadata["expected_version"] = 0
        cursor.rowcount = 1

        fs.write_node(node, _ctx())

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "UPDATE context_nodes SET" in sql
        assert "WHERE uri = %s AND version = %s" in sql
        assert params[-1] == 0
        assert "ON CONFLICT (uri) DO NOTHING" not in sql
        conn.commit.assert_called_once()

    def test_internal_expected_version_is_not_persisted(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _profile_node()
        node.metadata["_expected_version"] = 0
        cursor.rowcount = 1

        fs.write_node(node, _ctx())

        insert_params = cursor.execute.call_args.args[1]
        persisted_metadata = insert_params[10]
        assert "_expected_version" not in persisted_metadata


class TestSQLContextFSCodeResourceMutationGuard:
    def test_mutation_lock_key_is_stable_namespaced_and_signed(self):
        uri = "ctx://acme/resources/chunk-a"

        key = _resource_mutation_lock_key("acme", uri)

        assert key == -6666838223963259778
        assert key == _resource_mutation_lock_key("acme", uri)
        assert key == _resource_mutation_lock_key(" acme ", uri + "/")
        assert key != _resource_mutation_lock_key("other", uri)
        assert key != _resource_mutation_lock_key("acme", uri + "-other")
        assert -(2**63) <= key < 2**63

    def test_plain_code_write_requires_atomic_outbox_before_mutation(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _code_resource_node()

        with pytest.raises(
            RuntimeError,
            match="code resource writes require write_node_with_outbox",
        ):
            fs.write_node(node, _ctx())

        cursor.execute.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_not_called()

    def test_atomic_code_write_locks_and_checks_delete_before_mutation(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _code_resource_node()
        cursor.fetchone.return_value = None

        fs.write_node_with_outbox(node, _ctx(), _write_event(node))

        sql_calls = _sql_calls(cursor)
        lock_index = next(i for i, sql in enumerate(sql_calls) if "pg_advisory_xact_lock" in sql)
        marker_index = next(
            i for i, sql in enumerate(sql_calls) if sql.startswith("SELECT 1 FROM outbox_events")
        )
        node_write_index = next(
            i for i, sql in enumerate(sql_calls) if sql.startswith("INSERT INTO context_nodes")
        )
        event_write_index = next(
            i for i, sql in enumerate(sql_calls) if sql.startswith("INSERT INTO outbox_events")
        )
        assert lock_index < marker_index < node_write_index < event_write_index
        conn.commit.assert_called_once()

    def test_atomic_code_update_increments_column_and_metadata_version(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _code_resource_node()
        node.metadata["_expected_version"] = 2
        cursor.fetchone.return_value = None
        cursor.rowcount = 1

        fs.write_node_with_outbox(node, _ctx(), _write_event(node))

        update_call = next(
            call
            for call in cursor.execute.call_args_list
            if "UPDATE context_nodes SET" in call.args[0]
        )
        update_sql = " ".join(update_call.args[0].split())
        update_params = update_call.args[1]
        assert "version = version + 1" in update_sql
        assert update_params[4]["version"] == 3
        assert update_params[-1] == 2
        conn.commit.assert_called_once()

    def test_atomic_code_write_rejects_pending_durable_delete(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        node = _code_resource_node()
        cursor.fetchone.return_value = (1,)

        with pytest.raises(RuntimeError, match="pending durable deletion"):
            fs.write_node_with_outbox(node, _ctx(), _write_event(node))

        sql_calls = _sql_calls(cursor)
        assert not any(sql.startswith("INSERT INTO context_nodes") for sql in sql_calls)
        assert not any(sql.startswith("INSERT INTO outbox_events") for sql in sql_calls)
        marker_sql = next(sql for sql in sql_calls if sql.startswith("SELECT 1 FROM outbox_events"))
        assert "'PENDING', 'PROCESSING', 'FAILED'" in marker_sql
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()


class TestSQLContextFSWorkspaceRootBinding:
    def test_reserve_uses_the_protected_workspace_sync_row(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        fingerprint = "a" * 64
        cursor.fetchone.return_value = ({},)

        binding = fs.reserve_code_workspace_root_binding(
            ctx=_ctx(),
            root_slug="repo",
            root_fingerprint=fingerprint,
        )

        root_id = compute_workspace_root_id("acme", fingerprint)
        assert binding == (root_id, "repo", fingerprint)
        sql_calls = _sql_calls(cursor)
        assert any(sql.startswith("INSERT INTO code_workspace_sync_roots") for sql in sql_calls)
        assert any(
            sql.startswith("SELECT state FROM code_workspace_sync_roots")
            and sql.endswith("FOR UPDATE")
            for sql in sql_calls
        )
        update_call = next(
            call
            for call in cursor.execute.call_args_list
            if "UPDATE code_workspace_sync_roots" in call.args[0]
        )
        state = update_call.args[1][0]
        assert state == {
            "version": 1,
            "by_fingerprint": {
                fingerprint: {
                    "root_id": root_id,
                    "resource_root_slug": "repo",
                }
            },
            "by_slug": {
                "repo": {
                    "root_id": root_id,
                    "root_fingerprint": fingerprint,
                }
            },
        }
        conn.commit.assert_called_once()

    def test_reserve_rejects_conflict_before_any_resource_write(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        fingerprint = "a" * 64
        root_id = compute_workspace_root_id("acme", fingerprint)
        cursor.fetchone.return_value = (
            {
                "version": 1,
                "by_fingerprint": {
                    fingerprint: {
                        "root_id": root_id,
                        "resource_root_slug": "repo",
                    }
                },
                "by_slug": {
                    "repo": {
                        "root_id": root_id,
                        "root_fingerprint": fingerprint,
                    }
                },
            },
        )

        with pytest.raises(RootBindingError, match="different root slug"):
            fs.reserve_code_workspace_root_binding(
                ctx=_ctx(),
                root_slug="other",
                root_fingerprint=fingerprint,
            )

        assert not any(sql.startswith("INSERT INTO context_nodes") for sql in _sql_calls(cursor))
        assert not any(sql.startswith("INSERT INTO outbox_events") for sql in _sql_calls(cursor))
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()


class TestSQLContextFSAccess:
    def test_session_state_matching_session_allowed(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, _ctx(session_id="session-1")) is True

    def test_session_state_different_session_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, _ctx(session_id="session-2")) is False

    def test_session_history_same_account_remains_accessible(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/sessions/session-1/history"
        assert fs._is_accessible(uri, _ctx(session_id="session-2")) is True

    def test_session_archive_same_account_remains_accessible(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/sessions/session-1/history/archive-1"
        assert fs._is_accessible(uri, _ctx(session_id="session-2")) is True


class TestSQLContextFSReadNode:
    def test_read_existing_node(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        cursor.fetchone.return_value = (
            "ctx://acme/users/alice/memories/profile",
            "MEMORY",
            "profile",
            0,
            "user:alice",
            "Short abstract",
            "## Overview",
            "Profile content",
            [],  # relations
            {"tags": ["test"], "version": 1},  # metadata
            "ACTIVE",
            datetime.now(UTC),
            datetime.now(UTC),
            1,
        )

        node = fs.read_node("ctx://acme/users/alice/memories/profile", ctx)
        assert node.uri == "ctx://acme/users/alice/memories/profile"
        assert node.content == "Profile content"
        assert node.abstract == "Short abstract"

    def test_read_not_found(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        cursor.fetchone.return_value = None

        with pytest.raises(NodeNotFoundError):
            fs.read_node("ctx://acme/users/alice/memories/profile", ctx)

    def test_read_broken_node(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        cursor.fetchone.return_value = (
            "ctx://acme/users/alice/memories/profile",
            "MEMORY",
            "profile",
            0,
            "user:alice",
            "",
            "",
            "",
            [],
            {},
            "BROKEN",
            None,
            None,
            1,
        )

        with pytest.raises(NodeBrokenError):
            fs.read_node("ctx://acme/users/alice/memories/profile", ctx)

    def test_read_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx(account_id="other")

        with pytest.raises(AccessDeniedError):
            fs.read_node("ctx://acme/users/alice/memories/profile", ctx)

    def test_read_with_relations(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        relations = [
            {
                "from_uri": "ctx://acme/users/alice/memories/profile",
                "to_uri": "ctx://acme/users/alice/memories/preferences/coffee",
                "relation_type": "related_to",
                "weight": 0.8,
                "reason": "test",
            }
        ]

        cursor.fetchone.return_value = (
            "ctx://acme/users/alice/memories/profile",
            "MEMORY",
            "profile",
            0,
            "user:alice",
            "abs",
            "ov",
            "cont",
            relations,  # JSONB relations
            {"version": 1},
            "ACTIVE",
            datetime.now(UTC),
            datetime.now(UTC),
            1,
        )

        node = fs.read_node("ctx://acme/users/alice/memories/profile", ctx)
        assert len(node.metadata["_relations"]) == 1
        assert node.metadata["_relations"][0].relation_type == "related_to"


class TestSQLContextFSExists:
    def test_exists_active(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchone.return_value = (1,)

        assert fs.exists("ctx://acme/users/alice/memories/profile", ctx) is True

    def test_not_exists(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchone.return_value = None

        assert fs.exists("ctx://acme/users/alice/memories/profile", ctx) is False

    def test_archived_node_version(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchone.return_value = (4,)

        assert fs.archived_node_version("ctx://acme/users/alice/memories/preferences/pets", ctx) == 4
        assert "status = 'ARCHIVED'" in _sql_calls(cursor)[-1]

    def test_exists_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx(account_id="other")

        assert fs.exists("ctx://acme/users/alice/memories/profile", ctx) is False


class TestSQLContextFSDeleteNode:
    def test_delete_existing(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.rowcount = 1

        fs.delete_node("ctx://acme/users/alice/memories/profile", ctx)
        conn.commit.assert_called_once()

    def test_delete_not_found(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.rowcount = 0

        with pytest.raises(NodeNotFoundError):
            fs.delete_node("ctx://acme/users/alice/memories/profile", ctx)

    def test_delete_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx(account_id="other")

        with pytest.raises(AccessDeniedError):
            fs.delete_node("ctx://acme/users/alice/memories/profile", ctx)

    def test_plain_code_delete_requires_atomic_outbox_before_mutation(
        self,
        mock_sql_fs,
    ):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/resources/chunk-a"
        cursor.fetchone.return_value = (
            ContextType.RESOURCE.value,
            {"resource_type": "code"},
        )

        with pytest.raises(
            RuntimeError,
            match="code resource deletes require delete_nodes_with_outbox",
        ):
            fs.delete_node(uri, _ctx())

        sql_calls = _sql_calls(cursor)
        assert any("pg_advisory_xact_lock" in sql for sql in sql_calls)
        assert any(sql.endswith("FOR UPDATE") for sql in sql_calls)
        assert not any(sql.startswith("DELETE FROM context_nodes") for sql in sql_calls)
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()


class TestSQLContextFSDurableDelete:
    @staticmethod
    def _events(*uris):
        from commit.sql_outbox_store import SQLOutboxStore

        return [SQLOutboxStore.build_delete_event(uri) for uri in uris]

    def test_delete_nodes_locks_cancels_and_enqueues_in_one_transaction(
        self,
        mock_sql_fs,
    ):
        fs, conn, cursor = mock_sql_fs
        uris = [
            "ctx://acme/resources/chunk-a",
            "ctx://acme/resources/chunk-b",
        ]
        cursor.rowcount = 2

        deleted = fs.delete_nodes_with_outbox(uris, _ctx(), self._events(*uris))

        assert deleted == 2
        sql_calls = _sql_calls(cursor)
        advisory_indexes = [i for i, sql in enumerate(sql_calls) if "pg_advisory_xact_lock" in sql]
        row_lock_index = next(i for i, sql in enumerate(sql_calls) if "FOR UPDATE" in sql)
        delete_index = next(
            i for i, sql in enumerate(sql_calls) if sql.startswith("DELETE FROM context_nodes")
        )
        cancel_index = next(
            i for i, sql in enumerate(sql_calls) if sql.startswith("DELETE FROM outbox_events")
        )
        insert_indexes = [
            i for i, sql in enumerate(sql_calls) if sql.startswith("INSERT INTO outbox_events")
        ]
        assert [cursor.execute.call_args_list[index].args[1] for index in advisory_indexes] == [
            (_resource_mutation_lock_key("acme", uri),) for uri in sorted(uris)
        ]
        assert max(advisory_indexes) < row_lock_index < delete_index < cancel_index
        assert cancel_index < min(insert_indexes)
        assert len(insert_indexes) == 2
        assert sum("SELECT pg_notify" in sql for sql in sql_calls) == 2
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()

    def test_delete_nodes_rolls_back_every_step_when_event_insert_fails(
        self,
        mock_sql_fs,
    ):
        fs, conn, cursor = mock_sql_fs
        uri = "ctx://acme/resources/chunk-a"
        cursor.rowcount = 1

        def execute(sql, params=None):
            if "INSERT INTO outbox_events" in sql:
                raise RuntimeError("outbox insert failed")
            return None

        cursor.execute.side_effect = execute

        with pytest.raises(RuntimeError, match="outbox insert failed"):
            fs.delete_nodes_with_outbox([uri], _ctx(), self._events(uri))

        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()


class TestSQLContextFSListChildren:
    def test_list_children(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        # Return URIs under preferences/
        cursor.fetchall.return_value = [
            ("ctx://acme/users/alice/memories/preferences/",),
            ("ctx://acme/users/alice/memories/preferences/coffee",),
            ("ctx://acme/users/alice/memories/preferences/tea",),
            (
                "ctx://acme/users/alice/memories/preferences/food/sushi",
            ),  # nested, should be excluded
        ]

        children = fs.list_children("ctx://acme/users/alice/memories/preferences", ctx)

        # Only immediate children (no nested paths)
        assert len(children) == 2
        assert "ctx://acme/users/alice/memories/preferences/coffee" in children
        assert "ctx://acme/users/alice/memories/preferences/tea" in children

    def test_list_children_empty(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchall.return_value = []

        children = fs.list_children("ctx://acme/users/alice/memories/preferences", ctx)
        assert children == []

    def test_list_children_filters_by_account(self, mock_sql_fs):
        """list_children SQL must include account_id filter."""
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchall.return_value = []

        fs.list_children("ctx://acme/users/alice/memories/preferences", ctx)

        # Verify the SQL parameters include account_id
        execute_call = cursor.execute.call_args
        sql, params = execute_call[0]
        assert "account_id" in sql
        assert "acme" in params  # account_id extracted from URI

    def test_list_children_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx(account_id="other")

        with pytest.raises(AccessDeniedError):
            fs.list_children("ctx://acme/users/alice/memories/preferences", ctx)

    def test_list_node_summaries_uses_summary_columns_and_limit(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.fetchall.return_value = [
            (
                "ctx://acme/users/alice/memories/preferences/coffee",
                "MEMORY",
                "preference",
                0,
                "user:alice",
                "Likes coffee",
                {"version": 2},
                datetime.now(UTC),
                datetime.now(UTC),
                2,
                True,
            )
        ]

        summaries = fs.list_node_summaries(
            "ctx://acme/users/alice/memories/preferences",
            ctx,
            limit=1,
        )

        assert len(summaries) == 1
        assert summaries[0].uri.endswith("/coffee")
        assert summaries[0].content == ""
        assert summaries[0].metadata["has_content"] is True
        sql, params = cursor.execute.call_args[0]
        assert " content," not in sql
        assert "POSITION('/' IN SUBSTRING" in sql
        assert "CHAR_LENGTH(uri) > CHAR_LENGTH(%s)" in sql
        assert params[2].endswith("/preferences/")
        assert params[-1] == 1


class TestSQLContextFSMoveNode:
    def test_move_existing(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        # First UPDATE (move node itself) matches 1 row
        cursor.rowcount = 1

        fs.move_node(
            "ctx://acme/users/alice/memories/preferences/coffee",
            "ctx://acme/users/alice/memories/preferences/latte",
            ctx,
        )
        conn.commit.assert_called_once()

    def test_move_source_not_found(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()

        # First UPDATE matches 0 rows → NodeNotFoundError
        cursor.rowcount = 0

        with pytest.raises(NodeNotFoundError):
            fs.move_node(
                "ctx://acme/users/alice/memories/preferences/coffee",
                "ctx://acme/users/alice/memories/preferences/latte",
                ctx,
            )

    def test_move_access_denied(self, mock_sql_fs):
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx(account_id="other")

        with pytest.raises(AccessDeniedError):
            fs.move_node(
                "ctx://acme/users/alice/memories/preferences/coffee",
                "ctx://acme/users/alice/memories/preferences/latte",
                ctx,
            )

    def test_move_cascades_to_relations_and_outbox(self, mock_sql_fs):
        """move_node must UPDATE relation_edges and emit DELETE + UPSERT outbox events."""
        import hashlib

        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.rowcount = 1  # main node UPDATE matches 1 row

        # to_regclass returns table names → relation_edges and outbox_events exist
        # fetchone is called for: to_regclass('relation_edges'), to_regclass('outbox_events')
        cursor.fetchone.side_effect = [
            ("relation_edges",),  # to_regclass('relation_edges')
            ("outbox_events",),  # to_regclass('outbox_events')
        ]

        new_uri = "ctx://acme/users/alice/memories/preferences/latte"
        old_uri = "ctx://acme/users/alice/memories/preferences/coffee"

        # fetchall is called twice:
        # 1) SELECT uri ... for DELETE events (all moved nodes)
        # 2) SELECT uri,abstract,... for UPSERT events (re-index)
        cursor.fetchall.side_effect = [
            [(old_uri,)],  # ordered endpoint lock
            [],  # no graph jobs on this legacy move
            [(new_uri,)],  # all_new_uris — just the moved node
            [(new_uri, "abs", "ov", "content", "user:alice", "preference", "MEMORY", {}, 1)],  # moved_rows
        ]

        fs.move_node(old_uri, new_uri, ctx)
        conn.commit.assert_called_once()

        sql_calls = [str(c[0][0]) for c in cursor.execute.call_args_list]

        # 1) relation_edges cascade
        relation_updates = [s for s in sql_calls if "relation_edges" in s]
        assert len(relation_updates) >= 2, (
            f"Expected >= 2 relation_edges updates, got {len(relation_updates)}"
        )

        # 2) DELETE_CONTEXT: verify correct ids including L2 /content.md
        delete_calls = [
            c for c in cursor.execute.call_args_list if "DELETE_CONTEXT" in str(c[0][0])
        ]
        assert len(delete_calls) == 1, f"Expected 1 DELETE_CONTEXT, got {len(delete_calls)}"
        # call[0] = (sql, (event_id, uri, account_id, payload_dict))
        delete_params = delete_calls[0][0][1]  # params tuple
        payload = delete_params[3]  # 4th element = Json(payload)
        assert isinstance(payload, dict), f"Expected dict payload, got {type(payload)}"
        ids = payload["ids_to_delete"]
        assert len(ids) == 3, f"Expected 3 ids (L0/L1/L2), got {len(ids)}"
        expected_l2 = hashlib.sha256(f"{old_uri}/content.md:2".encode()).hexdigest()[:16]
        assert expected_l2 in ids, f"L2 id {expected_l2} not in {ids}"

        # 3) UPSERT_CONTEXT: verify records with correct new-URI ids
        upsert_calls = [
            c for c in cursor.execute.call_args_list if "UPSERT_CONTEXT" in str(c[0][0])
        ]
        assert len(upsert_calls) == 1, f"Expected 1 UPSERT_CONTEXT, got {len(upsert_calls)}"
        upsert_params = upsert_calls[0][0][1]
        records = upsert_params[3]["records"]
        assert len(records) >= 1, "Expected at least 1 record"
        expected_new_l2 = hashlib.sha256(f"{new_uri}/content.md:2".encode()).hexdigest()[:16]
        all_ids = [r["id"] for r in records]
        assert expected_new_l2 in all_ids, f"New L2 id {expected_new_l2} not in {all_ids}"

        # 4) UPSERT metadata must include parent_uri, category, context_type,
        #    has_overview, has_content — matching build_index_records() output.
        for rec in records:
            meta = rec["metadata"]
            assert "category" in meta, f"Missing 'category' in metadata: {meta}"
            assert "context_type" in meta, f"Missing 'context_type' in metadata: {meta}"
            assert "parent_uri" in meta, f"Missing 'parent_uri' in metadata: {meta}"
            assert "has_overview" in meta, f"Missing 'has_overview' in metadata: {meta}"
            assert "has_content" in meta, f"Missing 'has_content' in metadata: {meta}"
        # Spot-check values
        l0 = [r for r in records if r["level"] == 0][0]
        assert l0["metadata"]["category"] == "preference"
        assert l0["metadata"]["context_type"] == "MEMORY"
        assert l0["metadata"]["has_overview"] is True
        assert l0["metadata"]["has_content"] is True
        # L2 must have parent_uri pointing to the directory URI (not /content.md)
        l2 = [r for r in records if r["level"] == 2][0]
        assert l2["metadata"]["parent_uri"] == new_uri

    def test_move_leaf_node_includes_root_in_upsert(self, mock_sql_fs):
        """Moving a leaf node: the root to_uri must appear in UPSERT query."""
        fs, conn, cursor = mock_sql_fs
        ctx = _ctx()
        cursor.rowcount = 1

        cursor.fetchone.side_effect = [
            ("relation_edges",),
            ("outbox_events",),
        ]
        new_uri = "ctx://acme/users/alice/memories/preferences/latte"
        cursor.fetchall.side_effect = [
            [],  # endpoint lock result
            [],  # no graph jobs
            [(new_uri,)],
            [(new_uri, "a", "o", "c", "user:alice", "preference", "MEMORY", {}, 1)],
        ]

        fs.move_node(
            "ctx://acme/users/alice/memories/preferences/coffee",
            new_uri,
            ctx,
        )

        # Verify the SELECT for re-index uses `uri = %s OR uri LIKE %s`
        # so the root to_uri is included (not just children under prefix)
        sql_calls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        select_for_upsert = [
            s for s in sql_calls if "SELECT uri, abstract" in s and "uri = %s OR uri LIKE %s" in s
        ]
        assert len(select_for_upsert) == 1, (
            "Expected SELECT with `uri = %s OR uri LIKE %s` for re-index"
        )
