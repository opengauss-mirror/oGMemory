"""Business-level tests for SQL lifecycle trace persistence."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from fs.sql_adapter.sql_trace_store import (
    SQLTraceEventStore,
    _turn_id,
)
from skills.artifact import validate_skill_publication


@pytest.fixture
def store_and_db():
    store = SQLTraceEventStore.__new__(SQLTraceEventStore)
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda _: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    store._get_connection = lambda: conn
    store._return_connection = lambda _: None
    store.bind_tenant = MagicMock()
    store._max_turn_events = 128
    return store, conn, cursor


def _event_row(event_id="evt-1", event_index=17):
    return (
        event_id,
        f"ctx://acct/traces/sess/events/{event_id}",
        "acct",
        "user/alice",
        "agent",
        "sess",
        "PreToolUse",
        event_index,
        datetime(2026, 7, 15, tzinfo=UTC),
        {"event_type": "tool_start"},
        "[PreToolUse] Read",
        "[PreToolUse] Read",
    )


def test_insert_is_idempotent_by_stable_event_id(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.side_effect = [None, _event_row()]

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.insert(
            event_id="evt-1",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="PreToolUse",
            content={"event_type": "tool_start", "tool_name": "Read"},
        )

    assert result == {
        "ok": True,
        "event_id": "evt-1",
        "uri": "ctx://acct/traces/sess/events/evt-1",
        "event_index": 17,
        "inserted": False,
    }
    assert "ON CONFLICT (account_id, event_id) DO NOTHING" in cursor.execute.call_args_list[0].args[0]
    assert "WHERE account_id = %s AND event_id = %s" in cursor.execute.call_args_list[1].args[0]
    assert cursor.execute.call_args_list[1].args[1] == ("acct", "evt-1")
    store.bind_tenant.assert_called_once_with(conn, "acct")
    conn.commit.assert_called_once()


def test_insert_rolls_back_database_errors(store_and_db):
    store, conn, cursor = store_and_db
    cursor.execute.side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        store.insert(
            event_id="evt-1",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="Stop",
            content={"event_type": "assistant_message"},
        )

    conn.rollback.assert_called_once()


def test_insert_persists_canonical_turn_end_type(store_and_db):
    store, _, cursor = store_and_db
    cursor.fetchone.return_value = _event_row(event_id="evt-turn-end", event_index=18)

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.insert(
            event_id="evt-turn-end",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="turn_end",
            content={"event_type": "turn_end"},
        )

    assert result["inserted"] is True
    insert_params = cursor.execute.call_args_list[0].args[1]
    assert insert_params[6] == "turn_end"
    assert insert_params[8] == {"event_type": "turn_end"}


@pytest.mark.parametrize("write_batch_id", ["", "batch-A"])
@pytest.mark.parametrize("boundary_type", ["turn_end", "Stop"])
def test_finalize_aggregates_only_the_turn_end_bounded_turn(
    store_and_db, boundary_type, write_batch_id
):
    store, conn, cursor = store_and_db
    started = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    ended = datetime(2026, 7, 15, 8, 1, tzinfo=UTC)
    cursor.fetchone.side_effect = [
        (ended, write_batch_id),
        *([(9,)] if not write_batch_id else []),
        (3, started, ended),
        ("ctx://acct/traces/sess/turns/t",),
    ]
    cursor.fetchall.return_value = [
        (10, "UserPromptSubmit", started, {"event_type": "user_message", "prompt": "hello"}, "hello"),
        (11, "PreToolUse", started, {"event_type": "tool_start", "tool_name": "Read"}, "[PreToolUse] Read"),
        (12, boundary_type, ended, {"event_type": "turn_end", "assistant_message": "done"}, "done"),
    ]

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.finalize_session(
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            stop_event_index=12,
        )

    assert result["event_count"] == 3
    assert result["prev_stop_event_index"] == (None if write_batch_id else 9)
    boundary_query = cursor.execute.call_args_list[0].args[0]
    assert "hook_type IN ('turn_end', 'Stop')" in boundary_query
    offset = 0 if write_batch_id else 1
    range_query = cursor.execute.call_args_list[2 + offset]
    assert range_query.args[1][:-1] == (
        "acct",
        "sess",
        "user/alice",
        "agent",
        0 if write_batch_id else 9,
        12,
        write_batch_id,
    )
    upsert = cursor.execute.call_args_list[3 + offset]
    assert "INSERT INTO trace_sessions" in upsert.args[0]
    assert "ON CONFLICT (account_id, turn_id) DO UPDATE" in upsert.args[0]
    assert "hello" in upsert.args[1][9]
    assert "done" in upsert.args[1][9]
    assert upsert.args[1][10] == upsert.args[1][9]
    executed_sql = "\n".join(call.args[0] for call in cursor.execute.call_args_list)
    assert "INSERT INTO trace_session_events" in executed_sql
    event_projection = cursor.execute.call_args_list[5 + offset]
    assert "FROM trace_events" in event_projection.args[0]
    assert "ROW_NUMBER() OVER (ORDER BY event_index)" in event_projection.args[0]
    assert "jsonb_array_elements" not in event_projection.args[0]
    assert event_projection.args[1] == (
        "acct",
        "ctx://acct/traces/sess/turns/"
        + _turn_id("acct", "user/alice", "agent", "sess", 12),
        "acct",
        "sess",
        "user/alice",
        "agent",
        0 if write_batch_id else 9,
        12,
        write_batch_id,
    )
    assert "INSERT INTO skill_candidates" not in executed_sql
    assert "INSERT INTO skill_work_accounts" not in executed_sql
    conn.commit.assert_called_once()


def test_enqueue_skill_candidate_from_trace_is_idempotent(store_and_db):
    store, conn, cursor = store_and_db

    result = store.enqueue_skill_candidate_from_trace(
        account_id="acct",
        agent_id="agent",
        session_id="sess",
        turn_id="turn_123",
        trace_uri="ctx://acct/traces/sess/turns/turn_123",
    )

    executed_sql = "\n".join(call.args[0] for call in cursor.execute.call_args_list)
    assert "INSERT INTO skill_candidates" in executed_sql
    assert "INSERT INTO skill_candidate_evidence" in executed_sql
    assert "INSERT INTO skill_work_accounts" in executed_sql
    assert "ON CONFLICT (account_id) DO UPDATE" in executed_sql
    assert "SET updated_at = EXCLUDED.updated_at" in executed_sql
    assert result == {"ok": True, "candidate_id": "ctx://acct/skills/candidates/turn_123"}
    conn.commit.assert_called_once()


def test_fetch_turn_events_caps_stored_projection(store_and_db):
    store, _, cursor = store_and_db
    started = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    ended = datetime(2026, 7, 15, 8, 5, tzinfo=UTC)
    cursor.fetchone.side_effect = [
        (200, started, ended),
        (1, "UserPromptSubmit", started, {"event_type": "user_message"}, "hello"),
    ]
    cursor.fetchall.return_value = [
        (index, "PreToolUse", ended, {"event_type": "tool_start"}, "tool")
        for index in range(200, 73, -1)
    ]

    events, total, actual_start, actual_end = store._fetch_turn_events(
        cursor, "acct", "user:alice", "agent", "sess", 0, 200
    )

    assert total == 200
    assert len(events) == 128
    assert events[0]["event_index"] == 1
    assert events[1]["event_index"] == 74
    assert events[-1]["event_index"] == 200
    assert (actual_start, actual_end) == (started, ended)
    assert cursor.execute.call_args_list[2].args[1][-1] == 127

    metadata = store._session_metadata(events, total)
    assert metadata["truncated"] is True
    assert metadata["stored_event_count"] == 128


def test_truncated_metadata_labels_retained_sample_counts():
    events = [
        {
            "hook_type": "UserPromptSubmit",
            "content": {"event_type": "user_message"},
            "abstract": "hello",
        },
        {
            "hook_type": "Stop",
            "content": {"event_type": "assistant_message"},
            "abstract": "done",
        },
    ]

    metadata = SQLTraceEventStore._session_metadata(events, total_event_count=4)

    assert metadata["truncated"] is True
    assert metadata["stored_tool_event_count"] == 0
    assert metadata["stored_tools_used"] == []
    assert metadata["stored_hook_event_counts"] == {
        "UserPromptSubmit": 1,
        "Stop": 1,
    }
    assert "tool_call_count" not in metadata


def test_turn_id_includes_full_tenant_and_owner_identity():
    first = _turn_id("acct-a", "user:alice", "agent", "session-full", 7)
    other_tenant = _turn_id("acct-b", "user:alice", "agent", "session-full", 7)
    other_owner = _turn_id("acct-a", "user:bob", "agent", "session-full", 7)

    assert first.startswith("turn_")
    assert len(first) == 37
    assert len({first, other_tenant, other_owner}) == 3


def test_claim_skill_candidates_uses_skip_locked_and_lease(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchall.return_value = [
        ("turn", "trace:turn", "acct", "user:alice", "agent", "session",
         {}, 1, {})
    ]
    rows = store.claim_skill_candidates(
        account_id="acct", worker_id="worker-1", limit=10, lease_seconds=300,
    )
    query, params = cursor.execute.call_args.args
    assert "FOR UPDATE SKIP LOCKED" in query
    assert "FROM skill_candidates c" in query
    assert "JOIN trace_sessions t" in query
    assert "c.lease_until <= NOW()" in query
    assert params == ("acct", 10, "worker-1", 300)
    assert rows[0]["turn_id"] == "turn"
    assert rows[0]["skill_attempts"] == 1
    assert "content" not in rows[0]
    conn.commit.assert_called_once()


def test_claim_skill_evidence_serializes_and_leases_diverse_batch(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.side_effect = [(True,), None, (False,), (5, 3, False)]
    cursor.fetchall.return_value = [
        (
            "trace:1", "session-1", [],
            {"summary": "one", "staged_publication": {"uri": "ctx://skill"}}, 2,
        ),
        ("trace:2", "session-2", [], {"summary": "two"}, 2),
    ]

    evidence = store.claim_skill_evidence(
        account_id="acct", agent_id="agent", capability_key="review-code",
        worker_id="worker", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=300,
    )

    lock_query = cursor.execute.call_args_list[0].args[0]
    staged_query = cursor.execute.call_args_list[1].args[0]
    lease_query = cursor.execute.call_args_list[2].args[0]
    claim_query, claim_params = cursor.execute.call_args_list[4].args
    assert "pg_try_advisory_xact_lock" in lock_query
    assert "staged_publication" in staged_query
    assert "FROM skill_candidates" in lease_query
    assert "state = 'processing'" in lease_query
    assert "capability_key IS NOT NULL" in lease_query
    assert cursor.execute.call_args_list[2].args[1] == ("acct", "agent")
    assert "ROW_NUMBER() OVER" in claim_query
    assert "lease_owner = %s" in claim_query
    assert "compile_attempts = compile_attempts + 1" in claim_query
    assert claim_params[-4:] == (20, "worker", 300, "acct")
    assert evidence[0]["skill_compile_attempts"] == 2
    assert evidence[0]["staged_publication"] == {"uri": "ctx://skill"}
    assert evidence[0]["content"] == []
    conn.commit.assert_called_once()


def test_claim_skill_evidence_prefers_compacted_source_over_legacy_content(store_and_db):
    store, _conn, cursor = store_and_db
    cursor.fetchone.side_effect = [(True,), None, (False,), (5, 3, False)]
    compacted = [{"event_type": "user_message", "prompt": "bounded"}]
    cursor.fetchall.return_value = [
        ("trace:1", "session-1", None, {"source_events": compacted}, 1),
    ]

    evidence = store.claim_skill_evidence(
        account_id="acct", agent_id="agent", capability_key="review-code",
        worker_id="worker", min_evidence_turns=5, min_distinct_sessions=3,
        max_evidence_turns=20, lease_seconds=300,
    )

    assert evidence[0]["content"] == compacted


def test_claim_skill_evidence_requires_two_rows_when_support_is_pending(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.side_effect = [(True,), None, (False,), (1, 1, True), (False,)]

    evidence = store.claim_skill_evidence(
        account_id="acct", agent_id="agent", capability_key="review-code",
        worker_id="worker", min_evidence_turns=1, min_distinct_sessions=1,
        max_evidence_turns=20, lease_seconds=300,
    )

    count_query = cursor.execute.call_args_list[3].args[0]
    assert "eligibility_state" in count_query
    assert evidence == []
    conn.rollback.assert_called_once()


def test_read_skill_trace_event_page_rolls_back_before_reusing_connection(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        ({"event_type": "user_message", "prompt": "do it"}, 1),
        ({"event_type": "assistant_message", "assistant_message": "done"}, 2),
    ]

    page = store.read_skill_trace_event_page(
        account_id="acct", evidence_uri="trace:1", after_ordinal=0, limit=1,
    )

    assert page == {
        "events": [{
            "ordinal": 1,
            "event": {"event_type": "user_message", "prompt": "do it"},
        }],
        "next_ordinal": 1,
        "done": False,
    }
    query, params = cursor.execute.call_args.args
    assert "FROM trace_session_events" in query
    assert "jsonb_array_elements" not in query
    assert params == ("acct", "trace:1", 0, 2)
    conn.rollback.assert_called_once()


@pytest.mark.parametrize("write_batch_id", ["", "batch-A"])
def test_read_skill_trace_event_page_backfills_legacy_content_once(store_and_db, write_batch_id):
    store, conn, cursor = store_and_db
    legacy_content = [{"event_type": "user_message", "prompt": "do it", "write_batch_id": write_batch_id}]
    cursor.fetchone.side_effect = [
        None,
        (legacy_content, "user/alice", "agent", "sess", 12, 9, 1),
        None,
    ]
    cursor.rowcount = 1
    cursor.fetchall.return_value = [
        ({"event_type": "user_message", "prompt": "do it"}, 1),
    ]

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        page = store.read_skill_trace_event_page(
            account_id="acct", evidence_uri="trace:1", after_ordinal=0, limit=1,
        )

    assert page["done"] is True
    lock_query = cursor.execute.call_args_list[1]
    backfill = cursor.execute.call_args_list[3]
    assert "FOR UPDATE" in lock_query.args[0]
    assert "FROM trace_events" in backfill.args[0]
    assert "ROW_NUMBER() OVER (ORDER BY event_index)" in backfill.args[0]
    assert backfill.args[1] == (
        "acct", "trace:1", "acct", "sess", "user/alice", "agent", 9, 12, write_batch_id,
    )
    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()


def test_read_skill_trace_event_page_rolls_back_database_errors(store_and_db):
    store, conn, cursor = store_and_db
    cursor.execute.side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        store.read_skill_trace_event_page(
            account_id="acct", evidence_uri="trace:1", after_ordinal=0, limit=100,
        )

    conn.rollback.assert_called_once()


def test_save_skill_compaction_checkpoint_requires_owned_active_lease(store_and_db):
    store, conn, cursor = store_and_db
    cursor.rowcount = 1

    saved = store.save_skill_compaction_checkpoint(
        account_id="acct", uri="trace:1", worker_id="worker",
        checkpoint={"next_ordinal": 10}, lease_seconds=300,
    )

    query, params = cursor.execute.call_args.args
    assert "c.lease_owner = %s" in query
    assert "c.state = 'processing'" in query
    assert "c.lease_until > NOW()" in query
    assert params[1:] == (300, "acct", "worker", "trace:1")
    assert saved is True
    conn.commit.assert_called_once()


def test_mark_skill_assessed_preserves_staged_publication(store_and_db):
    store, conn, cursor = store_and_db
    cursor.rowcount = 1

    changed = store.mark_skill_assessed(
        account_id="acct", uri="trace:1", worker_id="worker",
        capability_key="review-code", assessment={"summary": "safe"},
    )

    query, params = cursor.execute.call_args.args
    assert "UPDATE skill_candidates" in query
    assert "COALESCE(assessment" in query
    assert "(%s::jsonb - 'staged_publication')" in query
    assert params[0] == "assessed"
    assert changed is True
    conn.commit.assert_called_once()


def test_stage_skill_publication_requires_owned_active_lease(store_and_db):
    store, conn, cursor = store_and_db
    cursor.rowcount = 2

    changed = store.stage_skill_publication(
        account_id="acct", uris=["trace:1", "trace:2"], worker_id="worker",
        publication={"uri": "ctx://skill", "content_hash": "abc"},
    )

    query, params = cursor.execute.call_args.args
    assert "jsonb_build_object" in query
    assert "'staged_publication'" in query
    assert "UPDATE skill_candidates" in query
    assert "c.state = 'processing'" in query
    assert "c.lease_owner = %s" in query
    assert params[0] is None
    assert params[2:] == ("acct", ["trace:1", "trace:2"], "worker")
    assert changed == 2
    conn.commit.assert_called_once()


def test_count_skill_distinct_sessions_unions_history_and_current_lease(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.return_value = (2,)

    count = store.count_skill_distinct_sessions(
        account_id="acct",
        skill_uri="ctx://acct/agents/agent/skills/generated/review-code",
        evidence_uris=["trace:1", "trace:2"],
        worker_id="worker",
    )

    query, params = cursor.execute.call_args.args
    assert "COUNT(DISTINCT e.session_id)" in query
    assert "c.state = 'consumed' AND c.skill_uri = %s" in query
    assert "c.state = 'processing'" in query
    assert "c.lease_owner = %s" in query
    assert params == (
        "acct", "ctx://acct/agents/agent/skills/generated/review-code",
        ["trace:1", "trace:2"], "worker",
    )
    assert count == 2
    conn.rollback.assert_called_once()


def test_count_skill_example_sessions_unions_exact_hash_history_and_current_lease(
    store_and_db,
):
    store, conn, cursor = store_and_db
    cursor.fetchone.side_effect = [(1,), (3,)]
    examples = [
        {
            "example_id": "primary", "example_hash": "a" * 64,
            "evidence_uris": ["trace:1"],
        },
        {
            "example_id": "branch-1", "example_hash": "b" * 64,
            "evidence_uris": ["trace:2"],
        },
    ]

    counts = store.count_skill_example_distinct_sessions(
        account_id="acct",
        skill_uri="ctx://acct/agents/agent/skills/generated/review-code",
        examples=examples,
        worker_id="worker",
    )

    assert counts == {f"primary:{'a' * 64}": 1, f"branch-1:{'b' * 64}": 3}
    assert len(cursor.execute.call_args_list) == 2
    query, params = cursor.execute.call_args_list[0].args
    assert "FROM skill_example_evidence AS mapped" in query
    assert "UNION ALL" in query
    assert "candidate.state = 'processing'" in query
    assert params == (
        "acct", "ctx://acct/agents/agent/skills/generated/review-code",
        "primary", "a" * 64, "acct", ["trace:1"], "worker",
    )
    conn.rollback.assert_called_once()


def test_publication_preserves_per_example_evidence_uris():
    publication = {
        "evidence_uris": ["trace:1", "trace:2"],
        "example_evidence_keys": [
            {
                "example_id": "primary", "example_hash": "a" * 64,
                "evidence_uris": ["trace:1"],
            },
            {
                "example_id": "branch-1", "example_hash": "b" * 64,
                "evidence_uris": ["trace:2"],
            },
        ],
    }

    assert validate_skill_publication(publication)["example_evidence_keys"] == publication[
        "example_evidence_keys"
    ]


def test_publication_rejects_example_evidence_outside_batch():
    with pytest.raises(ValueError, match="example evidence URIs"):
        validate_skill_publication({
            "evidence_uris": ["trace:1"],
            "example_evidence_keys": [{
                "example_id": "primary", "example_hash": "a" * 64,
                "evidence_uris": ["trace:other"],
            }],
        })


def test_mark_skill_terminal_requires_matching_processing_owner(store_and_db):
    store, conn, cursor = store_and_db
    cursor.rowcount = 2

    changed = store.mark_skill_terminal(
        account_id="acct", uris=["trace:1", "trace:2"], worker_id="worker",
        state="consumed", skill_uri="ctx://acct/agents/agent/skills/generated/review-code",
    )

    insert_query, insert_params = cursor.execute.call_args_list[0].args
    assert "INSERT INTO skill_example_evidence" in insert_query
    assert "jsonb_array_elements_text" in insert_query
    assert "evidence.evidence_uri = source.evidence_uri" in insert_query
    assert "ON CONFLICT" in insert_query
    assert insert_params == (
        "ctx://acct/agents/agent/skills/generated/review-code",
        "acct", ["trace:1", "trace:2"], "worker",
    )
    query, params = cursor.execute.call_args_list[1].args
    assert "UPDATE skill_candidates" in query
    assert "c.state = 'processing'" in query
    assert "c.lease_owner = %s" in query
    assert params[-1] == "worker"
    assert changed == 2
    conn.commit.assert_called_once()


def test_retry_skill_evidence_uses_configured_backoff(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.return_value = (3, 0, False)

    state = store.retry_skill_evidence(
        account_id="acct", uris=["trace:1"], worker_id="worker",
        max_attempts=5, backoff_base_seconds=7, backoff_cap_seconds=20,
        error="temporary",
    )

    assert state == "assessed"
    select_query = cursor.execute.call_args_list[0].args[0]
    assert "MAX(c.compile_attempts)" in select_query
    _, params = cursor.execute.call_args_list[1].args
    assert params[0] == "assessed"
    assert params[3] == 20
    conn.commit.assert_called_once()


def test_retry_skill_evidence_holds_at_compile_attempt_limit(store_and_db):
    store, _, cursor = store_and_db
    cursor.fetchone.return_value = (5, 0, False)

    state = store.retry_skill_evidence(
        account_id="acct", uris=["trace:1"], worker_id="worker",
        max_attempts=5, backoff_base_seconds=7, backoff_cap_seconds=20,
        error="persistent",
    )

    assert state == "held"
    _, params = cursor.execute.call_args_list[1].args
    assert params[0] == "held"
    assert params[3] == 20


def test_list_ready_skill_capabilities_groups_assessed_evidence(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchall.return_value = [("agent", "review-code", 6, 3)]

    rows = store.list_ready_skill_capabilities(
        account_id="acct", min_evidence_turns=5,
        min_distinct_sessions=3, limit=20,
    )

    query, params = cursor.execute.call_args.args
    assert "FROM skill_candidates c" in query
    assert "c.state = 'assessed'" in query
    assert "staged_publication" in query
    assert "GROUP BY c.agent_id, c.capability_key" in query
    assert "COUNT(*) >= %s" in query
    assert "BOOL_OR" in query
    assert "eligibility_state" in query
    assert "COUNT(*) >= 2" in query
    assert params == ("acct", 5, 3, 20)
    assert rows == [{
        "agent_id": "agent",
        "capability_key": "review-code",
        "evidence_count": 6,
        "distinct_session_count": 3,
    }]
    conn.rollback.assert_called_once()

def test_list_skill_candidates_rolls_back_read_transaction(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchall.return_value = []

    assert store.list_skill_candidates(account_id="acct", agent_id="agent") == []

    conn.rollback.assert_called_once()


def test_scheduler_backlog_count_rolls_back_read_transaction(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.return_value = (7,)

    assert store.count_skill_work_accounts() == 7

    conn.rollback.assert_called_once()


@pytest.mark.parametrize("invalid", [1, 1025])
def test_rejects_invalid_turn_projection_configuration(invalid):
    with pytest.raises(ValueError, match="between 2 and 1024"):
        SQLTraceEventStore(connection_string="unused", max_turn_events=invalid)
