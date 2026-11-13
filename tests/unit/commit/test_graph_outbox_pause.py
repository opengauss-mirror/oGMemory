"""Paused graph wakeups do not impose predecessor barriers on SQL indexing."""
import pytest

from tests.unit.commit.test_sql_outbox_store import mock_sql_outbox_store as mock_sql_outbox_store


@pytest.mark.parametrize("excluded,paused", [
    (None, []), (["EXTRACT_AGENT_MEMORY"], []),
    (["PROJECT_GRAPH"], ["PROJECT_GRAPH"]),
    (["EXTRACT_AGENT_MEMORY", "PROJECT_GRAPH"], ["PROJECT_GRAPH"]),
])
def test_claim_filter_preserves_explicit_feature_pauses(mock_sql_outbox_store, excluded, paused):
    store, _, _, cursor = mock_sql_outbox_store
    cursor.fetchall.return_value = []
    assert store.claim_batch("worker", limit=7, exclude_event_types=excluded) == []
    sql, params = cursor.execute.call_args.args
    assert "candidate.event_type <> ALL(%s)" in sql
    assert "earlier.event_type <> ALL(%s)" in sql
    assert "earlier.account_id = candidate.account_id" in sql
    assert "earlier.uri = candidate.uri" in sql
    assert "earlier.seq < candidate.seq" in sql
    assert params == (300, [], excluded or [], paused, 7)


def test_counter_exclusion_is_explicit_and_does_not_change_default(mock_sql_outbox_store):
    store, _, _, cursor = mock_sql_outbox_store
    cursor.fetchone.return_value = (0, 0, 0)
    assert store.count_unfinished("account", exclude_event_types=["PROJECT_GRAPH"])["total"] == 0
    sql, params = cursor.execute.call_args.args
    assert "event_type <> ALL(%s)" in sql
    assert params == ("account", ["PROJECT_GRAPH"])
    store.count_unfinished("account")
    sql, params = cursor.execute.call_args.args
    assert "event_type <> ALL(%s)" not in sql
    assert params == ("account",)
