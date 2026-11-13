"""Additional tests for OutboxStore edge cases."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from commit.outbox_store import OutboxStore
from core.models import ContextNode, OutboxEvent, RequestContext


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="u1",
        agent_id="a1",
        session_id="session",
        trace_id="trace",
    )


@pytest.fixture
def node() -> ContextNode:
    return ContextNode(
        uri="ctx://acct-1/users/u1/memories/preferences/coffee",
        context_type="MEMORY",
        category="preference",
        level=0,
        owner_space="user:u1",
        abstract="likes coffee",
        overview="overview",
        content="content",
    )


@pytest.fixture
def store() -> OutboxStore:
    return OutboxStore(Mock(), Mock(), mount_prefix="/local")


def test_collect_sibling_abstracts_truncates_to_first_twenty_and_ignores_errors(store, ctx):
    store._fs.list_children.return_value = [f"ctx://acct-1/users/u1/memories/preferences/item-{i}" for i in range(25)]

    def read_node(uri, _ctx):
        if uri.endswith("item-5"):
            raise RuntimeError("boom")
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:u1",
            abstract="x" * 250,
            overview="",
            content="",
        )

    store._fs.read_node.side_effect = read_node

    abstracts = store._collect_sibling_abstracts("ctx://acct-1/users/u1/memories/preferences/coffee", ctx)

    assert len(abstracts) == 19
    assert all(len(item) == 200 for item in abstracts)


def test_collect_sibling_abstracts_returns_empty_when_listing_fails(store, ctx):
    store._fs.list_children.side_effect = RuntimeError("boom")

    assert store._collect_sibling_abstracts("ctx://acct-1/users/u1/memories/preferences/coffee", ctx) == []


def test_register_directory_writes_event_to_parent_outbox_even_if_mkdir_fails(store, node, ctx):
    store._client.mkdir.side_effect = RuntimeError("exists")
    store._collect_sibling_abstracts = Mock(return_value=["one", "two"])

    event = store.register_directory(node, ctx)

    assert event.event_type == "UPSERT_DIRECTORY"
    assert event.uri == "ctx://acct-1/users/u1/memories/preferences/"
    written_path, written_bytes = store._client.write.call_args.args
    assert written_path.endswith("/accounts/acct-1/users/u1/memories/preferences/.outbox/" + event.event_id + ".json")
    payload = json.loads(written_bytes.decode("utf-8"))
    assert payload["payload"]["child_abstracts"] == ["one", "two"]


def test_write_event_publishes_staged_event_atomically(store):
    event = OutboxEvent(
        event_id="evt-1",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )

    store._write_event(event, event.uri)

    temp_path, event_bytes = store._client.write.call_args.args
    assert temp_path.endswith(".tmp_evt-1.json")
    final_path = temp_path.replace(".tmp_evt-1.json", "evt-1.json")
    store._client.mv.assert_called_once_with(temp_path, final_path)
    assert json.loads(event_bytes)["event_id"] == event.event_id


def test_write_event_reraises_when_temp_write_fails(store):
    store._client.write.side_effect = RuntimeError("disk full")
    event = OutboxEvent(
        event_id="evt-1",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )

    with pytest.raises(RuntimeError, match="disk full"):
        store._write_event(event, event.uri)

    store._client.rename.assert_not_called()


def test_mark_processing_writes_processing_status(store):
    event = OutboxEvent(
        event_id="evt-2",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )

    store.mark_processing(event, event.uri)

    _, raw = store._client.write.call_args.args
    payload = json.loads(raw.decode("utf-8"))
    assert payload["status"] == "PROCESSING"


def test_mark_done_ignores_delete_errors(store):
    event = OutboxEvent(
        event_id="evt-3",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )
    store._client.rm.side_effect = FileNotFoundError("No such file")

    store.mark_done(event, event.uri)


def test_collect_outbox_events_skips_future_retry_and_tolerates_invalid_lock(store):
    future_time = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    event_json = json.dumps(
        {
            "event_id": "evt-4",
            "event_type": "UPSERT_CONTEXT",
            "uri": "ctx://acct-1/users/u1/memories/profile",
            "payload": {"records": []},
            "status": "PENDING",
            "retry_count": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "next_retry_at": future_time,
        }
    ).encode("utf-8")
    invalid_lock = b"{bad json"

    def read_side_effect(path):
        if path.endswith(".processing"):
            return invalid_lock
        return event_json

    store._client.read.side_effect = read_side_effect
    results = []

    store._collect_outbox_events("/local/accounts/acct-1/.outbox", [{"name": "evt-4.json", "is_dir": False}], results)

    assert results == []


def test_collect_outbox_events_includes_event_when_next_retry_is_invalid(store):
    event_json = json.dumps(
        {
            "event_id": "evt-5",
            "event_type": "UPSERT_CONTEXT",
            "uri": "ctx://acct-1/users/u1/memories/profile",
            "payload": {"records": []},
            "status": "PENDING",
            "retry_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
            "next_retry_at": "not-a-date",
        }
    ).encode("utf-8")
    store._client.read.side_effect = lambda path: (_ for _ in ()).throw(FileNotFoundError(path)) if path.endswith(".processing") else event_json
    results = []

    store._collect_outbox_events("/local/accounts/acct-1/.outbox", [{"name": "evt-5.json", "is_dir": False}], results)

    assert len(results) == 1
    assert results[0][1].event_id == "evt-5"


def test_move_to_dlq_marks_failed_and_ignores_source_delete_errors(store):
    store._client.rm.side_effect = RuntimeError("already removed")
    event = OutboxEvent(
        event_id="evt-6",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )

    store.move_to_dlq(event, event.uri)

    path, raw = store._client.write.call_args.args
    payload = json.loads(raw.decode("utf-8"))
    assert path.endswith("/.outbox/dlq/evt-6.json")
    assert payload["status"] == "FAILED"


def test_increment_retry_updates_counter_and_next_retry_time(store):
    event = OutboxEvent(
        event_id="evt-7",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PROCESSING",
        retry_count=1,
    )
    next_retry_at = datetime.now(UTC) + timedelta(seconds=30)

    store.increment_retry(event, event.uri, next_retry_at)

    _, raw = store._client.write.call_args.args
    payload = json.loads(raw.decode("utf-8"))
    assert payload["retry_count"] == 2
    assert payload["status"] == "PENDING"
    assert payload["next_retry_at"] == next_retry_at.isoformat()
