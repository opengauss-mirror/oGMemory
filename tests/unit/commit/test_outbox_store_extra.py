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


def test_write_event_fails_closed_when_atomic_move_fails(store):
    store._client.mv.side_effect = RuntimeError("move failed")
    store._client.read.side_effect = RuntimeError("No such file")
    event = OutboxEvent(
        event_id="evt-1",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )

    with pytest.raises(RuntimeError, match="move failed"):
        store._write_event(event, event.uri)

    assert store._client.write.call_count == 1
    temp_path = store._client.write.call_args_list[0].args[0]
    assert temp_path.endswith(".tmp_evt-1.json")
    store._client.rm.assert_called_once_with(temp_path)


def test_write_event_accepts_lost_move_response_when_exact_event_is_visible(store):
    event = OutboxEvent(
        event_id="evt-committed",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )
    staged: dict[str, bytes] = {}
    committed: dict[str, bytes] = {}

    def write(path, raw):
        staged[path] = bytes(raw)

    def move(source, destination):
        committed[destination] = staged.pop(source)
        raise RuntimeError("lost event mv response")

    def read(path):
        try:
            return committed[path]
        except KeyError as exc:
            raise RuntimeError("No such file") from exc

    store._client.write.side_effect = write
    store._client.mv.side_effect = move
    store._client.read.side_effect = read

    store.persist_event(event, event.uri)

    event_path = next(path for path in committed if path.endswith("evt-committed.json"))
    assert json.loads(committed[event_path].decode("utf-8"))["event_id"] == event.event_id


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

    store._client.mv.assert_not_called()


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


def test_cancel_pending_upserts_removes_event_and_claim_lock(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-upsert",
        event_type="UPSERT_CONTEXT",
        uri=uri,
        payload={"records": []},
        status="PROCESSING",
    )
    store._client.ls.return_value = [
        {"name": "evt-upsert.json", "is_dir": False},
        {"name": "evt-upsert.processing", "is_dir": False},
    ]
    store._client.read.return_value = json.dumps(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
        }
    )

    assert store.cancel_pending_upserts(uri) == ["evt-upsert"]

    removed = {call.args[0] for call in store._client.rm.call_args_list}
    assert any(path.endswith("evt-upsert.json") for path in removed)
    assert any(path.endswith("evt-upsert.processing") for path in removed)


def test_durable_delete_cancels_upserts_before_processing_lock(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": [], "durable_delete": True},
        status="PENDING",
    )
    operations = []
    store.cancel_pending_upserts = Mock(
        side_effect=lambda node_uri: operations.append(("cancel", node_uri)) or []
    )
    store._client.read.side_effect = FileNotFoundError("Not found")
    store._client.write.side_effect = (
        lambda path, _payload: operations.append(("lock", path))
    )

    assert store.try_acquire(event, uri, "worker") is True

    assert operations[0] == ("cancel", uri)
    assert operations[1][0] == "lock"


def test_agfs_liveness_rejects_upsert_when_delete_tombstone_exists(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event_json = json.dumps(
        {
            "event_id": "evt-upsert",
            "event_type": "UPSERT_CONTEXT",
            "uri": uri,
            "payload": {"records": []},
            "status": "PROCESSING",
        }
    ).encode("utf-8")
    tombstone_json = json.dumps(
        {
            "kind": "DELETE_TOMBSTONE",
            "uri": uri,
            "status": "PENDING",
        }
    ).encode("utf-8")
    store._client.read.side_effect = [event_json, tombstone_json]

    assert store.is_event_current("evt-upsert", uri) is False


def test_delete_mark_done_keeps_live_event_when_payload_exists(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    store._client.read.return_value = b"payload"

    with pytest.raises(
        RuntimeError,
        match="cannot acknowledge DELETE_CONTEXT while payload exists",
    ):
        store.mark_done(event, uri)

    store._client.rm.assert_not_called()
    store._client.write.assert_not_called()
    store._client.mv.assert_not_called()


def test_delete_mark_done_keeps_live_event_when_payload_verification_fails(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    store._client.read.side_effect = RuntimeError("AGFS unavailable")

    with pytest.raises(RuntimeError, match="AGFS unavailable"):
        store.mark_done(event, uri)

    store._client.rm.assert_not_called()
    store._client.write.assert_not_called()
    store._client.mv.assert_not_called()


def test_delete_mark_done_publishes_ack_before_removing_event_and_marker_last(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    processing_path = node_path + ".outbox/evt-delete.processing"
    outbox_path = node_path + ".outbox"

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        raise RuntimeError("No such file")

    store._client.read.side_effect = read
    store._client.ls.return_value = []

    store.mark_done(event, uri)

    temp_path, raw = store._client.write.call_args_list[0].args
    acknowledging = json.loads(raw.decode("utf-8"))
    assert acknowledging == {**marker, "status": "ACKNOWLEDGING"}
    store._client.mv.assert_any_call(temp_path, tombstone_path)
    receipt_temp, receipt_raw = store._client.write.call_args_list[1].args
    assert json.loads(receipt_raw)["status"] == "DONE"
    store._client.mv.assert_any_call(receipt_temp, store._done_receipt_path(uri, event.event_id))

    calls = store._client.mock_calls

    def call_index(method, path):
        return next(
            index
            for index, item in enumerate(calls)
            if item[0] == method and item.args and item.args[0] == path
        )

    assert call_index("mv", temp_path) < call_index("rm", event_path)
    assert call_index("rm", event_path) < call_index("rm", processing_path)
    assert call_index("rm", processing_path) < call_index("rm", outbox_path)
    assert call_index("rm", outbox_path) < call_index("rm", tombstone_path)
    assert not any(
        item[0] == "rm"
        and item.args
        and item.args[0] == node_path
        and item.kwargs.get("recursive") is True
        for item in calls
    )


def test_delete_mark_done_continues_when_ambiguous_event_remove_is_absent(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    pending_temp: dict[str, bytes] = {}
    event_remove_attempts = 0

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        raise RuntimeError("No such file")

    def write(path, raw):
        pending_temp[path] = raw

    def move(from_path, to_path):
        if to_path == store._done_receipt_path(uri, event.event_id):
            assert json.loads(pending_temp[from_path])["status"] == "DONE"
            return
        assert to_path == tombstone_path
        marker.update(json.loads(pending_temp[from_path].decode("utf-8")))

    def remove(path, recursive=False):
        nonlocal event_remove_attempts
        if path == event_path:
            event_remove_attempts += 1
            if event_remove_attempts == 1:
                raise RuntimeError("ambiguous event remove")
        if path == tombstone_path:
            marker.clear()

    store._client.read.side_effect = read
    store._client.write.side_effect = write
    store._client.mv.side_effect = move
    store._client.rm.side_effect = remove
    store._client.ls.return_value = []

    store.mark_done(event, uri)

    assert event_remove_attempts == 1
    assert marker == {}
    assert store._client.write.call_count == 2


def test_delete_mark_done_raises_when_ambiguous_event_remove_is_still_live(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    pending_temp: dict[str, bytes] = {}

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        if path == event_path:
            return b'{"status":"PROCESSING"}'
        raise RuntimeError("No such file")

    def write(path, raw):
        pending_temp[path] = raw

    def move(from_path, to_path):
        if to_path == store._done_receipt_path(uri, event.event_id):
            assert json.loads(pending_temp[from_path])["status"] == "DONE"
            return
        assert to_path == tombstone_path
        marker.update(json.loads(pending_temp[from_path].decode("utf-8")))

    def remove(path, recursive=False):
        if path == event_path:
            raise RuntimeError("ambiguous event remove")

    store._client.read.side_effect = read
    store._client.write.side_effect = write
    store._client.mv.side_effect = move
    store._client.rm.side_effect = remove

    with pytest.raises(RuntimeError, match="ambiguous event remove"):
        store.mark_done(event, uri)

    assert marker["status"] == "ACKNOWLEDGING"
    assert store._client.write.call_count == 2
    assert not any(
        item.args[0] == tombstone_path
        for item in store._client.rm.call_args_list
    )


def test_delete_mark_done_retains_ack_when_event_remove_outcome_is_unknown(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    processing_path = node_path + ".outbox/evt-delete.processing"
    pending_temp: dict[str, bytes] = {}

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        if path == event_path:
            raise RuntimeError("event verification unavailable")
        raise RuntimeError("No such file")

    def write(path, raw):
        pending_temp[path] = raw

    def move(from_path, to_path):
        if to_path == store._done_receipt_path(uri, event.event_id):
            assert json.loads(pending_temp[from_path])["status"] == "DONE"
            return
        assert to_path == tombstone_path
        marker.update(json.loads(pending_temp[from_path].decode("utf-8")))

    def remove(path, recursive=False):
        if path == event_path:
            raise RuntimeError("ambiguous event remove")

    store._client.read.side_effect = read
    store._client.write.side_effect = write
    store._client.mv.side_effect = move
    store._client.rm.side_effect = remove

    store.mark_done(event, uri)

    assert marker["status"] == "ACKNOWLEDGING"
    removed = [item.args[0] for item in store._client.rm.call_args_list]
    assert [path for path in removed if not path.endswith(".tmp")] == [event_path]
    assert processing_path not in removed
    assert tombstone_path not in removed


@pytest.mark.parametrize("failure_point", ["processing", "outbox"])
def test_delete_mark_done_cleanup_failure_retains_ack_marker(store, failure_point):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    processing_path = node_path + ".outbox/evt-delete.processing"
    pending_temp: dict[str, bytes] = {}

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        raise RuntimeError("No such file")

    def write(path, raw):
        pending_temp[path] = raw

    def move(from_path, to_path):
        if to_path == store._done_receipt_path(uri, event.event_id):
            assert json.loads(pending_temp[from_path])["status"] == "DONE"
            return
        assert to_path == tombstone_path
        marker.update(json.loads(pending_temp[from_path].decode("utf-8")))

    def remove(path, recursive=False):
        if failure_point == "processing" and path == processing_path:
            raise RuntimeError("processing cleanup failed")

    store._client.read.side_effect = read
    store._client.write.side_effect = write
    store._client.mv.side_effect = move
    store._client.rm.side_effect = remove
    if failure_point == "outbox":
        store._client.ls.side_effect = RuntimeError("Outbox cleanup failed")
    else:
        store._client.ls.return_value = []

    store.mark_done(event, uri)

    assert marker["status"] == "ACKNOWLEDGING"
    removed = [item.args[0] for item in store._client.rm.call_args_list]
    assert event_path in removed
    assert tombstone_path not in removed


def test_delete_mark_done_keeps_event_when_ack_marker_publish_fails(store):
    uri = "ctx://acct-1/resources/chunk-1"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    marker = {
        "kind": "DELETE_TOMBSTONE",
        "uri": uri,
        "status": "PENDING",
        "created_at": "2026-08-25T08:00:00+00:00",
        "delete_event_id": event.event_id,
        "record_ids": ["l0", "l1", "l2"],
    }
    node_path = store._uri_to_agfs_path(uri)
    tombstone_path = node_path + ".delete_tombstone.json"
    event_path = node_path + ".outbox/evt-delete.json"
    processing_path = node_path + ".outbox/evt-delete.processing"

    def read(path):
        if path == tombstone_path:
            return json.dumps(marker).encode("utf-8")
        raise RuntimeError("No such file")

    store._client.read.side_effect = read
    store._client.mv.side_effect = RuntimeError("marker publish failed")

    with pytest.raises(RuntimeError, match="marker publish failed"):
        store.mark_done(event, uri)

    removed = [item.args[0] for item in store._client.rm.call_args_list]
    assert event_path not in removed
    assert processing_path not in removed
    assert tombstone_path not in removed
    assert len(removed) == 1
    assert removed[0].startswith(tombstone_path + ".tmp-")


def test_delete_mark_done_without_tombstone_keeps_generic_delete_working(store):
    uri = "ctx://acct-1/users/u1/memories/preferences/coffee"
    event = OutboxEvent(
        event_id="evt-delete",
        event_type="DELETE_CONTEXT",
        uri=uri,
        payload={"ids_to_delete": ["l0", "l1", "l2"]},
        status="PROCESSING",
    )
    store._client.read.side_effect = RuntimeError("No such file")
    store._client.ls.return_value = []

    store.mark_done(event, uri)

    removed = {item.args[0] for item in store._client.rm.call_args_list}
    assert any(path.endswith("evt-delete.json") for path in removed)
    assert any(path.endswith("evt-delete.processing") for path in removed)


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


def test_recursive_scans_hide_only_atomic_resource_staging(store):
    account_root = "/local/accounts/acct-1/"
    hidden_node = ".retained-resource"
    staging_node = ".chunk.resource-create-" + "a" * 32
    hidden_path = account_root.rstrip("/") + "/" + hidden_node + "/"
    staging_path = account_root.rstrip("/") + "/" + staging_node + "/"
    outbox_path = hidden_path.rstrip("/") + "/.outbox"
    event = OutboxEvent(
        event_id="evt-hidden",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/resources/retained-resource",
        payload={"records": []},
        status="PENDING",
    )
    event_raw = json.dumps(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
            "next_retry_at": event.next_retry_at,
        }
    ).encode("utf-8")
    visited: list[str] = []

    def list_directory(path):
        visited.append(path)
        if path == account_root:
            return [
                {"name": hidden_node, "is_dir": True},
                {"name": staging_node, "is_dir": True},
            ]
        if path == hidden_path:
            return [{"name": ".outbox", "is_dir": True}]
        if path in {outbox_path, outbox_path + "/"}:
            return [{"name": event.event_id + ".json", "is_dir": False}]
        raise RuntimeError(f"unexpected scan path: {path}")

    def read(path):
        if path.endswith(".processing"):
            raise RuntimeError("No such file")
        if path == outbox_path + "/" + event.event_id + ".json":
            return event_raw
        raise RuntimeError(f"unexpected read path: {path}")

    store._client.ls.side_effect = list_directory
    store._client.read.side_effect = read

    pending = store.list_pending("acct-1")

    assert [(uri, item.event_id) for uri, item in pending] == [
        (event.uri, event.event_id)
    ]
    assert hidden_path in visited
    assert staging_path not in visited

    visited.clear()
    assert store.count_unfinished("acct-1") == {
        "pending": 1,
        "processing": 0,
        "total": 1,
    }
    assert hidden_path in visited
    assert staging_path not in visited


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
