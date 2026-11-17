"""Accumulate partial extraction acknowledgements across archived source batches."""

from copy import deepcopy

import pytest

from core.write_contract import merge_source_outcome


def receipt(archive_id, writes, extracted, *, failed=False):
    result = {"archive_id": archive_id, "source_sha256": f"sha256:{archive_id}",
              "source_status": "ARCHIVED", "semantic_status": "DEGRADED" if failed else "COMPLETE",
              "writes_completed": writes, "candidates_extracted": extracted,
              "outbox_tracking_complete": True, "outbox_events": []}
    if failed:
        result.update(recovery_mode="partial" if writes else "source_only", failed_nodes=[{
            "slot": "chunk:0", "target": "ctx://account/users/alice/memories/entities/melanie",
            "draft_ids": [1], "error": "L2 edit search must occur once in target content"}])
    return result


@pytest.mark.parametrize("replay", [False, True], ids=["new-batches", "replayed-batches"])
def test_partial_then_complete_preserves_failure_and_counts_each_archive_once(replay):
    pending = {"source_status": "BUFFERED", "source_message_ids": ["message-a", "message-b"]}
    first = receipt("archive-a", 2, 3, failed=True)
    second = receipt("archive-b", 4, 4)
    preserved = deepcopy((pending, first, second))
    result = merge_source_outcome(pending, first, ["message-a"])
    assert result["writes_completed"] == 2
    assert result["source_status"] == "BUFFERED"
    if replay:
        result = merge_source_outcome(result, first, ["message-a"])
    result = merge_source_outcome(result, second, ["message-b"])
    if replay:
        result = merge_source_outcome(result, first, ["message-a"])
        result = merge_source_outcome(result, second, ["message-b"])
    assert result["writes_completed"] == 6
    assert result["candidates_extracted"] == 7
    assert result["semantic_status"] == result["request_status"] == "DEGRADED"
    assert result["recovery_mode"] == "partial"
    assert result["archived_source_ids"] == ["message-a", "message-b"]
    assert result["archive_receipts"] == {"archive-a": "sha256:archive-a", "archive-b": "sha256:archive-b"}
    assert result["failed_nodes"] == [{"archive_id": "archive-a", **first["failed_nodes"][0]}]
    assert (pending, first, second) == preserved


def test_each_partial_archive_retains_its_own_failed_draft_reference():
    pending = {"source_message_ids": ["message-a", "message-b"]}
    first = receipt("archive-a", 2, 3, failed=True)
    second = receipt("archive-b", 1, 2, failed=True)
    result = merge_source_outcome(pending, first, ["message-a"])
    result = merge_source_outcome(result, second, ["message-b"])
    result = merge_source_outcome(result, first, ["message-a"])
    assert result["writes_completed"] == 3
    assert result["candidates_extracted"] == 5
    assert [(failure["archive_id"], failure["draft_ids"]) for failure in result["failed_nodes"]] == [
        ("archive-a", [1]), ("archive-b", [1])]


def test_later_successful_nodes_complete_a_partially_recovered_request():
    pending = {"source_message_ids": ["message-a", "message-b"]}
    result = merge_source_outcome(pending, receipt("archive-a", 0, 1, failed=True), ["message-a"])
    assert result["recovery_mode"] == "source_only"
    result = merge_source_outcome(result, receipt("archive-b", 4, 4), ["message-b"])
    assert result["recovery_mode"] == "partial"
    assert result["writes_completed"] == 4
    assert result["candidates_extracted"] == 5
    assert result["failed_nodes"][0]["archive_id"] == "archive-a"
