"""Persisted v1 operations replay and independently upgrade during revision."""

import hashlib
from dataclasses import asdict, replace

from commit.write_receipts import RECEIPTS_KEY, operation_digest
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_request_outbox_events import make_runtime, request_context
from tests.unit.server.test_extraction_batch_recovery import NoModel


SLOT = "chunk:0"
ARCHIVE_ID = "legacy-archive"


def legacy_checkpoint(fs, ctx, candidates):
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager.add_message(ctx.session_id, "user", "The receiver and antenna were repaired.", ctx,
                        message_id="source", created_at="2026-10-27T14:00:00+00:00")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(
        manager, buffer, ctx, list(buffer.messages), archive_id=ARCHIVE_ID,
        messages=[{"role": "user", "content": buffer.messages[0].content}],
        session_time="2026-10-27T14:00:00+00:00", session_summary="", tool_stats_text="",
        client_request_id="legacy-request",
    )
    accepted = [replace(candidate, write_operation_id=hashlib.sha256(
        f"{ARCHIVE_ID}:{SLOT}:{index}".encode()).hexdigest()) for index, candidate in enumerate(candidates)]
    record = {
        "candidates": [asdict(candidate) for candidate in accepted],
        "candidate_digests": [operation_digest(candidate, ctx) for candidate in accepted],
        "extracted": len(accepted), "filtered": 0, "operations": {},
    }
    journal.mutate(lambda batch: batch["slots"].update({SLOT: record}))
    return journal.checkpoint(SLOT), accepted


def restart_checkpoint(client, ctx):
    fs = AGFSContextFS(client)
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    manager.enable_extraction_recovery(ctx.session_id, ctx, fs)
    journal = ExtractionBatchJournal.resume(manager, buffer, ctx)
    return fs, journal.checkpoint(SLOT)


def candidate(key):
    return replace(memory(key, content=f"The {key} was repaired."),
                   provenance_ids=[f"prov:1:archive:{ARCHIVE_ID}:source"])


def test_persisted_v1_checkpoint_replays_source_receipt_and_loads_completed_plan():
    client, fs, _ = make_runtime()
    ctx = request_context("legacy-checkpoint")
    checkpoint, accepted = legacy_checkpoint(fs, ctx, [candidate("receiver")])
    assert set(checkpoint.record) == {
        "candidates", "candidate_digests", "extracted", "filtered", "operations"}
    checkpoint.begin(0)
    created = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(accepted[0], ctx)

    restored_fs, restored = restart_checkpoint(client, ctx)
    loaded, extracted, filtered = restored.load()
    assert (loaded, extracted, filtered) == (accepted, 1, 0)
    plans = MemoryWriteAPI(restored_fs, NoModel(), directory_summary_enabled=False)._write_checkpointed_candidates(
        loaded, ctx, restored)
    assert plans[0].target_uri == created["target_uri"]
    assert plans[0].merged_fields["status"] == "operation_committed"
    source = restored_fs.read_node(created["target_uri"], ctx)
    assert source.metadata["version"] == 1
    assert source.metadata[RECEIPTS_KEY][accepted[0].write_operation_id] == operation_digest(accepted[0], ctx)

    final_fs, final = restart_checkpoint(client, ctx)
    assert final.load() == (accepted, 1, 0)
    assert final.completed(0)["target_uri"] == source.uri
    assert final.record["operations"]["0"]["status"] == "complete"
    replay = MemoryWriteAPI(final_fs, NoModel(), directory_summary_enabled=False)._write_checkpointed_candidates(
        final.load()[0], ctx, final)
    assert replay == plans
    assert final_fs.read_node(source.uri, ctx).metadata["version"] == 1


def test_rejected_v1_operation_revises_to_v2_alongside_legacy_receipt_replay():
    client, fs, _ = make_runtime()
    ctx = request_context("revised-checkpoint")
    checkpoint, accepted = legacy_checkpoint(fs, ctx, [candidate("receiver"), candidate("antenna")])
    checkpoint.reject(0, "Refresh the prepared target before writing")
    checkpoint.begin(1)
    original = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(accepted[1], ctx)

    _, restored = restart_checkpoint(client, ctx)
    assert restored.load() == (accepted, 2, 0)
    revised = restored.revise_unwritten(0, replace(accepted[0],
        abstract="The receiver was repaired and tested.", overview="The receiver was repaired and tested.",
        content="The receiver was repaired and tested."))
    expected_id = "receipt-v2:" + hashlib.sha256(f"{ARCHIVE_ID}:{SLOT}:0:revision:1".encode()).hexdigest()
    assert revised.write_operation_id == expected_id
    assert restored.record["operation_id_versions"] == {"0": 2}
    assert restored.record["revision_history"][0]["candidate"] == asdict(accepted[0])
    assert restored.record["revision_history"][0]["operation"]["status"] == "rejected_before_write"

    resumed_fs, resumed = restart_checkpoint(client, ctx)
    loaded, extracted, filtered = resumed.load()
    assert (loaded, extracted, filtered) == ([revised, accepted[1]], 2, 0)
    plans = MemoryWriteAPI(resumed_fs, NoModel(), directory_summary_enabled=False)._write_checkpointed_candidates(
        loaded, ctx, resumed)
    assert [plan.action for plan in plans] == ["create", "skip"]
    assert plans[1].target_uri == original["target_uri"]
    assert plans[1].merged_fields["status"] == "operation_committed"
    for entry, plan in zip(loaded, plans):
        source = resumed_fs.read_node(plan.target_uri, ctx)
        assert source.metadata["version"] == 1
        assert source.metadata[RECEIPTS_KEY][entry.write_operation_id] == operation_digest(entry, ctx)

    _, completed = restart_checkpoint(client, ctx)
    assert completed.load() == ([revised, accepted[1]], 2, 0)
    assert [completed.completed(index)["target_uri"] for index in range(2)] == [plan.target_uri for plan in plans]
    assert [completed.record["operations"][str(index)]["status"] for index in range(2)] == ["complete", "complete"]
