from dataclasses import asdict
from datetime import UTC, datetime

import pytest

from core.write_contract import OUTCOMES_KEY
from service.extraction_checkpoint import ExtractionBatchJournal, request_message_id
from service.extraction_receipts import compact_plans, read
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import candidate, params, setup_service
from tests.unit.server.test_extraction_batch_recovery import ctx as ctx

pytestmark = pytest.mark.usefixtures("service_threading")


def stage_completed_writes(service, manager, writer, ctx):
    manager.get_or_create(ctx.session_id, ctx)
    manager.enable_extraction_recovery(ctx.session_id, ctx, writer._fs)
    source = params(ctx)
    manager.add_message(
        ctx.session_id,
        "user",
        source["messages"][0]["content"],
        ctx,
        message_id=request_message_id(ctx, "original", 0),
    )
    buffer = manager.get_or_create(ctx.session_id, ctx)
    messages = [
        {"id": message.id, "role": message.role, "content": message.content}
        for message in buffer.messages
    ]
    journal = ExtractionBatchJournal(
        manager,
        buffer,
        ctx,
        list(buffer.messages),
        archive_id="staged-original",
        messages=messages,
        session_time=datetime.now(UTC),
        session_summary="",
        tool_stats_text="",
        client_request_id="original",
        request_messages=source["messages"],
    )
    service._chunk_and_extract(
        messages,
        writer,
        ctx,
        datetime.now(UTC),
        "",
        "",
        "staged",
        ctx.session_id,
        extraction_journal=journal,
    )
    return buffer


def test_forced_capture_finishes_its_own_source_after_saved_backlog(ctx):
    entries = iter([candidate(), candidate("entity", body="I own the blue release.")])
    service, writer, manager = setup_service(
        MemoryFS(), ctx, extract=lambda **kwargs: [next(entries)]
    )
    buffer = stage_completed_writes(service, manager, writer, ctx)
    request = {
        **params(ctx, request="new-turn", content="I own the blue release."),
        "forceExtract": True,
    }

    result = service.after_turn(request)

    assert result["status"] == "completed"
    assert result["client_request_id"] == "new-turn"
    assert len(result["plans"]) == 1
    assert "/entities/" in result["plans"][0]["target_uri"]
    assert [plan["target_uri"] for plan in read(buffer, ctx, "new-turn")["plans"]] == [
        result["plans"][0]["target_uri"]
    ]
    assert buffer.request_completed("original") and buffer.request_completed("new-turn")
    assert buffer.messages == []
    assert buffer.extraction_active_count == 0
    assert writer._pipeline.extract.call_count == 2


def test_accepted_source_can_continue_with_same_id_when_extraction_is_requested(ctx):
    service, writer, manager = setup_service(MemoryFS(), ctx)
    service._cfg.after_turn_threshold = 10000
    request = params(ctx)
    accepted = service.after_turn(request)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    receipt = read(buffer, ctx, "original", include_accepted=True)
    assert accepted["status"] == "accumulating"
    assert receipt["status"] == "accepted"
    assert receipt["plans"] == []
    assert receipt["message_ids"] == [request_message_id(ctx, "original", 0)]
    assert buffer.runtime_extensions[OUTCOMES_KEY]["entries"]["original"]["source_status"] == "BUFFERED"
    assert service.after_turn(request)["status"] == "duplicate"

    result = service.after_turn({**request, "forceExtract": True})

    assert result["status"] == "completed"
    assert read(buffer, ctx, "original")["status"] == "completed"
    assert buffer.messages == []
    assert writer._pipeline.extract.call_count == 1


def test_forced_capture_continues_after_assistant_archive(ctx):
    service, writer, manager = setup_service(MemoryFS(), ctx)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    manager.enable_extraction_recovery(ctx.session_id, ctx, writer._fs)
    manager.add_message(ctx.session_id, "assistant", "Previous response.", ctx, message_id="reply")
    ExtractionBatchJournal(
        manager,
        buffer,
        ctx,
        list(buffer.messages),
        archive_id="assistant-archive",
        messages=[],
        session_time=datetime.now(UTC),
        session_summary="",
        tool_stats_text="",
        client_request_id="previous",
        request_messages=[{"role": "assistant", "content": "Previous response."}],
    )

    result = service.after_turn({**params(ctx), "forceExtract": True})

    assert result["status"] == "completed"
    assert read(buffer, ctx, "original")["status"] == "completed"
    assert buffer.messages == []
    assert buffer.extraction_active_count == 0
    assert writer._pipeline.extract.call_count == 1


def test_accepted_ingest_handoff_keeps_its_saved_source_id(ctx):
    service, writer, manager = setup_service(MemoryFS(), ctx)
    service._cfg.after_turn_threshold = 10000
    request = params(ctx)
    service.ingest(
        {
            "_ctx": ctx,
            "sessionId": ctx.session_id,
            "messages": [{"id": "ingested-source", **request["messages"][0]}],
        }
    )

    assert service.after_turn(request)["status"] == "accumulating"
    buffer = manager.get_or_create(ctx.session_id, ctx)
    assert read(buffer, ctx, "original", include_accepted=True)["message_ids"] == [
        "ingested-source"
    ]
    result = service.after_turn({**request, "forceExtract": True})

    assert result["status"] == "completed"
    assert read(buffer, ctx, "original")["status"] == "completed"
    assert buffer.messages == []
    assert writer._pipeline.extract.call_count == 1


def test_completed_capture_restores_native_receipt_after_restart(ctx):
    fs = MemoryFS()
    service, _, _ = setup_service(fs, ctx)
    original = service.after_turn({**params(ctx), "forceExtract": True})

    restarted, _, manager = setup_service(fs, ctx)
    duplicate = restarted.after_turn({**params(ctx), "forceExtract": True})
    receipt = read(manager.get_or_create(ctx.session_id, ctx), ctx, "original")

    assert duplicate["status"] == "duplicate"
    assert receipt["status"] == "completed"
    assert receipt["source_status"] == "ARCHIVED"
    assert receipt["semantic_status"] == "COMPLETE"
    assert receipt["plans"] == compact_plans(original["plans"]) == duplicate["plans"]


def test_accepted_capture_combines_plans_from_saved_source_batches(ctx):
    entries = iter([candidate(), candidate("entity", body="I own the blue release.")])
    service, writer, manager = setup_service(
        MemoryFS(), ctx, extract=lambda **kwargs: [next(entries)]
    )
    service._cfg.after_turn_threshold = 10000
    request = params(ctx)
    request["messages"].append({"role": "user", "content": "I own the blue release."})
    assert service.after_turn(request)["status"] == "accumulating"
    buffer = manager.get_or_create(ctx.session_id, ctx)
    first = buffer.messages[0]
    journal = ExtractionBatchJournal(
        manager, buffer, ctx, [first], archive_id="first-source",
        messages=[asdict(first)], session_time=datetime.now(UTC),
        session_summary="", tool_stats_text="", client_request_id="original",
        request_messages=request["messages"],
        request_source_ids=[message.id for message in buffer.messages],
    )
    service._chunk_and_extract(
        [asdict(first)], writer, ctx, datetime.now(UTC), "", "", "staged",
        ctx.session_id, extraction_journal=journal,
    )

    result = service.after_turn({**request, "forceExtract": True})

    receipt = read(buffer, ctx, "original")
    assert result["status"] == receipt["status"] == "completed"
    assert receipt["source_status"] == "ARCHIVED"
    assert receipt["source_message_ids"] == receipt["archived_source_ids"]
    assert len(receipt["archive_receipts"]) == 2
    assert len(receipt["plans"]) == 2
    assert buffer.messages == []
    assert writer._pipeline.extract.call_count == 2
