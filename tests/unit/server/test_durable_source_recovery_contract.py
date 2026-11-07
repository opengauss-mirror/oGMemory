"""Checkpointed sources receive durable recovery acknowledgements and complete."""

from commit.outbox_scope import capture_outbox_events

import json
from unittest.mock import Mock

import pytest

from core.write_contract import write_request_contract
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from service.extraction_retry import retry_failed_source
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.commit.test_request_outbox_events import request_context
from tests.unit.server.test_extraction_batch_recovery import nodes, params, setup_service


@pytest.mark.parametrize("background", [None, False, True], ids=["default", "false", "true"])
@pytest.mark.parametrize("registered", [False, True], ids=["checkpoint", "existing-task"])
@capture_outbox_events
def test_checkpointed_source_is_acknowledged_and_completes(background, registered):
    fs = MemoryFS()
    ctx = request_context("durable-recovery-contract")
    service, writer, manager = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()
    request = params(ctx)
    if background is not None:
        request["retryInBackground"] = background
    manager.add_message(ctx.session_id, "user", request["messages"][0]["content"], ctx,
                        message_id="msg-one")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
        archive_id="durable-source", messages=request["messages"], session_time=None,
        session_summary="", tool_stats_text="", client_request_id=request["clientRequestId"],
        request_messages=request["messages"])
    if registered:
        service._extraction_retry.register_pending_source(ctx, journal, wake=False)

    @write_request_contract
    @retry_failed_source
    def acknowledge_attempt(self, incoming):
        return {"ok": False, "status": "failed", "retryable": True,
                "extraction_archive_id": "durable-source", "client_request_id": "original"}

    result = acknowledge_attempt(service, request)
    assert result["ok"] is True
    assert result["request_status"] == "RECOVERING"
    assert result["source_status"] == "CHECKPOINTED"
    assert result["accepted"] is True and result["terminal"] is False
    assert result["http_status"] == 202 and result["retry_action"] == "poll"
    saved = json.loads(nodes(fs, "state")[0].content)
    batch = saved["runtime_extensions"][EXTENSION_KEY]["batch"]
    assert batch["retry_task"] == result["recovery_task"]
    assert batch["retry_task"]["attempts"] == 0
    assert batch["request_messages"] == request["messages"]
    assert batch["message_ids"] == ["msg-one"]
    assert acknowledge_attempt(service, request)["recovery_task"] == result["recovery_task"]

    journal.mutate(lambda batch: batch["retry_task"].update(next_attempt_at=0))
    assert service._extraction_retry.run_once(ctx, "durable-source") is None
    final = service.get_write_request_status(request)
    assert final["request_status"] == "SUCCEEDED"
    assert final["source_status"] == "ARCHIVED" and final["semantic_status"] == "COMPLETE"
    assert final["archive_id"] == "durable-source"
    assert buffer.request_completed("original")
    assert len(nodes(fs, "event")) == 1 and len(nodes(fs, "session_archive")) == 1
    writer._pipeline.extract.assert_called_once()
    manager.commit_snapshot.assert_called_once()
