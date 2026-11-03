"""Asynchronous sources finish through the durable retry worker."""

import time
from unittest.mock import Mock

import pytest

from core.errors import LLMOutputTruncatedError, ToolCallProtocolError
from core.models import RequestContext
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from service.extraction_checkpoint import EXTENSION_KEY
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import candidate, nodes, setup_service


@pytest.mark.parametrize("failed_field", ["client_request_id", "retry_task"])
def test_registration_save_recovers_same_source(failed_field):
    ctx = RequestContext("recovery", "reader", "agent", "registration", "trace")
    fs = MemoryFS()
    service, _, manager = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()
    save = manager.save_session_state
    failed = []

    def reconnect_once(session_id, request):
        batch = manager.get_or_create(session_id).runtime_extensions.get(EXTENSION_KEY, {}).get("batch", {})
        if batch.get(failed_field) and not failed:
            failed.append(True)
            return False
        return save(session_id, request)

    manager.save_session_state = reconnect_once
    request = {"_ctx": ctx, "sessionId": ctx.session_id, "wait": False,
               "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver."}]}
    result = service.after_turn(request)
    buf = manager.get_or_create(ctx.session_id)
    assert result["status"] == "failed" and result["retryable"]
    assert buf.extraction_active_count == 0
    assert buf.extraction_watermark == 0

    assert service.after_turn({**request, "wait": True})["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    assert len(nodes(fs, "session_archive")) == 1
    assert buf.messages == []


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("error_type", [LLMOutputTruncatedError, ToolCallProtocolError, SessionMutationBusyError])
def test_background_source_is_recovered_once(restart, error_type):
    ctx = RequestContext("recovery", "reader", "agent", "truncated", "trace")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()
    writer._pipeline.extract.side_effect = error_type("temporary extraction failure")
    result = service.after_turn({"_ctx": ctx, "sessionId": ctx.session_id, "wait": False,
                                "messages": [{"role": "user", "content": "I repaired the receiver."}]})
    assert result["status"] == "processing"
    buf = manager.get_or_create(ctx.session_id)
    deadline = time.monotonic() + 5
    while buf.extraction_active_count and time.monotonic() < deadline:
        time.sleep(.01)
    task = buf.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
    assert (task["status"], task["attempts"]) == ("queued", 0)
    if restart:
        service, writer, manager = setup_service(fs, ctx)
        service._extraction_retry.wake = Mock()
        status = service.wait_until_idle({"_ctx": ctx, "sessionId": ctx.session_id})
        assert status["recovery_task"]["id"] == task["id"]
        buf = manager.get_or_create(ctx.session_id)
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    buf.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]["next_attempt_at"] = 0
    assert manager.save_session_state(ctx.session_id, ctx)

    assert service._extraction_retry.run_once(ctx, task["id"]) is None

    assert len(nodes(fs, "event")) == 1
    assert len(nodes(fs, "session_archive")) == 1
    assert len(buf.messages) == 0
    assert buf.runtime_extensions.get(EXTENSION_KEY) is None
