import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from commit.write_receipts import WriteNotStartedError
from core.errors import LLMTransientError, ToolCallProtocolError
from core.models import RequestContext
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal, RecoveryNeedsConfirmationError
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import candidate, nodes, params, setup_service


@pytest.fixture
def ctx():
    return RequestContext(account_id="retry", user_id="alice", agent_id="bot",
                          session_id="source-one", trace_id="test")


def queued(ctx, error_type=ToolCallProtocolError):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()  # deterministically drive the real worker below
    writer._pipeline.extract.side_effect = error_type("temporary extraction failure")
    result = service.after_turn({**params(ctx), "retryInBackground": True})
    assert result["ok"] is False and result["recovery_task"]["status"] == "queued"
    assert not nodes(fs, "event")
    return fs, service, writer, manager, result["recovery_task"]["id"]


def ready(manager, ctx):
    manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]["next_attempt_at"] = 0
    assert manager.save_session_state(ctx.session_id, ctx)


def test_delayed_registration_keeps_new_request_checkpoint(ctx):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()
    writer._pipeline.extract.side_effect = ToolCallProtocolError("first source failed")
    register = service._extraction_retry.register
    expected = []

    def delayed_register(request_ctx, failure):
        writer._pipeline.extract.side_effect = None
        writer._pipeline.extract.return_value = [candidate()]
        assert service.after_turn(params(ctx))["status"] == "completed"
        writer._pipeline.extract.side_effect = ToolCallProtocolError("second source failed")
        assert service.after_turn(params(ctx, request="next", content="Another source."))["status"] == "failed"
        expected.append(deepcopy(manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]))
        return register(request_ctx, failure)

    service._extraction_retry.register = delayed_register
    service.after_turn({**params(ctx), "retryInBackground": True})
    assert manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY] == expected[0]
    durable = json.loads(nodes(fs, "state")[0].content)
    assert durable["runtime_extensions"][EXTENSION_KEY] == expected[0]


@pytest.mark.parametrize("background", [False, True])
def test_prewrite_failure_uses_remaining_retry_budget(ctx, background):
    if background:
        fs, service, writer, manager, identity = queued(ctx)
        writer._pipeline.extract.side_effect = None
        writer._pipeline.extract.return_value = [candidate()]
    else:
        fs = MemoryFS()
        service, writer, manager = setup_service(fs, ctx)
        service._extraction_retry.wake = Mock()
    write = writer._writer.write_candidate

    def temporary_failure(entry, request_ctx):
        if entry.category == "event":
            raise WriteNotStartedError("merge model temporarily unavailable")
        return write(entry, request_ctx)

    writer._writer.write_candidate = temporary_failure
    if background:
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, identity)
    else:
        service.after_turn({**params(ctx), "retryInBackground": True})
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch.get("retry_task", {}).get("status") == "queued"
    assert batch["retry_task"]["attempts"] == int(background)
    identity = batch["archive_id"]
    extraction_calls = writer._pipeline.extract.call_count
    writer._writer.write_candidate = write
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert len(nodes(fs, "event")) == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert writer._pipeline.extract.call_count == extraction_calls


def test_busy_request_preserves_retry_budget_until_claim_available(ctx):
    fs, service, writer, manager, identity = queued(ctx)
    buf = manager.get_or_create(ctx.session_id)
    buf.claim_request("original")
    try:
        for _ in range(3):
            ready(manager, ctx)
            assert service._extraction_retry.run_once(ctx, identity) > 0
            task = buf.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
            assert task["status"] == "queued"
            assert task["attempts"] == 0
    finally:
        buf.release_request("original")
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert len(nodes(fs, "event")) == 1
    assert buf.request_completed("original")


@pytest.mark.parametrize("error_type", [ToolCallProtocolError, SessionMutationBusyError])
def test_directory_contention_preserves_retry_budget_until_available(ctx, error_type):
    fs, service, writer, manager, identity = queued(ctx, error_type)
    writer._pipeline.extract.side_effect = SessionMutationBusyError("directory identity is busy")
    for _ in range(3):
        ready(manager, ctx)
        assert service._extraction_retry.run_once(ctx, identity) > 0
        task = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        assert task["status"] == "queued"
        assert task["attempts"] == 0
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert len(nodes(fs, "event")) == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")


@pytest.mark.parametrize("initial_busy", [False, True])
def test_node_format_feedback_survives_directory_contention(ctx, initial_busy):
    from tests.unit.extraction.test_node_processing import NodeLLM, operation
    from tests.unit.server.test_memory_card_recovery import card, enable

    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True

    def decide(kwargs):
        result = operation([0])
        if len(writer._llm.calls) == 1:
            result.pop("edits")
            result.pop("sections")
        else:
            feedback = kwargs["prompt"].split("Previous processing error to correct: ")[1]
            assert "'edits' is a required property" in feedback
            assert "'sections' is a required property" in feedback
        return [result]

    writer._llm = NodeLLM(decide)
    resolver = writer._writer._identity_resolver
    if initial_busy:
        with patch.object(resolver, "lock_directories", side_effect=SessionMutationBusyError("directory busy")):
            response = service.after_turn({**params(ctx), "retryInBackground": True})
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, response["recovery_task"]["id"])
    else:
        response = service.after_turn(params(ctx))
    identity = response["recovery_task"]["id"]
    buf = manager.get_or_create(ctx.session_id)
    before = deepcopy(buf.runtime_extensions[EXTENSION_KEY]["batch"])
    assert before["retry_task"]["failure_kind"] == "memory_card_format"
    assert before["retry_task"]["attempts"] == int(initial_busy)
    with patch.object(resolver, "lock_directories", side_effect=SessionMutationBusyError("directory busy")):
        ready(manager, ctx)
        assert service._extraction_retry.run_once(ctx, identity) > 0
    after = buf.runtime_extensions[EXTENSION_KEY]["batch"]
    for key in ("last_error", "failure_kind", "attempts"):
        assert after["retry_task"][key] == before["retry_task"][key]
    assert after["node_drafts"] == before["node_drafts"]
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert buf.request_completed("original")
    assert len(nodes(fs, "entity")) == 1
    assert len(writer._llm.calls) == 2
    assert writer._pipeline.extract.call_count == 1


def test_durable_retry_survives_service_restart_and_finishes_once(ctx):
    fs, service, _, _, identity = queued(ctx)
    restarted, writer, manager = setup_service(fs, ctx)
    restarted._extraction_retry.wake = Mock()
    status = restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
    assert status["recovery_task"]["id"] == identity
    ready(manager, ctx)
    assert restarted._extraction_retry.run_once(ctx, identity) is None
    assert len(nodes(fs, "event")) == 1
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
    assert restarted._extraction_retry.run_once(ctx, identity) is None
    assert len(nodes(fs, "event")) == 1
    assert writer._pipeline.extract.call_count == 1


def test_exhausted_source_finishes_with_original_archive_and_other_session(ctx):
    fs, service, writer, manager, identity = queued(ctx)
    other = replace(ctx, session_id="source-two")
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    assert service.after_turn(params(other))["status"] == "completed"
    writer._pipeline.extract.side_effect = ToolCallProtocolError("still invalid")
    for _ in range(2):
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, identity)
    buf = manager.get_or_create(ctx.session_id)
    task = buf.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
    assert task["attempts"] == 2 and task["status"] == "queued"
    service._extraction_retry.run_once(ctx, identity)
    assert buf.request_completed("original")
    assert service.wait_until_idle({"_ctx": ctx, "sessionId": ctx.session_id})["idle"] is True
    assert len(nodes(fs, "session_archive")) == 2
    assert service.after_turn(params(ctx))["status"] == "duplicate"


def test_transient_provider_budget_finishes_original_source_archive(ctx):
    fs, service, writer, manager, identity = queued(ctx, LLMTransientError)
    for attempt in range(1, 4):
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, identity)
        task = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        assert task["attempts"] == 0
        assert task["transient_attempts"] == attempt
        assert task["status"] == "queued"
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert service.wait_until_idle({"_ctx": ctx, "sessionId": ctx.session_id})["idle"] is True
    assert len(nodes(fs, "session_archive")) == 1


def test_transient_retry_preserves_node_feedback_and_completes_saved_drafts(ctx):
    from tests.unit.extraction.test_node_processing import NodeLLM, operation
    from tests.unit.server.test_memory_card_recovery import card, enable

    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True

    def decide(kwargs):
        if len(writer._llm.calls) == 2:
            raise LLMTransientError("provider reconnecting")
        item = operation([0])
        if len(writer._llm.calls) == 1:
            item.pop("edits")
        else:
            assert "'edits' is a required property" in kwargs["prompt"]
        return [item]

    writer._llm = NodeLLM(decide)
    response = service.after_turn(params(ctx))
    identity = response["recovery_task"]["id"]
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, identity)
    task = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
    assert task["attempts"] == 0 and task["transient_attempts"] == 1
    assert "'edits' is a required property" in task["format_error"]
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert len(nodes(fs, "entity")) == 1
    assert writer._pipeline.extract.call_count == 1


def test_discovery_wakes_saved_task_and_finishes_source(ctx):
    fs, service, writer, manager, identity = queued(ctx)
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)
    task = deepcopy(manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"])
    from service.extraction_retry import ExtractionRetryWorker
    service._extraction_retry.wake = ExtractionRetryWorker.wake.__get__(service._extraction_retry)
    completed = threading.Event()
    original = service._extraction_retry.run_once

    def run_once(request_ctx, archive_id):
        result = original(request_ctx, archive_id)
        if manager.get_or_create(ctx.session_id).request_completed("original"):
            completed.set()
        return result

    service._extraction_retry.run_once = run_once
    try:
        service._extraction_retry.start_discovery(lambda: [(ctx, task)], interval=.05)
        assert completed.wait(3)
    finally:
        service._extraction_retry.close(timeout=3)
    assert len(nodes(fs, "event")) == 1
    assert service.after_turn(params(ctx))["status"] == "duplicate"


def test_unknown_write_is_not_enqueued(ctx):
    fs = MemoryFS()
    service, _, _ = setup_service(fs, ctx)
    service._after_turn = Mock(return_value={"ok": False, "status": "needs_confirmation", "retryable": False})
    service._extraction_retry.register = Mock()
    result = service.after_turn({**params(ctx), "retryInBackground": True})
    assert result["status"] == "needs_confirmation"
    service._extraction_retry.register.assert_not_called()


def test_retry_cannot_read_another_owner_or_run_a_replaced_source(ctx):
    fs, service, _, manager, identity = queued(ctx)
    assert service._extraction_retry.run_once(ctx, "different-archive") is None
    with pytest.raises(Exception):
        service._extraction_retry.run_once(replace(ctx, user_id="mallory"), identity)
    assert manager.get_or_create(ctx.session_id).pending_extraction_ids()


def test_retry_requires_stable_id_and_does_not_adopt_legacy_batch(ctx):
    fs = MemoryFS()
    service, writer, _ = setup_service(fs, ctx)
    service._extraction_retry.wake = Mock()
    writer._pipeline.extract.side_effect = ToolCallProtocolError("invalid")
    with pytest.raises(ValueError, match="stable clientRequestId"):
        service.after_turn({**params(ctx, request=""), "retryInBackground": True})
    service.after_turn(params(ctx, request=""))
    result = service.after_turn({**params(ctx, request="new"), "retryInBackground": True})
    assert "recovery_task" not in result
    service._extraction_retry.wake.assert_not_called()


def test_backoff_and_missing_task_do_not_drain_account(ctx):
    _, service, writer, manager, identity = queued(ctx)
    service._async_drain.reset_mock()
    assert service._extraction_retry.run_once(ctx, identity) > 0
    service._async_drain.assert_not_called()

    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, identity)
    assert service._async_drain.called
    service._async_drain.reset_mock()
    service._extraction_retry.run_once(ctx, identity)
    service._async_drain.assert_not_called()


@pytest.mark.parametrize("foreground", ["accept", "finish", "replace"])
def test_retry_update_preserves_concurrent_foreground_checkpoint(ctx, foreground):
    fs, service, writer, manager, identity = queued(ctx)
    buf = manager.get_or_create(ctx.session_id)
    journal = ExtractionBatchJournal.resume(manager, buf, ctx)
    reached, release = threading.Event(), threading.Event()
    save_lock = buf.session_state_save_lock

    class GatedSaveLock:
        def __enter__(self):
            if threading.current_thread().name.startswith("retry-update") and not reached.is_set():
                reached.set()
                assert release.wait(5)
            return save_lock.__enter__()

        def __exit__(self, *args):
            return save_lock.__exit__(*args)

    buf.session_state_save_lock = GatedSaveLock()

    def update():
        try:
            journal.mutate(lambda batch: batch["retry_task"].update(status="running"))
        except RecoveryNeedsConfirmationError:
            pass

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="retry-update") as pool:
        pending = pool.submit(update)
        try:
            assert reached.wait(5)
            if foreground == "accept":
                journal.checkpoint("chunk:0").accept([candidate()], 1, 0)
                expected = deepcopy(journal.batch["slots"])
            else:
                journal.finish()
                if foreground == "replace":
                    assert service.after_turn(params(ctx, request="next", content="A separate source."))["status"] == "failed"
                expected = deepcopy(buf.runtime_extensions.get(EXTENSION_KEY))
        finally:
            release.set()
        pending.result(timeout=5)
    durable = json.loads(nodes(fs, "state")[0].content).get("runtime_extensions", {})
    if foreground == "accept":
        assert durable[EXTENSION_KEY]["batch"]["slots"] == expected
    else:
        assert durable.get(EXTENSION_KEY) == expected


@pytest.mark.parametrize("restart", [False, True])
def test_background_retry_preserves_governed_turn_source(ctx, tmp_path, restart):
    from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.unit.server.test_memory_service_tool_results import _user_message_turn

    fs = MemoryFS()
    def start_service():
        _, writer, _ = setup_service(fs, ctx)
        service = MemoryService(OgMemConfig(tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message", after_turn_threshold=1,
            session_state_bridge_enabled=False), tool_result_store=LocalToolResultArtifactStore(tmp_path))
        service._write_api = writer
        service._generate_and_write_summary = Mock()
        service.drain_outbox_sync = Mock(return_value={})
        service._async_drain = Mock()
        service._extraction_retry.wake = Mock()
        manager = service.get_session_manager()
        manager.commit_snapshot = Mock(return_value={"ok": True, "status": "completed", "archived": True})
        return service, writer, manager

    service, writer, manager = start_service()
    writer._pipeline.extract.side_effect = ToolCallProtocolError("invalid generated output")
    request = {**params(ctx), "retryInBackground": True, "prePromptMessageCount": 1,
               "messages": [{"role": "user", "content": "Earlier context"}] + _user_message_turn("Receiver repaired.")}
    result = service.after_turn(request)
    identity = result["recovery_task"]["id"]
    ledger_key = service._tool_result_turn_ledger.STATE_KEY
    ledger = deepcopy(manager.get_or_create(ctx.session_id).runtime_extensions[ledger_key])
    if restart:
        service, writer, manager = start_service()
        status = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})
        assert status["recovery_task"]["id"] == identity
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)

    assert service._extraction_retry.run_once(ctx, identity) is None

    assert len(nodes(fs, "event")) == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert manager.get_or_create(ctx.session_id).runtime_extensions[ledger_key] == ledger


def test_recovered_source_emits_completion_after_durable_receipt(ctx, caplog):
    import logging

    fs, service, writer, manager, identity = queued(ctx)
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [candidate()]
    ready(manager, ctx)
    with caplog.at_level(logging.INFO, logger="ogmem.service"):
        assert service._extraction_retry.run_once(ctx, identity) is None
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert len(nodes(fs, "event")) == 1
    assert len(nodes(fs, "session_archive")) == 1
    completed = [r for r in caplog.records if "after_turn background extract done" in r.getMessage()]
    assert len(completed) == 1
    assert "session=" + ctx.session_id[:8] in completed[0].getMessage()
