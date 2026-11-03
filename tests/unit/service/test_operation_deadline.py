"""Explicit request budgets stop follow-on work without losing completed usage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.operation_deadline as deadline
from core.errors import OperationDeadlineExceededError
from providers.llm.openai_llm import OpenAILLM


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(deadline, "monotonic", lambda: value[0])
    return value


@pytest.mark.parametrize("value", [0, -1, True, "1", float("nan"), float("inf"), 10 ** 1000])
def test_invalid_budget_rejected(value):
    with pytest.raises(ValueError):
        with deadline.operation_deadline(value):
            pass


def test_nested_budget_cannot_extend_parent_and_context_resets(clock):
    assert deadline.remaining_seconds() is None
    with deadline.operation_deadline(5):
        clock[0] += 1
        with deadline.operation_deadline(10):
            assert deadline.remaining_seconds() == 4
        with deadline.operation_deadline(1):
            clock[0] += 1
            with pytest.raises(OperationDeadlineExceededError):
                deadline.check_deadline()
        assert deadline.remaining_seconds() == 3
    assert deadline.remaining_seconds() is None


@pytest.mark.parametrize("params", [{"timeoutSeconds": None, "wait": True},
                                    {"timeoutSeconds": 1}, {"timeoutSeconds": 0, "wait": True}])
def test_invalid_after_turn_budget_is_rejected_before_accumulation(params):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    service = MemoryService(config=OgMemConfig())
    service._after_turn = Mock()
    with pytest.raises(ValueError):
        service.after_turn({**params, "messages": [{"role": "user", "content": "source"}]})
    service._after_turn.assert_not_called()


def test_rejected_format_does_not_launch_fallback_after_expiry(clock, monkeypatch):
    from tests.unit.providers.llm.test_openai_llm import _bad_request
    def reject(**kwargs):
        clock[0] += 2
        raise _bad_request("json_object unsupported")
    llm, client = llm_with_response(monkeypatch, reject)
    llm._json_mode = True
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        llm.complete_json("source", {"type": "object"})
    client.chat.completions.create.assert_called_once()
    llm.token_tracker.record_failure.assert_called_once_with("llm")


def test_required_tool_probe_wait_uses_remaining_budget(clock, monkeypatch):
    llm, client = llm_with_response(monkeypatch, lambda **kwargs: response())
    lock = Mock()
    lock.acquire.return_value = False
    llm._required_tool_choice_lock = lock
    with deadline.operation_deadline(.5), pytest.raises(OperationDeadlineExceededError):
        llm.complete_with_tools("source", [], tool_choice="required")
    lock.acquire.assert_called_once_with(timeout=.5)
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("usage_arrives", [False, True])
def test_stream_expiry_stops_consumption_and_preserves_arrived_usage(clock, monkeypatch, usage_arrives):
    import json

    import httpx
    from openai import OpenAI
    consumed = []
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "{}"}, "finish_reason": None}]},
                ({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
                 if usage_arrives else {"choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": None}]}),
                {"choices": []},
            ]
            for index, chunk in enumerate(chunks):
                consumed.append(index)
                if index == 1:
                    clock[0] += 2
                yield ("data: " + json.dumps({"id": "round", "created": 1, "object": "chat.completion.chunk",
                                              "model": "test-model", **chunk}) + "\n\n").encode()
    llm, _ = llm_with_response(monkeypatch, None)
    llm._extra_body = {"stream": True}
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=Chunks()))
    with OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport)) as client:
        llm._client = client
        with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
            llm.complete_json("source", {"type": "object"})
    assert consumed == [0, 1]
    if usage_arrives:
        llm.token_tracker.record_llm.assert_called_once_with(10, 2, 0, 0, usage_known=True, model="test-model")
        llm.token_tracker.record_failure.assert_not_called()
    else:
        llm.token_tracker.record_llm.assert_not_called()
        llm.token_tracker.record_failure.assert_called_once_with("llm")


def test_public_checkpoint_expiry_before_accept_can_retry_under_new_budget(clock):
    from core.models import RequestContext
    from service.extraction_checkpoint import EXTENSION_KEY
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import (
        candidate,
        nodes,
        params,
        setup_service,
    )
    ctx = RequestContext("deadline", "reader", "agent", "session", "trace")
    fs = MemoryFS()
    def late_extract(**kwargs):
        clock[0] += 2
        return [candidate()]
    service, writer, manager = setup_service(fs, ctx, extract=late_extract)
    result = service.after_turn({**params(ctx), "timeoutSeconds": 1})
    assert result["status"] == "failed" and result["error"] == "Operation deadline exceeded"
    assert not nodes(fs, "event")
    buf = manager.get_or_create(ctx.session_id)
    assert buf.messages and buf.extraction_watermark == 0 and buf.extraction_done_event.is_set()
    batch = buf.runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch["slots"]["raw:0"]["operations"]["0"]["status"] == "complete"
    writer._pipeline.extract = Mock(return_value=[candidate()])
    assert service.after_turn({**params(ctx), "timeoutSeconds": 5})["status"] == "completed"
    assert len(nodes(fs, "event")) == 1 and not buf.messages


@pytest.mark.parametrize("typed_prewrite_error", [False, True])
def test_expiry_after_storage_commit_keeps_unknown_outcome_and_receipt_recovery(clock, typed_prewrite_error):
    from commit.write_receipts import WriteDeadlineExceededError
    from core.models import RequestContext
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import nodes, params, setup_service
    class LostAcknowledgementFS(MemoryFS):
        lost = False
        unread_receipt = None

        def read_node(self, uri, ctx):
            if uri == self.unread_receipt:
                self.unread_receipt = None
                raise ConnectionError("receipt temporarily unavailable after commit")
            return super().read_node(uri, ctx)

        def write_node(self, node, ctx):
            super().write_node(node, ctx)
            if node.category == "event" and not self.lost:
                self.lost = True
                self.unread_receipt = node.uri
                clock[0] += 2
                error = WriteDeadlineExceededError if typed_prewrite_error else OperationDeadlineExceededError
                raise error("lost acknowledgement after commit")
    ctx = RequestContext("deadline", "reader", "agent", "session", "trace")
    fs = LostAcknowledgementFS()
    service, writer, manager = setup_service(fs, ctx)
    result = service.after_turn({**params(ctx), "timeoutSeconds": 1})
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract = Mock(side_effect=AssertionError("accepted batch must not re-extract"))
    assert service.after_turn({**params(ctx), "timeoutSeconds": 5})["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


def test_expiry_during_candidate_build_stops_physical_write(clock):
    from core.models import RequestContext
    from service.api import MemoryWriteAPI
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import NoModel, candidate
    fs = MemoryFS()
    api = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    original = api._writer._archive_builder.build
    def late_build(*args, **kwargs):
        node = original(*args, **kwargs)
        clock[0] += 2
        return node
    api._writer._archive_builder.build = late_build
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        api.write_memory(candidate(write_operation_id="accepted-operation"),
                         RequestContext("deadline", "reader", "agent", "session", "trace"))
    assert fs.nodes == {}


@pytest.mark.parametrize("phase", ["begin", "build", "build_error"])
def test_accepted_prewrite_deadline_retries_without_reextracting(clock, monkeypatch, phase):
    from core.models import RequestContext
    from service.extraction_checkpoint import EXTENSION_KEY, CandidateCheckpoint
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import nodes, params, setup_service
    ctx = RequestContext("deadline", "reader", "agent", "session", "trace")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    build = writer._writer._archive_builder.build
    begin = CandidateCheckpoint.begin
    def late_begin(self, index):
        begin(self, index)
        if self.slot == "chunk:0":
            clock[0] += 2
    def late_build(*args, **kwargs):
        node = build(*args, **kwargs)
        if node.category == "event":
            clock[0] += 2
            if phase == "build_error":
                deadline.check_deadline()
        return node
    if phase == "begin":
        monkeypatch.setattr(CandidateCheckpoint, "begin", late_begin)
    else:
        writer._writer._archive_builder.build = late_build
    result = service.after_turn({**params(ctx), "timeoutSeconds": 1})
    assert result["status"] == "failed" and result["retryable"] is True
    assert not nodes(fs, "event")
    buf = manager.get_or_create(ctx.session_id)
    slot = buf.runtime_extensions[EXTENSION_KEY]["batch"]["slots"]["chunk:0"]
    assert slot["operations"]["0"]["status"] == "rejected_before_write"
    assert buf.messages and buf.extraction_watermark == 0 and buf.extraction_done_event.is_set()
    monkeypatch.setattr(CandidateCheckpoint, "begin", begin)
    writer._writer._archive_builder.build = build
    writer._pipeline.extract = Mock(side_effect=AssertionError("accepted batch must not re-extract"))
    assert service.after_turn({**params(ctx), "timeoutSeconds": 5})["status"] == "completed"
    assert len(nodes(fs, "event")) == 1 and not buf.messages
    writer._pipeline.extract.assert_not_called()


def llm_with_response(monkeypatch, response):
    client = Mock()
    client.with_options.return_value = client
    client.chat.completions.create.side_effect = response
    monkeypatch.setattr("providers.llm.openai_llm.OpenAI", Mock(return_value=client))
    llm = OpenAILLM(api_key="test-key")
    llm.token_tracker = Mock()
    return llm, client


def response():
    return SimpleNamespace(model="test-model", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                           choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                               content='{"ok": true}', tool_calls=[]))])


def test_late_response_usage_recorded_then_deadline_propagated(clock, monkeypatch):
    def finish(**kwargs):
        clock[0] += 2
        return response()
    llm, client = llm_with_response(monkeypatch, finish)
    with deadline.operation_deadline(1):
        with pytest.raises(OperationDeadlineExceededError):
            llm.complete_json("source", {"type": "object"})
        with pytest.raises(OperationDeadlineExceededError):
            llm.complete_with_tools_messages([{"role": "user", "content": "next"}])
    client.with_options.assert_called_once_with(timeout=1.0, max_retries=0)
    client.chat.completions.create.assert_called_once()
    llm.token_tracker.record_llm.assert_called_once_with(10, 2, 0, 0, usage_known=True, model="test-model")
    llm.token_tracker.record_failure.assert_not_called()


def test_default_provider_path_unchanged(clock, monkeypatch):
    llm, client = llm_with_response(monkeypatch, lambda **kwargs: response())
    assert llm.complete_json("source", {"type": "object"}) == {"ok": True}
    client.with_options.assert_not_called()


@pytest.mark.parametrize("timeout", [.25, 10, None])
def test_per_call_timeout_cannot_override_operation_budget(clock, monkeypatch, timeout):
    llm, client = llm_with_response(monkeypatch, lambda **kwargs: response())
    with deadline.operation_deadline(1):
        llm._create_completion(timeout=timeout)
    assert client.chat.completions.create.call_args.kwargs["timeout"] == (.25 if timeout == .25 else 1)


def test_custom_http_timeouts_preserve_tighter_components(clock, monkeypatch):
    import httpx
    llm, client = llm_with_response(monkeypatch, lambda **kwargs: response())
    with deadline.operation_deadline(1):
        llm._create_completion(timeout=httpx.Timeout(connect=.25, read=None, write=5, pool=.5))
    assert client.chat.completions.create.call_args.kwargs["timeout"].as_dict() == {
        "connect": .25, "read": 1, "write": 1, "pool": .5}


def test_extraction_expiry_does_not_accept_or_write_candidates(clock):
    from service.api import MemoryWriteAPI
    api = object.__new__(MemoryWriteAPI)
    api._node_processing_enabled = False
    api._fs = object()
    api._pipeline = Mock()
    api._writer = Mock()
    checkpoint = Mock()
    checkpoint.journal.batch = {"archive_id": "archive"}
    checkpoint.load.return_value = None
    def extract(*args, **kwargs):
        assert kwargs["parallel"] is False
        clock[0] += 2
        return []
    api._pipeline.extract.side_effect = extract
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        api.commit_session([], Mock(), archive_id="archive", candidate_checkpoint=checkpoint)
    checkpoint.accept.assert_not_called()
    api._writer.write_candidate.assert_not_called()


def test_completed_write_receipt_preserved_and_next_write_not_started(clock):
    from core.models import WritePlan
    from service.api import MemoryWriteAPI
    api = object.__new__(MemoryWriteAPI)
    api._writer = Mock()
    api._writer.prewrite_skip.return_value = None
    checkpoint = Mock()
    checkpoint.completed.return_value = None
    checkpoint.outcome_uncertain.return_value = False
    plan = WritePlan(action="create", target_uri="memory://done", merged_fields={})
    def write(*args):
        clock[0] += 2
        return plan
    api._writer.write_candidate.side_effect = write
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        api._write_checkpointed_candidates([Mock(), Mock()], Mock(), checkpoint)
    checkpoint.complete.assert_called_once_with(0, plan)
    checkpoint.begin.assert_called_once_with(0)
    checkpoint.reject.assert_not_called()


def test_after_turn_deadline_keeps_source_and_releases_claim(clock):
    from core.models import RequestContext
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    ctx = RequestContext("deadline", "user", "agent", "session", "trace")
    service = MemoryService(config=OgMemConfig(after_turn_threshold=1, session_state_bridge_enabled=False))
    service._write_api = Mock()
    service.drain_outbox_sync = Mock()
    mgr = service.get_session_manager()
    mgr.save_session_state = Mock(return_value=True)
    mgr.commit_snapshot = Mock()
    def extract(**kwargs):
        assert deadline.remaining_seconds() == 1
        clock[0] += 2
        deadline.check_deadline()
    service._chunk_and_extract = Mock(side_effect=extract)
    result = service.after_turn({"sessionId": "session", "clientRequestId": "request-1", "_ctx": ctx,
                                 "timeoutSeconds": 1, "messages": [{"role": "user", "content": "fact " * 80}]})
    assert result["ok"] is False and result["retryable"] is True
    assert result["error"] == "Operation deadline exceeded"
    buf = mgr.get_or_create("session", ctx=ctx)
    assert len(buf.messages) == 1 and buf.extraction_watermark == 0
    assert buf.extraction_done_event.is_set()
    assert buf.claim_request("request-1") != "processing"
    service.drain_outbox_sync.assert_not_called()
    mgr.commit_snapshot.assert_not_called()
    assert deadline.remaining_seconds() is None
