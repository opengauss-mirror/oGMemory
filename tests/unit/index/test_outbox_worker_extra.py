"""Additional tests for OutboxWorker edge cases."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from core.models import OutboxEvent
from index.outbox_worker import OutboxWorker, WorkerResult


def test_build_request_context_from_user_uri():
    worker = OutboxWorker(vector_index=Mock(), embedder=Mock())
    event = OutboxEvent(
        event_id="evt-1",
        event_type="UPSERT_DIRECTORY",
        uri="ctx://acct-1/users/u1/memories/preferences/",
        payload={},
        status="PENDING",
    )

    ctx = worker._build_request_context(event)

    assert ctx is not None
    assert ctx.account_id == "acct-1"
    assert ctx.user_id == "u1"
    assert ctx.agent_id == "outbox-worker"


def test_usage_context_attributes_user_uri_to_user_owner():
    worker = OutboxWorker(vector_index=Mock(), embedder=Mock())
    event = OutboxEvent(
        event_id="evt-usage",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/preferences/item",
        payload={},
        status="PENDING",
    )

    worker_ctx = worker._build_request_context(event)
    usage_ctx = worker._usage_context_for_event(event, worker_ctx)

    assert worker_ctx.agent_id == "outbox-worker"
    assert usage_ctx.account_id == "acct-1"
    assert usage_ctx.user_id == "u1"
    assert usage_ctx.agent_id == ""


def test_run_once_passes_built_context_to_process_event(monkeypatch):
    worker = OutboxWorker(vector_index=Mock(), embedder=Mock())
    event = OutboxEvent(
        event_id="evt-2",
        event_type="UPSERT_DIRECTORY",
        uri="ctx://acct-1/users/u1/memories/preferences/",
        payload={},
        status="PENDING",
    )
    store = Mock()
    store.list_pending.return_value = [(event.uri, event)]
    store.try_acquire.return_value = True
    process_event = Mock(return_value=WorkerResult(event_id=event.event_id, success=True))
    monkeypatch.setattr(worker, "process_event", process_event)
    monkeypatch.setattr(worker, "_handle_processing_result", Mock())

    worker.run_once(store, ["acct-1"], worker_id="worker-1")

    passed_ctx = process_event.call_args.args[1]
    assert passed_ctx.account_id == "acct-1"
    assert passed_ctx.user_id == "u1"


def test_build_request_context_returns_none_for_invalid_uri():
    worker = OutboxWorker(vector_index=Mock(), embedder=Mock())
    event = OutboxEvent(
        event_id="evt-3",
        event_type="UPSERT_CONTEXT",
        uri="not-a-ctx-uri",
        payload={},
        status="PENDING",
    )

    assert worker._build_request_context(event) is None


def test_run_once_uses_batch_claim_when_supported(monkeypatch):
    worker = OutboxWorker(vector_index=Mock(), embedder=Mock())
    event = OutboxEvent(
        event_id="evt-4",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct-1/users/u1/memories/profile",
        payload={"records": []},
        status="PENDING",
    )
    store = Mock()
    store.supports_batch_claim = True
    store.claim_batch.return_value = [(event.uri, event)]
    process_event = Mock(
        return_value=WorkerResult(event_id=event.event_id, success=True)
    )
    monkeypatch.setattr(worker, "process_event", process_event)

    stats = worker.run_once(store, ["acct-1"], worker_id="worker-2")

    store.claim_batch.assert_called_once_with(worker_id="worker-2", exclude_event_types=["EXTRACT_AGENT_MEMORY"])
    process_event.assert_called_once()
    store.mark_done.assert_called_once_with(event, event.uri)
    assert stats["processed"] == 1
    assert stats["succeeded"] == 1
