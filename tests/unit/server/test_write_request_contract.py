"""Positive write receipts retain quality and exact projection acknowledgements."""

import json

import pytest

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from commit.write_receipts import PROJECTION_RECEIPTS_KEY, receipt_plan
from core.errors import ExtractionBudgetError, ExtractionSemanticError, LLMTransientError
from core.write_contract import OUTCOMES_KEY, error_fields, normalize_write_result, projection_outcome, store_outcome, write_http_status
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from service.extraction_recovery_discovery import DISCOVERY_KEY, ExtractionRecoveryDiscovery
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_request_outbox_events import make_runtime, memory_node, request_context
from tests.unit.server.test_extraction_batch_recovery import NoModel


def prepare():
    client, fs, store = make_runtime()
    ctx = request_context("write-contract")
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager.add_message(ctx.session_id, "user", "The receiver was repaired.", ctx,
                        message_id="source-one", created_at="2026-10-29T01:00:00+00:00")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
        archive_id="archive-one", messages=[{"role": "user", "content": buffer.messages[0].content}],
        session_time="2026-10-29T01:00:00+00:00", session_summary="", tool_stats_text="",
        client_request_id="write-one")
    return client, fs, store, ctx, manager, journal


@pytest.mark.parametrize("semantic,state", [("COMPLETE", "SUCCEEDED"), ("DEGRADED", "DEGRADED")])
@capture_outbox_events
def test_finished_receipt_and_buffer_are_persisted_together(semantic, state):
    _, fs, store, ctx, manager, journal = prepare()
    receipt = journal.finish(outcome={"semantic_status": semantic})
    payload = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert payload["completed_request_ids"] == ["write-one"]
    assert payload["buffer_messages"] == []
    assert payload["runtime_extensions"][OUTCOMES_KEY]["entries"]["write-one"] == receipt
    assert projection_outcome(receipt, store)["request_status"] == state


@capture_outbox_events
def test_atomic_operation_restores_projection_identity_after_normal_restart():
    client, fs, store, ctx, manager, journal = prepare()
    writer = MemoryWriteAPI(fs, NoModel(), store, directory_summary_enabled=False)
    checkpoint = journal.checkpoint("chunk:0")
    accepted = checkpoint.accept([memory("receiver", content="The receiver was repaired.")], 1, 0)

    @capture_outbox_events
    def publish():
        checkpoint.begin(0)
        plan = writer._writer.write_candidate(accepted[0], ctx)
        checkpoint.complete(0, plan)
        return plan, current_outbox_events()

    plan, references = publish()
    source = fs.read_node(plan.target_uri, ctx)
    bound = source.metadata[PROJECTION_RECEIPTS_KEY][accepted[0].write_operation_id]["event_id"]
    assert references == [(plan.target_uri, bound)]

    @capture_outbox_events
    def replay():
        proof = receipt_plan(fs, writer._writer._identity_resolver.uris, accepted[0], ctx)
        return proof, current_outbox_events()

    proof, replayed = replay()
    assert proof.merged_fields["projection_evidence_known"] is True
    assert replayed == references
    receipt = journal.finish(outcome={"semantic_status": "COMPLETE"})
    assert projection_outcome(receipt, store)["request_status"] == "PROJECTING"
    for uri, event in store.list_pending_events(references):
        store.mark_done(event, uri)

    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service.get_session_manager = lambda: manager
    service._get_outbox_store = lambda _fs: store
    result = service.get_write_request_status({"_ctx": ctx, "sessionId": ctx.session_id,
                                                "clientRequestId": "write-one"})
    assert result["request_status"] == "SUCCEEDED"
    assert result["projection_status"] == "DONE"
    assert result["retry_action"] == "none"
    assert store.get_event_status(plan.target_uri, bound) == "DONE"


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("semantic,state", [("COMPLETE", "SUCCEEDED"), ("DEGRADED", "DEGRADED")])
@capture_outbox_events
def test_discovery_persists_completed_projection_and_preserves_next_turn(restart, semantic, state):
    _, fs, store, ctx, manager, journal = prepare()
    node = memory_node(ctx)
    fs.write_node(node, ctx)
    source = fs.read_node(node.uri, ctx)
    event = store.register_write(source, ctx)
    references = [[node.uri, event.event_id]]
    journal.mutate(lambda batch: batch.update(outbox_events=references))
    pending = journal.finish(outcome={"semantic_status": semantic},
                             resolve_outcome=lambda value: projection_outcome(value, store))
    assert pending["projection_status"] == "PENDING"

    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    recovered = (SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
                 if restart else manager)
    service.get_session_manager = lambda: recovered
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda _fs: store
    service._load_extraction_recovery_before_accumulation(ctx.session_id, ctx)

    manager.add_message(ctx.session_id, "user", "The antenna was repaired.", ctx,
                        message_id="source-two", created_at="2026-10-29T01:01:00+00:00")
    next_receipt = store_outcome(manager.get_or_create(ctx.session_id, ctx), "write-two", {
        "request_id": "write-two", "request_status": "ACCEPTED", "source_status": "BUFFERED",
        "semantic_status": "PENDING", "source_message_ids": ["source-two"]})
    assert manager.save_session_state(ctx.session_id, ctx)

    index = InMemoryVectorIndex(dimension=16)
    stats = OutboxWorker(index, MockEmbedder(16), fs=fs, directory_summary_enabled=False).run_once(
        store, [ctx.account_id], pending_events=store.list_pending_events([tuple(item) for item in references]))
    assert stats["succeeded"] == 1
    assert store.get_event_status(node.uri, event.event_id) == "DONE"
    state_uri = manager._session_state.state_uri(ctx.account_id, ctx.session_id)
    before = fs.read_node(state_uri, ctx).content
    params = {"_ctx": ctx, "sessionId": ctx.session_id, "clientRequestId": "write-one"}
    assert service.get_write_request_status(params)["request_status"] == state
    assert fs.read_node(state_uri, ctx).content == before

    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id],
                                            projection_handler=service._refresh_write_outcomes)
    discover()
    saved = fs.read_node(state_uri, ctx)
    payload = json.loads(saved.content)
    receipts = payload["runtime_extensions"][OUTCOMES_KEY]["entries"]
    assert receipts["write-one"]["request_status"] == state
    assert receipts["write-one"]["semantic_status"] == semantic
    assert receipts["write-one"]["projection_status"] == "DONE"
    assert receipts["write-one"]["outbox_events"] == references
    assert receipts["write-two"] == next_receipt
    assert payload["buffer_messages"] == [{"id": "source-two", "role": "user",
        "content": "The antenna was repaired.", "created_at": "2026-10-29T01:01:00+00:00"}]
    assert payload["completed_request_ids"] == ["write-one"]
    assert saved.metadata[DISCOVERY_KEY]["projection_pending"] is False
    assert fs.read_node(node.uri, ctx).content == source.content
    assert fs.read_node(node.uri, ctx).metadata["version"] == source.metadata["version"]
    assert service.get_write_request_status(params) == receipts["write-one"]
    assert fs.read_node(state_uri, ctx).content == saved.content


@capture_outbox_events
def test_buffered_request_receipt_advances_with_its_original_source_batch():
    _, fs, store, ctx, manager, journal = prepare()
    store_outcome(journal.buffer, "prior-turn", {"source_status": "BUFFERED",
                                               "source_message_ids": ["source-one"]})
    journal.finish(outcome={"semantic_status": "COMPLETE"})
    payload = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    receipt = payload["runtime_extensions"][OUTCOMES_KEY]["entries"]["prior-turn"]
    assert receipt["request_id"] == "prior-turn"
    assert projection_outcome(receipt, store)["request_status"] == "SUCCEEDED"


@pytest.mark.parametrize("error,code", [
    (ExtractionBudgetError("complete input budget"), "EXTRACTION_CONTEXT_BUDGET"),
    (ExtractionSemanticError("identity qualification"), "EXTRACTION_SEMANTIC_INCOMPLETE"),
    (LLMTransientError("provider busy"), "MODEL_TEMPORARILY_UNAVAILABLE"),
])
@capture_outbox_events
def test_accepted_recovery_keeps_domain_cause_and_returns_poll(error, code):
    result = normalize_write_result({"request_status": "RECOVERING", "accepted": True,
                                    **error_fields(error, "source-request")}, "source-request")
    assert result["error_detail"]["code"] == code
    assert result["retry_action"] == "poll"
    assert result["error_detail"]["retry_action"] == "poll"
    assert write_http_status(result) == 202


@capture_outbox_events
def test_buffered_acceptance_waits_for_normal_extraction_trigger():
    result = normalize_write_result({"request_status": "ACCEPTED", "source_status": "BUFFERED",
                                     "semantic_status": "PENDING"}, "buffered-source")
    assert result["accepted"] is True
    assert result["terminal"] is False
    assert result["retry_action"] == "none"


@capture_outbox_events
def test_repeated_checkpointed_request_reports_the_active_write():
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_extraction_batch_recovery import setup_service
    fs = MemoryFS()
    ctx = request_context("active-write-contract")
    service, writer, manager = setup_service(fs, ctx)
    text = "The receiver was repaired."
    manager.add_message(ctx.session_id, "user", text, ctx, message_id="active-source")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    assert buffer.claim_request("active-request") == "claimed"
    ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages), archive_id="active-archive",
        messages=[{"role": "user", "content": text}], session_time=None,
        session_summary="", tool_stats_text="", client_request_id="active-request")
    result = service.after_turn({"_ctx": ctx, "sessionId": ctx.session_id,
        "clientRequestId": "active-request", "messages": [{"role": "user", "content": text}]})
    assert result["request_status"] == "RUNNING"
    assert result["accepted"] is True
    assert result["http_status"] == 202
    assert result["retry_action"] == "poll"
    assert [message.id for message in buffer.messages] == ["active-source"]
    assert writer._pipeline.extract.call_count == 0


@capture_outbox_events
def test_one_request_accumulates_source_receipts_across_two_batches():
    _, fs, store, ctx, manager, journal = prepare()
    manager.add_message(ctx.session_id, "user", "The antenna was repaired.", ctx,
                        message_id="source-two", created_at="2026-10-29T01:01:00+00:00")
    store_outcome(journal.buffer, "write-one", {"request_status": "ACCEPTED", "source_status": "BUFFERED",
        "source_message_ids": ["source-one", "source-two"]})
    first = journal.finish(outcome={"semantic_status": "COMPLETE"})
    assert first["source_status"] == "BUFFERED"
    assert first["archived_source_ids"] == ["source-one"]
    buffer = manager.get_or_create(ctx.session_id, ctx)
    second = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages), archive_id="archive-two",
        messages=[{"role": "user", "content": buffer.messages[0].content}],
        session_time="2026-10-29T01:01:00+00:00", session_summary="", tool_stats_text="")
    second.finish(outcome={"semantic_status": "COMPLETE"})
    receipt = buffer.runtime_extensions[OUTCOMES_KEY]["entries"]["write-one"]
    assert receipt["archived_source_ids"] == ["source-one", "source-two"]
    assert set(receipt["archive_receipts"]) == {"archive-one", "archive-two"}
    assert projection_outcome(receipt, store)["request_status"] == "SUCCEEDED"


@capture_outbox_events
def test_active_receipts_and_retained_request_ids_survive_receipt_maintenance():
    _, _, _, _, manager, journal = prepare()
    buffer = journal.buffer
    for index in range(270):
        store_outcome(buffer, f"active-{index}", {"request_status": "ACCEPTED", "source_status": "BUFFERED"})
    buffer.mark_request_completed("retained-terminal")
    store_outcome(buffer, "retained-terminal", {"request_status": "SUCCEEDED"})
    store_outcome(buffer, "old-terminal", {"request_status": "SUCCEEDED"})
    entries = buffer.runtime_extensions[OUTCOMES_KEY]["entries"]
    assert len(entries) == 271
    assert all(f"active-{index}" in entries for index in range(270))
    assert entries["retained-terminal"]["request_status"] == "SUCCEEDED"


@pytest.mark.parametrize("projected,state", [(False, "PROJECTING"), (True, "SUCCEEDED")])
@capture_outbox_events
def test_finished_source_checkpoint_persists_exact_projection_state(projected, state):
    _, fs, store, ctx, manager, journal = prepare()
    node = memory_node(ctx)
    fs.write_node(node, ctx)
    event = store.register_write(fs.read_node(node.uri, ctx), ctx)
    journal.mutate(lambda batch: batch.update(outbox_events=[[node.uri, event.event_id]]))
    if projected:
        store.mark_done(event, node.uri)
    receipt = journal.finish(outcome={"semantic_status": "COMPLETE"},
                             resolve_outcome=lambda value: projection_outcome(value, store))
    payload = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert receipt["request_status"] == state
    assert payload["runtime_extensions"][OUTCOMES_KEY]["entries"]["write-one"] == receipt
    assert payload["completed_request_ids"] == ["write-one"]
    assert payload["buffer_messages"] == []


@capture_outbox_events
def test_receipt_window_reconciles_completed_events_and_retains_pending_sources():
    _, fs, store, ctx, manager, journal = prepare()
    buffer = journal.buffer

    def receipt(request_id, projected):
        node = memory_node(ctx, key=request_id)
        fs.write_node(node, ctx)
        event = store.register_write(fs.read_node(node.uri, ctx), ctx)
        if projected:
            store.mark_done(event, node.uri)
        return {"request_id": request_id, "request_status": "PROJECTING", "source_status": "ARCHIVED",
                "semantic_status": "COMPLETE", "projection_status": "UNKNOWN",
                "outbox_tracking_complete": True, "outbox_events": [[node.uri, event.event_id]]}

    store_outcome(buffer, "pending-source", receipt("pending-source", False))
    for index in range(260):
        request_id = f"completed-{index}"
        buffer.mark_request_completed(request_id)
        store_outcome(buffer, request_id, receipt(request_id, True))

    resolved = []
    def resolve(value):
        resolved.append(value["request_id"])
        return projection_outcome(value, store)

    buffer.mark_request_completed("latest-source")
    result = store_outcome(buffer, "latest-source", receipt("latest-source", True), resolve_outcome=resolve)
    entries = buffer.runtime_extensions[OUTCOMES_KEY]["entries"]
    assert result["request_status"] == "SUCCEEDED"
    assert entries["pending-source"]["projection_status"] == "PENDING"
    assert set(entries) == set(buffer.completed_request_ids) | {"pending-source"}
    assert len(entries) == 257
    assert resolved == ["latest-source", "pending-source"] + [f"completed-{index}" for index in range(5)]
    assert manager.save_session_state(ctx.session_id, ctx)
    payload = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert payload["runtime_extensions"][OUTCOMES_KEY]["entries"] == entries


@capture_outbox_events
def test_storing_confirmed_receipt_reuses_its_projection_proof():
    _, fs, store, ctx, _, journal = prepare()
    node = memory_node(ctx)
    fs.write_node(node, ctx)
    event = store.register_write(fs.read_node(node.uri, ctx), ctx)
    store.mark_done(event, node.uri)
    journal.buffer.mark_request_completed("write-one")
    calls = []

    def resolve(value):
        calls.append(value["request_id"])
        return projection_outcome(value, store)

    receipt = {"request_id": "write-one", "request_status": "PROJECTING", "source_status": "ARCHIVED",
               "semantic_status": "COMPLETE", "outbox_tracking_complete": True,
               "outbox_events": [[node.uri, event.event_id]]}
    confirmed = store_outcome(journal.buffer, "write-one", receipt, resolve_outcome=resolve)
    saved = store_outcome(journal.buffer, "write-one", confirmed, resolve_outcome=resolve)
    assert confirmed["request_status"] == "SUCCEEDED"
    assert saved == confirmed
    assert calls == ["write-one"]


@pytest.mark.parametrize("repeats,state", [(1, "DEGRADED"), (50, "SUCCEEDED")])
@capture_outbox_events
def test_buffered_conversation_dispose_preserves_all_roles_and_quality(repeats, state):
    from unittest.mock import Mock
    from tests.unit.server.test_extraction_batch_recovery import setup_service
    _, fs, store = make_runtime()
    ctx = request_context("dispose-write-contract")
    service, writer, manager = setup_service(fs, ctx, outbox=store)
    service._cfg.after_turn_threshold = 100000
    service.get_llm = lambda: None
    service._generate_summary_candidate = Mock(return_value=None)
    service._get_outbox_store = lambda _fs: store
    service._flush_session_tool_usage_async = Mock()
    source = [{"role": "user", "content": "I repaired the receiver. " * repeats},
              {"role": "assistant", "content": "The repair is recorded."}]
    accepted = service.after_turn({"_ctx": ctx, "sessionId": ctx.session_id,
        "clientRequestId": "dispose-source", "messages": source, "wait": True})
    assert accepted["source_status"] == "BUFFERED"
    buffer = manager.get_or_create(ctx.session_id, ctx)
    service._run_dispose_background_impl(manager, ctx.session_id, {"_ctx": ctx}, ctx, buffer)
    receipt = service.get_write_request_status({"_ctx": ctx, "sessionId": ctx.session_id,
                                                "clientRequestId": "dispose-source"})
    assert receipt["source_status"] == "RAW_ARCHIVED"
    archived = "\n".join(fs.read_node(uri, ctx).content for uri in receipt["raw_source_uris"])
    assert all(f"{message['role']}: {message['content']}" in archived for message in source)
    payload = json.loads(fs.read_node(manager._session_state.state_uri(ctx.account_id, ctx.session_id), ctx).content)
    assert payload["buffer_messages"] == []
    saved = payload["runtime_extensions"][OUTCOMES_KEY]["entries"]["dispose-source"]
    assert saved["source_snapshot_sha256"]
    for uri, event in store.list_pending_events([tuple(reference) for reference in saved["outbox_events"]]):
        store.mark_done(event, uri)
    completed = service.get_write_request_status({"_ctx": ctx, "sessionId": ctx.session_id,
                                                  "clientRequestId": "dispose-source"})
    assert completed["request_status"] == state
    assert completed["projection_status"] == "DONE"
    if repeats == 1:
        assert completed["error_detail"]["code"] == "SEMANTIC_EXTRACTION_SKIPPED"
        assert writer._pipeline.extract.call_count == 0
    else:
        assert writer._pipeline.extract.call_count == 1


@capture_outbox_events
def test_http_accepted_recovery_and_readonly_terminal_query_share_contract(monkeypatch):
    import server.app as app_module
    from types import SimpleNamespace
    ctx = request_context("http-write-contract")
    accepted = normalize_write_result({"request_status": "RECOVERING", "source_status": "CHECKPOINTED"}, "write-one")
    complete = normalize_write_result({"request_status": "SUCCEEDED", "source_status": "ARCHIVED",
                                       "semantic_status": "COMPLETE", "projection_status": "DONE"}, "write-one")
    calls = []
    service = SimpleNamespace(after_turn=lambda params: accepted,
                              get_write_request_status=lambda params: calls.append(params) or complete)
    monkeypatch.setattr(app_module, "_get_service", lambda: service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    client = app_module.app.test_client()
    response = client.post("/api/v1/after_turn", json={"sessionId": ctx.session_id,
        "clientRequestId": "write-one", "messages": [{"role": "user", "content": "Remember the receiver repair."}]})
    assert response.status_code == 202
    assert response.json["retry_action"] == "poll"
    result = client.get(f"/api/v1/sessions/{ctx.session_id}/write-requests/write-one")
    assert result.status_code == 200
    assert result.json["request_status"] == "SUCCEEDED"
    assert calls[0]["_ctx"] == ctx
    assert calls[0]["clientRequestId"] == "write-one"


@pytest.mark.asyncio
async def test_mcp_status_tool_uses_shared_request_query():
    from mcp import Client
    from server.mcp_server import create_mcp_server
    from tests.unit.server.test_mcp_tools_write import _FakeService
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True))
    calls = []
    receipt = normalize_write_result({"request_status": "SUCCEEDED", "source_status": "ARCHIVED",
                                       "semantic_status": "COMPLETE", "projection_status": "DONE"}, "write-one")
    service.get_write_request_status = lambda params: calls.append(params) or receipt
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        result = await client.call_tool("ogm_get_write_request_status", {
            "session_id": "write-contract-session", "client_request_id": "write-one"})
    assert result.is_error is False
    assert calls[0]["clientRequestId"] == "write-one"
    assert calls[0]["_ctx"].session_id == "write-contract-session"
