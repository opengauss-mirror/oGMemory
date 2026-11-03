"""Public-entry recovery tests use deterministic candidates and no model/service."""

import json
import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError
from core.models import CandidateMemory, RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import MemoryFS


class NoModel:
    def __getattr__(self, name):
        raise AssertionError(f"Recovery must not call a model: {name}")


def candidate(category="event", *, body="I repaired the receiver.", **fields):
    return CandidateMemory(category=category, owner_scope="user", routing_key="receiver_repair",
                           abstract=body, overview=body, content=body, title="Receiver repair",
                           when="2026-09-30", who="Nora", confidence=.9,
                           provenance_ids=["prov:1:archive:will-be-bound:msg-one"], **fields)


@pytest.fixture
def ctx():
    return RequestContext(account_id="recovery-test", user_id="reader", agent_id="agent",
                          session_id="batch-session", trace_id="proof-trace")


def setup_service(fs, ctx, *, entries=None, extract=None, outbox=None):
    writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    if extract is None:
        def extract(**kwargs):
            return [replace(entry, provenance_ids=[f"prov:1:archive:{kwargs['archive_id']}:msg-one"])
                    for entry in (entries or [candidate()])]
    writer._pipeline.extract = Mock(side_effect=lambda *args, **kwargs: extract(**kwargs))
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._cfg.after_turn_threshold = 1
    service._write_api = writer
    service._generate_and_write_summary = Mock()
    service.drain_outbox_sync = Mock(return_value={})
    service._async_drain = Mock()
    manager = service.get_session_manager()
    manager.commit_snapshot = Mock(return_value={"ok": True, "status": "completed", "archived": True})
    return service, writer, manager


def params(ctx, request="original", content="I repaired the receiver."):
    return {"sessionId": ctx.session_id, "clientRequestId": request,
            "messages": [{"role": "user", "content": content}], "wait": True, "_ctx": ctx}


def nodes(fs, category):
    return [node for node in fs.nodes.values() if node.category == category]


def test_user_only_turn_skips_agent_memory_and_completes_once(ctx):
    ctx = replace(ctx, agent_id="")
    fs = MemoryFS()
    skill = replace(candidate("skill"), owner_scope="agent")
    service, writer, manager = setup_service(fs, ctx, entries=[skill])

    result = service.after_turn(params(ctx))

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert not nodes(fs, "skill")
    assert len(nodes(fs, "session_archive")) == 1
    buffer = manager.get_or_create(ctx.session_id)
    assert not buffer.messages
    assert not buffer.pending_extraction_ids()
    assert not buffer.runtime_extensions.get(EXTENSION_KEY)
    assert buffer.request_completed("original")
    assert service.after_turn(params(ctx))["status"] == "duplicate"
    manager.commit_snapshot.assert_called_once()
    writer._pipeline.extract.assert_called_once()


@pytest.mark.parametrize("legacy_writing", [False, True])
def test_agent_skip_recovers_after_result_checkpoint_failure(ctx, legacy_writing):
    ctx = replace(ctx, agent_id="")
    fs = MemoryFS()
    skill = replace(candidate("skill"), owner_scope="agent")
    service, writer, manager = setup_service(fs, ctx, entries=[skill])
    save = manager.save_session_state
    failed = []

    def fail_skip_completion_once(session_id, request):
        batch = manager.get_or_create(session_id).runtime_extensions.get(EXTENSION_KEY, {}).get("batch", {})
        completed_skip = any(
            operation.get("plan", {}).get("action") == "skip"
            for slot in batch.get("slots", {}).values()
            for operation in slot.get("operations", {}).values())
        if completed_skip and not failed:
            failed.append(True)
            return False
        return save(session_id, request)

    manager.save_session_state = fail_skip_completion_once
    assert service.after_turn(params(ctx))["status"] == "failed"
    assert failed and not nodes(fs, "skill")
    state = nodes(fs, "state")[0]
    payload = json.loads(state.content)
    batch = payload["runtime_extensions"][EXTENSION_KEY]["batch"]
    slot = next(slot for slot in batch["slots"].values() if slot["candidates"][0]["category"] == "skill")
    if legacy_writing:
        # Older writers persisted 'writing' before recognizing this no-write case.
        slot["operations"]["0"] = {"status": "writing"}
        state.content = json.dumps(payload)
        fs.nodes[state.uri] = state
    else:
        assert slot["operations"].get("0", {}).get("status") != "writing"

    restarted, next_writer, next_manager = setup_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted skip was re-extracted"))
    result = restarted.after_turn(params(ctx))
    assert result["status"] == "completed"
    assert not nodes(fs, "skill")
    assert len(nodes(fs, "session_archive")) == 1
    buffer = next_manager.get_or_create(ctx.session_id)
    assert not buffer.messages and not buffer.pending_extraction_ids()
    assert not buffer.runtime_extensions.get(EXTENSION_KEY)
    assert buffer.request_completed("original")
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
    writer._pipeline.extract.assert_called_once()
    next_writer._pipeline.extract.assert_not_called()


def test_uncertain_agent_write_cannot_be_recovered_as_missing_agent_skip(ctx):
    fs = MemoryFS()
    skill = replace(candidate("skill"), owner_scope="agent")
    service, writer, _ = setup_service(fs, ctx, entries=[skill])
    writer._writer.write_candidate = Mock(side_effect=SystemExit("lost before write acknowledgement"))
    with pytest.raises(SystemExit):
        service.after_turn(params(ctx))
    restarted, next_writer, _ = setup_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted candidate was re-extracted"))
    assert restarted.after_turn(params(replace(ctx, agent_id="")))["status"] == "needs_confirmation"
    result = restarted.after_turn(params(ctx))
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    assert not nodes(fs, "skill")
    next_writer._pipeline.extract.assert_not_called()


def test_early_recovery_uses_the_configured_session_backend(ctx):
    fs = MemoryFS()
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False,
                                             after_turn_threshold=999_999))
    service._session_mgr = SessionManager(get_context_fs=lambda: fs,
                                         session_state_bridge_enabled=False)
    service._get_context_fs = Mock(side_effect=AssertionError("default backend must not be read"))

    result = service.after_turn(params(ctx))

    assert result["ok"] is True
    assert result["status"] == "accumulating"
    assert service._session_mgr.extraction_recovery_loaded(ctx.session_id, ctx)
    assert [message.content for message in service._session_mgr.get_or_create(ctx.session_id).messages] == [
        "I repaired the receiver."]
    service._get_context_fs.assert_not_called()


def test_public_partial_rejection_reuses_accepted_candidates_after_restart(ctx):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[candidate(), candidate("entity")])
    original = writer._writer.write_candidate
    rejected = False

    def reject_once(entry, request):
        nonlocal rejected
        if entry.category == "entity" and not rejected:
            rejected = True
            raise WriteNotStartedError("deterministic planning rejection before storage")
        return original(entry, request)

    writer._writer.write_candidate = reject_once
    first = service.after_turn(params(ctx))
    assert first["status"] == "failed"
    old_batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    archive_id = old_batch["archive_id"]
    assert len(nodes(fs, "event")) == 1

    restarted, next_writer, next_manager = setup_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted candidates were re-extracted"))
    retried = restarted.after_turn(params(ctx))
    assert retried["status"] == "completed"
    assert len(nodes(fs, "event")) == len(nodes(fs, "entity")) == 1
    assert nodes(fs, "event")[0].metadata["provenance_ids"] == [f"prov:1:archive:{archive_id}:msg-one"]
    next_writer._pipeline.extract.assert_not_called()
    assert next_manager.get_or_create(ctx.session_id).request_completed("original")


class CommitThenFailFS(MemoryFS):
    def __init__(self):
        super().__init__()
        self.once = True
        self.unread_receipt_uri = None

    def read_node(self, uri, ctx):
        if uri == self.unread_receipt_uri:
            self.unread_receipt_uri = None
            raise ConnectionError("receipt unavailable after connection loss")
        return super().read_node(uri, ctx)

    def write_node(self, node, ctx):
        super().write_node(node, ctx)
        if node.category == "event" and self.once:
            self.once = False
            self.unread_receipt_uri = node.uri
            raise RuntimeError("connection lost after durable commit")


def test_committed_write_without_acknowledgement_recovers_only_with_receipt(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
    assert len(nodes(fs, "event")) == 1
    assert nodes(fs, "event")[0].metadata[RECEIPTS_KEY]
    restarted, writer, _ = setup_service(fs, ctx, extract=lambda **kwargs: pytest.fail("model resumed"))
    assert restarted.after_turn(params(ctx))["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


def test_crash_after_writing_checkpoint_before_node_write_fails_closed(ctx):
    fs = MemoryFS()
    service, writer, _ = setup_service(fs, ctx)
    writer._writer.write_candidate = Mock(side_effect=SystemExit("process terminated before storage"))
    with pytest.raises(SystemExit):
        service.after_turn(params(ctx))
    assert not nodes(fs, "event")
    restarted, next_writer, _ = setup_service(fs, ctx, extract=lambda **kwargs: pytest.fail("model resumed"))
    result = restarted.after_turn(params(ctx))
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    assert not nodes(fs, "event")
    next_writer._pipeline.extract.assert_not_called()


@pytest.mark.parametrize("remove", ["delete", "archive"])
def test_unacknowledged_committed_then_removed_node_is_not_recreated(ctx, remove):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    uri = nodes(fs, "event")[0].uri
    if remove == "archive":
        fs.archive_node(uri, ctx)
    else:
        fs.nodes.pop(uri)
    restarted, _, _ = setup_service(fs, ctx, extract=lambda **kwargs: pytest.fail("model resumed"))
    assert restarted.after_turn(params(ctx))["status"] == "needs_confirmation"
    assert not nodes(fs, "event")


def test_storage_error_without_commit_is_not_guessed_retryable(ctx):
    class UncertainFS(MemoryFS):
        def write_node(self, node, request):
            if node.category == "event":
                raise TimeoutError("unknown storage result")
            super().write_node(node, request)

    fs = UncertainFS()
    service, _, _ = setup_service(fs, ctx)
    assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
    restarted, writer, _ = setup_service(fs, ctx)
    assert restarted.after_turn(params(ctx))["status"] == "needs_confirmation"
    writer._pipeline.extract.assert_not_called()
    assert not nodes(fs, "event")


@pytest.mark.parametrize("fail_at", [1, 3])
def test_checkpoint_failure_prevents_first_memory_write(ctx, fail_at):
    class FailStateFS(MemoryFS):
        saves = 0

        def write_node(self, node, request):
            if node.category == "state":
                self.saves += 1
                if self.saves == fail_at:
                    raise RuntimeError("state checkpoint unavailable")
            super().write_node(node, request)

    fs = FailStateFS()
    service, _, _ = setup_service(fs, ctx)
    assert service.after_turn(params(ctx))["status"] == "failed"
    assert not nodes(fs, "event")


def test_receipt_recovers_node_commit_before_result_checkpoint(ctx):
    class FailResultFS(MemoryFS):
        failed = False

        def write_node(self, node, request):
            if node.category == "state":
                batch = json.loads(node.content).get("runtime_extensions", {}).get(EXTENSION_KEY, {}).get("batch", {})
                operation = batch.get("slots", {}).get("chunk:0", {}).get("operations", {}).get("0", {})
                if operation.get("status") == "complete" and not self.failed:
                    self.failed = True
                    raise RuntimeError("result checkpoint unavailable")
            super().write_node(node, request)

    fs = FailResultFS()
    service, _, _ = setup_service(fs, ctx)
    assert service.after_turn(params(ctx))["status"] == "failed"
    assert len(nodes(fs, "event")) == 1
    restarted, writer, _ = setup_service(fs, ctx, extract=lambda **kwargs: pytest.fail("model resumed"))
    assert restarted.after_turn(params(ctx))["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


def test_new_tail_survives_original_batch_recovery(ctx):
    fs = CommitThenFailFS()
    service, writer, manager = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    manager.add_message(ctx.session_id, "user", "A later independent fact.", ctx, message_id="new-tail")
    original_message_ids = set(manager.get_or_create(ctx.session_id).pending_extraction_ids())
    assert service.after_turn(params(ctx))["status"] == "completed"
    buffer = manager.get_or_create(ctx.session_id)
    assert [message.id for message in buffer.messages] == ["new-tail"]
    assert buffer.extraction_watermark == 0
    assert not original_message_ids & {message.id for message in buffer.messages}
    assert not buffer.pending_extraction_ids()
    assert writer._pipeline.extract.call_count == 1


def test_overlapping_requests_do_not_claim_or_extract_original_batch_twice(ctx):
    started, release = threading.Event(), threading.Event()

    def slow_extract(**kwargs):
        started.set()
        assert release.wait(5)
        return [candidate()]

    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, extract=slow_extract)
    result = []
    thread = threading.Thread(target=lambda: result.append(service.after_turn(params(ctx))))
    thread.start()
    assert started.wait(5)
    overlap = service.after_turn(params(ctx, request="tail-request", content="A later independent fact."))
    assert overlap["status"] == "processing" and overlap["retryable"] is True
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result[0]["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    assert writer._pipeline.extract.call_count == 1
    assert [message.content for message in manager.get_or_create(ctx.session_id).messages] == ["A later independent fact."]


def test_pending_batch_source_is_immutable_and_other_lifecycle_paths_fail_closed(ctx):
    fs = CommitThenFailFS()
    service, _, manager = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    buffer = manager.get_or_create(ctx.session_id)
    message = buffer.messages[0]
    with pytest.raises(ValueError, match="extraction snapshot"):
        buffer.add("user", "Changed facts.", message_id=message.id)
    with pytest.raises(ValueError, match="extraction snapshot"):
        buffer.remove_unextracted(message.id)
    assert service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "needs_confirmation"
    assert service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})["disposed"] is False
    assert buffer.messages


def test_corrupt_operation_identity_is_rejected_without_writing(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    state = nodes(fs, "state")[0]
    payload = json.loads(state.content)
    record = payload["runtime_extensions"][EXTENSION_KEY]["batch"]["slots"]["chunk:0"]
    record["candidates"][0]["write_operation_id"] = "different"
    state.content = json.dumps(payload)
    fs.nodes[state.uri] = state
    restarted, _, _ = setup_service(fs, ctx)
    assert restarted.after_turn(params(ctx))["status"] == "needs_confirmation"
    assert len(nodes(fs, "event")) == 1


def test_raw_chunk_storage_recovers_before_semantic_extraction(ctx):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    original = writer._writer.write_candidate

    def reject_raw(entry, request):
        if entry.category == "session_archive":
            raise WriteNotStartedError("raw archive rejected before storage")
        return original(entry, request)

    writer._writer.write_candidate = reject_raw
    for _ in range(2):
        assert service.after_turn(params(ctx))["status"] == "failed"
        assert manager.get_or_create(ctx.session_id).pending_extraction_ids()
        assert not nodes(fs, "session_archive")
    writer._writer.write_candidate = original
    assert service.after_turn(params(ctx))["status"] == "completed"
    assert len(nodes(fs, "session_archive")) == 1
    assert len(nodes(fs, "event")) == 1
    assert writer._pipeline.extract.call_count == 1


def test_ingest_handoff_does_not_add_the_same_source_again_with_request_id(ctx):
    fs = MemoryFS()
    service, _, manager = setup_service(fs, ctx)
    service._cfg.after_turn_threshold = 10000
    service.ingest({"sessionId": ctx.session_id, "messages": [{"role": "user", "content": "I repaired the receiver."}],
                    "_ctx": ctx})
    assert service.after_turn(params(ctx))["status"] == "accumulating"
    assert len(manager.get_or_create(ctx.session_id).messages) == 1


def test_same_pending_request_id_cannot_change_source_facts(ctx):
    fs = CommitThenFailFS()
    service, _, manager = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    original_sources = list(manager.get_or_create(ctx.session_id).messages)
    response = service.after_turn(params(ctx, content="I did not repair the receiver."))
    assert response["status"] == "needs_confirmation"
    assert manager.get_or_create(ctx.session_id).messages == original_sources
    assert len(nodes(fs, "event")) == 1


def lazy_service(fs, ctx):
    service, writer, _ = setup_service(fs, ctx, extract=lambda **kwargs: pytest.fail("model resumed"))
    service._session_mgr = None
    service._write_api = None
    service._get_context_fs = Mock(return_value=fs)
    service._cfg.after_turn_threshold = 10000
    service.get_llm = Mock(side_effect=AssertionError("lazy recovery initialized a model"))
    service._get_shared_embedder = Mock(side_effect=AssertionError("lazy recovery initialized an embedder"))

    def initialize():
        service._write_api = writer
        return writer

    service.get_write_api = Mock(side_effect=initialize)
    manager = service.get_session_manager()
    manager.commit_snapshot = Mock(return_value={"ok": True, "status": "completed", "archived": True})
    return service, writer, manager


def test_lazy_writer_restart_loads_pending_before_threshold_or_accumulation(ctx):
    fs = CommitThenFailFS()
    original, _, _ = setup_service(fs, ctx)
    assert original.after_turn(params(ctx))["status"] == "needs_confirmation"
    service, writer, manager = lazy_service(fs, ctx)
    assert service.after_turn(params(ctx))["status"] == "completed"
    assert not manager.get_or_create(ctx.session_id).messages
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()
    service.get_llm.assert_not_called()
    service._get_shared_embedder.assert_not_called()


def test_lazy_load_is_once_and_ingest_preserves_explicit_source_id(ctx):
    fs = MemoryFS()
    service, _, manager = lazy_service(fs, ctx)
    assert service.ingest({"sessionId": ctx.session_id, "messages": [
        {"id": "stable-source", "role": "user", "content": "An independent fact."}], "_ctx": ctx})["ingested"]
    assert manager.get_or_create(ctx.session_id).messages[0].id == "stable-source"
    assert service.after_turn(params(ctx, request="turn-two", content="A second fact."))["status"] == "accumulating"
    assert service.after_turn(params(ctx, request="turn-three", content="A third fact."))["status"] == "accumulating"
    service._get_context_fs.assert_called_once()
    service.get_write_api.assert_not_called()


@pytest.mark.parametrize("entry", ["after_turn", "ingest"])
def test_initial_durable_read_error_preserves_buffer_and_stops_accumulation(ctx, entry):
    class UnreadableFS(MemoryFS):
        def read_node(self, uri, request):
            raise TimeoutError("durable journal unavailable")

    fs = UnreadableFS()
    service, _, manager = lazy_service(fs, ctx)
    manager.add_message(ctx.session_id, "user", "Existing message.", ctx, message_id="existing")
    prior = list(manager.get_or_create(ctx.session_id).messages)
    response = getattr(service, entry)(params(ctx))
    assert response["status"] == "needs_confirmation" and response["retryable"] is False
    assert manager.get_or_create(ctx.session_id).messages == prior
    assert not manager.extraction_recovery_loaded(ctx.session_id, ctx)
    service.get_write_api.assert_not_called()
    assert not fs.nodes


def test_ingest_pending_replay_requires_proven_source_identity(ctx):
    fs = CommitThenFailFS()
    original, _, _ = setup_service(fs, ctx)
    original.after_turn(params(ctx))
    service, _, manager = lazy_service(fs, ctx)
    assert service.ingest(params(ctx))["count"] == 0
    no_identity = params(ctx)
    no_identity.pop("clientRequestId")
    assert service.ingest(no_identity)["status"] == "needs_confirmation"
    assert len(manager.get_or_create(ctx.session_id).messages) == 1
    assert service.after_turn(params(ctx))["status"] == "completed"
    assert not manager.get_or_create(ctx.session_id).messages


def test_new_request_tail_reports_accumulation_after_original_recovery(ctx):
    fs = CommitThenFailFS()
    original, _, _ = setup_service(fs, ctx)
    original.after_turn(params(ctx))
    service, writer, manager = lazy_service(fs, ctx)
    response = service.after_turn(params(ctx, request="new-turn", content="A later independent fact."))
    assert response["status"] == "accumulating"
    assert [message.content for message in manager.get_or_create(ctx.session_id).messages] == ["A later independent fact."]
    assert manager.get_or_create(ctx.session_id).extraction_watermark == 0
    writer._pipeline.extract.assert_not_called()


def test_first_read_cannot_load_another_session_owner_even_with_bridge_off(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    restarted, _, manager = lazy_service(fs, ctx)
    other = replace(ctx, user_id="other-reader")
    assert restarted.after_turn(params(other))["status"] == "needs_confirmation"
    assert not manager.extraction_recovery_loaded(ctx.session_id, other)
    assert len(nodes(fs, "event")) == 1


def test_initialized_writer_strict_load_cannot_expose_another_owners_source(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    other = replace(ctx, user_id="other-reader")
    restarted, _, manager = setup_service(fs, other)
    assert restarted.after_turn(params(other))["status"] == "needs_confirmation"
    buf = manager.get_or_create(ctx.session_id, ctx=other)
    assert not buf.messages and not buf.runtime_extensions
    assert not manager.extraction_recovery_loaded(ctx.session_id, other)
    assert restarted.after_turn(params(ctx))["status"] == "completed"
    assert manager.extraction_recovery_loaded(ctx.session_id, ctx)


def test_initial_source_collision_is_rejected_before_replacing_current_buffer(ctx):
    fs = CommitThenFailFS()
    service, _, original_manager = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    source_id = original_manager.get_or_create(ctx.session_id).messages[0].id
    restarted, _, manager = lazy_service(fs, ctx)
    manager.add_message(ctx.session_id, "user", "Changed current content.", ctx, message_id=source_id)
    before = list(manager.get_or_create(ctx.session_id).messages)
    assert restarted.after_turn(params(ctx))["status"] == "needs_confirmation"
    assert manager.get_or_create(ctx.session_id, ctx=ctx).messages == before
    assert not manager.extraction_recovery_loaded(ctx.session_id, ctx)


def test_distinct_request_same_text_remains_a_new_source_after_pending_recovery(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    restarted, writer, manager = lazy_service(fs, ctx)
    response = restarted.after_turn(params(ctx, request="new-source-request"))
    assert response["status"] == "accumulating"
    buf = manager.get_or_create(ctx.session_id)
    assert [message.content for message in buf.messages] == ["I repaired the receiver."]
    assert buf.extraction_watermark == 0 and len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


def test_distinct_buffered_requests_same_text_do_not_use_legacy_ingest_dedup(ctx):
    service, _, manager = lazy_service(MemoryFS(), ctx)
    assert service.after_turn(params(ctx, request="first-source"))["status"] == "accumulating"
    assert service.after_turn(params(ctx, request="second-source"))["status"] == "accumulating"
    assert len(manager.get_or_create(ctx.session_id).messages) == 2


def test_invalid_durable_source_shape_is_rejected_before_new_tail_accumulation(ctx):
    fs = CommitThenFailFS()
    service, _, _ = setup_service(fs, ctx)
    service.after_turn(params(ctx))
    state = nodes(fs, "state")[0]
    payload = json.loads(state.content)
    payload["buffer_messages"][0]["created_at"] = None
    payload["runtime_extensions"][EXTENSION_KEY]["batch"]["snapshot"][0]["created_at"] = None
    state.content = json.dumps(payload)
    fs.nodes[state.uri] = state
    restarted, _, manager = lazy_service(fs, ctx)
    response = restarted.after_turn(params(ctx, request="new-tail", content="A later fact."))
    assert response["status"] == "needs_confirmation"
    assert not manager.get_or_create(ctx.session_id, ctx=ctx).messages
    assert not manager.extraction_recovery_loaded(ctx.session_id, ctx)


def test_recovery_and_concurrent_append_preserve_sources(ctx):
    import threading

    fs = MemoryFS()
    service, _, manager = setup_service(fs, ctx)
    manager.add_message(ctx.session_id, "user", "Retain the source", ctx, message_id="source")
    manager._get_context_fs = lambda: fs
    assert manager.save_session_state(ctx.session_id, ctx)
    _, _, restored = setup_service(fs, ctx)
    restored.get_or_create(ctx.session_id, ctx)
    restore = restored._restore_runtime_state_from_payload
    restoring = threading.Event()
    appending = threading.Event()
    owned = []
    errors = []

    def atomic_restore(*args):
        owned.append(restored._lock._is_owned())
        restoring.set()
        assert appending.wait(2)
        return restore(*args)

    def reload():
        restored.enable_extraction_recovery(ctx.session_id, ctx, fs)

    def append():
        assert restoring.wait(2)
        appending.set()
        restored.add_message(ctx.session_id, "user", "Append after restore", ctx, message_id="tail")

    def run(operation):
        try:
            operation()
        except Exception as exc:
            errors.append(exc)

    restored._restore_runtime_state_from_payload = atomic_restore
    workers = [threading.Thread(target=run, args=(operation,), daemon=True) for operation in (reload, append)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(3)
    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert owned == [True]
    assert [m.id for m in restored.get_or_create(ctx.session_id, ctx).messages] == ["source", "tail"]


def test_ingest_request_handoff_keeps_one_source(ctx):
    fs = MemoryFS()
    service, _, manager = setup_service(fs, ctx)
    service._cfg.after_turn_threshold = 10000
    request = params(ctx)
    assert service.ingest(request)["ingested"]
    handoff = {key: value for key, value in request.items() if key != "clientRequestId"}
    assert service.after_turn(handoff)["ok"]
    assert len(manager.get_or_create(ctx.session_id, ctx).messages) == 1


def test_unidentified_replay_retains_original_recovery_source(ctx):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    extract = writer._pipeline.extract.side_effect
    writer._pipeline.extract.side_effect = OSError("temporary extraction outage")
    request = {key: value for key, value in params(ctx).items() if key != "clientRequestId"}
    assert service.after_turn(request)["ok"] is False
    original = list(manager.get_or_create(ctx.session_id, ctx).messages)
    writer._pipeline.extract.side_effect = extract
    retried = service.after_turn(request)
    assert retried["status"] == "needs_confirmation"
    assert manager.get_or_create(ctx.session_id, ctx).messages == original
    request["messages"][0]["id"] = original[0].id
    assert service.after_turn(request)["ok"]
    assert manager.get_or_create(ctx.session_id, ctx).messages == []


def test_dispose_waits_for_normal_non_sql_extraction(ctx):
    import threading
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    service.wake_skill_sedimentation = Mock()
    service._flush_session_tool_usage_async = Mock()
    entered, release = threading.Event(), threading.Event()
    extract = writer._pipeline.extract.side_effect

    def slow_extract(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return extract(*args, **kwargs)

    writer._pipeline.extract.side_effect = slow_extract
    request = {**params(ctx), "wait": False}
    request.pop("clientRequestId")
    service.after_turn(request)
    assert entered.wait(3)
    try:
        disposed = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert disposed["status"] == "dispose_deferred"
    finally:
        release.set()
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
        "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})["idle"]
    assert manager.get_or_create(ctx.session_id, ctx).messages == []


def test_spawn_failure_completes_claimed_source_synchronously(ctx, monkeypatch):
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    request = {key: value for key, value in params(ctx).items() if key != "clientRequestId"}
    request["wait"] = False
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("thread capacity exhausted")))
    result = service.after_turn(request)
    assert result["status"] == "completed"
    assert result["ok"]
    writer._pipeline.extract.assert_called_once()
    manager.commit_snapshot.assert_called_once()
    buf = manager.get_or_create(ctx.session_id, ctx)
    assert not buf.messages
    assert not buf.pending_extraction_ids()
    assert buf.extraction_active_count == 0
