"""Lifecycle recovery uses public entries, deterministic candidates and no model."""

import json
import threading
from copy import deepcopy
from unittest.mock import Mock

import pytest

from commit.write_receipts import WriteNotStartedError
from core.models import RequestContext
from service.extraction_checkpoint import EXTENSION_KEY, RecoveryNeedsConfirmationError
from session.archive_store import SessionArchiveStore
from session.session_manager import SessionManager, SessionOperationInProgress
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import (
    CommitThenFailFS,
    candidate,
    nodes,
    params,
    setup_service,
)


@pytest.fixture
def ctx():
    return RequestContext("lifecycle-test", "reader", "agent", "session", "proof")


def lifecycle_service(fs, ctx, **kwargs):
    service, writer, manager = setup_service(fs, ctx, **kwargs)
    service._get_context_fs = Mock(return_value=fs)
    service._get_archive_store = Mock(return_value=None)
    manager._get_llm = lambda: None
    manager._get_agfs = lambda: fs
    manager._get_archive_store = lambda: None
    return service, writer, manager


def reject_entity_once(writer):
    original = writer._writer.write_candidate
    rejected = False

    def write(entry, ctx):
        nonlocal rejected
        if entry.category == "entity" and not rejected:
            rejected = True
            raise WriteNotStartedError("known rejection before storage")
        return original(entry, ctx)

    writer._writer.write_candidate = write


def seed(manager, ctx, content="I repaired the receiver. " * 40):
    manager.add_message(ctx.session_id, "user", content, ctx, message_id="msg-one")


class InlineThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def test_first_dispose_binds_lazily_initialized_writer_storage(ctx, monkeypatch):
    fs = MemoryFS()
    service, writer, _ = lifecycle_service(fs, ctx)
    service._write_api = None
    service._session_mgr = None
    service._get_outbox_store = Mock(return_value=object())
    service._get_shared_vector_index = Mock(return_value=None)
    service._get_shared_embedder = Mock(return_value=None)
    service.get_llm = Mock(return_value=None)
    service._start_outbox_worker = Mock()
    service._flush_session_tool_usage_async = Mock()
    monkeypatch.setattr("server.memory_service.MemoryWriteAPI", Mock(return_value=writer))
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    request = {"sessionId": ctx.session_id, "_ctx": ctx}
    assert service.add_session_message({**request, "role": "user", "content": "I repaired the receiver. " * 50})["ok"]
    manager = service.get_session_manager()

    result = service.dispose(request)

    assert result.get("status") == "background_flush_dispatched", result
    assert len(nodes(fs, "event")) == 1
    assert json.loads(nodes(fs, "state")[0].content)["buffer_messages"] == []
    assert manager._get_context_fs() is fs


def test_dispose_captures_tail_when_ordinary_extraction_finishes_after_prepare(ctx, monkeypatch):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    buf = manager.get_or_create(ctx.session_id, ctx)
    buf.extraction_watermark = len(buf.messages)
    buf.begin_extraction()
    tail = "The antenna needs replacement. " * 50
    manager.add_message(ctx.session_id, "user", tail, ctx, message_id="tail")
    prepare = service._prepare_dispose_source

    def finish_ordinary(*args):
        result = prepare(*args)
        buf.remove_messages_by_id({"msg-one"})
        buf.end_extraction()
        return result

    service._prepare_dispose_source = finish_ordinary
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})

    assert result.get("status") == "background_flush_dispatched", result
    assert writer._pipeline.extract.call_args.args[0][0]["content"] == tail
    assert len(nodes(fs, "event")) == 1
    assert json.loads(nodes(fs, "state")[0].content)["buffer_messages"] == []


@pytest.mark.parametrize("fault", ["source_parse", "fast_path_save", "thread_start", "worker_entry"])
def test_dispose_exception_releases_only_its_owned_operation(ctx, monkeypatch, fault):
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx, content="short pending fact" if fault == "fast_path_save" else "A pending fact. " * 80)
    buffer = manager.begin_managed_operation(ctx.session_id, ctx)
    assert buffer.active_operation_count == 1
    try:
        if fault == "source_parse":
            service._prepare_dispose_source = Mock(return_value={})
            with pytest.raises(KeyError, match="source_snapshot"):
                service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        elif fault == "fast_path_save":
            service._save_session_state_best_effort = Mock(side_effect=ValueError("save failure"))
            with pytest.raises(ValueError, match="save failure"):
                service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        else:
            if fault == "thread_start":
                monkeypatch.setattr("server.memory_service.threading.Thread",
                                    Mock(side_effect=RuntimeError("thread creation failure")))
            else:
                monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
                service._run_dispose_background_impl = Mock(side_effect=RuntimeError("worker entry failure"))
            assert service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})["reason"] == "thread_unavailable"
        assert buffer.active_operation_count == 1
        assert not buffer.dispose_in_progress
        assert not buffer.lifecycle_prepare_in_progress and not buffer.lifecycle_source_ids
    finally:
        buffer.end_operation()


@pytest.mark.parametrize("invalid_kind", [False, True])
def test_dispose_preclaim_exit_preserves_an_existing_worker_owner(ctx, monkeypatch, invalid_kind):
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    buffer = manager.begin_managed_operation(ctx.session_id, ctx)
    buffer.dispose_in_progress = True
    try:
        if invalid_kind:
            monkeypatch.setattr("server.memory_service.lifecycle_kind",
                                Mock(side_effect=RecoveryNeedsConfirmationError("invalid checkpoint")))
            with pytest.raises(RecoveryNeedsConfirmationError, match="invalid checkpoint"):
                service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        else:
            assert service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})["reason"] == "dispose_in_progress"
        assert buffer.active_operation_count == 1
        assert buffer.dispose_in_progress
        assert not buffer.lifecycle_prepare_in_progress and not buffer.lifecycle_source_ids
    finally:
        buffer.dispose_in_progress = False
        buffer.end_operation()


def real_archives(fs, ctx, **kwargs):
    service, writer, manager = lifecycle_service(fs, ctx, **kwargs)
    store = SessionArchiveStore(fs)
    service._get_archive_store = Mock(return_value=store)
    manager._get_archive_store = lambda: store
    manager._archive_store_required = True
    manager.commit_snapshot = SessionManager.commit_snapshot.__get__(manager)
    return service, writer, manager


@pytest.mark.parametrize("entry", ["public_commit", "manager_commit", "snapshot_and_clear"])
@pytest.mark.parametrize("failure", ["extraction", "partial_write"])
def test_ordinary_pending_batch_blocks_source_clear_until_recovered(ctx, entry, failure):
    fs = MemoryFS()
    service, writer, manager = real_archives(fs, ctx, entries=[candidate(), candidate("entity")])
    if failure == "extraction":
        writer._pipeline.extract.side_effect = RuntimeError("extraction interrupted")
    else:
        reject_entity_once(writer)
    assert service.after_turn(params(ctx))["status"] == "failed"
    buffer = manager.get_or_create(ctx.session_id)
    source_before = buffer.capture_rollback_state()
    extension_before = deepcopy(buffer.runtime_extensions)
    nodes_before = deepcopy(fs.nodes)
    tasks_before = deepcopy(manager._tasks)
    assert buffer.pending_extraction_ids()
    assert "lifecycle" not in buffer.runtime_extensions[EXTENSION_KEY]["batch"]

    with pytest.raises(SessionOperationInProgress) as conflict:
        if entry == "public_commit":
            service.commit_session_buffer({"sessionId": ctx.session_id, "_ctx": ctx, "wait": True})
        elif entry == "manager_commit":
            manager.commit(ctx.session_id, ctx, wait=True)
        else:
            buffer.snapshot_and_clear()
    assert conflict.value.conflict_type == "extraction_source_reserved"
    assert conflict.value.retryable is True
    assert buffer.capture_rollback_state() == source_before
    assert buffer.runtime_extensions == extension_before
    assert fs.nodes == nodes_before
    assert manager._tasks == tasks_before
    assert not buffer.commit_in_progress and buffer.active_operation_count == 0
    assert not nodes(fs, "history")

    kwargs = ({"extract": lambda **kw: pytest.fail("accepted candidates were re-extracted")}
              if failure == "partial_write" else {"entries": [candidate(), candidate("entity")]})
    restarted, next_writer, next_manager = real_archives(fs, ctx, **kwargs)
    assert restarted.after_turn(params(ctx))["status"] == "completed"
    restored = next_manager.get_or_create(ctx.session_id)
    assert not restored.messages and not restored.runtime_extensions.get(EXTENSION_KEY)
    assert restored.request_completed("original")
    assert len(nodes(fs, "event")) == len(nodes(fs, "entity")) == len(nodes(fs, "history")) == 1
    if failure == "partial_write":
        next_writer._pipeline.extract.assert_not_called()
        event = nodes(fs, "event")[0]
        assert event == nodes_before[event.uri]
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
    assert restarted.ingest({"sessionId": ctx.session_id, "_ctx": ctx,
        "messages": [{"role": "user", "content": "A later message."}]})["ingested"] is True
    committed = restarted.commit_session_buffer({"sessionId": ctx.session_id, "_ctx": ctx, "wait": True})
    assert committed["archived"] is True and committed["status"] == "completed"
    assert not restored.messages and len(nodes(fs, "history")) == 2


@pytest.mark.parametrize("entry", ["after_turn", "prepare_compaction", "dispose"])
def test_initial_receipt_read_failure_can_retry_after_restart(ctx, entry, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)

    class ReadFailureFS(MemoryFS):
        failed = False

        def list_children(self, uri, request):
            if uri.endswith("/memories/events") and not self.failed:
                self.failed = True
                raise OSError("transient receipt directory read failure")
            return super().list_children(uri, request)

    fs = ReadFailureFS()
    service, _, manager = lifecycle_service(fs, ctx)
    if entry == "after_turn":
        assert service.after_turn(params(ctx))["status"] == "failed"
    else:
        seed(manager, ctx)
        if entry == "prepare_compaction":
            with pytest.raises(RuntimeError, match="compaction extraction write failed"):
                service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
        else:
            service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert fs.failed and not nodes(fs, "event")
    restarted, writer, _ = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted candidate re-extracted"))
    result = getattr(restarted, entry)(params(ctx) if entry == "after_turn"
                                     else {"sessionId": ctx.session_id, "_ctx": ctx})
    if entry == "after_turn":
        assert result["status"] == "completed"
    elif entry == "prepare_compaction":
        assert result["prepareToken"]
    else:
        assert result["disposed"]
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


def configure_inline_merge(manager):
    manager._archive_max_count = 1
    manager._archive_merge_threshold = 2
    original = manager._maybe_merge_archives
    manager._maybe_merge_archives = lambda session_id, ctx: original(session_id, ctx, wait=True)


@pytest.mark.parametrize("fail_finish", [False, True])
def test_compaction_merges_only_after_durable_source_clear(ctx, fail_finish):
    fs = MemoryFS()
    service, _, manager = real_archives(fs, ctx)
    store = manager._get_archive_store()
    assert store.write_archive(ctx.session_id, "Older turn", "Older turn", [],
                               ctx, archive_id="older").success
    seed(manager, ctx)
    configure_inline_merge(manager)
    original_save = manager.save_session_state
    failed = False

    def save(session_id, request):
        nonlocal failed
        buffer = manager.get_or_create(session_id)
        if fail_finish and not failed and EXTENSION_KEY not in buffer.runtime_extensions and not buffer.messages:
            failed = True
            return False
        return original_save(session_id, request)

    manager.save_session_state = save
    result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx})
    if fail_finish:
        assert failed and result["status"] == "failed" and result["retryable"]
        assert len(store.list_archives(ctx.session_id, ctx)) == 2
        restarted, writer, next_manager = real_archives(
            fs, ctx, extract=lambda **kwargs: pytest.fail("completed lifecycle re-extracted"))
        configure_inline_merge(next_manager)
        result = restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx})
        assert not next_manager.get_or_create(ctx.session_id).pending_extraction_ids()
        writer._pipeline.extract.assert_not_called()
    assert result["compacted"]
    active = store.list_archives(ctx.session_id, ctx)
    assert len(active) == 1 and active[0].archive_id.startswith("merged_")
    merged = store.read_archive(ctx.session_id, active[0].archive_id, ctx)
    assert any(message["id"] == "msg-one" for message in merged.messages)


def test_compaction_partial_failure_keeps_accepted_candidates_across_restart(ctx):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    seed(manager, ctx)
    reject_entity_once(writer)
    with pytest.raises(RuntimeError, match="compaction extraction write failed"):
        service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    archive_id = batch["archive_id"]
    assert batch["lifecycle"]["kind"] == "compaction"
    assert len(nodes(fs, "event")) == 1

    restarted, next_writer, next_manager = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted candidates were re-extracted"))
    prepared = restarted.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    assert prepared["archive_id"] == archive_id
    assert len(nodes(fs, "event")) == len(nodes(fs, "entity")) == 1
    next_writer._pipeline.extract.assert_not_called()
    assert next_manager.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}


def test_dispose_partial_failure_can_resume_the_same_batch_after_restart(ctx, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    seed(manager, ctx)
    reject_entity_once(writer)
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    archive_id = batch["archive_id"]
    assert batch["lifecycle"]["kind"] == "dispose"
    assert len(nodes(fs, "event")) == 1

    restarted, next_writer, next_manager = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("accepted candidates were re-extracted"))
    result = restarted.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["disposed"] is True
    assert len(nodes(fs, "event")) == len(nodes(fs, "entity")) == 1
    assert nodes(fs, "event")[0].metadata["provenance_ids"] == [f"prov:1:archive:{archive_id}:msg-one"]
    next_writer._pipeline.extract.assert_not_called()
    assert not next_manager.has_session(ctx.session_id)


def test_dispose_cold_start_restores_pending_batch_before_lazy_writer_initialization(ctx, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    seed(manager, ctx)
    reject_entity_once(writer)
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})

    restarted, next_writer, _ = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("cold dispose re-extracted accepted candidates"))
    restarted._write_api = None
    restarted.get_write_api = Mock(return_value=next_writer)
    restarted._session_mgr = SessionManager(session_state_bridge_enabled=False, get_llm=lambda: None,
                                             get_agfs=lambda: fs, get_archive_store=lambda: None)
    result = restarted.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "background_flush_dispatched"
    assert len(nodes(fs, "event")) == len(nodes(fs, "entity")) == 1
    assert not restarted._session_mgr.has_session(ctx.session_id)
    next_writer._pipeline.extract.assert_not_called()


def test_prepared_token_survives_restart_and_archives_only_its_fixed_source(ctx):
    fs = MemoryFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    prepared = service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    manager.add_message(ctx.session_id, "user", "Later fact remains pending.", ctx, message_id="tail")
    assert manager.save_session_state(ctx.session_id, ctx)
    restarted, writer, next_manager = real_archives(
        fs, ctx, extract=lambda **kwargs: pytest.fail("prepared batch re-extracted"))
    result = restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                                "prepared": True, "prepareToken": prepared["prepareToken"]})
    assert result["compacted"] is True
    archive = nodes(fs, "history")[0]
    assert archive.metadata["archive_id"] == prepared["archive_id"]
    assert [value["id"] for value in json.loads(archive.content)] == ["msg-one"]
    buffer = next_manager.get_or_create(ctx.session_id)
    assert [message.id for message in buffer.messages] == ["tail"]
    assert buffer.extraction_watermark == 0
    assert not buffer.pending_extraction_ids()
    writer._pipeline.extract.assert_not_called()
    with pytest.raises(ValueError, match="invalid prepare token"):
        restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                           "prepared": True, "prepareToken": prepared["prepareToken"]})


def test_compaction_preserves_the_new_tail_task_state(ctx):
    fs = MemoryFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    prepared = service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    task = "Please investigate the amplifier."
    manager.add_message(ctx.session_id, "user", task, ctx, message_id="tail")
    assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                            "prepareToken": prepared["prepareToken"]})["compacted"] is True
    buffer = manager.get_or_create(ctx.session_id)
    assert [message.id for message in buffer.messages] == ["tail"]
    assert buffer.window_state.active_task == task


@pytest.mark.parametrize("remove", [False, True])
def test_compaction_uncertain_candidate_requires_its_existing_receipt(ctx, remove):
    fs = CommitThenFailFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    assert service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "needs_confirmation"
    event = nodes(fs, "event")[0]
    if remove:
        fs.archive_node(event.uri, ctx)
    restarted, writer, _ = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("uncertain candidate re-extracted"))
    if remove:
        assert restarted.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "needs_confirmation"
        assert not nodes(fs, "event")
    else:
        restarted.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
        assert len(nodes(fs, "event")) == 1
        assert nodes(fs, "event")[0].metadata["version"] == 1
    writer._pipeline.extract.assert_not_called()


class ArchiveCommitThenFailFS(MemoryFS):
    once = True

    def write_node(self, node, ctx):
        super().write_node(node, ctx)
        if node.category == "history" and self.once:
            self.once = False
            raise RuntimeError("archive commit acknowledged only by durable data")


def test_compaction_uncertain_archive_recovers_by_complete_exact_read(ctx):
    fs = ArchiveCommitThenFailFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "failed"
    assert len(nodes(fs, "history")) == 1
    original = deepcopy(nodes(fs, "history")[0])
    restarted, writer, next_manager = real_archives(
        fs, ctx, extract=lambda **kwargs: pytest.fail("archive recovery re-extracted"))
    next_manager._compress = Mock(side_effect=AssertionError("archive must not be recompressed"))
    assert restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx})["compacted"] is True
    assert nodes(fs, "history") == [original]
    assert not next_manager.get_or_create(ctx.session_id).pending_extraction_ids()
    writer._pipeline.extract.assert_not_called()
    next_manager._compress.assert_not_called()


@pytest.mark.parametrize("change", ["remove", "owner", "source", "digest", "inactive"])
def test_compaction_uncertain_archive_is_never_recreated_or_guessed(ctx, change):
    fs = ArchiveCommitThenFailFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "failed"
    archive = nodes(fs, "history")[0]
    stored = fs.nodes[archive.uri]
    if change == "remove":
        fs.nodes.pop(archive.uri)
    elif change == "owner":
        stored.metadata["owner_user_id"] = "another-owner"
    elif change == "source":
        stored.content = "[]"
    elif change == "digest":
        stored.metadata["memory_lifecycle_source_sha256"] = "changed"
    else:
        stored.metadata["archive_status"] = "MERGED"
    restarted, writer, next_manager = real_archives(
        fs, ctx, extract=lambda **kwargs: pytest.fail("archive recovery re-extracted"))
    next_manager._write_archive = Mock(side_effect=AssertionError("uncertain archive was recreated"))
    result = restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    assert next_manager.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}
    next_manager._write_archive.assert_not_called()
    writer._pipeline.extract.assert_not_called()


def test_dispose_dispatch_persists_fixed_source_before_the_worker_runs(ctx, monkeypatch):
    queued = []

    class QueuedThread(InlineThread):
        def start(self):
            queued.append(self)

    monkeypatch.setattr("server.memory_service.threading.Thread", QueuedThread)
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "background_flush_dispatched" and result["flushed"] is False
    state = json.loads(nodes(fs, "state")[0].content)
    assert state["runtime_extensions"][EXTENSION_KEY]["batch"]["message_ids"] == ["msg-one"]
    writer._pipeline.extract.assert_not_called()
    manager.add_message(ctx.session_id, "user", "A tail arriving after dispatch.", ctx, message_id="tail")
    queued[0].target(*queued[0].args)
    assert [message.id for message in manager.get_or_create(ctx.session_id).messages] == ["tail"]
    assert manager.get_or_create(ctx.session_id).extraction_watermark == 0
    assert not manager.get_or_create(ctx.session_id).pending_extraction_ids()


def test_dispose_real_background_failure_reports_pending_then_can_retry(ctx):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    service._count_outbox_unfinished = Mock(return_value={
        "pending": 0, "processing": 0, "failed": 0, "total": 0, "supported": True,
    })
    seed(manager, ctx)
    reject_entity_once(writer)
    assert service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})["flushed"] is False
    buffer = manager.get_or_create(ctx.session_id)
    # Join the actual lifecycle worker by polling the public idle entry with a bounded timeout.
    result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                      "timeoutSeconds": 3, "drainOutbox": False})
    assert result["idle"] is False and result["reason"] == "lifecycle_recovery_pending"
    assert result["status"] == "failed" and result["retryable"] is True
    assert not buffer.dispose_in_progress
    writer._pipeline.extract.side_effect = AssertionError("retry re-extracted accepted candidates")
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                    "timeoutSeconds": 3, "drainOutbox": False})["idle"] is True
    assert not manager.has_session(ctx.session_id)


@pytest.mark.parametrize("entry", ["after_turn", "ingest", "dispose"])
def test_compaction_batch_cannot_be_consumed_by_another_lifecycle(ctx, entry):
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    original = deepcopy(nodes(fs, "state")[0])
    restarted, writer, _ = lifecycle_service(fs, ctx)
    result = getattr(restarted, entry)({"sessionId": ctx.session_id, "_ctx": ctx,
                                       "messages": [{"role": "user", "content": "new independent fact"}]})
    assert result["status"] == "needs_confirmation"
    assert nodes(fs, "state") == [original]
    writer._pipeline.extract.assert_not_called()


class FailFinalStateOnceFS(MemoryFS):
    failed = False

    def write_node(self, node, ctx):
        if (node.category == "state" and not self.failed and nodes(self, "event")
                and EXTENSION_KEY not in json.loads(node.content).get("runtime_extensions", {})):
            self.failed = True
            raise RuntimeError("final state acknowledgement unavailable")
        super().write_node(node, ctx)


def test_compaction_finalize_failure_keeps_source_and_recovers_archive_receipt(ctx):
    fs = FailFinalStateOnceFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "failed" and result["compacted"] is False
    assert manager.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}
    archive = deepcopy(nodes(fs, "history")[0])
    restarted, writer, next_manager = real_archives(
        fs, ctx, extract=lambda **kwargs: pytest.fail("finalize retry re-extracted"))
    assert restarted.compact({"sessionId": ctx.session_id, "_ctx": ctx})["compacted"] is True
    assert nodes(fs, "history") == [archive]
    assert not next_manager.get_or_create(ctx.session_id).runtime_extensions.get(EXTENSION_KEY)
    writer._pipeline.extract.assert_not_called()


def test_dispose_finalize_failure_keeps_source_and_resumes_without_extraction(ctx, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs = FailFinalStateOnceFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert manager.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}
    assert len(nodes(fs, "event")) == 1
    restarted, writer, next_manager = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("dispose finalize re-extracted"))
    restarted.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert not next_manager.has_session(ctx.session_id)
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


@pytest.mark.parametrize("entry", ["prepare_compaction", "dispose"])
def test_lifecycle_checkpoint_failure_prevents_dispatch_or_first_memory_write(ctx, entry, monkeypatch):
    class FailStateFS(MemoryFS):
        def write_node(self, node, ctx):
            if node.category == "state":
                raise RuntimeError("source journal unavailable")
            super().write_node(node, ctx)

    thread = Mock(side_effect=AssertionError("worker dispatched without durable source"))
    monkeypatch.setattr("server.memory_service.threading.Thread", thread)
    fs = FailStateFS()
    service, writer, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    result = getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "failed" and result["retryable"] is True
    assert not nodes(fs, "event")
    assert len(manager.get_or_create(ctx.session_id).messages) == 1
    writer._pipeline.extract.assert_not_called()
    thread.assert_not_called()
    buffer = manager.get_or_create(ctx.session_id)
    assert not buffer.lifecycle_prepare_in_progress and not buffer.lifecycle_source_ids
    assert not buffer.extraction_in_progress and not buffer.dispose_in_progress
    assert buffer.active_operation_count == 0


def test_expired_prepare_token_changes_only_token_and_preserves_accepted_batch(ctx):
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    prepared = service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    state = nodes(fs, "state")[0]
    payload = json.loads(state.content)
    payload["runtime_extensions"][EXTENSION_KEY]["batch"]["lifecycle"]["prepared"]["created_at"] = "2000-01-01T00:00:00+00:00"
    fs.nodes[state.uri].content = json.dumps(payload)
    restarted, writer, _ = lifecycle_service(
        fs, ctx, extract=lambda **kwargs: pytest.fail("expired token re-extracted"))
    renewed = restarted.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    assert renewed["archive_id"] == prepared["archive_id"]
    assert renewed["prepareToken"] != prepared["prepareToken"]
    assert len(nodes(fs, "event")) == 1
    writer._pipeline.extract.assert_not_called()


@pytest.mark.parametrize("change", ["source", "token_digest", "owner"])
def test_lifecycle_restart_rejects_changed_source_token_digest_or_owner(ctx, change):
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    state = nodes(fs, "state")[0]
    payload = json.loads(state.content)
    batch = payload["runtime_extensions"][EXTENSION_KEY]["batch"]
    if change == "source":
        payload["buffer_messages"][0]["content"] = "mutated source"
        batch["snapshot"][0]["content"] = "mutated source"
    elif change == "token_digest":
        batch["lifecycle"]["prepared"]["source_digest"] = "wrong"
    else:
        fs.nodes[state.uri].metadata["owner_user_id"] = "other-owner"
    fs.nodes[state.uri].content = json.dumps(payload)
    restarted, writer, _ = lifecycle_service(fs, ctx)
    result = restarted.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    writer._pipeline.extract.assert_not_called()


@pytest.mark.parametrize("pause_at", ["before_claim", "after_claim"])
def test_compaction_cannot_take_source_reserved_by_an_ordinary_turn(ctx, pause_at):
    fs = MemoryFS()
    service, writer, manager = real_archives(fs, ctx)
    buffer = manager.get_or_create(ctx.session_id, ctx=ctx)
    reached, release = threading.Event(), threading.Event()
    original = (buffer.claim_incremental_extraction if pause_at == "before_claim"
                else service._build_incremental_extraction_state)

    def paused(*args, **kwargs):
        if threading.current_thread().name == "ordinary-source":
            reached.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    if pause_at == "before_claim":
        buffer.claim_incremental_extraction = paused
    else:
        service._build_incremental_extraction_state = paused
    results, failures = [], []

    def ordinary():
        try:
            results.append(service.after_turn(params(ctx)))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=ordinary, name="ordinary-source")
    worker.start()
    try:
        assert reached.wait(5)
        prepared = service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
        assert prepared["ok"] is False and prepared["status"] == "processing"
        assert prepared["retryable"] is True
        if pause_at == "before_claim":
            disposed = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
            assert disposed["disposed"] is False and disposed["retryable"] is True
        assert not buffer.pending_extraction_ids()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert results[0]["status"] == "completed"
    assert len(nodes(fs, "event")) == 1
    assert writer._pipeline.extract.call_count == 1
    assert buffer.ordinary_turns_in_progress == 0


@pytest.mark.parametrize("entry", ["after_turn", "ingest"])
def test_source_intake_cannot_change_compaction_journal_preparation(ctx, entry):
    fs = MemoryFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    reached, release = threading.Event(), threading.Event()
    original = service._compaction_journal

    def paused(result):
        reached.set()
        assert release.wait(5)
        return original(result)

    service._compaction_journal = paused
    results, failures = [], []

    def prepare():
        try:
            results.append(service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx}))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=prepare)
    worker.start()
    try:
        assert reached.wait(5)
        result = getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx, "wait": True,
            "messages": [{"id": "msg-one", "role": "user", "content": "A changed independent fact. " * 40}]})
        assert result.get("ok") is False, result
        assert result["status"] == "processing" and result["retryable"] is True
        assert result["reason"] == "lifecycle_preparing"
        assert [message.id for message in manager.get_or_create(ctx.session_id).messages] == ["msg-one"]
        assert manager.get_or_create(ctx.session_id).messages[0].content == "I repaired the receiver. " * 40
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert results[0]["prepareToken"]
    buffer = manager.get_or_create(ctx.session_id)
    assert not buffer.lifecycle_prepare_in_progress and not buffer.extraction_in_progress
    assert buffer.active_operation_count == 0


def test_cold_idle_loads_failed_dispose_without_initializing_a_writer(ctx, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    seed(manager, ctx)
    reject_entity_once(writer)
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    restarted, next_writer, _ = lifecycle_service(fs, ctx)
    restarted._write_api = None
    restarted.get_write_api = Mock(side_effect=AssertionError("idle must not create a writer"))
    restarted._session_mgr = SessionManager(session_state_bridge_enabled=False,
        get_context_fs=lambda: fs, get_llm=lambda: None, get_archive_store=lambda: None)
    result = restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
        "timeoutSeconds": 1, "drainOutbox": False, "waitOutbox": False})
    assert result["idle"] is False and result["reason"] == "lifecycle_recovery_pending"
    assert result["status"] == "failed" and result["retryable"] is True
    assert restarted._session_mgr.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}
    next_writer._pipeline.extract.assert_not_called()
    restarted.get_write_api.assert_not_called()


def test_cold_idle_cannot_treat_unknown_durable_read_as_a_missing_session(ctx):
    class UnavailableStateFS(MemoryFS):
        def read_node(self, uri, ctx):
            raise TimeoutError("durable session read unavailable")

    fs = UnavailableStateFS()
    service, _, _ = lifecycle_service(fs, ctx)
    service._write_api = None
    service.get_write_api = Mock(side_effect=AssertionError("idle must not create a writer"))
    service._session_mgr = SessionManager(session_state_bridge_enabled=False,
        get_context_fs=lambda: fs, get_llm=lambda: None, get_archive_store=lambda: None)
    result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                     "drainOutbox": False, "waitOutbox": False})
    assert result["idle"] is False and result["reason"] == "recovery_state_unavailable"
    assert result["status"] == "needs_confirmation" and result["retryable"] is False
    assert not fs.nodes
    service.get_write_api.assert_not_called()


@pytest.mark.parametrize("restart", [False, True])
def test_idle_cannot_report_success_for_an_ordinary_pending_batch(ctx, restart):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx, entries=[candidate(), candidate("entity")])
    reject_entity_once(writer)
    assert service.after_turn(params(ctx))["status"] == "failed"
    if restart:
        service, writer, manager = lifecycle_service(fs, ctx)
    result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                     "drainOutbox": False, "waitOutbox": False})
    assert result["idle"] is False and result["reason"] == "extraction_recovery_pending"
    assert manager.get_or_create(ctx.session_id).pending_extraction_ids()
    assert len(nodes(fs, "event")) == 1 and not nodes(fs, "entity")
    writer._pipeline.extract.side_effect = AssertionError("idle retry re-extracted accepted candidates")
    assert service.after_turn(params(ctx))["status"] == "completed"
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                   "drainOutbox": False, "waitOutbox": False})["idle"] is True


def test_idle_does_not_recreate_a_finished_dispose_buffer(ctx, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs = MemoryFS()
    service, _, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert not manager.has_session(ctx.session_id)
    result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                     "drainOutbox": False, "waitOutbox": False})
    assert result["idle"] is True and result["reason"] == "session_not_found"
    assert not manager.has_session(ctx.session_id)


@pytest.mark.parametrize("fail", [False, True])
def test_dispose_can_defer_to_an_active_ordinary_durable_worker(ctx, fail):
    fs = MemoryFS()
    service, writer, manager = real_archives(fs, ctx, entries=[candidate(), candidate("entity")] if fail else None)
    if fail:
        reject_entity_once(writer)
        service._extraction_retry.wake = Mock()
    original = writer._pipeline.extract.side_effect
    reached, release = threading.Event(), threading.Event()

    def paused(*args, **kwargs):
        reached.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    writer._pipeline.extract.side_effect = paused
    result = service.after_turn({"sessionId": ctx.session_id, "_ctx": ctx,
        "messages": [{"id": "msg-one", "role": "user", "content": "I repaired the receiver. " * 40}]})
    assert result["status"] == "processing"
    try:
        assert reached.wait(5)
        result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert result["disposed"] is True and result["status"] == "dispose_deferred", result
    finally:
        release.set()
    idle = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
        "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})
    if fail:
        assert idle["idle"] is False and idle["reason"] == "extraction_retry_task"
        assert manager.get_or_create(ctx.session_id).pending_extraction_ids() == {"msg-one"}
        assert len(nodes(fs, "event")) == 1 and not nodes(fs, "entity")
        task = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        task["next_attempt_at"] = 0
        assert manager.save_session_state(ctx.session_id, ctx)
        assert service._extraction_retry.run_once(ctx, task["id"]) is None
        assert len(nodes(fs, "entity")) == len(nodes(fs, "history")) == 1
        return
    assert idle["idle"] is True
    assert not manager.has_session(ctx.session_id)
    assert len(nodes(fs, "event")) == len(nodes(fs, "history")) == 1


@pytest.mark.parametrize("kind", ["compaction", "dispose"])
@pytest.mark.parametrize("mutator", ["after_turn", "ingest", "replace", "remove", "commit"])
def test_fixed_lifecycle_source_is_reserved_until_its_journal_exists(ctx, kind, mutator):
    fs = MemoryFS()
    service, _, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    buffer = manager.get_or_create(ctx.session_id)
    reached, release = threading.Event(), threading.Event()
    helper = "_compaction_journal" if kind == "compaction" else "_dispose_journal"
    original = getattr(service, helper)
    results, failures = [], []

    def paused(*args, **kwargs):
        reached.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    setattr(service, helper, paused)

    def prepare():
        try:
            results.append(getattr(service, "prepare_compaction" if kind == "compaction" else "dispose")(
                {"sessionId": ctx.session_id, "_ctx": ctx}))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=prepare)
    worker.start()
    try:
        assert reached.wait(5)
        if mutator in {"after_turn", "ingest"}:
            result = getattr(service, mutator)({"sessionId": ctx.session_id, "_ctx": ctx, "wait": True,
                "messages": [{"id": "msg-one", "role": "user", "content": "Changed source. " * 80}]})
            assert result["ok"] is False and result["status"] == "processing" and result["retryable"]
        elif mutator == "replace":
            with pytest.raises(ValueError, match="cannot rewrite"):
                manager.add_message(ctx.session_id, "user", "Changed source.", ctx, message_id="msg-one")
        elif mutator == "remove":
            with pytest.raises(ValueError, match="cannot remove"):
                manager.remove_message(ctx.session_id, "msg-one", ctx)
        else:
            with pytest.raises(SessionOperationInProgress) as conflict:
                service.commit_session_buffer({"sessionId": ctx.session_id, "_ctx": ctx, "wait": True})
            assert conflict.value.conflict_type == "lifecycle_source_reserved"
        assert buffer.messages[0].content == "I repaired the receiver. " * 40
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    if kind == "dispose":
        assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
            "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})["idle"] is True
    else:
        assert results[0]["prepareToken"]
    assert not buffer.lifecycle_prepare_in_progress and not buffer.lifecycle_source_ids


@pytest.mark.parametrize("kind", ["compaction", "dispose"])
def test_public_append_after_lifecycle_snapshot_stays_outside_the_frozen_batch(ctx, kind):
    fs = MemoryFS()
    service, writer, manager = real_archives(fs, ctx)
    seed(manager, ctx)
    reached, release = threading.Event(), threading.Event()
    helper = "_compaction_journal" if kind == "compaction" else "_dispose_journal"
    original = getattr(service, helper)
    results, failures = [], []
    tail = "The amplifier needs a separate follow-up investigation."

    def paused(*args, **kwargs):
        if not reached.is_set():
            reached.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    setattr(service, helper, paused)

    def prepare():
        try:
            results.append(getattr(service, "prepare_compaction" if kind == "compaction" else "dispose")(
                {"sessionId": ctx.session_id, "_ctx": ctx}))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=prepare)
    worker.start()
    try:
        assert reached.wait(5)
        added = service.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx,
                                            "role": "user", "content": tail})
        assert added["ok"] is True
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    if kind == "compaction":
        batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
        assert batch["message_ids"] == ["msg-one"]
        assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                               "prepareToken": results[0]["prepareToken"]})["compacted"] is True
        assert [item["id"] for item in json.loads(nodes(fs, "history")[0].content)] == ["msg-one"]
    else:
        assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
            "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})["idle"] is True
    buffer = manager.get_or_create(ctx.session_id)
    assert [(message.id, message.content) for message in buffer.messages] == [(added["message_id"], tail)]
    assert buffer.extraction_watermark == 0
    assert all(item["content"] != tail for item in writer._pipeline.extract.call_args.args[0])


def test_below_threshold_dispose_clears_only_its_original_source_and_keeps_new_tail(ctx):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx, content="A short source.")
    original = service._prepare_dispose_source
    tail = "A newly appended source is pending. " * 50

    def append_after_capture(*args, **kwargs):
        state = original(*args, **kwargs)
        assert service.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx,
                                           "role": "user", "content": tail})["ok"]
        return state

    service._prepare_dispose_source = append_after_capture
    result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
    assert result["reason"] == "below_token_threshold"
    assert [message.content for message in manager.get_or_create(ctx.session_id).messages] == [tail]
    assert manager.get_or_create(ctx.session_id).extraction_watermark == 0
    writer._pipeline.extract.assert_not_called()


def test_a_changed_source_cannot_create_an_invalid_new_lifecycle_checkpoint(ctx):
    fs = MemoryFS()
    service, writer, manager = lifecycle_service(fs, ctx)
    seed(manager, ctx)
    buffer = manager.get_or_create(ctx.session_id)
    original_source = deepcopy(buffer.messages)
    manager.add_message(ctx.session_id, "user", "Changed source.", ctx, message_id="msg-one")
    with pytest.raises(RecoveryNeedsConfirmationError, match="source snapshot changed before checkpoint"):
        service._dispose_journal(manager, buffer, ctx, original_source)
    assert not nodes(fs, "state") and not buffer.pending_extraction_ids()
    writer._pipeline.extract.assert_not_called()
