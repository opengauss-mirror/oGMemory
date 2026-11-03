"""Lifecycle checkpoints on an explicitly disposable PostgreSQL database."""

import os
from unittest.mock import Mock
from uuid import uuid4

import pytest

from core.models import RequestContext
from session.models import ArchiveWriteResult
from session.session_manager import SessionManager
from session.sql_archive_store import SQLSessionArchiveStore
from tests.integration.test_extraction_batch_recovery_sql import category_nodes
from tests.integration.test_occurrence_replay_sql import close_stores, stores
from tests.unit.server.test_extraction_batch_recovery import candidate, params
from tests.unit.server.test_lifecycle_batch_recovery import (
    InlineThread,
    configure_inline_merge,
    lifecycle_service,
    reject_entity_once,
)


@pytest.fixture
def ctx():
    return RequestContext("lifecycle-sql-" + uuid4().hex, "reader", "agent", "session", "trace")


def sql_service(fs, outbox, ctx, *, archive_store=None, **kwargs):
    service, writer, manager = lifecycle_service(fs, ctx, outbox=outbox, **kwargs)
    service._cfg.storage_backend = "sql"
    service._cfg.sql_connection_string = os.environ["OGMEM_IDENTITY_TEST_DSN"]
    if archive_store is not None:
        service._get_archive_store = Mock(return_value=archive_store)
        manager._get_archive_store = lambda: archive_store
        manager._archive_store_required = True
        manager.commit_snapshot = SessionManager.commit_snapshot.__get__(manager)
    return service, writer, manager


def seed_sql(service, ctx):
    """Persist the source through the guarded public entry before recovery."""
    threshold = service._cfg.after_turn_threshold
    service._cfg.after_turn_threshold = 1_000_000_000
    try:
        result = service.after_turn({
            "sessionId": ctx.session_id, "_ctx": ctx, "wait": True,
            "clientRequestId": "seed",
            "messages": [{"id": "msg-one", "role": "user",
                          "content": "I repaired the receiver. " * 40}],
        })
        assert result["status"] == "accumulating"
    finally:
        service._cfg.after_turn_threshold = threshold


@pytest.mark.parametrize("entry", ["after_turn", "prepare_compaction", "dispose"])
def test_sql_initial_receipt_read_failure_reconnect_can_retry(ctx, entry, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(fs, outbox, ctx)
        seed_sql(service, ctx)
        original = type(fs).list_node_summaries_page
        failed = False

        def read(store, uri, request, **kwargs):
            nonlocal failed
            if store is fs and uri.endswith("/memories/events") and not failed:
                failed = True
                raise OSError("injected transient SQL receipt read failure")
            return original(store, uri, request, **kwargs)

        monkeypatch.setattr(type(fs), "list_node_summaries_page", read)
        if entry == "prepare_compaction":
            with pytest.raises(RuntimeError, match="compaction extraction write failed"):
                service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
        elif entry == "after_turn":
            assert service.after_turn(params(ctx))["status"] == "failed"
        else:
            service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert failed and not category_nodes(writer, "event", ctx)
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(
            fs, outbox, ctx, extract=lambda **kwargs: pytest.fail("accepted SQL candidate re-extracted"))
        result = getattr(service, entry)(params(ctx) if entry == "after_turn"
                                         else {"sessionId": ctx.session_id, "_ctx": ctx})
        if entry == "after_turn":
            assert result["status"] == "completed"
        elif entry == "prepare_compaction":
            assert result["prepareToken"]
        else:
            assert result["disposed"]
            assert not manager.has_session(ctx.session_id)
        assert len(category_nodes(writer, "event", ctx)) == 1
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("fail_finish", [False, True])
def test_sql_compaction_merge_waits_for_durable_source_clear(ctx, fail_finish):
    fs, outbox = stores()
    archive_store = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, archive_store=archive_store)
        seed_sql(service, ctx)
        assert archive_store.write_archive(ctx.session_id, "Older turn", "Older turn", [],
                                           ctx, archive_id="older").success
        configure_inline_merge(manager)
        original = manager.save_session_state
        failed = False

        def save(session_id, request):
            nonlocal failed
            buffer = manager.get_or_create(session_id)
            if fail_finish and not failed and "memory_extraction_recovery" not in buffer.runtime_extensions and not buffer.messages:
                failed = True
                return False
            return original(session_id, request)

        manager.save_session_state = save
        result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx})
        if fail_finish:
            assert failed and result["status"] == "failed" and result["retryable"]
            assert len(archive_store.list_archives(ctx.session_id, ctx)) == 2
        else:
            assert result["compacted"]
            active = archive_store.list_archives(ctx.session_id, ctx)
            assert len(active) == 1 and active[0].archive_id.startswith("merged_")
            assert not manager.get_or_create(ctx.session_id).pending_extraction_ids()
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)
    if not fail_finish:
        return

    fs, outbox = stores()
    archive_store = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, archive_store=archive_store,
            extract=lambda **kwargs: pytest.fail("acknowledged SQL lifecycle re-extracted"))
        configure_inline_merge(manager)
        assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx})["compacted"]
        active = archive_store.list_archives(ctx.session_id, ctx)
        assert len(active) == 1 and active[0].archive_id.startswith("merged_")
        merged = archive_store.read_archive(ctx.session_id, active[0].archive_id, ctx)
        assert any(message["id"] == "msg-one" for message in merged.messages)
        assert not manager.get_or_create(ctx.session_id).pending_extraction_ids()
        assert len(category_nodes(writer, "event", ctx)) == 1
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)


@pytest.mark.parametrize("entry", ["prepare_compaction", "dispose"])
def test_lifecycle_partial_rejection_reconnect_reuses_accepted_operations(ctx, entry, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, entries=[candidate(), candidate("entity")])
        seed_sql(service, ctx)
        reject_entity_once(writer)
        if entry == "prepare_compaction":
            with pytest.raises(RuntimeError, match="compaction extraction write failed"):
                service.prepare_compaction({"sessionId": ctx.session_id, "_ctx": ctx})
        else:
            service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert len(category_nodes(writer, "event", ctx)) == 1
        original = category_nodes(writer, "event", ctx)[0]
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(
            fs, outbox, ctx, extract=lambda **kwargs: pytest.fail("SQL lifecycle re-extracted accepted candidates"))
        getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx})
        assert category_nodes(writer, "event", ctx) == [original]
        assert len(category_nodes(writer, "entity", ctx)) == 1
        writer._pipeline.extract.assert_not_called()
        if entry == "dispose":
            assert not manager.has_session(ctx.session_id)
    finally:
        close_stores(fs, outbox)


class CommitThenFailArchiveStore(SQLSessionArchiveStore):
    once = True

    def write_archive(self, *args, **kwargs):
        result = super().write_archive(*args, **kwargs)
        if result.success and self.once:
            self.once = False
            return ArchiveWriteResult(archive_id=result.archive_id, session_id=result.session_id,
                                      uri=result.uri, success=False, error="lost acknowledgement after SQL commit")
        return result


@pytest.mark.parametrize("remove", [False, True])
def test_lifecycle_archive_reconnect_requires_exact_existing_archive(ctx, remove):
    fs, outbox = stores()
    archive_store = CommitThenFailArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, archive_store=archive_store)
        seed_sql(service, ctx)
        assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx})["status"] == "failed"
        batch = manager.get_or_create(ctx.session_id).runtime_extensions["memory_extraction_recovery"]["batch"]
        archive_id = batch["archive_id"]
        original = archive_store.read_archive(ctx.session_id, archive_id, ctx)
        assert original is not None
        if remove:
            assert archive_store.delete_archive(ctx.session_id, archive_id, ctx)
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)


    fs, outbox = stores()
    archive_store = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    try:
        service, writer, manager = sql_service(
            fs, outbox, ctx, archive_store=archive_store,
            extract=lambda **kwargs: pytest.fail("SQL archive recovery re-extracted"))
        manager._write_archive = Mock(side_effect=AssertionError("SQL uncertain archive recreated"))
        result = service.compact({"sessionId": ctx.session_id, "_ctx": ctx})
        if remove:
            assert result["status"] == "needs_confirmation"
            assert archive_store.read_archive(ctx.session_id, archive_id, ctx) is None
            assert manager.get_or_create(ctx.session_id).pending_extraction_ids()
        else:
            assert result["compacted"] is True
            assert archive_store.read_archive(ctx.session_id, archive_id, ctx) == original
            assert not manager.get_or_create(ctx.session_id).pending_extraction_ids()
        assert len(category_nodes(writer, "event", ctx)) == 1
        manager._write_archive.assert_not_called()
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)


def test_sql_dispose_default_background_failure_remains_recoverable(ctx):
    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, entries=[candidate(), candidate("entity")])
        seed_sql(service, ctx)
        reject_entity_once(writer)
        result = service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert result["status"] == "background_flush_dispatched" and result["flushed"] is False
        result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                          "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})
        assert result["idle"] is False and result["reason"] == "lifecycle_recovery_pending"
        assert result["status"] == "failed" and result["retryable"] is True
        writer._pipeline.extract.side_effect = AssertionError("SQL background retry re-extracted")
        service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                        "timeoutSeconds": 5, "drainOutbox": False, "waitOutbox": False})["idle"] is True
        assert not manager.has_session(ctx.session_id)
        assert len(category_nodes(writer, "event", ctx)) == len(category_nodes(writer, "entity", ctx)) == 1
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("entry", ["after_turn", "dispose"])
def test_sql_cold_idle_observes_failed_batch_without_initializing_writer(ctx, entry, monkeypatch):
    monkeypatch.setattr("server.memory_service.threading.Thread", InlineThread)
    fs, outbox = stores()
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, entries=[candidate(), candidate("entity")])
        reject_entity_once(writer)
        if entry == "after_turn":
            assert service.after_turn(params(ctx))["status"] == "failed"
        else:
            seed_sql(service, ctx)
            service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        assert manager.get_or_create(ctx.session_id).pending_extraction_ids()
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        service, writer, _ = sql_service(fs, outbox, ctx)
        service._write_api = None
        service.get_write_api = Mock(side_effect=AssertionError("SQL idle must not create a writer"))
        service._session_mgr = SessionManager(session_state_bridge_enabled=False,
            get_context_fs=lambda: fs, get_llm=lambda: None, get_archive_store=lambda: None)
        result = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx,
                                         "drainOutbox": False, "waitOutbox": False})
        assert result["idle"] is False
        assert result["reason"] == ("extraction_recovery_pending" if entry == "after_turn"
                                    else "lifecycle_recovery_pending")
        assert service._session_mgr.get_or_create(ctx.session_id).pending_extraction_ids()
        assert len(category_nodes(writer, "event", ctx)) == 1
        assert not category_nodes(writer, "entity", ctx)
        service.get_write_api.assert_not_called()
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("entry", ["prepare_compaction", "dispose"])
def test_sql_fixed_source_and_public_new_tail_remain_valid_after_reconnect(ctx, entry, monkeypatch):
    fs, outbox = stores()
    archive_store = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    prepared_token = None
    tail = "The amplifier needs a separate follow-up investigation."
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, archive_store=archive_store,
            entries=[candidate(), candidate("entity")] if entry == "dispose" else None)
        seed_sql(service, ctx)
        if entry == "dispose":
            reject_entity_once(writer)
        helper = "_compaction_journal" if entry == "prepare_compaction" else "_dispose_journal"
        original = getattr(service, helper)
        appended = False

        def append_after_capture(*args, **kwargs):
            nonlocal appended
            if not appended:
                appended = True
                assert service.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx,
                    "role": "user", "content": tail})["ok"]
            return original(*args, **kwargs)

        setattr(service, helper, append_after_capture)
        with monkeypatch.context() as patch:
            patch.setattr("server.memory_service.threading.Thread", InlineThread)
            result = getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx})
        if entry == "prepare_compaction":
            prepared_token = result["prepareToken"]
        batch = manager.get_or_create(ctx.session_id).runtime_extensions["memory_extraction_recovery"]["batch"]
        archive_id = batch["archive_id"]
        assert batch["message_ids"] == ["msg-one"]
        assert all(message["content"] != tail for message in batch["messages"])
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)

    fs, outbox = stores()
    archive_store = SQLSessionArchiveStore(os.environ["OGMEM_IDENTITY_TEST_DSN"])
    try:
        service, writer, manager = sql_service(fs, outbox, ctx, archive_store=archive_store,
            extract=lambda **kwargs: pytest.fail("fixed-source SQL recovery re-extracted accepted candidates"))
        with monkeypatch.context() as patch:
            patch.setattr("server.memory_service.threading.Thread", InlineThread)
            if entry == "prepare_compaction":
                assert service.compact({"sessionId": ctx.session_id, "_ctx": ctx,
                    "prepareToken": prepared_token})["compacted"] is True
                archive = archive_store.read_archive(ctx.session_id, archive_id, ctx)
                assert [message["id"] for message in archive.messages] == ["msg-one"]
            else:
                service.dispose({"sessionId": ctx.session_id, "_ctx": ctx})
        buffer = manager.get_or_create(ctx.session_id)
        assert [message.content for message in buffer.messages] == [tail]
        assert buffer.extraction_watermark == 0 and not buffer.pending_extraction_ids()
        assert len(category_nodes(writer, "event", ctx)) == 1
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)
        close_stores(archive_store, archive_store)
