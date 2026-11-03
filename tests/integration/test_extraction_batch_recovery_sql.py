"""Durable extraction recovery against an explicitly disposable PostgreSQL DB."""

import json
import os
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError, WriteOutcomeUnknownError
from core.models import RequestContext
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from tests.integration.test_occurrence_replay_sql import close_stores, stores
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_profile_merge import profile
from tests.unit.server.test_extraction_batch_recovery import (
    NoModel,
    candidate,
    params,
    setup_service,
)


@pytest.fixture
def ctx():
    return RequestContext("batch-sql-" + uuid4().hex, "reader", "agent", "session", "trace")


def custom_stores(fs_type):
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    fs = fs_type(dsn)
    return fs, SQLOutboxStore(dsn, fs=fs)


def category_nodes(writer, category, ctx):
    directory = writer._writer._identity_resolver.uris.get_directory_uri(category, ctx)
    return [writer._fs.read_node(uri, ctx) for uri in writer._fs.list_children(directory, ctx)]


class CommitThenFailSQL(SQLContextFS):
    once = True
    unread_receipt_uri = None

    def read_node(self, uri, ctx):
        if uri == self.unread_receipt_uri:
            self.unread_receipt_uri = None
            raise ConnectionError("receipt unavailable after connection loss")
        return super().read_node(uri, ctx)

    def write_node_with_outbox(self, node, ctx, outbox_event=None):
        super().write_node_with_outbox(node, ctx, outbox_event)
        if node.category == "event" and self.once:
            self.once = False
            self.unread_receipt_uri = node.uri
            raise RuntimeError("SQL connection lost after acknowledged transaction commit")


@pytest.mark.parametrize("category", ["profile", "entity", "preference", "event", "case"])
def test_receipt_reconnect_preserves_history_and_requires_new_operation_cas(ctx, category):
    fs, outbox = stores()
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        original = profile("identity", "Initial fact.") if category == "profile" else memory(
            "topic", category=category, content="Initial fact.")
        if category == "case":
            original = replace(original, owner_scope="agent")
        uri = writer.write_memory(original, ctx)["target_uri"]
        accepted = replace(original, target_uri=uri, expected_version=1,
                           content="Initial fact. Added fact.", write_operation_id="accepted")
        assert writer.write_memory(accepted, ctx)["action"] == "merge"
        writer.write_memory(replace(accepted, expected_version=2, write_operation_id="",
                                    content="Initial fact. Added fact. Later fact."), ctx)
        assert len(outbox.list_pending(ctx.account_id)) == 3
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        before = fs.read_node(uri, ctx)
        assert before.metadata["version"] == 3 and before.metadata[RECEIPTS_KEY]["accepted"]
        result = writer.write_memory(accepted, replace(ctx, trace_id="retry-trace"))
        assert result["action"] == "skip" and result["merged_fields"]["status"] == "operation_committed"
        with pytest.raises(WriteNotStartedError):
            writer.write_memory(replace(accepted, write_operation_id="new-operation"), ctx)
        assert fs.read_node(uri, ctx) == before
        assert len(outbox.list_pending(ctx.account_id)) == 3
    finally:
        close_stores(fs, outbox)


def test_receipt_node_and_outbox_rollback_together(ctx):
    fs, outbox = stores()
    original_build = outbox.build_write_event

    def invalid_outbox_payload(node):
        event = original_build(node)
        event.payload["not_json_serializable"] = object()
        return event

    outbox.build_write_event = invalid_outbox_payload
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        with pytest.raises(WriteOutcomeUnknownError):
            writer.write_memory(replace(candidate(), write_operation_id="rollback-operation"), ctx)
        assert category_nodes(writer, "event", ctx) == []
        assert outbox.list_pending(ctx.account_id) == []
    finally:
        close_stores(fs, outbox)


def test_public_unknown_commit_reconnect_resumes_accepted_candidates_and_receipt(ctx):
    fs, outbox = custom_stores(CommitThenFailSQL)
    try:
        service, writer, manager = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
        batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
        archive_id = batch["archive_id"]
        events = category_nodes(writer, "event", ctx)
        assert len(events) == 1 and events[0].metadata[RECEIPTS_KEY]
        assert len(outbox.list_pending(ctx.account_id)) == 2
        assert category_nodes(writer, "session_archive", ctx)[0].content == "user: I repaired the receiver."
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        service, writer, manager = setup_service(
            fs, ctx, outbox=outbox, extract=lambda **kwargs: pytest.fail("SQL recovery re-extracted"))
        assert service.after_turn(params(ctx))["status"] == "completed"
        events = category_nodes(writer, "event", ctx)
        assert len(events) == 1 and events[0].metadata["version"] == 1
        assert events[0].metadata["provenance_ids"] == [f"prov:1:archive:{archive_id}:msg-one"]
        assert len(outbox.list_pending(ctx.account_id)) == 2  # one event and its raw archive
        writer._pipeline.extract.assert_not_called()
        assert not manager.get_or_create(ctx.session_id).pending_extraction_ids()
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("remove", ["delete", "archive"])
def test_removed_unacknowledged_sql_node_never_reappears(ctx, remove):
    fs, outbox = custom_stores(CommitThenFailSQL)
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
        uri = category_nodes(writer, "event", ctx)[0].uri
        (fs.delete_node if remove == "delete" else fs.archive_node)(uri, ctx)
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
        assert category_nodes(writer, "event", ctx) == []
        writer._pipeline.extract.assert_not_called()
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)


def test_writing_checkpoint_before_sql_write_never_guesses_uncommitted(ctx):
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        writer._writer.write_candidate = Mock(side_effect=SystemExit("terminated before SQL write"))
        with pytest.raises(SystemExit):
            service.after_turn(params(ctx))
        assert category_nodes(writer, "event", ctx) == []
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
        assert category_nodes(writer, "event", ctx) == []
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("source_only", [False, True])
def test_sql_commit_result_checkpoint_recovery_preserves_receipt_and_outbox(ctx, source_only):
    class FailResultSQL(SQLContextFS):
        failed = False

        def write_node(self, node, request):
            if node.category == "state":
                batch = json.loads(node.content).get("runtime_extensions", {}).get(EXTENSION_KEY, {}).get("batch", {})
                operation = batch.get("slots", {}).get("chunk:0", {}).get("operations", {}).get("0", {})
                if not self.failed and operation.get("status") == "complete":
                    self.failed = True
                    raise RuntimeError("SQL checkpoint acknowledgement unavailable")
            super().write_node(node, request)

    fs, outbox = custom_stores(FailResultSQL)
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "failed"
        assert len(category_nodes(writer, "event", ctx)) == 1
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        if source_only:
            with service._session_mutation_scope(ctx):
                archive_id = service._extraction_retry._journal(ctx).batch["archive_id"]
                result = service._complete_extraction_source(ctx, archive_id, "model budget exhausted")
        else:
            result = service.after_turn(params(ctx))
        assert result["status"] == "completed"
        assert len(category_nodes(writer, "event", ctx)) == 1
        assert len(outbox.list_pending(ctx.account_id)) == 2
        assert writer._pipeline.extract.call_count == 0
        assert category_nodes(writer, "event", ctx)[0].metadata["version"] == 1
        assert category_nodes(writer, "session_archive", ctx)[0].metadata["version"] == 1
    finally:
        close_stores(fs, outbox)


def test_before_write_rejection_reconnect_retries_same_candidate_without_model(ctx):
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox,
                                           entries=[candidate(), candidate("entity")])
        original = writer._writer.write_candidate

        def reject_entity(entry, request):
            if entry.category == "entity":
                raise WriteNotStartedError("definite planning rejection")
            return original(entry, request)

        writer._writer.write_candidate = reject_entity
        assert service.after_turn(params(ctx))["status"] == "failed"
        assert len(category_nodes(writer, "event", ctx)) == 1
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "completed"
        assert len(category_nodes(writer, "event", ctx)) == len(category_nodes(writer, "entity", ctx)) == 1
        assert len(outbox.list_pending(ctx.account_id)) == 3
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)


def test_receipt_context_and_category_never_cross_sql_owner_boundaries(ctx):
    fs, outbox = stores()
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        accepted = replace(memory("Atlas", object_name="Atlas", content="Atlas fact."),
                           write_operation_id="shared-operation-id")
        uri = writer.write_memory(accepted, ctx)["target_uri"]
        for changed in [replace(ctx, session_id="other-session"), replace(ctx, agent_id="other-agent")]:
            with pytest.raises(WriteNotStartedError, match="does not match"):
                writer.write_memory(accepted, changed)
        for changed in [replace(ctx, account_id=ctx.account_id + "-other"),
                        replace(ctx, user_id="another-reader")]:
            result = writer.write_memory(accepted, changed)
            assert result["action"] == "create" and result["target_uri"] != uri
        other_category = replace(accepted, category="event")
        result = writer.write_memory(other_category, ctx)
        assert result["action"] == "create" and result["target_uri"] != uri
        assert fs.read_node(uri, ctx).metadata["version"] == 1
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("fail_at", [1, 3])
def test_sql_checkpoint_rejection_is_acknowledged_before_first_memory_write(ctx, fail_at):
    class FailStateSQL(SQLContextFS):
        saves = 0

        def write_node(self, node, request):
            if node.category == "state":
                self.saves += 1
                if self.saves == fail_at:
                    raise RuntimeError("SQL state save rejected")
            super().write_node(node, request)

    fs, outbox = custom_stores(FailStateSQL)
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "failed"
        assert category_nodes(writer, "event", ctx) == []
        assert outbox.list_pending(ctx.account_id) == []
    finally:
        close_stores(fs, outbox)


def test_unknown_sql_storage_error_without_commit_is_not_automatic_retry(ctx):
    class TimeoutSQL(SQLContextFS):
        def write_node_with_outbox(self, node, request, outbox_event=None):
            if node.category == "event":
                raise TimeoutError("unknown write outcome")
            super().write_node_with_outbox(node, request, outbox_event)

    fs, outbox = custom_stores(TimeoutSQL)
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        assert service.after_turn(params(ctx))["status"] == "needs_confirmation"
        assert category_nodes(writer, "event", ctx) == []
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        service, writer, _ = setup_service(fs, ctx, outbox=outbox)
        result = service.after_turn(params(ctx))
        assert result["status"] == "needs_confirmation" and result["retryable"] is False
        assert category_nodes(writer, "event", ctx) == []
        pending = outbox.list_pending(ctx.account_id)
        assert len(pending) == 1
        assert pending[0][0] == category_nodes(writer, "session_archive", ctx)[0].uri
        writer._pipeline.extract.assert_not_called()
    finally:
        close_stores(fs, outbox)
