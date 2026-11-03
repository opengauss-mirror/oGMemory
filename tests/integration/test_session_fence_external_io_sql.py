"""Final-review regressions: real SQL archives and artifact I/O, no models."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import psycopg2
import pytest

from core.models import Role
from features.tool_result_externalization.externalizer import (
    BoundedDaemonIORunner,
    ExternalizationTimeoutError,
)
from features.tool_result_externalization.store_sql import SQLToolResultArtifactStore
from fs.sql_adapter.session_mutation import SessionMutationLostError, active_session_guard
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.session_manager import SessionManager
from session.session_state import SessionState
from tests.integration.test_session_mutation_sql import ctx as ctx
from tests.integration.test_session_mutation_sql import database as database
from tests.unit.features.test_tool_result_externalization import _foundation_metadata


@pytest.mark.parametrize("method", ["commit", "commit_snapshot"])
@pytest.mark.parametrize("required", [True, False])
def test_archive_only_real_sql_manager_rejects_unowned_commit(database, ctx, method, required):
    _, make = database
    fs, _, archive = make()
    held, release = threading.Event(), threading.Event()

    def owner():
        with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            held.set()
            assert release.wait(10)

    manager = SessionManager(get_archive_store=lambda: archive, archive_store_required=required,
                             session_state_bridge_enabled=False)
    manager._compress = Mock(side_effect=AssertionError("unguarded archive reached compression"))
    manager.add_message(ctx.session_id, "user", "Original archive source.", ctx, message_id="source")
    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        args = dict(session_id=ctx.session_id, ctx=ctx, wait=False, archive_id="stable-archive")
        if method == "commit_snapshot":
            args["snapshot"] = list(manager.get_or_create(ctx.session_id).messages)
        try:
            with pytest.raises(SessionMutationLostError, match="requires its guard"):
                getattr(manager, method)(**args)
            assert not holding.done()
            assert archive.list_archives(ctx.session_id, ctx) == []
            assert len(manager.get_or_create(ctx.session_id).messages) == 1
            manager._compress.assert_not_called()
        finally:
            release.set()
            holding.result(timeout=5)


@pytest.mark.parametrize("method", ["commit", "commit_snapshot"])
def test_guarded_sql_manager_keeps_archive_and_acknowledged_state(database, ctx, method):
    _, make = database
    fs, _, archive = make()
    manager = SessionManager(get_archive_store=lambda: archive, archive_store_required=True,
                             get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager._compress = Mock(return_value=("Original source", "Original source"))
    manager.add_message(ctx.session_id, "user", "Original archive source.", ctx, message_id="source")
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        args = dict(session_id=ctx.session_id, ctx=ctx, wait=False, archive_id="stable-archive")
        if method == "commit_snapshot":
            args["snapshot"] = list(manager.get_or_create(ctx.session_id).messages)
        response = getattr(manager, method)(**args)
        assert response["status"] == "completed" and response["archived"]
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert fs.exists(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx)


def test_sql_runner_lazy_store_and_all_artifact_io_keep_owner_pid_and_release_slots(database, ctx):
    _, make = database
    fs, _, _ = make()
    runner = BoundedDaemonIORunner(max_inflight=1)
    text = "A complete artifact source."
    metadata = replace(_foundation_metadata(text), account_id=ctx.account_id, session_id=ctx.session_id,
                       owner_user_id=ctx.user_id, owner_agent_id=ctx.agent_id)
    observed = []
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        # Construction is intentionally lazy, inside the actual I/O closure.
        store = runner.run(lambda: SQLToolResultArtifactStore(pool=fs._shared_pool),
                           deadline=time.monotonic() + 5, operation_key=ctx.session_id)
        original = store._get_connection

        def borrow():
            assert active_session_guard() is guard and threading.get_ident() == guard.thread_id
            connection = original()
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                observed.append(cursor.fetchone()[0])
            return connection

        store._get_connection = borrow
        operations = [lambda: store.write(metadata, text, ctx),
                      lambda: store.read_all(metadata.artifact_id, ctx.session_id, ctx),
                      lambda: store.read_metadata(metadata.artifact_id, ctx.session_id, ctx),
                      lambda: store.read_range(metadata.artifact_id, ctx.session_id, ctx, 0, 16),
                      lambda: store.list(ctx.session_id, ctx),
                      lambda: store.delete_session_artifacts(ctx.session_id, ctx)]
        for operation in operations:
            runner.run(operation, deadline=time.monotonic() + 5, operation_key=ctx.session_id)
            assert not runner.has_inflight(ctx.session_id)
        assert len(observed) == 6 and len(set(observed)) == 1
        assert runner._slots.acquire(blocking=False)
        runner._slots.release()


@pytest.mark.parametrize("field", ["account_id", "user_id", "agent_id", "session_id"])
@pytest.mark.parametrize("method", ["write", "read_all", "read_metadata", "read_range", "list", "delete_session_artifacts"])
def test_sql_artifact_every_public_method_checks_full_guard_identity(database, ctx, field, method):
    _, make = database
    fs, _, _ = make()
    store = SQLToolResultArtifactStore(pool=fs._shared_pool)
    foreign = replace(ctx, **{field: "another-identity"})
    text = "Authorized content."
    metadata = replace(_foundation_metadata(text), account_id=foreign.account_id,
                       session_id=foreign.session_id, owner_user_id=foreign.user_id,
                       owner_agent_id=foreign.agent_id)
    with pytest.raises(SessionMutationLostError), fs.session_mutation_lock(
            SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        if method == "write":
            store.write(metadata, text, foreign)
        elif method in ("read_all", "read_metadata"):
            getattr(store, method)(metadata.artifact_id, foreign.session_id, foreign)
        elif method == "read_range":
            store.read_range(metadata.artifact_id, foreign.session_id, foreign, 0, 16)
        else:
            getattr(store, method)(foreign.session_id, foreign)


def test_sql_inline_timeout_and_error_cleanup_keep_slots_and_inflight_balanced(database, ctx):
    _, make = database
    fs, _, _ = make()
    runner = BoundedDaemonIORunner(max_inflight=1)
    calls = []
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        def slow():
            calls.append(threading.get_ident())
            time.sleep(.04)
        with pytest.raises(ExternalizationTimeoutError):
            runner.run(slow, deadline=time.monotonic() + .02, operation_key=ctx.session_id)
        assert len(calls) == 1 and not runner.has_inflight(ctx.session_id)
        assert runner.run(lambda: "next", deadline=time.monotonic() + 5, operation_key=ctx.session_id) == "next"
        with pytest.raises(ValueError, match="ordinary I/O rejection"):
            runner.run(lambda: (_ for _ in ()).throw(ValueError("ordinary I/O rejection")),
                       deadline=time.monotonic() + 5, operation_key=ctx.session_id)
        assert not runner.has_inflight(ctx.session_id) and runner._slots.acquire(blocking=False)
        runner._slots.release()


@pytest.mark.parametrize("entry", ["first_view", "after_turn", "add_message", "compose"])
def test_public_artifact_after_owner_loss_and_fenced_delete_cannot_publish(database, ctx, entry):
    dsn, _ = database
    cfg = OgMemConfig(storage_backend="sql", sql_connection_string=dsn, provider="mock",
                      tool_result_externalization_enabled=True, tool_result_single_chars=500,
                      tool_result_write_timeout_ms=5000, after_turn_threshold=100000,
                      session_state_bridge_enabled=False)
    service, latest = MemoryService(config=cfg), MemoryService(config=cfg)
    for instance in (service, latest):
        instance._flush_session_tool_usage_async = Mock()
        instance.get_llm = Mock(side_effect=AssertionError("artifact path initialized a model"))
    entered, release = threading.Event(), threading.Event()
    observed = {}
    original = service._tool_result_store.write

    def delayed(metadata, text, request):
        guard = active_session_guard()
        assert guard is not None and guard.thread_id == threading.get_ident()
        with guard.connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            observed["pid"] = cursor.fetchone()[0]
        guard.connection.rollback()
        entered.set()
        assert release.wait(10)
        return original(metadata, text, request)

    service._tool_result_store.write = delayed
    request = dict(sessionId=ctx.session_id, adapterId="regression.test", turnId="turn-one",
                   toolCallId="call-one", toolName="sql_result", _ctx=ctx,
                   content=[{"type": "text", "text": "Observed test component reading 25 units.\n" * 150}])

    def dispatch():
        message = {"role": "toolResult", "toolCallId": "call-one", "toolName": "sql_result", "content": request["content"]}
        if entry == "first_view":
            return service.govern_first_view_tool_result(request)
        if entry == "add_message":
            return service.add_session_message({"sessionId": ctx.session_id, "_ctx": ctx, "message": message})
        return getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx, "messages": [message]})

    control = psycopg2.connect(dsn)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(dispatch)
            assert entered.wait(5)
            with control.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(%s)", (observed["pid"],))
                assert cursor.fetchone()[0]
            control.commit()
            deleted = latest.delete_session({"sessionId": ctx.session_id, "_ctx": replace(ctx, role=Role.ADMIN)})
            assert deleted["cleaned"] and deleted["status"] == 200
            release.set()
            assert pending.result(timeout=10)["status"] == "needs_confirmation"
        with control.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM tool_result_artifacts WHERE account_id=%s AND session_id=%s",
                           (ctx.account_id, ctx.session_id))
            assert cursor.fetchone()[0] == 0
        assert not service._tool_result_io_runner.has_inflight(ctx.session_id)
    finally:
        release.set()
        control.close()
        for instance in (service, latest):
            instance.get_llm.assert_not_called()
            if instance._sql_pool is not None:
                instance._sql_pool.close_all()


@pytest.mark.parametrize("entry", ["first_view", "after_turn", "add_message", "compose"])
def test_sql_public_governance_paths_run_actual_artifact_io_under_owner(database, ctx, entry):
    dsn, _ = database
    service = MemoryService(config=OgMemConfig(
        storage_backend="sql", sql_connection_string=dsn, provider="mock",
        tool_result_externalization_enabled=True, tool_result_single_chars=500,
        after_turn_threshold=100000, session_state_bridge_enabled=False))
    service._flush_session_tool_usage_async = Mock()
    service.get_llm = Mock(side_effect=AssertionError("artifact governance initialized a model"))
    service._get_shared_embedder = Mock(side_effect=AssertionError("artifact governance initialized embeddings"))
    original = service._tool_result_store.write
    observed = []

    def owned_write(metadata, text, request):
        guard = active_session_guard()
        assert guard is not None and guard.thread_id == threading.get_ident()
        guard.ensure_owned()
        observed.append(guard)
        return original(metadata, text, request)

    service._tool_result_store.write = owned_write
    content = [{"type": "text", "text": "Observed test component reading 25 units.\n" * 150}]
    message = {"role": "toolResult", "toolCallId": "governance-call", "toolName": "sql_result", "content": content}
    request = {"sessionId": ctx.session_id, "_ctx": ctx}
    try:
        if entry == "first_view":
            response = service.govern_first_view_tool_result({**request, "adapterId": "regression.test",
                "turnId": "turn-one", "toolCallId": "governance-call", "toolName": "sql_result", "content": content})
        elif entry == "after_turn":
            response = service.after_turn({**request, "messages": [message]})
        elif entry == "add_message":
            response = service.add_session_message({**request, "message": message})
        else:
            response = service.compose({**request, "messages": [message]})
        assert response.get("status") != "needs_confirmation", response
        assert observed
        assert len(service._tool_result_store.list(ctx.session_id, ctx)["artifacts"]) == 1
        assert not service._tool_result_io_runner.has_inflight(ctx.session_id)
        service.get_llm.assert_not_called()
        service._get_shared_embedder.assert_not_called()
    finally:
        if service._sql_pool is not None:
            service._sql_pool.close_all()


def _seed_archives(fs, archive, ctx):
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        for index in range(2):
            assert archive.write_archive(session_id=ctx.session_id, ctx=ctx, archive_id=f"seed-{index}",
                overview="Original overview", abstract="Original abstract",
                messages=[{"role": "user", "content": f"Original source {index}"}]).success


def test_optional_auto_merge_late_sql_getter_rejects_before_first_list(database, ctx):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    generic, current = Mock(), []
    generic.list_archives.return_value = []
    current.append(generic)

    def generic_write(**kwargs):
        current[0] = archive
        return Mock(success=True, uri="generic://archive", archive_id=kwargs["archive_id"])

    generic.write_archive.side_effect = generic_write
    manager = SessionManager(get_archive_store=lambda: current[0], archive_merge_threshold=2, archive_max_count=1)
    manager._compress = Mock(return_value=("New original source", "New original source"))
    manager.add_message(ctx.session_id, "user", "New original source.", ctx, message_id="new-source")
    attempted, held, release = threading.Event(), threading.Event(), threading.Event()
    original_merge = manager._merge_archives_once
    original_list = archive.list_archives
    archive.list_archives = Mock(wraps=original_list)
    archive.write_archive = Mock(wraps=archive.write_archive)

    def merge_once(*args):
        try:
            return original_merge(*args)
        finally:
            attempted.set()

    manager._merge_archives_once = merge_once

    def owner():
        with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        try:
            result = manager.commit(ctx.session_id, ctx, wait=True, archive_id="generic-new")
            assert result["status"] == "completed"
            assert attempted.wait(5) and not holding.done()
            archive.list_archives.assert_not_called()
            archive.write_archive.assert_not_called()
            assert len(original_list(ctx.session_id, ctx)) == 2
        finally:
            release.set()
            holding.result(timeout=5)


@pytest.mark.parametrize("method", ["delete_session_archives", "delete_persisted_session_state"])
def test_direct_sql_cleanup_rejects_while_another_session_owner_is_active(database, ctx, method):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    manager = SessionManager(get_archive_store=lambda: archive, get_context_fs=lambda: fs)
    manager.add_message(ctx.session_id, "user", "Original state source.", ctx, message_id="state-source")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with fs.session_mutation_lock(uri, ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
    before = fs.read_node(uri, ctx).content
    if method == "delete_session_archives":
        # The reported bypass is the public fixed SQL getter with no ContextFS.
        manager = SessionManager(get_archive_store=lambda: archive)
    held, release = threading.Event(), threading.Event()

    def owner():
        with fs.session_mutation_lock(uri, ctx):
            held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        try:
            with pytest.raises(SessionMutationLostError):
                getattr(manager, method)(ctx.session_id, ctx)
            assert not holding.done()
            assert len(archive.list_archives(ctx.session_id, ctx)) == 2
            assert fs.read_node(uri, ctx).content == before
        finally:
            release.set()
            holding.result(timeout=5)
    with fs.session_mutation_lock(uri, ctx):
        assert getattr(manager, method)(ctx.session_id, ctx) == (2 if method == "delete_session_archives" else 1)


@pytest.mark.parametrize("fail_mark", [False, True])
def test_actual_sql_merge_all_mutations_keep_one_guarded_store_and_connection(database, ctx, fail_mark):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    manager = SessionManager(get_archive_store=lambda: archive, archive_merge_threshold=2, archive_max_count=1)
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    seen = []
    with fs.session_mutation_lock(uri, ctx) as guard:
        original_borrow = archive._get_connection

        def borrow():
            assert active_session_guard() is guard and threading.get_ident() == guard.thread_id
            connection = original_borrow()
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                seen.append(cursor.fetchone()[0])
            return connection

        archive._get_connection = borrow
        if fail_mark:
            original_mark = archive.mark_archive_merged
            calls = []

            def mark(session, archive_id, request, merged_id):
                calls.append(archive_id)
                return original_mark(session, archive_id, request, merged_id) if len(calls) == 1 else False

            archive.mark_archive_merged = mark
        assert manager._merge_archives_once(ctx.session_id, ctx) is not fail_mark
        guard.ensure_owned()
        active = archive.list_archives(ctx.session_id, ctx)
        if fail_mark:
            assert {entry.archive_id for entry in active} == {"seed-0", "seed-1"}
        else:
            assert len(active) == 1 and active[0].archive_id.startswith("merged_")
        assert len(seen) >= 6 and len(set(seen)) == 1


@pytest.mark.parametrize("field", ["account_id", "user_id", "agent_id", "session_id"])
@pytest.mark.parametrize("method", ["_merge_archives_once", "delete_session_archives"])
def test_archive_mutation_helper_rejects_full_context_mismatch_before_access(database, ctx, field, method):
    _, make = database
    fs, _, archive = make()
    manager = SessionManager(get_archive_store=lambda: archive)
    request = replace(ctx, **{field: "foreign-identity"})
    archive.list_archives = Mock(side_effect=AssertionError("mismatched mutation reached first list"))
    archive.delete_session_archives = Mock(side_effect=AssertionError("mismatched mutation reached delete"))
    with pytest.raises(SessionMutationLostError), fs.session_mutation_lock(
            SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        getattr(manager, method)(request.session_id, request)
    archive.list_archives.assert_not_called()
    archive.delete_session_archives.assert_not_called()
