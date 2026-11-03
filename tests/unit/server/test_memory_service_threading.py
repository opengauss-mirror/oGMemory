from __future__ import annotations

import threading
import time
import logging
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


@pytest.mark.parametrize("archive_failed", [False, True])
def test_background_completion_marker_waits_for_snapshot_commit(caplog, archive_failed):
    caplog.set_level(logging.INFO, logger="ogmem.service")
    service = MemoryService(config=OgMemConfig(
        after_turn_threshold=1, session_state_bridge_enabled=False,
    ))
    session_id = "archive-completion-barrier"
    ctx = _threaded_ctx(session_id)
    service._write_api = Mock()
    service._async_drain = Mock()
    service._chunk_and_extract = Mock(return_value=_successful_write_result())
    mgr = service.get_session_manager()
    mgr.save_session_state = Mock(return_value=True)

    def commit(*args, **kwargs):
        assert "after_turn background extract done:" not in caplog.text
        return {"status": "failed", "error": "archive failed"} if archive_failed else {
            "status": "completed", "archived": True,
        }

    mgr.commit_snapshot = Mock(side_effect=commit)
    service.after_turn({
        "sessionId": session_id,
        "messages": [{"role": "user", "content": "a" * 100}],
        "_ctx": ctx,
    })
    assert mgr.get_or_create(session_id, ctx=ctx).extraction_done_event.wait(timeout=3)
    mgr.commit_snapshot.assert_called_once()
    assert ("after_turn background extract done:" in caplog.text) is not archive_failed


def test_session_manager_lazy_initialization_is_thread_safe():
    service = MemoryService(config=OgMemConfig())
    barrier = threading.Barrier(16)
    managers = []

    def get_manager() -> None:
        barrier.wait()
        managers.append(service.get_session_manager())

    threads = [threading.Thread(target=get_manager) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(managers) == 16
    assert len({id(manager) for manager in managers}) == 1


def test_concurrent_after_turn_claims_each_message_once():
    service = MemoryService(
        config=OgMemConfig(
            after_turn_threshold=1,
            session_state_bridge_enabled=False,
        )
    )
    ctx = RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="session-threaded",
        trace_id="trace-threaded",
    )
    service._write_api = Mock()
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr.save_session_state = Mock(return_value=True)

    claim_barrier = threading.Barrier(2)
    buf = mgr.get_or_create("session-threaded", ctx=ctx)
    real_claim = buf.claim_incremental_extraction

    def synchronized_claim():
        claim_barrier.wait(timeout=3)
        return real_claim()

    buf.claim_incremental_extraction = synchronized_claim

    extract_started = threading.Event()
    release_extract = threading.Event()
    extraction_message_ids = []

    def fake_extract(**kwargs):
        extraction_message_ids.append(tuple(message["id"] for message in kwargs["messages"]))
        extract_started.set()
        release_extract.wait(timeout=3)
        return {
            "ok": True,
            "candidates_extracted": 1,
            "candidates_filtered": 0,
            "writes_completed": 1,
            "writes_skipped": 0,
            "writes_failed": 0,
            "plans": [],
        }

    service._chunk_and_extract = fake_extract
    mgr.commit_snapshot = Mock(
        return_value={"ok": True, "archived": True, "status": "completed"}
    )

    results = []

    def invoke_after_turn(index: int):
        results.append(service.after_turn({
            "sessionId": "session-threaded",
            "messages": [{
                "role": "user",
                "content": f"message-{index}-" + ("x" * 80),
            }],
            "_ctx": ctx,
        }))

    request_threads = [
        threading.Thread(target=invoke_after_turn, args=(index,))
        for index in range(2)
    ]
    for thread in request_threads:
        thread.start()
    for thread in request_threads:
        thread.join(timeout=3)

    assert extract_started.wait(timeout=3)
    assert len(extraction_message_ids) == 1
    assert len(extraction_message_ids[0]) == 2
    assert sorted(result["status"] for result in results) == ["processing", "processing"]

    release_extract.set()
    assert buf.extraction_done_event.wait(timeout=3)
    mgr.commit_snapshot.assert_called_once()


def test_after_turn_returns_retryable_while_session_extraction_active():
    service = MemoryService(
        config=OgMemConfig(
            after_turn_threshold=1,
            session_state_bridge_enabled=False,
        )
    )
    session_id = "session-single-flight"
    ctx = _threaded_ctx(session_id)
    service._write_api = Mock()
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr.save_session_state = Mock(return_value=True)
    mgr.commit_snapshot = Mock(
        return_value={"ok": True, "archived": True, "status": "completed"}
    )

    extract_started = threading.Event()
    release_extract = threading.Event()

    def fake_extract(**_kwargs):
        extract_started.set()
        release_extract.wait(timeout=3)
        return _successful_write_result()

    service._chunk_and_extract = Mock(side_effect=fake_extract)

    first = service.after_turn({
        "sessionId": session_id,
        "messages": [{"role": "user", "content": "first " + ("x" * 80)}],
        "_ctx": ctx,
    })
    assert first["status"] == "processing"
    assert extract_started.wait(timeout=3)

    second = service.after_turn({
        "sessionId": session_id,
        "messages": [{"role": "user", "content": "second " + ("y" * 80)}],
        "_ctx": ctx,
        "wait": True,
    })

    assert second["ok"] is False
    assert second["status"] == "processing"
    assert second["retryable"] is True
    assert mgr.get_or_create(session_id, ctx=ctx).extraction_active_count == 1
    assert service._chunk_and_extract.call_count == 1

    release_extract.set()
    assert mgr.get_or_create(session_id, ctx=ctx).extraction_done_event.wait(timeout=3)


def _threaded_ctx(session_id: str) -> RequestContext:
    return RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id=session_id,
        trace_id=f"trace-{session_id}",
    )


def _successful_write_result() -> dict:
    return {
        "ok": True,
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [],
    }


def test_dispose_preserves_messages_added_after_its_snapshot():
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._get_context_fs = Mock(return_value=None)
    mgr = service.get_session_manager()
    session_id = "dispose-race"
    ctx = _threaded_ctx(session_id)
    initial = "initial-" + ("a" * 1000)
    late = "late-" + ("b" * 400)
    mgr.add_message(session_id, "user", initial, ctx)
    buf = mgr.get_or_create(session_id, ctx=ctx)

    write_api_requested = threading.Event()
    release_write_api = threading.Event()
    flushed_contents = []

    def blocked_write_api():
        write_api_requested.set()
        release_write_api.wait(timeout=3)
        return object()

    def fake_extract(**kwargs):
        flushed_contents.extend(message["content"] for message in kwargs["messages"])
        return _successful_write_result()

    service.get_write_api = blocked_write_api
    service._chunk_and_extract = fake_extract
    service.drain_outbox_sync = Mock(return_value={"processed": 0})
    mgr.save_session_state = Mock(return_value=True)

    results = []
    dispose_thread = threading.Thread(
        target=lambda: results.append(service.dispose({"sessionId": session_id, "_ctx": ctx}))
    )
    dispose_thread.start()
    assert write_api_requested.wait(timeout=3)
    mgr.add_message(session_id, "user", late, ctx)
    release_write_api.set()
    dispose_thread.join(timeout=3)

    assert buf.extraction_done_event.wait(timeout=3)
    assert results[0]["flushed"] is False
    assert initial in flushed_contents
    assert late not in flushed_contents
    assert mgr.has_session(session_id, ctx=ctx)
    remaining = mgr.get_or_create(session_id, ctx=ctx).messages
    assert [message.content for message in remaining] == [late]


def test_dispose_fast_path_preserves_messages_added_during_cleanup():
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._get_context_fs = Mock(return_value=None)
    mgr = service.get_session_manager()
    session_id = "dispose-fast-race"
    ctx = _threaded_ctx(session_id)
    initial = "small pending message"
    late = "late message"
    mgr.add_message(session_id, "user", initial, ctx)

    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(*_args, **_kwargs):
        save_started.set()
        release_save.wait(timeout=3)

    service._save_session_state_best_effort = blocked_save
    result_holder = []
    dispose_thread = threading.Thread(
        target=lambda: result_holder.append(
            service.dispose({"sessionId": session_id, "_ctx": ctx})
        )
    )
    dispose_thread.start()

    assert save_started.wait(timeout=3)
    mgr.add_message(session_id, "user", late, ctx)
    release_save.set()
    dispose_thread.join(timeout=3)

    assert result_holder[0]["reason"] == "below_token_threshold"
    assert mgr.has_session(session_id, ctx=ctx)
    remaining = mgr.get_or_create(session_id, ctx=ctx).messages
    assert [message.content for message in remaining] == [late]


def test_dispose_flush_failure_keeps_snapshot_for_retry():
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._get_context_fs = Mock(return_value=None)
    mgr = service.get_session_manager()
    session_id = "dispose-failure"
    ctx = _threaded_ctx(session_id)
    content = "important-" + ("x" * 1000)
    mgr.add_message(session_id, "user", content, ctx)
    buf = mgr.get_or_create(session_id, ctx=ctx)
    service.get_write_api = Mock(return_value=object())
    service._chunk_and_extract = Mock(side_effect=RuntimeError("simulated failure"))

    result = service.dispose({"sessionId": session_id, "_ctx": ctx})

    assert result["status"] == "background_flush_dispatched"
    assert result["flushed"] is False
    assert buf.extraction_done_event.wait(timeout=3)
    assert mgr.has_session(session_id, ctx=ctx)
    assert [message.content for message in buf.messages] == [content]
    assert buf.extraction_watermark == 0
    assert buf.dispose_in_progress is False


def test_dispose_does_not_block_request_thread_while_extraction_is_active():
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    service._get_context_fs = Mock(return_value=None)
    mgr = service.get_session_manager()
    session_id = "dispose-deferred"
    ctx = _threaded_ctx(session_id)
    buf = mgr.get_or_create(session_id, ctx=ctx)
    buf.add("user", "already extracted")
    buf.extraction_watermark = len(buf.messages)
    buf.begin_extraction()
    extraction_wait = Mock(wraps=buf.extraction_done_event.wait)
    buf.extraction_done_event.wait = extraction_wait

    started_at = time.monotonic()
    result = service.dispose({"sessionId": session_id, "_ctx": ctx})
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert result["status"] == "dispose_deferred"
    assert result["reason"] == "extraction_in_progress"
    assert mgr.has_session(session_id, ctx=ctx)

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not extraction_wait.called:
        time.sleep(0.01)
    extraction_wait.assert_called_once_with(timeout=600)

    buf.end_extraction()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and mgr.has_session(session_id, ctx=ctx):
        time.sleep(0.01)
    assert not mgr.has_session(session_id, ctx=ctx)
