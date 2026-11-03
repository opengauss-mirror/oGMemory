"""Independent service caches share AGFS session ownership."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from core.errors import ToolCallProtocolError
from core.models import RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.extraction_checkpoint import EXTENSION_KEY
from service.extraction_recovery_discovery import DISCOVERY_KEY, ExtractionRecoveryDiscovery
from session.archive_store import SessionArchiveStore
from session.session_manager import SessionManager
from session.session_state import SessionState
from tests.unit.service.test_extraction_recovery_discovery import state
from tests.unit.server.test_extraction_batch_recovery import params, setup_service


def context_fs(address):
    return AGFSContextFS(AGFSClient(address), mount_prefix=os.environ.get("OGMEM_AGFS_TEST_MOUNT", "/local"))


@pytest.mark.parametrize("retry_first", [False, True])
def test_services_reload_completed_source_under_shared_lease(retry_first):
    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    ctx = RequestContext("recovery-" + uuid4().hex, "reader", "agent", "source", "trace")
    first_fs = context_fs(address)
    first, writer, manager = setup_service(first_fs, ctx)
    first._extraction_retry.wake = Mock()
    extract = writer._pipeline.extract.side_effect
    writer._pipeline.extract.side_effect = ToolCallProtocolError("temporary response failure")
    task = first.after_turn({**params(ctx), "retryInBackground": True})["recovery_task"]
    with first._session_mutation_scope(ctx):
        manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]["next_attempt_at"] = 0
        assert manager.save_session_state(ctx.session_id, ctx)
    second, second_writer, _ = setup_service(context_fs(address), ctx)
    second._load_extraction_recovery_before_accumulation(ctx.session_id, ctx)
    entered, release = Event(), Event()

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return extract(*args, **kwargs)

    writer._pipeline.extract.side_effect = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(first._extraction_retry.run_once, ctx, task["id"]) if retry_first else pool.submit(
            first.after_turn, params(ctx))
        try:
            assert entered.wait(5)
            assert second._extraction_retry.run_once(ctx, task["id"]) > 0
        finally:
            release.set()
        active.result(timeout=10)
    assert second._extraction_retry.run_once(ctx, task["id"]) is None
    buf = second.get_session_manager().get_or_create(ctx.session_id)
    assert buf.request_completed("original")
    assert buf.messages == []
    assert second_writer._pipeline.extract.call_count == 0
    assert first_fs.read_node(f"ctx://{ctx.account_id}/sessions/source/state.json", ctx).category == "state"


@pytest.mark.parametrize("filename", ["content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"])
@pytest.mark.parametrize("pause_at", ["before_write", "after_write", "before_publish", "after_publish", "during_reclaim"])
def test_successor_checkpoint_survives_expired_writer(filename, pause_at, monkeypatch):
    import json
    from datetime import UTC, datetime, timedelta

    import fs.agfs_adapter.agfs_context_fs as agfs_module
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from session.session_state import SessionState

    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    monkeypatch.setattr(agfs_module, "_RESOURCE_MUTATION_LOCK_RENEW_INTERVAL_SECONDS", 3600)
    ctx = RequestContext("fence-" + uuid4().hex, "reader", "agent", "source", "trace")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)

    def service():
        fs = context_fs(address)
        svc = MemoryService(OgMemConfig(session_state_bridge_enabled=False))
        svc._get_context_fs = lambda: fs
        manager = svc.get_session_manager()
        manager._get_context_fs = lambda: fs
        return fs, svc, manager

    old_fs, old_service, old_manager = service()
    new_fs, new_service, new_manager = service()
    paused, resume, staged, reclaim = Event(), Event(), Event(), Event()
    lease_holder = []
    original_write, original_move = old_fs._client.write, old_fs._client.mv

    def pause(stage, path):
        if not path.endswith(filename):
            return
        if stage == "before_write" and pause_at == "during_reclaim":
            lease_holder.append(old_fs.active_session_recovery()[0])
            paused.set()
            assert reclaim.wait(10)
        elif stage == "before_publish" and pause_at == "during_reclaim":
            staged.set()
            assert resume.wait(10)
        elif stage == pause_at:
            lease_holder.append(old_fs.active_session_recovery()[0])
            paused.set()
            assert resume.wait(10)

    def write(path, content, *args, **kwargs):
        pause("before_write", path)
        result = original_write(path, content, *args, **kwargs)
        pause("after_write", path)
        return result

    def move(source, destination, *args, **kwargs):
        if "/.publish-" in source:
            pause("before_publish", destination)
        result = original_move(source, destination, *args, **kwargs)
        if "/.publish-" in source:
            pause("after_publish", destination)
        return result

    monkeypatch.setattr(old_fs._client, "write", write)
    monkeypatch.setattr(old_fs._client, "mv", move)

    def save(svc, manager, request):
        with svc._session_mutation_scope(ctx):
            buf = manager.get_or_create(ctx.session_id, ctx=ctx)
            manager.add_message(ctx.session_id, "user", request, ctx, message_id=request)
            buf.completed_request_ids = [request]
            return manager.save_session_state(ctx.session_id, ctx)

    def old_save():
        try:
            return save(old_service, old_manager, "old-completed")
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(old_save)
        try:
            assert paused.wait(5)
            lease = lease_holder[0]
            lease_path = lease.lock_path + f"/lease-{lease.owner_token}.json"
            expiry = json.loads(new_fs._read_file(lease_path))
            expiry["lease_updated_at"] = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
            new_fs._write_file(lease_path, json.dumps(expiry))

            if pause_at == "during_reclaim":
                original_remove = new_fs._client.rm

                def remove(path, *args, **kwargs):
                    if path == lease.lock_path and not reclaim.is_set():
                        reclaim.set()
                        assert staged.wait(5)
                    return original_remove(path, *args, **kwargs)

                monkeypatch.setattr(new_fs._client, "rm", remove)

            assert save(new_service, new_manager, "new-completed")
            node_path = new_fs._uri_to_agfs_path(uri)
            paths = [node_path + name for name in (
                "content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json")]
            committed = {path: new_fs._read_file(path) for path in paths}
        finally:
            reclaim.set()
            resume.set()
        future.result(timeout=5)

    assert {path: new_fs._read_file(path) for path in paths} == committed
    assert json.loads(new_fs.read_node(uri, ctx).content)["completed_request_ids"] == ["new-completed"]


@pytest.mark.parametrize("metadata_hint", ["empty", "previous", "missing"])
def test_published_source_recovers_with_earlier_metadata(metadata_hint):
    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    ctx = RequestContext("discovery-" + uuid4().hex, "reader", "agent", "source", "trace")
    fs = context_fs(address)
    first, writer, manager = setup_service(fs, ctx)
    first._extraction_retry.wake = Mock()
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    path = fs._uri_to_agfs_path(uri)
    with first._session_mutation_scope(ctx):
        manager.get_or_create(ctx.session_id, ctx=ctx)
        assert manager.save_session_state(ctx.session_id, ctx)
    metadata = json.loads(fs._read_file(path + fs.FILE_META))
    if metadata_hint == "previous":
        metadata[DISCOVERY_KEY] = state(ctx).metadata[DISCOVERY_KEY]
    elif metadata_hint == "missing":
        metadata.pop(DISCOVERY_KEY)

    writer._pipeline.extract.side_effect = ToolCallProtocolError("interrupted extraction")
    task = first.after_turn({**params(ctx), "retryInBackground": True})["recovery_task"]
    with first._session_mutation_scope(ctx):
        manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]["next_attempt_at"] = 0
        assert manager.save_session_state(ctx.session_id, ctx)
    fs._write_file(path + fs.FILE_META, json.dumps(metadata))

    completed_ctx = replace(ctx, session_id="a-completed")
    completed = state(completed_ctx)
    completed.content = json.dumps({"completed_request_ids": ["completed"], "runtime_extensions": {}})
    fs.write_node(completed, completed_ctx)
    damaged_ctx = replace(ctx, session_id="b-damaged")
    damaged = state(damaged_ctx)
    fs.write_node(damaged, damaged_ctx)
    fs._write_file(fs._uri_to_agfs_path(damaged.uri) + fs.FILE_CONTENT, '{"runtime_extensions":')

    restarted_fs = context_fs(address)
    discover = ExtractionRecoveryDiscovery(lambda: restarted_fs, lambda: [ctx.account_id])
    found = discover()
    assert [(owner.session_id, recovered["id"]) for owner, recovered in found] == [(ctx.session_id, task["id"])]

    restarted, _, manager = setup_service(restarted_fs, ctx)
    store = SessionArchiveStore(restarted_fs)
    manager.commit_snapshot = SessionManager.commit_snapshot.__get__(manager)
    manager._get_archive_store = lambda: store
    manager._archive_store_required = True
    manager._get_llm = lambda: None
    restarted._get_context_fs = lambda: restarted_fs
    restarted._extraction_recovery_account_ids = lambda: [ctx.account_id]
    try:
        restarted.start_extraction_recovery()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if manager.get_or_create(ctx.session_id, ctx=ctx).request_completed("original"):
                break
            time.sleep(.02)
    finally:
        restarted.shutdown(join_timeout=5)
    saved = json.loads(restarted_fs.read_node(uri, ctx).content)
    assert "original" in saved["completed_request_ids"]
    archive = store.read_archive(ctx.session_id, task["id"], ctx)
    assert [message["content"] for message in archive.messages] == ["I repaired the receiver."]


@pytest.mark.parametrize("archive_failure", [False, True])
def test_buffer_commit_preserves_source_until_archive_completes(archive_failure):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    ctx = RequestContext("archive-" + uuid4().hex, "reader", "agent", "source", "trace")
    fs = context_fs(address)
    store = SessionArchiveStore(fs)

    def service():
        instance = MemoryService(OgMemConfig(session_state_bridge_enabled=False))
        instance._get_context_fs = lambda: fs
        manager = instance.get_session_manager()
        manager._get_context_fs = lambda: fs
        manager._get_archive_store = lambda: store
        manager._archive_store_required = True
        manager._get_llm = lambda: None
        return instance, manager

    first, manager = service()
    first.add_session_message({"_ctx": ctx, "sessionId": ctx.session_id,
                               "role": "user", "content": "Preserve this source."})
    process = manager._process_snapshot

    def archive(*args, **kwargs):
        lease, _ = fs.active_session_recovery()
        lease.ensure_owned()
        state = json.loads(fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx).content)
        assert [message["content"] for message in state["buffer_messages"]] == ["Preserve this source."]
        if archive_failure:
            raise RuntimeError("temporary archive write failure")
        return process(*args, **kwargs)

    manager._process_snapshot = archive
    result = first.commit_session_buffer({"_ctx": ctx, "sessionId": ctx.session_id, "wait": False})
    expected = ["Preserve this source."]
    if archive_failure:
        assert result["status"] == "failed"
        restarted, _ = service()
        restarted.add_session_message({"_ctx": ctx, "sessionId": ctx.session_id,
                                        "role": "user", "content": "Followup message."})
        expected.append("Followup message.")
        result = restarted.commit_session_buffer({"_ctx": ctx, "sessionId": ctx.session_id, "wait": False})
    assert result["status"] == "completed"
    archive = store.read_archive(ctx.session_id, result["archive_id"], ctx)
    assert [message["content"] for message in archive.messages] == expected
