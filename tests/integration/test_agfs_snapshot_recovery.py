"""Exercise committed node recovery against an isolated native AGFS server."""

import json
import threading
import time
from dataclasses import asdict
from pathlib import PurePosixPath

import pytest
from pyagfs import AGFSClientError

from core.operation_deadline import operation_deadline
from service.api import MemoryWriteAPI
from tests.integration.test_agfs_node_snapshot_reads import native_agfs, _runtime, _preference
from tests.unit.commit.test_memory_identity import memory


@pytest.mark.parametrize("method", ["read_node", "read_node_fields"])
def test_reader_preserves_successor_writer_snapshot(native_agfs, monkeypatch, method):
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "lock-takeover")
    other_client, writer, _ = _runtime(address, "lock-takeover")
    fs.write_node(_preference(ctx, 1), ctx)
    remove = client.rm
    def keep_lock(path, *args, **kwargs):
        if str(path).endswith(".update-lock"):
            raise AGFSClientError("cleanup interrupted")
        return remove(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(client, "rm", keep_lock)
        fs.write_node(_preference(ctx, 2), ctx)
    partial = threading.Event()
    errors = []
    write_files = writer._write_node_files
    def slow_write(path, node, *args, **kwargs):
        writer._write_file(path + writer.FILE_CONTENT, node.content)
        partial.set()
        time.sleep(.5)
        return write_files(path, node, *args, **kwargs)
    monkeypatch.setattr(writer, "_write_node_files", slow_write)
    def write_next():
        try:
            writer.write_node(_preference(ctx, 3), ctx)
        except Exception as exc:
            errors.append(type(exc).__name__)
    worker = threading.Thread(target=write_next)
    started = False
    def takeover():
        nonlocal started
        if started:
            return
        started = True
        path = fs._uri_to_agfs_path(_preference(ctx, 2).uri)
        final = PurePosixPath(path.rstrip("/"))
        remove(str(final.parent) + f"/.{final.name}.update-lock", recursive=True)
        worker.start()
        assert partial.wait(5)
    lease = fs._read_update_lease
    calls = 0
    def read_lease(*args):
        nonlocal calls
        result = lease(*args)
        calls += 1
        if calls == 2:
            takeover()
        return result
    read_method = "_read_node_once" if method == "read_node" else "_read_node_fields_once"
    read_once = getattr(fs, read_method)
    def read_and_takeover(*args):
        result = read_once(*args)
        takeover()
        return result
    monkeypatch.setattr(fs, "_read_update_lease", read_lease)
    monkeypatch.setattr(fs, read_method, read_and_takeover)
    emit("REQUEST concurrent write/read", {"write": asdict(_preference(ctx, 3)), "read_uri": _preference(ctx, 2).uri})
    try:
        uri = _preference(ctx, 2).uri
        result = fs.read_node(uri, ctx) if method == "read_node" else fs.read_node_fields(uri, ctx, "all")
        emit("RESPONSE " + method, asdict(result))
    finally:
        if started:
            worker.join(10)
    emit("RESPONSE successor write", {"errors": errors, "finished": not worker.is_alive()})
    expected = _preference(ctx, result.metadata["version"])
    assert (result.abstract, result.overview, result.content) == (expected.abstract, expected.overview, expected.content)
    assert started and not worker.is_alive() and errors == []
    client.session.close()
    other_client.session.close()


def test_prewrite_read_expiry_resumes_checkpoint(native_agfs, monkeypatch):
    from session.session_state import SessionState
    from tests.unit.server.test_extraction_batch_recovery import setup_service, params
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "prewrite-retry")
    service, api, manager = setup_service(fs, ctx)
    manager._get_context_fs = lambda: fs
    build = api._writer._archive_builder.build
    def delayed_build(*args, **kwargs):
        node = build(*args, **kwargs)
        if node.category == "event":
            time.sleep(1.1)
        return node
    request = {**params(ctx), "timeoutSeconds": 1}
    emit("REQUEST after_turn", request)
    with monkeypatch.context() as patch:
        patch.setattr(api._writer._archive_builder, "build", delayed_build)
        first = service.after_turn(request)
    emit("RESPONSE after_turn", first)
    saved = fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx)
    state = json.loads(saved.content)
    emit("RESPONSE persisted checkpoint", state)
    batch = state["runtime_extensions"]["memory_extraction_recovery"]["batch"]
    assert batch["slots"]["chunk:0"]["operations"]["0"]["status"] == "rejected_before_write"
    emit("REQUEST retry after_turn", {**request, "timeoutSeconds": 10})
    retried = service.after_turn({**request, "timeoutSeconds": 10})
    emit("RESPONSE retry after_turn", retried)
    assert first["request_status"] == "RECOVERING"
    assert first["retry_action"] == "poll" and first["http_status"] == 202
    assert retried["status"] == "completed"
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    client.session.close()


def emit(kind, payload):
    print(kind + " " + json.dumps(payload, ensure_ascii=False, default=str))


@pytest.mark.parametrize("method", ["read_node", "read_node_fields"])
def test_read_committed_revision_after_lock_cleanup_interruption(native_agfs, monkeypatch, method):
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "committed-lock")
    fs.write_node(_preference(ctx, 1), ctx)
    remove = client.rm

    def interrupted_cleanup(path, *args, **kwargs):
        if str(path).endswith(".update-lock"):
            raise AGFSClientError("lock cleanup interrupted")
        return remove(path, *args, **kwargs)

    node = _preference(ctx, 2)
    emit("REQUEST write_node", asdict(node))
    with monkeypatch.context() as patch:
        patch.setattr(client, "rm", interrupted_cleanup)
        fs.write_node(node, ctx)
    emit("RESPONSE write_node", {"committed_version": fs._read_node_version(fs._uri_to_agfs_path(node.uri))})
    emit("REQUEST " + method, {"uri": node.uri, "level": "all"})
    try:
        result = fs.read_node(node.uri, ctx) if method == "read_node" else fs.read_node_fields(node.uri, ctx, "all")
    except Exception as exc:
        emit("RESPONSE " + method, {"error": type(exc).__name__, "message": str(exc)})
        raise
    emit("RESPONSE " + method, asdict(result))
    assert (result.abstract, result.overview, result.content) == (node.abstract, node.overview, node.content)
    assert result.metadata["version"] == 2
    client.session.close()


def test_confirm_committed_write_after_response_arrives_past_deadline(native_agfs, monkeypatch):
    address, _, _ = native_agfs
    client, fs, ctx = _runtime(address, "late-receipt")
    api = MemoryWriteAPI(fs, None, directory_summary_enabled=False)
    entry = memory("atlas_release", content="Atlas releases every Friday.", write_operation_id="atlas-write-001")
    write = fs.write_node

    def committed_then_late_response(node, request):
        write(node, request)
        time.sleep(1.1)
        raise OSError("write response lost after commit")

    monkeypatch.setattr(fs, "write_node", committed_then_late_response)
    emit("REQUEST write_memory", {"candidate": asdict(entry), "timeoutSeconds": 1})
    try:
        with operation_deadline(1):
            result = api.write_memory(entry, ctx)
    except Exception as exc:
        emit("RESPONSE write_memory", {"error": type(exc).__name__, "message": str(exc)})
        raise
    emit("RESPONSE write_memory", result)
    assert result["merged_fields"]["status"] == "operation_committed"
    assert fs.read_node(result["target_uri"], ctx).content == entry.content
    client.session.close()
