"""Canonical session state reuses committed static files under its lease."""

import json

from core.models import RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from session.session_manager import SessionManager
from session.session_state import SessionState
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


def state_stack():
    client = _AtomicMemoryAGFSClient()
    client.api_base = "memory://session-state"
    fs = AGFSContextFS(client)
    ctx = RequestContext("narrow-state", "reader", "agent", "session", "trace")
    manager = SessionManager(get_context_fs=lambda: fs)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    buffer.add("user", "I repaired the receiver.", message_id="message-one")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    return client, fs, ctx, manager, buffer, uri


def published(operations):
    return [entry[2].rsplit("/", 1)[-1] for entry in operations
            if entry[0] == "mv" and "/.publish-" in entry[1]]


def test_twenty_two_saves_publish_forty_seven_state_files():
    client, fs, ctx, manager, buffer, uri = state_stack()
    saves = []
    with fs.session_recovery_lock(uri, ctx):
        for index in range(22):
            start = len(client.operations)
            buffer.extraction_summary = f"saved revision {index}"
            assert manager.save_session_state(ctx.session_id, ctx)
            saves.append(published(client.operations[start:]))
    assert saves[0] == ["content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"]
    assert saves[1:] == [["content.md", ".meta.json"]] * 21
    assert sum(map(len, saves)) == 47
    assert len([entry for entry in client.operations if entry[0] == "write"]) == 49
    state = fs.read_node(uri, ctx)
    assert state.metadata["session_state_layout"]["version"] == 1
    assert len(state.metadata["session_state_layout"]["sha256"]) == 64
    restored = SessionManager(get_context_fs=lambda: fs)
    restored.get_or_create(ctx.session_id, ctx)
    assert restored.load_session_state(ctx.session_id, ctx)
    assert restored.get_or_create(ctx.session_id).extraction_summary == "saved revision 21"
    assert json.loads(state.content)["buffer_messages"][0]["id"] == "message-one"


def test_legacy_layout_initializes_then_reuses_static_files_after_reload():
    client, fs, ctx, manager, buffer, uri = state_stack()
    assert manager.save_session_state(ctx.session_id, ctx)
    node = fs.read_node(uri, ctx)
    legacy_meta = fs._build_node_metadata(node, None)
    fs._write_file(fs._uri_to_agfs_path(uri) + fs.FILE_META, json.dumps(legacy_meta))
    with fs.session_recovery_lock(uri, ctx):
        start = len(client.operations)
        assert manager.save_session_state(ctx.session_id, ctx)
        assert published(client.operations[start:]) == ["content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"]
    next_fs = AGFSContextFS(client)
    next_manager = SessionManager(get_context_fs=lambda: next_fs)
    next_manager.get_or_create(ctx.session_id, ctx)
    assert next_manager.load_session_state(ctx.session_id, ctx)
    with next_fs.session_recovery_lock(uri, ctx):
        start = len(client.operations)
        assert next_manager.save_session_state(ctx.session_id, ctx)
        assert published(client.operations[start:]) == ["content.md", ".meta.json"]
    assert next_fs.read_node(uri, ctx).abstract == node.abstract
    assert next_fs.read_node(uri, ctx).overview == node.overview


def test_updated_static_layout_is_published_then_reused():
    client, fs, ctx, manager, buffer, uri = state_stack()
    with fs.session_recovery_lock(uri, ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
        node = fs.read_node(uri, ctx)
        node.abstract = "Session tasks and extraction progress"
        node.overview = "Current session tasks, commitments and extraction progress."
        start = len(client.operations)
        fs.write_node(node, ctx)
        assert published(client.operations[start:]) == ["content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"]
        start = len(client.operations)
        fs.write_node(node, ctx)
        assert published(client.operations[start:]) == ["content.md", ".meta.json"]
    stored = fs.read_node(uri, ctx)
    assert (stored.abstract, stored.overview) == (node.abstract, node.overview)

