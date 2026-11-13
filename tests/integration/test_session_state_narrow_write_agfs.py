"""Real AGFS write-count evidence for twenty-two canonical state saves."""

import json
import os
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from core.models import RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from session.session_manager import SessionManager
from session.session_state import SessionState


def test_twenty_two_saves_use_forty_seven_real_agfs_payload_writes():
    address = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not address:
        pytest.skip("set OGMEM_AGFS_TEST_URL to a disposable AGFS server")
    client = AGFSClient(address)
    client.session.trust_env = False
    client.session.proxies.clear()
    fs = AGFSContextFS(client, mount_prefix=os.environ.get("OGMEM_AGFS_TEST_MOUNT", "/local"))
    ctx = RequestContext("narrow-" + uuid4().hex, "reader", "agent", "session", "trace")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    manager = SessionManager(get_context_fs=lambda: fs)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    buffer.add("user", "I repaired the receiver.", message_id="message-one")
    writes, publications = [], []
    write, move = client.write, client.mv

    def record_write(path, content, *args, **kwargs):
        result = write(path, content, *args, **kwargs)
        writes.append((str(path), len(content)))
        return result

    def record_move(source, destination, *args, **kwargs):
        result = move(source, destination, *args, **kwargs)
        if "/.publish-" in str(source):
            publications.append(str(destination).rsplit("/", 1)[-1])
        return result

    client.write, client.mv = record_write, record_move
    per_save = []
    with fs.session_recovery_lock(uri, ctx):
        for index in range(22):
            start = len(publications)
            buffer.extraction_summary = f"revision {index}"
            assert manager.save_session_state(ctx.session_id, ctx)
            per_save.append(publications[start:])
    assert per_save[0] == ["content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"]
    assert per_save[1:] == [["content.md", ".meta.json"]] * 21
    payload_writes = [(path, size) for path, size in writes if "/.publish-" in path]
    assert len(payload_writes) == 47
    loaded = SessionManager(get_context_fs=lambda: fs)
    loaded.get_or_create(ctx.session_id, ctx)
    assert loaded.load_session_state(ctx.session_id, ctx)
    assert loaded.get_or_create(ctx.session_id).extraction_summary == "revision 21"
    print(json.dumps({"saves": 22, "baseline_payload_writes": 110,
                      "payload_writes": len(payload_writes), "total_agfs_writes": len(writes),
                      "payload_bytes": sum(size for _, size in payload_writes),
                      "per_save_payload_writes": [len(items) for items in per_save],
                      "canonical_uri": uri, "loaded_revision": 21}))
