"""Recovery discovery pages canonical session metadata after restart."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from service.extraction_recovery_discovery import (
    DISCOVERY_KEY,
    ExtractionRecoveryDiscovery,
    recovery_discovery_hint,
)
from tests.unit.fs.test_agfs_context_fs import _MemoryAGFSClient


def state(ctx, *, status="queued", legacy=False):
    owner = {key: getattr(ctx, key) for key in ("account_id", "user_id", "agent_id", "session_id")}
    batch = {"owner": owner, "archive_id": "archive-" + ctx.session_id, "slots": {},
             "snapshot": [{"id": "message", "role": "user", "content": "Saved source."}],
             "messages": [{"role": "user", "content": "Saved source."}]}
    if status is not None:
        batch["retry_task"] = {"id": batch["archive_id"], "status": status, "attempts": 1}
    payload = {"runtime_extensions": {"memory_extraction_recovery": {"version": 1, "batch": batch}}}
    metadata = {"session_id": ctx.session_id, "owner_user_id": ctx.user_id, "owner_agent_id": ctx.agent_id}
    if not legacy:
        metadata[DISCOVERY_KEY] = recovery_discovery_hint(payload)
    return ContextNode(uri=f"ctx://{ctx.account_id}/sessions/{ctx.session_id}/state.json",
                       context_type="RESOURCE", category="state", level=0,
                       owner_space="session:" + ctx.session_id,
                       abstract="state", overview="state", content=json.dumps(payload), metadata=metadata)


@pytest.mark.parametrize("status", ["queued", "running", "failed", None])
@pytest.mark.parametrize("legacy", [False, True])
def test_restores_durable_owner_and_task(status, legacy):
    ctx = RequestContext("recovery-account", "reader", "agent", "source", "old-trace")
    fs = AGFSContextFS(_MemoryAGFSClient())
    fs.write_node(state(ctx, status=status, legacy=legacy), ctx)
    assert fs.recovery_account_ids() == [ctx.account_id]
    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id])

    [(recovered, task)] = discover()

    assert recovered == replace(ctx, trace_id="extraction-recovery-discovery")
    assert task["id"] == "archive-source"
    assert task["status"] == (status or "queued")
    if status is None:
        assert task["recovered_source"] is True


def test_agfs_pages_share_one_directory_snapshot_and_cycle_interval(monkeypatch):
    ctx = RequestContext("recovery-account", "reader", "agent", "source", "trace")
    client = _MemoryAGFSClient()
    fs = AGFSContextFS(client)
    for index in range(5):
        current = replace(ctx, session_id=f"source-{index}")
        fs.write_node(state(current), current)
    client.ls = Mock(wraps=client.ls)
    client.read = Mock(wraps=client.read)
    clock = [100]
    monkeypatch.setattr("service.extraction_recovery_discovery.time.monotonic", lambda: clock[0])
    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id], page_size=2)

    pages = [discover(), discover(), discover()]

    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item[0].session_id for page in pages for item in page] == [f"source-{index}" for index in range(5)]
    assert client.ls.call_count == 1
    assert len([call for call in client.read.call_args_list if call.args[0].endswith(".meta.json")]) == 5
    clock[0] = 160
    assert len(discover()) == 2
    assert client.ls.call_count == 2


def test_agfs_discovers_new_content_with_unchanged_metadata(monkeypatch):
    ctx = RequestContext("legacy", "reader", "agent", "source", "trace")
    fs = AGFSContextFS(_MemoryAGFSClient())
    node = state(ctx, legacy=True)
    fs.write_node(node, ctx)
    clock = [100]
    monkeypatch.setattr("service.extraction_recovery_discovery.time.monotonic", lambda: clock[0])
    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id])
    assert discover()[0][1]["id"] == "archive-source"

    path = fs._uri_to_agfs_path(node.uri)
    metadata = fs._read_file(path + fs.FILE_META)
    payload = json.loads(node.content)
    batch = payload["runtime_extensions"]["memory_extraction_recovery"]["batch"]
    batch["archive_id"] = batch["retry_task"]["id"] = "archive-next"
    fs._write_file(path + fs.FILE_CONTENT, json.dumps(payload))
    clock[0] = 160

    assert discover()[0][1]["id"] == "archive-next"
    assert fs._read_file(path + fs.FILE_META) == metadata


def test_metadata_only_legacy_projection_reuses_unchanged_state_version(monkeypatch):
    class MetadataOnlyFS(AGFSContextFS):
        def scan_session_recovery(self, account_id, **kwargs):
            rows, cursor = super().scan_session_recovery(account_id, **kwargs)
            for row in rows:
                row.pop("content", None)
            return rows, cursor

    ctx = RequestContext("legacy", "reader", "agent", "source", "trace")
    fs = MetadataOnlyFS(_MemoryAGFSClient())
    node = state(ctx, legacy=True)
    fs.write_node(node, ctx)
    fs.read_node = Mock(wraps=fs.read_node)
    clock = [100]
    monkeypatch.setattr("service.extraction_recovery_discovery.time.monotonic", lambda: clock[0])
    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: [ctx.account_id])

    assert len(discover()) == 1
    clock[0] = 160
    assert len(discover()) == 1
    assert fs.read_node.call_count == 1
    fs.write_node(state(ctx, legacy=True, status="running"), ctx)
    clock[0] = 220
    assert discover()[0][1]["status"] == "running"
    assert fs.read_node.call_count == 2


def test_later_account_recovers_while_first_account_reconnects():
    class ReconnectingFS(AGFSContextFS):
        def scan_session_recovery(self, account_id, **kwargs):
            if account_id == "reconnecting":
                raise ConnectionError("account storage reconnecting")
            return super().scan_session_recovery(account_id, **kwargs)

    ctx = RequestContext("ready", "reader", "agent", "source", "trace")
    fs = ReconnectingFS(_MemoryAGFSClient())
    fs.write_node(state(ctx), ctx)
    discover = ExtractionRecoveryDiscovery(lambda: fs, lambda: ["reconnecting", "ready"])

    discover()
    [(recovered, task)] = discover()

    assert recovered.account_id == "ready"
    assert task["id"] == "archive-source"
