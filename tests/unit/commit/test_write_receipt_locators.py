"""Versioned accepted operations locate their committed source after restart."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Barrier
from unittest.mock import Mock

import pytest

from commit.write_receipts import (
    LOCATED_OPERATION_PREFIX, RECEIPTS_KEY, bind_operation_target, operation_digest, receipt_plan,
)
from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.sql_context_fs import parse_uri as parse_sql_uri
from index.index_record_builder import build_index_records
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS, memory
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient
from tests.unit.fs.test_sql_context_fs import mock_sql_fs
from tests.unit.server.test_extraction_batch_recovery import NoModel


@pytest.fixture
def ctx():
    return RequestContext(account_id="receipt-locator", user_id="reader", agent_id="agent",
                          session_id="source-session", trace_id="trace")


class CountingMemoryFS(MemoryFS):
    def __init__(self):
        super().__init__()
        self.listed = []
        self.reads = []
        self.writes = []

    def list_children(self, uri, ctx):
        self.listed.append(uri)
        return super().list_children(uri, ctx)

    def read_node(self, uri, ctx):
        self.reads.append(uri)
        return super().read_node(uri, ctx)

    def write_node(self, node, ctx):
        super().write_node(node, ctx)
        self.writes.append(node.uri)


@pytest.mark.parametrize("category,owner", [
    ("session_archive", "user"), ("session_summary", "user"),
    ("entity", "user"), ("preference", "user"), ("event", "user"), ("case", "agent"),
    ("skill", "agent"), ("tool", "agent"),
])
def test_versioned_operation_replays_with_direct_source_proof(ctx, category, owner):
    fs = CountingMemoryFS()
    accepted = replace(memory("receiver", category=category), owner_scope=owner,
                       write_operation_id=LOCATED_OPERATION_PREFIX + "accepted")
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    result = writer.write_memory(accepted, ctx)
    source = fs.read_node(result["target_uri"], ctx)
    locator = next(node for node in fs.nodes.values() if node.category == "write_operation")
    assert fs.writes == [locator.uri, source.uri]
    assert json.loads(locator.content)["target_uri"] == source.uri
    assert source.metadata[RECEIPTS_KEY][accepted.write_operation_id] == operation_digest(accepted, ctx)
    assert build_index_records(locator) == []
    assert parse_sql_uri(locator.uri)["owner_id"] == (ctx.agent_id if owner == "agent" else ctx.user_id)

    restarted_fs = CountingMemoryFS()
    restarted_fs.nodes = deepcopy(fs.nodes)
    outbox = Mock()
    restarted = MemoryWriteAPI(restarted_fs, NoModel(), outbox, directory_summary_enabled=False)
    restarted._writer._require_index_event = True
    replay = restarted.write_memory(accepted, replace(ctx, trace_id="replay"))
    assert replay["merged_fields"]["status"] == "operation_committed"
    assert restarted_fs.listed == []
    assert restarted_fs.reads == [locator.uri, source.uri, source.uri]
    assert restarted_fs.nodes[source.uri] == source
    outbox.register_write.assert_called_once_with(source, replace(ctx, trace_id="replay"))


def test_bound_operation_replays_after_later_valid_update(ctx):
    fs = CountingMemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    original = memory("receiver", content="Original fact.")
    uri = writer.write_memory(original, ctx)["target_uri"]
    accepted = replace(original, target_uri=uri, expected_version=1, content="Original and new fact.",
                       abstract="Original and new fact.", overview="Original and new fact.",
                       write_operation_id=LOCATED_OPERATION_PREFIX + "update")
    writer.write_memory(accepted, ctx)
    later = replace(accepted, expected_version=2, content="Original, new and later fact.",
                    abstract="Original, new and later fact.", overview="Original, new and later fact.",
                    write_operation_id=LOCATED_OPERATION_PREFIX + "later")
    writer.write_memory(later, ctx)
    stored = fs.read_node(uri, ctx)
    fs.listed.clear()

    replay = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(accepted, ctx)
    assert replay["merged_fields"]["status"] == "operation_committed"
    assert fs.read_node(uri, ctx) == stored
    assert stored.metadata["version"] == 3
    assert fs.listed == []


def test_single_file_operation_keeps_direct_source_receipt(ctx):
    from tests.unit.commit.test_profile_merge import profile

    fs = CountingMemoryFS()
    accepted = replace(profile("identity", "Original fact."),
                       write_operation_id=LOCATED_OPERATION_PREFIX + "profile")
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    created = writer.write_memory(accepted, ctx)
    replay = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(accepted, ctx)
    assert replay["target_uri"] == created["target_uri"]
    assert replay["merged_fields"]["status"] == "operation_committed"
    assert fs.listed == []
    assert len(fs.nodes) == 1


def test_concurrent_writers_share_one_versioned_source(ctx):
    fs = CountingMemoryFS()
    accepted = replace(memory("chunk", category="session_archive"),
                       write_operation_id=LOCATED_OPERATION_PREFIX + "concurrent")
    writers = [MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False) for _ in range(2)]
    ready = Barrier(2)

    def write(writer):
        ready.wait()
        return writer.write_memory(accepted, ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, writers))
    assert sorted(result["action"] for result in results) == ["create", "skip"]
    assert len({result["target_uri"] for result in results}) == 1
    assert fs.read_node(results[0]["target_uri"], ctx).metadata["version"] == 1
    assert len(fs.nodes) == 2
    assert fs.listed == []


def test_legacy_operation_retains_single_locked_discovery_and_replay(ctx):
    fs = CountingMemoryFS()
    accepted = replace(memory("chunk", category="session_archive"), write_operation_id="legacy-operation")
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    created = writer.write_memory(accepted, ctx)
    directory = writer._writer._identity_resolver.uris.get_directory_uri(accepted.category, ctx)
    assert fs.listed == [directory]
    fs.listed.clear()

    replay = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(accepted, ctx)
    assert replay["target_uri"] == created["target_uri"]
    assert replay["merged_fields"]["status"] == "operation_committed"
    assert fs.listed == [directory]
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("dialect,statement", [
    ("postgresql", "INSERT INTO context_nodes"), ("opengauss", "MERGE INTO context_nodes"),
])
def test_sql_locator_uses_atomic_initial_create(ctx, mock_sql_fs, monkeypatch, dialect, statement):
    fs, connection, cursor = mock_sql_fs
    fs._dialect = resolve_sql_dialect(dialect)
    create_node = Mock(wraps=fs._dialect.write_context_node)
    monkeypatch.setattr(fs._dialect, "write_context_node", create_node)
    cursor.fetchone.return_value = None
    cursor.rowcount = 1
    uris = URIResolver(SchemaRegistry())
    accepted = replace(memory("receiver"), write_operation_id=LOCATED_OPERATION_PREFIX + "sql-create")
    target = uris.resolve(accepted.category, {"routing_key": accepted.routing_key}, ctx)

    bind_operation_target(fs, uris, accepted, ctx, target)

    inserted = next(call for call in cursor.execute.call_args_list if statement in call.args[0])
    values = inserted.args[1]
    assert values[1:5] == (ctx.account_id, ctx.user_space_name(), "write_operation", "RESOURCE")
    assert values[11] == 1
    assert values[10]["indexing_disabled"] is True
    assert json.loads(values[6]) == {
        "operation_id": accepted.write_operation_id, "category": accepted.category,
        "digest": operation_digest(accepted, ctx), "target_uri": target,
    }
    assert create_node.call_args.kwargs["create_only"] is True
    connection.commit.assert_called_once()


def test_agfs_525_archive_directory_uses_constant_lookup_and_persistent_replay(ctx):
    class CountingClient(_AtomicMemoryAGFSClient):
        def __init__(self):
            super().__init__()
            self.listed = []

        def ls(self, path):
            self.listed.append(path.rstrip("/"))
            return super().ls(path)

    client = CountingClient()
    fs = AGFSContextFS(client)
    directory = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/session_archives"
    for index in range(525):
        fs.write_node(ContextNode(
            uri=f"{directory}/history_{index}", context_type="MEMORY", category="session_archive",
            level=0, owner_space=ctx.user_space_name(), abstract="History.", overview="History.",
            content="History.",
        ), ctx)
    client.operations.clear()
    client.listed.clear()
    accepted = replace(memory("current_chunk", category="session_archive"),
                       write_operation_id=LOCATED_OPERATION_PREFIX + "current-chunk")
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    created = writer.write_memory(accepted, ctx)
    category_path = fs._uri_to_agfs_path(directory).rstrip("/")
    assert client.listed.count(category_path) == 0
    locator_writes = [op for op in client.operations
                      if op[0] == "write" and "/.write_operations/" in op[1]]
    assert len(locator_writes) == 5
    source_path = fs._uri_to_agfs_path(created["target_uri"])
    locator_publish = next(index for index, op in enumerate(client.operations)
                           if op[0] == "mv" and "/.write_operations/" in op[2])
    source_publish = next(index for index, op in enumerate(client.operations)
                          if op[0] == "mv" and op[2].rstrip("/") == source_path.rstrip("/"))
    assert locator_publish < source_publish

    restarted = AGFSContextFS(client)
    replay = receipt_plan(restarted, writer._writer._identity_resolver.uris, accepted, ctx)
    assert replay.target_uri == created["target_uri"]
    assert replay.merged_fields["status"] == "operation_committed"
    assert client.listed.count(category_path) == 0
    assert fs.read_node(replay.target_uri, ctx).metadata["version"] == 1
