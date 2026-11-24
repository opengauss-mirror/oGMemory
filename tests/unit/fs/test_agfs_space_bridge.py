"""Positive publication and content-read scenarios for the AGFS/SQL bridge."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pyagfs.exceptions import AGFSClientError

from core.errors import OperationDeadlineExceededError
from core.models import ContextNode, RequestContext
from core.operation_deadline import operation_deadline
from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from fs.sql_adapter.sql_context_fs import SQLContextFS


class MemoryAGFS:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        self.active_reads = 0
        self.max_reads = 0
        self.read_delay = 0

    def mkdir(self, path):
        pass

    def write(self, path, content):
        with self.lock:
            self.files[path] = content

    def read(self, path):
        with self.lock:
            if path not in self.files:
                raise AGFSClientError("file not found")
            self.active_reads += 1
            self.max_reads = max(self.max_reads, self.active_reads)
            content = self.files[path]
        try:
            if self.read_delay:
                time.sleep(self.read_delay)
            return content
        finally:
            with self.lock:
                self.active_reads -= 1

    def mv(self, source, target):
        with self.lock:
            self.files[target] = self.files.pop(source)

    def rm(self, path, recursive=False):
        with self.lock:
            for name in list(self.files):
                if name == path or (recursive and name.startswith(path + "/")):
                    del self.files[name]


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(SQLContextFS, "_ensure_table", lambda self: None)
    monkeypatch.setattr("fs.agfs_adapter.space_bridge.SharedConnectionPool", lambda *args, **kwargs: MagicMock())
    value = AGFSSpaceBridge(MagicMock(), MemoryAGFS(), "/local")
    yield value
    value.close()


def source_node(content="排查连接池等待，再定位慢事务。"):
    return ContextNode(uri="ctx://acme/agents/ops/memories/case/pool", context_type="MEMORY",
                       category="case", level=2, owner_space="agent:ops", content=content,
                       abstract="连接池排查", overview="## 过程\n定位慢事务", metadata={})


def test_source_publication_stages_agfs_before_sql_and_reads_body(bridge, monkeypatch):
    ctx = RequestContext("acme", "alice", "ops", "session", "trace")
    saved = []
    cursor = MagicMock()

    def commit(self, cur, node, account_id, expected_version, create_only,
               relations_json, extra_meta, new_version, now):
        pointer = extra_meta["agfs_content"]
        payload = bridge._agfs._client.read(pointer["path"])
        assert hashlib.sha256(payload).hexdigest() == pointer["sha256"]
        assert json.loads(payload)["content"] == node.content
        assert cur is cursor
        assert account_id == ctx.account_id
        saved.append(replace(node, metadata={**node.metadata, **extra_meta}))

    monkeypatch.setattr(SQLContextFS, "_execute_write", commit)
    bridge._execute_write(cursor, source_node(), ctx.account_id, None, False, "[]", {}, 1, datetime.now(UTC))
    monkeypatch.setattr(SQLContextFS, "read_node", lambda *args: replace(saved[0], content="SQL projection"))
    read = bridge.read_node(saved[0].uri, ctx)
    assert read.content == source_node().content
    assert read.abstract == source_node().abstract


def test_immutable_revisions_reuse_content_and_preserve_old_reads(bridge):
    node = source_node()
    old = bridge.store_shared(node.uri, "版本一", "摘要", "概览")
    current = bridge.store_shared(node.uri, "版本二", "摘要", "概览")
    repeated = bridge.store_shared(node.uri, "版本二", "摘要", "概览")
    assert old["agfs_content"]["sha256"] != current["agfs_content"]["sha256"]
    assert repeated == current
    rows = [{"uri": node.uri, "metadata": old}, {"uri": node.uri, "metadata": current}]
    assert [r["content"] for r in bridge.load_content(rows)] == ["版本一", "版本二"]


def test_field_and_summary_reads_hydrate_agfs(bridge, monkeypatch):
    node = source_node()
    node.metadata.update(bridge.store_shared(node.uri, node.content, node.abstract, node.overview))
    monkeypatch.setattr(SQLContextFS, "read_node_fields", lambda *args: replace(node, content="projection"))
    monkeypatch.setattr(SQLContextFS, "list_node_summaries_page", lambda *args, **kwargs: [node])
    ctx = RequestContext("acme", "alice", "ops", "session", "trace")
    assert bridge.read_node_fields(node.uri, ctx, "content").content == node.content
    assert bridge.read_node_fields(node.uri, ctx, "abstract").abstract == node.abstract
    assert bridge.read_node_fields(node.uri, ctx, "overview").overview == node.overview
    assert bridge.list_node_summaries(node.uri.rsplit("/", 1)[0], ctx)[0].abstract == node.abstract


def test_batch_reads_hydrate_agfs(bridge, monkeypatch):
    nodes = [replace(source_node(f"正文 {index}"), uri=source_node().uri + str(index))
             for index in range(10)]
    for node in nodes:
        node.metadata.update(bridge.store_shared(node.uri, node.content, node.abstract, node.overview))
    monkeypatch.setattr(SQLContextFS, "read_nodes", lambda *args: {
        node.uri: replace(node, content="projection") for node in nodes})
    ctx = RequestContext("acme", "alice", "ops", "session", "trace")
    result = bridge.read_nodes([node.uri for node in nodes], ctx)
    assert [result[node.uri].content for node in nodes] == [node.content for node in nodes]


def test_queued_hydration_inherits_deadline(bridge, monkeypatch):
    import core.operation_deadline as deadline

    uri = source_node().uri
    metadata = bridge.store_shared(uri, "原始正文")
    calls = []
    now = 0
    read = bridge._agfs._client.read

    def cross_deadline(path):
        nonlocal now
        calls.append(path)
        result = read(path)
        now = 1
        return result

    monkeypatch.setattr(deadline, "monotonic", lambda: now)
    monkeypatch.setattr(bridge._agfs._client, "read", cross_deadline)
    with ThreadPoolExecutor(max_workers=1) as executor, monkeypatch.context() as patch:
        patch.setattr(bridge, "_content_reads", executor)
        with operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
            bridge.load_content([{"uri": uri, "metadata": metadata} for _ in range(3)])
    assert calls == [metadata["agfs_content"]["path"]]


def test_shared_skill_hydration_and_purge(bridge):
    uri = "ctx://acme/spaces/ops/skills/restart"
    package = {"package_skill_md": "# Restart", "package_files": {"check.txt": "validated"}}
    pointer = bridge.store_shared(uri, "# Restart", "重启步骤", metadata=package)
    row = bridge.load_content([{"uri": uri, "metadata": pointer}])[0]
    assert row["metadata"]["package_files"] == package["package_files"]
    bridge.cleanup_shared(uri)
    assert len(bridge._agfs._client.files) == 0
    bridge.cleanup_shared(uri)


def test_bounded_batch_hydration_and_catalog_uri_move(bridge):
    uri = source_node().uri
    metadata = bridge.store_shared(uri, "原始正文")
    bridge._agfs._client.read_delay = 0.005
    rows = [{"uri": uri, "metadata": metadata} for _ in range(50)]
    bridge.load_content(rows)
    assert all(row["content"] == "原始正文" for row in rows)
    assert 1 < bridge._agfs._client.max_reads <= 8
    moved = bridge.load_content([{"uri": uri + "-renamed", "metadata": metadata}])[0]
    assert moved["content"] == "原始正文"


def test_close_drains_active_content_reads(bridge, monkeypatch):
    uri = source_node().uri
    metadata = bridge.store_shared(uri, "读取完成")
    reading = threading.Event()
    closing = threading.Event()
    release = threading.Event()
    read = bridge._agfs._client.read

    def read_content(path):
        reading.set()
        release.wait(2)
        return read(path)

    def close_content():
        closing.set()
        return bridge.close(timeout=2)

    monkeypatch.setattr(bridge._agfs._client, "read", read_content)
    rows = [{"uri": uri, "metadata": metadata} for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as requests:
        hydrated = requests.submit(bridge.load_content, rows)
        assert reading.wait(1)
        drained = requests.submit(close_content)
        assert closing.wait(1)
        release.set()
        assert drained.result(timeout=3) is True
        assert [row["content"] for row in hydrated.result(timeout=3)] == ["读取完成"] * 2
