"""Occurrence replay receipts persist across SQL connections and writers."""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import RequestContext
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import DecisionLLM
from tests.unit.commit.test_occurrence_replay import occurrence


def stores():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    fs = SQLContextFS(dsn)
    outbox = SQLOutboxStore(dsn, fs=fs)
    return fs, outbox


def close_stores(fs, outbox):
    for store in (fs, outbox):
        while store._pool:
            store._pool.pop().close()


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("scope", ["Project A", "Project " + "设备互联与运行状态追踪平台升级验证-" * 5])
def test_sql_replay_survives_reconnect_and_preserves_updated_history(category, scope):
    ctx = RequestContext("replay-" + uuid4().hex, "reader", "agent", "session", "trace")
    first = occurrence(category, scope_ref=scope)
    uri = None
    for stage in range(5):
        fs, outbox = stores()
        try:
            writer = MemoryWriteAPI(fs, DecisionLLM(), outbox, directory_summary_enabled=False)
            draft = first
            if stage == 1:
                draft = replace(first, routing_key="renamed_inspection", confidence=.95)
            elif stage == 2:
                draft = replace(first, target_uri=uri, expected_version=1,
                                content=first.content + " The repair was approved.",
                                abstract="Inspection and approved repair", overview="Repair approved.",
                                provenance_ids=["source:approval"])
            elif stage == 4:
                draft = replace(first, when="2026-10-02 09:00")
            result = writer.write_memory(draft, replace(ctx, session_id=f"session-{stage}"))
            if stage == 0:
                uri = result["target_uri"]
            if stage in (1, 3):
                assert result["action"] == "skip"
                assert result["target_uri"] == uri
            if stage == 4:
                assert result["action"] == "create"
                assert result["target_uri"] != uri
            old = fs.read_node(uri, ctx)
            assert old.metadata["scope_ref"] == scope
            assert old.metadata["version"] == (1 if stage < 2 else 2)
            if stage >= 2:
                assert "repair was approved" in old.content
                assert "source:approval" in old.metadata["provenance_ids"]
            assert len(outbox.list_pending(ctx.account_id)) == (1, 1, 2, 2, 3)[stage]
            directory = uri.rsplit("/", 1)[0]
            assert len(fs.list_children(directory, ctx)) == (2 if stage == 4 else 1)
        finally:
            close_stores(fs, outbox)


def test_sql_concurrent_replay_creates_one_node_and_one_outbox_event():
    ctx = RequestContext("race-" + uuid4().hex, "reader", "agent", "session", "trace")
    pairs = [stores(), stores()]
    barrier = Barrier(2)

    def write(index):
        fs, outbox = pairs[index]
        writer = MemoryWriteAPI(fs, DecisionLLM(), outbox, directory_summary_enabled=False)
        draft = occurrence(routing_key=f"inspection_{index}")
        barrier.wait(timeout=20)
        return writer.write_memory(draft, replace(ctx, session_id=f"session-{index}"))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, range(2)))
        assert sorted(result["action"] for result in results) == ["create", "skip"]
        assert results[0]["target_uri"] == results[1]["target_uri"]
        fs, outbox = pairs[0]
        uri = results[0]["target_uri"]
        assert len(fs.list_children(uri.rsplit("/", 1)[0], ctx)) == 1
        assert fs.read_node(uri, ctx).metadata["version"] == 1
        assert len(outbox.list_pending(ctx.account_id)) == 1
    finally:
        for fs, outbox in pairs:
            close_stores(fs, outbox)
