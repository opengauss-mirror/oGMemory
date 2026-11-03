from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

from core.models import RequestContext
from providers.token_tracker import TokenTracker, usage_scope
from providers.unified_config import OgMemConfig
from server.control_plane_store import ControlPlaneStore
from server.internal_tool_usage import InternalToolUsageStore, InternalToolUsageTracker
from server.memory_service import MemoryService


def _summary_subset(summary: dict) -> dict:
    keys = ("llm_tool_rounds", "tool_calls", "input_tokens", "output_tokens", "total_tokens")
    return {key: summary[key] for key in keys}


def test_internal_tool_usage_tracker_filters_tools_by_session():
    tracker = InternalToolUsageTracker(max_rounds_per_session=10)

    tracker.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read", "extract_profile"],
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
        duration_ms=10,
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="extract_profile",
        status="success",
        duration_ms=2,
    )
    tracker.record_round(
        account_id="acct-1",
        session_id="session-b",
        pipeline_name="extraction.lazy",
        round_id="round-b",
        tool_names=["read"],
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-b",
        pipeline_name="extraction.lazy",
        round_id="round-b",
        tool_name="read",
        status="error",
        duration_ms=30,
    )

    stats = tracker.get_stats(session_id="session-a")

    assert stats["source"] == "internal"
    assert _summary_subset(stats["summary"]) == {
        "llm_tool_rounds": 1,
        "tool_calls": 2,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    tools = {tool["tool_name"]: tool for tool in stats["tools"]}
    assert tools["extract_profile"]["call_count"] == 1
    assert tools["extract_profile"]["avg_duration_ms"] == 2.0
    assert tools["extract_profile"]["attributed_tokens"] == 0
    assert tools["extract_profile"]["allocated_tokens"] == 0
    assert tools["read"]["call_count"] == 1
    assert tools["read"]["avg_duration_ms"] == 10.0
    assert tools["read"]["attributed_tokens"] == 0
    assert tools["read"]["allocated_tokens"] == 0


def test_internal_tool_usage_filters_by_user_and_time_range():
    tracker = InternalToolUsageTracker(max_rounds_per_session=10)
    tracker.record_round(
        account_id="acct-1",
        user_id="u-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read"],
        input_tokens=80,
        output_tokens=20,
        ended_at="2026-05-13T10:00:00+00:00",
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
        duration_ms=10,
    )
    tracker.record_round(
        account_id="acct-1",
        user_id="u-2",
        session_id="session-b",
        pipeline_name="extraction.lazy",
        round_id="round-b",
        tool_names=["read"],
        input_tokens=30,
        output_tokens=10,
        ended_at="2026-05-13T11:00:00+00:00",
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-b",
        pipeline_name="extraction.lazy",
        round_id="round-b",
        tool_name="read",
        status="success",
        duration_ms=10,
    )

    stats = tracker.get_stats(
        user_id="u-1",
        start_time="2026-05-13T09:30:00+00:00",
        end_time="2026-05-13T10:30:00+00:00",
        include_rounds=True,
    )

    assert stats["filters"]["user_id"] == "u-1"
    assert _summary_subset(stats["summary"]) == {
        "llm_tool_rounds": 1,
        "tool_calls": 1,
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
    }
    assert stats["tools"][0]["tool_name"] == "read"
    assert stats["tools"][0]["total_tokens"] == 100
    assert stats["tools"][0]["attributed_tokens"] == 100
    assert stats["tools"][0]["allocated_tokens"] == 100
    assert stats["rounds"][0]["round_id"] == "round-a"
    assert stats["sessions"] == [
        {
            "session_id": "session-a",
            "user_id": "u-1",
            "llm_tool_rounds": 1,
            "tool_calls": 1,
        }
    ]


def test_internal_tool_usage_store_round_trips_session_files(tmp_path):
    tracker = InternalToolUsageTracker(max_rounds_per_session=10)
    tracker.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.eager",
        round_id="round-a",
        tool_names=["extract_profile"],
        input_tokens=100,
        output_tokens=20,
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.eager",
        round_id="round-a",
        tool_name="extract_profile",
        status="success",
        duration_ms=3,
    )
    snapshot = tracker.session_snapshot("session-a")

    store = InternalToolUsageStore(ControlPlaneStore(mount_prefix="", local_root=str(tmp_path)))
    store.write_session(snapshot)

    loaded = store.read_session("acct-1", "session-a", include_rounds=True)

    assert loaded["ok"] is True
    assert loaded["source"] == "internal"
    assert _summary_subset(loaded["summary"]) == {
        "llm_tool_rounds": 1,
        "tool_calls": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert loaded["tools"][0]["tool_name"] == "extract_profile"
    assert loaded["tools"][0]["attributed_tokens"] == 120
    assert loaded["tools"][0]["allocated_tokens"] == 120
    assert loaded["rounds"][0]["round_id"] == "round-a"


def test_internal_tool_usage_session_snapshot_is_json_serializable():
    tracker = InternalToolUsageTracker(max_rounds_per_session=10)
    tracker.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.eager",
        round_id="round-a",
        tool_names=["extract_profile"],
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.eager",
        round_id="round-a",
        tool_name="extract_profile",
        status="success",
    )

    snapshot = tracker.session_snapshot("session-a")

    json.dumps(snapshot)


def test_token_stats_reports_internal_tool_round_token_summary():
    service = MemoryService(config=OgMemConfig())
    service._internal_tool_usage.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read", "extract_profile"],
        input_tokens=100,
        output_tokens=20,
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="extract_profile",
        status="success",
    )

    stats = service.get_cumulative_token_usage()

    assert stats["internal_tool_rounds"] == {
        "rounds": 1,
        "tool_calls": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 120,
    }


def test_model_usage_metrics_combines_llm_and_embedding_trackers():
    service = MemoryService(config=OgMemConfig())
    llm_tracker = TokenTracker(model="qwen-plus")
    embedding_tracker = TokenTracker(model="text-embedding-v3")
    with usage_scope("acct-a", "user:alice"):
        llm_tracker.record_llm(120, 30)
        embedding_tracker.record_embed(50)
    with usage_scope("acct-b", "user:bob"):
        llm_tracker.record_llm(999, 999)
    service._llm = SimpleNamespace(token_tracker=llm_tracker)
    service._embedder = SimpleNamespace(token_tracker=embedding_tracker)

    metrics = service.get_model_usage_metrics(days=1, account_id="acct-a")

    assert metrics["models"] == {
        "llm": "qwen-plus",
        "embedding": "text-embedding-v3",
    }
    assert metrics["today"]["total"] == 200
    assert metrics["cumulative"]["llm"]["calls"] == 1
    assert metrics["cumulative"]["embedding"]["calls"] == 1
    assert metrics["scope"]["account_id"] == "acct-a"


def test_internal_tool_usage_reports_attributed_and_allocated_tokens():
    tracker = InternalToolUsageTracker(max_rounds_per_session=10)
    tracker.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read", "extract_profile"],
        input_tokens=90,
        output_tokens=30,
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
        duration_ms=10,
    )
    tracker.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="extract_profile",
        status="success",
        duration_ms=2,
    )

    stats = tracker.get_stats(session_id="session-a")

    assert stats["summary"]["input_tokens"] == 90
    assert stats["summary"]["output_tokens"] == 30
    assert stats["summary"]["total_tokens"] == 120
    tools = {tool["tool_name"]: tool for tool in stats["tools"]}
    assert tools["read"]["attributed_tokens"] == 120
    assert tools["extract_profile"]["attributed_tokens"] == 120
    assert tools["read"]["allocated_tokens"] == 60
    assert tools["extract_profile"]["allocated_tokens"] == 60
    assert sum(tool["allocated_tokens"] for tool in stats["tools"]) == 120


def test_session_tool_usage_flush_writes_session_files_and_read_by_session(tmp_path):
    service = MemoryService(config=OgMemConfig())
    service._internal_tool_usage_store = InternalToolUsageStore(
        ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    )
    service._internal_tool_usage.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read"],
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
        duration_ms=4,
    )
    ctx = RequestContext(
        account_id="acct-1",
        user_id="u-1",
        agent_id="main",
        session_id="session-a",
        trace_id="trace-1",
    )

    service._flush_session_tool_usage_async("session-a", ctx)
    tool_calls_path = (
        tmp_path
        / "accounts"
        / "acct-1"
        / "sessions"
        / "session-a"
        / "tool_usage"
        / "tool_calls.json"
    )
    deadline = time.time() + 2
    while time.time() < deadline and not tool_calls_path.exists():
        time.sleep(0.01)

    assert tool_calls_path.exists()
    service._internal_tool_usage.reset()

    stats = service.get_tool_usage_stats(
        {
            "accountId": "acct-1",
            "session_id": "session-a",
            "include_rounds": "true",
        }
    )

    assert _summary_subset(stats["summary"]) == {
        "llm_tool_rounds": 1,
        "tool_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert stats["tools"][0]["tool_name"] == "read"
    assert stats["tools"][0]["attributed_tokens"] == 0
    assert stats["tools"][0]["allocated_tokens"] == 0
    assert stats["rounds"][0]["round_id"] == "round-a"


def test_session_tool_usage_flush_clears_in_memory_session_after_success(tmp_path):
    service = MemoryService(config=OgMemConfig())
    service._internal_tool_usage_store = InternalToolUsageStore(
        ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    )
    service._internal_tool_usage.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read"],
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
    )
    ctx = RequestContext(
        account_id="acct-1",
        user_id="u-1",
        agent_id="main",
        session_id="session-a",
        trace_id="trace-1",
    )

    service._flush_session_tool_usage_async("session-a", ctx)
    tool_calls_path = (
        tmp_path
        / "accounts"
        / "acct-1"
        / "sessions"
        / "session-a"
        / "tool_usage"
        / "tool_calls.json"
    )
    deadline = time.time() + 2
    while time.time() < deadline and not tool_calls_path.exists():
        time.sleep(0.01)

    assert tool_calls_path.exists()
    live_stats = service._internal_tool_usage.get_stats(session_id="session-a")
    assert live_stats["summary"]["llm_tool_rounds"] == 0
    assert live_stats["summary"]["tool_calls"] == 0


def test_tool_usage_flush_failure_is_available_from_pending_snapshot():
    class FailingStore:
        def write_session(self, snapshot):
            raise RuntimeError("boom")

        def read_session(self, account_id, session_id, *, include_rounds=False):
            raise FileNotFoundError(session_id)

    service = MemoryService(config=OgMemConfig())
    service._internal_tool_usage_store = FailingStore()
    service._internal_tool_usage.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read"],
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
    )
    ctx = RequestContext(
        account_id="acct-1",
        user_id="u-1",
        agent_id="main",
        session_id="session-a",
        trace_id="trace-1",
    )

    service._flush_session_tool_usage_async("session-a", ctx)
    deadline = time.time() + 2
    while time.time() < deadline:
        if service._pending_tool_usage_snapshots.get("session-a"):
            break
        time.sleep(0.01)

    service._internal_tool_usage.reset()
    stats = service.get_tool_usage_stats(
        {
            "accountId": "acct-1",
            "session_id": "session-a",
            "include_rounds": "true",
        }
    )

    assert stats["summary"]["llm_tool_rounds"] == 1
    assert stats["summary"]["tool_calls"] == 1
    assert stats["rounds"][0]["round_id"] == "round-a"


def test_dispose_flushes_tool_usage_for_completed_session(tmp_path):
    service = MemoryService(config=OgMemConfig())
    service._get_context_fs = Mock(return_value=None)
    service._internal_tool_usage_store = InternalToolUsageStore(
        ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    )
    mgr = service.get_session_manager()
    buf = mgr.get_or_create("session-a")
    buf.add("user", "already extracted")
    buf.extraction_watermark = len(buf.messages)
    service._internal_tool_usage.record_round(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_names=["read"],
    )
    service._internal_tool_usage.record_tool_call(
        account_id="acct-1",
        session_id="session-a",
        pipeline_name="extraction.lazy",
        round_id="round-a",
        tool_name="read",
        status="success",
    )
    ctx = RequestContext(
        account_id="acct-1",
        user_id="u-1",
        agent_id="main",
        session_id="session-a",
        trace_id="trace-1",
    )

    result = service.dispose({"sessionId": "session-a", "_ctx": ctx})
    tool_calls_path = (
        tmp_path
        / "accounts"
        / "acct-1"
        / "sessions"
        / "session-a"
        / "tool_usage"
        / "tool_calls.json"
    )
    deadline = time.time() + 2
    while time.time() < deadline and not tool_calls_path.exists():
        time.sleep(0.01)

    assert result["reason"] == "no_pending_messages"
    assert tool_calls_path.exists()
