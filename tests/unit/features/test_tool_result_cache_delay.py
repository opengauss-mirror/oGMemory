from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import providers.unified_config as unified_config
from core.models import RequestContext
from features.tool_result_externalization.externalizer import (
    ToolResultExternalizer,
    parse_stub,
)
from features.tool_result_externalization.normalizer import ToolResultNormalizer
from features.tool_result_externalization.policy import ToolResultPolicy
from features.tool_result_externalization.store_base import (
    ArtifactNotVisibleError,
    ToolResultArtifactStore,
)
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder
from features.tool_result_optimization.cache_delay import ToolResultCacheDelay
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _ctx(session_id: str = "session") -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id=session_id,
        trace_id="trace",
    )


def _tool_message(content: str, message_id: str, turn_id: str = "turn") -> dict:
    return {
        "role": "toolResult",
        "tool_call_id": f"call-{message_id}",
        "content": content,
        "_ogmem_source": {
            "source_message_id": message_id,
            "turn_id": turn_id,
        },
    }


def _user_message_tool_turn(content: str, message_id: str) -> list[dict]:
    call_id = f"call-{message_id}"
    return [
        {"role": "user", "content": "inspect the tool result"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": call_id,
                "name": "read",
                "arguments": {"path": "/tmp/fixture"},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "read",
            "content": content,
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]


def _register_user_message_turn(
    service: MemoryService,
    messages: list[dict],
    *,
    session_id: str = "session",
) -> dict:
    return service.after_turn({
        "sessionId": session_id,
        "messages": messages,
        "prePromptMessageCount": 0,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(session_id),
    })


def _isolate_service(service: MemoryService) -> MemoryService:
    service._get_archive_store = lambda: None
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


class _MemoryStore(ToolResultArtifactStore):
    def __init__(self) -> None:
        self.rows = {}

    def write(self, metadata, output_text, ctx):
        existing = self.rows.get(metadata.artifact_id)
        if existing is not None:
            if existing == (metadata, output_text):
                return True
            raise FileExistsError(metadata.artifact_id)
        self.rows[metadata.artifact_id] = (metadata, output_text)
        return False

    def read_all(self, artifact_id, session_id, ctx):
        try:
            return self.rows[artifact_id]
        except KeyError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc

    def list(self, session_id, ctx, limit=50, cursor=None):
        return {
            "artifacts": [metadata.to_dict() for metadata, _ in self.rows.values()],
            "truncated": False,
            "next_cursor": None,
        }

    def delete_session_artifacts(self, session_id, ctx):
        count = len(self.rows)
        self.rows.clear()
        return count


def _cache_delay(
    store,
    clock,
    *,
    delayed=10,
    immediate=100,
    turn_inline=10_000,
    ttl=30,
    summary_artifact=200,
    summary_turn=12_000,
):
    persistence_externalizer = ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(delayed),
        ToolResultSynopsisBuilder(200),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=summary_artifact,
        summary_per_turn_chars=summary_turn,
    )
    replay_externalizer = ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(immediate),
        ToolResultSynopsisBuilder(200),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=summary_artifact,
        summary_per_turn_chars=summary_turn,
    )
    return ToolResultCacheDelay(
        persistence_externalizer=persistence_externalizer,
        replay_externalizer=replay_externalizer,
        store=store,
        delayed_result_chars=delayed,
        immediate_result_chars=immediate,
        idle_ttl_seconds=ttl,
        cache_hit_price_ratio=0.1,
        clock=clock,
    )


def test_optimization_config_defaults_off_and_cache_delay_requires_externalization():
    config = OgMemConfig()
    assert config.tool_result_single_chars == 20_000
    assert config.tool_result_cache_delay_chars == 2_000
    assert config.tool_result_stub_summary_turn_chars == 12_000
    assert config.tool_result_semantic_turn_mode == "user_message"
    assert config.tool_result_cache_delay_enabled is False

    try:
        OgMemConfig(tool_result_cache_delay_enabled=True)
    except ValueError as exc:
        assert "requires tool-result externalization" in str(exc)
    else:
        raise AssertionError("cache delay must not implicitly enable externalization")

    try:
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=1,
        )
    except ValueError as exc:
        assert "must be at least 2" in str(exc)
    else:
        raise AssertionError("one-character delayed threshold is not supported")


def test_cache_delay_config_loads_yaml_and_environment(monkeypatch):
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {
        "tool_result_externalization": {
            "enabled": True,
            "single_result_chars": 20_000,
        },
        "tool_result_optimization": {
            "cache_delay": {
                "enabled": True,
                "delayed_result_chars": 1_000,
                "idle_ttl_seconds": 45,
                "cache_hit_price_ratio": 0.02,
            },
        },
    })
    yaml_config = OgMemConfig.load("unused.yaml")
    assert yaml_config.tool_result_cache_delay_enabled is True
    assert yaml_config.tool_result_cache_delay_chars == 1_000
    assert yaml_config.tool_result_cache_delay_idle_ttl_seconds == 45
    assert yaml_config.tool_result_cache_hit_price_ratio == 0.02

    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED", "1")
    monkeypatch.setenv("OGMEM_TOOL_RESULT_SINGLE_CHARS", "50000")
    monkeypatch.setenv("OGMEM_TOOL_RESULT_CACHE_DELAY_ENABLED", "1")
    monkeypatch.setenv("OGMEM_TOOL_RESULT_CACHE_DELAY_CHARS", "1000")
    monkeypatch.setenv(
        "OGMEM_TOOL_RESULT_CACHE_DELAY_IDLE_TTL_SECONDS", "60",
    )
    monkeypatch.setenv("OGMEM_TOOL_RESULT_CACHE_HIT_PRICE_RATIO", "0.02")
    env_config = OgMemConfig.load("unused.yaml")
    assert env_config.tool_result_externalization_enabled is True
    assert env_config.tool_result_single_chars == 50_000
    assert env_config.tool_result_cache_delay_enabled is True
    assert env_config.tool_result_cache_delay_chars == 1_000
    assert env_config.tool_result_cache_delay_idle_ttl_seconds == 60
    assert env_config.tool_result_cache_hit_price_ratio == 0.02


def test_cache_delay_state_transitions_once_after_ttl_and_never_moves_backwards():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = [start]
    controller = _cache_delay(_MemoryStore(), lambda: now[0])

    first = controller.advance(None)
    assert not first.transitioned
    assert first.active_cutoff == ""

    now[0] = start + timedelta(seconds=29)
    hot = controller.advance(first.state.to_dict())
    assert not hot.transitioned
    assert hot.active_cutoff == ""

    now[0] = start + timedelta(seconds=60)
    cold = controller.advance(hot.state.to_dict())
    assert cold.transitioned
    assert cold.active_cutoff == hot.state.last_model_request_at
    assert cold.cache_hit_price_ratio == 0.1

    now[0] = start + timedelta(seconds=61)
    resumed = controller.advance(cold.state.to_dict())
    assert not resumed.transitioned
    assert resumed.active_cutoff == cold.active_cutoff

    now[0] = start - timedelta(seconds=1)
    backwards = controller.advance(resumed.state.to_dict())
    assert backwards.state.last_model_request_at == resumed.state.last_model_request_at
    assert backwards.active_cutoff == cold.active_cutoff
def test_cache_delay_keeps_hot_caller_raw_but_persists_stub_and_cold_rewrites():
    store = _MemoryStore()
    controller = _cache_delay(store, lambda: datetime.now(UTC), immediate=3_000)
    message = _tool_message("x" * 2_000, "message-1")

    governed = controller.govern_persistent_messages(
        [message], ctx=_ctx(), session_id="session", complete_turn=False,
    )
    assert "[ogmemory-tool-result v=2]" in governed.messages[0]["content"]
    assert len(store.rows) == 1

    hot = controller.rewrite_cold_view(
        [message], ctx=_ctx(), session_id="session", cutoff="",
        hot_seen_artifacts={},
    )
    assert hot.messages[0]["content"] == "x" * 2_000
    assert hot.replaced_count == 0

    metadata, _ = next(iter(store.rows.values()))
    cutoff = (_parse(metadata.created_at) + timedelta(seconds=1)).isoformat()
    cold = controller.rewrite_cold_view(
        [message], ctx=_ctx(), session_id="session", cutoff=cutoff,
        hot_seen_artifacts={metadata.artifact_id: metadata.created_at},
    )
    assert cold.replaced_count == 1
    assert "[ogmemory-tool-result v=2]" in cold.messages[0]["content"]
    assert message["content"] == "x" * 2_000
def test_cache_delay_cutoff_does_not_rewrite_new_hot_epoch_or_fallback_identity():
    store = _MemoryStore()
    controller = _cache_delay(store, lambda: datetime.now(UTC), immediate=3_000)
    old = _tool_message("a" * 2_000, "old-message")
    new = _tool_message("b" * 2_000, "new-message", turn_id="turn-2")
    controller.govern_persistent_messages(
        [old, new], ctx=_ctx(), session_id="session", complete_turn=False,
    )
    rows = list(store.rows.items())
    boundary = datetime(2026, 1, 1, tzinfo=UTC)
    old_id, (old_meta, old_output) = rows[0]
    new_id, (new_meta, new_output) = rows[1]
    store.rows[old_id] = (replace(old_meta, created_at=(boundary - timedelta(seconds=1)).isoformat()), old_output)
    store.rows[new_id] = (replace(new_meta, created_at=(boundary + timedelta(seconds=1)).isoformat()), new_output)

    fallback = {"role": "tool", "content": "c" * 20}
    result = controller.rewrite_cold_view(
        [old, new, fallback],
        ctx=_ctx(),
        session_id="session",
        cutoff=boundary.isoformat(),
        hot_seen_artifacts={
            old_id: (boundary - timedelta(seconds=2)).isoformat(),
            new_id: (boundary + timedelta(seconds=2)).isoformat(),
        },
    )

    assert result.replaced_count == 1
    assert "[ogmemory-tool-result v=2]" in result.messages[0]["content"]
    assert result.messages[1]["content"] == "b" * 2_000
    assert result.messages[2]["content"] == "c" * 20


def test_cache_delay_store_read_or_metadata_mismatch_fails_open():
    store = _MemoryStore()
    controller = _cache_delay(store, lambda: datetime.now(UTC))
    message = _tool_message("z" * 20, "message-1")
    controller.govern_persistent_messages(
        [message], ctx=_ctx(), session_id="session", complete_turn=False,
    )
    artifact_id, (metadata, output) = next(iter(store.rows.items()))
    cutoff = (_parse(metadata.created_at) + timedelta(seconds=1)).isoformat()

    store.rows[artifact_id] = (replace(metadata, sha256="0" * 64), output)
    mismatch = controller.rewrite_cold_view(
        [message], ctx=_ctx(), session_id="session", cutoff=cutoff,
        hot_seen_artifacts={artifact_id: metadata.created_at},
    )
    assert mismatch.replaced_count == 0
    assert mismatch.messages[0]["content"] == "z" * 20

    store.rows.clear()
    unavailable = controller.rewrite_cold_view(
        [message], ctx=_ctx(), session_id="session", cutoff=cutoff,
        hot_seen_artifacts={artifact_id: metadata.created_at},
    )
    assert unavailable.replaced_count == 0
    assert unavailable.messages[0]["content"] == "z" * 20


def test_cache_delay_reclaims_inactive_cold_capacity_before_observing_new_artifact():
    store = _MemoryStore()
    controller = _cache_delay(
        store,
        lambda: datetime.now(UTC),
        immediate=3_000,
    )
    message = _tool_message("n" * 2_000, "new-message")
    controller.govern_persistent_messages(
        [message], ctx=_ctx(), session_id="session", complete_turn=False,
    )
    metadata, _output = next(iter(store.rows.values()))
    old_seen_at = "2026-01-01T00:00:00+00:00"
    state = {
        "last_model_request_at": "2026-01-03T00:00:00+00:00",
        "cold_cutover_before": "2026-01-02T00:00:00+00:00",
        "hot_seen_artifacts": {
            f"tr_{index:064x}": old_seen_at
            for index in range(controller.MAX_TRACKED_ARTIFACTS)
        },
    }

    observation = controller.observe_hot_view(
        [message],
        state=state,
        ctx=_ctx(),
        session_id="session",
        observed_at="2026-01-03T00:00:00+00:00",
    )

    assert observation.pruned_count == controller.MAX_TRACKED_ARTIFACTS
    assert observation.observed_count == 1
    assert observation.overflow_count == 0
    assert observation.state.hot_seen_artifacts == {
        metadata.artifact_id: "2026-01-03T00:00:00+00:00",
    }
    cold = controller.rewrite_cold_view(
        [message],
        ctx=_ctx(),
        session_id="session",
        cutoff="2026-01-04T00:00:00+00:00",
        hot_seen_artifacts=observation.state.hot_seen_artifacts,
    )
    assert cold.replaced_count == 1
    assert "[ogmemory-tool-result v=2]" in cold.messages[0]["content"]


def test_cache_delay_does_not_prune_active_artifact_when_store_read_fails():
    store = _MemoryStore()
    controller = _cache_delay(store, lambda: datetime.now(UTC))
    message = _tool_message("u" * 20, "active-message")
    controller.govern_persistent_messages(
        [message], ctx=_ctx(), session_id="session", complete_turn=False,
    )
    artifact_id = next(iter(store.rows))
    store.rows.clear()
    state = {
        "last_model_request_at": "2026-01-03T00:00:00+00:00",
        "cold_cutover_before": "2026-01-02T00:00:00+00:00",
        "hot_seen_artifacts": {
            artifact_id: "2026-01-01T00:00:00+00:00",
        },
    }

    observation = controller.observe_hot_view(
        [message],
        state=state,
        ctx=_ctx(),
        session_id="session",
        observed_at="2026-01-03T00:00:00+00:00",
    )

    assert observation.pruned_count == 0
    assert observation.observed_count == 0
    assert observation.state.hot_seen_artifacts == state["hot_seen_artifacts"]
def test_memory_service_optimization_off_preserves_main_medium_result_path(tmp_path):
    service = _isolate_service(MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            vector_db_type="memory",
            tool_result_single_chars=100,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    ))
    turn = _user_message_tool_turn("m" * 20, "message-1")
    result = _register_user_message_turn(service, turn)
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    tool_result = next(
        message for message in buffer.messages if message.role == "toolResult"
    )

    assert service._tool_result_governance._tool_result_cache_delay is None
    assert result["ok"] is True
    assert tool_result.content == "m" * 20
    assert ToolResultCacheDelay.STATE_KEY not in buffer.runtime_extensions
    assert service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"] == []
def test_memory_service_cache_delay_persists_stub_then_switches_caller_after_ttl(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            vector_db_type="memory",
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=3_000,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=10,
            tool_result_cache_delay_idle_ttl_seconds=30,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    now = [datetime.now(UTC) + timedelta(seconds=1)]
    service._tool_result_governance._tool_result_cache_delay.clock = lambda: now[0]
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    turn = _user_message_tool_turn("d" * 2_500, "message-1")
    _register_user_message_turn(service, turn)
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert any(
        "[ogmemory-tool-result v=2]" in message.content
        for message in buffer.messages
        if message.role == "toolResult"
    )

    hot = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert any(row.get("content") == "d" * 2_500 for row in hot["messages"])

    now[0] += timedelta(seconds=31)
    cold = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    assert any(
        "[ogmemory-tool-result v=2]" in str(row.get("content", ""))
        for row in cold["messages"]
    )


def test_cache_delay_never_cold_rewrites_bounded_read_source_fragments(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=1_000,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=10,
            tool_result_cache_delay_idle_ttl_seconds=30,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = [start]
    service._tool_result_governance._tool_result_cache_delay.clock = lambda: now[0]
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    first_view = service.govern_first_view_tool_result({
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "source-turn",
        "toolCallId": "source-call",
        "toolName": "fixture",
        "content": [{"type": "text", "text": "read-source-" * 1_000}],
        "rankingPrompt": "inspect",
        "_ctx": _ctx(),
    })
    assert first_view["externalized"] == 1
    artifact = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"][0]
    envelope = service.read_tool_result_artifact({
        "sessionId": "session",
        "artifactId": artifact["artifact_id"],
        "length_bytes": 64,
        "source_tool_call_id": "read-call",
        "_ctx": _ctx(),
    })
    projection = "structuredContent:\n" + json.dumps(
        envelope, ensure_ascii=False, indent=2,
    )
    turn = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "read-call",
                "name": "ogmemory__read_tool_result_artifact",
                "arguments": {},
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "read-call",
            "toolName": "ogmemory__read_tool_result_artifact",
            "content": projection,
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]

    hot = service.compose({
        "sessionId": "session", "prompt": "hot", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    _register_user_message_turn(service, turn)
    now[0] = start + timedelta(seconds=31)
    cold_epoch = service.compose({
        "sessionId": "session", "prompt": "cold", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })

    for result in (hot, cold_epoch):
        assert any(
            row.get("content") == projection
            for row in result["messages"]
            if isinstance(row, dict)
        )
    buffer = manager.get_or_create("session", ctx=_ctx())
    assert all(projection not in message.content for message in buffer.messages)
    assert any(
        (fields := parse_stub(message.content)) is not None
        and fields["ref"] == artifact["ref"]
        for message in buffer.messages
    )


def test_cache_delay_never_cold_rewrites_precise_search_source_collection(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=1_000,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=10,
            tool_result_cache_delay_idle_ttl_seconds=30,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = [start]
    service._tool_result_governance._tool_result_cache_delay.clock = lambda: now[0]
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    first_view = service.govern_first_view_tool_result({
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "search-source-turn",
        "toolCallId": "search-source-call",
        "toolName": "fixture",
        "content": [{
            "type": "text",
            "text": ("prefix\n" * 300) + "NEEDLE request=req-42\n" + ("suffix\n" * 300),
        }],
        "rankingPrompt": "find NEEDLE",
        "_ctx": _ctx(),
    })
    assert first_view["externalized"] == 1
    search = service.search_tool_result_artifacts({
        "sessionId": "session",
        "query": "NEEDLE",
        "top_k": 1,
        "max_chars": 500,
        "source_tool_call_id": "search-call",
        "_ctx": _ctx(),
    })
    assert search["results"]
    projection = "structuredContent:\n" + json.dumps(
        search, ensure_ascii=False, indent=2,
    )
    turn = [
        {"role": "user", "content": "find the request id"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "search-call",
                "name": "ogmemory__search_tool_result_artifacts",
                "arguments": {},
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "search-call",
            "toolName": "ogmemory__search_tool_result_artifacts",
            "content": projection,
        },
        {"role": "assistant", "content": "req-42", "stopReason": "stop"},
    ]

    hot = service.compose({
        "sessionId": "session", "prompt": "hot", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })
    _register_user_message_turn(service, turn)
    now[0] = start + timedelta(seconds=31)
    cold_epoch = service.compose({
        "sessionId": "session", "prompt": "cold", "messages": deepcopy(turn),
        "semanticTurnMode": "user_message", "_ctx": _ctx(),
    })

    for result in (hot, cold_epoch):
        assert any(
            row.get("content") == projection
            for row in result["messages"]
            if isinstance(row, dict)
        )
    buffer = manager.get_or_create("session", ctx=_ctx())
    assert all(projection not in message.content for message in buffer.messages)
    assert any(
        parse_stub(message.content) is not None
        for message in buffer.messages
    )
def test_cache_delay_turn_completion_anchors_first_post_ttl_cold_view(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            vector_db_type="memory",
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=3_000,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=10,
            tool_result_cache_delay_idle_ttl_seconds=300,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    now = [start]
    service._tool_result_governance._tool_result_cache_delay.clock = lambda: now[0]
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    turn = _user_message_tool_turn("a" * 2_500, "async-message")

    service.compose({
        "sessionId": "session",
        "prompt": "start",
        "messages": [{"role": "user", "content": "start"}],
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    now[0] = start + timedelta(seconds=10)
    _register_user_message_turn(service, turn)
    state = manager.get_or_create(
        "session", ctx=_ctx(),
    ).runtime_extensions["tool_result_cache_delay"]
    assert state["last_model_request_at"] == (start + timedelta(seconds=10)).isoformat()

    now[0] = start + timedelta(seconds=310)
    first_visible = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    first_content = next(
        row["content"] for row in first_visible["messages"]
        if row.get("role") == "toolResult"
    )
    assert "[ogmemory-tool-result v=2]" in first_content

    now[0] = start + timedelta(seconds=611)
    cold = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": deepcopy(turn),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    cold_content = next(
        row["content"] for row in cold["messages"]
        if row.get("role") == "toolResult"
    )
    assert "[ogmemory-tool-result v=2]" in cold_content
def test_memory_service_cache_delay_threshold_boundaries(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=20_000,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=2_000,
            tool_result_cache_delay_idle_ttl_seconds=300,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True

    expected = {
        1_999: (False, False),
        2_000: (True, False),
        19_999: (True, False),
        20_000: (True, True),
        20_001: (True, True),
    }
    for length, (buffer_is_stub, compose_is_stub) in expected.items():
        session_id = f"session-{length}"
        ctx = _ctx(session_id)
        turn = _user_message_tool_turn("x" * length, f"message-{length}")
        _register_user_message_turn(service, turn, session_id=session_id)
        buffer = manager.get_or_create(session_id, ctx=ctx)
        buffered_tool_content = next(
            message.content
            for message in buffer.messages
            if message.role == "toolResult"
        )
        assert (
            "[ogmemory-tool-result v=2]" in buffered_tool_content
        ) is buffer_is_stub

        composed = service.compose({
            "sessionId": session_id,
            "prompt": "continue",
            "messages": deepcopy(turn),
            "semanticTurnMode": "user_message",
            "_ctx": ctx,
        })
        tool_content = next(
            row["content"]
            for row in composed["messages"]
            if row.get("role") == "toolResult"
        )
        assert ("[ogmemory-tool-result v=2]" in tool_content) is compose_is_stub
def test_memory_service_cache_delay_state_save_failure_preserves_hot_caller(tmp_path):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_single_chars=100,
            tool_result_cache_delay_enabled=True,
            tool_result_cache_delay_chars=10,
            tool_result_cache_delay_idle_ttl_seconds=30,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_context_fs = lambda: None
    message = _tool_message("f" * 20, "message-1")
    service.ingest({
        "sessionId": "session", "messages": [message], "_ctx": _ctx(),
    })
    controller = service._tool_result_governance._tool_result_cache_delay
    controller.clock = lambda: datetime(2026, 1, 2, tzinfo=UTC)
    manager = service.get_session_manager()
    buffer = manager.get_or_create("session", ctx=_ctx())
    buffer.runtime_extensions[controller.STATE_KEY] = {
        "last_model_request_at": "2026-01-01T00:00:00+00:00",
        "cold_cutover_before": "2026-01-01T00:00:00+00:00",
    }
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: False

    result = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": [message],
        "_ctx": _ctx(),
    })

    assert any(row.get("content") == "f" * 20 for row in result["messages"])
def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
