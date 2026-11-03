from __future__ import annotations

import json
import re

from core.models import RequestContext
from features.tool_result_externalization.store_base import ToolResultArtifactStore
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _params(tool_text, **overrides):
    values = {
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "turn-1",
        "toolCallId": "call-1",
        "toolName": "fixture",
        "content": [{"type": "text", "text": tool_text}],
        "rankingPrompt": "Find the exact failing request id",
        "_ctx": _ctx(),
    }
    values.update(overrides)
    return values


def _service(tmp_path, **overrides) -> MemoryService:
    values = {
        "tool_result_externalization_enabled": True,
        "tool_result_first_view_enabled": True,
        "tool_result_single_chars": 20,
        "tool_result_semantic_turn_mode": "user_message",
        "after_turn_threshold": 999_999,
    }
    values.update(overrides)
    service = MemoryService(
        OgMemConfig(**values),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_context_fs = lambda: None
    service._get_archive_store = lambda: None
    manager = service.get_session_manager()
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _artifacts(service: MemoryService):
    return service.list_tool_result_artifacts({
        "sessionId": "session",
        "_ctx": _ctx(),
    })["artifacts"]
def test_first_view_identity_conflict_creates_immutable_conflict_artifact(tmp_path):
    service = _service(tmp_path)
    original = "A" * 80
    changed_retry = "B" * 80

    first = service.govern_first_view_tool_result(_params(original))
    conflict = service.govern_first_view_tool_result(_params(changed_retry))

    assert first["externalized"] == 1
    assert conflict["externalized"] == 1
    assert conflict["failed"] == 0
    assert conflict["reason"] == "externalized"
    artifacts = _artifacts(service)
    assert len(artifacts) == 2
    stored_outputs = set()
    artifact_ids = set()
    for artifact in artifacts:
        metadata, stored = service._tool_result_store.read_all(
            artifact["artifact_id"], "session", _ctx(),
        )
        assert metadata.identity_strength == "strong"
        stored_outputs.add(stored)
        artifact_ids.add(metadata.artifact_id)
    assert stored_outputs == {original, changed_retry}
    assert any(re.fullmatch(r"tr_[a-f0-9]{32}_[a-f0-9]{12}", item) for item in artifact_ids)
def test_first_view_uses_primary_threshold_not_cache_delay_threshold(tmp_path):
    service = _service(
        tmp_path,
        tool_result_single_chars=100,
        tool_result_cache_delay_enabled=True,
        tool_result_cache_delay_chars=10,
    )
    payload = "medium-result" * 2

    result = service.govern_first_view_tool_result(_params(payload))

    assert result["externalized"] == 0
    assert result["content"] == [{"type": "text", "text": payload}]
    assert _artifacts(service) == []
def test_first_view_preserves_bounded_read_fragment_even_above_threshold(tmp_path):
    service = _service(tmp_path)
    payload = "ERROR request_id=req-123 failed\n" * 4
    service.govern_first_view_tool_result(_params(payload))
    artifact = _artifacts(service)[0]
    envelope = service.read_tool_result_artifact({
        "sessionId": "session",
        "artifactId": artifact["artifact_id"],
        "length_bytes": 12,
        "source_tool_call_id": "read-1",
        "_ctx": _ctx(),
    })
    content = [{
        "type": "text",
        "text": "structuredContent:\n" + json.dumps(
            envelope, ensure_ascii=False, indent=2,
        ),
    }]

    result = service.govern_first_view_tool_result(_params(
        "unused",
        toolCallId="read-1",
        toolName="ogmemory__read_tool_result_artifact",
        content=content,
    ))

    assert result["externalized"] == 0
    assert result["failed"] == 0
    assert result["content"] == content
    assert len(_artifacts(service)) == 1
class _FailingStore(ToolResultArtifactStore):
    def write(self, metadata, output_text, ctx):
        raise OSError("store unavailable")

    def read_all(self, artifact_id, session_id, ctx):
        raise FileNotFoundError

    def list(self, session_id, ctx, limit=50, cursor=None):
        return {"artifacts": [], "truncated": False, "next_cursor": None}

    def delete_session_artifacts(self, session_id, ctx):
        return 0


def test_first_view_store_failure_returns_original_model_content():
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_first_view_enabled=True,
            tool_result_single_chars=1,
            tool_result_retry_attempts=1,
        ),
        tool_result_store=_FailingStore(),
    )
    service._get_context_fs = lambda: None
    payload = "secret payload"

    result = service.govern_first_view_tool_result(_params(payload))

    assert result["externalized"] == 0
    assert result["failed"] == 1
    assert result["reason"] == "fail_open"
    assert result["content"] == [{"type": "text", "text": payload}]
