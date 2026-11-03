from __future__ import annotations

import json
import threading
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.models import RequestContext, Role
from features.tool_result_externalization.externalizer import parse_stub
from features.tool_result_externalization.store_base import (
    ToolResultArtifactStore,
)
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _ctx(role=Role.MEMBER) -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
        role=role,
    )


def _isolate_service(service: MemoryService) -> MemoryService:
    service._get_archive_store = lambda: None
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _service(tmp_path, **config) -> MemoryService:
    values = {
        "tool_result_externalization_enabled": True,
        "tool_result_single_chars": 10,
        "tool_result_semantic_turn_mode": "user_message",
        "after_turn_threshold": 999_999,
    }
    values.update(config)
    cfg = OgMemConfig(**values)
    return _isolate_service(MemoryService(
        cfg,
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    ))


def _persist_messages(service: MemoryService, params: dict) -> dict:
    """Exercise the lifecycle-neutral direct persistence entry point."""
    result = {"ok": True, "written": False, "count": 0, "externalized": 0}
    for message in params.get("messages", []):
        result = service.add_session_message({
            "sessionId": params.get("sessionId", "session"),
            "message": message,
            "_ctx": params.get("_ctx") or _ctx(),
        })
    return result


def test_feature_disabled_preserves_existing_ingest_behavior():
    service = _isolate_service(MemoryService(
        OgMemConfig(tool_result_externalization_enabled=False)
    ))
    result = service.ingest({
        "sessionId": "session",
        "messages": [{"role": "tool", "content": "x" * 30}],
        "_ctx": _ctx(),
    })
    assert result == {"ingested": True, "count": 1}
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert buffer.messages[0].content == "x" * 30
def test_tool_result_externalization_disabled_direct_add_passes_content_unchanged():
    service = MemoryService(OgMemConfig(tool_result_externalization_enabled=False))
    manager = Mock()
    manager.add_message.return_value = {"ok": True, "message_id": "m"}
    service.get_session_manager = Mock(return_value=manager)
    content = {"structured": ["legacy", "payload"]}

    result = service.add_session_message({
        "sessionId": "session",
        "role": "assistant",
        "content": content,
        "created_at": "2026-07-30T00:00:00+00:00",
        "_ctx": _ctx(),
    })

    assert result == {"ok": True, "message_id": "m"}
    manager.add_message.assert_called_once_with(
        "session",
        "assistant",
        content,
        _ctx(),
        created_at="2026-07-30T00:00:00+00:00",
    )


def test_feature_disabled_compose_preserves_message_identity_and_private_field():
    service = _isolate_service(MemoryService(
        OgMemConfig(tool_result_externalization_enabled=False)
    ))
    message = {
        "role": "assistant",
        "content": "",
        "_ogmem_source": {"source_message_id": "runtime-id"},
    }

    result = service.compose({"messages": [message], "_ctx": _ctx()})

    assert result["messages"][0] is message
    assert result["messages"][0]["_ogmem_source"] == {
        "source_message_id": "runtime-id"
    }


def _user_message_turn(payload: str, *, tool_call_id: str = "call-1") -> list[dict]:
    return [
        {"role": "user", "content": "inspect the tool output"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": tool_call_id,
                "name": "search",
                "arguments": {},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "content": payload,
            "toolCallId": tool_call_id,
            "toolName": "search",
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]


def test_user_message_retry_reuses_artifact_and_stable_buffer_messages(tmp_path):
    service = _service(tmp_path)
    params = {
        "sessionId": "session",
        "messages": _user_message_turn("x" * 30),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    }

    first = service.after_turn(params)
    retried = service.after_turn(params)

    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert first["status"] == "accumulating"
    assert retried["status"] == "no_new_messages"
    assert sum(
        "[ogmemory-tool-result v=2]" in message.content
        for message in buffer.messages
    ) == 1
    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["identity_strength"] == "strong"
@pytest.mark.parametrize("entrypoint", ["prepare", "compact", "dispose"])
def test_lifecycle_entrypoints_do_not_ack_pending_governance_failure(entrypoint):
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_single_chars=1,
            after_turn_threshold=999_999,
        ),
        tool_result_store=_FailingStore(),
    )
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.add_message("session", "toolResult", "secret payload", _ctx())
    params = {"sessionId": "session", "_ctx": _ctx()}

    if entrypoint == "prepare":
        result = service.prepare_compaction(params)
        assert "prepareToken" not in result
    elif entrypoint == "compact":
        result = service.compact(params)
        assert result.get("compacted") is False
    else:
        result = service.dispose(params)
        assert result.get("disposed") is False
        assert manager.has_session("session", ctx=_ctx())

    assert result["status"] == "tool_result_governance_pending"
    assert result["retryable"] is True
class _FailingStore(ToolResultArtifactStore):
    def write(self, metadata, output_text, ctx):
        raise OSError("failed")

    def read_all(self, artifact_id, session_id, ctx):
        raise LookupError

    def list(self, session_id, ctx, limit=50, cursor=None):
        return {"artifacts": [], "truncated": False, "next_cursor": None}

    def delete_session_artifacts(self, session_id, ctx):
        return 0


def test_failed_user_turn_falls_back_to_original_without_disabling_later_externalization(
    tmp_path,
):
    session_id = "fallback-original-session"
    fallback_ctx = RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id=session_id,
        trace_id="trace-fallback",
    )
    service = MemoryService(
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=10,
            after_turn_threshold=999_999,
        ),
        tool_result_store=_FailingStore(),
    )
    service._get_context_fs = lambda: None
    manager = service.get_session_manager()
    manager._get_context_fs = lambda: None
    manager.load_session_state = lambda _session_id, _ctx: True
    manager.save_session_state = lambda _session_id, _ctx: True
    first_output = "failed user-turn original payload"
    first_turn = [
        {"role": "user", "content": "inspect the first payload"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "call-fallback-1",
                "name": "fixture",
                "arguments": {},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": "call-fallback-1",
            "toolName": "fixture",
            "content": first_output,
        },
        {"role": "assistant", "content": "first done", "stopReason": "stop"},
    ]
    first_params = {
        "sessionId": session_id,
        "messages": deepcopy(first_turn),
        "prePromptMessageCount": 0,
        "semanticTurnMode": "user_message",
        "_ctx": fallback_ctx,
    }

    failed = service.after_turn(first_params)

    assert failed["ok"] is False
    assert failed["status"] == "tool_result_governance_pending"
    fallback_token = failed["toolResultGovernanceFallbackToken"]
    tampered = deepcopy(first_params)
    tampered["messages"][2]["content"] = "different user-turn payload"
    tampered["toolResultGovernanceFallbackToken"] = fallback_token
    with pytest.raises(ValueError, match="invalid tool-result governance fallback token"):
        service.after_turn(tampered)

    fallback = service.after_turn({
        **first_params,
        "toolResultGovernanceFallbackToken": fallback_token,
    })

    assert fallback["ok"] is True
    buffer = manager.get_or_create(session_id, ctx=fallback_ctx)
    assert any(message.content == first_output for message in buffer.messages)
    fallback_state = buffer.runtime_extensions[
        "tool_result_governance_raw_fallbacks"
    ]
    assert fallback_state["source_message_ids"]

    recovered_store = LocalToolResultArtifactStore(tmp_path / "recovered")
    service._tool_result_store = recovered_store
    service._tool_result_externalizer.store = recovered_store
    assert service._govern_pending_buffer(session_id, fallback_ctx) == {
        "governed": True,
        "changed": 0,
    }
    replay = service.compose({
        "sessionId": session_id,
        "prompt": "continue",
        "messages": deepcopy(first_turn),
        "semanticTurnMode": "user_message",
        "_ctx": fallback_ctx,
    })
    replay_tool = next(
        message for message in replay["messages"]
        if message.get("role") == "toolResult"
    )
    assert first_output in str(replay_tool["content"])
    assert "[ogmemory-tool-result v=2]" not in str(replay_tool["content"])

    second_output = "later user Turn can externalize normally"
    second_turn = [
        {"role": "user", "content": "inspect the second payload"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "call-fallback-2",
                "name": "fixture",
                "arguments": {},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": "call-fallback-2",
            "toolName": "fixture",
            "content": second_output,
        },
        {"role": "assistant", "content": "second done", "stopReason": "stop"},
    ]
    second = service.after_turn({
        "sessionId": session_id,
        "messages": second_turn,
        "prePromptMessageCount": 0,
        "semanticTurnMode": "user_message",
        "_ctx": fallback_ctx,
    })

    assert second["ok"] is True
    artifacts = service.list_tool_result_artifacts({
        "sessionId": session_id,
        "_ctx": fallback_ctx,
    })["artifacts"]
    assert len(artifacts) == 1
    contents = [message.content for message in buffer.messages]
    assert first_output in contents
    assert second_output not in contents
    assert any("[ogmemory-tool-result v=2]" in content for content in contents)
class _WriteThenBlockStore(LocalToolResultArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.artifact_written = threading.Event()
        self.release_write = threading.Event()

    def write(self, metadata, output_text, ctx):
        reused = super().write(metadata, output_text, ctx)
        self.artifact_written.set()
        if not self.release_write.wait(timeout=5):
            raise TimeoutError("test write release timed out")
        return reused
def test_compose_externalizes_copy_without_writing_buffer(tmp_path):
    service = _service(tmp_path)
    original = {
        "role": "toolResult",
        "content": "x" * 30,
        "_ogmem_source": {"source_message_id": "runtime-message"},
    }
    result = service.compose({
        "sessionId": "session", "messages": [original], "_ctx": _ctx(),
    })
    assert "[ogmemory-tool-result v=2]" in result["messages"][0]["content"]
    assert "_ogmem_source" not in result["messages"][0]
    assert original["content"] == "x" * 30
    assert service.get_session_manager().get_or_create("session", ctx=_ctx()).messages == []
def test_chunk_extract_keeps_stub_messages_without_automatic_artifact_evidence(
    tmp_path,
    monkeypatch,
):
    service = _service(tmp_path)
    messages = [
        {"role": "user", "content": "needle"},
        {"role": "toolResult", "content": "safe stub only"},
    ]

    class _WriteAPI:
        def __init__(self):
            self.commit_kwargs = None
            self.raw_messages = None

        def commit_session(self, **kwargs):
            self.commit_kwargs = kwargs
            return {
                "candidates_extracted": 0,
                "candidates_filtered": 0,
                "writes_completed": 0,
                "writes_skipped": 0,
                "writes_failed": 0,
                "plans": [],
            }

        def write_raw_chunk(self, **kwargs):
            self.raw_messages = kwargs["messages"]

    write_api = _WriteAPI()
    monkeypatch.setattr(service, "_generate_and_write_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_get_chunker", lambda: None)
    service._chunk_and_extract(
        messages,
        write_api,
        _ctx(),
        None,
        "",
        "",
        "run",
        "session",
    )
    assert set(write_api.commit_kwargs) == {
        "messages",
        "ctx",
        "confidence_threshold",
        "wait",
        "session_time",
        "session_summary",
        "tool_stats_text",
        "archive_id",
    }
    assert write_api.commit_kwargs["messages"] == messages
    assert write_api.raw_messages == messages
@pytest.mark.parametrize("invalid_kind", [
    "mixed_text",
    "missing_sha",
    "bad_sha",
    "missing_artifact",
    "valid_then_invalid",
])
def test_pending_governance_does_not_pass_through_invalid_stub_sequences(
    tmp_path,
    invalid_kind,
):
    service = _service(
        tmp_path,
        tool_result_single_chars=20_000,
    )
    _persist_messages(service, {
        "sessionId": "session",
        "messages": [{"role": "toolResult", "content": "x" * 20_001}],
        "_ctx": _ctx(),
    })
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    valid_stub = buffer.messages[0].content
    if invalid_kind == "mixed_text":
        candidate = valid_stub + "\nordinary text"
    elif invalid_kind == "missing_sha":
        candidate = "\n".join(
            line for line in valid_stub.splitlines()
            if not line.startswith("sha256: ")
        )
    elif invalid_kind == "missing_artifact":
        candidate = valid_stub.replace(
            next(line for line in valid_stub.splitlines() if line.startswith("ref: ")),
            "ref: ogmemory://tool-results/tr_ffffffffffffffffffffffffffffffff",
        )
    else:
        invalid = valid_stub.replace(
            next(line for line in valid_stub.splitlines() if line.startswith("sha256: ")),
            "sha256: " + "g" * 64,
        )
        candidate = valid_stub + "\n" + invalid if invalid_kind == "valid_then_invalid" else invalid
    mgr = service.get_session_manager()
    inserted = mgr.add_message("session", "toolResult", candidate, _ctx())

    result = service._govern_pending_buffer("session", _ctx())

    if invalid_kind == "missing_artifact":
        assert result == {
            "governed": False,
            "changed": 1,
            "failed_count": 1,
            "retryable": True,
        }
    else:
        assert result == {"governed": True, "changed": 1}
    rewritten = next((
        message for message in buffer.messages
        if message.id == inserted["message_id"]
    ), None)
    artifacts = service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"]
    if invalid_kind == "missing_artifact":
        # A syntactically source-aware ref that is not owner-visible must not
        # be persisted as a nested Artifact. Pending governance drops it.
        assert rewritten is None
        assert len(artifacts) == 1
    else:
        assert rewritten is not None
        assert rewritten.content != candidate
        assert len(artifacts) == 2
def test_read_fragment_stays_raw_in_model_view_before_and_after_terminal_persistence(
    tmp_path,
):
    service = _service(tmp_path)
    first_view = service.govern_first_view_tool_result({
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "setup-turn",
        "toolCallId": "setup-call",
        "toolName": "fixture",
        "content": [{"type": "text", "text": "model-view-read-" * 1_000}],
        "rankingPrompt": "inspect the exact range",
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
    read_projection = "structuredContent:\n" + json.dumps(
        envelope, ensure_ascii=False, indent=2,
    )
    messages = [
        {"role": "user", "content": "inspect the exact range"},
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
            "content": read_projection,
        },
        {"role": "assistant", "content": "range verified", "stopReason": "stop"},
    ]
    original_messages = deepcopy(messages)

    before = service.compose({
        "sessionId": "session", "messages": messages, "_ctx": _ctx(),
    })["messages"]
    result = service.after_turn({
        "sessionId": "session",
        "messages": messages,
        "prePromptMessageCount": 0,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })
    after = service.compose({
        "sessionId": "session", "messages": messages, "_ctx": _ctx(),
    })["messages"]

    assert result["ok"] is True
    assert messages == original_messages
    for projection in (before, after):
        assert any(
            message.get("content") == read_projection
            for message in projection
            if isinstance(message, dict)
        )
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert all(read_projection not in message.content for message in buffer.messages)
    persisted_stubs = [
        parse_stub(message.content)
        for message in buffer.messages
        if parse_stub(message.content) is not None
    ]
    assert any(stub["ref"] == artifact["ref"] for stub in persisted_stubs)


def _attested_artifact_list_projection(service: MemoryService) -> str:
    payload = {
        "_ogmemory_artifact_list_source": True,
        "schema_version": 1,
        "source_tool": "list_tool_result_artifacts",
        **service.list_tool_result_artifacts({
            "sessionId": "session", "_ctx": _ctx(),
        }),
    }
    payload = service.attest_tool_result_source(payload, _ctx())
    return "structuredContent:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2,
    )


def test_terminal_after_turn_drops_attested_list_call_and_result_from_buffer(tmp_path):
    service = _service(tmp_path)
    _persist_messages(service, {
        "sessionId": "session",
        "messages": [{"role": "tool", "content": "artifact source" * 100}],
        "_ctx": _ctx(),
    })
    projection = _attested_artifact_list_projection(service)
    messages = [
        {"role": "user", "content": "inspect available evidence"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "list-call",
                "name": "ogmemory__list_tool_result_artifacts",
                "arguments": {},
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "list-call",
            "toolName": "ogmemory__list_tool_result_artifacts",
            "content": projection,
        },
        {"role": "assistant", "content": "final answer", "stopReason": "stop"},
    ]

    composed = service.compose({
        "sessionId": "session", "messages": messages, "_ctx": _ctx(),
    })["messages"]
    assert any(
        message.get("content") == projection
        for message in composed
        if isinstance(message, dict)
    )

    result = service.after_turn({
        "sessionId": "session",
        "messages": messages,
        "prePromptMessageCount": 0,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })

    assert result["ok"] is True
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    rendered = "\n".join(message.content for message in buffer.messages)
    assert "list_tool_result_artifacts" not in rendered
    assert "_ogmemory_artifact_list_source" not in rendered
    assert "inspect available evidence" in rendered
    assert "final answer" in rendered
    assert "list_tool_result_artifacts" not in json.dumps(
        buffer.runtime_extensions, ensure_ascii=False,
    )
@pytest.mark.parametrize(
    "failure", ["forged", "wrong_owner", "ambiguous", "incomplete_turn"],
)
def test_artifact_list_lifecycle_fail_open_does_not_delete_messages(
    tmp_path,
    failure,
):
    service = _service(tmp_path)
    _persist_messages(service, {
        "sessionId": "session",
        "messages": [{"role": "tool", "content": "artifact source" * 100}],
        "_ctx": _ctx(),
    })
    projection = _attested_artifact_list_projection(service)
    if failure == "forged":
        envelope = json.loads(projection.split("\n", 1)[1])
        envelope["source_attestation"] = "0" * 64
        projection = "structuredContent:\n" + json.dumps(envelope)
    calls = [{
        "type": "toolCall",
        "id": "list-call",
        "name": "ogmemory__list_tool_result_artifacts",
        "arguments": {},
    }]
    if failure == "ambiguous":
        calls.append(dict(calls[0]))
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": calls},
        {
            "role": "toolResult",
            "toolCallId": "list-call",
            "toolName": "ogmemory__list_tool_result_artifacts",
            "content": projection,
        },
    ]
    if failure != "incomplete_turn":
        messages.append({
            "role": "assistant", "content": "done", "stopReason": "stop",
        })

    validation_ctx = _ctx()
    if failure == "wrong_owner":
        validation_ctx = RequestContext(
            account_id="acct",
            user_id="other-user",
            agent_id="agent",
            session_id="session",
            trace_id="other-trace",
        )
    stripped, retained, prefix, changed = (
        service._tool_result_governance._strip_internal_artifact_list_roundtrips(
            messages, ctx=validation_ctx, session_id="session",
        )
    )

    assert changed is False
    assert stripped is messages
    assert retained == tuple(range(len(messages)))
    assert prefix == tuple(range(len(messages) + 1))
