"""End-to-end verification for the compact mechanism PRD.

These tests exercise MemoryService through public lifecycle methods while using
in-memory fakes for storage, read, write, and LLM dependencies. They are meant
to cover the compact, rolling, after_turn, SessionState persistence, and
TopicBuffer integration paths without requiring a local AGFS/SQL/HTTP stack.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from core.models import RequestContext, RetrievedBlock, SearchMemoryResult
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.archive_store import SessionArchiveStore


class StructuredCompactLLM:
    def __init__(self):
        self.calls: list[str] = []

    def complete_json(self, prompt, schema):
        self.calls.append(prompt)
        if "Analyze this conversation and extract structured state" in prompt:
            return {
                "active_task": "Validate compact mechanism end to end",
                "confirmed_constraints": ["Use deterministic in-memory fakes"],
                "recent_decisions": ["Enable rolling compression by default"],
                "open_loops": ["Verify archive and state persistence"],
                "uncertainties": ["None"],
                "summary": "The compact mechanism E2E validates rolling state.",
            }
        return {
            "overview": (
                "## Compact Archive\n"
                "- Prepared compaction extracted user memories.\n"
                "- Commit archived the buffered session."
            ),
            "abstract": "Compact archive summary",
        }


class RecordingWriteAPI:
    def __init__(self):
        self.commit_calls: list[dict] = []
        self.raw_chunks: list[dict] = []

    def commit_session(self, **kwargs):
        self.commit_calls.append(kwargs)
        archive_id = kwargs.get("archive_id") or "archive-e2e"
        return {
            "candidates_extracted": 1,
            "candidates_filtered": 0,
            "writes_completed": 1,
            "writes_skipped": 0,
            "writes_failed": 0,
            "plans": [
                {
                    "action": "create",
                    "category": "case",
                    "target_uri": (
                        "ctx://acct-e2e/users/user-e2e/memories/case/"
                        f"{archive_id}"
                    ),
                }
            ],
        }

    def write_raw_chunk(self, **kwargs):
        self.raw_chunks.append(kwargs)

    def write_memory(self, candidate, ctx):
        return None


@dataclass
class StaticPrefetchReadAPI:
    hits: list[RetrievedBlock]

    def read_memory(self, uri, ctx, **kwargs):
        return next(hit for hit in self.hits if hit.uri == uri)

    def search_memory(self, **kwargs):
        categories = kwargs.get("categories")
        if categories == ["session_summary"]:
            return SearchMemoryResult(query=kwargs.get("query", ""), hits=[])
        if kwargs.get("fill_content_for_top_k") != 0:
            return SearchMemoryResult(query=kwargs.get("query", ""), hits=[])
        return SearchMemoryResult(query=kwargs.get("query", ""), hits=list(self.hits))


def _ctx(session_id: str = "compact-e2e") -> RequestContext:
    return RequestContext(
        account_id="acct-e2e",
        user_id="user-e2e",
        agent_id="agent-e2e",
        session_id=session_id,
        trace_id="trace-e2e",
    )


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _service(
    memory_fs,
    *,
    session_id: str = "compact-e2e",
    after_turn_threshold: int = 999999,
) -> tuple[MemoryService, RequestContext, RecordingWriteAPI]:
    cfg = OgMemConfig(
        account_id="acct-e2e",
        user_id="user-e2e",
        agent_id="agent-e2e",
        rolling_compress_enabled=True,
        rolling_compress_fallback_enabled=False,
        after_turn_threshold=after_turn_threshold,
        prefetch_enabled=True,
        topic_detection_enabled=True,
        compact_prepare_token_ttl=300,
    )
    service = MemoryService(config=cfg)
    service._get_context_fs = lambda: memory_fs
    archive_store = SessionArchiveStore(memory_fs)
    service._get_archive_store = lambda: archive_store
    service.drain_outbox_sync = lambda account_id=None: {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
    }
    service._async_drain = lambda account_id=None: None
    service._llm = StructuredCompactLLM()
    write_api = RecordingWriteAPI()
    service._write_api = write_api
    service._read_api = StaticPrefetchReadAPI(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-e2e/users/user-e2e/memories/case/compact-runtime",
                score=0.93,
                category="case",
                abstract="Prefetched compact runtime memory",
                overview="Prefetched compact runtime memory",
                content_excerpt="Prefetched compact runtime memory",
            )
        ]
    )
    return service, _ctx(session_id), write_api


def _wait_for_extraction(service: MemoryService, session_id: str, timeout: float = 5.0) -> None:
    buf = service.get_session_manager().get_or_create(session_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        event = buf.extraction_done_event
        if not buf.extraction_in_progress:
            return
        if event is not None:
            event.wait(timeout=0.05)
        else:
            time.sleep(0.05)
    raise AssertionError("background extraction did not finish")


def test_compact_mechanism_end_to_end(memory_fs):
    service, ctx, write_api = _service(memory_fs)
    session_id = ctx.session_id

    for idx in range(10):
        service.after_turn({
            "sessionId": session_id,
            "_ctx": ctx,
            "messages": [
                _msg("user", f"Please validate compact runtime state turn {idx}."),
                _msg("assistant", f"ack {idx}"),
            ],
            "commitTokenThreshold": 999999,
        })

    prepared = service.prepare_compaction({"sessionId": session_id, "_ctx": ctx})
    assert prepared["prepareToken"]
    assert prepared["archive_id"]
    assert service.get_session_manager().get_or_create(session_id).extraction_watermark == 20
    assert write_api.commit_calls[-1]["archive_id"] == prepared["archive_id"]

    compacted = service.compact({
        "sessionId": session_id,
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
        "shortTermIndexMode": "off",
    })
    assert compacted["ok"] is True
    assert compacted["compacted"] is True
    assert "Prepared compaction extracted user memories" in compacted["result"]["summary"]
    assert compacted["result"]["tokensBefore"] > compacted["result"]["tokensAfter"]

    archive_uris = [
        uri for uri in memory_fs.stored_uris
        if f"/sessions/{session_id}/history/" in uri
    ]
    assert len(archive_uris) == 1
    archive_node = memory_fs.read_node(archive_uris[0], ctx)
    assert archive_node.metadata["archive_id"] == prepared["archive_id"]
    assert len(json.loads(archive_node.content)) == 20

    for idx in range(10):
        service.after_turn({
            "sessionId": session_id,
            "_ctx": ctx,
            "messages": [
                _msg("user", f"Please verify rolling context {idx}."),
                _msg("assistant", f"ok {idx}"),
            ],
            "commitTokenThreshold": 999999,
        })
    compose = service.compose({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [_msg("user", "Please debug this compact regression")],
    })
    assert "## Archive History" in compose["episodicContext"]
    assert "## Active Task\nPlease validate compact runtime state turn 9." in compose["sessionContext"]
    assert "## Recent Session Summary\nThe compact mechanism E2E validates rolling state." in compose["sessionContext"]
    assert compose["archiveCount"] == 1
    assert service.get_session_manager().get_topic_buffer(session_id).get_current_topic().label == "debugging"

    service.prefetch({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [_msg("user", "review compact runtime state")],
    })
    first_prefetch_compose = service.compose({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [_msg("user", "review compact runtime state finding")],
    })
    assert "Prefetched compact runtime memory" in first_prefetch_compose["retrievedEvidence"]

    service.prefetch({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [_msg("user", "plan compact runtime follow-up")],
    })
    reinjected_compose = service.compose({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [_msg("user", "plan compact runtime follow-up requirements")],
    })
    assert "Prefetched compact runtime memory" in reinjected_compose["retrievedEvidence"]
    assert service.get_session_manager().get_topic_buffer(session_id).get_current_topic().label == "planning"

    disposed = service.dispose({
        "sessionId": session_id,
        "_ctx": ctx,
    })
    assert disposed["disposed"] is True
    assert disposed["reason"] == "below_token_threshold"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and service.get_session_manager().has_session(
        session_id,
        ctx=ctx,
    ):
        time.sleep(0.01)

    state_uri = f"ctx://acct-e2e/sessions/{session_id}/state.json"
    assert memory_fs.exists(state_uri, ctx)
    state_payload = json.loads(memory_fs.read_node(state_uri, ctx).content)
    assert state_payload["window_state"]["active_task"] == "Please validate compact runtime state turn 9."
    assert state_payload["session_meta"]["account_id"] == "acct-e2e"

    reloaded_service, reloaded_ctx, _ = _service(memory_fs, session_id=session_id)
    reloaded_state = reloaded_service._get_session_state(session_id, reloaded_ctx)
    assert reloaded_state.active_task == "Please validate compact runtime state turn 9."
    assert reloaded_state.compressed_text == "The compact mechanism E2E validates rolling state."


def test_after_turn_threshold_archives_snapshot_and_persists_state(memory_fs):
    service, ctx, write_api = _service(
        memory_fs,
        session_id="compact-e2e-after-turn",
        after_turn_threshold=1,
    )
    session_id = ctx.session_id

    result = service.after_turn({
        "sessionId": session_id,
        "_ctx": ctx,
        "messages": [
            _msg("user", "Please archive this large automatic after_turn batch. " + "x" * 500),
            _msg("assistant", "ack " + "y" * 500),
        ],
        "commitTokenThreshold": 1,
    })
    assert result["ok"] is True
    assert result["status"] == "processing"

    _wait_for_extraction(service, session_id)

    session = service.get_session_manager().get_session(session_id, ctx)
    assert session["message_count"] == 0
    assert session["commit_count"] == 1
    assert write_api.commit_calls
    assert write_api.raw_chunks
    assert any(f"/sessions/{session_id}/history/" in uri for uri in memory_fs.stored_uris)

    state_uri = f"ctx://acct-e2e/sessions/{session_id}/state.json"
    assert memory_fs.exists(state_uri, ctx)
