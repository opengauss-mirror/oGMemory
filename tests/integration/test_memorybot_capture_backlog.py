from __future__ import annotations

from datetime import UTC, datetime

import pytest

from commit.outbox_scope import capture_outbox_events
from memorybot.gateway import MemoryGateway
from memorybot.models import new_id
from server.memory_service import MemoryService
from service.extraction_checkpoint import ExtractionBatchJournal
from service.extraction_receipts import read
from session.session_manager import generate_archive_id
from tests.integration.test_memorybot_runtime import DeterministicLLM, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


class CaptureLLM(DeterministicLLM):
    def __init__(self):
        super().__init__()
        self.extractions = 0

    def complete_with_tools(self, prompt, tools, **kwargs):
        if any(tool.get("name") == "extract_entity" for tool in tools):
            self.extractions += 1
            name = "青禾发布" if self.extractions == 1 else "蓝溪发布"
            return [
                {
                    "tool": "extract_entity",
                    "input": {
                        "owner_scope": "user",
                        "routing_key": name,
                        "object_name": name,
                        "abstract": name + "由我负责。",
                        "overview": name + "由我负责。",
                        "content": name + "由我负责。",
                        "confidence": 0.99,
                    },
                }
            ]
        return []


def test_capture_continues_saved_backlog_and_rechecks_own_completed_receipt(runtime):
    memory, ctx = runtime.memory, runtime.ctx
    llm = CaptureLLM()
    memory._llm = llm
    original_id = new_id("original")
    original_messages = [
        {"id": original_id + "_user", "role": "user", "content": "我负责青禾发布。"}
    ]

    @capture_outbox_events
    def stage_original():
        with memory._session_mutation_scope(ctx):
            writer = memory.get_write_api()
            manager = memory.get_session_manager()
            manager.get_or_create(ctx.session_id, ctx)
            manager.enable_extraction_recovery(ctx.session_id, ctx, writer._fs)
            manager.add_message(
                ctx.session_id,
                "user",
                original_messages[0]["content"],
                ctx,
                message_id=original_messages[0]["id"],
            )
            buffer = manager.get_or_create(ctx.session_id, ctx)
            journal = ExtractionBatchJournal(
                manager,
                buffer,
                ctx,
                list(buffer.messages),
                archive_id=generate_archive_id(),
                messages=original_messages,
                session_time=datetime.now(UTC),
                session_summary="",
                tool_stats_text="",
                client_request_id=original_id,
                request_messages=original_messages,
            )
            staged = memory._chunk_and_extract(
                original_messages,
                writer,
                ctx,
                datetime.now(UTC),
                "",
                "",
                "staged",
                ctx.session_id,
                extraction_journal=journal,
            )
            return staged["plans"][0]["target_uri"]

    original_uri = stage_original()

    item = task(runtime, "memory_write", {"action": "remember", "new_text": "我负责蓝溪发布。"})
    operation_id = new_id("capture")
    frozen = {
        "messages": [
            {"id": item["task_id"] + "_user", "role": "user", "content": "我负责蓝溪发布。"}
        ]
    }
    result = runtime.gateway.capture(item, frozen["messages"], operation_id)
    assert result["memory_status"] == "stored"
    assert len(result["sources"]) == 1
    own_uri = result["sources"][0]["uri"]
    assert own_uri != original_uri
    assert "蓝溪发布" in result["sources"][0]["text"]
    assert result["receipt"]["client_request_id"] == operation_id
    assert llm.extractions == 2
    with memory._session_mutation_scope(ctx):
        buffer = memory.get_session_manager().get_or_create(ctx.session_id, ctx)
        assert read(buffer, ctx, original_id)["status"] == "completed"
        assert [plan["target_uri"] for plan in read(buffer, ctx, operation_id)["plans"]] == [
            own_uri
        ]
        assert buffer.messages == []
        assert buffer.extraction_active_count == 0

    fresh = MemoryService(memory._cfg)
    fresh._llm = CaptureLLM()
    gateway = MemoryGateway(fresh, runtime.gateway.context_resolver)
    try:
        restored = gateway.capture_recheck(item, operation_id, frozen)
        assert restored["receipt"]["status"] == "completed"
        assert [source["uri"] for source in restored["sources"]] == [own_uri]
        assert restored["memory_status"] == "stored"
    finally:
        fresh.shutdown()
        fresh._sql_pool.close_all()
