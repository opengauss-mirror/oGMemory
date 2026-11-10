"""Summary/model overlap with owner-thread durable writes."""

import threading
from contextvars import ContextVar
from datetime import datetime
from types import SimpleNamespace

import pytest

from core.operation_deadline import operation_deadline, remaining_seconds
from extraction.summary_generator import SessionSummaryGenerator
from server.memory_service import MemoryService
from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_manager import SessionManager
from tests.unit.extraction.test_node_processing import card, operation, setup


@pytest.mark.parametrize("checkpointed", [False, True])
def test_summary_overlaps_node_model_and_writes_on_owner(checkpointed):
    summary_started, node_started = threading.Event(), threading.Event()
    context = ContextVar("summary-test-context", default="")
    owner = threading.get_ident()
    observations = []

    def decide(kwargs):
        assert summary_started.wait(3)
        observations.append(("node", threading.get_ident(), context.get(), remaining_seconds()))
        node_started.set()
        return [operation([0])]

    ctx, fs, llm, api = setup(decide)

    def summarize(prompt, schema):
        observations.append(("summary", threading.get_ident(), context.get(), remaining_seconds()))
        summary_started.set()
        assert node_started.wait(3)
        return {"facts_and_details": "Caroline's grandmother gave her a necklace."}

    llm.complete_json = summarize
    frozen_summaries = []

    def extract(*args, **kwargs):
        frozen_summaries.append(kwargs["session_summary"])
        assert summary_started.wait(3)
        return [card()]

    api._pipeline.extract = extract
    writes = []
    write_node = fs.write_node

    def save(node, request):
        writes.append((node.category, threading.get_ident()))
        return write_node(node, request)

    fs.write_node = save
    service = MemoryService.__new__(MemoryService)
    service.get_llm = lambda: llm
    service._get_chunker = lambda: None
    messages = [{"role": "user", "content": "Caroline's grandmother gave her a necklace.", "id": "source-1"}]
    session_time = datetime(2026, 10, 27)
    journal = None
    if checkpointed:
        manager = SessionManager(get_context_fs=lambda: fs)
        buffer = manager.get_or_create(ctx.session_id, ctx)
        source = buffer.add("user", messages[0]["content"], message_id="source-1")
        journal = ExtractionBatchJournal(
            manager, buffer, ctx, [source], archive_id="parallel-archive", messages=messages,
            session_time=session_time, session_summary="frozen previous summary", tool_stats_text="",
        )
    token = context.set("inherited")
    try:
        with operation_deadline(30):
            result = service._chunk_and_extract(
                messages, api, ctx, session_time, "frozen previous summary", "", "run", ctx.session_id,
                archive_id="parallel-archive", extraction_journal=journal,
            )
    finally:
        context.reset(token)

    assert result["writes_completed"] == 1
    assert frozen_summaries == ["frozen previous summary"]
    stages = {stage: (ident, value, budget) for stage, ident, value, budget in observations}
    assert stages["node"][0] != owner
    assert stages["summary"][0] != owner
    assert all(value == "inherited" and 0 < budget <= 30 for _, value, budget in stages.values())
    categories = [category for category, ident in writes]
    assert categories.index("session_summary") < categories.index("entity")
    assert all(ident == owner for _, ident in writes)
    assert sum(node.category == "session_summary" for node in fs.nodes.values()) == 1
    if checkpointed:
        assert journal.checkpoint("summary").load()[1:] == (1, 0)
        assert journal.checkpoint("chunk:0").load()[1:] == (1, 0)


@pytest.mark.parametrize("restored", [False, True])
def test_empty_candidate_batch_joins_summary_before_completion(restored):
    ctx, fs, llm, api = setup(lambda _: [])
    llm.complete_json = lambda prompt, schema: {"facts_and_details": "Caroline owns a necklace."}
    api._pipeline.extract = lambda *args, **kwargs: []
    saved = SimpleNamespace(journal=SimpleNamespace(batch={"archive_id": "existing"}), load=lambda: ([], 0, 0))
    service = MemoryService.__new__(MemoryService)
    service.get_llm = lambda: llm
    messages = [{"role": "user", "content": "Caroline owns a necklace."}]
    calls = []
    with service._parallel_session_summary(messages, None, ctx, ctx.session_id, api) as join_summary:
        def before_write():
            join_summary()
            calls.append("ready")

        result = api.commit_session(messages, ctx, candidate_checkpoint=saved if restored else None,
                                    before_write=before_write)
        assert sum(node.category == "session_summary" for node in fs.nodes.values()) == 1
    assert calls == ["ready"]
    assert result["writes_completed"] == 0


@pytest.mark.parametrize("summary_content", ["Caroline owns a necklace.", ""])
def test_accepted_summary_replays_on_owner_with_accepted_chunk(summary_content):
    ctx, fs, llm, api = setup(lambda _: [])
    manager = SessionManager(get_context_fs=lambda: fs)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    message = buffer.add("user", "Caroline owns a necklace.", message_id="source")
    messages = [{"role": "user", "content": message.content}]
    journal = ExtractionBatchJournal(manager, buffer, ctx, [message], archive_id="resume",
                                     messages=messages, session_time=None, session_summary="old", tool_stats_text="")
    summary = SessionSummaryGenerator.build_candidate({}, summary_content, session_id=ctx.session_id)
    checkpoint = journal.checkpoint("summary")
    candidates = [summary] if summary_content else []
    checkpoint.accept(candidates, len(candidates), 0)
    journal.checkpoint("chunk:0").accept([], 0, 0)
    service = MemoryService.__new__(MemoryService)
    service._get_chunker = lambda: None
    result = service._chunk_and_extract_recoverable(api, ctx, ctx.session_id, journal)
    assert result["writes_completed"] == 0
    if candidates:
        assert checkpoint.completed(0)["action"] == "create"
    assert sum(node.category == "session_summary" for node in fs.nodes.values()) == len(candidates)
