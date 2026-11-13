"""Archive and summary preparation overlap with owner-thread checkpoints."""

import threading
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from core.models import RequestContext
from server.memory_service import MemoryService
from service.extraction_checkpoint import ExtractionBatchJournal
from core.model_stage import run_model_call
from session.session_manager import SessionManager
from session.models import ArchiveEntry
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import setup_service, params


class StageModel:
    model = "archive-stage-test"

    def __init__(self):
        self.calls = []

    def complete_json(self, prompt, schema):
        self.calls.append((prompt, threading.get_ident()))
        if "overview" in schema["properties"]:
            return {"overview": "## Main Topics\n- Receiver repair.", "abstract": "Receiver repaired."}
        return {"facts_and_details": "Nora repaired the receiver."}


def stage_setup():
    ctx = RequestContext("stage-test", "reader", "agent", "stage-session", "trace")
    fs = MemoryFS()
    model = StageModel()
    manager = SessionManager(get_context_fs=lambda: fs, get_llm=lambda: model)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    source = buffer.add("user", "Nora repaired the receiver.", message_id="source-one")
    messages = [asdict(source)]
    journal = ExtractionBatchJournal(manager, buffer, ctx, [source], archive_id="staged-archive",
        messages=messages, session_time=None, session_summary="", tool_stats_text="")
    return ctx, fs, model, manager, source, journal


@pytest.mark.parametrize("background", [False, True])
def test_after_turn_overlaps_archive_summary_and_main_with_owner_writes(background):
    ctx = RequestContext("overlap-test", "reader", "agent", "overlap-session", "trace")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    del service._generate_summary_candidate
    manager.commit_snapshot = SessionManager.commit_snapshot.__get__(manager)
    manager._get_archive_store = lambda: None
    manager._get_agfs = lambda: None
    started = {name: threading.Event() for name in ("archive", "summary", "main")}
    observations, saves, archives, frozen = [], [], [], []
    model = StageModel()
    complete = model.complete_json

    def compute(prompt, schema):
        stage = "archive" if "overview" in schema["properties"] else "summary"
        observations.append((stage, threading.get_ident()))
        started[stage].set()
        assert started["main"].wait(5)
        return complete(prompt, schema)

    model.complete_json = compute
    service.get_llm = lambda: model
    manager._get_llm = lambda: model
    original_extract = writer._pipeline.extract

    def extract(*args, **kwargs):
        observations.append(("main", threading.get_ident()))
        started["main"].set()
        assert started["archive"].wait(5)
        assert started["summary"].wait(5)
        return original_extract(*args, **kwargs)

    writer._pipeline.extract = extract
    original_inputs = manager._archive_compression_inputs

    def inputs(messages, *args):
        frozen.append(deepcopy(messages))
        return original_inputs(messages, *args)

    manager._archive_compression_inputs = inputs
    original_write = fs.write_node

    def write(node, request):
        saves.append(threading.get_ident())
        return original_write(node, request)

    fs.write_node = write

    def archive(session_id, archive_id, overview, abstract, messages, request, metadata=None):
        archives.append((overview, abstract, deepcopy(messages), threading.get_ident()))
        return {"success": True, "uri": "memory://archive", "archive_id": archive_id}

    manager._write_archive = archive
    request = params(ctx, request="" if background else "parallel")
    request["wait"] = not background
    result = service.after_turn(request)
    if background:
        deadline = monotonic() + 5
        while manager.get_or_create(ctx.session_id).extraction_active_count and monotonic() < deadline:
            sleep(.01)
        assert result["status"] == "processing"
    else:
        assert result["status"] == "completed"
    stages = dict(observations)
    owner = stages["main"]
    assert stages["archive"] != owner
    assert stages["summary"] != owner
    assert archives == [("## Main Topics\n- Receiver repair.", "Receiver repaired.", frozen[0], owner)]
    assert frozen[0] == frozen[-1]
    assert len(model.calls) == 2
    assert manager.get_or_create(ctx.session_id).extraction_active_count == 0
    assert all(ident == owner for ident in saves[-3:])
    assert len([node for node in fs.nodes.values() if node.category == "session_summary"]) == 1


def test_prepared_archive_restores_and_owner_persists_completed_stage():
    ctx, fs, model, manager, source, journal = stage_setup()
    owner = threading.get_ident()
    writes = []
    original_write = fs.write_node

    def save(node, request):
        writes.append((node.category, threading.get_ident()))
        return original_write(node, request)

    fs.write_node = save
    with manager.parallel_archive_compression([source], ctx.session_id, ctx, journal) as prepared:
        run_model_call(lambda: sleep(.15))
        assert journal.load_stage("history_archive", prepared["inputs"]) == {
            "overview": "## Main Topics\n- Receiver repair.", "abstract": "Receiver repaired."}
    restored = SessionManager(get_context_fs=lambda: fs, get_llm=lambda: model)
    restored.get_or_create(ctx.session_id, ctx)
    restored.enable_extraction_recovery(ctx.session_id, ctx, fs)
    recovered = ExtractionBatchJournal.resume(restored, restored.get_or_create(ctx.session_id, ctx), ctx)
    with restored.parallel_archive_compression([source], ctx.session_id, ctx, recovered) as replay:
        assert replay["result"] == prepared["result"]
    assert len(model.calls) == 1
    assert all(ident == owner for _, ident in writes)
    assert any(category == "stage_result" for category, _ in writes)


def test_archive_dependency_refresh_recomputes_its_summary():
    ctx, fs, model, manager, source, journal = stage_setup()
    previous = ["Original overview", "Original abstract"]
    manager._get_latest_archive_context = lambda *args: tuple(previous)
    with manager.parallel_archive_compression([source], ctx.session_id, ctx, journal) as first:
        pass
    previous[:] = ["Updated overview", "Updated abstract"]
    result = manager._resolve_archive_compression([asdict(source)], ctx.session_id, ctx, first, journal)
    refreshed = manager._archive_compression_inputs(
        [asdict(source)], ctx.session_id, ctx, model, journal.batch["archive_id"])
    assert journal.load_stage("history_archive", refreshed) == {"overview": result[0], "abstract": result[1]}
    assert len(model.calls) == 2
    assert "Previous overview:\nUpdated overview" in model.calls[-1][0]
    assert "Previous abstract: Updated abstract" in model.calls[-1][0]
    with manager.parallel_archive_compression([source], ctx.session_id, ctx, journal) as replay:
        assert replay["result"] == {"overview": result[0], "abstract": result[1]}
    assert len(model.calls) == 2


def test_completed_summary_stage_restores_before_acceptance():
    ctx, fs, model, manager, source, journal = stage_setup()
    service = MemoryService.__new__(MemoryService)
    service.get_llm = lambda: model
    api = SimpleNamespace(_write_checkpointed_candidates=lambda *args: [SimpleNamespace(action="create")])
    messages, session_time = [asdict(source)], datetime(2026, 10, 27)
    with service._parallel_session_summary(messages, session_time, ctx, ctx.session_id,
                                            api, journal.checkpoint("summary")):
        run_model_call(lambda: sleep(.15))
        snapshot = deepcopy(fs.nodes)
    recovered_fs = MemoryFS()
    recovered_fs.nodes = snapshot
    restored = SessionManager(get_context_fs=lambda: recovered_fs)
    restored.get_or_create(ctx.session_id, ctx)
    restored.enable_extraction_recovery(ctx.session_id, ctx, recovered_fs)
    recovered = ExtractionBatchJournal.resume(restored, restored.get_or_create(ctx.session_id, ctx), ctx)
    with service._parallel_session_summary(messages, session_time, ctx, ctx.session_id,
                                            api, recovered.checkpoint("summary")):
        pass
    accepted, extracted, filtered = recovered.checkpoint("summary").load()
    assert (extracted, filtered) == (1, 0)
    assert accepted[0].content == "### Facts\nNora repaired the receiver."
    assert len(model.calls) == 1


def test_written_archive_reuses_its_prepared_predecessor_on_resume():
    ctx, fs, model, manager, source, journal = stage_setup()
    entries = [ArchiveEntry("predecessor", ctx.session_id, "Previous overview", "Previous abstract", [],
                            created_at="2026-10-26T00:00:00+00:00")]
    manager._get_archive_store = lambda: SimpleNamespace(list_archives=lambda *args: list(entries))
    with manager.parallel_archive_compression([source], ctx.session_id, ctx, journal) as prepared:
        pass
    entries.append(ArchiveEntry(journal.batch["archive_id"], ctx.session_id,
        prepared["result"]["overview"], prepared["result"]["abstract"], [asdict(source)],
        created_at="2026-10-27T00:00:00+00:00"))
    resumed = ExtractionBatchJournal.resume(manager, manager.get_or_create(ctx.session_id, ctx), ctx)
    with manager.parallel_archive_compression([source], ctx.session_id, ctx, resumed) as restored:
        assert restored["result"] == prepared["result"]
        assert restored["inputs"]["prev_overview"] == "Previous overview"
        assert restored["inputs"]["prev_abstract"] == "Previous abstract"
    assert len(model.calls) == 1
