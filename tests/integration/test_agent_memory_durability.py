"""Exercise extraction checkpoints, maintenance and ownership on live AGFS."""

import json
import multiprocessing
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from commit.context_writer import ContextWriter
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from index.outbox_worker import OutboxWorker
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.agent_memory import process_trace_memory, trace_memory_event
from tests.integration.test_agent_memory_review import _context, _storage
from tests.unit.server.test_agent_memory import ExtractionLLM, SameEmbedding, finalized_trace


def _process(event, store, fs, outbox, llm):
    process_trace_memory(event, trace_store=store, fs=fs, outbox=outbox, llm=llm)


def test_receipt_retry_publishes_index_before_checkpoint(agfs_server_info, monkeypatch):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    llm, index = ExtractionLLM(), InMemoryVectorIndex()
    worker = OutboxWorker(index, SameEmbedding(), fs=fs,
                          agent_memory_handler=lambda job: _process(job, store, fs, outbox, llm))
    register = outbox.register_write

    def interrupted(node, ctx):
        if node.category == "tool":
            raise OSError("index event publication interrupted")
        return register(node, ctx)

    with monkeypatch.context() as patch:
        patch.setattr(outbox, "register_write", interrupted)
        first = worker.process_event(event)
    saved = next(job for _, job in outbox.list_pending(ctx.account_id) if job.event_id == event.event_id)
    before = saved.payload["committed_count"]
    assert worker.process_event(saved).success
    outbox.mark_done(saved, saved.uri)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    hits = index.search_by_vector([1, 0], {"account_id": ctx.account_id, "level": 2}, 10)
    assert {hit.category for hit in hits} == {"tool", "experience"}
    assert before == 0, (first, before)
    assert saved.payload["committed_count"] == 2
    assert {fs.read_node(hit.uri.removesuffix("/content.md"), ctx).metadata["version"] for hit in hits} == {1}
    assert len([call for call in llm.calls if call[0] == "maintain_agent_memory"]) == 2


class TwoExperienceLLM(ExtractionLLM):
    def complete_with_tools(self, prompt, tools, tool_choice):
        result = super().complete_with_tools(prompt, tools, tool_choice)
        if tools[0]["name"] == "maintain_agent_memory" and prompt.startswith("Extract only experience;"):
            template = result[0]["input"]["changes"][0]
            result[0]["input"]["changes"] = [dict(deepcopy(template), routing_key=key)
                                                 for key in ("batch_planning", "response_planning")]
        return result


def test_one_repair_supplies_all_existing_experience_targets(agfs_server_info):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    for session in ("initial", "fresh"):
        ctx = replace(ctx, session_id=session)
        store, turn = finalized_trace(client, ctx)
        event = trace_memory_event(turn["uri"], ctx)
        outbox.register_once(event, ctx)
        llm = TwoExperienceLLM()
        _process(event, store, fs, outbox, llm)
    repairs = [prompt for name, prompt in llm.calls
               if name == "maintain_agent_memory" and prompt.startswith("Extract only experience;")]
    assert len(repairs) == 2
    snapshots = json.loads(repairs[1].split("Current maintenance state:\n")[1].split("\nPrevious proposal")[0])
    assert {node["routing_key"] for node in snapshots} == {"batch_planning", "response_planning"}
    assert event.payload["committed_count"] == 3
    for key in ("batch_planning", "response_planning"):
        node = fs.read_node(f"ctx://{ctx.account_id}/agents/a/memories/experiences/{key}", ctx)
        assert node.metadata["version"] == 2


class RenamedExperienceLLM(ExtractionLLM):
    def complete_with_tools(self, prompt, tools, tool_choice):
        result = super().complete_with_tools(prompt, tools, tool_choice)
        if tools[0]["name"] == "maintain_agent_memory" and prompt.startswith("Extract only experience;"):
            result[0]["input"]["changes"][0]["key"] = "batch_strategy"
        return result


def test_concurrent_experience_key_reextracts_and_finishes(agfs_server_info, monkeypatch):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    original = ContextWriter.write_candidate
    concurrent = False

    def write(writer, candidate, request):
        nonlocal concurrent
        if candidate.category == "experience" and not concurrent:
            concurrent = True
            other_ctx = replace(ctx, session_id="concurrent")
            other_store, other_turn = finalized_trace(client, other_ctx)
            other = trace_memory_event(other_turn["uri"], other_ctx)
            outbox.register_once(other, other_ctx)
            _process(other, other_store, fs, outbox, RenamedExperienceLLM())
        return original(writer, candidate, request)

    worker = OutboxWorker(InMemoryVectorIndex(), SameEmbedding(), fs=fs,
                          agent_memory_handler=lambda job: _process(job, store, fs, outbox, ExtractionLLM()))
    with monkeypatch.context() as patch:
        patch.setattr(ContextWriter, "write_candidate", write)
        first = worker.process_event(event)
    saved = next(job for _, job in outbox.list_pending(ctx.account_id) if job.event_id == event.event_id)
    assert saved.payload.get("reextract") is True, first
    assert saved.payload["committed_count"] == 1
    _process(saved, store, fs, outbox, RenamedExperienceLLM())
    assert saved.payload["committed_count"] == 2
    node = fs.read_node(f"ctx://{ctx.account_id}/agents/a/memories/experiences/batch_planning", ctx)
    assert node.metadata["structured_fields"]["key"] == "batch_strategy"
    assert node.metadata["version"] == 2
    assert len(node.metadata["provenance_ids"]) == 2


def _extract_worker(url, ctx, reached, resume, results, worker_id):
    client, fs, outbox = _storage(url)
    store = AGFSTraceEventStore(client=client)

    class PausedLLM(ExtractionLLM):
        def complete_with_tools(self, *args, **kwargs):
            if reached is not None and not reached.is_set():
                reached.set()
                assert resume.wait(20)
            return super().complete_with_tools(*args, **kwargs)

    worker = OutboxWorker(InMemoryVectorIndex(), SameEmbedding(), fs=fs,
                          agent_memory_handler=lambda job: _process(job, store, fs, outbox, PausedLLM()))
    results.put((worker_id, worker.run_once(outbox, [ctx.account_id], worker_id=worker_id)))


def test_live_extraction_keeps_ownership_after_claim_expiry(agfs_server_info):
    url = agfs_server_info["url"]
    client, fs, outbox = _storage(url)
    ctx = _context()
    _, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    mp = multiprocessing.get_context("spawn")
    reached, resume, results = mp.Event(), mp.Event(), mp.Queue()
    first = mp.Process(target=_extract_worker, args=(url, ctx, reached, resume, results, "first"))
    second = mp.Process(target=_extract_worker, args=(url, ctx, None, None, results, "second"))
    first.start()
    try:
        assert reached.wait(10)
        marker = outbox._uri_to_agfs_path(event.uri) + ".outbox/" + event.event_id + ".processing"
        stale = json.loads(client.read(marker))
        stale["acquired_at"] = (datetime.now(UTC) - timedelta(seconds=301)).isoformat()
        client.write(marker, json.dumps(stale).encode())
        second.start()
        second.join(10)
        resume.set()
        first.join(20)
        assert first.exitcode == second.exitcode == 0
        stats = dict(results.get(timeout=2) for _ in range(2))
        assert stats["second"]["skipped"] == 1, stats
        assert stats["first"]["succeeded"] == 1, stats
        assert outbox._agent_event_done(event)
        node = fs.read_node(f"ctx://{ctx.account_id}/agents/a/memories/experiences/batch_planning", ctx)
        assert node.metadata["version"] == 1
    finally:
        resume.set()
        for child in (first, second):
            if child.is_alive():
                child.terminate()
                child.join(5)
        results.close()
