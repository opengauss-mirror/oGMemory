"""Run Agent extraction recovery and concurrent index publication against AGFS."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from multiprocessing.managers import BaseManager
from types import SimpleNamespace
from uuid import uuid4

from pyagfs import AGFSClient

from commit.outbox_store import OutboxStore
from core.models import RequestContext
from extraction.agent_memory import render_agent_memory
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from index.outbox_worker import OutboxWorker
from providers.token_tracker import TokenTracker
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.agent_memory import process_trace_memory, trace_memory_event
from tests.unit.retrieval.test_agent_units import unit_memory
from tests.unit.server.test_agent_memory import ExtractionLLM, SameEmbedding, finalized_trace


def _storage(url):
    client = AGFSClient(api_base_url=url)
    fs = AGFSContextFS(client)
    return client, fs, OutboxStore(client, fs)


def _context():
    return RequestContext("review-" + uuid4().hex, "u", "a", "s", "request")


def test_background_extraction_attributes_all_provider_usage(agfs_server_info):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    tracker, persisted = TokenTracker(), []
    tracker.set_usage_store(SimpleNamespace(record=persisted.append))

    class LLM(ExtractionLLM):
        def complete_with_tools(self, *args, **kwargs):
            tracker.record_llm(10, 5)
            return super().complete_with_tools(*args, **kwargs)

    class Embedding(SameEmbedding):
        def embed_texts(self, texts):
            tracker.record_embed(7)
            return super().embed_texts(texts)

    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    def handler(event):
        process_trace_memory(event, trace_store=store, outbox=outbox, fs=fs, llm=LLM(),
                             vector_index=InMemoryVectorIndex(), embedder=Embedding())
    worker = OutboxWorker(InMemoryVectorIndex(), Embedding(), fs=fs, agent_memory_handler=handler)
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(worker.run_once, outbox, [ctx.account_id]).result(timeout=20)
    assert result["succeeded"] == 1
    total = tracker.snapshot()
    scoped = tracker.scoped_snapshot(ctx.account_id, (ctx.agent_space_name(),))
    assert total.llm_calls == 6
    assert total.embed_calls == 1
    assert (scoped.input_tokens, scoped.output_tokens, scoped.embed_tokens) == (60, 30, 7)
    assert {key for event in persisted for key in event["allocations"]} == {
        (ctx.account_id, ctx.agent_space_name())}


def test_fresh_trace_restores_archived_tool_and_finishes_experience(agfs_server_info):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    process_trace_memory(event, trace_store=store,
                         outbox=outbox, fs=fs, llm=ExtractionLLM())
    tools = f"ctx://{ctx.account_id}/agents/a/memories/tool_units"
    uri = fs.list_children(tools, ctx)[0]
    fs.archive_node(uri, ctx)
    archived_version = fs.archived_node_version(uri, ctx)
    ctx = replace(ctx, session_id="new-evidence")
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    process_trace_memory(event, trace_store=store, outbox=outbox, fs=fs, llm=ExtractionLLM())
    restored = fs.read_node(uri, ctx)
    assert restored.metadata["version"] == archived_version + 1
    assert restored.metadata["provenance_ids"] == [turn["uri"]]
    experience = fs.read_node(f"ctx://{ctx.account_id}/agents/a/memories/experiences/batch_planning", ctx)
    assert experience.metadata["version"] == 2
    assert event.payload["committed_count"] == 2


class _IndexManager(BaseManager):
    pass


_IndexManager.register("index", InMemoryVectorIndex)


def _index_event(url, ctx, index, started, reached, resume, completed):
    _, fs, outbox = _storage(url)

    class Embedding(SameEmbedding):
        def embed_texts(self, texts):
            if reached is not None:
                reached.set()
                assert resume.wait(15)
            return super().embed_texts(texts)

    started.set()
    result = OutboxWorker(index, Embedding(), fs=fs).run_once(outbox, [ctx.account_id])
    assert result["succeeded"] == 1, result
    completed.set()


def test_concurrent_workers_publish_current_units(agfs_server_info):
    url = agfs_server_info["url"]
    _, fs, outbox = _storage(url)
    ctx = _context()
    _, template, _, _ = unit_memory.__wrapped__()
    node = replace(template, uri=f"ctx://{ctx.account_id}/agents/a/memories/tool_units/search",
                   owner_space=ctx.agent_space_name())
    fs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    mp = multiprocessing.get_context("spawn")
    started, reached, resume, old_done, new_done = [mp.Event() for _ in range(5)]
    with _IndexManager(ctx=mp) as manager:
        index = manager.index()
        old = mp.Process(target=_index_event, args=(url, ctx, index, started, reached, resume, old_done))
        new_started = mp.Event()
        new = mp.Process(target=_index_event, args=(url, ctx, index, new_started, None, None, new_done))
        old.start()
        try:
            assert reached.wait(10)
            current = deepcopy(node)
            current.metadata["structured_fields"]["units"] = current.metadata["structured_fields"]["units"][:1]
            current.metadata["agent_memory"]["units"]["cursor"]["withdrawn"] = True
            current.content = render_agent_memory("tool", current.metadata["structured_fields"])
            fs.write_node(current, ctx)
            outbox.register_write(current, ctx)
            new.start()
            assert new_started.wait(10)
            new_done.wait(1)
            resume.set()
            old.join(15)
            new.join(15)
            assert old.exitcode == new.exitcode == 0
            assert old_done.is_set() and new_done.is_set()
            hits = index.search_by_vector([1, 0], {"account_id": ctx.account_id, "level": 2}, 10)
            assert [hit.metadata["unit_key"] for hit in hits] == ["batch"]
            assert outbox.list_pending(ctx.account_id) == []
        finally:
            resume.set()
            for child in (old, new):
                if child.is_alive():
                    child.terminate()
                    child.join(5)
