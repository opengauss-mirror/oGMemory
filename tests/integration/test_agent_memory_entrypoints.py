"""Exercise shared extraction queues and HTTP unit reads through service entrypoints."""

from dataclasses import replace

import pytest

import server.app as app_module
from extraction.agent_memory import render_agent_memory
from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from index.outbox_worker import OutboxWorker
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.agent_memory import trace_memory_event
from server.mcp_trace import message_events
from server.memory_service import MemoryService
from service.index_service import IndexService
from tests.integration.test_agent_memory_hybrid import (
    _read_api,
)
from tests.integration.test_agent_memory_hybrid import (
    sql_database as sql_database,
)
from tests.integration.test_agent_memory_hybrid import (
    sql_storage as sql_storage,
)
from tests.integration.test_agent_memory_review import _context, _storage
from tests.unit.retrieval.test_agent_units import unit_memory
from tests.unit.server.test_agent_memory import MESSAGES, ExtractionLLM, SameEmbedding
from tests.unit.server.test_mcp_tools_readonly import _FakeService


@pytest.fixture(params=["agfs", "sql"])
def storage(request):
    if request.param == "sql":
        fs, outbox = request.getfixturevalue("sql_storage")
        trace_store = SQLTraceEventStore(pool=fs._shared_pool)
    else:
        client, fs, outbox = _storage(request.getfixturevalue("agfs_server_info")["url"])
        trace_store = AGFSTraceEventStore(client=client)
    return request.param, fs, outbox, trace_store


def _queued_trace(store, outbox, ctx):
    events = message_events(MESSAGES) + [{"event_type": "turn_end", "status": "ok"}]
    for i, event in enumerate(events):
        saved = store.insert(
            event_id=str(i), account_id=ctx.account_id, owner_space=ctx.user_space_name(),
            agent_id=ctx.agent_id, session_id=ctx.session_id,
            hook_type=event["event_type"], content=event,
        )
    turn = store.finalize_session(
        account_id=ctx.account_id, owner_space=ctx.user_space_name(), agent_id=ctx.agent_id,
        session_id=ctx.session_id, stop_event_index=saved["event_index"],
    )
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    return event


def _tool_node(ctx):
    _, node, _, _ = unit_memory.__wrapped__()
    return replace(node, uri=f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tool_units/search",
                   owner_space=ctx.agent_space_name())


def test_index_service_preserves_extraction_for_memory_service(storage):
    backend, fs, outbox, store = storage
    ctx = _context()
    event = _queued_trace(store, outbox, ctx)
    queued = next(job for _, job in outbox.list_pending(ctx.account_id)
                  if job.event_id == event.event_id)
    node = _tool_node(ctx)
    fs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    index = InMemoryVectorIndex()
    service = IndexService(outbox, SameEmbedding(), index, lambda: [ctx.account_id],
                           interval_seconds=3600, worker_count=1, fs=fs, llm=ExtractionLLM())
    service.start()
    try:
        for thread in service._threads:
            thread.join(5)
        for _ in range(3):
            stats = service.schedulers[0]._run_once_wrapped()
        pending = next(job for _, job in outbox.list_pending(ctx.account_id)
                       if job.event_id == event.event_id)
        assert (pending.status, pending.retry_count, pending.payload) == (
            "PENDING", 0, queued.payload), stats
        assert index.count({"account_id": ctx.account_id}) == 4

        cfg = {"storage_backend": backend, "index_workers": 0}
        if backend == "sql":
            cfg["sql_connection_string"] = outbox._connection_string
        memory = MemoryService(config=OgMemConfig(**cfg))
        memory._get_context_fs = lambda: fs
        memory._get_outbox_store = lambda fs: outbox
        memory._get_trace_event_store = lambda: store
        memory._get_shared_vector_index = lambda: index
        memory._get_shared_projection_index = lambda: index
        memory._get_shared_embedder = lambda: SameEmbedding()
        memory.get_llm = lambda: ExtractionLLM()
        try:
            assert memory.drain_outbox_sync(ctx.account_id)["succeeded"] == 1
        finally:
            memory.shutdown()
        assert service.schedulers[0]._run_once_wrapped()["succeeded"] == 2
        hits = index.search_by_vector([1, 0], {"account_id": ctx.account_id, "level": 2}, 10)
        assert {hit.category for hit in hits} == {"tool", "experience"}
        assert len(hits) == 4
        assert outbox.list_pending(ctx.account_id) == []
    finally:
        service.stop()


def test_http_query_reads_selected_unit_with_small_budget(storage, monkeypatch):
    _, fs, outbox, _ = storage
    ctx = _context()
    node = _tool_node(ctx)
    fields = node.metadata["structured_fields"]
    fields["units"][0]["content"] = "Batch settings. " * 500
    node = replace(node, content=render_agent_memory("tool", fields))
    fs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    index = InMemoryVectorIndex()
    assert OutboxWorker(index, SameEmbedding(), fs=fs).run_once(outbox, [ctx.account_id])["succeeded"] == 1
    service = _FakeService()
    service.fs = fs
    service.read_api = _read_api(index, fs)
    service.build_context = lambda params, identity=None: ctx
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    query = client.post("/api/v1/memories/query", json={"query": "pagination", "top_k": 10})
    assert query.status_code == 200, query.get_json()
    hit = next(hit for hit in query.get_json()["hits"] if hit.get("unit_key") == "cursor")
    read = client.post("/api/v1/memories/read", json={
        "uri": hit["uri"], "unit_key": hit["unit_key"], "level": "content", "max_chars": 200,
    })
    assert read.status_code == 200, read.get_json()
    result = read.get_json()
    assert result["content"] == "search\n\nSearch pagination\n\nReuse the returned cursor for the next page."
    assert (result["unit_key"], result["unit_revision"]) == ("cursor", 2)
    assert result["grounding"] == fields["units"][1]["grounding"]


def test_sql_claim_selects_supported_events_before_limit(sql_storage):
    fs, outbox = sql_storage
    ctx = _context()
    extraction = trace_memory_event(f"ctx://{ctx.account_id}/traces/s/turns/one", ctx)
    outbox.register_once(extraction, ctx)
    node = _tool_node(ctx)
    fs.write_node(node, ctx)
    writes = [outbox.register_write(node, ctx) for _ in range(2)]
    for event in writes:
        claimed = outbox.claim_batch("index", limit=1, exclude_event_types=["EXTRACT_AGENT_MEMORY"])
        assert [job.event_id for _, job in claimed] == [event.event_id]
        outbox.mark_done(claimed[0][1], node.uri)
    claimed = outbox.claim_batch("memory", limit=1)
    assert [job.event_id for _, job in claimed] == [extraction.event_id]
    assert claimed[0][1].retry_count == 0
