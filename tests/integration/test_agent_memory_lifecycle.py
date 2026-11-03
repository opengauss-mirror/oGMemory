"""Exercise production finalization and cross-process Outbox registration on AGFS."""

import multiprocessing
import time
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from commit.outbox_store import OutboxStore
from core.models import RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from server.agent_memory import trace_memory_event


def _finish_registration(url, ctx, reached, resume):
    client = AGFSClient(api_base_url=url)
    outbox = OutboxStore(client, AGFSContextFS(client))
    event = trace_memory_event(f"ctx://{ctx.account_id}/traces/s/turns/t", ctx)
    original = client.read

    def read(path, *args, **kwargs):
        if path.endswith(".agent_memory_done") and not reached.is_set():
            try:
                return original(path, *args, **kwargs)
            except Exception:
                reached.set()
                assert resume.wait(10)
                raise
        return original(path, *args, **kwargs)

    client.read = read
    outbox.register_once(event, ctx)


def _complete_event(url, ctx, completed):
    client = AGFSClient(api_base_url=url)
    outbox = OutboxStore(client, AGFSContextFS(client))
    event = trace_memory_event(f"ctx://{ctx.account_id}/traces/s/turns/t", ctx)
    outbox.mark_done(event, event.uri)
    completed.set()


def test_completed_event_serializes_with_registration_across_processes(agfs_server_info):
    url = agfs_server_info["url"]
    ctx = RequestContext("race-" + uuid4().hex, "u", "a", "s", "request")
    client = AGFSClient(api_base_url=url)
    outbox = OutboxStore(client, AGFSContextFS(client))
    event = trace_memory_event(f"ctx://{ctx.account_id}/traces/s/turns/t", ctx)
    outbox.register_once(event, ctx)
    mp = multiprocessing.get_context("spawn")
    reached, resume, completed = mp.Event(), mp.Event(), mp.Event()
    registration = mp.Process(target=_finish_registration, args=(url, ctx, reached, resume))
    completion = mp.Process(target=_complete_event, args=(url, ctx, completed))
    registration.start()
    try:
        assert reached.wait(10)
        completion.start()
        completed.wait(2)
        resume.set()
        registration.join(10)
        completion.join(10)
        assert registration.exitcode == completion.exitcode == 0
        assert completed.is_set()
        assert outbox.list_pending(ctx.account_id) == []
    finally:
        resume.set()
        for child in (registration, completion):
            if child.is_alive():
                child.terminate()
                child.join(5)


def _run_service(url, workers):
    from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
    from providers.unified_config import OgMemConfig
    from providers.vector_index.in_memory_index import InMemoryVectorIndex
    from server.mcp_trace import message_events
    from server.memory_service import MemoryService
    from tests.unit.server.test_agent_memory import MESSAGES, ExtractionLLM, SameEmbedding

    client = AGFSClient(api_base_url=url)
    fs = AGFSContextFS(client)
    outbox = OutboxStore(client, fs)
    store = AGFSTraceEventStore(client=client)
    index = InMemoryVectorIndex()
    ctx = RequestContext("lifecycle-" + uuid4().hex, "u", "a", "s", "request")
    service = MemoryService(config=OgMemConfig(storage_backend="agfs", index_workers=workers, agent_memory_enabled=True))
    service._default_account_id = ctx.account_id
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda fs: outbox
    service._get_trace_event_store = lambda: store
    service._get_shared_projection_index = lambda: index
    service._get_shared_vector_index = lambda: index
    service._get_shared_embedder = lambda: SameEmbedding()
    service.get_llm = lambda: ExtractionLLM()
    events = message_events(MESSAGES) + [{"event_type": "turn_end", "status": "ok"}]
    for i, event in enumerate(events):
        saved = service.add_session_trace(
            {
                "_ctx": ctx,
                "sessionId": "s",
                "eventId": str(i),
                "hookType": event["event_type"],
                "event": event,
            }
        )
        assert saved["ok"]
    done = service.finalize_session_trace(
        {"_ctx": ctx, "sessionId": "s", "stopEventIndex": saved["event_index"]}
    )
    assert done["ok"]
    filters = {"account_id": ctx.account_id, "owner_space": ctx.agent_space_name()}
    deadline = time.monotonic() + 15
    while index.count(filters) < 4 and time.monotonic() < deadline:
        if workers == 0:
            assert service.drain_outbox_sync(ctx.account_id)["failed"] == 0
        time.sleep(0.05)
    hits = index.search_by_vector([1.0, 0.0], filters, 10)
    assert len(hits) == 4
    assert {hit.category for hit in hits} == {"tool", "experience"}
    for uri in {hit.uri.removesuffix("/content.md") for hit in hits}:
        node = fs.read_node(uri, ctx)
        assert node.metadata["version"] == 1
        assert node.metadata["provenance_ids"] == [done["uri"]]
    replay = service.finalize_session_trace(
        {"_ctx": ctx, "sessionId": "s", "stopEventIndex": saved["event_index"]}
    )
    assert replay["agent_memory_event_id"] == done["agent_memory_event_id"]


@pytest.mark.parametrize("workers", [0, 1])
def test_service_finalizes_extracts_and_indexes_on_agfs(agfs_server_info, workers):
    mp = multiprocessing.get_context("spawn")
    child = mp.Process(target=_run_service, args=(agfs_server_info["url"], workers))
    child.start()
    try:
        child.join(25)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
