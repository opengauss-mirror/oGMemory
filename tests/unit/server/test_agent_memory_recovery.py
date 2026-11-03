from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event
from unittest.mock import patch

from commit.archive_builder import ArchiveBuilder
from commit.outbox_store import OutboxStore
from core.models import CandidateMemory, SeedHit, SeedResult, TypedQuery, WritePlan
from core.uri_resolver import URIResolver
from extraction.agent_memory import AgentMemoryExtractor, render_agent_memory
from extraction.schemas.registry import SchemaRegistry
from index.outbox_worker import OutboxWorker
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.agent_memory import process_trace_memory, trace_memory_event
from tests.unit.retrieval.test_agent_units import unit_memory
from tests.unit.server.test_agent_memory import (
    CTX,
    MESSAGES,
    ExtractionLLM,
    NodeFS,
    SameEmbedding,
    _MemoryAGFS,
    finalized_trace,
)


def setup():
    client = _MemoryAGFS()
    fs = NodeFS(client)
    outbox = OutboxStore(client, fs)
    return client, fs, outbox, ExtractionLLM()


def process(fs, outbox, llm, store, event):
    return process_trace_memory(event, trace_store=store, outbox=outbox, fs=fs, llm=llm)


def test_service_finalization_writes_memory():
    from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
    from providers.unified_config import OgMemConfig
    from server.mcp_trace import message_events
    from server.memory_service import MemoryService

    client, fs, outbox, llm = setup()
    svc = MemoryService(config=OgMemConfig(storage_backend="agfs", index_workers=0, agent_memory_enabled=True))
    svc._get_context_fs = lambda: fs
    svc._get_outbox_store = lambda fs: outbox
    svc._get_trace_event_store = lambda: store
    svc.get_llm = lambda: llm
    svc._get_shared_projection_index = lambda: index
    svc._get_shared_vector_index = lambda: index
    svc._get_shared_embedder = lambda: SameEmbedding()
    store, index = AGFSTraceEventStore(client=client, mount_prefix="/local"), InMemoryVectorIndex()
    for i, event in enumerate(
        message_events(MESSAGES) + [{"event_type": "turn_end", "status": "ok"}]
    ):
        result = svc.add_session_trace(
            {
                "_ctx": CTX,
                "sessionId": CTX.session_id,
                "eventId": str(i),
                "event": event,
                "hookType": event["event_type"],
            }
        )
        assert result["ok"]
    done = svc.finalize_session_trace(
        {"_ctx": CTX, "sessionId": CTX.session_id, "stopEventIndex": result["event_index"]}
    )
    assert done["ok"]
    for _ in range(3):
        assert svc.drain_outbox_sync("acct")["failed"] == 0
    assert {n.category for n in fs.nodes.values()} == {"tool", "experience"}


def test_legacy_tool_and_typed_tool_coexist():
    client, fs, outbox, llm = setup()
    candidate = CandidateMemory(
        category="tool",
        owner_scope="agent",
        routing_key="read_file",
        abstract="Read files",
        overview="Read local documents",
        content="Read local text files.",
        confidence=1.0,
    )
    uri = URIResolver(SchemaRegistry()).resolve("tool", {"routing_key": "read_file"}, CTX)
    node = ArchiveBuilder(llm).build(candidate, WritePlan(action="create", target_uri=uri), CTX)
    fs.write_node(node, CTX)
    original = fs.read_node(uri, CTX)
    store, turn = finalized_trace(client)
    process(fs, outbox, llm, store, trace_memory_event(turn["uri"], CTX))
    assert len(fs.nodes) == 3
    assert fs.read_node(uri, CTX) == original


def test_distinct_tool_names_survive_persistence():
    client, fs, outbox, llm = setup()
    for i, name in enumerate(("Read", "read", "api.search", "api-search")):
        ctx = replace(CTX, session_id="s" + str(i))
        messages = deepcopy(MESSAGES)
        messages[1]["tool_calls"][0]["function"]["name"] = name
        with patch("tests.unit.server.test_agent_memory.MESSAGES", messages):
            store, turn = finalized_trace(client, ctx)
        process(fs, outbox, llm, store, trace_memory_event(turn["uri"], ctx))
    assert {
        n.metadata["structured_fields"]["tool_name"]
        for n in fs.nodes.values()
        if n.category == "tool"
    } == {"Read", "read", "api.search", "api-search"}


def test_units_survive_fusion_hierarchy_and_ranking():
    uri = "ctx://acct/agents/codex/memories/tools/read/content.md"
    hits = [
        SeedHit(uri=uri, score=score, metadata={"agent_memory_version": 1, "unit_key": key})
        for key, score in [("batch", 0.95), ("cursor", 0.55)]
    ]
    query = TypedQuery("batch cursor", "MEMORY", ["tool"])
    for fused in (
        SeedRetriever._fuse_results(hits, hits),
        SeedRetriever._weighted_rrf(hits, hits, semantic_weight=0.7, lexical_weight=0.3, rrf_k=60),
    ):
        assert len(fused) == 2
        leaves = HierarchicalSearcher(InMemoryVectorIndex()).expand(
            query, SeedResult(initial_candidates=fused), CTX, limit=10
        )
        blocks = ResultRanker().assemble(query, leaves, None)
        assert {b.metadata["unit_key"] for b in blocks} == {"batch", "cursor"}


def test_conflict_retry_retains_committed_candidate():
    from commit.context_writer import ContextWriter

    client, fs, outbox, llm = setup()
    store, turn = finalized_trace(client)
    event = trace_memory_event(turn["uri"], CTX)
    outbox.register_once(event, CTX)
    original = ContextWriter.write_candidate
    interrupted = False

    def write(writer, candidate, ctx):
        nonlocal interrupted
        if candidate.category == "experience" and not interrupted:
            interrupted = True
            other_ctx = replace(ctx, session_id="concurrent")
            other_store, other_turn = finalized_trace(client, other_ctx)
            page = other_store.read_skill_trace_event_page(
                account_id=ctx.account_id, evidence_uri=other_turn["uri"], after_ordinal=0, limit=32
            )
            other = next(
                c
                for c in AgentMemoryExtractor(fs, llm).extract_candidates(
                    other_turn["uri"], page["events"], other_ctx
                )
                if c.category == "experience"
            )
            original(writer, other, other_ctx)
        return original(writer, candidate, ctx)

    with patch.object(ContextWriter, "write_candidate", write):
        try:
            process(fs, outbox, llm, store, event)
        except Exception:
            pass
        restored = next(
            job for _, job in outbox.list_pending("acct") if job.event_id == event.event_id
        )
        process(fs, outbox, llm, store, restored)
    assert {n.category for n in fs.nodes.values()} == {"tool", "experience"}
    assert {n.category: n.metadata["version"] for n in fs.nodes.values()} == {
        "tool": 1,
        "experience": 2,
    }


def test_completed_event_stays_completed_during_registration():
    client, fs, outbox, llm = setup()
    store, turn = finalized_trace(client)
    event = trace_memory_event(turn["uri"], CTX)
    outbox.register_once(event, CTX)
    reached, resume = Event(), Event()
    original = client.read

    def read(path):
        if path.endswith(".agent_memory_done") and not reached.is_set():
            try:
                return original(path)
            except Exception:
                reached.set()
                assert resume.wait(5)
                raise
        return original(path)

    client.read = read
    with ThreadPoolExecutor(max_workers=2) as pool:
        registration = pool.submit(outbox.register_once, deepcopy(event), CTX)
        assert reached.wait(5)
        completion = pool.submit(outbox.mark_done, event, event.uri)
        try:
            completion.result(timeout=0.3)
        except TimeoutError:
            pass
        resume.set()
        registration.result(timeout=5)
        completion.result(timeout=5)
    assert outbox.list_pending("acct") == []


def test_worker_indexes_current_units_after_withdrawal():
    ctx, node, _, _ = unit_memory.__wrapped__()
    client = _MemoryAGFS()
    fs = NodeFS(client)
    outbox = OutboxStore(client, fs)
    index = InMemoryVectorIndex()
    worker = OutboxWorker(index, SameEmbedding(), fs=fs)
    fs.write_node(node, ctx)
    outbox.register_write(node, ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    final = deepcopy(node)
    final.metadata["structured_fields"]["units"] = final.metadata["structured_fields"]["units"][:1]
    final.overview = ""
    final.content = render_agent_memory("tool", final.metadata["structured_fields"])
    final.metadata["agent_memory"]["units"]["cursor"]["withdrawn"] = True
    fs.write_node(final, ctx)
    outbox.register_write(final, ctx)
    assert worker.run_once(outbox, [ctx.account_id])["failed"] == 0
    hits = index.search_by_vector([1.0, 0.0], {"account_id": ctx.account_id}, 10)
    assert {(hit.level, hit.metadata.get("unit_key")) for hit in hits} == {(0, None), (2, "batch")}
