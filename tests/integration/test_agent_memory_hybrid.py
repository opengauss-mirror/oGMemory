"""Execute hybrid retrieval and SQL extraction recovery scenarios."""

import json
import multiprocessing
import os
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.lexical import LexicalHit, lexical_record_from_index_record
from core.models import ContextNode, RetrievalConfig, TypedQuery
from commit.sql_outbox_store import SQLOutboxStore
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.composite_index import CompositeProjectionWriter
from providers.lexical_index.native_sql_index import DatabaseNativeLexicalIndex
from providers.lexical_index.sql_index import SQLNativeProjectionStore
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.agent_memory import process_trace_memory, trace_memory_event
from server.mcp_tools import query_memory
from server.memory_service import MemoryService
from service.api import ReadAPI
from tests.integration.test_agent_memory_review import _context
from tests.unit.retrieval.test_agent_units import unit_memory
from tests.unit.server.test_agent_memory import (
    ExtractionLLM,
    SameEmbedding,
    _MemoryAGFS,
    finalized_trace,
)
from tests.unit.server.test_mcp_tools_readonly import _FakeService


class ProjectionStore(SQLNativeProjectionStore):
    _prepare_record = DatabaseNativeLexicalIndex._prepare_record


@pytest.fixture
def sql_database():
    dsn = os.environ.get("TEST_POSTGRES_CONNECTION_STRING")
    if not dsn:
        pytest.skip("TEST_POSTGRES_CONNECTION_STRING is required")
    import psycopg2
    from psycopg2 import sql

    name = "agent_review_" + uuid4().hex
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    params = psycopg2.extensions.parse_dsn(dsn)
    params["dbname"] = name
    connection = psycopg2.extensions.make_dsn(**params)
    try:
        yield connection
    finally:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        admin.close()


@pytest.fixture
def sql_storage(sql_database):
    pool = SharedConnectionPool(sql_database)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    try:
        yield fs, outbox
    finally:
        pool.close_all()


def _read_api(index, fs=None, *, expand=False):
    cfg = RetrievalConfig(expand_hierarchy=expand, default_score_threshold=0.01)
    return ReadAPI(RetrievalPipeline(QueryPlanner(cfg), SeedRetriever(index, SameEmbedding(), cfg),
                                    HierarchicalSearcher(index, cfg), ResultRanker(), cfg,
                                    context_reader=ContextReader(fs)), config=cfg)


def _indexed_units():
    ctx, template, _, fs = unit_memory.__wrapped__()
    node = replace(template, uri=template.uri.replace("/tools/", "/tool_units/"))
    fs.nodes[node.uri] = node
    index = InMemoryVectorIndex()
    records = build_index_records(node)
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index.upsert(records)
    return ctx, node, fs, index, records


def test_hybrid_service_extracts_using_readable_vector_index(sql_database, sql_storage):
    fs, outbox = sql_storage
    client, ctx = _MemoryAGFS(), _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    service = MemoryService(config=OgMemConfig(storage_backend="sql", index_workers=0,
                                              retrieval_hybrid_enabled=True, sql_connection_string=sql_database))
    service._vector_index = InMemoryVectorIndex()
    service._projection_index = CompositeProjectionWriter(service._vector_index, SimpleNamespace())
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda fs: outbox
    service._get_trace_event_store = lambda: store
    service._get_shared_embedder = lambda: SameEmbedding()
    service.get_llm = lambda: ExtractionLLM()
    service._process_agent_memory(event)
    assert event.payload["committed_count"] == 2
    assert {candidate["category"] for candidate in event.payload["candidates"]} == {"tool", "experience"}


def test_sql_lexical_projection_preserves_unit_read_identity(sql_database):
    import psycopg2

    ctx, node, fs, _, records = _indexed_units()
    store = ProjectionStore(connection_string=sql_database, table_name="agent_lexical")
    try:
        store.upsert_many([lexical_record_from_index_record(record) for record in records if record.level == 2])
        with closing(psycopg2.connect(sql_database)) as conn, conn, conn.cursor() as cur:
            cur.execute("SELECT record_id,metadata_json FROM agent_lexical_docs ORDER BY record_id")
            hits = [LexicalHit(ident, 1.0, json.loads(metadata) if isinstance(metadata, str) else metadata)
                    for ident, metadata in cur.fetchall()]
        lexical = SimpleNamespace(search=lambda **kwargs: hits,
                                  supported_filters={"account_id", "owner_space", "level", "category", "context_type"})
        cfg = RetrievalConfig(hybrid_enabled=True, semantic_weight=0, lexical_weight=1, expand_hierarchy=False)
        retriever = SeedRetriever(InMemoryVectorIndex(), SameEmbedding(), cfg, lexical,
                                  SimpleNamespace(ensure_ready=lambda *args: True))
        seeds = retriever.search(TypedQuery("batch cursor", "MEMORY", ["tool"]), ctx)
        blocks = ResultRanker().assemble(TypedQuery("batch cursor", "MEMORY", ["tool"]),
                                        RetrievalPipeline._fallback_l2_from_seed(seeds), None,
                                        ctx=ctx, fill_content_for_top_k=2, context_reader=ContextReader(fs))
        assert {block.metadata.get("unit_key") for block in blocks} == {"batch", "cursor"}
        assert {hit.metadata.get("unit_revision") for hit in hits} == {1, 2}
        assert {hit.metadata.get("agent_memory_unit_key") for hit in hits} == {"batch", "cursor"}
        assert [hit.metadata["grounding"] for hit in hits] == [
            next(unit["grounding"] for unit in node.metadata["structured_fields"]["units"]
                 if unit["key"] == hit.metadata["unit_key"]) for hit in hits]
        assert [len(block.metadata["structured_fields"]["units"]) for block in blocks] == [1, 1]
    finally:
        store.close()


def test_skill_enabled_search_keeps_each_tool_unit(monkeypatch):
    ctx, _, fs, index, _ = _indexed_units()
    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    service = MemoryService(config=OgMemConfig(skill_sedimentation_enabled=True, workers=1))
    result = service._search_without_generated_skills(
        _read_api(index, fs), query="batch cursor", ctx=ctx, top_k=2, fill_content_for_top_k=2)
    assert {hit.metadata.get("unit_key") for hit in result.hits} == {"batch", "cursor"}
    assert all(hit.content_excerpt for hit in result.hits)


@pytest.fixture(params=["memory", "chroma", "postgres"])
def search_index(request, tmp_path, monkeypatch):
    backend = request.param
    if backend == "chroma":
        pytest.importorskip("chromadb")
        from providers.vector_index.chroma_index import ChromaVectorIndex

        index = ChromaVectorIndex(collection_name="type_filters", persist_directory=str(tmp_path), dimension=2)
    elif backend == "postgres":
        import psycopg2

        from providers.vector_index import opengauss_index
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        monkeypatch.setattr(opengauss_index, "Json", psycopg2.extras.Json)
        dsn = request.getfixturevalue("sql_database")
        with closing(psycopg2.connect(dsn)) as conn, conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION vector")
        index = OpenGaussVectorIndex(connection_string=dsn, dimension=2, dialect="postgresql")
    else:
        index = InMemoryVectorIndex()
    try:
        yield index
    finally:
        if hasattr(index, "close"):
            index.close()


@pytest.mark.parametrize("requested", ["tool_memory", "tool_evidence"])
@pytest.mark.parametrize("expand", [False, True])
def test_public_type_is_filtered_before_candidate_limit(requested, expand, search_index):
    ctx, template, fs, index, _ = _indexed_units()
    index = search_index
    wanted_dir = "tools" if requested == "tool_memory" else "tool_units"
    other_dir = "tool_units" if wanted_dir == "tools" else "tools"
    for i in range(61):
        directory = wanted_dir if i == 60 else other_dir
        uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/{directory}/search-{i}"
        if directory == "tool_units":
            node = replace(deepcopy(template), uri=uri)
        else:
            node = ContextNode(uri, "MEMORY", "tool", 4, ctx.agent_space_name(),
                               "Search pagination", "", "Use the returned cursor.")
        fs.nodes[uri] = node
        records = build_index_records(node)
        for record in records:
            record.metadata["_embedding"] = [0.7, 0.7] if i == 60 else [1.0, 0.0]
        index.upsert(records)
    service = _FakeService()
    service.read_api = _read_api(index, fs, expand=expand)
    result = query_memory(service, query="cursor", types=[requested], top_k=1, request_context=ctx)
    assert len(result["hits"]) == 1
    assert result["hits"][0]["type"] == requested
    assert result["hits"][0]["id"] == "search-60"


def test_mixed_public_types_share_recall(search_index):
    ctx, node, fs, _, records = _indexed_units()
    user = fs.nodes["ctx://acct/users/user/memories/preferences/research"]
    records += build_index_records(user)
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    search_index.upsert(records)
    service = _FakeService()
    service.read_api = _read_api(search_index, fs)
    result = query_memory(service, query="batch cursor", types=["preference", "tool_evidence"],
                          top_k=3, request_context=ctx)
    assert sorted(hit["type"] for hit in result["hits"]) == ["preference", "tool_evidence", "tool_evidence"]


@pytest.mark.parametrize("directory", ["tools", "tool_units"])
def test_sql_lexical_type_filter_applies_before_limit(sql_database, directory):
    import psycopg2

    ctx, template, _, _, records = _indexed_units()

    store = ProjectionStore(connection_string=sql_database, table_name="typed_lexical")
    try:
        items = []
        for i in range(61):
            kind = directory if i == 60 else ("tools" if directory == "tool_units" else "tool_units")
            uri = f"ctx://acct/agents/agent/memories/{kind}/search-{i}"
            items.append(replace(records[-1], id=str(i), uri=uri))
        store.upsert_many([lexical_record_from_index_record(record) for record in items])
        where, params = store._filter_where({"account_id": ctx.account_id,
                                             "owner_space": ctx.agent_space_name(),
                                             "memory_directory": [directory], "level": 2})
        with closing(psycopg2.connect(sql_database)) as conn, conn, conn.cursor() as cur:
            cur.execute(f"SELECT record_id FROM typed_lexical_docs d WHERE {where} ORDER BY record_id::int LIMIT 1", params)
            assert cur.fetchone() == ("60",)
    finally:
        store.close()


def _sql_extraction(dsn, ctx, turn, reached, resume):
    pool = SharedConnectionPool(dsn)
    try:
        fs, client = SQLContextFS(pool=pool), _MemoryAGFS()
        outbox = SQLOutboxStore(pool=pool, fs=fs)
        store, recreated = finalized_trace(client, ctx)
        assert recreated["uri"] == turn["uri"]

        class PausedLLM(ExtractionLLM):
            def complete_with_tools(self, *args, **kwargs):
                if not reached.is_set():
                    reached.set()
                    assert resume.wait(20)
                return super().complete_with_tools(*args, **kwargs)

        worker = OutboxWorker(InMemoryVectorIndex(), SameEmbedding(), fs=fs,
            agent_memory_handler=lambda job: process_trace_memory(job, trace_store=store,
                fs=fs, outbox=outbox, llm=PausedLLM()))
        assert worker.run_once(outbox, [ctx.account_id], worker_id="first")["succeeded"] == 1
    finally:
        pool.close_all()


@pytest.mark.parametrize("state", ["running", "completed"])
def test_sql_expired_claim_keeps_current_extraction_state(sql_database, sql_storage, state):
    import psycopg2

    fs, outbox = sql_storage
    client, ctx = _MemoryAGFS(), _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    mp = multiprocessing.get_context("spawn")
    reached, resume = mp.Event(), mp.Event()
    child = mp.Process(target=_sql_extraction, args=(sql_database, ctx, turn, reached, resume))
    child.start()
    try:
        assert reached.wait(10)
        with closing(psycopg2.connect(sql_database)) as conn, conn, conn.cursor() as cur:
            cur.execute("UPDATE outbox_events SET locked_at=NOW()-interval '301 seconds' WHERE event_id=%s",
                        (event.event_id,))
        claimed = outbox.claim_batch("second")
        assert len(claimed) == 1
        if state == "completed":
            resume.set()
            child.join(15)
            assert child.exitcode == 0
        worker = OutboxWorker(InMemoryVectorIndex(), SameEmbedding(), fs=fs,
            agent_memory_handler=lambda job: process_trace_memory(job, trace_store=store,
                fs=fs, outbox=outbox, llm=ExtractionLLM()))
        stats = dict(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0)
        worker._process_claimed_events(claimed, outbox, stats)
        resume.set()
        child.join(15)
        assert child.exitcode == 0
        current = outbox.get_event(ctx.account_id, event.event_id)
        assert current.status == "DONE", (current, stats)
        assert current.retry_count == 0, stats
        assert stats["skipped"] == 1, stats
    finally:
        resume.set()
        if child.is_alive():
            child.terminate()
            child.join(5)


def test_sql_takeover_resumes_persisted_checkpoint(sql_storage):
    fs, outbox = sql_storage
    client, ctx = _MemoryAGFS(), _context()
    store, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    claimed = outbox.claim_batch("first")
    save_payload = outbox.save_payload

    def checkpoint_then_interrupt(job):
        save_payload(job)
        if job.payload.get("committed_count") == 1:
            raise InterruptedError("worker stopped after first unit commit")

    outbox.save_payload = checkpoint_then_interrupt
    with pytest.raises(InterruptedError):
        process_trace_memory(event, trace_store=store, fs=fs, outbox=outbox, llm=ExtractionLLM())
    outbox.save_payload = save_payload
    resumed = []

    def finish(job):
        resumed.append(job.payload["committed_count"])
        process_trace_memory(job, trace_store=store, fs=fs, outbox=outbox, llm=ExtractionLLM())
        resumed.append(job.payload["committed_count"])

    worker = OutboxWorker(InMemoryVectorIndex(), SameEmbedding(), fs=fs, agent_memory_handler=finish)
    stats = dict(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0)
    worker._process_claimed_events(claimed, outbox, stats, worker_id="second")
    completed = outbox.get_event(ctx.account_id, event.event_id)
    assert resumed == [1, 2]
    assert completed.status == "DONE"
    assert stats["succeeded"] == 1
