"""MCP queries observe canonical deletion before an index worker runs."""
from __future__ import annotations

from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, RequestContext, RetrievalConfig, Role
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from index.index_record_builder import build_index_records
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.mcp_tools import delete_memory, query_memory
from server.memory_service import MemoryService
from service.api import ReadAPI
from tests.integration.test_mcp_streamable_http_protocol import _required_sql_dsn
from tests.unit.retrieval.test_agent_units import SameEmbedding


@pytest.mark.integration
@pytest.mark.parametrize("category", ["preference", "session_archive"])
@pytest.mark.parametrize("deleted_rank", [0, 4])
def test_first_query_after_delete_excludes_stale_index_hit(category, deleted_rank):
    dsn = _required_sql_dsn()
    pool = SharedConnectionPool(connection_string=dsn, pool_size=2)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    account = "query-after-delete-" + uuid4().hex[:12]
    ctx = RequestContext(account, "alice", "bot", "session", "trace", Role.MEMBER,
                         ("user:alice", "agent:bot"))
    cfg = RetrievalConfig(expand_hierarchy=False, default_score_threshold=0, hotness_alpha=0)
    index = InMemoryVectorIndex(dimension=2)
    reader = ContextReader(fs, storage_backend="sql")
    pipeline = RetrievalPipeline(QueryPlanner(cfg), SeedRetriever(index, SameEmbedding(), cfg),
                                 None, ResultRanker(cfg), cfg, context_reader=reader)
    service = MemoryService(config=OgMemConfig(storage_backend="sql", sql_connection_string=dsn,
                                              index_workers=0, provider="mock"))
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda _fs: outbox
    service._read_api = ReadAPI(pipeline, reader, cfg)
    try:
        directory = "preferences" if category == "preference" else "session_archives"
        for position in range(7):
            uri = f"ctx://{account}/users/alice/memories/{directory}/item-{position}"
            node = ContextNode(uri, "MEMORY", category, 2, "user:alice",
                               f"private archive summary {position}", "overview", f"content {position}")
            fs.write_node(node, ctx)
            records = build_index_records(node)
            for record in records:
                record.metadata["_embedding"] = [1.0, 0.0]
            index.upsert(records)
        before = query_memory(service, query="private archive summary", types=[category], top_k=7,
                              request_context=ctx)
        before_uris = [hit["uri"] for hit in before["hits"]]
        assert len(before_uris) == 7
        target = before_uris[deleted_rank]
        index_ids = set(index._records)

        deleted = delete_memory(service, uri=target, request_context=ctx)
        assert deleted["ok"] and deleted["deleted"]
        pending = outbox.list_pending(account)
        assert len(pending) == 1
        assert pending[0][1].status == "PENDING"
        assert set(index._records) == index_ids

        first_query = query_memory(service, query="private archive summary", types=[category], top_k=7,
                                   request_context=ctx)

        assert [hit["uri"] for hit in first_query["hits"]] == [uri for uri in before_uris if uri != target]
        assert len(outbox.list_pending(account)) == 1
        assert set(index._records) == index_ids
    finally:
        service.shutdown()
        pool.close_all()
