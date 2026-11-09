"""Ordinary hybrid reads become available before relation projection completes."""

import json
import os
from types import SimpleNamespace
from uuid import uuid4

from core.lexical import LexicalHit, lexical_record_from_index_record
from core.models import RelationEdge, RetrievalConfig
from fs.sql_adapter.node_scanner import SQLCanonicalNodeScanner
from fs.sql_adapter.pool import SharedConnectionPool
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.composite_index import CompositeProjectionWriter
from providers.lexical_backfill import LexicalBackfill
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.owner_scope import expand_owner_space_aliases
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from service.api import ReadAPI
from tests.integration.test_agent_memory_hybrid import ProjectionStore
from tests.integration.test_semantic_context_graph import graph_db as graph_db, _graph_sql


def observe_graph_readiness(runtime):
    ctx, fs, index, relations, projections, outbox, original_worker, nodes, profile = runtime
    source, target, _ = nodes
    prefix = "graph_lex_" + uuid4().hex[:12]
    lexical = ProjectionStore(connection_string=os.environ["GRAPH_TEST_DSN"], table_name=prefix)
    scan_pool = SharedConnectionPool(os.environ["GRAPH_TEST_DSN"])
    scanner = SQLCanonicalNodeScanner(scan_pool)
    coordinator = LexicalBackfill(scanner, lexical)
    scopes = expand_owner_space_aliases([ctx.user_space_name(), ctx.agent_space_name()])
    embedder = original_worker._embedder
    worker = OutboxWorker(CompositeProjectionWriter(index, lexical), embedder, fs=fs,
                          directory_summary_enabled=False)
    records = [lexical_record_from_index_record(record)
               for node in nodes for record in build_index_records(node)]
    assert lexical.replace_scopes(rebuild_id=lexical.begin_rebuild(), account_id=ctx.account_id,
                                  owner_spaces=scopes, records=records)

    def search(*, query, top_k, filters=None):
        rows = _graph_sql(fs, ctx,
            f"SELECT record_id,metadata_json FROM {prefix}_docs "
            "WHERE account_id=%s AND owner_space=ANY(%s) AND level=2 "
            "AND content ILIKE %s ORDER BY record_id LIMIT %s",
            (ctx.account_id, scopes, "%telescope%", top_k))
        return [LexicalHit(key, 1., json.loads(meta) if isinstance(meta, str) else meta)
                for key, meta in rows]

    cfg = RetrievalConfig(hybrid_enabled=True, semantic_weight=0, lexical_weight=1,
                          expand_hierarchy=False)
    reader = ContextReader(fs, storage_backend="sql")
    lexical_search = SimpleNamespace(search=search, supported_filters=lexical.supported_filters)
    pipeline = RetrievalPipeline(QueryPlanner(cfg),
        SeedRetriever(index, embedder, cfg, lexical_search, coordinator),
        HierarchicalSearcher(index, cfg), ResultRanker(cfg), cfg, context_reader=reader)
    read_api = ReadAPI(pipeline, reader, cfg)
    try:
        relations.upsert_edges([RelationEdge(source.uri, target.uri, "owns", 1., "evidence")], ctx)
        current = fs.read_node(target.uri, ctx)
        current.content = "The observatory telescope is ready for nightly observations."
        current.metadata["_expected_version"] = current.metadata["version"]
        fs.write_node(current, ctx)
        ordinary = outbox.register_write(current, ctx)
        worker.run_once(outbox, [ctx.account_id])
        ready = coordinator.ensure_ready(ctx, scopes)
        result = (read_api.search_memory("telescope", ctx, categories=["entity"],
                  top_k=5, score_threshold=0, fill_content_for_top_k=5)
                  if ready else None)
        observation = dict(
            ordinary_status=outbox.get_event(ctx.account_id, ordinary.event_id).status,
            ready_scopes=sorted(lexical.ready_scopes(ctx.account_id, scopes)),
            lexical_ready=ready,
            graph_jobs=_graph_sql(fs, ctx,
                "SELECT status,attempt_count FROM edge_projection_jobs WHERE account_id=%s",
                (ctx.account_id,)),
            global_unfinished=outbox.count_unfinished(ctx.account_id)["total"],
            hits=[hit.uri for hit in result.hits] if result else [],
            expected_uri=target.uri,
            content=read_api.read_memory(target.uri, ctx).content_excerpt,
        )
        original_worker.run_once(outbox, [ctx.account_id])
        observation["final_graph_jobs"] = _graph_sql(fs, ctx,
            "SELECT status,target_revision FROM edge_projection_jobs WHERE account_id=%s",
            (ctx.account_id,))
        return observation
    finally:
        coordinator.close()
        lexical.close()
        scan_pool.close_all()


def test_hybrid_reads_complete_while_graph_projection_waits(graph_db):
    result = observe_graph_readiness(graph_db)
    assert result["ordinary_status"] == "DONE"
    assert len(result["ready_scopes"]) == 4
    assert result["lexical_ready"]
    assert result["graph_jobs"] == [("PENDING", 0)]
    assert result["global_unfinished"] > 0
    assert result["hits"] == [result["expected_uri"] + "/content.md"]
    assert "telescope" in result["content"]
    assert result["final_graph_jobs"] == [("COMPLETE", 2)]
