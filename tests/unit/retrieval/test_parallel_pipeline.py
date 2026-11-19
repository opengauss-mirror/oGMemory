"""Scoped multi-route retrieval with shared embeddings and bounded workers."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, Event, Lock, Semaphore, get_ident
from unittest.mock import MagicMock

import pytest

from core.models import RequestContext, RetrievalConfig, SeedHit, TypedQuery
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever

CTX = RequestContext(account_id="acct", user_id="alice", agent_id="assistant",
                     session_id="session", trace_id="trace")
REQUEST = ContextVar("parallel_test_request", default="")


class Embedder:
    def __init__(self):
        self.inputs = []

    def embed_texts(self, texts):
        self.inputs.extend(texts)
        return [[0.1, 0.2] for _ in texts]


def pipeline(index, embedder, **options):
    cfg = RetrievalConfig(expand_hierarchy=False, hotness_alpha=0.0)
    return RetrievalPipeline(
        QueryPlanner(cfg), SeedRetriever(index, embedder, cfg),
        None, ResultRanker(cfg), cfg, **options,
    )


def test_routes_share_embedding_context_and_ordered_trace():
    barrier = Barrier(3)
    calls = []

    class Index:
        def search_by_vector(self, query_vector, filters, top_k):
            calls.append((filters, query_vector, REQUEST.get()))
            barrier.wait(timeout=10)
            count = 1 if filters["context_type"] == "MEMORY" else 3 if filters.get("category") else 2
            return [SeedHit(uri=f"ctx://acct/users/alice/memories/{count}-{i}",
                            score=0.9, level=2) for i in range(count)]

    embedder = Embedder()
    pipe = pipeline(Index(), embedder)
    token = REQUEST.set("request-123")
    try:
        result = pipe.run("Alice preferences", CTX, hints={"include_all_memories": True})
    finally:
        REQUEST.reset(token)

    assert embedder.inputs == ["Alice preferences"]
    assert len(result.hits) == 6
    assert len(calls) == 3
    assert all(vector == [0.1, 0.2] and request == "request-123" for _, vector, request in calls)
    assert all(filters["account_id"] == "acct" for filters, _, _ in calls)
    assert [stage.output_count for stage in result.trace.stages if stage.stage == "seed_retrieval"] == [1, 2, 3]
    assert all(hit.level_hit == "L2" for hit in result.hits)


def test_supplied_vector_matches_sanitized_query_and_new_text_is_embedded():
    calls = []

    class Index:
        def search_by_vector(self, query_vector, filters, top_k):
            calls.append(query_vector)
            return []

    class Planner:
        def plan(self, query, ctx, **kwargs):
            return [TypedQuery(text=text, context_type="MEMORY", categories=[])
                    for text in ("Alice", "Bob", "Bob")]

    embedder = Embedder()
    pipe = pipeline(Index(), embedder)
    pipe.planner = Planner()
    result = pipe.run("  Alice  ", CTX, query_vector=[0.9, 0.8])

    assert embedder.inputs == ["Bob"]
    assert sorted(calls) == [[0.1, 0.2], [0.1, 0.2], [0.9, 0.8]]
    assert len(result.typed_queries) == 3


def test_identical_ranking_inputs_validate_each_selected_node_once():
    checked = []

    class Index:
        def search_by_vector(self, query_vector, filters, top_k):
            return [SeedHit(uri="ctx://acct/users/alice/memories/preference", score=0.9, level=2)]

    class Reader:
        def is_visible(self, uri, ctx):
            checked.append((uri, ctx.account_id))
            return True

    result = pipeline(Index(), Embedder(), context_reader=Reader()).run(
        "Alice preferences", CTX, hints={"include_all_memories": True},
    )

    assert [hit.uri for hit in result.hits] == ["ctx://acct/users/alice/memories/preference"]
    assert checked == [("ctx://acct/users/alice/memories/preference", "acct")]


def test_concurrent_requests_share_eight_seed_workers():
    gate, capacity = Event(), Event()
    lock = Lock()
    active = peak = completed = 0

    class Index:
        def search_by_vector(self, query_vector, filters, top_k):
            nonlocal active, peak, completed
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 8:
                    capacity.set()
            gate.wait(timeout=10)
            with lock:
                active -= 1
                completed += 1
            return []

    pipe = pipeline(Index(), Embedder())
    with ThreadPoolExecutor(max_workers=8) as callers:
        futures = [callers.submit(pipe.run, "Alice preferences", CTX,
                                  hints={"include_all_memories": True}) for _ in range(8)]
        try:
            assert capacity.wait(timeout=10)
        finally:
            gate.set()
        results = [future.result(timeout=10) for future in futures]

    assert len(results) == 8
    assert completed == 24
    assert peak == 8


def test_lexical_mode_prepares_empty_vector():
    embedder = Embedder()
    seed = SeedRetriever(None, embedder, RetrievalConfig(hybrid_enabled=True, semantic_weight=0))
    assert seed.embed_query("Alice preferences") == []
    assert embedder.inputs == []


@pytest.mark.parametrize("query_vector", [None, [0.1, 0.2]])
def test_guarded_routes_keep_owning_sql_session_and_return_all_hits(monkeypatch, query_vector):
    from fs.sql_adapter import session_mutation

    dsn = "dbname=test host=localhost"
    raw = MagicMock(closed=0)
    raw.cursor.return_value.__enter__.return_value.fetchone.return_value = (True,)
    monkeypatch.setattr("fs.sql_adapter.pool._resolved_hosts", lambda *args: {})
    monkeypatch.setattr(session_mutation.psycopg2, "connect", lambda *args, **kwargs: raw)
    calls = []

    class Index:
        _sql_dialect = "postgresql"

        def _get_connection(self):
            return raw

        def _return_connection(self, conn):
            conn.rollback()

        def search_by_vector(self, query_vector, filters, top_k):
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                calls.append((get_ident(), conn, filters["context_type"]))
                return [SeedHit(uri=f"ctx://acct/users/alice/memories/route-{len(calls)}",
                                score=0.9, level=2)]
            finally:
                self._return_connection(conn)

    index, embedder = Index(), Embedder()
    session_mutation.guard_pool_methods(index, dsn)
    with session_mutation.session_mutation_scope(dsn, CTX, 7, Semaphore(1)) as guard:
        result = pipeline(index, embedder).run(
            "Alice preferences", CTX, hints={"include_all_memories": True},
            query_vector=query_vector,
        )
        guard.check_active()
        assert [(thread, conn) for thread, conn, _ in calls] == [(guard.thread_id, guard.connection)] * 3
        assert guard.borrowed == 0

    assert [kind for _, _, kind in calls] == ["MEMORY", "SKILL", "SKILL"]
    assert [hit.uri for hit in result.hits] == [f"ctx://acct/users/alice/memories/route-{i}" for i in (1, 2, 3)]
    assert embedder.inputs == (["Alice preferences"] if query_vector is None else [])
