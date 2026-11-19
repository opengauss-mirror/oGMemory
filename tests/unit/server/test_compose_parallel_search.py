"""Compose keeps its two retrieval quotas while sharing query preparation."""

from contextvars import ContextVar
from threading import Barrier, get_ident
from types import SimpleNamespace

from core.models import RequestContext, RetrievedBlock, SearchMemoryResult
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import ReadAPI


def test_compose_queries_share_one_vector_and_request_context():
    prepared = []
    calls = []
    vector = [0.3, 0.7]
    overlap = Barrier(2)
    scope = ContextVar("compose_test_request", default="")
    scope.set("request-1")
    ctx = RequestContext(account_id="account", user_id="user", agent_id="agent", session_id="session", trace_id="trace")

    def embed_query(text):
        prepared.append(text)
        return vector

    def run(query, read_ctx, **options):
        assert read_ctx is ctx
        assert options["query_vector"] is vector
        assert scope.get() == "request-1"
        calls.append(options)
        overlap.wait(timeout=5)
        category = "session_summary" if options["categories"] else "preference"
        return SearchMemoryResult(hits=[RetrievedBlock(
            uri=f"ctx://account/users/user/memories/{category}/one",
            category=category, abstract=category, score=0.9,
        )])

    api = ReadAPI(SimpleNamespace(seed_retriever=SimpleNamespace(embed_query=embed_query), run=run))
    service = MemoryService(config=OgMemConfig())
    service.get_read_api = lambda: api
    items, summaries, stats = service._search_working_set("  Remember my preferences  ", ctx)

    assert prepared == ["Remember my preferences"]
    assert sorted(call["top_k"] for call in calls) == [1, 15]
    assert [item["category"] for item in items] == ["preference"]
    assert [item["category"] for item in summaries] == ["session_summary"]
    assert stats["hit_count"] == 1


def test_compose_keeps_agfs_searches_and_reports_partial_summary():
    ctx = RequestContext("account", "user", "agent", "session", "trace")
    calls = []

    def run(query, read_ctx, *, top_k, categories, **options):
        assert query == "Remember my preferences"
        assert read_ctx is ctx
        calls.append((top_k, categories, get_ident()))
        category = "session_summary" if categories else "preference"
        hit = RetrievedBlock(
            uri=f"ctx://account/users/user/memories/{category}/one",
            category=category, abstract=category, score=0.9,
        )
        return SearchMemoryResult(hits=[hit], partial=bool(categories))

    api = ReadAPI(SimpleNamespace(run=run))
    service = MemoryService(config=OgMemConfig(retrieval_mode="agfs"))
    service.get_read_api = lambda: api
    items, summaries, stats = service._search_working_set("Remember my preferences", ctx)

    assert calls == [(15, None, get_ident()), (1, ["session_summary"], get_ident())]
    assert [item["category"] for item in items] == ["preference"]
    assert [item["category"] for item in summaries] == ["session_summary"]
    assert stats["retrieval"] == {"mode": "agfs", "status": "partial"}


def test_compose_preserves_sql_session_thread_ownership():
    from core.models import RetrievalConfig, SeedHit
    from fs.sql_adapter.session_mutation import _ACTIVE, SQLSessionGuard
    from retrieval.pipeline import RetrievalPipeline
    from retrieval.query_planner import QueryPlanner
    from retrieval.result_ranker import ResultRanker
    from retrieval.seed_retriever import SeedRetriever

    ctx = RequestContext("account", "user", "agent", "session", "trace")
    guard = SQLSessionGuard(SimpleNamespace(closed=False), "dbname=test host=localhost", ctx, 7)
    calls, embeddings = [], []

    class Index:
        def search_by_vector(self, query_vector, filters, top_k):
            guard.check_context(ctx)
            calls.append(get_ident())
            category = filters.get("category", "preference")
            if isinstance(category, list):
                category = category[0]
            return [SeedHit(uri=f"ctx://account/users/user/memories/{category}/one",
                            category=category, score=0.9, level=2)]

    def embed_texts(texts):
        embeddings.extend(texts)
        return [[0.3, 0.7] for _ in texts]

    cfg = RetrievalConfig(expand_hierarchy=False)
    pipe = RetrievalPipeline(
        QueryPlanner(cfg), SeedRetriever(Index(), SimpleNamespace(embed_texts=embed_texts), cfg),
        None, ResultRanker(cfg), cfg,
    )
    service = MemoryService(config=OgMemConfig())
    service.get_read_api = lambda: ReadAPI(pipe, config=cfg)
    token = _ACTIVE.set(guard)
    try:
        items, summaries, _ = service._search_working_set("Remember my preferences", ctx)
        guard.check_context(ctx)
    finally:
        _ACTIVE.reset(token)

    assert items and [item["category"] for item in summaries] == ["session_summary"]
    assert calls == [get_ident()] * 4
    assert embeddings == ["Remember my preferences"]
