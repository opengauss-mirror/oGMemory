"""Real storage/provider boundary checks for persisted active code retrieval.

This module is intentionally an integration test, not an end-to-end transport test.
The SQL ContextFS, SQL Outbox, OutboxWorker, shared BM25, and selected vector
provider are real implementations.  A deterministic embedder and a small
ReadAPI adapter are test doubles for the model and transport-independent
retrieval assembly; neither replaces a storage or vector backend.

Set ``OGMEM_REQUIRE_REAL_BACKENDS=1`` for formal acceptance.  In that mode a
missing dependency or DSN fails the test instead of being reported as a skip.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import (
    ContextNode,
    RelationEdge,
    RequestContext,
    RetrievedBlock,
    SearchMemoryResult,
)
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.relation_store.sql_relation_store import SQLRelationStore
from providers.unified_config import OgMemConfig
from retrieval.bm25_index import Bm25Index
from retrieval.code_retrieval import (
    ActiveCodeSearchRequest,
    CodeRetrievalService,
    CodeSearchBudget,
    CodeSearchScope,
)

pytestmark = pytest.mark.integration

_DIMENSION = 8


def _formal_real_run() -> bool:
    return os.environ.get("OGMEM_REQUIRE_REAL_BACKENDS", "").strip() == "1"


def _require_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    message = f"{name} is required for the formal real-backend run"
    if _formal_real_run():
        pytest.fail(message)
    pytest.skip(message)


def _request_context(account_id: str) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="integration-user",
        agent_id="integration-agent",
        session_id=uuid.uuid4().hex,
        trace_id=uuid.uuid4().hex,
    )


def _code_node(
    account_id: str,
    slug: str,
    *,
    root: str,
    relative_path: str,
    symbol: str,
    content: str,
    graph: dict[str, Any] | None = None,
) -> ContextNode:
    return ContextNode(
        uri=f"ctx://{account_id}/resources/{slug}",
        context_type="RESOURCE",
        category="code",
        level=0,
        owner_space=f"resource:{account_id}",
        abstract=f"{symbol} code implementation",
        overview=f"{symbol}({relative_path})",
        content=content,
        metadata={
            "resource_type": "code",
            "file_source": content,
            "resource_root_slug": root,
            "root_fingerprint": ("f" * 63) + account_id[-1],
            "file_key": uuid.uuid4().hex * 2,
            "file_slug": uuid.uuid4().hex,
            "relative_path": relative_path,
            "source_extension": ".py",
            "language": "python",
            "start_line": 1,
            "end_line": max(1, len(content.splitlines())),
            "symbol": symbol,
            "symbol_kind": "function",
            "signature": f"{symbol}(value)",
            "chunk_hash": uuid.uuid4().hex * 2,
            "chunk_key": uuid.uuid4().hex * 2,
            "graph": graph or {},
        },
    )


@dataclass
class _SqlBackend:
    dsn: str
    pool: Any
    fs: Any
    outbox: SQLOutboxStore
    relations: SQLRelationStore
    account_id: str
    other_account_id: str
    ctx: RequestContext


@pytest.fixture
def sql_backend() -> _SqlBackend:
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get(
        "TEST_SQL_CONNECTION_STRING", ""
    ).strip()
    if not dsn:
        dsn = _require_value("OGMEM_TEST_SQL_DSN")

    from fs.sql_adapter.pool import SharedConnectionPool
    from fs.sql_adapter.sql_context_fs import SQLContextFS

    account_id = f"active-real-{uuid.uuid4().hex[:12]}"
    other_account_id = f"active-other-{uuid.uuid4().hex[:12]}"
    pool = SharedConnectionPool(dsn, pool_size=8)
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    relations = SQLRelationStore(pool=pool)
    backend = _SqlBackend(
        dsn=dsn,
        pool=pool,
        fs=fs,
        outbox=outbox,
        relations=relations,
        account_id=account_id,
        other_account_id=other_account_id,
        ctx=_request_context(account_id),
    )
    try:
        yield backend
    finally:
        # Cleanup uses a short-lived connection so it does not depend on the
        # request account's RLS binding or leave task rows in a shared runtime.
        try:
            import psycopg2

            conn = psycopg2.connect(dsn)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM relation_edges WHERE account_id = ANY(%s)",
                        ([account_id, other_account_id],),
                    )
                    cur.execute(
                        "DELETE FROM outbox_events WHERE account_id = ANY(%s)",
                        ([account_id, other_account_id],),
                    )
                    cur.execute(
                        "DELETE FROM context_nodes WHERE account_id = ANY(%s)",
                        ([account_id, other_account_id],),
                    )
                conn.commit()
            finally:
                conn.close()
        finally:
            pool.close_all()


@pytest.fixture(params=("chroma", "opengauss"), ids=("real-chroma", "real-opengauss"))
def vector_backend(request: pytest.FixtureRequest, tmp_path):
    provider_name = str(request.param)
    embedder = MockEmbedder(dimension=_DIMENSION)
    if provider_name == "chroma":
        try:
            import chromadb  # noqa: F401
        except ImportError:
            if _formal_real_run():
                pytest.fail("chromadb is required for the formal real-backend run")
            pytest.skip("chromadb is required for real Chroma")
        from providers.vector_index.chroma_index import ChromaVectorIndex

        index = ChromaVectorIndex(
            collection_name=f"active_code_{uuid.uuid4().hex[:16]}",
            persist_directory=str(tmp_path / "chroma"),
            dimension=_DIMENSION,
        )
        yield provider_name, index, embedder
        return

    dsn = os.environ.get("OGMEM_TEST_OPENGAUSS_DSN") or os.environ.get(
        "OPENGAUSS_CONNECTION_STRING", ""
    ).strip()
    if not dsn:
        dsn = _require_value("OGMEM_TEST_OPENGAUSS_DSN")
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        if _formal_real_run():
            pytest.fail("psycopg2 is required for the formal real-backend run")
        pytest.skip("psycopg2 is required for real OpenGauss")
    from providers.vector_index.opengauss_index import OpenGaussVectorIndex

    table_name = f"active_code_{uuid.uuid4().hex[:12]}"
    index = OpenGaussVectorIndex(
        dsn,
        dimension=_DIMENSION,
        table_name=table_name,
        pool_size=2,
    )
    try:
        yield provider_name, index, embedder
    finally:
        # The table is task-owned and is dropped after the test.  Dropping the
        # whole table also removes any L0/L2 rows if setup failed halfway
        # through, which is more reliable than guessing account IDs here.
        try:
            conn = index._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
            finally:
                index._return_connection(conn)
        finally:
            index.close()


class _RealProviderReadAPI:
    """Transport-neutral adapter that calls the production vector provider."""

    def __init__(self, index, embedder):
        self._index = index
        self._embedder = embedder
        self.calls: list[dict[str, Any]] = []

    def search_memory(self, *, query, ctx, top_k, context_type, resource_type, **kwargs):
        del kwargs
        filters = {
            "account_id": ctx.account_id,
            "context_type": context_type,
            "resource_type": resource_type,
        }
        self.calls.append(dict(filters))
        query_vector = self._embedder.embed_texts([query])[0]
        hits = self._index.search_by_vector(query_vector, filters, top_k)
        return SearchMemoryResult(
            query=query,
            hits=[
                RetrievedBlock(
                    uri=hit.uri,
                    score=hit.score,
                    level_hit=f"L{hit.level}",
                    category=hit.category or "code",
                    owner_space=hit.owner_space,
                    abstract=hit.abstract,
                    metadata=dict(hit.metadata or {}),
                )
                for hit in hits
            ],
        )


def _seed_real_data(backend: _SqlBackend) -> list[ContextNode]:
    source = _code_node(
        backend.account_id,
        "source",
        root="repo",
        relative_path="src/source.py",
        symbol="load_config",
        content="def load_config(value):\n    return parse_config(value)\n",
    )
    target = _code_node(
        backend.account_id,
        "target",
        root="repo",
        relative_path="src/target.py",
        symbol="parse_config",
        content="def parse_config(value):\n    return value.strip()\n",
    )
    foreign = _code_node(
        backend.other_account_id,
        "foreign",
        root="repo",
        relative_path="src/target.py",
        symbol="parse_config",
        content="def parse_config(value):\n    return 'foreign'\n",
    )
    for node in (source, target, foreign):
        node_ctx = backend.ctx if node.uri.startswith(f"ctx://{backend.account_id}/") else _request_context(backend.other_account_id)
        event = backend.outbox.build_write_event(node)
        backend.fs.write_node_with_outbox(node, node_ctx, event)

    backend.relations.upsert_edges(
        [
            RelationEdge(
                from_uri=source.uri,
                to_uri=target.uri,
                relation_type="calls",
                weight=1.0,
                reason="real active retrieval integration",
            )
        ],
        backend.ctx,
    )
    return [source, target, foreign]


def test_active_retrieval_crosses_real_sql_outbox_bm25_and_provider(
    sql_backend: _SqlBackend,
    vector_backend,
):
    provider_name, vector_index, embedder = vector_backend
    nodes = _seed_real_data(sql_backend)
    bm25 = Bm25Index()
    worker = OutboxWorker(
        vector_index=vector_index,
        embedder=embedder,
        fs=sql_backend.fs,
        directory_summary_enabled=False,
        bm25_index=bm25,
    )
    stats = worker.run_once(
        sql_backend.outbox,
        [sql_backend.account_id, sql_backend.other_account_id],
        worker_id=f"active-real-{uuid.uuid4().hex[:10]}",
    )
    assert stats["processed"] == len(nodes)
    assert stats["succeeded"] == len(nodes)
    assert sql_backend.outbox.list_pending(sql_backend.account_id) == []
    assert sql_backend.outbox.list_pending(sql_backend.other_account_id) == []
    assert bm25.doc_count == 6

    read_api = _RealProviderReadAPI(vector_index, embedder)
    service = CodeRetrievalService(
        context_fs=sql_backend.fs,
        read_api=read_api,
        bm25_index=bm25,
        relation_store=sql_backend.relations,
        config=OgMemConfig(
            code_resources_enabled=True,
            code_retrieval_enabled=True,
            code_retrieval_candidate_limit=12,
            code_retrieval_result_limit=4,
            code_retrieval_snippet_chars=80,
            code_retrieval_batch_chars=160,
            code_retrieval_route_deadline_ms=5000,
            storage_backend="sql",
            sql_connection_string=sql_backend.dsn,
        ),
    )
    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "parse_config",
            scope=CodeSearchScope(
                root_slugs=("repo",),
                paths=("src/target.py",),
                grep_terms=("parse_config",),
            ),
            budget=CodeSearchBudget(
                candidate_limit=8,
                result_limit=2,
                snippet_chars=60,
                batch_chars=100,
            ),
        ),
        sql_backend.ctx,
    )

    assert outcome.status == "ok"
    assert outcome.evidence
    assert [item.uri for item in outcome.evidence] == [
        f"ctx://{sql_backend.account_id}/resources/target"
    ]
    evidence = outcome.evidence[0]
    assert evidence.excerpt.startswith("def parse_config")
    assert evidence.locator == "repo/src/target.py:1-2"
    assert "/home/" not in evidence.locator
    assert all("foreign" not in item.uri for item in outcome.evidence)
    assert outcome.diagnostics.scope_match_count == 1
    assert outcome.diagnostics.routes["semantic"]["count"] >= 1
    assert outcome.diagnostics.routes["bm25"]["count"] >= 1
    assert outcome.diagnostics.routes["symbolic"]["count"] >= 1
    assert read_api.calls
    assert all(call["account_id"] == sql_backend.account_id for call in read_api.calls)
    assert all(call["resource_type"] == "code" for call in read_api.calls)
    assert provider_name in {"chroma", "opengauss"}


def test_active_retrieval_real_relation_store_renders_bounded_graph_context(
    sql_backend: _SqlBackend,
    vector_backend,
):
    _, vector_index, embedder = vector_backend
    _seed_real_data(sql_backend)
    bm25 = Bm25Index()
    worker = OutboxWorker(
        vector_index=vector_index,
        embedder=embedder,
        fs=sql_backend.fs,
        directory_summary_enabled=False,
        bm25_index=bm25,
    )
    assert worker.run_once(
        sql_backend.outbox,
        [sql_backend.account_id, sql_backend.other_account_id],
        worker_id=f"active-graph-{uuid.uuid4().hex[:10]}",
    )["succeeded"] == 3

    service = CodeRetrievalService(
        context_fs=sql_backend.fs,
        read_api=_RealProviderReadAPI(vector_index, embedder),
        bm25_index=bm25,
        relation_store=sql_backend.relations,
        config=OgMemConfig(
            code_resources_enabled=True,
            code_retrieval_enabled=True,
            code_retrieval_candidate_limit=8,
            code_retrieval_result_limit=3,
            code_retrieval_graph_neighbor_limit=1,
            code_retrieval_route_deadline_ms=5000,
            storage_backend="sql",
            sql_connection_string=sql_backend.dsn,
        ),
    )
    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "load_config",
            scope=CodeSearchScope(root_slugs=("repo",)),
            budget=CodeSearchBudget(result_limit=3, graph_neighbor_limit=1),
        ),
        sql_backend.ctx,
    )

    assert outcome.status == "ok"
    assert any(
        context.get("relation_type") == "calls"
        and context.get("target_locator") == "repo/src/target.py"
        for item in outcome.evidence
        for context in item.graph_context
    )
    assert all(len(item.graph_context) <= 1 for item in outcome.evidence)
    assert all(sql_backend.other_account_id not in item.uri for item in outcome.evidence)
