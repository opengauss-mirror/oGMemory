"""Integration tests for the SQL storage backend using a real PostgreSQL DSN.

These tests are optional. They look for a DSN in this order:
TEST_SQL_CONNECTION_STRING -> SQL_CONNECTION_STRING -> OgMemConfig.load().
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.errors import ConcurrentModificationError
from core.models import CandidateMemory, ContextNode, RelationEdge, RequestContext
from fs.sql_adapter import SQLContextFS
from providers.embedder.mock_embedder import MockEmbedder
from providers.llm import MockLLM
from providers.relation_store.sql_relation_store import SQLRelationStore
from providers.unified_config import OgMemConfig
from providers.vector_index import OpenGaussVectorIndex
from server.mcp_resources import (
    external_to_internal_uri_candidates,
    internal_to_external_uri,
)
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI
from session.sql_archive_store import SQLSessionArchiveStore

try:
    import psycopg2
except ImportError:  # pragma: no cover - fixture skips when missing
    psycopg2 = None


pytestmark = pytest.mark.integration


def _wait_for_outbox_listener(dsn: str, timeout_seconds: float = 5.0) -> bool:
    """Poll pg_stat_activity until a LISTEN connection is visible."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND query ILIKE 'LISTEN ogmem_outbox%'
                    """
                )
                if cur.fetchone()[0] > 0:
                    return True
        finally:
            conn.close()
        time.sleep(0.1)
    return False


def _wait_for_pgvector_rows(
    dsn: str,
    account_id: str,
    timeout_seconds: float = 5.0,
) -> tuple[int, list[tuple]]:
    """Wait until async listener finishes vector upsert for the account."""
    deadline = time.time() + timeout_seconds
    last_outbox_rows = 0
    last_vector_rows: list[tuple] = []

    while time.time() < deadline:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM outbox_events WHERE account_id = %s",
                    (account_id,),
                )
                last_outbox_rows = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT id, uri, level, vector_dims(embedding), filters->>'account_id'
                    FROM vector_index
                    WHERE filters->>'account_id' = %s
                    ORDER BY id
                    """,
                    (account_id,),
                )
                last_vector_rows = cur.fetchall()
                if len(last_vector_rows) == 2 and last_outbox_rows == 0:
                    return last_outbox_rows, last_vector_rows
        finally:
            conn.close()
        time.sleep(0.1)

    return last_outbox_rows, last_vector_rows


@pytest.fixture
def sql_dsn():
    cfg = OgMemConfig.load()
    dsn = (
        os.environ.get("TEST_SQL_CONNECTION_STRING")
        or os.environ.get("SQL_CONNECTION_STRING")
        or cfg.sql_connection_string
    )
    if not dsn:
        pytest.skip("No SQL DSN configured in TEST_SQL_CONNECTION_STRING, SQL_CONNECTION_STRING, or ogmem.yaml")
    if psycopg2 is None:
        pytest.skip("psycopg2 is not installed")
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
    except Exception as exc:  # pragma: no cover - depends on local env
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    return dsn


@pytest.fixture
def pgvector_dsn(sql_dsn):
    cfg = OgMemConfig.load()
    dsn = (
        os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING")
        or os.environ.get("OPENGAUSS_CONNECTION_STRING")
        or cfg.opengauss_connection_string
        or sql_dsn
    )
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                pytest.skip("pgvector extension is not enabled on the configured vector database")
    finally:
        conn.close()
    return dsn


@pytest.fixture
def sql_account():
    return f"acct-sql-it-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sql_ctx(sql_account):
    return RequestContext(
        account_id=sql_account,
        user_id="u-sql",
        agent_id="agent-sql",
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        trace_id=f"trace-{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture
def cleanup_sql_account(sql_dsn, sql_account):
    yield

    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            for table in ("relation_edges", "outbox_events", "context_nodes", "session_archives"):
                cur.execute(f"DELETE FROM {table} WHERE account_id = %s", (sql_account,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def cleanup_pgvector_account(pgvector_dsn, sql_account):
    yield

    conn = psycopg2.connect(pgvector_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.vector_index')")
            if cur.fetchone()[0] is not None:
                cur.execute(
                    "DELETE FROM vector_index WHERE filters->>'account_id' = %s",
                    (sql_account,),
                )
        conn.commit()
    finally:
        conn.close()


def test_sql_session_archive_roundtrip(sql_dsn, sql_ctx, cleanup_sql_account):
    cfg = OgMemConfig(
        provider="mock",
        after_turn_threshold=1,
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string=sql_dsn,
        account_id=sql_ctx.account_id,
        user_id=sql_ctx.user_id,
        agent_id=sql_ctx.agent_id,
    )
    service = MemoryService(config=cfg)
    try:
        result = service.after_turn(
            {
                "sessionId": sql_ctx.session_id,
                "messages": [
                    {"role": "user", "content": "Remember that I prefer pour-over coffee."},
                    {"role": "assistant", "content": "Stored."},
                ],
                "prePromptMessageCount": 0,
                "wait": True,
            }
        )

        assert result["ok"] is True
        assert result["status"] == "completed"

        store = SQLSessionArchiveStore(connection_string=sql_dsn)
        archives = store.list_archives(sql_ctx.session_id, sql_ctx)
        assert len(archives) == 1
        assert archives[0].archive_id
        loaded = store.read_archive(sql_ctx.session_id, archives[0].archive_id, sql_ctx)
        assert loaded.session_id == sql_ctx.session_id
        assert loaded.messages[0]["content"] == "Remember that I prefer pour-over coffee."
    finally:
        service.shutdown()


def test_sql_context_fs_roundtrip_via_write_api(sql_dsn, sql_ctx, cleanup_sql_account):
    fs = SQLContextFS(connection_string=sql_dsn)
    outbox = SQLOutboxStore(connection_string=sql_dsn, fs=fs)
    api = MemoryWriteAPI(fs=fs, llm=MockLLM(), outbox_store=outbox)

    result = api.write_memory(
        CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="Backend engineer profile",
            overview="## Profile\n\nBackend engineer who prefers SQL storage.",
            content="Backend engineer who prefers SQL storage.",
            confidence=0.95,
        ),
        sql_ctx,
    )

    assert result["action"] in {"create", "merge"}

    uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/profile"
    assert fs.exists(uri, sql_ctx) is True

    node = fs.read_node(uri, sql_ctx)
    assert "SQL storage" in node.content


@pytest.mark.parametrize("directory,category,slug", [
    ("entities", "entity", "李雷"),
    ("preferences", "preference", "coffee%20shop"),
])
def test_sql_memory_uri_write_read_delete(
    sql_dsn, sql_ctx, cleanup_sql_account, directory, category, slug,
):
    fs = SQLContextFS(connection_string=sql_dsn)
    uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/{directory}/{slug}"
    fs.write_node(
        ContextNode(
            uri=uri, context_type="MEMORY", category=category, level=0,
            owner_space=f"user:{sql_ctx.user_id}", abstract="Stored memory", overview="",
            content="Stored memory", metadata={},
        ),
        sql_ctx,
    )
    public_uri = internal_to_external_uri(uri, category=category)
    existing_uri = next(
        candidate for candidate in external_to_internal_uri_candidates(public_uri, sql_ctx)
        if fs.exists(candidate, sql_ctx)
    )

    assert existing_uri == uri
    assert fs.read_node(existing_uri, sql_ctx).content == "Stored memory"
    fs.delete_node(existing_uri, sql_ctx)
    assert not fs.exists(uri, sql_ctx)


def test_sql_relation_store_roundtrip(sql_dsn, sql_ctx, cleanup_sql_account):
    store = SQLRelationStore(connection_string=sql_dsn)
    source_uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/profile"
    target_uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/entities/coffee"

    store.upsert_edges(
        [
            RelationEdge(
                from_uri=source_uri,
                to_uri=target_uri,
                relation_type="related_to",
                weight=0.88,
                reason="Profile mentions coffee preference",
            )
        ],
        sql_ctx,
    )

    edges = store.get_one_hop(source_uri, sql_ctx, limit=3)
    assert len(edges) == 1
    assert edges[0].to_uri == target_uri
    assert edges[0].weight == 0.88


def test_sql_pgvector_roundtrip_via_memory_service(
    sql_dsn,
    pgvector_dsn,
    sql_ctx,
    cleanup_sql_account,
    cleanup_pgvector_account,
):
    cfg = OgMemConfig(
        provider="mock",
        embedding_provider="mock",
        vector_db_type="opengauss",
        opengauss_connection_string=pgvector_dsn,
        opengauss_dimension=1536,
        opengauss_table_name="vector_index",
        storage_backend="sql",
        sql_connection_string=sql_dsn,
        account_id=sql_ctx.account_id,
        user_id=sql_ctx.user_id,
        agent_id=sql_ctx.agent_id,
    )
    service = MemoryService(config=cfg)
    try:
        write_api = service.get_write_api()

        assert write_api is not None
        assert _wait_for_outbox_listener(sql_dsn) is True

        result = write_api.write_memory(
            CandidateMemory(
                category="profile",
                owner_scope="user",
                routing_key="profile",
                abstract="PostgreSQL vector storage verification profile",
                overview="## Profile\n\nTesting LISTEN/NOTIFY plus pgvector persistence.",
                content="Testing LISTEN/NOTIFY plus pgvector persistence with mock embeddings.",
                confidence=0.99,
            ),
            sql_ctx,
        )

        assert result["action"] in {"create", "merge"}

        # The listener may process the event before this synchronous sweep runs.
        service.drain_outbox_sync()

        outbox_rows, vector_rows = _wait_for_pgvector_rows(
            pgvector_dsn,
            sql_ctx.account_id,
        )

        assert outbox_rows == 0
        assert len(vector_rows) == 2
        assert {row[2] for row in vector_rows} == {0, 2}
        assert all(row[3] == 1536 for row in vector_rows)
        assert all(row[4] == sql_ctx.account_id for row in vector_rows)

        embedder = MockEmbedder(dimension=1536)
        index = OpenGaussVectorIndex(
            connection_string=pgvector_dsn,
            dialect="postgresql",
            dimension=1536,
            table_name="vector_index",
        )
        query_vector = embedder.embed_texts(
            ["Testing LISTEN/NOTIFY plus pgvector persistence with mock embeddings."]
        )[0]
        results = index.search_by_vector(
            query_vector=query_vector,
            filters={
                "account_id": sql_ctx.account_id,
                "owner_space": f"user:{sql_ctx.user_id}",
            },
            top_k=3,
        )

        assert len(results) == 2
        assert results[0].uri.endswith("/content.md")
        assert results[0].score > 0.99
    finally:
        service.shutdown()


# ------------------------------------------------------------------
# A.1  Outbox atomicity rollback
# ------------------------------------------------------------------


def test_write_node_with_outbox_is_atomic(
    sql_dsn, sql_ctx, cleanup_sql_account
):
    """Business write and outbox registration must be atomic:
    when outbox INSERT fails, context_nodes INSERT must also roll back."""
    fs = SQLContextFS(connection_string=sql_dsn)
    outbox = SQLOutboxStore(connection_string=sql_dsn, fs=fs)

    uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/profile"
    node = ContextNode(
        uri=uri,
        context_type="MEMORY",
        category="profile",
        level=0,
        owner_space=sql_ctx.user_space_name(),
        abstract="test atomicity",
        overview="atomicity overview",
        content="atomicity content",
        metadata={},
    )

    event = outbox.build_write_event(node)
    # Corrupt: event_id NOT NULL violation → outbox INSERT fails
    event.event_id = None  # type: ignore[assignment]

    with pytest.raises(Exception):
        fs.write_node_with_outbox(node, sql_ctx, event)

    # Critical assertion: context_nodes must NOT contain the row
    assert fs.exists(uri, sql_ctx) is False


# ------------------------------------------------------------------
# A.2  Optimistic lock concurrent race
# ------------------------------------------------------------------


def test_optimistic_lock_prevents_concurrent_lost_update(
    sql_dsn, sql_ctx, cleanup_sql_account
):
    """Two threads writing with the same expected_version: exactly one
    succeeds, the other gets ConcurrentModificationError."""
    fs = SQLContextFS(connection_string=sql_dsn)

    uri = f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}/memories/profile"

    # Write initial node → version=1
    base_node = ContextNode(
        uri=uri,
        context_type="MEMORY",
        category="profile",
        level=0,
        owner_space=sql_ctx.user_space_name(),
        abstract="base",
        overview="base overview",
        content="base",
        metadata={},
    )
    fs.write_node(base_node, sql_ctx)

    barrier = threading.Barrier(2)
    results: list[Exception | None] = [None, None]

    def writer(i: int, content: str) -> None:
        try:
            node = ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="profile",
                level=0,
                owner_space=sql_ctx.user_space_name(),
                abstract=content,
                overview=f"{content} overview",
                content=content,
                metadata={"expected_version": 1},
            )
            barrier.wait(timeout=5)
            fs.write_node(node, sql_ctx)
        except Exception as e:
            results[i] = e

    t1 = threading.Thread(target=writer, args=(0, "base + X"))
    t2 = threading.Thread(target=writer, args=(1, "base + Y"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    failed = [r for r in results if isinstance(r, ConcurrentModificationError)]
    assert len(failed) == 1, (
        f"expected exactly one ConcurrentModificationError, got {results!r}"
    )

    # Final content must be exactly one of the two payloads (XOR)
    final = fs.read_node(uri, sql_ctx).content
    assert ("base + X" in final) != ("base + Y" in final)


# ------------------------------------------------------------------
# A.3  move_node end-to-end cascade
# ------------------------------------------------------------------


def test_move_node_end_to_end_consistency(
    sql_dsn, sql_ctx, cleanup_sql_account
):
    """move_node on real PG: relation_edges migrate, no dangling edges,
    outbox has no PENDING/PROCESSING events for old URI."""
    fs = SQLContextFS(connection_string=sql_dsn)
    relations = SQLRelationStore(connection_string=sql_dsn)

    old_uri = (
        f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}"
        "/memories/preferences/draft-x"
    )
    target_uri = (
        f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}"
        "/memories/entities/coffee"
    )
    new_uri = (
        f"ctx://{sql_ctx.account_id}/users/{sql_ctx.user_id}"
        "/memories/preferences/x"
    )

    # Write two nodes
    fs.write_node(
        ContextNode(
            uri=old_uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space=sql_ctx.user_space_name(),
            abstract="draft preference",
            overview="draft overview",
            content="draft content",
            metadata={},
        ),
        sql_ctx,
    )
    fs.write_node(
        ContextNode(
            uri=target_uri,
            context_type="MEMORY",
            category="entity",
            level=0,
            owner_space=sql_ctx.user_space_name(),
            abstract="coffee entity",
            overview="coffee overview",
            content="coffee content",
            metadata={},
        ),
        sql_ctx,
    )

    # Create relation edge
    relations.upsert_edges(
        [
            RelationEdge(
                from_uri=old_uri,
                to_uri=target_uri,
                relation_type="mentions",
                weight=1.0,
                reason="",
            )
        ],
        sql_ctx,
    )

    # Move
    fs.move_node(old_uri, new_uri, sql_ctx)

    # 1) Edge now originates from new_uri
    edges = relations.get_edges(new_uri, sql_ctx)
    assert len(edges) == 1
    assert edges[0].to_uri == target_uri

    # 2) No dangling edges from old_uri
    assert relations.get_edges(old_uri, sql_ctx) == []

    # 3) outbox: any PENDING/PROCESSING events for old URI must be
    #    DELETE_CONTEXT (cleanup), not UPSERT_CONTEXT (would write stale
    #    vectors back to old URI).
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT event_type FROM outbox_events "
                "WHERE uri = %s AND status IN ('PENDING', 'PROCESSING')",
                (old_uri,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    for (event_type,) in rows:
        assert event_type == "DELETE_CONTEXT", (
            f"Found dangling {event_type} event for old URI {old_uri}"
        )


def test_move_node_relations_jsonb_rewrite(
    sql_dsn, sql_ctx, cleanup_sql_account
):
    """move_node on real PG: context_nodes.relations JSONB is rewritten
    so that stale from_uri/to_uri do not leak via the fallback read path.

    This exercises the to_jsonb(%s::text) fix: without the explicit ::text
    cast, PostgreSQL cannot infer the polymorphic type for the untyped
    parameter and raises 'could not determine polymorphic type'.
    """
    fs = SQLContextFS(connection_string=sql_dsn)
    relations = SQLRelationStore(connection_string=sql_dsn)

    acct = sql_ctx.account_id
    space = sql_ctx.user_space_name()

    old_uri = f"ctx://{acct}/users/{sql_ctx.user_id}/memories/draft-x"
    new_uri = f"ctx://{acct}/users/{sql_ctx.user_id}/memories/published-x"
    target_uri = f"ctx://{acct}/users/{sql_ctx.user_id}/memories/coffee"
    # A child node under old_uri to exercise prefix-based rewrite
    child_old = old_uri + "/child-y"
    child_new = new_uri + "/child-y"
    # A third-party node that *mentions* old_uri in its relations JSONB
    third_uri = f"ctx://{acct}/users/{sql_ctx.user_id}/memories/third-z"

    # 1) Write nodes with embedded relations JSONB (via metadata._relations)
    #    The third-party node covers both from_uri and to_uri rewrite paths:
    #    - from_uri exact match (old_uri), from_uri prefix match (child_old)
    #    - to_uri exact match (old_uri), to_uri prefix match (child_old)
    fs.write_node(
        ContextNode(
            uri=old_uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space=space,
            abstract="draft",
            overview="draft overview",
            content="draft content",
            metadata={
                "_relations": [
                    RelationEdge(old_uri, target_uri, "mentions", 1.0, ""),
                ],
            },
        ),
        sql_ctx,
    )
    fs.write_node(
        ContextNode(
            uri=target_uri,
            context_type="MEMORY",
            category="entity",
            level=0,
            owner_space=space,
            abstract="coffee",
            overview="coffee overview",
            content="coffee content",
            metadata={},
        ),
        sql_ctx,
    )
    # Third-party node referencing old_uri/child_old in both from_uri AND
    # to_uri to exercise the to_uri rewrite branch of the JSONB UPDATE.
    fs.write_node(
        ContextNode(
            uri=third_uri,
            context_type="MEMORY",
            category="pattern",
            level=0,
            owner_space=space,
            abstract="third",
            overview="third overview",
            content="third content",
            metadata={
                "_relations": [
                    # from_uri rewrite paths
                    RelationEdge(old_uri, third_uri, "related_to", 0.5, ""),
                    RelationEdge(child_old, third_uri, "derived_from", 0.3, ""),
                    # to_uri rewrite paths
                    RelationEdge(third_uri, old_uri, "refers_to", 0.7, ""),
                    RelationEdge(third_uri, child_old, "follows", 0.4, ""),
                ],
            },
        ),
        sql_ctx,
    )

    # Also add relation_edges so the move cascades there too
    relations.upsert_edges(
        [
            RelationEdge(old_uri, target_uri, "mentions", 1.0, ""),
        ],
        sql_ctx,
    )

    # 2) Move the node
    fs.move_node(old_uri, new_uri, sql_ctx)

    # 3) Verify: relation_edges migrated (exact match)
    edges_new = relations.get_edges(new_uri, sql_ctx)
    assert len(edges_new) == 1
    assert edges_new[0].to_uri == target_uri
    assert relations.get_edges(old_uri, sql_ctx) == []

    # 4) Verify: context_nodes.relations JSONB was rewritten
    #    Use fs.read_node (binds RLS tenant) instead of raw psycopg2 query
    #    which would fail under FORCE ROW LEVEL SECURITY without BYPASSRLS.
    third_node = fs.read_node(third_uri, sql_ctx)
    third_rels = third_node.metadata["_relations"]
    from_uris = [e.from_uri for e in third_rels]
    to_uris = [e.to_uri for e in third_rels]

    # from_uri: exact-match rewrite old_uri → new_uri
    assert old_uri not in from_uris, (
        f"Stale from_uri={old_uri} still present in relations JSONB"
    )
    assert new_uri in from_uris, (
        f"Expected from_uri={new_uri} not found in relations JSONB"
    )
    # from_uri: prefix-match rewrite child_old → child_new
    assert child_old not in from_uris, (
        f"Stale from_uri={child_old} still present in relations JSONB"
    )
    assert child_new in from_uris, (
        f"Expected from_uri={child_new} not found in relations JSONB"
    )
    # to_uri: exact-match rewrite old_uri → new_uri
    assert old_uri not in to_uris, (
        f"Stale to_uri={old_uri} still present in relations JSONB"
    )
    assert new_uri in to_uris, (
        f"Expected to_uri={new_uri} not found in relations JSONB"
    )
    # to_uri: prefix-match rewrite child_old → child_new
    assert child_old not in to_uris, (
        f"Stale to_uri={child_old} still present in relations JSONB"
    )
    assert child_new in to_uris, (
        f"Expected to_uri={child_new} not found in relations JSONB"
    )

    # 5) Verify: fallback path returns correct URIs (not stale ones)
    #    Delete the relation_edges row so the fallback path is exercised.
    #    Bind RLS tenant to ensure DELETE succeeds under FORCE ROW LEVEL
    #    SECURITY — without SET LOCAL app.account_id a non-BYPASSRLS role
    #    would affect 0 rows and never actually clear the edge.
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET LOCAL app.account_id = %s", (acct,)
            )
            cur.execute(
                "DELETE FROM relation_edges WHERE from_uri = %s",
                (new_uri,),
            )
            assert cur.rowcount == 1, (
                f"DELETE affected {cur.rowcount} rows — expected 1; "
                "RLS may have blocked the delete"
            )
        conn.commit()
    finally:
        conn.close()

    fallback_edges = relations.get_edges(new_uri, sql_ctx)
    # Fallback must return at least one edge (not silently empty).
    assert len(fallback_edges) >= 1, (
        "Fallback path returned empty edges — expected at least one"
    )
    fallback_from = [e.from_uri for e in fallback_edges]
    assert new_uri in fallback_from, (
        f"Fallback missing expected from_uri={new_uri}"
    )
    assert old_uri not in fallback_from, (
        f"Fallback returned stale from_uri={old_uri}"
    )
    # to_uri: moved node's relations JSONB had to_uri=target_uri which
    # was untouched by the move, so fallback should still see it.
    fallback_to = [e.to_uri for e in fallback_edges]
    assert target_uri in fallback_to, (
        f"Fallback missing expected to_uri={target_uri}"
    )
    assert old_uri not in fallback_to, (
        f"Fallback returned stale to_uri={old_uri}"
    )
