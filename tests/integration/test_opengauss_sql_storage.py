import os
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from threading import Event

import psycopg2
import pytest
from psycopg2.extensions import cursor

from commit.sql_outbox_store import SQLOutboxStore
from core.errors import ConcurrentModificationError, NodeNotFoundError
from core.models import CandidateMemory, ContextNode, RelationEdge, RequestContext
from fs.sql_adapter import SharedConnectionPool, SQLContextFS
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from providers.relation_store.sql_relation_store import SQLRelationStore
from session.sql_archive_store import SQLSessionArchiveStore


@pytest.fixture
def opengauss_sql_dsn():
    dsn = os.environ.get("OGMEM_OPENGAUSS_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_OPENGAUSS_SQL_DSN is not configured")
    return dsn


def test_opengauss_sql_context_storage_round_trip(opengauss_sql_dsn):
    account_id = f"ogmem-it-{uuid.uuid4().hex[:12]}"
    ctx = RequestContext(
        account_id=account_id,
        user_id="alice",
        agent_id="main",
        session_id="session-1",
        trace_id="trace-1",
    )
    old_uri = f"ctx://{account_id}/users/alice/memories/profile"
    new_uri = old_uri + "-moved"
    related_uri = f"ctx://{account_id}/users/alice/memories/other"
    pool = SharedConnectionPool(
        opengauss_sql_dsn,
        pool_size=5,
        dialect="opengauss",
    )
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    relations = SQLRelationStore(pool=pool)
    archives = SQLSessionArchiveStore(pool=pool)

    try:
        node = ContextNode(
            uri=old_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="",
            overview="",
            content="initial",
            metadata={
                "_expected_version": 0,
                "_relations": [
                    RelationEdge(
                        from_uri=old_uri,
                        to_uri=related_uri,
                        relation_type="related_to",
                        weight=0.5,
                        reason="integration",
                    )
                ]
            },
        )
        event = outbox.build_write_event(node)
        fs.write_node_with_outbox(node, ctx, event)
        assert fs.read_node(old_uri, ctx).content == "initial"
        pending = outbox.list_pending(account_id)
        assert len(pending) == 1
        pending_uri, pending_event = pending[0]
        assert outbox.try_acquire(pending_event, pending_uri, "opengauss-it") is True
        outbox.release(pending_event, pending_uri)

        with pytest.raises(ConcurrentModificationError):
            fs.write_node(node, ctx)

        node.metadata.pop("_expected_version")
        node.content = "updated"
        fs.write_node(node, ctx)
        assert fs.read_node(old_uri, ctx).content == "updated"

        relations.upsert_edges(
            [
                RelationEdge(
                    from_uri=old_uri,
                    to_uri=related_uri,
                    relation_type="related_to",
                    weight=0.5,
                    reason="integration",
                )
            ],
            ctx,
        )
        assert relations.get_edges(old_uri, ctx)[0].reason == "integration"

        fs.move_node(old_uri, new_uri, ctx)
        moved = fs.read_node(new_uri, ctx)
        assert moved.content == "updated"
        assert moved.metadata["_relations"][0].from_uri == new_uri

        result = archives.write_archive(
            "session-1",
            "overview",
            "abstract",
            [{"role": "user", "content": "hello"}],
            ctx,
            archive_id="archive-1",
        )
        assert result.success is True
        assert archives.read_archive("session-1", "archive-1", ctx).abstract == "abstract"
    finally:
        conn = pool.get_connection()
        try:
            SQLContextFS.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                for table in (
                    "relation_edges",
                    "outbox_events",
                    "context_nodes",
                    "session_archives",
                ):
                    cur.execute(
                        f"DELETE FROM {table} WHERE account_id = %s",
                        (account_id,),
                    )
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()


def test_opengauss_tenant_isolation_blocks_cross_account_access(opengauss_sql_dsn):
    account_a = f"ogmem-rls-a-{uuid.uuid4().hex[:12]}"
    account_b = f"ogmem-rls-b-{uuid.uuid4().hex[:12]}"
    uri = f"ctx://{account_a}/users/alice/memories/private"
    pool = SharedConnectionPool(opengauss_sql_dsn, dialect="opengauss")
    fs = SQLContextFS(pool=pool)
    fs.write_node(
        ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="private",
            overview="private",
            content="private",
        ),
        RequestContext(
            account_id=account_a,
            user_id="alice",
            agent_id="main",
            session_id="session-rls",
            trace_id="trace-rls",
        ),
    )
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, account_b)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM context_nodes WHERE uri = %s", (uri,))
            assert cur.fetchone()[0] == 0
            with pytest.raises(psycopg2.Error, match="tenant_write_guard"):
                cur.execute(
                    """
                    INSERT INTO context_nodes
                        (uri, account_id, owner_space, category, context_type,
                         level, status, relations, metadata, version)
                    VALUES (%s, %s, %s, %s, %s, 0, 'ACTIVE', '[]', '{}', 1)
                    """,
                    (
                        "ctx://acct-b/users/bob/memories/blocked",
                        "acct-b",
                        "user:bob",
                        "profile",
                        "MEMORY",
                    ),
                )
        conn.rollback()
    finally:
        SQLContextFS.bind_tenant(conn, account_a)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM context_nodes WHERE uri = %s", (uri,))
        conn.commit()
        pool.return_connection(conn)
        pool.close_all()


def test_opengauss_sql_trace_storage_round_trip(opengauss_sql_dsn):
    account_id = f"ogmem-trace-it-{uuid.uuid4().hex[:12]}"
    owner_space = "user:alice"
    agent_id = "main"
    session_id = "session-1"
    pool = SharedConnectionPool(
        opengauss_sql_dsn,
        pool_size=3,
        dialect="opengauss",
    )
    traces = SQLTraceEventStore(pool=pool)

    try:
        event = traces.insert(
            event_id="turn-end-1",
            account_id=account_id,
            owner_space=owner_space,
            agent_id=agent_id,
            session_id=session_id,
            hook_type="turn_end",
            content={"event_type": "turn_end"},
        )
        finalized = traces.finalize_session(
            account_id=account_id,
            owner_space=owner_space,
            agent_id=agent_id,
            session_id=session_id,
            stop_event_index=event["event_index"],
        )

        assert event["inserted"] is True
        assert finalized["status"] == "finalized"
    finally:
        conn = pool.get_connection()
        try:
            SQLContextFS.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                for table in (
                    "trace_session_events",
                    "trace_sessions",
                    "trace_events",
                ):
                    cur.execute(
                        f"DELETE FROM {table} WHERE account_id = %s",
                        (account_id,),
                    )
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()


def test_dreaming_merge_serializes_inline_relation_writes(dreaming_storage, monkeypatch):
    pool, fs, _, ctx, canonical = dreaming_storage
    source = replace(canonical, uri=canonical.parent_uri + "source")
    peer = replace(canonical, uri=canonical.parent_uri + "peer")
    relation = RelationEdge(peer.uri, source.uri, "related_to", 1.0, "Shared fact")
    writer = SQLContextFS(pool=pool, dreaming_enabled=True)
    relations = SQLRelationStore(pool=pool)
    started = Event()

    def write_relation():
        started.set()
        writer.write_node(replace(peer, metadata={"_expected_version": 1, "_relations": [relation]}), ctx)

    for node in (canonical, source, peer):
        fs.write_node(node, ctx)
    stored = [fs.read_node(node.uri, ctx) for node in (canonical, source)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = []

        class MergeCursor(cursor):
            def execute(self, query, params=None):
                super().execute(query, params)
                if query.startswith("SELECT EXISTS (SELECT 1 FROM relation_edges"):
                    pending.append(executor.submit(write_relation))
                    assert started.wait(5)
                    wait(pending, timeout=1)

        conn = pool.get_connection()
        previous_factory = conn.cursor_factory
        conn.cursor_factory = MergeCursor
        try:
            with monkeypatch.context() as patch:
                patch.setattr(fs, "_get_connection", lambda: conn)
                patch.setattr(fs, "_return_connection", lambda _: None)
                assert fs.merge_dreaming_duplicate(*stored, ctx)
        finally:
            conn.cursor_factory = previous_factory
            pool.return_connection(conn)
        with pytest.raises(NodeNotFoundError):
            pending[0].result(timeout=10)
    assert fs.read_dreaming_alias(source.uri, ctx).uri == canonical.uri
    assert relations.get_edges(peer.uri, ctx) == []


@pytest.fixture
def dreaming_storage(opengauss_sql_dsn):
    account = f"ogmem-dream-{uuid.uuid4().hex[:12]}"
    ctx = RequestContext(account, "alice", "main", "session-1", "trace-1")
    pool = SharedConnectionPool(opengauss_sql_dsn, pool_size=5, dialect="opengauss")
    fs = SQLContextFS(pool=pool, dreaming_enabled=True)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    template = ContextNode(f"ctx://{account}/users/alice/memories/patterns/a-primary", "MEMORY", "pattern",
                           0, ctx.user_space_name(), "Shared fact", "Shared fact", "Shared fact",
                           {"_expected_version": 0, "routing_key": "primary"})
    try:
        yield pool, fs, outbox, ctx, template
    finally:
        conn = pool.get_connection()
        try:
            fs.bind_tenant(conn, account)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM outbox_events WHERE account_id=%s", (account,))
                cur.execute("DELETE FROM context_nodes WHERE account_id=%s", (account,))
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()


def test_dreaming_merge_projects_the_committed_revision(dreaming_storage):
    from index.index_record_builder import build_index_records

    _, fs, outbox, ctx, canonical = dreaming_storage
    canonical.metadata["expected_version"] = 0
    source = replace(canonical, uri=canonical.parent_uri + "z-source")
    for node in (canonical, source):
        fs.write_node(node, ctx)
    stored = [fs.read_node(node.uri, ctx) for node in (canonical, source)]
    assert fs.merge_dreaming_duplicate(*stored, ctx)
    fresh = fs.read_node(canonical.uri, ctx)
    [project] = [event for uri, event in outbox.list_pending(ctx.account_id)
                 if uri == canonical.uri and event.event_type == "UPSERT_CONTEXT"]
    assert fresh.metadata["version"] == 2
    assert {record["metadata"]["canonical_revision"] for record in project.payload["records"]} == {2}
    assert {record.metadata["canonical_revision"] for record in build_index_records(fresh)} == {2}


def test_dreaming_merge_serializes_paged_identity_writes(dreaming_storage, monkeypatch):
    from service.api import MemoryWriteAPI
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    _, fs, _, ctx, canonical = dreaming_storage
    directory = canonical.parent_uri
    source = replace(canonical, uri=directory + "z-source",
                     metadata={**canonical.metadata, "routing_key_aliases": ["old_name"]})
    for node in (canonical, source):
        fs.write_node(node, ctx)
    for i in range(99):
        fs.write_node(replace(canonical, uri=f"{directory}m-{i:03}",
                              metadata={"_expected_version": 0, "routing_key": f"unrelated_{i}"}), ctx)
    stored = [fs.read_node(node.uri, ctx) for node in (canonical, source)]
    started = Event()
    real_page = SQLContextFS.list_node_summaries_page

    def merge():
        started.set()
        try:
            fs.merge_dreaming_duplicate(*stored, ctx)
        except ConcurrentModificationError:
            pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = []

        def first_page_then_merge(self, path, context, *, limit=100, after_uri=None):
            page = real_page(self, path, context, limit=limit, after_uri=after_uri)
            if not pending and limit == 100 and after_uri is None:
                assert len(page) == 100 and page[0].uri == canonical.uri
                assert all(node.uri != source.uri for node in page)
                pending.append(executor.submit(merge))
                assert started.wait(5)
                wait(pending, timeout=1)
            return page

        with monkeypatch.context() as patch:
            patch.setattr(SQLContextFS, "list_node_summaries_page", first_page_then_merge)
            candidate = CandidateMemory("pattern", "user", "old_name", "Added detail", "Added detail",
                                        "Added detail", 0.8)
            result = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(candidate, ctx)
        pending[0].result(timeout=10)
    assert result["target_uri"] in {canonical.uri, source.uri}
