"""Canonical scene storage contracts against an isolated PostgreSQL database.

Set SCENE_TEST_ADMIN_DSN to a local test server role with CREATEDB permission.
Each test owns and removes a uniquely named database.
"""

import json
import os
from dataclasses import replace
from threading import Event, Thread
from uuid import uuid4

import pytest

from core.models import TypedQuery
from fs.sql_adapter.sql_context_fs import SQLContextFS
from scene.adapters import CanonicalSceneAdapter, _scope_context
from scene.config import SceneConfig
from session.archive_store import SessionArchiveStore
from session.sql_archive_store import SQLSessionArchiveStore
from tests.unit.scene.test_adapters import Embedder, IndexedVectors, Repository, node, scene

psycopg2 = pytest.importorskip("psycopg2")
sql = pytest.importorskip("psycopg2.sql")
make_dsn = pytest.importorskip("psycopg2.extensions").make_dsn

pytestmark = pytest.mark.integration
CTX = _scope_context("acme", "user:alice")


@pytest.fixture
def database():
    admin_dsn = os.getenv("SCENE_TEST_ADMIN_DSN")
    if not admin_dsn:
        pytest.skip("SCENE_TEST_ADMIN_DSN is required")
    name = "scene_contract_" + uuid4().hex
    admin = psycopg2.connect(admin_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_dsn(admin_dsn, dbname=name)
    finally:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


@pytest.mark.parametrize("store_type", ["archives", "canonical"])
def test_sql_archive_identity_selects_current_owner(database, store_type):
    fs = SQLContextFS(database)
    store = SQLSessionArchiveStore(database) if store_type == "archives" else SessionArchiveStore(fs)
    for ctx in (CTX, replace(CTX, account_id="other"), replace(CTX, user_id="bob")):
        result = store.write_archive(
            ctx.user_id, "Planning", "Friday release",
            [{"id": "m1", "role": "user", "content": f"{ctx.account_id}/{ctx.user_id}"}],
            ctx, archive_id="release",
        )
        assert result.success, result.error
    entry = store.read_archive_by_id("release", CTX)
    assert entry.messages[0]["content"] == "acme/alice"
    assert entry.session_id == "alice"
    source = CanonicalSceneAdapter(Repository(), fs, IndexedVectors([]), Embedder(), archive_store=store)
    from core.provenance_resolver import ProvenanceResolver

    reference = ProvenanceResolver.build_id("archive", "release", ["m1"])
    card = source.read_source_context("acme", "user:alice", [reference], 1)[0]
    assert card["text"] == "[m1]  user: acme/alice\n"


@pytest.mark.parametrize("filter_name", ["category", "directory", "target"])
def test_sql_scene_candidates_fill_filtered_query(database, filter_name):
    fs = SQLContextFS(database)
    preference, event = node("style", category="preference"), node("release")
    for item in (preference, event):
        fs.write_node(item, CTX)
    repository = Repository([scene("a", [preference]), scene("b", [event], centroid=(0.8, 0.6))])
    source = CanonicalSceneAdapter(repository, fs, IndexedVectors([preference, event]), Embedder())
    query = TypedQuery("release", "MEMORY", ["event"] if filter_name == "category" else [],
                       memory_directories=["events"] if filter_name == "directory" else None,
                       target_uri=event.uri if filter_name == "target" else None)
    hits = source.search(query.text, CTX, limit=1, query_vector=[1, 0], typed_query=query)
    members = source.resolve_members(hits, CTX, limit=1, query_vector=[1, 0], typed_query=query)
    assert [member.uri for member in members] == [event.uri]
    assert hits[0].scene_id == "b"


@pytest.mark.parametrize("store_type", ["memory", "archives", "canonical"])
@pytest.mark.parametrize("fragment", ['"', "\x01"])
def test_sql_source_card_preserves_budgeted_evidence(database, store_type, fragment):
    fs = SQLContextFS(database)
    content = fragment * 2000
    store = None
    if store_type == "memory":
        item = replace(node("source"), content=content)
        fs.write_node(item, CTX)
        reference, text = item.uri, content
    else:
        store = SQLSessionArchiveStore(database) if store_type == "archives" else SessionArchiveStore(fs)
        saved = store.write_archive(
            "planning", "Plan", "Plan", [{"id": "m1", "role": "user", "content": content}],
            CTX, archive_id="source",
        )
        assert saved.success, saved.error
        reference, text = saved.uri, f"[m1]  user: {content}\n"
    source = CanonicalSceneAdapter(
        Repository(), fs, IndexedVectors([]), Embedder(), archive_store=store,
        config=replace(SceneConfig(), candidate_card_budget_tokens=256),
    )
    cards = source.read_source_context("acme", "user:alice", [reference], 1)
    assert len(cards) == 1
    card = cards[0]
    assert fragment in card["text"]
    assert card["text"] == text[:len(card["text"])]
    assert len(json.dumps(card, ensure_ascii=False)) <= 1024
    extended = {**card, "text": text[:len(card["text"]) + 1]}
    assert len(json.dumps(extended, ensure_ascii=False)) > 1024


class TransactionalRepository(Repository):
    """Contract backend using a SQL lock shared with actual canonical writes."""

    def __init__(self, dsn):
        super().__init__()
        self.dsn = dsn
        self.before_commit = lambda: None
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE scene_publication (revision integer, source_version integer)")

    def compare_and_swap_checked(self, expected, replacement, check):
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            SQLContextFS.bind_tenant(conn, expected.account_id)
            cur.execute("LOCK TABLE context_nodes IN SHARE MODE")
            cur.execute("LOCK TABLE scene_publication IN EXCLUSIVE MODE")
            if not check(lambda uri, ctx: SQLContextFS.read_node_with_cursor(cur, uri, ctx)):
                return False
            self.before_commit()
            if not super().compare_and_swap(expected, replacement):
                return False
            cur.execute("INSERT INTO scene_publication VALUES (%s, %s)", (
                replacement.revision, replacement.scenes[0].cluster.members[0].node_version,
            ))
        return True


def test_sql_publication_and_canonical_writer_serialize(database):
    fs, writer_fs = SQLContextFS(database), SQLContextFS(database)
    item = node("release")
    fs.write_node(item, CTX)
    repository = TransactionalRepository(database)
    source = CanonicalSceneAdapter(repository, fs, IndexedVectors([item]), Embedder())
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri])
    updated = replace(item, content="Updated", metadata={"expected_version": 1})
    checked, connected, attempted = Event(), Event(), Event()
    errors, writer_pid = [], []
    acquire = writer_fs._get_connection

    def writer_connection():
        conn = acquire()
        writer_pid.append(conn.get_backend_pid())
        connected.set()
        return conn

    writer_fs._get_connection = writer_connection

    def writer():
        try:
            assert checked.wait(10)
            attempted.set()
            writer_fs.write_node(updated, CTX)
        except Exception as exc:
            errors.append(exc)

    def observe_writer_wait():
        checked.set()
        assert attempted.wait(10) and connected.wait(10)
        with psycopg2.connect(database) as observer, observer.cursor() as cur:
            # SQL lock observation makes the concurrent-write schedule deterministic.
            for _ in range(1000):
                cur.execute("SELECT wait_event_type FROM pg_stat_activity WHERE pid = %s", (writer_pid[0],))
                waiting = cur.fetchone()
                if waiting and waiting[0] == "Lock":
                    break
                cur.execute("SELECT pg_stat_clear_snapshot(), pg_sleep(0.01)")
            assert waiting[0] == "Lock"
        assert fs.read_node(item.uri, CTX).metadata["version"] == 1

    repository.before_commit = observe_writer_wait
    worker = Thread(target=writer)
    worker.start()
    try:
        record = scene("release", batch.nodes)
        assert source.compare_and_swap(batch.snapshot, replace(batch.snapshot, revision=1, scenes=(record,)))
    finally:
        checked.set()
        worker.join(15)
    assert errors == []
    assert fs.read_node(item.uri, CTX).metadata["version"] == 2
    with psycopg2.connect(database) as conn, conn.cursor() as cur:
        cur.execute("SELECT revision, source_version FROM scene_publication")
        assert cur.fetchall() == [(1, 1)]
    repository.before_commit = lambda: None
    for _ in range(2):
        if source.compare_and_swap(batch.snapshot, replace(batch.snapshot, revision=2, scenes=(record,))):
            break
        batch = source.load_input("acme", "user:alice", node_uris=[item.uri])
        record = replace(record, cluster=replace(record.cluster, members=(
            replace(record.cluster.members[0], node_version=batch.nodes[0].metadata["version"]),
        )))
    with psycopg2.connect(database) as conn, conn.cursor() as cur:
        cur.execute("SELECT source_version FROM scene_publication ORDER BY revision DESC LIMIT 1")
        assert cur.fetchone() == (2,)
