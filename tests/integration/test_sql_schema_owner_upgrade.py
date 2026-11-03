"""Canonical-schema upgrades on explicitly configured disposable SQL databases."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, operation_digest
from core.errors import AccessDeniedError, ConcurrentModificationError
from core.models import CandidateMemory, ContextNode, RequestContext
from fs.sql_adapter import schema as schema_module
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from session.session_manager import SessionManager
from session.session_state import SessionState
from session.sql_archive_store import SQLSessionArchiveStore


@pytest.fixture(params=["postgresql", "opengauss"])
def schema_database(request):
    dialect = request.param
    dsn = os.environ.get(f"OGMEM_SCHEMA_{dialect.upper()}_TEST_DSN")
    if not dsn:
        pytest.skip(f"configure OGMEM_SCHEMA_{dialect.upper()}_TEST_DSN for a disposable database")
    name = "schema_upgrade_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    isolated_dsn = psycopg2.extensions.make_dsn(dsn, options=f"-c search_path={name}")
    prior_ensured = set(schema_module._schema_ensured)
    schema_module._schema_ensured.clear()
    pool = SharedConnectionPool(isolated_dsn, pool_size=1, dialect=dialect)
    try:
        yield dialect, name, pool
    finally:
        pool.close_all()
        schema_module._schema_ensured.clear()
        schema_module._schema_ensured.update(prior_ensured)
        with control.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
        control.close()


def reapply_schema(pool, dialect):
    schema_module._schema_ensured.clear()
    SQLContextFS(pool=pool)


def owner_columns(pool, schema_name):
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name, data_type, character_maximum_length, is_nullable, column_default "
                "FROM information_schema.columns WHERE table_schema=%s AND table_name='session_archives' "
                "AND column_name IN ('owner_user_id', 'owner_agent_id') ORDER BY column_name",
                (schema_name,),
            )
            return cursor.fetchall()
    finally:
        conn.rollback()
        pool.return_connection(conn)


def assert_owner_columns(columns):
    assert [row[0] for row in columns] == ["owner_agent_id", "owner_user_id"]
    assert all(row[1:4] == ("character varying", 128, "NO") and "''" in row[4] for row in columns)


def test_new_schema_and_reapplication_keep_owner_column_contract(schema_database):
    dialect, name, pool = schema_database
    SQLContextFS(pool=pool)
    first = owner_columns(pool, name)
    assert_owner_columns(first)
    reapply_schema(pool, dialect)
    assert owner_columns(pool, name) == first


def test_legacy_archive_upgrade_preserves_rows_and_adds_both_owners(schema_database):
    dialect, name, pool = schema_database
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE session_archives (archive_id VARCHAR(64) NOT NULL, "
                "session_id VARCHAR(128) NOT NULL, account_id VARCHAR(128) NOT NULL, "
                "abstract TEXT NOT NULL DEFAULT '', overview TEXT NOT NULL DEFAULT '', "
                "messages JSONB NOT NULL DEFAULT '[]', metadata JSONB NOT NULL DEFAULT '{}', "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "PRIMARY KEY (account_id, session_id, archive_id))"
            )
            cursor.execute(
                "INSERT INTO session_archives (archive_id,session_id,account_id,abstract,messages) "
                "VALUES ('legacy-archive','legacy-session','legacy-account','keep-me','[]')"
            )
        conn.commit()
    finally:
        pool.return_connection(conn)
    SQLSessionArchiveStore(pool=pool)
    assert_owner_columns(owner_columns(pool, name))
    reapply_schema(pool, dialect)
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, "legacy-account")
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT abstract, owner_user_id, owner_agent_id FROM session_archives "
                "WHERE archive_id='legacy-archive'"
            )
            assert cursor.fetchone() == ("keep-me", "", "")
    finally:
        conn.rollback()
        pool.return_connection(conn)


def test_unrelated_ddl_failure_is_not_acknowledged_as_schema_success(schema_database):
    dialect, _, pool = schema_database
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TABLE session_archives (archive_id VARCHAR(64))")
        conn.commit()
    finally:
        pool.return_connection(conn)
    with pytest.raises(RuntimeError) as error:
        SQLContextFS(pool=pool)
    assert isinstance(error.value.__cause__, psycopg2.Error)
    assert error.value.__cause__.pgcode == "42703"
    assert not schema_module._schema_ensured


class NoModel:
    def __getattr__(self, name):
        raise AssertionError(f"SQL schema/component validation must not call a model: {name}")


@pytest.mark.parametrize("user_id,agent_id,category,owner_scope", [
    ("reader", "agent", "event", "user"),
    ("Nora研发", "Agent.Lab+2", "event", "user"),
    ("Nora研发", "Agent.Lab+2", "case", "agent"),
])
def test_normal_schema_public_archive_journal_receipt_outbox_and_session_guard(
    schema_database, user_id, agent_id, category, owner_scope,
):
    _, _, pool = schema_database
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    archive = SQLSessionArchiveStore(pool=pool)
    ctx = RequestContext("Schema-Public-" + uuid4().hex, user_id, agent_id, "session", "trace")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    accepted = CandidateMemory(
        category=category, owner_scope=owner_scope, routing_key="receiver-repair", title="Receiver repair",
        abstract="I repaired the receiver.", overview="I repaired the receiver.",
        content="I repaired the receiver.", confidence=.9, when="2026-09-30", who="Reader",
        write_operation_id="schema-public-operation",
    )
    with fs.session_mutation_lock(uri, ctx) as guard:
        for store in (fs, outbox, archive):
            conn = store._get_connection()
            assert conn is guard.connection
            conn.rollback()
            store._return_connection(conn)
        buffer = manager.get_or_create(ctx.session_id, ctx)
        buffer.add("user", accepted.content, message_id="source-message")
        messages = [{"id": "source-message", "role": "user", "content": accepted.content}]
        result = archive.write_archive(ctx.session_id, "overview", "abstract", messages, ctx, archive_id="source-archive")
        assert result.success, result.error
        assert archive.read_archive(ctx.session_id, "source-archive", ctx).messages == messages
        journal = ExtractionBatchJournal(
            manager, buffer, ctx, list(buffer.messages), archive_id="source-archive", messages=messages,
            session_time=datetime(2026, 9, 30, tzinfo=UTC), session_summary="", tool_stats_text="",
        )
        journal.mutate(lambda batch: batch.update(session_summary="durable-journal"))
        payload = json.loads(fs.read_node(uri, ctx).content)
        assert payload["runtime_extensions"][EXTENSION_KEY]["batch"]["session_summary"] == "durable-journal"
        plan = writer.write_memory(accepted, ctx)
        target = plan["target_uri"]
        stored = fs.read_node(target, ctx)
        expected_owner = ctx.user_space_name() if owner_scope == "user" else ctx.agent_space_name()
        owner_type, owner_id = ("users", ctx.user_id) if owner_scope == "user" else ("agents", ctx.agent_id)
        assert target.startswith(f"ctx://{ctx.account_id}/{owner_type}/{owner_id}/")
        assert stored.owner_space == expected_owner
        assert stored.metadata[RECEIPTS_KEY][accepted.write_operation_id] == operation_digest(accepted, ctx)
        assert writer.write_memory(accepted, ctx)["action"] == "skip"
        assert fs.read_node(target, ctx).metadata["version"] == stored.metadata["version"]
        assert len(outbox.list_pending(ctx.account_id)) == 1
        stale = ContextNode(target, "MEMORY", category, stored.level, stored.owner_space,
                            "stale", "stale", "stale", {"_expected_version": 0})
        with pytest.raises(ConcurrentModificationError):
            fs.write_node(stale, ctx)
        assert fs.read_node(target, ctx).content == stored.content
        assert archive.write_archive(ctx.session_id, "other", "other", messages, ctx,
                                     archive_id="source-archive").success
        assert len(archive.list_archives(ctx.session_id, ctx)) == 1
        assert archive.get_session_owner(ctx.session_id, ctx) == (ctx.user_id, ctx.agent_id)
        guard.ensure_owned()
    restarted = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    restored = restarted.get_or_create(ctx.session_id, ctx)
    assert restored.runtime_extensions[EXTENSION_KEY]["batch"]["session_summary"] == "durable-journal"
    with pytest.raises(AccessDeniedError):
        fs.read_node(target, replace(ctx, account_id=ctx.account_id.lower()))
    owner_field = "user_id" if owner_scope == "user" else "agent_id"
    if getattr(ctx, owner_field).lower() != getattr(ctx, owner_field):
        with pytest.raises(AccessDeniedError):
            fs.read_node(target, replace(ctx, **{owner_field: getattr(ctx, owner_field).lower()}))
    with pytest.raises(PermissionError):
        restarted.get_or_create(ctx.session_id, replace(ctx, user_id="another-reader"))
