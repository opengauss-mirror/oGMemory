"""Original-case owner namespaces and receipts in a disposable PostgreSQL DB."""

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError
from core.errors import AccessDeniedError
from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from session.sql_archive_store import SQLSessionArchiveStore
from tests.integration.test_occurrence_replay_sql import close_stores, stores
from tests.integration.test_session_mutation_sql import (
    test_union_typed_partial_retry_fixed_source_new_tail_and_owning_connection as _verify_lifecycle_recovery,
)
from tests.integration.test_sql_guard_native_reads import (
    test_guarded_real_native_multispan_reads_preserve_owner_and_tool_protocol as _verify_native_reads,
)
from tests.integration.test_sql_guard_native_reads import (
    test_public_guarded_source_receipt_restart_does_not_reextract_or_duplicate as _verify_source_recovery,
)
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_occurrence_replay import occurrence
from tests.unit.commit.test_profile_merge import profile
from tests.unit.server.test_extraction_batch_recovery import NoModel


def cleanup(fs, accounts):
    """Delete only this test's fresh UUID tenants, including their outbox rows."""
    conn = fs._get_connection()
    try:
        with conn.cursor() as cur:
            for account in accounts:
                fs.bind_tenant(conn, account)
                for table in ("relation_edges", "outbox_events", "session_archives", "context_nodes"):
                    cur.execute(f"DELETE FROM {table} WHERE account_id = %s", (account,))
        conn.commit()
    finally:
        fs._return_connection(conn)


@pytest.mark.parametrize("category", ["profile", "entity", "preference", "pattern", "event", "case", "skill", "tool"])
def test_original_case_receipt_reconnect_preserves_owner_and_stale_cas(category):
    account = "Owner-Uri-" + uuid4().hex
    ctx = RequestContext(account, "Nora研发", "Agent.Lab+2", "session", "trace")
    fs, outbox = stores()
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        first = profile("identity", "Initial supported fact.") if category == "profile" else memory(
            "supported_topic", category=category, content="Initial supported fact.")
        first = replace(first, owner_scope="agent" if category in {"case", "skill", "tool"} else "user")
        uri = writer.write_memory(first, ctx)["target_uri"]
        accepted = replace(first, target_uri=uri, expected_version=1,
                           content="Initial supported fact. Added fact.", write_operation_id="accepted-operation")
        assert writer.write_memory(accepted, ctx)["action"] == "merge"
        later = replace(accepted, expected_version=2, write_operation_id="later-operation",
                        content="Initial supported fact. Added fact. Later fact.")
        assert writer.write_memory(later, ctx)["action"] == "merge"
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
        before = fs.read_node(uri, ctx)
        assert before.owner_space == (ctx.agent_space_name() if first.owner_scope == "agent" else ctx.user_space_name())
        assert before.metadata["version"] == 3
        assert set(before.metadata[RECEIPTS_KEY]) == {"accepted-operation", "later-operation"}
        assert writer.write_memory(accepted, replace(ctx, trace_id="retry"))["action"] == "skip"
        with pytest.raises(WriteNotStartedError):
            writer.write_memory(replace(accepted, write_operation_id="stale-new-operation"), ctx)
        assert fs.read_node(uri, ctx) == before
        assert len(outbox.list_pending(account)) == 3
        with pytest.raises(AccessDeniedError):
            fs.read_node(uri, replace(ctx, account_id=account.lower()))
        owner_field = "agent_id" if first.owner_scope == "agent" else "user_id"
        with pytest.raises(AccessDeniedError):
            fs.read_node(uri, replace(ctx, **{owner_field: getattr(ctx, owner_field).lower()}))
    finally:
        cleanup(fs, [account])
        close_stores(fs, outbox)


@pytest.mark.parametrize("category,field", [
    ("event", "account_id"), ("case", "account_id"), ("event", "user_id"), ("case", "agent_id"),
])
def test_case_distinct_principals_have_separate_sql_draft_replay_and_receipts(category, field):
    fs, outbox = stores()
    upper = RequestContext("Owner-Uri-" + uuid4().hex, "Nora", "Agent", "session", "trace")
    lower = replace(upper, **{field: getattr(upper, field).lower()})
    candidate = occurrence(category)
    targets = []
    accounts = {upper.account_id, lower.account_id}
    try:
        for ctx in (upper, lower):
            writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
            created = writer.write_memory(candidate, ctx)
            replay = writer.write_memory(candidate, replace(ctx, session_id="retry"))
            assert created["action"] == "create" and replay["action"] == "skip"
            assert created["target_uri"] == replay["target_uri"]
            targets.append(created["target_uri"])
            accepted = replace(candidate, target_uri=created["target_uri"], expected_version=1,
                               content=candidate.content + " Repair approved.", write_operation_id="same-operation")
            assert writer.write_memory(accepted, ctx)["action"] == "merge"
            directory = URIResolver(SchemaRegistry()).get_directory_uri(category, ctx)
            assert fs.list_children(directory, ctx) == [created["target_uri"]]
        assert targets[0] != targets[1]
        assert len(outbox.list_pending(upper.account_id)) == (2 if field == "account_id" else 4)
    finally:
        cleanup(fs, accounts)
        close_stores(fs, outbox)


@pytest.fixture
def guarded_owner_database():
    """Existing real SQL setup, with new original-case Unicode owner identities."""
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    ctx = RequestContext("Owner-Union-" + uuid4().hex, "Nora研发", "Agent.Lab+2", "session", "trace")
    created = []

    def make():
        pool = SharedConnectionPool(dsn, pool_size=1)
        fs = SQLContextFS(pool=pool)
        outbox = SQLOutboxStore(pool=pool, fs=fs)
        archive = SQLSessionArchiveStore(pool=pool)
        created.append((pool, fs))
        return fs, outbox, archive

    yield dsn, make, ctx
    try:
        cleanup(created[-1][1], [ctx.account_id])
    finally:
        for pool, _ in created:
            pool.close_all()


def native_owner_nodes(make, ctx):
    fs, outbox, archive = make()
    directory = URIResolver(SchemaRegistry()).get_directory_uri("entity", ctx)
    uris = {}
    for key in ("alpha", "beta"):
        uri = f"{directory}/{key}.md"
        fs.write_node(ContextNode(
            uri=uri, context_type="MEMORY", category="entity", level=2,
            owner_space=ctx.user_space_name(), abstract=f"Device {key} is uncalibrated.",
            overview=f"Device {key} on the test bench is uncalibrated.",
            content=f"Device {key} is uncalibrated.",
            metadata={"routing_key": key, "object_name": f"Device {key}", "scope_ref": "Test bench",
                      "owner_scope": "user", "confidence": .9},
        ), ctx)
        uris[key] = uri
    return fs, outbox, archive, ctx, uris


@pytest.mark.parametrize("mode", ["lazy", "eager"])
def test_original_case_sql_guard_actual_extractor_preserves_native_call_budget(guarded_owner_database, mode):
    _, make, ctx = guarded_owner_database
    _verify_native_reads(native_owner_nodes(make, ctx), mode, "commit_session")


def test_original_case_sql_guard_source_receipt_reconnect_preserves_accepted_identity(guarded_owner_database):
    _, make, ctx = guarded_owner_database
    _verify_source_recovery(native_owner_nodes(make, ctx))


@pytest.mark.parametrize("entry", ["compact", "dispose"])
def test_original_case_lifecycle_guard_retry_keeps_fixed_batch_and_tail(guarded_owner_database, entry):
    dsn, make, ctx = guarded_owner_database
    _verify_lifecycle_recovery((dsn, make), ctx, entry)
