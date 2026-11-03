"""Exercise durable receipt races and guarded extraction on a real SQL store."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, WriteOutcomeUnknownError
from core.models import CandidateMemory, RequestContext
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor
from fs.sql_adapter.session_mutation import SessionMutationLostError, SQLSessionGuard
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from session.session_state import SessionState
from tests.integration.test_sql_schema_owner_upgrade import NoModel
from tests.integration.test_sql_schema_owner_upgrade import schema_database as schema_database


def candidate(category="event"):
    return CandidateMemory(
        category, "agent" if category == "case" else "user", "receiver_repair",
        "Repaired the receiver.", "Repaired the receiver.", "Repaired the receiver.", .9,
        when="2026-10-05", who="Nora", title="Repair", write_operation_id="repair-operation",
    )


@pytest.fixture
def store(schema_database):
    _, _, pool = schema_database
    return SQLContextFS(pool=pool), RequestContext("receipt-race", "reader", "agent", "session", "trace")


@pytest.mark.parametrize("category", ["profile", "entity", "preference", "pattern", "event", "case"])
@pytest.mark.parametrize("atomic", [False, True])
def test_duplicate_operation_receipt_survives_later_update(store, schema_database, category, atomic):
    fs, ctx = store
    outbox = SQLOutboxStore(pool=schema_database[2], fs=fs) if atomic else None
    writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    original = replace(candidate(category), write_operation_id="",
                       provenance_ids=["source-message"] if category in {"event", "case"} else [])
    uri = writer.write_memory(original, ctx)["target_uri"]
    initial = fs.read_node(uri, ctx)
    accepted = replace(original, write_operation_id="duplicate-operation")
    if category == "profile":
        accepted = replace(accepted, target_uri=uri, expected_version=initial.metadata["version"])

    skipped = writer.write_memory(accepted, ctx)
    stored = fs.read_node(uri, ctx)
    assert skipped["action"] == "skip" and skipped["target_uri"] == uri
    assert accepted.write_operation_id in stored.metadata[RECEIPTS_KEY]
    assert (stored.abstract, stored.overview, stored.content) == (
        initial.abstract, initial.overview, initial.content)

    updated = replace(original, target_uri=uri, expected_version=stored.metadata["version"],
                      abstract="Receiver upgraded.", overview="Receiver upgraded.",
                      content="Receiver upgraded.", write_operation_id="later-update")
    writer.write_memory(updated, ctx)
    latest = fs.read_node(uri, ctx)
    pending = outbox.list_pending(ctx.account_id) if outbox else []

    restarted = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    retried = restarted.write_memory(accepted, ctx)
    assert retried["action"] == "skip" and retried["target_uri"] == uri
    assert retried["merged_fields"]["status"] == "operation_committed"
    assert fs.read_node(uri, ctx) == latest
    if outbox:
        assert outbox.list_pending(ctx.account_id) == pending


@pytest.mark.parametrize("bound", [False, True])
def test_receipt_discovery_reads_only_matching_bodies(store, bound, monkeypatch):
    from commit.write_receipts import receipt_plan

    fs, ctx = store
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    original = replace(candidate("entity"), write_operation_id="")
    uri = writer.write_memory(original, ctx)["target_uri"]
    accepted = replace(original, write_operation_id="indexed-operation")
    if bound:
        accepted = replace(accepted, target_uri=uri, expected_version=1)
    writer.write_memory(accepted, ctx)
    node = fs.read_node(uri, ctx)
    directory = uri.rsplit("/", 1)[0]
    for index in range(205):
        fs.write_node(replace(node, uri=f"{directory}/background_{index:03}",
                              metadata={"routing_key": f"background_{index:03}"}), ctx)
    reads = []
    original_read = fs.read_node

    def read(*args):
        reads.append(args[0])
        return original_read(*args)

    monkeypatch.setattr(fs, "read_node", read)
    uris = writer._writer._identity_resolver.uris
    assert receipt_plan(fs, uris, replace(accepted, write_operation_id="new-operation"), ctx) is None
    assert reads == []
    assert receipt_plan(fs, uris, accepted, ctx).target_uri == uri
    assert reads == [uri]


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("sources", [[], ["source-message"]])
def test_concurrent_operation_has_one_committed_receipt(store, schema_database, category, sources, monkeypatch):
    import commit.context_writer as module

    fs, ctx = store
    accepted = replace(candidate(category), provenance_ids=sources)
    outbox = SQLOutboxStore(pool=schema_database[2], fs=fs)
    barrier = threading.Barrier(2)
    original = module.receipt_plan
    local = threading.local()

    def synchronized_lookup(*args):
        result = original(*args)
        if not getattr(local, "checked", False):
            local.checked = True
            barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(module, "receipt_plan", synchronized_lookup)
    writers = [MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(writer.write_memory, accepted, ctx) for writer in writers]
        results = [future.result(timeout=15) for future in futures]
    assert len({result["target_uri"] for result in results}) == 1
    monkeypatch.setattr(module, "receipt_plan", original)
    assert writers[0].write_memory(accepted, ctx)["merged_fields"]["status"] == "operation_committed"
    uri = results[0]["target_uri"]
    assert fs.list_children(uri.rsplit("/", 1)[0], ctx) == [uri]
    assert fs.read_node(uri, ctx).metadata["version"] == 1
    assert len(outbox.list_pending(ctx.account_id)) == 1


@pytest.mark.parametrize("category", ["entity", "event", "case", "tool"])
@pytest.mark.parametrize("atomic", [False, True])
def test_committed_receipt_recovers_after_lock_cleanup_error(store, schema_database, category, atomic, monkeypatch):
    fs, ctx = store
    outbox = SQLOutboxStore(pool=schema_database[2], fs=fs) if atomic else None
    writer = MemoryWriteAPI(fs, NoModel(), outbox, directory_summary_enabled=False)
    accepted = replace(candidate(category), owner_scope="agent" if category in {"case", "tool"} else "user")
    resolver = writer._writer._identity_resolver
    original = resolver._lock

    @contextmanager
    def cleanup_error(*args):
        with original(*args):
            yield
        raise RuntimeError("lock cleanup failed after commit")

    monkeypatch.setattr(resolver, "_lock", cleanup_error)
    result = writer.write_memory(accepted, ctx)
    assert result["merged_fields"]["status"] == "operation_committed"
    assert fs.read_node(result["target_uri"], ctx).metadata["version"] == 1


def test_connection_loss_during_unlock_requires_receipt_recovery(store, monkeypatch):
    fs, ctx = store
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    accepted = candidate("entity")
    original = SQLSessionGuard.call

    def lose_connection(self, function, *args, **kwargs):
        if args and isinstance(args[0], str) and "pg_advisory_unlock" in args[0]:
            self.raw.close()
        return original(self, function, *args, **kwargs)

    monkeypatch.setattr(SQLSessionGuard, "call", lose_connection)
    with pytest.raises(SessionMutationLostError):
        with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            with pytest.raises(WriteOutcomeUnknownError):
                writer.write_memory(accepted, ctx)
    result = writer.write_memory(accepted, ctx)
    assert result["merged_fields"]["status"] == "operation_committed"
    assert accepted.write_operation_id in fs.read_node(result["target_uri"], ctx).metadata[RECEIPTS_KEY]


def test_guarded_lazy_multispan_reads_and_following_write_share_owner_thread(store):
    fs, ctx = store
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    uri = writer.write_memory(candidate("entity"), ctx)["target_uri"]
    owner = threading.get_ident()
    reads = []

    class ReadModel:
        def complete_json(self, prompt, schema):
            return {"spans": [{"start": index, "end": index, "categories": ["entity"]}
                              for index in (0, 8)]}

        def complete_with_tools_messages(self, messages, **kwargs):
            results = [message for message in messages if message["role"] == "tool"]
            if results:
                reads.append((threading.get_ident(), results[-1]["content"]))
                return [], "[]"
            return [{"name": "read", "input": {"uri": uri}, "id": "read-receiver"}], ""

    registry = SchemaRegistry()
    extractor = Extractor(ReadModel(), fs=fs, schema_registry=registry,
                          uri_resolver=URIResolver(registry), mode="lazy")
    messages = [{"role": "user", "content": f"Receiver observation {index}."} for index in range(9)]
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx) as guard:
        assert extractor.extract(messages, ctx) == []
        guard.ensure_owned()
        result = writer.write_memory(replace(candidate(), write_operation_id="following-write"), ctx)
        assert fs.read_node(result["target_uri"], ctx).content == candidate().content
    assert len(reads) == 4
    assert all(thread == owner and "Repaired the receiver." in body for thread, body in reads)
