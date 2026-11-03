"""Keep reads available and recover acknowledged Agent extraction jobs."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.enums import EventType
from core.lexical import LexicalHit, lexical_record_from_index_record
from core.models import OutboxEvent, RetrievalConfig, TypedQuery
from fs.sql_adapter.node_scanner import SQLCanonicalNodeScanner
from index.outbox_worker import OutboxWorker
from providers.lexical_backfill import LexicalBackfill
from retrieval.owner_scope import expand_owner_space_aliases
from retrieval.seed_retriever import SeedRetriever
from server.agent_memory import trace_memory_event
from tests.integration.test_agent_memory_hybrid import (
    ProjectionStore,
    _indexed_units,
)
from tests.integration.test_agent_memory_hybrid import sql_database as sql_database
from tests.integration.test_agent_memory_hybrid import sql_storage as sql_storage
from tests.integration.test_agent_memory_review import _context, _storage
from tests.unit.server.test_agent_memory import SameEmbedding, finalized_trace


@pytest.mark.parametrize("state", ["pending", "processing", "retrying"])
def test_sql_hybrid_reads_while_extraction_is_unfinished(sql_database, sql_storage, state):
    import psycopg2

    fs, outbox = sql_storage
    ctx, _, _, index, records = _indexed_units()
    scanner = SQLCanonicalNodeScanner(fs._shared_pool)
    projection = ProjectionStore(connection_string=sql_database, table_name="queue_lexical")
    coordinator = LexicalBackfill(scanner, projection)
    scopes = expand_owner_space_aliases([ctx.user_space_name(), ctx.agent_space_name()])
    try:
        assert projection.replace_scopes(
            rebuild_id=projection.begin_rebuild(), account_id=ctx.account_id, owner_spaces=scopes,
            records=[lexical_record_from_index_record(record) for record in records],
        )
        assert coordinator.ensure_ready(ctx, scopes)
        with closing(psycopg2.connect(sql_database)) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT record_id,metadata_json FROM queue_lexical_docs WHERE level=2")
            hits = [LexicalHit(ident, 1.0, json.loads(meta) if isinstance(meta, str) else meta)
                    for ident, meta in cursor.fetchall()]
        lexical = SimpleNamespace(search=lambda **kwargs: hits, supported_filters=projection.supported_filters)
        retriever = SeedRetriever(index, SameEmbedding(), RetrievalConfig(
            hybrid_enabled=True, expand_hierarchy=False,
        ), lexical, coordinator)
        query = TypedQuery("batch cursor", "MEMORY", ["tool"])
        expected = {(hit.uri, hit.metadata["unit_key"]) for hit in retriever.search(query, ctx).initial_candidates}
        event = trace_memory_event(f"ctx://{ctx.account_id}/traces/turn", ctx)
        outbox.register_once(event, ctx)
        if state == "processing":
            assert outbox.try_acquire(event, event.uri, "extractor")
        elif state == "retrying":
            outbox.increment_retry(event, event.uri, datetime.now(UTC) + timedelta(hours=1))
        assert outbox.count_unfinished(ctx.account_id)["total"] == 1
        actual = retriever.search(query, ctx).initial_candidates
        assert {(hit.uri, hit.metadata["unit_key"]) for hit in actual} == expected
        assert {hit.metadata["unit_key"] for hit in actual} == {"batch", "cursor"}
    finally:
        coordinator.close()
        projection.close()


@pytest.mark.parametrize("event_type", [kind.value for kind in EventType if kind != EventType.EXTRACT_AGENT_MEMORY])
def test_index_completion_releases_sql_projection_barrier(sql_storage, event_type):
    fs, outbox = sql_storage
    ctx = _context()
    uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tools/test"
    extraction = trace_memory_event(uri, ctx)
    outbox.register_once(extraction, ctx)
    index_event = OutboxEvent("index-event", event_type, uri, {}, "PENDING")
    outbox.register_once(index_event, ctx)
    scanner = SQLCanonicalNodeScanner(fs._shared_pool)
    assert scanner.projection_is_quiescent(account_id=ctx.account_id) is False
    assert outbox.try_acquire(index_event, uri, "indexer")
    assert scanner.projection_is_quiescent(account_id=ctx.account_id) is False
    outbox.mark_done(index_event, uri)
    assert scanner.projection_is_quiescent(account_id=ctx.account_id)
    assert outbox.count_unfinished(ctx.account_id)["total"] == 1


@pytest.mark.parametrize("recovery", ["register", "claim", "acquire"])
@pytest.mark.parametrize("interrupted_file", ["json", "processing"])
def test_agfs_acknowledgement_cleanup_recovers(agfs_server_info, monkeypatch, recovery, interrupted_file):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    _, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    assert outbox.try_acquire(event, event.uri, "first")
    prefix = outbox._uri_to_agfs_path(event.uri) + ".outbox/" + event.event_id
    remove = client.rm

    def interrupted(path, *args, **kwargs):
        if path == prefix + "." + interrupted_file:
            raise OSError("acknowledgement interrupted")
        return remove(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(client, "rm", interrupted)
        with pytest.raises(OSError, match="acknowledgement interrupted"):
            outbox.mark_done(event, event.uri)
    marker = client.read(outbox._uri_to_agfs_path(event.uri) + ".agent_memory_done")
    _, _, restarted = _storage(agfs_server_info["url"])
    for _ in range(2):
        if recovery == "register":
            restarted.register_once(event, ctx)
        elif recovery == "claim":
            with restarted.claim_agent_event(event, "recovery") as acquired:
                assert acquired is False
        else:
            assert restarted.try_acquire(event, event.uri, "recovery") is False
        assert restarted.count_unfinished(ctx.account_id) == {"pending": 0, "processing": 0, "total": 0}
        names = {entry["name"].rsplit("/", 1)[-1] for entry in client.ls(prefix.rsplit("/", 1)[0])}
        assert names == set()
        assert client.read(outbox._uri_to_agfs_path(event.uri) + ".agent_memory_done") == marker


def test_agfs_worker_cleans_acknowledged_retry(agfs_server_info):
    client, fs, outbox = _storage(agfs_server_info["url"])
    ctx = _context()
    _, turn = finalized_trace(client, ctx)
    event = trace_memory_event(turn["uri"], ctx)
    outbox.register_once(event, ctx)
    assert outbox.try_acquire(event, event.uri, "first")
    node_path = outbox._uri_to_agfs_path(event.uri)
    client.write(node_path + ".agent_memory_done", b'{}')
    client.write(node_path + ".outbox/" + event.event_id + ".processing",
                 json.dumps({"worker_id": "first", "acquired_at": "2000-01-01T00:00:00+00:00"}).encode())
    _, _, restarted = _storage(agfs_server_info["url"])
    worker = OutboxWorker(_indexed_units()[3], SameEmbedding(), fs=fs,
                         agent_memory_handler=lambda job: pytest.fail("completed job was executed again"))
    assert worker.run_once(restarted, [ctx.account_id])["failed"] == 0
    assert restarted.count_unfinished(ctx.account_id)["total"] == 0
