"""Real SQL claims serialize normal projection, handoff, and request scopes."""

import os
from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import IndexRecord, OutboxEvent, RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from index.outbox_worker import create_upsert_event


@pytest.fixture
def sql_runtime():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("Set OGMEM_IDENTITY_TEST_DSN to the isolated PostgreSQL database")
    import psycopg2
    from psycopg2 import sql

    namespace = "outbox_review_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    dsn = psycopg2.extensions.make_dsn(dsn, options=f"-c search_path={namespace}")
    pools = []

    def store():
        pool = SharedConnectionPool(dsn, pool_size=2)
        pools.append(pool)
        return SQLOutboxStore(pool=pool)

    ctx = RequestContext("outbox-owner-" + uuid4().hex, "reader", "agent", "session", "trace")
    try:
        yield ctx, store
    finally:
        for pool in pools:
            pool.close_all()
        with control.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))
        control.close()


def publish(store, ctx, name, text):
    uri = f"ctx://{ctx.account_id}/users/reader/memories/entities/{name}"
    record = IndexRecord(IndexRecord.generate_id(uri, 0), uri, 0, text,
                         {"account_id": ctx.account_id, "owner_space": "user:reader"}, {})
    event = create_upsert_event(uri, [record])
    store._insert_event(event, ctx.account_id)
    return event, record


def test_agent_claim_keeps_checkpoint_and_completion_with_same_ticket(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    uri = f"ctx://{ctx.account_id}/agents/agent/memories/agent_extraction/job"
    event = OutboxEvent(uuid4().hex, "EXTRACT_AGENT_MEMORY", uri,
                        {"context": vars(ctx), "cursor": 0}, "PENDING")
    store._insert_event(event, ctx.account_id)
    claimed = next(item for _, item in store.claim_batch("agent-owner") if item.event_id == event.event_id)
    with store.claim_agent_event(claimed, "agent-owner") as acquired:
        assert acquired
        claimed.payload["cursor"] = 1
        store.save_payload(claimed)
        assert store.get_event(ctx.account_id, event.event_id).payload["cursor"] == 1
        store.mark_done(claimed, uri)
    assert store.get_event(ctx.account_id, event.event_id).status == "DONE"


def test_individual_claim_release_and_completion_share_uri_lock(sql_runtime):
    ctx, make = sql_runtime
    store = make()
    event, _ = publish(store, ctx, "individual", "Individual projection")
    with store.claim_processing_event(event, event.uri, "first-owner") as acquired:
        assert acquired
        with store.processing_lock(event.uri):
            store.ensure_processing_owned(event, event.uri)
        store.release(event, event.uri)
    with store.claim_processing_event(event, event.uri, "next-owner") as acquired:
        assert acquired
        store.mark_done(event, event.uri)
    assert store.get_event(ctx.account_id, event.event_id).status == "DONE"
