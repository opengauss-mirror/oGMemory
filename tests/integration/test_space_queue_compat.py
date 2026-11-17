"""Queue processing through the configured SQL dialect and real AGFS storage."""

import json
import os
from pathlib import Path

import pytest
from pyagfs import AGFSClient

from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from server.space_repository import SpaceRepository
from tests.integration.test_space_job_review import memory, publish, snapshot, source_node
from tests.integration.test_space_job_review import pool as pool
from tests.integration.test_space_job_review import workspace as workspace


def test_worker_publishes_and_reclaims_memory(workspace, pool):
    config = os.environ.get("OGMEM_SPACE_TEST_CONFIG")
    if not config:
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects an isolated AGFS instance")
    _, ctx = workspace
    settings = json.loads(Path(config).read_text())
    bridge = AGFSSpaceBridge(pool, AGFSClient(settings["agfs_url"]))
    jobs = SpaceRepository(pool, content_store=bridge)
    try:
        node = source_node(ctx, "queue", "Connection pool advice")
        bridge.write_node(node, ctx)
        dispatched = jobs.dispatch_source_events(ctx.account_id)
        print("request", {"method": "dispatch_source_events", "uri": node.uri}, "response", dispatched)
        assert dispatched == 1
        job, data = snapshot(jobs, ctx.account_id)
        result = publish(jobs, job, data, lambda p: [memory(p, "pool")])
        [item] = result["items"]
        print("request", {"method": "claim_job/complete_job", "action": "ADD"}, "response", result)
        assert item["state"] == "ACTIVE"
        with jobs.transaction(ctx.account_id) as cur:
            cur.execute("""UPDATE space_items SET expires_at=NOW()-INTERVAL '1 day',
                purge_after=NOW()-INTERVAL '1 day' WHERE account_id=%s AND item_id=%s""",
                (ctx.account_id, item["item_id"]))
        for state in ("EXPIRED", "PURGED"):
            assert jobs.maintain_due_items(ctx.account_id) == 1
            with jobs.transaction(ctx.account_id) as cur:
                cur.execute("""SELECT i.state,n.metadata->>'agfs_purged' AS reclaimed
                    FROM space_items i JOIN context_nodes n USING(account_id,uri)
                    WHERE i.account_id=%s AND i.item_id=%s""", (ctx.account_id, item["item_id"]))
                row = dict(cur.fetchone())
            print("request", {"method": "maintain_due_items", "item_id": item["item_id"]}, "response", row)
            assert row["state"] == state
        assert row["reclaimed"] == "true"
    finally:
        bridge.close(10)


def test_agfs_package_update_advances_source_event(workspace, pool):
    config = os.environ.get("OGMEM_SPACE_TEST_CONFIG")
    if not config:
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects an isolated AGFS instance")
    _, ctx = workspace
    settings = json.loads(Path(config).read_text())
    bridge = AGFSSpaceBridge(pool, AGFSClient(settings["agfs_url"]))
    jobs = SpaceRepository(pool, content_store=bridge)
    node = source_node(ctx, "package", "Connection pool advice")
    node.metadata["fields"] = {"steps": "Check active sessions"}
    try:
        bridge.write_node(node, ctx)
        with jobs.transaction(ctx.account_id) as cur:
            cur.execute("SELECT version FROM context_nodes WHERE uri=%s", (node.uri,))
            previous = cur.fetchone()["version"]
        node.metadata["fields"] = {"steps": "Check active sessions and waiting transactions"}
        bridge.write_node(node, ctx)
        with jobs.transaction(ctx.account_id) as cur:
            cur.execute("SELECT version FROM context_nodes WHERE uri=%s", (node.uri,))
            current = cur.fetchone()["version"]
            cur.execute("""SELECT (payload->>'content_rev')::integer AS revision
                FROM outbox_events WHERE account_id=%s AND uri=%s
                AND event_type='SOURCE_CHANGED' ORDER BY seq""", (ctx.account_id, node.uri))
            revisions = [row["revision"] for row in cur.fetchall()]
        print("request", {"method": "write_node", "uri": node.uri, "fields": node.metadata["fields"]},
              "response", {"previous_version": previous, "current_version": current,
                           "source_event_revisions": revisions})
        assert current == previous + 1
        assert revisions == [previous, current]
    finally:
        bridge.close(10)
