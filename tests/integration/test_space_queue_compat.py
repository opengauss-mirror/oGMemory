"""Queue processing through the configured SQL dialect and real AGFS storage."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from tests.integration.test_space_job_review import memory, publish, snapshot, source_node
from tests.integration.test_space_job_review import pool as pool
from tests.integration.test_space_job_review import workspace as workspace


def test_worker_publishes_and_reclaims_memory(workspace, pool):
    config = os.environ.get("OGMEM_SPACE_TEST_CONFIG")
    if not config:
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects an isolated AGFS instance")
    jobs, ctx = workspace
    settings = json.loads(Path(config).read_text())
    bridge = AGFSSpaceBridge(pool, AGFSClient(settings["agfs_url"]))
    jobs.content_store = bridge
    try:
        node = source_node(ctx, "queue", "Connection pool advice")
        bridge.write_node(node, ctx)
        with jobs.transaction(ctx.account_id) as cur:
            cur.execute("""INSERT INTO outbox_events(event_id,event_type,uri,account_id,status)
                VALUES (%s,'SOURCE_CHANGED',%s,%s,'SOURCE_PENDING')""",
                (str(uuid4()), node.uri, ctx.account_id))
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
