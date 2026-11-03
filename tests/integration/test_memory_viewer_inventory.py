"""Inventory counts against the same PostgreSQL ContextFS rows as directory browsing."""

import os
from uuid import uuid4

import pytest

from core.models import RequestContext, Role
from fs.sql_adapter.sql_context_fs import SQLContextFS


def test_sql_memory_inventory_counts_active_direct_children():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required")
    fs = SQLContextFS(connection_string=dsn)
    account = "inventory-" + uuid4().hex
    other = "inventory-" + uuid4().hex
    ctx = RequestContext(account, "alice", "", "test", "test", role=Role.ADMIN)
    rows = [
        (f"ctx://{account}/users/alice/memories/profile", account, "user:alice", "profile", "ACTIVE"),
        (f"ctx://{account}/agents/helper/memories/tools/example", account, "agent:helper", "tool", "ACTIVE"),
        (f"ctx://{account}/users/张 三/memories/events/one", account, "user:张 三", "event", "ACTIVE"),
        (f"ctx://{account}/agents/bot@ops/memories/tools/one", account, "agent:bot@ops", "tool", "ACTIVE"),
        (f"ctx://{account}/users/alice/memories/events/old", account, "user:alice", "event", "ARCHIVED"),
        (f"ctx://{account}/users/alice/memories/events/nested/child", account, "user:alice", "event", "ACTIVE"),
        (f"ctx://{other}/users/alice/memories/events/foreign", other, "user:alice", "event", "ACTIVE"),
    ]
    rows.extend(
        (f"ctx://{account}/users/alice/memories/events/item-{i:03}", account,
         "user:alice", "event", "ACTIVE")
        for i in range(125)
    )
    conn = fs._get_connection()
    try:
        for tenant in (account, other):
            fs.bind_tenant(conn, tenant)
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO context_nodes (uri, account_id, owner_space, category, context_type, status) "
                    "VALUES (%s, %s, %s, %s, 'memory', %s)",
                    [row for row in rows if row[1] == tenant],
                )
            conn.commit()

        counts = fs.count_memory_children(ctx)
        assert counts[f"ctx://{account}/users/alice/memories/profile"] == 1
        assert counts[f"ctx://{account}/users/alice/memories/events"] == 125
        assert counts[f"ctx://{account}/agents/helper/memories/tools"] == 1
        assert counts[f"ctx://{account}/users/张 三/memories/events"] == 1
        assert counts[f"ctx://{account}/agents/bot@ops/memories/tools"] == 1
        assert all(other not in uri for uri in counts)
    finally:
        for tenant in (account, other):
            fs.bind_tenant(conn, tenant)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM context_nodes WHERE account_id = %s", (tenant,))
            conn.commit()
        fs._return_connection(conn)
