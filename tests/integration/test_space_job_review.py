"""Runtime checks for team job repair and bridged AGFS deletion."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from psycopg2.extras import Json, RealDictCursor
from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, RequestContext
from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from server.space_job_repository import SpaceJobRepository
from server.space_rules import digest, validate_rule
from service.space_lifecycle import generate_plans


class Jobs(SpaceJobRepository):
    @contextmanager
    def transaction(self, account_id, space_id=None, *, write=True):
        conn = self.pool.get_connection()
        try:
            with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT set_config('app.account_id', %s, true)", (account_id,))
                yield cur
        finally:
            self.pool.return_connection(conn)


@pytest.fixture(scope="module")
def pool():
    dsn = os.environ.get("OGMEM_SPACE_TEST_DSN")
    if not dsn:
        pytest.skip("OGMEM_SPACE_TEST_DSN requires a disposable database")
    pool = SharedConnectionPool(dsn, pool_size=4, dialect="auto")
    SQLContextFS(pool=pool)
    conn = pool.get_connection()
    try:
        with conn, conn.cursor() as cur:
            ddl = Path("server/space_schema.sql").read_text(encoding="utf-8")
            cur.execute(pool.sql_dialect.prepare_schema(ddl))
    finally:
        pool.return_connection(conn)
    yield pool
    pool.close_all()


def save_rule(jobs, account, rev, **overrides):
    spec = validate_rule({"description": "Summarize pool advice", "memory_types": ["case"],
                          "trigger_policy": {"min_interval_seconds": 0}, **overrides})
    with jobs.transaction(account) as cur:
        cur.execute("""INSERT INTO space_rule_revisions(account_id,space_id,name,state,rule_id,rule_rev,spec,spec_hash)
                       VALUES (%s,'ops','Pool','ACTIVE','pool',%s,%s,%s)""",
                    (account, rev, Json(spec), digest(spec)))
        cur.execute("UPDATE space_rules SET rule_rev=%s WHERE account_id=%s", (rev, account))


@pytest.fixture
def workspace(pool):
    jobs = Jobs()
    jobs.pool = pool
    account = "space-review-" + uuid4().hex[:12]
    with jobs.transaction(account) as cur:
        cur.execute("INSERT INTO cp_accounts(account_id) VALUES (%s)", (account,))
        cur.execute("INSERT INTO cp_users(account_id,user_id,key) VALUES (%s,'alice','test')", (account,))
        cur.execute("INSERT INTO spaces(account_id,space_id,name,owner_user_id) VALUES (%s,'ops','Ops','alice')", (account,))
        cur.execute("INSERT INTO space_members(account_id,space_id,principal,user_id) VALUES (%s,'ops','user:alice','alice')", (account,))
        cur.execute("""INSERT INTO space_source_bindings(account_id,space_id,binding_id,principal,authorized_by,memory_types)
                       VALUES (%s,'ops','alice','user:alice','alice','["case"]')""", (account,))
        cur.execute("INSERT INTO space_rules(account_id,space_id,rule_id,name) VALUES (%s,'ops','pool','Pool')", (account,))
        cur.execute("INSERT INTO space_rule_runtime(account_id,space_id,rule_id) VALUES (%s,'ops','pool')", (account,))
    save_rule(jobs, account, 1)
    return jobs, RequestContext(account, "alice", "", "", "space-review")


def source_node(ctx, name, content):
    return ContextNode(f"ctx://{ctx.account_id}/users/alice/memories/case/{name}",
                       "MEMORY", "case", 2, "user:alice", "Pool advice", "", content)


def snapshot(jobs, account):
    with jobs.transaction(account) as cur:
        cur.execute("UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s", (account,))
    jobs.reconcile_due_rules(account)
    job = jobs.claim_job(account)
    return job, jobs.job_snapshot(job)


def publish(jobs, job, data, operations):
    class Model:
        def complete_json(self, prompt, schema):
            return {"operations": operations(json.loads(prompt.split("\n", 1)[1]))}

    plans = generate_plans(Model(), data)
    return jobs.complete_job(job, plans)


def memory(payload, topic, **target):
    source = payload["sources"][0]
    return {"action": "ADD", "topic": topic, "abstract": "Pool advice", "content": source["content"],
            "quality_score": 0.9, "source_uris": [source["uri"]], **target}


def test_keep_advances_to_new_source_without_resetting_lifetime(workspace, pool):
    jobs, ctx = workspace
    storage = SQLContextFS(pool=pool)
    first = source_node(ctx, "first", "existing notes")
    first.metadata["shared_source_epoch"] = "a" * 32
    storage.write_node(first, ctx)
    job, data = snapshot(jobs, ctx.account_id)
    [published] = publish(jobs, job, data, lambda p: [memory(p, "pool")])["items"]
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("SELECT created_at,expires_at FROM space_items WHERE item_id=%s", (published["item_id"],))
        lifetime = dict(cur.fetchone())
    save_rule(jobs, ctx.account_id, 2)
    added = source_node(ctx, "second", "new notes")
    storage.write_node(added, ctx)
    job, data = snapshot(jobs, ctx.account_id)
    keep = [{"action": "KEEP", "item_id": published["item_id"]}]
    result = publish(jobs, job, data, lambda p: keep)
    _, following = snapshot(jobs, ctx.account_id)
    observed = [row["uri"] for row in following["sources"]]
    print("rule_rev=2", keep, result, "next_sources", observed)
    assert observed == [added.uri]
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("SELECT created_at,expires_at FROM space_items WHERE item_id=%s", (published["item_id"],))
        assert dict(cur.fetchone()) == lifetime


def test_maintenance_accepts_fractional_minimum_age(workspace, pool):
    jobs, ctx = workspace
    save_rule(jobs, ctx.account_id, 2, maintenance_policy={"min_age_days": 0.5})
    SQLContextFS(pool=pool).write_node(source_node(ctx, "first", "existing notes"), ctx)
    job, data = snapshot(jobs, ctx.account_id)
    publish(jobs, job, data, lambda p: [memory(p, "pool")])
    print("request", {"min_age_days": 0.5, "operation": "maintain_due_items"}, flush=True)
    result = jobs.maintain_due_items(ctx.account_id)
    print("response", result)
    assert result == 0


@pytest.mark.parametrize("job_state", ["QUEUED", "RUNNING"])
def test_maintenance_reaches_idle_rule(workspace, pool, job_state):
    jobs, ctx = workspace
    SQLContextFS(pool=pool).write_node(source_node(ctx, "first", "pool notes"), ctx)
    job, data = snapshot(jobs, ctx.account_id)
    items = publish(jobs, job, data, lambda p: [memory(p, topic) for topic in ("busy", "idle")])["items"]
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO space_rules(account_id,space_id,rule_id,name)
            VALUES (%s,'ops','idle','Idle')""", (ctx.account_id,))
        cur.execute("""INSERT INTO space_rule_revisions(account_id,space_id,rule_id,rule_rev,name,state,spec,spec_hash)
            SELECT account_id,space_id,'idle',rule_rev,'Idle',state,spec,spec_hash
            FROM space_rule_revisions WHERE account_id=%s AND rule_id='pool'""", (ctx.account_id,))
        cur.execute("UPDATE space_items SET rule_id='idle' WHERE account_id=%s AND item_id=%s",
                    (ctx.account_id, items[1]["item_id"]))
        cur.execute("""UPDATE space_items SET expires_at=NOW()-INTERVAL '1 day',
            updated_at=NOW()-INTERVAL '2 days' WHERE account_id=%s""", (ctx.account_id,))
        cur.execute("""UPDATE space_items SET updated_at=NOW()-INTERVAL '3 days'
            WHERE account_id=%s AND item_id=%s""", (ctx.account_id, items[0]["item_id"]))
        cur.execute("UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s", (ctx.account_id,))
    assert jobs.reconcile_due_rules(ctx.account_id) == 1
    if job_state == "RUNNING":
        assert jobs.claim_job(ctx.account_id)

    assert jobs.maintain_due_items(ctx.account_id, limit=1) == 1
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("SELECT item_id,state FROM space_items WHERE account_id=%s", (ctx.account_id,))
        states = {row["item_id"]: row["state"] for row in cur.fetchall()}
    assert states == {items[0]["item_id"]: "ACTIVE", items[1]["item_id"]: "EXPIRED"}


def test_same_uri_recreated_with_version_one_reenters_job(workspace, pool):
    jobs, ctx = workspace
    storage = SQLContextFS(pool=pool)
    node = source_node(ctx, "same", "first version")
    storage.write_node(node, ctx)
    job, data = snapshot(jobs, ctx.account_id)
    publish(jobs, job, data, lambda p: [memory(p, "pool")])
    old = data["sources"][0]
    storage.delete_node(node.uri, ctx)
    storage.write_node(source_node(ctx, "same", "recreated version"), ctx)
    job, data = snapshot(jobs, ctx.account_id)
    observed = [(row["uri"], row["content_rev"], row["content"]) for row in data["sources"]]
    print("request", {"uri": node.uri, "previous_version": old["content_rev"], "replacement": "recreated version"},
          "response", observed, "target_count", len(data["items"]))
    assert observed == [(node.uri, old["content_rev"], "recreated version")]
    assert len(data["items"]) == 1


def test_source_overwrite_schedules_existing_item_for_repair(workspace, pool):
    jobs, ctx = workspace
    storage = SQLContextFS(pool=pool)
    original = source_node(ctx, "rewrite", "first version")
    storage.write_node(original, ctx)
    job, data = snapshot(jobs, ctx.account_id)
    prior_version = data["sources"][0]["content_rev"]
    [item] = publish(jobs, job, data, lambda p: [memory(p, "rewrite")])["items"]

    updated = source_node(ctx, "rewrite", "edited without CAS")
    storage.write_node(updated, ctx)
    _, data = snapshot(jobs, ctx.account_id)

    [source] = data["sources"]
    assert (source["uri"], source["content_rev"], source["content"]) == (
        original.uri, prior_version + 1, updated.content,
    )
    assert [target["item_id"] for target in data["items"]] == [item["item_id"]]


def test_reusing_existing_output_supersedes_both_old_targets(workspace, pool):
    jobs, ctx = workspace
    SQLContextFS(pool=pool).write_node(source_node(ctx, "first", "pool notes"), ctx)
    job, data = snapshot(jobs, ctx.account_id)
    originals = publish(jobs, job, data, lambda p: [memory(p, t) for t in ("wait", "leak")])["items"]
    save_rule(jobs, ctx.account_id, 2)
    job, data = snapshot(jobs, ctx.account_id)
    result = publish(jobs, job, data, lambda p: [memory(p, "merged", action="UPDATE", item_id=i["item_id"])
                                                 for i in p["existing"]])["items"]
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("SELECT item_id,state,superseded_by FROM space_items WHERE account_id=%s ORDER BY item_id",
                    (ctx.account_id,))
        rows = {row["item_id"]: (row["state"], row["superseded_by"]) for row in cur.fetchall()}
    target = result[0]["item_id"]
    print("request", {"action": "UPDATE", "old_targets": [x["item_id"] for x in originals],
                      "topic": "merged"}, "response", rows)
    assert all(rows[row["item_id"]] == ("SUPERSEDED", target) for row in originals)


@pytest.mark.parametrize("rewrite", [False, True], ids=["delete-moved", "rewrite-moved"])
def test_deleted_agfs_body_drains_after_last_moved_reference(workspace, pool, monkeypatch, rewrite):
    config = os.environ.get("OGMEM_SPACE_TEST_CONFIG")
    if not config:
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects an isolated AGFS instance")
    jobs, ctx = workspace
    settings = json.loads(Path(config).read_text())
    client = AGFSClient(settings["agfs_url"])
    storage = AGFSSpaceBridge(pool, client)
    jobs.content_store = storage
    try:
        node = source_node(ctx, "first", "original notes")
        storage.write_node(node, ctx)
        first_path = storage.read_node(node.uri, ctx).metadata["agfs_content"]["path"]
        storage.write_node(source_node(ctx, "first", "revised notes"), ctx)
        second_path = storage.read_node(node.uri, ctx).metadata["agfs_content"]["path"]
        moved = source_node(ctx, "moved", "").uri
        storage.move_node(node.uri, moved, ctx)
        storage.write_node(source_node(ctx, "first", "recreated notes"), ctx)
        storage.delete_node(node.uri, ctx)
        jobs.maintain_due_items(ctx.account_id)
        assert storage.read_node(moved, ctx).content == "revised notes"
        if rewrite:
            storage.write_node(source_node(ctx, "moved", "new body at destination"), ctx)
        else:
            event = SQLOutboxStore.build_delete_event(moved)
            assert storage.delete_nodes_with_outbox([moved], ctx, [event]) == 1

        def unavailable(*args, **kwargs):
            raise AGFSClientError("temporary AGFS failure")

        with monkeypatch.context() as patch:
            patch.setattr(client, "rm", unavailable)
            jobs.maintain_due_items(ctx.account_id)
        with jobs.transaction(ctx.account_id) as cur:
            cur.execute("""UPDATE outbox_events SET next_retry_at=NOW()
                WHERE account_id=%s AND event_type='RECLAIM_AGFS' AND next_retry_at>NOW()""", (ctx.account_id,))
            assert cur.rowcount == 1
        jobs.maintain_due_items(ctx.account_id)
        readable = []
        for path in (first_path, second_path):
            try:
                client.read(path)
                readable.append(path)
            except AGFSClientError as exc:
                assert storage._missing(exc)
        print("request", {"method": "write_node" if rewrite else "delete_nodes_with_outbox", "uri": moved},
              "retry", "AGFS_remaining", len(readable))
        assert readable == []
        if rewrite:
            assert storage.read_node(moved, ctx).content == "new body at destination"
    finally:
        storage.close(10)


@pytest.mark.parametrize("origin,initial,updated", [
    ("rule", {"mode": "inherit"}, {"mode": "restricted", "readers": ["user:alice"]}),
    ("rule", {"mode": "restricted", "readers": ["user:alice"]},
     {"mode": "restricted", "readers": ["user:bob"]}),
    ("rule", {"mode": "restricted", "readers": ["user:alice"]}, {"mode": "inherit"}),
    ("manual", {"mode": "restricted", "readers": ["user:alice"]}, {"mode": "inherit"}),
])
def test_keep_applies_rule_acl_and_preserves_manual_acl(workspace, pool, origin, initial, updated):
    jobs, ctx = workspace
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("INSERT INTO cp_users(account_id,user_id,key) VALUES (%s,'bob','test')", (ctx.account_id,))
        cur.execute("""INSERT INTO space_members(account_id,space_id,principal,user_id)
            VALUES (%s,'ops','user:bob','bob')""", (ctx.account_id,))
    save_rule(jobs, ctx.account_id, 2, output_acl=initial)
    SQLContextFS(pool=pool).write_node(source_node(ctx, "first", "pool notes"), ctx)
    job, data = snapshot(jobs, ctx.account_id)
    [item] = publish(jobs, job, data, lambda p: [memory(p, "pool")])["items"]
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET acl_origin=%s WHERE account_id=%s AND item_id=%s",
                    (origin, ctx.account_id, item["item_id"]))
        cur.execute("SELECT created_at,expires_at FROM space_items WHERE account_id=%s AND item_id=%s",
                    (ctx.account_id, item["item_id"]))
        lifetime = dict(cur.fetchone())
    save_rule(jobs, ctx.account_id, 3, output_acl=updated)
    job, data = snapshot(jobs, ctx.account_id)
    request = [{"action": "KEEP", "item_id": item["item_id"]}]
    result = publish(jobs, job, data, lambda p: request)
    with jobs.transaction(ctx.account_id) as cur:
        cur.execute("""SELECT acl_mode,acl_origin,checked_rule_rev,i.created_at,expires_at,n.content
            FROM space_items i JOIN context_nodes n USING(account_id,uri)
            WHERE account_id=%s AND item_id=%s""", (ctx.account_id, item["item_id"]))
        row = dict(cur.fetchone())
        cur.execute("""SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s
            ORDER BY principal""", (ctx.account_id, item["item_id"]))
        readers = [entry["principal"] for entry in cur.fetchall()]
    print("request", {"output_acl": updated, "operations": request, "acl_origin": origin},
          "response", {"result": result, "acl": row, "readers": readers})
    expected = initial if origin == "manual" else updated
    assert (row["acl_mode"], row["acl_origin"], readers) == (
        expected["mode"], origin, expected.get("readers", []))
    assert row["checked_rule_rev"] == 3
    assert {key: row[key] for key in lifetime} == lifetime
    assert row["content"] == "pool notes"
