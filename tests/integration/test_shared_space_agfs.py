"""Positive shared-space flows through real openGauss vectors and AGFS bodies."""

import hashlib
import json
import os
import time
from uuid import uuid4

import pytest
from pyagfs import AGFSClient
from werkzeug.test import Client
from werkzeug.wrappers import Response

from core.models import ContextNode, IndexRecord, RequestContext
from providers.embedder.mock_embedder import MockEmbedder
from scripts.shared_space_agfs_qa import (
    create_service,
    load_environment,
    register_members,
    write_source,
)


def wait_for(operation, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = operation()
        if result:
            return result
        time.sleep(0.2)
    raise AssertionError("Shared-space background operation exceeded its QA deadline")


@pytest.fixture
def deployment():
    if not os.environ.get("OGMEM_SPACE_TEST_CONFIG"):
        pytest.skip("OGMEM_SPACE_TEST_CONFIG selects the openGauss + AGFS QA deployment")
    environment = load_environment()
    account = "agfs-test-" + uuid4().hex[:12]
    service = create_service(environment, account)
    state = {"service": service, "environment": environment, "account": account}
    try:
        state["credentials"] = register_members(service, account)
        yield state
    finally:
        state["service"].shutdown(30)


def configure_space(service, account, privacy_terms=()):
    ctx = RequestContext(account, "alice", "", "", uuid4().hex)
    repo = service.get_space_repository()
    repo.create_space(ctx, {"space_id": "ops", "name": "Operations"}, request_id=uuid4().hex)
    repo.set_members(ctx, "ops", {"members": [
        {"principal": "user:alice", "role": "manager"},
        {"principal": "user:bob", "role": "member"},
        {"principal": "agent:ops-agent", "role": "member"},
    ]}, expected_rev=1, request_id=uuid4().hex)
    repo.put_source(ctx, "ops", "ops-source", {
        "principal": "agent:ops-agent", "memory_types": ["case"],
    }, expected_rev=0, request_id=uuid4().hex)
    repo.save_rule(ctx, "ops", "pool", {"name": "Pool troubleshooting", "spec": {
        "memory_types": ["case"],
        "description": "Repeatable connection pool troubleshooting steps.",
        "output_kind": "experience",
        "privacy_policy": {"redact_terms": list(privacy_terms)},
        "trigger_policy": {"debounce_seconds": 0, "min_interval_seconds": 0, "reconcile_seconds": 1},
    }}, expected_rev=0, request_id=uuid4().hex)
    return repo, ctx


def await_item(repo, ctx, content=None, *, manager=None):
    def ready():
        jobs = repo.list_jobs(manager or ctx, "ops")["items"]
        for job in jobs:
            assert job["state"] in {"QUEUED", "RUNNING", "SUCCEEDED", "SUPERSEDED"}, job.get("last_error")
        items = repo.visible_items(ctx, ["ops"])
        return next((item for item in items if content is None or content in item["content"]), None)
    return wait_for(ready)


def verify_agfs(environment, repo, account, uri):
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT metadata,version FROM context_nodes WHERE account_id=%s AND uri=%s", (account, uri))
        catalog = dict(cur.fetchone())
    pointer = catalog["metadata"]["agfs_content"]
    raw = AGFSClient(api_base_url=environment["agfs_url"]).read(pointer["path"])
    assert hashlib.sha256(raw).hexdigest() == pointer["sha256"]
    assert len(raw) == pointer["bytes"]
    payload = json.loads(raw)
    assert payload["uri"] == uri
    return catalog, payload


def test_agfs_source_generation_acl_recompute_restart_and_expiry(deployment):
    service, account = deployment["service"], deployment["account"]
    repo, alice = configure_space(service, account)
    bob = RequestContext(account, "bob", "", "", uuid4().hex)
    original_rule = repo.list_rules(alice, "ops")["items"][0]
    source_ctx = RequestContext(account, "alice", "ops-agent", "", uuid4().hex)
    source = f"ctx://{account}/agents/ops-agent/memories/case/pool"
    receipts = {"qa-write": {"generation": 1, "action": "create"}}
    node = ContextNode(source, "MEMORY", "case", 2, "agent:ops-agent", "Pool exhaustion",
                       "Inspect connection leaks", "Check connection leaks and long transactions.",
                       {"_expected_version": 0, "write_receipts": receipts})
    context_fs = service._get_context_fs()
    event = service._get_outbox_store(context_fs).build_write_event(node)
    event.payload.update(operation_id="qa-" + uuid4().hex, source_generation=1)
    context_fs.write_node_with_outbox(node, source_ctx, event)
    stored = context_fs.read_node(source, source_ctx)
    assert stored.content == node.content
    assert stored.metadata["write_receipts"] == receipts
    assert context_fs.read_node_fields(source, source_ctx, "content").content == node.content
    assert context_fs.read_node_fields(source, source_ctx, "abstract").abstract == node.abstract
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT payload FROM outbox_events WHERE account_id=%s AND event_id=%s", (account, event.event_id))
        persisted = cur.fetchone()["payload"]
        assert persisted["operation_id"] == event.payload["operation_id"]
        assert persisted["source_generation"] == 1
    source_catalog, source_payload = verify_agfs(deployment["environment"], repo, account, source)
    assert source_payload["content"] == "Check connection leaks and long transactions."
    first = await_item(repo, bob, "Check connection leaks", manager=alice)
    catalog, payload = verify_agfs(deployment["environment"], repo, account, first["uri"])
    assert payload["content"] == first["content"]
    assert catalog["metadata"]["indexed_content_rev"] == catalog["version"]
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT id,uri,level,metadata FROM shared_space_qa_vectors WHERE filters->>'account_id'=%s AND filters->>'owner_space'='space:ops'", (account,))
        indexed = cur.fetchall()
        assert {row["level"] for row in indexed} == {0, 2}
        for row in indexed:
            assert row["id"] == IndexRecord.generate_id(row["uri"], row["level"])
            assert row["metadata"]["category"] == "experience"
            assert row["metadata"]["context_type"] == "MEMORY"
    revision = repo.list_spaces(alice, "ops")["space"]["authz_rev"]
    repo.set_acl(alice, "ops", {"item_id": first["item_id"], "mode": "restricted", "readers": ["user:bob"]},
                 expected_rev=revision, request_id=uuid4().hex)
    result = service.search_shared_memory(bob, query="connection pool", space_ids=["ops"])
    assert result["items"][0]["item_id"] == first["item_id"]
    assert result["items"][0]["content"] == payload["content"]

    write_source(service, account, "Check connection leaks. Verify pool recovery after release.")
    updated = await_item(repo, bob, "Verify pool recovery", manager=alice)
    updated_source, updated_payload = verify_agfs(deployment["environment"], repo, account, source)
    assert updated_source["version"] > source_catalog["version"]
    assert "Verify pool recovery" in updated_payload["content"]
    assert updated["acl_origin"] == "manual"
    assert updated["acl_mode"] == "restricted"
    assert repo.permissions(bob, "ops", item_id=updated["item_id"])["can_read"] is True
    revision = repo.list_spaces(alice, "ops")["space"]["authz_rev"]
    repo.set_members(alice, "ops", {"members": [
        {"principal": "user:alice", "role": "manager"},
        {"principal": "user:bob", "role": "manager"},
        {"principal": "agent:ops-agent", "role": "member"},
    ]}, expected_rev=revision, request_id=uuid4().hex)
    rule = repo.list_rules(alice, "ops")["items"][0]
    assert (rule["rule_rev"], rule["spec_hash"], rule["spec"]) == (
        original_rule["rule_rev"], original_rule["spec_hash"], original_rule["spec"])

    service.shutdown(30)
    service = create_service(deployment["environment"], account)
    deployment["service"] = service
    result = service.search_shared_memory(bob, query="pool recovery", space_ids=["ops"])
    assert result["items"][0]["item_id"] == updated["item_id"]
    assert "Verify pool recovery" in result["items"][0]["content"]
    repo = service.get_space_repository()
    with repo.transaction(account) as cur:
        cur.execute("UPDATE space_items SET expires_at=NOW()-INTERVAL '1 second' WHERE account_id=%s AND item_id=%s",
                    (account, updated["item_id"]))
    def await_maintenance(state):
        def maintained():
            repo.maintain_due_items(account)
            items = repo.list_items(alice, "ops")["items"]
            return next((item for item in items
                         if item["item_id"] == updated["item_id"] and item["state"] == state), None)
        return wait_for(maintained)
    assert await_maintenance("EXPIRED")["state"] == "EXPIRED"
    with repo.transaction(account) as cur:
        cur.execute("UPDATE space_items SET purge_after=NOW()-INTERVAL '1 second' WHERE account_id=%s AND item_id=%s",
                    (account, updated["item_id"]))
    client = AGFSClient(api_base_url=deployment["environment"]["agfs_url"])
    parent = f"{deployment['environment'].get('agfs_mount_prefix', '/local')}/accounts/{account}/.space-content"
    directory = hashlib.sha256(updated["uri"].encode()).hexdigest()

    def cleaned():
        repo.maintain_due_items(account)
        with repo.transaction(account, write=False) as cur:
            cur.execute("""SELECT i.state,n.content,n.metadata->>'agfs_purged' AS purged
                FROM space_items i JOIN context_nodes n USING(account_id,uri)
                WHERE i.account_id=%s AND i.item_id=%s""", (account, updated["item_id"]))
            row = cur.fetchone()
        if not row or row["state"] != "PURGED" or row["purged"] != "true":
            return None
        if any(entry["name"] == directory for entry in client.ls(parent)):
            return None
        return row

    assert wait_for(cleaned)["content"] == ""
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT count(*) AS total FROM space_item_sources WHERE account_id=%s AND item_id=%s", (account, updated["item_id"]))
        assert cur.fetchone()["total"] == 1



def test_agfs_text_privacy_delegation_and_system_archive(deployment, monkeypatch):
    service, account = deployment["service"], deployment["account"]
    repo, alice = configure_space(service, account, privacy_terms=["张三"])
    delegated = RequestContext(account, "alice", "ops-agent", "", uuid4().hex)
    source = "客户张三、邮箱zhang@example.com、手机号13812345678。检查连接泄漏并验证连接池恢复。"
    expected = "客户[已遮蔽]、邮箱[邮箱]、手机号[手机号]。检查连接泄漏并验证连接池恢复。"
    abstract = "客户张三的连接池经验，联系zhang@example.com或13812345678。"
    expected_abstract = "客户[已遮蔽]的连接池经验，联系[邮箱]或[手机号]。"
    model = service.get_llm()
    complete_json = model.complete_json
    embedded = []
    model_sources = []
    embed_texts = MockEmbedder.embed_texts

    def summarize(prompt, schema):
        model_sources.extend(source["content"] for source in json.loads(prompt.split("\n", 1)[1])["sources"])
        result = complete_json(prompt, schema)
        for operation in result["operations"]:
            if operation["action"] in {"ADD", "UPDATE"}:
                operation["abstract"] = abstract
        return result

    def record_embedding(embedder, texts):
        embedded.extend(texts)
        return embed_texts(embedder, texts)

    monkeypatch.setattr(model, "complete_json", summarize)
    monkeypatch.setattr(MockEmbedder, "embed_texts", record_embedding)
    write_source(service, account, source)
    item = await_item(repo, alice, "验证连接池恢复")
    assert item["content"] == expected
    assert item["abstract"] == expected_abstract
    catalog, payload = verify_agfs(deployment["environment"], repo, account, item["uri"])
    assert payload["content"] == payload["overview"] == expected
    assert payload["abstract"] == expected_abstract
    assert catalog["metadata"]["privacy_redacted"] is True
    assert expected in model_sources
    assert expected in embedded and expected_abstract in embedded
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT content,abstract,overview FROM context_nodes WHERE account_id=%s AND uri=%s", (account, item["uri"]))
        stored = cur.fetchone()
        assert stored["content"] == stored["overview"] == expected
        assert stored["abstract"] == expected_abstract
        cur.execute("SELECT level,text,metadata FROM shared_space_qa_vectors WHERE filters->>'account_id'=%s AND filters->>'owner_space'='space:ops'", (account,))
        indexed = cur.fetchall()
        assert {row["level"]: row["text"] for row in indexed} == {0: expected_abstract, 2: expected}
        assert all(row["metadata"]["abstract"] == expected_abstract for row in indexed)
    for mode in ("keyword", "semantic", "hybrid"):
        result = service.search_shared_memory(delegated, query="连接池", space_ids=["ops"], mode=mode)
        shared = next(row for row in result["items"] if row["item_id"] == item["item_id"])
        assert shared["content"] == expected
        assert shared["abstract"] == expected_abstract

    import server.app as public
    import server.management_app as gateway
    monkeypatch.setattr(public, "_service", service)
    monkeypatch.setattr(gateway, "_get_service", lambda: service)
    client = Client(gateway.create_app(), Response)
    response = client.get(
        "/management/spaces/ops/items", query_string={"item_id": item["item_id"]},
        headers={"X-API-Key": deployment["credentials"]["bob_key"]})
    assert response.status_code == 200, response.get_json()
    detail = response.get_json()["items"][0]
    assert detail["content"] == detail["overview"] == expected
    assert detail["abstract"] == expected_abstract
    for mode in ("keyword", "semantic", "hybrid"):
        response = client.post("/api/spaces/search", json={"query": "连接池", "space_ids": ["ops"], "mode": mode},
                               headers={"X-API-Key": deployment["credentials"]["bob_key"]})
        assert response.status_code == 200, response.get_json()
        shared = next(row for row in response.get_json()["items"] if row["item_id"] == item["item_id"])
        assert shared["content"] == expected
        assert shared["abstract"] == expected_abstract
    assert repo.list_rules(alice, "ops")["items"][0]["rule_rev"] == 1

    write_source(service, account, "QA_RETIRE: this troubleshooting procedure has been retired.")
    archived = wait_for(lambda: next((row for row in repo.list_items(alice, "ops")["items"]
                                    if row["item_id"] == item["item_id"] and row["state"] == "ARCHIVED"), None))
    assert archived["rule_rev"] == 1
    assert repo.list_rules(alice, "ops")["items"][0]["rule_rev"] == 1


def test_deleted_space_system_archives_and_purges_agfs_memory(deployment):
    service, account = deployment["service"], deployment["account"]
    repo, alice = configure_space(service, account)
    source_text = "Check connection leaks and verify available connections recover."
    write_source(service, account, source_text)
    item = await_item(repo, alice, "Check connection leaks")
    _, published = verify_agfs(deployment["environment"], repo, account, item["uri"])
    assert published["content"] == item["content"]
    revision = repo.list_spaces(alice, "ops")["space"]["authz_rev"]
    repo.delete_space(alice, "ops", expected_rev=revision, request_id=uuid4().hex)

    def item_state(expected):
        with repo.transaction(account, write=False) as cur:
            cur.execute("""SELECT i.state,n.content,n.metadata FROM space_items i
                JOIN context_nodes n USING(account_id,uri) WHERE i.account_id=%s AND i.item_id=%s""",
                        (account, item["item_id"]))
            row = cur.fetchone()
            return dict(row) if row["state"] == expected else None

    archived = wait_for(lambda: item_state("ARCHIVED"))
    assert archived["content"] == item["content"]
    with repo.transaction(account) as cur:
        cur.execute("SELECT state,deleted_at FROM spaces WHERE account_id=%s AND space_id='ops'", (account,))
        deleted = cur.fetchone()
        assert deleted["state"] == "ARCHIVED"
        assert deleted["deleted_at"] is not None
        cur.execute("""UPDATE space_items SET purge_after=NOW()-INTERVAL '1 second'
            WHERE account_id=%s AND item_id=%s""", (account, item["item_id"]))

    def cleaned():
        purged = item_state("PURGED")
        if purged and purged["metadata"].get("agfs_purged"):
            with repo.transaction(account, write=False) as cur:
                cur.execute("""SELECT count(*) AS total FROM shared_space_qa_vectors
                    WHERE filters->>'account_id'=%s AND filters->>'owner_space'='space:ops'""", (account,))
                if cur.fetchone()["total"] == 0:
                    return purged
        return None

    assert wait_for(cleaned)["content"] == ""
    with repo.transaction(account, write=False) as cur:
        cur.execute("SELECT count(*) AS total FROM space_item_sources WHERE account_id=%s AND item_id=%s",
                    (account, item["item_id"]))
        assert cur.fetchone()["total"] == 1
        cur.execute("""SELECT action FROM cp_audit_logs WHERE account_id=%s
            AND details->>'item_id'=%s AND action IN ('space.archived','space.purged')""", (account, item["item_id"]))
        assert {row["action"] for row in cur.fetchall()} == {"space.archived", "space.purged"}
    client = AGFSClient(api_base_url=deployment["environment"]["agfs_url"])
    parent = f"{deployment['environment'].get('agfs_mount_prefix', '/local')}/accounts/{account}/.space-content"
    remaining = [entry for entry in client.ls(parent)
                 if entry["name"] == hashlib.sha256(item["uri"].encode()).hexdigest()]
    assert remaining == []
    source_uri = f"ctx://{account}/agents/ops-agent/memories/case/pool"
    _, source = verify_agfs(deployment["environment"], repo, account, source_uri)
    assert source["content"] == source_text
