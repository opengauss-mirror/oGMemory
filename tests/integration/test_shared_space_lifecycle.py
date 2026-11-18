"""Positive shared-space lifecycle scenarios against a dedicated SQL database."""
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pyagfs import AGFSClient

from commit.sql_outbox_store import SQLOutboxStore
from core.models import ContextNode, IndexRecord, OutboxEvent, RequestContext
from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
from fs.sql_adapter.pool import SharedConnectionPool
from index.outbox_worker import OutboxWorker
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.space_repository import SpaceRepository
from service.shared_memory import SharedMemoryRetrievalService
from service.space_lifecycle import SpaceMemoryWorker


class EvidenceLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, schema):
        self.calls += 1
        data = json.loads(prompt.split("\n", 1)[1])
        operation = {"action": "ADD", "topic": "connection-pool", "quality_score": 0.95,
                     "abstract": "Database connection pool troubleshooting",
                     "source_uris": [source["uri"] for source in data["sources"]],
                     "content": "\n\n".join(source["content"] for source in data["sources"])}
        if data["existing"]:
            operation.update(action="UPDATE", item_id=data["existing"][0]["item_id"])
        return {"operations": [operation]}


@pytest.fixture(scope="module")
def space_pool():
    dsn = os.environ.get("OGMEM_SPACE_TEST_DSN")
    if not dsn:
        pytest.skip("OGMEM_SPACE_TEST_DSN selects the dedicated SQL test database")
    pool = SharedConnectionPool(dsn, pool_size=8, dialect="auto")
    SQLOutboxStore(pool=pool)
    SpaceRepository(pool)
    yield pool
    pool.close_all()


@pytest.fixture
def shared_space(space_pool):
    pool = space_pool
    repo = SpaceRepository(pool, initialize=False)
    account = "lifecycle-" + uuid4().hex[:12]
    ctx = RequestContext(account, "alice", "", "", "lifecycle-test")
    with repo.transaction(account) as cur:
        cur.execute("INSERT INTO cp_accounts(account_id) VALUES(%s)", (account,))
        cur.execute("INSERT INTO cp_users(account_id,user_id,role,key) VALUES(%s,'alice','user',%s),(%s,'bob','user',%s)",
                    (account, uuid4().hex, account, uuid4().hex))
        cur.execute("INSERT INTO cp_agents(account_id,agent_id,owner_user_id) VALUES(%s,'ops-agent','alice')", (account,))
    repo.create_space(ctx, {"space_id": "ops", "name": "Operations"}, request_id=uuid4().hex)
    repo.set_members(ctx, "ops", {"members": [
        {"principal": "user:alice", "role": "manager"}, {"principal": "user:bob", "role": "member"},
        {"principal": "agent:ops-agent", "role": "member"}]}, expected_rev=1, request_id=uuid4().hex)
    repo.put_source(ctx, "ops", "ops-source", {"principal": "agent:ops-agent", "memory_types": ["case"]},
                    expected_rev=0, request_id=uuid4().hex)
    spec = {"memory_types": ["case"],
            "description": "Repeatable connection pool troubleshooting steps.",
            "output_kind": "experience", "privacy_policy": {"redact_terms": []},
            "trigger_policy": {"debounce_seconds": 0, "min_interval_seconds": 0, "reconcile_seconds": 1}}
    saved = repo.save_rule(ctx, "ops", "pool", {"name": "Pool troubleshooting", "spec": spec}, expected_rev=0,
                           request_id=uuid4().hex)
    source_uri = f"ctx://{account}/agents/ops-agent/memories/case/pool"
    with repo.transaction(account) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:ops-agent','case','MEMORY','Check connection leaks and long running transactions')""", (source_uri, account))
    llm = EvidenceLLM()
    worker = SpaceMemoryWorker(repo, llm, worker_count=1)
    yield repo, ctx, source_uri, worker, llm, saved


@pytest.fixture(params=["sql", "agfs"])
def published_space(shared_space, request):
    repo = shared_space[0]
    if request.param == "agfs":
        path = os.environ.get("OGMEM_SPACE_TEST_CONFIG")
        if not path:
            pytest.skip("OGMEM_SPACE_TEST_CONFIG selects the AGFS content deployment")
        environment = json.loads(Path(path).read_text(encoding="utf8"))
        repo.content_store = AGFSSpaceBridge(repo.pool, AGFSClient(api_base_url=environment["agfs_url"]),
                                            environment.get("agfs_mount_prefix", "/local"))
    try:
        yield shared_space
    finally:
        if repo.content_store:
            repo.content_store.close(10)


def project_items(repo, ctx):
    store = SQLOutboxStore(pool=repo.pool)
    index = InMemoryVectorIndex()
    projector = OutboxWorker(index, MockEmbedder(), directory_summary_enabled=False)
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT event_id,event_type,uri,payload,status FROM outbox_events WHERE account_id=%s AND event_type='UPSERT_CONTEXT' AND status='PENDING'", (ctx.account_id,))
        events = [OutboxEvent(**dict(row)) for row in cur.fetchall()]
    for event in events:
        result = projector.process_event(event)
        assert result.success, result.error_message
        assert result.records_upserted == 2
        store.mark_done(event, event.uri)
    return index


def run_rule(repo, ctx, worker):
    repo.dispatch_source_events(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s", (ctx.account_id,))
    repo.reconcile_due_rules(ctx.account_id)
    job = repo.claim_job(ctx.account_id)
    assert job is not None
    worker._execute(job)
    jobs = repo.list_jobs(ctx, "ops")["items"]
    completed = next(item for item in jobs if item["job_id"] == job["job_id"])
    assert completed["state"] == "SUCCEEDED", completed.get("last_error")
    project_items(repo, ctx)
    return completed


def test_source_to_shared_retrieval_and_stable_rule(shared_space):
    repo, ctx, uri, worker, llm, saved = shared_space
    completed = run_rule(repo, ctx, worker)
    assert len(completed["result"]["items"]) == 1
    bob = RequestContext(ctx.account_id, "bob", "", "", "bob-query")
    result = SharedMemoryRetrievalService(repo).search(bob, "connection", ["ops"])
    assert result["total"] == 1
    assert "connection leaks" in result["items"][0]["content"]
    rule = repo.list_rules(ctx, "ops")["items"][0]
    assert rule["rule_rev"] == 1
    assert rule["spec"]["description"] == "Repeatable connection pool troubleshooting steps."
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT metadata FROM context_nodes WHERE account_id=%s AND owner_space='space:ops'", (ctx.account_id,))
        nodes = cur.fetchall()
        assert len(nodes) == 1
        assert nodes[0]["metadata"]["shared_projection_ready"] is True
        cur.execute("SELECT COUNT(*) AS n FROM space_item_sources WHERE account_id=%s AND source_uri=%s", (ctx.account_id, uri))
        assert cur.fetchone()["n"] == 1


def test_keyword_pages_chinese_items_and_filters_kind(shared_space):
    repo, ctx, source_uri, worker, _, _ = shared_space

    class CatalogLLM:
        def complete_json(self, prompt, schema):
            data = json.loads(prompt.split('\n', 1)[1])
            common = {'action': 'ADD', 'quality_score': 0.95,
                      'source_uris': [source['uri'] for source in data['sources']]}
            return {'operations': [{**common, 'topic': f'pool-{number}', 'abstract': f'数据库运维经验{number}',
                'content': f'排查连接池的第{number}类等待，释放连接并验证恢复。' * 12}
                for number in range(3)]}

    worker.llm = CatalogLLM()
    with repo.transaction(ctx.account_id) as cur:
        cur.execute('UPDATE context_nodes SET content=%s,version=version+1 WHERE uri=%s',
                    ('排查连接池的长事务和连接泄漏，释放连接并验证等待恢复。', source_uri))
    repo.dispatch_source_events(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute('UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s', (ctx.account_id,))
    repo.reconcile_due_rules(ctx.account_id)
    job = repo.claim_job(ctx.account_id)
    assert job is not None
    worker._execute(job)
    assert next(row for row in repo.list_jobs(ctx, 'ops')['items']
                if row['job_id'] == job['job_id'])['state'] == 'SUCCEEDED'
    project_items(repo, ctx)
    bob = RequestContext(ctx.account_id, 'bob', '', '', 'keyword-pages')
    search = SharedMemoryRetrievalService(repo)
    base = {'query': '连接池', 'space_ids': ['ops'], 'mode': 'keyword', 'top_k': 100}
    all_kinds = search.search(bob, **base, limit=10)
    assert all_kinds['total'] == 3
    assert all(item['kind'] == 'experience' for item in all_kinds['items'])
    listed, list_cursor = [], None
    for _ in range(3):
        page = repo.list_items(bob, 'ops', limit=1, cursor=list_cursor)
        listed.extend(item['item_id'] for item in page['items'])
        list_cursor = page['next_cursor']
    assert list_cursor is None
    assert listed == sorted(item['item_id'] for item in all_kinds['items'])
    first = search.search(bob, **base, kind='experience', limit=2)
    second = search.search(bob, **base, kind='experience', limit=2, cursor=first['next_cursor'])
    assert first['total'] == second['total'] == 3
    assert len(first['items']) == 2 and len(second['items']) == 1
    assert second['next_cursor'] is None
    complete = first['items'] + second['items']
    assert len({item['item_id'] for item in complete}) == 3
    assert all(item['kind'] == 'experience' and '连接池' in item['content'] for item in complete)
    filtered = search.search(bob, **{**base, 'top_k': 1}, kind='experience', limit=1)
    assert len(filtered['items']) == 1 and filtered['items'][0]['kind'] == 'experience'
    traversed, cursor = [], None
    for _ in range(3):
        page = search.search(bob, **base, kind='experience', limit=2, cursor=cursor, max_context_tokens=128)
        traversed.extend(item['item_id'] for item in page['items'])
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert cursor is None
    assert traversed == [item['item_id'] for item in complete]


def test_team_definition_uses_new_authorized_sources_without_revision_change(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    original = repo.list_rules(ctx, 'ops')['items'][0]
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("INSERT INTO cp_agents(account_id,agent_id,owner_user_id) VALUES(%s,'db-agent','alice')", (ctx.account_id,))
    team = repo.list_spaces(ctx, 'ops')['space']
    members = [{'principal': row['principal'], 'role': row['role']} for row in team['members']]
    members.append({'principal': 'agent:db-agent', 'role': 'member'})
    repo.set_members(ctx, 'ops', {'members': members}, expected_rev=team['authz_rev'])
    repo.put_source(ctx, 'ops', 'db-source', {'principal': 'agent:db-agent', 'memory_types': ['case']}, expected_rev=0)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:db-agent','case','MEMORY','Measure acquisition latency after pool recovery')""",
            (f'ctx://{ctx.account_id}/agents/db-agent/memories/case/recovery', ctx.account_id))
    run_rule(repo, ctx, worker)
    bob = RequestContext(ctx.account_id, 'bob', '', '', 'team-definition')
    current = repo.list_rules(bob, 'ops')['items'][0]
    assert current['spec'] == original['spec']
    assert current['spec_hash'] == original['spec_hash']
    assert current['rule_rev'] == original['rule_rev']
    memories = repo.visible_items(bob, ['ops'])
    assert sum(item['source_count'] for item in memories) == 2
    assert any('Measure acquisition latency' in item['content'] for item in memories)


def test_incremental_merge_preserves_rule_and_keep_expiration(shared_space):
    repo, ctx, uri, worker, llm, _ = shared_space
    run_rule(repo, ctx, worker)
    first = repo.visible_items(ctx, ["ops"])[0]
    run_rule(repo, ctx, worker)
    kept = repo.visible_items(ctx, ["ops"])[0]
    assert kept["item_id"] == first["item_id"]
    assert kept["expires_at"] == first["expires_at"]
    assert llm.calls == 1
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE context_nodes SET content=content||' Verify pool recovery after releasing leaked connections',version=version+1,updated_at=NOW() WHERE uri=%s", (uri,))
    run_rule(repo, ctx, worker)
    updated = repo.visible_items(ctx, ["ops"])[0]
    assert "Verify pool recovery" in updated["content"]
    assert updated["rule_rev"] == 1
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT state,superseded_by FROM space_items WHERE account_id=%s AND item_id=%s", (ctx.account_id, first["item_id"]))
        previous = cur.fetchone()
        assert previous == {"state": "SUPERSEDED", "superseded_by": updated["item_id"]}


def test_system_expiration_and_purge_keep_provenance(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    item = repo.visible_items(ctx, ["ops"])[0]
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET expires_at=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    assert repo.maintain_due_items(ctx.account_id) == 1
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT state,purge_after FROM space_items WHERE account_id=%s AND item_id=%s", (ctx.account_id, item["item_id"]))
        expired = cur.fetchone()
        assert expired["state"] == "EXPIRED"
        assert expired["purge_after"] > datetime.now(UTC)
        cur.execute("UPDATE space_items SET purge_after=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    assert repo.maintain_due_items(ctx.account_id) == 1
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT i.state,n.content FROM space_items i JOIN context_nodes n USING(account_id,uri) WHERE i.account_id=%s", (ctx.account_id,))
        assert cur.fetchone() == {"state": "PURGED", "content": ""}
        cur.execute("SELECT COUNT(*) AS n FROM space_item_sources WHERE account_id=%s", (ctx.account_id,))
        assert cur.fetchone()["n"] == 1


@pytest.mark.parametrize("manual_acl", [False, True])
def test_restored_source_content_publishes_current_version_and_preserves_readers(published_space, manual_acl):
    repo, ctx, source_uri, worker, _, _ = published_space
    versions = []
    bob = RequestContext(ctx.account_id, "bob", "", "", "restored-source")
    for content in ("Connection pool procedure A", "Connection pool procedure B",
                    "Connection pool procedure A", "Connection pool procedure B"):
        if repo.content_store:
            node = ContextNode(source_uri, "MEMORY", "case", 2, "agent:ops-agent", "Connection pool",
                               "Troubleshooting", content)
            repo.content_store.write_node(node, RequestContext(ctx.account_id, "alice", "ops-agent", "", uuid4().hex))
        else:
            with repo.transaction(ctx.account_id) as cur:
                cur.execute("UPDATE context_nodes SET content=%s,version=version+1 WHERE account_id=%s AND uri=%s",
                            (content, ctx.account_id, source_uri))
        completed = run_rule(repo, ctx, worker)
        current = repo.visible_items(ctx, ["ops"])
        assert len(current) == 1
        assert current[0]["content"] == content
        assert completed["result"]["items"] == [{"item_id": current[0]["item_id"],
            "uri": current[0]["uri"], "state": "ACTIVE"}]
        if manual_acl and not versions:
            revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
            repo.set_acl(ctx, "ops", {"item_id": current[0]["item_id"], "mode": "restricted",
                "readers": ["user:bob"]}, expected_rev=revision)
        visible = SharedMemoryRetrievalService(repo).search(bob, "Connection", ["ops"])["items"]
        assert [item["content"] for item in visible] == [content]
        with repo.transaction(ctx.account_id, write=False) as cur:
            cur.execute("SELECT source_content_rev FROM space_item_sources WHERE account_id=%s AND item_id=%s",
                        (ctx.account_id, current[0]["item_id"]))
            contribution = cur.fetchone()
            cur.execute("SELECT version FROM context_nodes WHERE account_id=%s AND uri=%s",
                        (ctx.account_id, source_uri))
            assert contribution["source_content_rev"] == cur.fetchone()["version"]
            if manual_acl:
                cur.execute("SELECT acl_mode,acl_origin FROM space_items WHERE account_id=%s AND item_id=%s",
                            (ctx.account_id, current[0]["item_id"]))
                assert dict(cur.fetchone()) == {"acl_mode": "restricted", "acl_origin": "manual"}
                cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s",
                            (ctx.account_id, current[0]["item_id"]))
                assert cur.fetchall() == [{"principal": "user:bob"}]
            if versions:
                cur.execute("SELECT state,superseded_by FROM space_items WHERE account_id=%s AND item_id=%s",
                            (ctx.account_id, versions[-1]))
                assert dict(cur.fetchone()) == {"state": "SUPERSEDED", "superseded_by": current[0]["item_id"]}
        versions.append(current[0]["item_id"])
    assert len(set(versions)) == len(versions)


def test_text_privacy_generation_and_delegated_retrieval(shared_space):
    repo, ctx, source_uri, worker, _, _ = shared_space
    rule = repo.list_rules(ctx, "ops")["items"][0]
    spec = rule["spec"]
    spec["description"] = "保留连接池故障处理经验，客户联系方式和姓名脱敏。"
    spec["privacy_policy"] = {"redact_terms": ["张三"]}
    repo.save_rule(ctx, "ops", "pool", {"name": "客户排障复盘", "spec": spec}, expected_rev=1, request_id=uuid4().hex)
    source = "客户张三、邮箱zhang@example.com、手机号13812345678。检查连接泄漏并验证连接池恢复。"
    expected = "客户[已遮蔽]、邮箱[邮箱]、手机号[手机号]。检查连接泄漏并验证连接池恢复。"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE context_nodes SET content=%s,version=version+1 WHERE uri=%s", (source, source_uri))

    class PrivacyLLM(EvidenceLLM):
        def complete_json(self, prompt, schema):
            result = super().complete_json(prompt, schema)
            result["operations"][0].update(abstract=source, topic="客户张三的连接池经验")
            return result

    worker.llm = PrivacyLLM()
    run_rule(repo, ctx, worker)
    delegated = RequestContext(ctx.account_id, "alice", "ops-agent", "", "delegated-query")
    result = SharedMemoryRetrievalService(repo).search(delegated, "连接池", ["ops"], kind="experience")
    assert result["items"][0]["content"] == expected
    assert result["items"][0]["abstract"] == expected
    row = repo.visible_items(ctx, ["ops"])[0]
    assert row["content"] == row["abstract"] == expected
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT content,abstract,metadata FROM context_nodes WHERE account_id=%s AND uri=%s", (ctx.account_id, row["uri"]))
        stored = cur.fetchone()
    assert stored["content"] == stored["abstract"] == expected
    assert stored["metadata"]["topic"] == "客户[已遮蔽]的连接池经验"
    assert stored["metadata"]["privacy_redacted"] is True


def test_resource_read_policy_survives_system_replacement(shared_space):
    repo, ctx, uri, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    first = repo.visible_items(ctx, ["ops"])[0]
    revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
    repo.set_acl(ctx, "ops", {"item_id": first["item_id"], "mode": "restricted", "readers": ["user:bob"]},
                 expected_rev=revision, request_id=uuid4().hex)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE context_nodes SET content=content||' Measure available connections',version=version+1 WHERE uri=%s", (uri,))
    run_rule(repo, ctx, worker)
    bob = RequestContext(ctx.account_id, "bob", "", "", "restricted-query")
    item = repo.visible_items(bob, ["ops"])[0]
    assert item["acl_origin"] == "manual"
    assert item["acl_mode"] == "restricted"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s", (ctx.account_id, item["item_id"]))
        assert cur.fetchall() == [{"principal": "user:bob"}]


def test_task_lease_recovery_retains_single_publication(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    repo.reconcile_due_rules(ctx.account_id)
    original = repo.claim_job(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_jobs SET lease_until=NOW()-INTERVAL '1 second' WHERE account_id=%s AND job_id=%s", (ctx.account_id, original["job_id"]))
    recovered = repo.claim_job(ctx.account_id)
    assert recovered["job_id"] == original["job_id"]
    assert recovered["attempts"] == 2
    assert recovered["claim_token"] != original["claim_token"]
    worker._execute(recovered)
    project_items(repo, ctx)
    assert len(repo.visible_items(ctx, ["ops"])) == 1
    jobs = repo.list_jobs(ctx, "ops")["items"]
    assert jobs[0]["state"] == "SUCCEEDED"


def test_source_pages_process_101_inputs_and_preserve_all_outputs(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    with repo.transaction(ctx.account_id) as cur:
        for index in range(100):
            uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/example-{index:03}"
            cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
                VALUES(%s,%s,'agent:ops-agent','case','MEMORY',%s)""",
                (uri, ctx.account_id, f"Connection pool troubleshooting evidence {index}"))
    run_rule(repo, ctx, worker)
    run_rule(repo, ctx, worker)
    items = repo.visible_items(ctx, ["ops"])
    assert len(items) == 2
    assert sum(item["source_count"] for item in items) == 101
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT COUNT(DISTINCT source_uri) AS count FROM space_item_sources WHERE account_id=%s", (ctx.account_id,))
        assert cur.fetchone()["count"] == 101
        cur.execute("SELECT source_after FROM space_rule_runtime WHERE account_id=%s", (ctx.account_id,))
        assert cur.fetchone()["source_after"] == ""


def test_renewed_source_grant_refreshes_existing_provenance(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    original = repo.visible_items(ctx, ["ops"])[0]
    repo.put_source(ctx, "ops", "ops-source", {"principal": "agent:ops-agent", "memory_types": ["case"]},
                    expected_rev=1, request_id=uuid4().hex)
    run_rule(repo, ctx, worker)
    refreshed = repo.visible_items(ctx, ["ops"])[0]
    assert refreshed["item_id"] == original["item_id"]
    assert refreshed["expires_at"] == original["expires_at"]
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT grant_rev FROM space_item_sources WHERE account_id=%s", (ctx.account_id,))
        assert cur.fetchone()["grant_rev"] == 2


def test_purged_manual_policy_is_inherited_by_new_evidence(shared_space):
    repo, ctx, uri, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    first = repo.visible_items(ctx, ["ops"])[0]
    revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
    repo.set_acl(ctx, "ops", {"item_id": first["item_id"], "mode": "restricted", "readers": ["user:bob"]},
                 expected_rev=revision, request_id=uuid4().hex)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET expires_at=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    repo.maintain_due_items(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET purge_after=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    repo.maintain_due_items(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE context_nodes SET content=content||' Measure pool wait time',version=version+1 WHERE uri=%s", (uri,))
    run_rule(repo, ctx, worker)
    successor = repo.visible_items(ctx, ["ops"])[0]
    assert successor["acl_origin"] == "manual"
    assert successor["acl_mode"] == "restricted"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s", (ctx.account_id, successor["item_id"]))
        assert cur.fetchall() == [{"principal": "user:bob"}]
        cur.execute("SELECT state,superseded_by FROM space_items WHERE account_id=%s AND item_id=%s", (ctx.account_id, first["item_id"]))
        assert cur.fetchone() == {"state": "PURGED", "superseded_by": successor["item_id"]}


def test_source_contribution_change_recomputes_from_remaining_evidence(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    repo.put_source(ctx, "ops", "alice-events", {"principal": "user:alice", "memory_types": ["event"]},
                    expected_rev=0, request_id=uuid4().hex)
    rule = repo.list_rules(ctx, "ops")["items"][0]
    spec = rule["spec"]
    spec["memory_types"].append("event")
    repo.save_rule(ctx, "ops", "pool", {"name": "Pool troubleshooting", "spec": spec}, expected_rev=1,
                   request_id=uuid4().hex)
    event_uri = f"ctx://{ctx.account_id}/users/alice/memories/event/connection"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'user:alice','event','MEMORY','Measure connection wait time after pool recovery')""", (event_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    current = repo.visible_items(ctx, ["ops"])[0]
    assert current["source_count"] == 2
    repo.put_source(ctx, "ops", "ops-source", {"principal": "agent:ops-agent", "memory_types": ["case"], "state": "PAUSED"},
                    expected_rev=1, request_id=uuid4().hex)
    run_rule(repo, ctx, worker)
    recomputed = repo.visible_items(ctx, ["ops"])[0]
    assert recomputed["content"] == "Measure connection wait time after pool recovery"
    assert recomputed["source_count"] == 1
    assert repo.list_rules(ctx, "ops")["items"][0]["spec"] == spec
    audit = repo.list_audit(ctx, "ops")["items"]
    assert any(entry["action"] == "space.generate" for entry in audit)


def test_new_source_window_inherits_purged_topic_read_policy(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    original = repo.visible_items(ctx, ["ops"])[0]
    revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
    repo.set_acl(ctx, "ops", {"item_id": original["item_id"], "mode": "restricted", "readers": ["user:bob"]},
                 expected_rev=revision, request_id=uuid4().hex)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET expires_at=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    repo.maintain_due_items(ctx.account_id)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_items SET purge_after=NOW()-INTERVAL '1 second' WHERE account_id=%s", (ctx.account_id,))
    repo.maintain_due_items(ctx.account_id)
    new_uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/new-pool-observation"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:ops-agent','case','MEMORY','Measure pool wait time and verify connection recovery')""", (new_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    current = repo.visible_items(ctx, ["ops"])[0]
    assert current["source_count"] == 1
    assert current["acl_mode"] == "restricted"
    assert current["acl_origin"] == "manual"
    assert current["metadata"]["source_window"] != original["metadata"]["source_window"]
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s", (ctx.account_id, current["item_id"]))
        assert cur.fetchall() == [{"principal": "user:bob"}]
        cur.execute("SELECT state FROM space_items WHERE account_id=%s AND item_id=%s", (ctx.account_id, original["item_id"]))
        assert cur.fetchone()["state"] == "PURGED"


def test_new_source_window_inherits_current_manual_readers_after_replacement(shared_space):
    repo, ctx, source_uri, worker, _, _ = shared_space
    reader_id = "运维:bob@example.com"
    reader = "user:" + reader_id
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("INSERT INTO cp_users(account_id,user_id,role,key) VALUES(%s,%s,'user',%s)",
                    (ctx.account_id, reader_id, uuid4().hex))
    team = repo.list_spaces(ctx, "ops")["space"]
    members = [{"principal": member["principal"], "role": member["role"]} for member in team["members"]]
    repo.set_members(ctx, "ops", {"members": members + [{"principal": reader, "role": "member"}]},
                     expected_rev=team["authz_rev"])
    run_rule(repo, ctx, worker)
    original = repo.visible_items(ctx, ["ops"])[0]
    revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
    repo.set_acl(ctx, "ops", {"item_id": original["item_id"], "mode": "restricted", "readers": ["user:alice"]},
                 expected_rev=revision)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE context_nodes SET content=content||' Verify recovery',version=version+1 WHERE uri=%s", (source_uri,))
    run_rule(repo, ctx, worker)
    successor = repo.visible_items(ctx, ["ops"])[0]
    revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
    repo.set_acl(ctx, "ops", {"item_id": successor["item_id"], "mode": "restricted", "readers": [reader]},
                 expected_rev=revision)
    new_uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/new-observation"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:ops-agent','case','MEMORY','Observe pool recovery after releasing acquired connections')""",
                    (new_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    bob = RequestContext(ctx.account_id, reader_id, "", "", "updated-readers")
    result = SharedMemoryRetrievalService(repo).search(bob, "pool", ["ops"])["items"]
    assert len(result) == 2
    new_item = next(item for item in result if "Observe pool recovery" in item["content"])
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s",
                    (ctx.account_id, new_item["item_id"]))
        assert cur.fetchall() == [{"principal": reader}]


def test_new_source_before_page_cursor_adds_only_new_contribution(shared_space):
    repo, ctx, _, worker, _, _ = shared_space
    with repo.transaction(ctx.account_id) as cur:
        for index in range(100):
            uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/example-{index:03}"
            cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
                VALUES(%s,%s,'agent:ops-agent','case','MEMORY',%s)""",
                (uri, ctx.account_id, f"Connection pool troubleshooting evidence {index}"))
    run_rule(repo, ctx, worker)
    run_rule(repo, ctx, worker)
    original = repo.visible_items(ctx, ["ops"])
    assert len(original) == 2
    before_uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/000-new-observation"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:ops-agent','case','MEMORY','Observe pending connection acquisition requests')""", (before_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    current = repo.visible_items(ctx, ["ops"])
    assert len(current) == 3
    assert {item["item_id"] for item in original}.issubset({item["item_id"] for item in current})
    assert sum(item["source_count"] for item in current) == 102
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("SELECT source_uri,COUNT(*) AS count FROM space_item_sources WHERE account_id=%s GROUP BY source_uri", (ctx.account_id,))
        counts = cur.fetchall()
        assert len(counts) == 102
        assert {row["count"] for row in counts} == {1}


def test_same_topic_windows_keep_target_policies_and_intersect_new_readers(shared_space):
    repo, ctx, original_uri, worker, _, _ = shared_space
    with repo.transaction(ctx.account_id) as cur:
        for index in range(100):
            uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/example-{index:03}"
            cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
                VALUES(%s,%s,'agent:ops-agent','case','MEMORY',%s)""",
                (uri, ctx.account_id, f"Connection pool troubleshooting evidence {index}"))
    run_rule(repo, ctx, worker)
    run_rule(repo, ctx, worker)
    items = repo.visible_items(ctx, ["ops"])
    alice_item = next(item for item in items if item["source_count"] == 100)
    bob_item = next(item for item in items if item["source_count"] == 1)
    for item, reader in ((alice_item, "user:alice"), (bob_item, "user:bob")):
        revision = repo.list_spaces(ctx, "ops")["space"]["authz_rev"]
        repo.set_acl(ctx, "ops", {"item_id": item["item_id"], "mode": "restricted", "readers": [reader]},
                     expected_rev=revision, request_id=uuid4().hex)
    alice_uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/example-000"
    for source_uri, expected_reader, source_count in ((alice_uri, "user:alice", 100), (original_uri, "user:bob", 1)):
        with repo.transaction(ctx.account_id) as cur:
            cur.execute("UPDATE context_nodes SET content=content||' Verify acquired connections',version=version+1 WHERE uri=%s", (source_uri,))
        run_rule(repo, ctx, worker)
        current = next(item for item in repo.visible_items(ctx, ["ops"]) if item["source_count"] == source_count)
        assert current["acl_origin"] == "manual"
        assert current["acl_mode"] == "restricted"
        with repo.transaction(ctx.account_id) as cur:
            cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND item_id=%s", (ctx.account_id, current["item_id"]))
            assert cur.fetchall() == [{"principal": expected_reader}]
    new_uri = f"ctx://{ctx.account_id}/agents/ops-agent/memories/case/000-new-window"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'agent:ops-agent','case','MEMORY','Observe connection latency after pool recovery')""", (new_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""SELECT i.item_id,i.acl_mode,i.acl_origin FROM space_items i
            JOIN space_item_sources x USING(account_id,space_id,item_id)
            WHERE i.account_id=%s AND i.state='ACTIVE' AND x.source_uri=%s""", (ctx.account_id, new_uri))
        combined = cur.fetchone()
        assert combined["acl_mode"] == "restricted"
        assert combined["acl_origin"] == "manual"
        cur.execute("SELECT COUNT(*) AS count FROM space_read_acl WHERE account_id=%s AND item_id=%s", (ctx.account_id, combined["item_id"]))
        assert cur.fetchone()["count"] == 0


@pytest.mark.parametrize("job_state", ["QUEUED", "RUNNING"])
def test_rule_deletion_settles_jobs_and_system_archives_memory(shared_space, job_state):
    repo, ctx, _, worker, _, _ = shared_space
    run_rule(repo, ctx, worker)
    item = repo.visible_items(ctx, ["ops"])[0]
    definition = repo.list_rules(ctx, "ops")["items"][0]
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("UPDATE space_rule_runtime SET next_run_at=NOW() WHERE account_id=%s", (ctx.account_id,))
    assert repo.reconcile_due_rules(ctx.account_id) == 1
    if job_state == "RUNNING":
        assert repo.claim_job(ctx.account_id)["state"] == "RUNNING"
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT job_id FROM space_jobs WHERE account_id=%s AND state=%s", (ctx.account_id, job_state))
        pending_id = cur.fetchone()["job_id"]

    repo.delete_rule(ctx, "ops", "pool", expected_rev=definition["rule_rev"], request_id=uuid4().hex)
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT state,claim_token,lease_until FROM space_jobs WHERE account_id=%s AND job_id=%s",
                    (ctx.account_id, pending_id))
        assert dict(cur.fetchone()) == {"state": "SUPERSEDED", "claim_token": None, "lease_until": None}
        cur.execute("""SELECT r.state,r.deleted_at,r.rule_rev,v.spec,v.spec_hash
            FROM space_rules r JOIN space_rule_revisions v USING(account_id,space_id,rule_id,rule_rev)
            WHERE r.account_id=%s AND r.space_id='ops' AND r.rule_id='pool'""", (ctx.account_id,))
        deleted = cur.fetchone()
        assert deleted["state"] == "ARCHIVED"
        assert deleted["deleted_at"] is not None
        assert deleted["rule_rev"] == definition["rule_rev"] + 1
        assert (deleted["spec"], deleted["spec_hash"]) == (definition["spec"], definition["spec_hash"])

    assert repo.maintain_due_items(ctx.account_id) == 1
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("SELECT state,purge_after FROM space_items WHERE account_id=%s AND item_id=%s",
                    (ctx.account_id, item["item_id"]))
        archived = cur.fetchone()
        assert archived["state"] == "ARCHIVED"
        assert archived["purge_after"] > datetime.now(UTC)
        cur.execute("""SELECT event_id,event_type,uri,payload,status FROM outbox_events
            WHERE account_id=%s AND uri=%s AND event_type='DELETE_CONTEXT'""", (ctx.account_id, item["uri"]))
        event = OutboxEvent(**dict(cur.fetchone()))
        assert set(event.payload["ids_to_delete"]) == {
            IndexRecord.generate_id(item["uri"], 0), IndexRecord.generate_id(item["uri"] + "/content.md", 2)}
    projector = OutboxWorker(InMemoryVectorIndex(), MockEmbedder(), directory_summary_enabled=False)
    projected = projector.process_event(event)
    assert projected.success, projected.error_message
    SQLOutboxStore(pool=repo.pool).mark_done(event, event.uri)


def test_source_deletion_recomputes_and_new_binding_contributes(shared_space):
    repo, ctx, source_uri, worker, _, _ = shared_space
    repo.put_source(ctx, "ops", "alice-events", {"principal": "user:alice", "memory_types": ["event"]},
                    expected_rev=0, request_id=uuid4().hex)
    definition = repo.list_rules(ctx, "ops")["items"][0]
    spec = {**definition["spec"], "memory_types": ["case", "event"]}
    repo.save_rule(ctx, "ops", "pool", {"name": definition["name"], "spec": spec},
                   expected_rev=definition["rule_rev"], request_id=uuid4().hex)
    event_uri = f"ctx://{ctx.account_id}/users/alice/memories/event/connection"
    with repo.transaction(ctx.account_id) as cur:
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,content)
            VALUES(%s,%s,'user:alice','event','MEMORY','Measure connection wait time after pool recovery')""",
                    (event_uri, ctx.account_id))
    run_rule(repo, ctx, worker)
    assert repo.visible_items(ctx, ["ops"])[0]["source_count"] == 2

    repo.delete_source(ctx, "ops", "ops-source", expected_rev=1, request_id=uuid4().hex)
    run_rule(repo, ctx, worker)
    remaining = repo.visible_items(ctx, ["ops"])[0]
    assert remaining["content"] == "Measure connection wait time after pool recovery"
    assert remaining["source_count"] == 1
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("""SELECT state,grant_rev,deleted_at FROM space_source_bindings
            WHERE account_id=%s AND space_id='ops' AND binding_id='ops-source'""", (ctx.account_id,))
        deleted = cur.fetchone()
        assert deleted["state"] == "PAUSED"
        assert deleted["grant_rev"] == 2
        assert deleted["deleted_at"] is not None

    repo.put_source(ctx, "ops", "ops-renewed", {"principal": "agent:ops-agent", "memory_types": ["case"]},
                    expected_rev=0, request_id=uuid4().hex)
    run_rule(repo, ctx, worker)
    memories = repo.visible_items(ctx, ["ops"])
    assert sum(item["source_count"] for item in memories) == 2
    renewed = next(item for item in memories if "Check connection leaks" in item["content"])
    with repo.transaction(ctx.account_id, write=False) as cur:
        cur.execute("""SELECT binding_id,grant_rev,source_uri FROM space_item_sources
            WHERE account_id=%s AND space_id='ops' AND item_id=%s ORDER BY source_uri""",
                    (ctx.account_id, renewed["item_id"]))
        assert [dict(row) for row in cur.fetchall()] == [
            {"binding_id": "ops-renewed", "grant_rev": 1, "source_uri": source_uri}]
