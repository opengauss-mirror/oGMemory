from dataclasses import replace

from core.lexical import LexicalHit
from scene.candidate_index import LazySceneIndex, LexicalSceneIndex, SceneIndexRefresh
from scene.models import SceneCluster, SceneMember, SceneRecord, SceneSnapshot


class MemoryIndex:
    def __init__(self):
        self.records = []
        self.events = []

    def begin_rebuild(self):
        self.events.append("begin")
        return "lease"

    def replace_scopes(self, *, rebuild_id, account_id, owner_spaces, records):
        assert rebuild_id == "lease"
        self.records = [r for r in self.records
                        if (r.account_id, r.owner_space) != (account_id, owner_spaces[0])]
        self.records.extend(records)
        self.events.append("publish")
        return True

    def abort_rebuild(self, rebuild_id):
        assert rebuild_id == "lease"
        self.events.append("release")

    def search(self, query, top_k, filters):
        self.filters = filters
        return [LexicalHit(r.id, 1.0, r.metadata) for r in self.records
                if r.account_id == filters["account_id"]
                and r.owner_space in filters["owner_space"]
                and r.metadata["context_type"] == filters["context_type"]
                and query in r.text][:top_k]

    def close(self, timeout=5.0):
        self.timeout = timeout
        return True


def snapshot(account="acme", owner="user:alice"):
    cluster = SceneCluster(
        owner_space=owner, members=(SceneMember("memory-1", 1.0),),
        centroid=(1.0, 0.0), quality=0.9, label="Deployment",
        summary="Service release", source_fingerprint="source-v1",
        boundary=(("activity", "rollback"),),
    )
    return SceneSnapshot(account, owner, 1, (SceneRecord("release", cluster),))


def test_projection_publishes_scene_card_and_searches_label_summary_boundary():
    index = MemoryIndex()
    projection = LexicalSceneIndex(index)
    current = snapshot()
    projection.publish(current)
    record = index.records[0]
    assert (record.id, record.uri, record.level) == ("acme/user:alice/release",) * 2 + (0,)
    assert record.text == "Deployment\nService release\nrollback"
    assert record.metadata == {"scene_id": "release", "scene_version": 1, "context_type": "SCENE"}
    for query in ("Deployment", "Service release", "rollback"):
        assert projection.search(query, current, 3) == ["release"]
    assert index.filters == {"account_id": "acme", "owner_space": ["user:alice"], "context_type": "SCENE"}
    assert index.events == ["begin", "publish", "release"]
    assert projection.close(timeout=2.0)
    assert index.timeout == 2.0


def test_projection_replaces_one_scope_and_preserves_other_accounts_and_owners():
    index = MemoryIndex()
    projection = LexicalSceneIndex(index)
    current = snapshot()
    other_owner = snapshot(owner="user:bob")
    other_account = snapshot(account="other")
    for scope in (current, other_owner, other_account):
        projection.publish(scope)
    updated = replace(current, revision=2, scenes=(replace(current.scenes[0], version=2),))
    projection.publish(updated)
    assert len(index.records) == 3
    assert {r.id: r.metadata["scene_version"] for r in index.records} == {
        "acme/user:alice/release": 2, "acme/user:bob/release": 1, "other/user:alice/release": 1,
    }
    for scope in (updated, other_owner, other_account):
        assert projection.search("Deployment", scope, 1) == ["release"]


def test_refresh_acknowledges_each_published_snapshot():
    index = MemoryIndex()
    scopes = [snapshot(), snapshot(owner="user:bob")]

    class Queue:
        def pending_projections(self):
            yield from scopes

        def publish_projection(self, current, publish):
            publish(current)
            assert index.events[-2:] == ["publish", "release"]
            assert any(r.owner_space == current.owner_space for r in index.records)
            index.events.append(("ack", current.owner_space, current.revision))
            return True

    queue = Queue()
    SceneIndexRefresh(queue, LazySceneIndex(lambda: index, projection_queue=queue))()
    assert index.events == [
        "begin", "publish", "release", ("ack", "user:alice", 1),
        "begin", "publish", "release", ("ack", "user:bob", 1),
    ]


def test_lazy_projection_shares_provider_and_refreshes_each_scope_revision_once():
    from concurrent.futures import ThreadPoolExecutor

    index = MemoryIndex()
    factories = []

    def create():
        factories.append(index)
        return index

    projection = LazySceneIndex(create)
    current = snapshot()
    with ThreadPoolExecutor(max_workers=4) as callers:
        results = list(callers.map(lambda _: projection.search("Deployment", current, 1), range(8)))
    assert results == [["release"]] * 8
    updated = replace(current, revision=2, scenes=(replace(current.scenes[0], version=2),))
    assert projection.search("Deployment", updated, 1) == ["release"]
    assert factories == [index]
    assert index.events.count("publish") == 2
    assert projection.close(timeout=1)
    assert projection.close(timeout=1)
