"""Scene management requests through the production service and HTTP blueprint."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from flask import Flask, jsonify

from core.errors import ConcurrentModificationError, NodeNotFoundError
from core.models import ContextNode, RequestContext
from scene.adapters import CanonicalSceneAdapter
from scene.config import InMemorySceneConfigProvider
from scene.jobs import InMemorySceneJobStore
from scene.models import SceneSnapshot
from scene.service import SceneService
from server.scene_api import create_scene_blueprint


class Repository:
    def __init__(self):
        self.scopes = {}

    def load_snapshot(self, account_id, owner_space):
        return deepcopy(self.scopes.get((account_id, owner_space), SceneSnapshot(account_id, owner_space, 0)))

    def compare_and_swap(self, expected, replacement):
        key = (expected.account_id, expected.owner_space)
        current = self.load_snapshot(*key)
        if current.revision != expected.revision:
            return False
        self.scopes[key] = deepcopy(replacement)
        return True

    def compare_and_swap_checked(self, expected, replacement, check):
        return check(None) and self.compare_and_swap(expected, replacement)


class MemoryStore:
    def __init__(self, nodes):
        self.nodes = {node.uri: node for node in nodes}

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return deepcopy(self.nodes[uri])

    def get_embeddings(self, uris, filters, level=0, *, expected_texts=None):
        return {uri: [1.0, 0.0] for uri in uris if uri in self.nodes}


@pytest.fixture
def management():
    ctx = RequestContext("acme", "alice", "helper", "session", "trace")
    uri = "ctx://acme/users/alice/memories/events/release"
    node = ContextNode(uri, "MEMORY", "event", 4, "user:alice", "Release planning", "",
                       "The release is scheduled for Friday.", {"version": 3})
    repository = Repository()
    store = MemoryStore([node])
    embedder = SimpleNamespace(embed_texts=lambda texts: [[1.0, 0.0] for _ in texts])
    config = InMemorySceneConfigProvider()
    adapter = CanonicalSceneAdapter(repository, store, store, embedder, config_provider=config)
    jobs = InMemorySceneJobStore()
    manager = SimpleNamespace(jobs=jobs, flush=jobs.flush, running=True, halted=False)
    service = SceneService(repository, adapter, manager, config, embedder)
    return SimpleNamespace(ctx=ctx, uri=uri, repository=repository, config=config, adapter=adapter, service=service)


def test_scene_management_create_edit_members_paginate_and_archive(management):
    fixture = management
    service, ctx = fixture.service, fixture.ctx
    created = service.save(ctx, {"revision": 0, "title": "Release planning",
                                 "description": "Coordinate release checks."})
    scene_id = created["scene"]["scene_id"]
    assert created["scene"]["origin"] == "manual"
    updated = service.save(ctx, {"revision": 1, "member_uris": [fixture.uri]}, scene_id=scene_id)
    assert updated["scene"]["cluster"]["members"][0]["context_uri"] == fixture.uri
    edited = service.save(ctx, {"revision": 2, "description": "Coordinate Friday release checks."}, scene_id=scene_id)
    assert edited["scene"]["cluster"]["summary"] == "Coordinate Friday release checks."
    assert edited["scene"]["description_edited"] is True
    service.save(ctx, {"revision": 3, "title": "Deployment review", "description": "Review deployment evidence."})
    first = service.list(ctx, limit=1)
    second = service.list(ctx, offset=first["next_offset"], limit=1)
    assert first["total"] == 2
    assert len(first["items"]) == len(second["items"]) == 1
    assert second["next_offset"] is None
    assert service.archive(ctx, scene_id, revision=4)["status"] == "ARCHIVED"
    assert service.list(ctx)["total"] == 1
    assert fixture.adapter.fs.read_node(fixture.uri, ctx).content.endswith("Friday.")


def test_scene_configuration_publish_reaches_next_search_request(management):
    fixture = management
    service = fixture.service
    current = service.configuration(fixture.ctx)
    updated = service.update_configuration(fixture.ctx, {"revision": current["revision"],
                                                        "values": {"scene_boost": 0.25}})
    assert updated["revision"] == current["revision"] + 1
    assert fixture.adapter.for_request(fixture.ctx, []).config.scene_boost == 0.25


def test_description_edit_preserves_scene_boundary_and_refreshes_member_version(management):
    fixture = management
    created = fixture.service.save(fixture.ctx, {
        "revision": 0, "title": "Release planning", "description": "Review the release.",
        "member_uris": [fixture.uri],
    })
    key = (fixture.ctx.account_id, fixture.ctx.user_space_name())
    snapshot = fixture.repository.scopes[key]
    record = snapshot.scenes[0]
    cluster = replace(record.cluster, boundary=(("activity", "release"),), time_start="2026-01-01")
    fixture.repository.scopes[key] = replace(snapshot, scenes=(replace(record, cluster=cluster),))
    fixture.adapter.fs.nodes[fixture.uri].metadata["version"] = 4
    edited = fixture.service.save(fixture.ctx, {
        "revision": 1, "description": "Review the Friday release.",
    }, scene_id=created["scene"]["scene_id"])
    assert edited["scene"]["cluster"]["boundary"] == (("activity", "release"),)
    assert edited["scene"]["cluster"]["time_start"] == "2026-01-01"
    assert edited["scene"]["cluster"]["members"][0]["node_version"] == 4


def test_agent_scene_management_keeps_its_own_scope(management):
    service, ctx = management.service, management.ctx
    service.save(ctx, {"revision": 0, "owner_space": "agent:helper", "title": "Tool execution",
                       "description": "Track reusable execution procedures."})
    assert service.list(ctx, owner_space="agent:helper")["total"] == 1
    assert service.list(ctx)["total"] == 0


def test_manual_scene_preserves_complete_canonical_sources(management):
    fixture = management
    refs = [f"archive:release_{index}" for index in range(5)]
    fixture.adapter.fs.nodes[fixture.uri].metadata["provenance_ids"] = refs
    created = fixture.service.save(fixture.ctx, {
        "revision": 0, "title": "Release planning", "description": "Review all release evidence.",
        "member_uris": [fixture.uri],
    })
    member = created["scene"]["cluster"]["members"][0]
    assert member["source_refs"] == tuple(refs)
    persisted = fixture.repository.load_snapshot("acme", "user:alice").scenes[0]
    assert persisted.cluster.members[0].source_refs == tuple(refs)


def test_manual_scene_saves_configured_member_capacity_with_full_content(management):
    fixture = management
    template = fixture.adapter.fs.nodes[fixture.uri]
    nodes = [replace(template, uri=f"{fixture.uri}_{index}", abstract="release " * 400,
                     content="release evidence " * 800, metadata={"version": index + 1})
             for index in range(64)]
    fixture.adapter.fs.nodes.update((node.uri, node) for node in nodes)
    created = fixture.service.save(fixture.ctx, {
        "revision": 0, "title": "Release evidence", "description": "Full release member set.",
        "member_uris": [node.uri for node in nodes],
    })
    persisted = fixture.repository.load_snapshot("acme", "user:alice").scenes[0]
    assert len(created["scene"]["cluster"]["members"]) == 64
    assert {member.context_uri for member in persisted.cluster.members} == {node.uri for node in nodes}
    assert {member.node_version for member in persisted.cluster.members} == set(range(1, 65))


def test_manual_save_recovers_with_current_evidence_after_embedding(management):
    fixture = management
    embed_texts = fixture.service.embedder.embed_texts

    def update_source(texts):
        fixture.adapter.fs.nodes[fixture.uri].metadata["version"] = 4
        fixture.service.embedder.embed_texts = embed_texts
        return embed_texts(texts)

    fixture.service.embedder.embed_texts = update_source
    payload = {"revision": 0, "title": "Release planning", "description": "Review release evidence.",
               "member_uris": [fixture.uri]}
    with pytest.raises(ConcurrentModificationError):
        fixture.service.save(fixture.ctx, payload)
    created = fixture.service.save(fixture.ctx, payload)
    assert created["revision"] == 1
    assert created["scene"]["cluster"]["members"][0]["node_version"] == 4
    assert fixture.service.list(fixture.ctx)["total"] == 1


@pytest.fixture
def client(management):
    app = Flask(__name__)
    app.config["TESTING"] = True
    facade = SimpleNamespace(get_scene_service=lambda: management.service,
                             get_auth_service=lambda: SimpleNamespace(role_control_active=lambda: False))

    def error_response(error):
        return jsonify({"error": type(error).__name__}), 400

    app.register_blueprint(create_scene_blueprint(lambda: facade, lambda payload: management.ctx, error_response))
    return app.test_client()


def test_scene_http_create_update_list_and_delete(client, management):
    response = client.post("/api/v1/scenes", json={"revision": 0, "title": "Release planning",
                                                  "description": "Prepare a verified release.",
                                                  "member_uris": [management.uri]})
    assert response.status_code == 200
    scene_id = response.json["scene"]["scene_id"]
    edited = client.put(f"/api/v1/scenes/{scene_id}", json={"revision": 1, "description": "Prepare Friday release."})
    assert edited.status_code == 200
    listed = client.get("/api/v1/scenes?limit=1&offset=0")
    assert listed.status_code == 200
    assert listed.json["items"][0]["cluster"]["summary"] == "Prepare Friday release."
    deleted = client.delete(f"/api/v1/scenes/{scene_id}", json={"revision": 2})
    assert deleted.status_code == 200
    assert deleted.json["status"] == "ARCHIVED"
    assert client.get("/api/v1/scenes").json["total"] == 0


def test_scene_http_configuration_hot_reload(client):
    initial = client.get("/api/v1/scenes/configuration")
    assert initial.status_code == 200
    updated = client.put("/api/v1/scenes/configuration", json={"revision": initial.json["revision"],
                                                              "values": {"neighbor_k": 12}})
    assert updated.status_code == 200
    assert updated.json["values"]["neighbor_k"] == 12
    assert client.get("/api/v1/scenes/configuration").json["revision"] == 1


def test_scene_http_flush_promotes_pending_evidence(client, management):
    management.service.manager.jobs.enqueue(
        "acme", "user:alice", [management.uri], token_count=2, trigger_tokens=10,
    )
    status = client.get("/api/v1/scenes/status")
    assert status.status_code == 200
    assert status.json["pending_nodes"] == 1
    flushed = client.post("/api/v1/scenes/flush", json={})
    assert flushed.status_code == 200
    assert flushed.json["active_job"]["node_uris"] == [management.uri]
    assert flushed.json["active_job"]["status"] == "PENDING"
    assert flushed.json["running"] is True
