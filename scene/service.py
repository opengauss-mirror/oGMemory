"""Authorized scene management over a versioned repository."""

from dataclasses import asdict, replace
from uuid import uuid4

from core.errors import AccessDeniedError, ConcurrentModificationError, NodeNotFoundError
from core.models import RequestContext, Role
from scene.decision_sources import node_source_refs
from scene.models import SceneCluster, SceneMember, SceneRecord


class SceneService:
    def __init__(self, repository, adapter, manager, config_provider, embedder):
        self.repository = repository
        self.adapter = adapter
        self.manager = manager
        self.config_provider = config_provider
        self.embedder = embedder

    @staticmethod
    def scope(ctx: RequestContext, owner_space: str | None = None) -> str:
        owner = owner_space or ctx.user_space_name()
        own = {ctx.user_space_name(), ctx.agent_space_name()}
        if owner not in own and ctx.role not in {Role.ROOT, Role.ADMIN}:
            raise AccessDeniedError(owner, ctx.account_id, "scene ownership required")
        if not owner.startswith(("user:", "agent:")) or not owner.split(":", 1)[1]:
            raise ValueError("owner_space must identify a user or agent")
        return owner

    def list(self, ctx, *, owner_space=None, offset=0, limit=20):
        owner = self.scope(ctx, owner_space)
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset nonnegative")
        snapshot = self.repository.load_snapshot(ctx.account_id, owner)
        scenes = [item for item in snapshot.scenes if item.status == "ACTIVE"]
        return {"revision": snapshot.revision, "total": len(scenes),
                "items": [asdict(item) for item in scenes[offset:offset + limit]],
                "next_offset": offset + limit if offset + limit < len(scenes) else None}

    def save(self, ctx, payload, *, scene_id=None):
        owner = self.scope(ctx, payload.get("owner_space"))
        expected = payload.get("revision")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("revision is required")
        snapshot = self.repository.load_snapshot(ctx.account_id, owner)
        self._check_revision(snapshot, expected)
        config = self.config_provider.get_effective(ctx.account_id, owner).config
        previous = next((record for record in snapshot.scenes if record.scene_id == scene_id), None)
        if scene_id and (previous is None or previous.status != "ACTIVE"):
            raise NodeNotFoundError(scene_id)
        title = payload.get("title", previous.cluster.label if previous else "")
        description = payload.get("description", previous.cluster.summary if previous else "")
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
            raise ValueError("title must contain 1..120 characters")
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 2000:
            raise ValueError("description must contain 1..2000 characters")
        title, description = title.strip(), description.strip()
        uris = payload.get("member_uris")
        if uris is None:
            uris = [member.context_uri for member in previous.cluster.members] if previous else []
        if not isinstance(uris, list) or any(not isinstance(uri, str) for uri in uris):
            raise ValueError("member_uris must be a list of URI strings")
        if len(uris) > config.max_members_per_scene or len(uris) != len(set(uris)):
            raise ValueError("member cardinality exceeds the configured limit")
        loaded = self.adapter.load_input(ctx.account_id, owner)
        snapshot = loaded.snapshot
        self._check_revision(snapshot, expected)
        nodes, _ = self.adapter.load_nodes(ctx.account_id, owner, uris)
        if {node.uri for node in nodes} != set(uris):
            raise ValueError("members must be active indexed memories in the scene scope")
        previous_members = {member.context_uri: member for member in previous.cluster.members} if previous else {}
        members = tuple(replace(
            previous_members.get(node.uri, SceneMember(node.uri, 1.0)),
            node_version=int(node.metadata.get("version", 0)),
            source_refs=node_source_refs(node),
        ) for node in nodes)
        if previous is None and sum(s.status == "ACTIVE" for s in snapshot.scenes) >= config.max_active_scenes_per_space:
            raise ValueError("active scene capacity reached")
        vector = self.embedder.embed_texts([title + "\n" + description])[0]
        cluster = SceneCluster(owner, members, tuple(vector), 1.0, title,
                               description, uuid4().hex, basis="manual")
        if previous is not None:
            cluster = replace(previous.cluster, members=members, centroid=tuple(vector),
                              label=title, summary=description,
                              source_fingerprint=cluster.source_fingerprint)
        record = SceneRecord(scene_id or uuid4().hex, cluster,
                             version=previous.version + 1 if previous else 1,
                             origin=previous.origin if previous else "manual",
                             description_edited=True)
        records = tuple(item for item in snapshot.scenes if item.scene_id != record.scene_id) + (record,)
        self._publish(snapshot, records, evidence_nodes=nodes)
        return {"revision": snapshot.revision + 1, "scene": asdict(record)}

    def archive(self, ctx, scene_id, *, revision, owner_space=None):
        owner = self.scope(ctx, owner_space)
        snapshot = self.repository.load_snapshot(ctx.account_id, owner)
        self._check_revision(snapshot, revision)
        previous = next((record for record in snapshot.scenes if record.scene_id == scene_id), None)
        if previous is None:
            raise NodeNotFoundError(scene_id)
        records = tuple(replace(item, status="ARCHIVED", version=item.version + 1)
                        if item.scene_id == scene_id else item for item in snapshot.scenes)
        self._publish(snapshot, records)
        return {"revision": snapshot.revision + 1, "scene_id": scene_id, "status": "ARCHIVED"}

    def configuration(self, ctx, owner_space=None):
        owner = self.scope(ctx, owner_space)
        snapshot = self.config_provider.get_effective(ctx.account_id, owner)
        return {"revision": snapshot.revision, "source": snapshot.source,
                "values": asdict(snapshot.config)}

    def status(self, ctx, owner_space=None):
        owner = self.scope(ctx, owner_space)
        return {**self.manager.jobs.summary(ctx.account_id, owner),
                "running": self.manager.running, "halted": self.manager.halted}

    def flush(self, ctx, owner_space=None):
        owner = self.scope(ctx, owner_space)
        self.manager.flush(ctx.account_id, owner)
        return self.status(ctx, owner)

    def update_configuration(self, ctx, payload):
        owner = self.scope(ctx, payload.get("owner_space"))
        if isinstance(payload.get("revision"), bool) or not isinstance(payload.get("revision"), int):
            raise ValueError("revision must be an integer")
        if not isinstance(payload.get("values"), dict):
            raise ValueError("values must contain configuration overrides")
        self.config_provider.publish(ctx.account_id, owner, payload["revision"], payload["values"])
        return self.configuration(ctx, owner)

    def _publish(self, snapshot, records, *, evidence_nodes=None):
        replacement = replace(snapshot, revision=snapshot.revision + 1, scenes=records)
        published = (self.repository.compare_and_swap(snapshot, replacement) if evidence_nodes is None
                     else self.adapter.compare_and_swap(snapshot, replacement, evidence_nodes=evidence_nodes))
        if not published:
            actual = self.repository.load_snapshot(snapshot.account_id, snapshot.owner_space)
            raise ConcurrentModificationError(snapshot.owner_space, snapshot.revision, actual.revision)

    @staticmethod
    def _check_revision(snapshot, expected):
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("revision must be an integer")
        if snapshot.revision != expected:
            raise ConcurrentModificationError(snapshot.owner_space, expected, snapshot.revision)
