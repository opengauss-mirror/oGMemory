"""Scene-card projection through the shared lexical-index contract."""

from collections.abc import Iterable, Sequence
from threading import RLock
from time import monotonic
from typing import Protocol

from core.lexical import LexicalIndex, LexicalRecord
from scene.models import SceneSnapshot


class SceneCandidateIndex(Protocol):
    def search(self, query: str, snapshot: SceneSnapshot, limit: int) -> Sequence[str]: ...

    def close(self, timeout: float = 5.0) -> bool: ...


class SceneProjectionQueue(Protocol):
    def pending_projections(self, limit: int = 20) -> Iterable[SceneSnapshot]:
        """Yield detached snapshots awaiting publication."""
        ...

    def publish_projection(self, snapshot: SceneSnapshot, publish) -> bool:
        """Confirm durable publication under a shared scope lock and revision fence."""
        ...


class SceneIndexRefresh:
    def __init__(self, queue: SceneProjectionQueue, index: "LazySceneIndex"):
        self.queue, self.index = queue, index

    def __call__(self):
        for snapshot in self.queue.pending_projections():
            self.index.publish(snapshot)


class LexicalSceneIndex:
    """Publish scene cards and validate search hits against canonical revisions."""

    def __init__(self, index: LexicalIndex):
        self.index = index

    def close(self, timeout: float = 5.0) -> bool:
        return self.index.close(timeout=timeout)

    def publish(self, snapshot: SceneSnapshot) -> None:
        records = []
        for scene in snapshot.scenes:
            if scene.status != "ACTIVE":
                continue
            cluster = scene.cluster
            identity = f"{snapshot.account_id}/{snapshot.owner_space}/{scene.scene_id}"
            records.append(LexicalRecord(
                identity, identity, 0,
                "\n".join((cluster.label, cluster.summary,
                           " ".join(value for _, value in cluster.boundary))),
                snapshot.account_id, snapshot.owner_space,
                {"scene_id": scene.scene_id, "scene_version": scene.version,
                 "context_type": "SCENE"},
            ))
        rebuild = self.index.begin_rebuild()
        try:
            if not self.index.replace_scopes(
                rebuild_id=rebuild, account_id=snapshot.account_id,
                owner_spaces=[snapshot.owner_space], records=records,
            ):
                raise RuntimeError("scene lexical projection changed during publication")
        finally:
            self.index.abort_rebuild(rebuild)

    def search(self, query: str, snapshot: SceneSnapshot, limit: int) -> list[str]:
        versions = {s.scene_id: s.version for s in snapshot.scenes if s.status == "ACTIVE"}
        hits = self.index.search(query, limit, {
            "account_id": snapshot.account_id, "owner_space": [snapshot.owner_space],
            "context_type": "SCENE",
        })
        return list(dict.fromkeys(
            h.metadata["scene_id"] for h in hits
            if h.score > 0 and h.metadata.get("scene_id") in versions
            and h.metadata.get("scene_version") == versions[h.metadata["scene_id"]]
        ))


class LazySceneIndex:
    """Initialize the shared lexical provider on the first enabled search."""

    def __init__(self, factory, *, projection_queue: SceneProjectionQueue | None = None):
        self._factory = factory
        self.projection_queue = projection_queue
        self._index = None
        self._revisions = {}
        self._lock = RLock()
        self._closed = False

    @property
    def initialized(self):
        with self._lock:
            return self._index is not None and not self._closed

    def _ensure_index(self):
        if self._closed:
            raise RuntimeError("scene lexical index is closed")
        if self._index is None:
            self._index = LexicalSceneIndex(self._factory())
        return self._index

    def publish(self, snapshot):
        with self._lock:
            index = self._ensure_index()
            scope = snapshot.account_id, snapshot.owner_space
            if snapshot.revision > self._revisions.get(scope, -1):
                if self.projection_queue is None:
                    index.publish(snapshot)
                elif not self.projection_queue.publish_projection(snapshot, index.publish):
                    return
                self._revisions[scope] = snapshot.revision

    def search(self, query, snapshot, limit):
        with self._lock:
            self.publish(snapshot)
            return self._index.search(query, snapshot, limit)

    def close(self, timeout=5.0):
        deadline = monotonic() + max(0, timeout)
        if not self._lock.acquire(timeout=max(0, timeout)):
            return False
        try:
            if self._closed:
                return True
            if self._index is not None and not self._index.close(timeout=max(0, deadline - monotonic())):
                return False
            self._closed = True
            return True
        finally:
            self._lock.release()
