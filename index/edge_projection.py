"""Rebuildable index projections for accepted relation edges.

Canonical edges remain owned by the graph/relation write path.  This module
only builds the disposable search projection consumed by Stage 8R retrieval.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass
from typing import Protocol

from providers.relation_store.projection_identity import REPRESENTATION_VERSION



@dataclass(frozen=True)
class AcceptedEdgeProjectionInput:
    edge_id: str
    source_uri: str
    target_uri: str
    relation: str
    reason: str
    target_text: str
    source_revision: int
    target_revision: int
    claim_token: str | None = None


@dataclass(frozen=True)
class EdgeProjection:
    edge_id: str
    text: str
    vector: tuple[float, ...]
    representation_version: str
    embedding_profile: str
    model: str
    dimension: int
    source_revision: int
    target_revision: int
    claim_token: str | None = None

    def as_graph_metadata(self) -> dict:
        metadata = asdict(self)
        metadata.pop("claim_token")  # A queue lease is not searchable evidence.
        return metadata


class EdgeProjectionStore(Protocol):
    def put_many(self, projections: list[EdgeProjection]) -> None: ...

    def delete(self, edge_id: str) -> None: ...


class InMemoryEdgeProjectionStore:
    """Small reference store used by workers and deterministic tests."""

    def __init__(self) -> None:
        self._items: dict[str, EdgeProjection] = {}

    def put_many(self, projections: list[EdgeProjection]) -> None:
        self._items.update((item.edge_id, item) for item in projections)

    def get(self, edge_id: str) -> EdgeProjection | None:
        return self._items.get(edge_id)

    def delete(self, edge_id: str) -> None:
        self._items.pop(edge_id, None)


def build_searchable_text(edge: AcceptedEdgeProjectionInput) -> str:
    """Embed relation and target only; keep reason as canonical provenance.

    The BEAM task-2 replay found identical retrieved sets without reason.
    Source text remains excluded; adding it did not recover required facts
    in that replay. Neither observation establishes general accuracy gains.
    """
    parts = [
        f"Relationship: {edge.relation.strip()}",
        f"Target memory: {edge.target_text.strip()}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


class EdgeProjectionWorker:
    """Batch projection worker called only after canonical edge acceptance."""

    def __init__(self, embedder, store: EdgeProjectionStore, profile: dict) -> None:
        self._embedder = embedder
        self._store = store
        self._profile = dict(profile)
        self._pending: dict[str, AcceptedEdgeProjectionInput] = {}

    def enqueue_accepted(self, edges: list[AcceptedEdgeProjectionInput]) -> None:
        """Queue only edges the canonical write path has already accepted."""
        self._pending.update((edge.edge_id, edge) for edge in edges)

    async def project_pending(self) -> list[EdgeProjection]:
        """Drain one deterministic batch outside the relation write request."""
        edges = [self._pending[key] for key in sorted(self._pending)]
        projected = await self.project_accepted(edges)
        for edge in edges:
            if self._pending.get(edge.edge_id) == edge:
                self._pending.pop(edge.edge_id, None)
        return projected

    async def project_accepted(
        self, edges: list[AcceptedEdgeProjectionInput]
    ) -> list[EdgeProjection]:
        """Project a batch without blocking an asynchronous caller."""
        return await asyncio.to_thread(self.project_accepted_sync, edges)

    def project_accepted_sync(
        self, edges: list[AcceptedEdgeProjectionInput]
    ) -> list[EdgeProjection]:
        """Project a batch for the synchronous outbox worker."""
        if not edges:
            return []
        texts = [build_searchable_text(edge) for edge in edges]
        vectors = self._embedder.embed_texts(texts)
        if len(vectors) != len(edges):
            raise ValueError("embedder returned a different number of vectors")
        projections: list[EdgeProjection] = []
        for edge, text, vector in zip(edges, texts, vectors, strict=True):
            dimension = len(vector)
            if not dimension or not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("edge projection vector must contain finite numbers")
            configured_dimension = self._profile.get("dimension")
            if configured_dimension and dimension != int(configured_dimension):
                raise ValueError("edge projection dimension does not match profile")
            projections.append(
                EdgeProjection(
                    edge_id=edge.edge_id,
                    text=text,
                    vector=tuple(float(value) for value in vector),
                    representation_version=REPRESENTATION_VERSION,
                    embedding_profile=str(self._profile["profile_id"]),
                    model=str(self._profile["model"]),
                    dimension=dimension,
                    source_revision=edge.source_revision,
                    target_revision=edge.target_revision,
                    claim_token=edge.claim_token,
                )
            )
        self._store.put_many(projections)
        return projections

    def edge_changed(self, edge: AcceptedEdgeProjectionInput) -> None:
        """Invalidate stale content and enqueue its replacement projection."""
        self._store.delete(edge.edge_id)
        self.enqueue_accepted([edge])

    def edge_deleted(self, edge_id: str) -> None:
        self._store.delete(edge_id)
