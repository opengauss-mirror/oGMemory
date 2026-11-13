"""Rebuildable index projections for accepted relation edges.

Canonical edges remain owned by the graph/relation write path.  This module
only builds the disposable search projection consumed by Stage 8R retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Protocol

from index.embedding_components import EmbeddingComponentCache, EmbeddingComponentKey
from providers.relation_store.projection_identity import (
    FACTORIZED_REPRESENTATION_VERSION,
    REPRESENTATION_VERSION,
)
from providers.token_tracker import embedding_input_scope

DEFAULT_PROJECTION_BATCH_BYTES = 8192
_logger = logging.getLogger(__name__)


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


def projection_embedding_key(edge: AcceptedEdgeProjectionInput) -> tuple[str, str]:
    """Reuse identical input within one canonical source owner."""
    owner = "/".join(edge.source_uri.split("/")[:5])
    return owner, build_searchable_text(edge)


def bounded_projection_batch(
    edges: list[AcceptedEdgeProjectionInput],
    max_bytes: int = DEFAULT_PROJECTION_BATCH_BYTES,
) -> list[AcceptedEdgeProjectionInput]:
    """Bound unique UTF-8 input per claim; preserve an oversized input whole."""
    selected: list[AcceptedEdgeProjectionInput] = []
    seen: set[tuple[str, str]] = set()
    total = 0
    for edge in edges:
        key = projection_embedding_key(edge)
        size = 0 if key in seen else len(key[1].encode("utf-8"))
        if selected and size and total + size > max_bytes:
            break
        selected.append(edge)
        seen.add(key)
        total += size
    return selected


class EdgeProjectionWorker:
    """Batch projection worker called only after canonical edge acceptance."""

    def __init__(self, embedder, store: EdgeProjectionStore, profile: dict,
                 component_cache: EmbeddingComponentCache | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._profile = dict(profile)
        self._component_cache = component_cache if component_cache is not None else EmbeddingComponentCache()
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
        representation = self._profile.get("representation_version", REPRESENTATION_VERSION)
        texts = [build_searchable_text(edge) for edge in edges]
        if representation == FACTORIZED_REPRESENTATION_VERSION:
            vectors = self._factorized_vectors(edges)
        elif representation == REPRESENTATION_VERSION:
            keys = [projection_embedding_key(edge) for edge in edges]
            unique = {}
            for index, key in enumerate(keys):
                unique.setdefault(key, index)
            with embedding_input_scope(list(unique.values())):
                embedded = self._embedder.embed_texts([key[1] for key in unique])
            if len(embedded) != len(unique):
                raise ValueError("embedder returned a different number of vectors")
            vectors_by_key = dict(zip(unique, embedded, strict=True))
            vectors = [vectors_by_key[key] for key in keys]
        else:
            raise ValueError("unsupported edge projection representation")
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
                    representation_version=representation,
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

    def _factorized_vectors(self, edges: list[AcceptedEdgeProjectionInput]) -> list[tuple[float, ...]]:
        preprocessing = (
            ("max_input_chars", str(self._profile.get("max_input_chars", getattr(self._embedder, "max_input_chars", None)))),
            ("multimodal", str(self._profile.get("multimodal", False))),
        )
        edge_components = []
        first_edge: dict[EmbeddingComponentKey, int] = {}
        for index, edge in enumerate(edges):
            owner = projection_embedding_key(edge)[0]
            components = []
            for kind, text in (("target", edge.target_text.strip()),
                               ("relation", f"Relationship: {edge.relation.strip()}")):
                if kind == "target" and not text:
                    components.append(None)
                    continue
                key = EmbeddingComponentKey(owner=owner, profile_id=str(self._profile["profile_id"]),
                    model=str(self._profile["model"]), dimension=int(self._profile.get("dimension") or 0),
                    representation=FACTORIZED_REPRESENTATION_VERSION, kind=kind, text=text,
                    preprocessing=preprocessing)
                components.append(key)
                first_edge.setdefault(key, index)
            edge_components.append(components)

        def compute(keys):
            weights = [max(1, len(key.text.encode("utf-8"))) for key in keys]
            with embedding_input_scope([first_edge[key] for key in keys], weights=weights):
                result = self._embedder.embed_texts([key.text for key in keys])
            if len(result) != len(keys):
                raise ValueError("component vector count must match input count")
            return [self._unit_vector(vector) for vector in result]

        values = {}
        batch = []
        size = 0
        for key in first_edge:
            length = len(key.text.encode("utf-8"))
            if batch and size + length > DEFAULT_PROJECTION_BATCH_BYTES:
                values.update(self._component_cache.resolve(batch, compute))
                batch = []
                size = 0
            batch.append(key)
            size += length
        if batch:
            values.update(self._component_cache.resolve(batch, compute))
        projected = [values[relation_key] if target_key is None else
                self._unit_vector([.75 * target + .25 * relation
                for target, relation in zip(values[target_key], values[relation_key], strict=True)])
                for target_key, relation_key in edge_components]
        _logger.info("graph_component_cache_totals %s", json.dumps(self._component_cache.snapshot(), sort_keys=True))
        return projected

    def _unit_vector(self, vector) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        dimension = self._profile.get("dimension")
        if dimension and len(values) != int(dimension):
            raise ValueError("edge projection dimension does not match profile")
        scale = math.sqrt(sum(value * value for value in values))
        if scale <= 0 or not math.isfinite(scale):
            raise ValueError("component vector must have a finite positive norm")
        return tuple(value / scale for value in values)

    def edge_changed(self, edge: AcceptedEdgeProjectionInput) -> None:
        """Invalidate stale content and enqueue its replacement projection."""
        self._store.delete(edge.edge_id)
        self.enqueue_accepted([edge])

    def edge_deleted(self, edge_id: str) -> None:
        self._store.delete(edge_id)
