"""Outbox projection with optional vector generation."""

from core.interfaces import Embedder, VectorIndex
from core.models import IndexRecord


class VectorProjection:
    def __init__(self, vector_index: VectorIndex, embedder: Embedder | None) -> None:
        self._vector_index = vector_index
        self._embedder = embedder

    @property
    def has_embeddings(self) -> bool:
        return self._embedder is not None

    def embed(self, records: list[IndexRecord]) -> list[list[float]]:
        assert self._embedder is not None
        return self._embedder.embed_texts([record.text for record in records])

    def upsert(self, records: list[IndexRecord]) -> None:
        self._vector_index.upsert(records)

    def delete(self, ids: list[str]) -> None:
        self._vector_index.delete(ids)
