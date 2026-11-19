"""Optional vector projection for Outbox records."""

from core.interfaces import Embedder, VectorIndex
from core.models import IndexRecord


class VectorProjection:
    def __init__(self, vector_index: VectorIndex, embedder: Embedder) -> None:
        self._vector_index = vector_index
        self._embedder = embedder

    def embed(self, records: list[IndexRecord]) -> list[list[float]]:
        return self._embedder.embed_texts([record.text for record in records])

    def upsert(self, records: list[IndexRecord]) -> None:
        self._vector_index.upsert(records)

    def delete(self, ids: list[str]) -> None:
        self._vector_index.delete(ids)
