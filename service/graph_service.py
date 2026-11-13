"""Canonical ContextNode relation reads for memory graph retrieval."""


class GraphService:
    """Expose authorized relation-store reads to graph retrieval."""

    def __init__(self, graph_store=None, vector_index=None, *, relation_reader=None):
        self._relation_reader = relation_reader

    @property
    def supports_relation_reads(self) -> bool:
        return self._relation_reader is not None

    def get_relation_neighbors(self, uri, ctx, *, limit=10, min_weight=0.0):
        if self._relation_reader is None:
            raise RuntimeError("semantic relation reader is not configured")
        return self._relation_reader.get_relation_neighbors(
            uri, ctx, limit=limit, min_weight=min_weight,
        )

    def get_traversal_neighbors(self, uri, ctx, *, limit=128, embedding_profile=None):
        if self._relation_reader is None:
            raise RuntimeError("semantic relation reader is not configured")
        return self._relation_reader.get_traversal_neighbors(
            uri, ctx, limit=limit, embedding_profile=embedding_profile,
        )
