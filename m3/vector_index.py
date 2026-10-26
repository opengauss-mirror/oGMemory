"""M3 vector-index construction helpers.

Centralises the "how do I turn an :class:`M3RuntimeConfig` into a
:class:`VectorIndex`" question so the M3 CLI drivers and the
pipeline runner don't each open-code the selection logic.

Currently supported backends:

- ``opengauss`` — Huawei openGauss. **The recommended production
  target** for the M3 vector index. The M3 benchmarks were
  validated against openGauss; deployments that already operate
  one should keep using it.
- ``pgvector`` — vanilla PostgreSQL 15+ with the pgvector
  extension. Convenience alias for users who want to share a
  single Postgres instance between the graph store and the
  vector index instead of running a separate openGauss cluster.
  Routes to the same :class:`OpenGaussVectorIndex` because the
  SQL it emits (CREATE EXTENSION vector, vector(N), hnsw, MERGE
  INTO) is portable across the two servers via the standard PG
  wire protocol.

ChromaDB and Qdrant are deliberately not exposed here yet; the M3
schema/filter semantics (``video_id`` × ``before_clip`` ×
``extractor_type``) need explicit SQL columns that openGauss /
pgvector's GIN + HNSW combo handles natively, but which Chroma's
closed-allowlist ``_build_where`` cannot represent.

To add a new backend:

1. Implement the :class:`VectorIndex` Protocol (see
   ``core/interfaces.py:175``).
2. Add a branch below keyed off ``runtime.vector_index_backend``.
3. Surface any backend-specific knobs as new fields on
   :class:`M3RuntimeConfig` — keep the YAML shape flat so users
   don't have to learn nested types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.vector_index import (
    ChromaVectorIndex,
    InMemoryVectorIndex,
    OpenGaussVectorIndex,
)

if TYPE_CHECKING:
    from core.interfaces import VectorIndex

    from m3.config import M3RuntimeConfig


def build_m3_vector_index(runtime: "M3RuntimeConfig") -> "VectorIndex":
    """Construct a :class:`VectorIndex` from the M3 runtime config.

    The returned instance is fully constructed (``_ensure_table`` has
    already run for OpenGauss) so callers can immediately upsert /
    search without a second bootstrap step. The caller owns the
    lifecycle — close any pool / persistent-client resources when
    the pipeline ends (OpenGaussVectorIndex cleans up on GC; Chroma
    flushes per upsert so no explicit close is needed).

    Raises:
        NotImplementedError: If ``runtime.vector_index_backend`` is
            a backend we don't support yet. Adding one means a new
            branch below.
        ValueError: If a required connection parameter is missing.
    """
    backend = (runtime.vector_index_backend or "opengauss").lower()

    if backend in ("opengauss", "pgvector"):
        # Backend selection controls extension bootstrap through the SQL dialect.
        if not runtime.vector_index_connection_string:
            raise ValueError(
                "vector_db.connection_string is required for the "
                f"{backend} backend; configure it in ogmemory.yaml "
                "(or export the env var it references) so M3 can "
                "reuse the same Postgres / openGauss instance as the "
                "graph store."
            )
        if runtime.embedding_dimension is None:
            raise ValueError(
                "m3.embedding.dimension is required so the "
                "vector column can be sized at table-creation time; "
                "set it in ogmemory.yaml."
            )
        return OpenGaussVectorIndex(
            connection_string=runtime.vector_index_connection_string,
            table_name=runtime.vector_index_table_name,
            pool_size=runtime.vector_index_pool_size,
            dimension=runtime.embedding_dimension,
            dialect="postgresql" if backend == "pgvector" else "opengauss",
        )

    if backend == "chroma":
        # Chroma is exposed for parity with the seed-retriever path
        # but M3's filter semantics (``before_clip`` ≤ N) currently
        # aren't supported by Chroma's `_build_where` allowlist. M3
        # callers selecting chroma will silently lose the
        # before-clip scope cut and pull more candidates than
        # expected — log a warning so it's visible.
        import logging

        logging.getLogger(__name__).warning(
            "m3.vector_index.backend=chroma is supported but ChromaDB "
            "does not implement the M3 ``before_clip`` filter; the QA "
            "scope cut will be applied post-fetch in Python instead of "
            "pushed down to the index."
        )
        # Reuse a default collection distinct from the directory
        # index so the two workloads don't share storage.
        return ChromaVectorIndex(
            collection_name="m3_episodic_vectors",
            persist_directory=".ogmem_data/chroma/m3",
            dimension=runtime.embedding_dimension or 1536,
        )

    if backend == "memory":
        # Used only by unit tests — no persistence, no cross-process
        # sharing. The constructor is lightweight so a test fixture
        # can create one per test without teardown.
        return InMemoryVectorIndex(dimension=runtime.embedding_dimension or 384)

    raise NotImplementedError(
        f"M3 vector_index backend {runtime.vector_index_backend!r} is "
        "not implemented; supported: opengauss (default), chroma, "
        "memory (tests only)."
    )
