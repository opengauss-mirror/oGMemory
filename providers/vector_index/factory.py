"""Construction and registration for vector providers."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from core.interfaces import VectorIndex

VectorFactory = Callable[[Any], VectorIndex]

_PROVIDERS: dict[str, VectorFactory] = {}
_PROVIDER_LOCK = RLock()


def _normalize_provider_name(name: str) -> str:
    key = name.strip().casefold()
    if not key:
        raise ValueError("vector provider name must not be empty")
    return key


def register_vector_provider(name: str, provider_factory: VectorFactory) -> None:
    """Register a vector provider factory."""

    key = _normalize_provider_name(name)
    if not callable(provider_factory):
        raise TypeError("vector provider factory must be callable")
    with _PROVIDER_LOCK:
        if key in _PROVIDERS:
            raise ValueError(f"vector provider {key!r} is already registered")
        _PROVIDERS[key] = provider_factory


def _create_memory(config) -> VectorIndex:
    from providers.vector_index.in_memory_index import InMemoryVectorIndex

    return InMemoryVectorIndex(dimension=config.opengauss_dimension)


def _create_opengauss(config) -> VectorIndex:
    if not config.opengauss_connection_string:
        raise ValueError(
            "OPENGAUSS_CONNECTION_STRING is required when vector_db_type='opengauss'"
        )

    from providers.vector_index.opengauss_index import OpenGaussVectorIndex

    return OpenGaussVectorIndex(
        connection_string=config.opengauss_connection_string,
        dimension=config.opengauss_dimension,
        table_name=config.opengauss_table_name,
        pool_size=config.opengauss_pool_size,
        dialect=getattr(config, "vector_db_dialect", None),
    )


def _create_chroma(config) -> VectorIndex:
    from providers.vector_index.chroma_index import ChromaVectorIndex

    return ChromaVectorIndex(
        collection_name=config.chroma_collection_name,
        persist_directory=config.chroma_persist_directory,
        dimension=config.embedding_dimension,
    )


register_vector_provider("memory", _create_memory)
register_vector_provider("opengauss", _create_opengauss)
register_vector_provider("chroma", _create_chroma)


def create_vector_index(config) -> VectorIndex:
    """Create the configured vector index."""

    name = _normalize_provider_name(str(config.vector_db_type))
    with _PROVIDER_LOCK:
        provider_factory = _PROVIDERS.get(name)
        if provider_factory is None:
            available = ", ".join(sorted(_PROVIDERS))
            raise ValueError(
                f"Unknown vector_db_type: {name!r}; available providers: {available}"
            )
    return provider_factory(config)
