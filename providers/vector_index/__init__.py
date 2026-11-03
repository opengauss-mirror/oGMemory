"""Vector Index providers.

Implementations of the VectorIndex protocol for different backends.
"""

from providers.vector_index.chroma_index import ChromaVectorIndex
from providers.vector_index.factory import create_vector_index, register_vector_provider
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.vector_index.milvus_index import MilvusVectorIndex
from providers.vector_index.opengauss_index import OpenGaussVectorIndex

__all__ = [
    "ChromaVectorIndex",
    "MilvusVectorIndex",
    "InMemoryVectorIndex",
    "OpenGaussVectorIndex",
    "create_vector_index",
    "register_vector_provider",
]
