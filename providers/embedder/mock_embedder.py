# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Mock Embedder for testing.

Returns fixed-dimension vectors for reproducible testing.
Each unique text gets a deterministic vector based on its hash.
"""

import hashlib
from typing import Final

from core.interfaces import Embedder

# Fixed embedding dimension for mock (matches common small models)
MOCK_EMBEDDING_DIM: Final[int] = 384


class MockEmbedder(Embedder):
    """Mock embedder for testing.

    Returns deterministic vectors based on text hash.
    Each unique text produces a consistent vector across runs.
    """

    def __init__(self, dimension: int = MOCK_EMBEDDING_DIM) -> None:
        """Initialize mock embedder.

        Args:
            dimension: Vector dimension (default: 384).
        """
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of vectors, one per input text.
            Vectors are deterministic based on text content.
        """
        results: list[list[float]] = []

        for text in texts:
            # Create deterministic vector from text hash
            vector = self._hash_to_vector(text)
            results.append(vector)

        return results

    def embed_texts_with_timeout(
        self,
        texts: list[str],
        *,
        timeout_seconds: float,
    ) -> list[list[float]]:
        """Expose the non-blocking test provider to candidate deadline callers."""

        if timeout_seconds <= 0:
            raise TimeoutError("embedding request deadline elapsed")
        return self.embed_texts(texts)

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to deterministic vector via hash.

        Uses SHA256 hash of text to seed a pseudo-random vector.
        Ensures same text always produces same vector.
        """
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to full dimension
        vector: list[float] = []
        for i in range(self._dimension):
            # Use hash bytes cyclically
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            normalized = (byte_val - 128) / 128.0
            vector.append(normalized)

        return vector


class ZeroEmbedder(Embedder):
    """Mock embedder that returns zero vectors.

    Useful for testing isolation of vector effects.
    All texts get identical zero vectors.
    """

    def __init__(self, dimension: int = MOCK_EMBEDDING_DIM) -> None:
        """Initialize zero embedder.

        Args:
            dimension: Vector dimension (default: 384).
        """
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return zero vectors for all texts.

        Args:
            texts: List of input texts (ignored).

        Returns:
            List of zero vectors, one per input text.
        """
        return [[0.0] * self._dimension for _ in texts]

    def embed_texts_with_timeout(
        self,
        texts: list[str],
        *,
        timeout_seconds: float,
    ) -> list[list[float]]:
        """Expose the non-blocking test provider to candidate deadline callers."""

        if timeout_seconds <= 0:
            raise TimeoutError("embedding request deadline elapsed")
        return self.embed_texts(texts)
