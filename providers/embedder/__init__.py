# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Embedder providers for text embedding.

Includes mock implementations for testing and production implementations.
"""

from core.interfaces import Embedder
from .mock_embedder import MockEmbedder, ZeroEmbedder, MOCK_EMBEDDING_DIM

def get_openai_embedder():
    """Lazy-load OpenAIEmbedder. Requires: pip install openai"""
    from .openai_embedder import OpenAIEmbedder, CachedOpenAIEmbedder
    return OpenAIEmbedder, CachedOpenAIEmbedder

__all__ = [
    "Embedder",
    # Mock implementations
    "MockEmbedder",
    "ZeroEmbedder",
    "MOCK_EMBEDDING_DIM",
    # Lazy loader for OpenAI
    "get_openai_embedder",
]
