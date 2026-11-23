# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

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
