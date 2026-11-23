# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations:
- MemoryWriteAPI: commit_session for writing memories

For browse operations:
- MemoryFS: file system-like memory browsing

For search/read operations:
- ReadAPI: search_memory, read_memory with vector search

For async index synchronization:
- IndexService: manages OutboxScheduler for background index updates
"""

from service.api import (
    MemoryWriteAPI,
    ReadAPI,
    init_write_api,
    get_write_api,
)
from service.memory_fs import MemoryFS
from service.index_service import (
    IndexService,
    init_index_service,
    get_index_service,
    setup_signal_handlers,
)

__all__ = [
    "MemoryWriteAPI",
    "ReadAPI",
    "MemoryFS",
    "init_write_api",
    "get_write_api",
    "IndexService",
    "init_index_service",
    "get_index_service",
    "setup_signal_handlers",
]
