# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Stable identity for a memory node or one independently retrievable unit."""


def canonical_uri(uri: str) -> str:
    uri = uri.rstrip("/")
    for suffix in ("/content.md", "/.abstract.md", "/.overview.md"):
        if uri.endswith(suffix):
            return uri[:-len(suffix)]
    return uri


def memory_unit_key(metadata: dict | None) -> str:
    metadata = metadata or {}
    return str(metadata.get("unit_key") or metadata.get("agent_memory_unit_key") or "")


def memory_identity(uri: str, metadata: dict | None = None) -> tuple[str, str]:
    return canonical_uri(uri), memory_unit_key(metadata)
