# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Schema-driven memory type definitions for ContextEngine."""

from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField
from extraction.schemas.registry import (
    SUPPORTED_SCHEMA_MAJOR_VERSIONS,
    SchemaRegistry,
    is_schema_version_compatible,
    schema_version_major,
)

__all__ = [
    "FieldType",
    "MemoryTypeSchema",
    "SchemaField",
    "SUPPORTED_SCHEMA_MAJOR_VERSIONS",
    "SchemaRegistry",
    "is_schema_version_compatible",
    "schema_version_major",
]
