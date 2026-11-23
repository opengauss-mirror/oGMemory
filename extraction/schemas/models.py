# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Schema data models for memory type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, List

logger = getLogger(__name__)


class FieldType(str, Enum):
    """Field type enum for schema definitions."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    LIST = "list"


@dataclass(frozen=True)
class SchemaField:
    """Field definition for memory type schemas."""

    name: str
    field_type: FieldType
    required: bool = False
    description: str = ""
    default: Any = None
    enum: List[str] | None = None


@dataclass(frozen=True)
class MemoryTypeSchema:
    """Schema definition for a memory type."""

    memory_type: str
    description: str
    directory: str  # Jinja template for directory path
    filename_template: str  # Jinja template for filename
    operation_mode: str  # "upsert" or "add_only"
    fields: List[SchemaField] = field(default_factory=list)
    enabled: bool = True
    owner_scope: str = "user"  # "user" or "agent"
    version: str = "1.0"

    @property
    def is_single_file(self) -> bool:
        """Check if this schema uses a single-file template."""
        return "{{" not in self.filename_template

    @property
    def is_multi_file(self) -> bool:
        """Check if this schema uses a multi-file template."""
        return "{{" in self.filename_template

    @property
    def is_add_only(self) -> bool:
        """Check if this schema is add-only mode."""
        return self.operation_mode == "add_only"

    @property
    def tool_name(self) -> str:
        """Get the tool name for this memory type."""
        return f"extract_{self.memory_type}"

    @property
    def required_fields(self) -> List[SchemaField]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]

    @property
    def optional_fields(self) -> List[SchemaField]:
        """Get all optional fields."""
        return [f for f in self.fields if not f.required]
