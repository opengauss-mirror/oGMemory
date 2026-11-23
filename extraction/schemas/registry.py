# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Schema registry for loading and managing memory type schemas."""

from logging import getLogger
from pathlib import Path
from typing import List

import yaml

from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField

logger = getLogger(__name__)

SUPPORTED_SCHEMA_MAJOR_VERSIONS = {"1"}


def schema_version_major(version: str | None) -> str:
    """Return the major component of a schema version string."""
    return str(version or "1.0").split(".", 1)[0]


def is_schema_version_compatible(schema: MemoryTypeSchema) -> bool:
    """Return whether the schema version is supported by this runtime."""
    return schema_version_major(schema.version) in SUPPORTED_SCHEMA_MAJOR_VERSIONS


class SchemaRegistry:
    """Registry for memory type schemas loaded from YAML files."""

    def __init__(self, schemas_dir: str | None = None):
        """Initialize the schema registry.

        Args:
            schemas_dir: Directory containing schema YAML files.
                        Defaults to extraction/schemas/definitions/ relative to package.
        """
        self._schemas: dict[str, MemoryTypeSchema] = {}

        if schemas_dir is None:
            # Default to extraction/schemas/definitions/ relative to this package
            import extraction.schemas as schemas_package

            schemas_dir = str(Path(schemas_package.__file__).parent / "definitions")

        # Load schemas from directory if it exists
        if Path(schemas_dir).exists():
            self._load_from_directory(schemas_dir)
        else:
            logger.warning(f"Schema directory not found: {schemas_dir}")

    def _load_from_directory(self, dir_path: str) -> int:
        """Load all schema YAML files from a directory.

        Args:
            dir_path: Path to directory containing schema YAML files.

        Returns:
            Number of schemas loaded.
        """
        count = 0
        dir_path_obj = Path(dir_path)

        if not dir_path_obj.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return 0

        # Load *.yaml files
        for yaml_file in sorted(dir_path_obj.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                schema = self._parse_schema(data)
                self.register(schema)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        # Load *.yml files
        for yaml_file in sorted(dir_path_obj.glob("*.yml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                schema = self._parse_schema(data)
                self.register(schema)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {count} schemas from {dir_path}")
        return count

    def _parse_schema(self, data: dict) -> MemoryTypeSchema:
        """Parse schema from YAML data.

        Args:
            data: Dictionary from loaded YAML file.

        Returns:
            Parsed MemoryTypeSchema instance.
        """
        if not isinstance(data, dict):
            raise ValueError("schema file must contain a YAML mapping")

        memory_type = data.get("memory_type", data.get("name", ""))
        if not memory_type:
            raise ValueError("schema missing required field: memory_type")
        for key in ("description", "directory", "filename_template"):
            if not data.get(key):
                raise ValueError(f"schema {memory_type!r} missing required field: {key}")

        # Parse fields
        fields_data = data.get("fields", [])
        if not isinstance(fields_data, list):
            raise ValueError(f"schema {memory_type!r} field 'fields' must be a list")
        fields = []

        for idx, field_data in enumerate(fields_data):
            if not isinstance(field_data, dict):
                raise ValueError(f"schema {memory_type!r} field #{idx + 1} must be a mapping")
            if not field_data.get("name"):
                raise ValueError(f"schema {memory_type!r} field #{idx + 1} missing field name")
            field = SchemaField(
                name=field_data.get("name", ""),
                field_type=FieldType(field_data.get("type", "string")),
                required=field_data.get("required", False),
                description=field_data.get("description", ""),
                default=field_data.get("default"),
                enum=field_data.get("enum"),
            )
            fields.append(field)

        # Parse main schema
        return MemoryTypeSchema(
            memory_type=memory_type,
            description=data.get("description", ""),
            directory=data.get("directory", ""),
            filename_template=data.get("filename_template", ""),
            operation_mode=data.get("operation_mode", "upsert"),
            fields=fields,
            enabled=data.get("enabled", data.get("enable", True)),
            owner_scope=data.get("owner_scope", "user"),
            version=str(data.get("version", "1.0")),
        )

    def register(self, schema: MemoryTypeSchema) -> None:
        """Register a schema.

        Args:
            schema: MemoryTypeSchema instance to register.
        """
        self._schemas[schema.memory_type] = schema
        logger.debug(f"Registered schema: {schema.memory_type}")

    def unregister(self, memory_type: str) -> MemoryTypeSchema | None:
        """Remove one schema from this registry instance."""

        return self._schemas.pop(memory_type, None)

    def get(self, memory_type: str) -> MemoryTypeSchema | None:
        """Get a schema by memory type name.

        Args:
            memory_type: Name of the memory type.

        Returns:
            MemoryTypeSchema if found, None otherwise.
        """
        return self._schemas.get(memory_type)

    def get_compatible(self, memory_type: str) -> MemoryTypeSchema | None:
        """Get a schema only when its version is compatible with this runtime."""
        schema = self.get(memory_type)
        if schema is None:
            return None
        if not self.is_compatible(schema):
            logger.warning(
                "Ignoring schema %s: incompatible schema version %s",
                schema.memory_type,
                schema.version,
            )
            return None
        return schema

    def is_compatible(self, schema: MemoryTypeSchema) -> bool:
        """Return whether a schema can be used by this runtime."""
        return is_schema_version_compatible(schema)

    def list_all(self) -> List[MemoryTypeSchema]:
        """List all registered schemas.

        Returns:
            List of all MemoryTypeSchema instances.
        """
        return list(self._schemas.values())

    def list_enabled(self) -> List[MemoryTypeSchema]:
        """List only enabled schemas.

        Returns:
            List of enabled MemoryTypeSchema instances.
        """
        return [s for s in self._schemas.values() if s.enabled]

    def list_compatible_enabled(self) -> List[MemoryTypeSchema]:
        """List enabled schemas whose versions are supported by this runtime."""
        return [s for s in self.list_enabled() if self.is_compatible(s)]

    def list_incompatible_enabled(self) -> List[MemoryTypeSchema]:
        """List enabled schemas skipped because their versions are unsupported."""
        return [s for s in self.list_enabled() if not self.is_compatible(s)]
