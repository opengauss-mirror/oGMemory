# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Dynamic tool definition builder from SchemaRegistry.

Generates OpenAI function calling tool definitions from schema metadata.
SchemaRegistry is the single source of truth for extraction tools.
"""

from logging import getLogger

from core.models import CandidateMemory
from extraction.schemas.models import FieldType
from extraction.schemas.registry import (
    SchemaRegistry,
    is_schema_version_compatible,
)

logger = getLogger(__name__)

# Field type mapping to JSON Schema types
_FIELD_TYPE_JSON_SCHEMA = {
    FieldType.STRING: "string",
    FieldType.NUMBER: "number",
    FieldType.INTEGER: "integer",
    FieldType.BOOLEAN: "boolean",
    FieldType.LIST: "array",
}

# Fields kept in schema definitions for internal metadata, but not exposed
# to the LLM as tool-call arguments.
_EXCLUDED_TOOL_FIELDS = {"provenance_ids"}

# Schemas used by storage/internal flows, not by LLM extraction.
_INTERNAL_MEMORY_TYPES = {"session_archive", "session_summary"}


def build_extraction_tools(registry: SchemaRegistry, *, activation_enabled: bool = False) -> list[dict]:
    """Generate OpenAI function calling tool definitions from enabled schemas.

    Each schema generates one tool definition with:
    - name: schema.tool_name (extract_{memory_type})
    - description: schema.description
    - input_schema: JSON Schema derived from schema.fields

    Args:
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        List of tool definition dicts compatible with OpenAI function calling
    """
    tools = []

    for schema in registry.list_enabled():
        if schema.memory_type in _INTERNAL_MEMORY_TYPES:
            continue
        # Relation proposals belong to graph-on ingestion, not memory storage.
        if schema.memory_type == "relation" and not activation_enabled:
            continue

        if not is_schema_version_compatible(schema):
            logger.warning(
                "Skipping tool %s: incompatible schema version %s",
                schema.tool_name,
                schema.version,
            )
            continue

        # Build JSON Schema for input parameters
        properties = {}
        required = []

        for field in schema.fields:
            if field.name in _EXCLUDED_TOOL_FIELDS:
                continue

            prop_def = {
                "type": _FIELD_TYPE_JSON_SCHEMA.get(field.field_type, "string"),
            }

            if field.description:
                prop_def["description"] = field.description

            if field.enum:
                prop_def["enum"] = field.enum

            properties[field.name] = prop_def

            if field.required:
                required.append(field.name)

        if activation_enabled and schema.memory_type != "relation":
            properties["update_activation"] = {
                "type": "string", "minLength": 1,
                "description": "One compact source-grounded declarative sentence describing this memory's explicit subject, concrete fact, and stated distinguishing context. Preserve uncertainty and negation. Generate alongside this candidate; do not invent scope, future queries, or links.",
            }
            required.append("update_activation")

        tools.append({
            "name": schema.tool_name,
            "description": schema.description,
            "schema_version": schema.version,
            "metadata": {
                "memory_type": schema.memory_type,
                "schema_version": schema.version,
            },
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })

        logger.debug(f"Built tool definition: {schema.tool_name}")

    return tools


def build_tool_to_category(registry: SchemaRegistry) -> dict[str, tuple[str, str]]:
    """Generate tool_name -> (memory_type, owner_scope) mapping.

    Args:
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        Dict mapping tool_name to (memory_type, owner_scope) tuple
    """
    return {
        schema.tool_name: (schema.memory_type, schema.owner_scope)
        for schema in registry.list_compatible_enabled()
        if schema.memory_type not in _INTERNAL_MEMORY_TYPES and schema.memory_type != "relation"
    }


def parse_tool_call(
    tool_name: str,
    tool_input: dict,
    registry: SchemaRegistry,
    *,
    activation_enabled: bool = False,
) -> tuple[str, str, CandidateMemory] | None:
    """Validate and parse a tool call using schema definitions.

    Args:
        tool_name: Name of the tool being called (e.g., "extract_profile")
        tool_input: Raw input dict from the tool call
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        Tuple of (memory_type, owner_scope, CandidateMemory) if valid,
        None if tool_name is unknown or validation fails
    """
    # Find a compatible schema by tool_name
    schema = None
    for s in registry.list_compatible_enabled():
        if s.memory_type in _INTERNAL_MEMORY_TYPES:
            continue
        if s.memory_type == "relation":
            continue  # handled as a proposal, never a CandidateMemory
        if s.tool_name == tool_name:
            schema = s
            break

    if schema is None:
        logger.warning(f"Unknown tool name: {tool_name}")
        return None

    activation = None
    if activation_enabled:
        value = tool_input.get("update_activation")
        if not isinstance(value, str) or not value.strip() or "\n" in value.strip() or "\r" in value.strip():
            logger.warning("Invalid or missing update_activation for %s", tool_name)
            return None
        activation = value.strip()

    # Pre-fill auto-generatable fields before validation so required checks pass
    tool_input = dict(tool_input)  # copy to avoid mutating caller's dict
    explicit_routing_key = tool_input.get("routing_key")
    if explicit_routing_key is not None:
        explicit_routing_key = str(explicit_routing_key)
        tool_input["routing_key"] = explicit_routing_key
    if "confidence" not in tool_input or tool_input.get("confidence") is None:
        tool_input["confidence"] = 0.5
    if "content" not in tool_input or not tool_input.get("content"):
        tool_input["content"] = tool_input.get("abstract", "")
    if "overview" not in tool_input or not tool_input.get("overview"):
        tool_input["overview"] = tool_input.get("abstract", "")
    # Auto-generate routing_key from abstract
    abstract = tool_input.get("abstract", "")
    if abstract:
        slug = "_".join(abstract.lower().split()[:3]).replace(",", "").replace(".", "")
    else:
        slug = "unnamed"
    if "routing_key" not in tool_input or not tool_input.get("routing_key"):
        tool_input["routing_key"] = slug

    # Validate fields against schema
    validated_fields = {}
    errors = []

    for field in schema.fields:
        if field.name in tool_input:
            value = tool_input[field.name]

            if value is None and field.required:
                errors.append(f"{field.name}: required field must not be null")
                continue

            # Type validation
            if field.field_type == FieldType.STRING:
                if value is not None and not isinstance(value, str):
                    try:
                        value = str(value)
                    except Exception:
                        errors.append(f"{field.name}: expected string, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.NUMBER:
                if value is not None and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{field.name}: expected number, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.INTEGER:
                if value is not None and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{field.name}: expected integer, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.BOOLEAN:
                if value is not None and not isinstance(value, bool):
                    # Try to parse common string representations
                    if isinstance(value, str):
                        value_lower = value.lower()
                        if value_lower in ("true", "1", "yes"):
                            value = True
                        elif value_lower in ("false", "0", "no"):
                            value = False
                        else:
                            errors.append(f"{field.name}: expected boolean, got '{value}'")
                            continue
                    else:
                        errors.append(f"{field.name}: expected boolean, got {type(value).__name__}")
                        continue

            # Enum validation
            if field.enum and value not in field.enum:
                errors.append(f"{field.name}: value '{value}' not in enum {field.enum}")
                continue

            validated_fields[field.name] = value

        elif field.required:
            errors.append(f"{field.name}: required field missing")
            if field.default is not None:
                validated_fields[field.name] = field.default
        elif field.default is not None:
            validated_fields[field.name] = field.default

    if errors:
        logger.warning(f"Validation failed for tool {tool_name}: {errors}")
        return None

    # Extract common fields for CandidateMemory
    abstract = validated_fields.get("abstract", "")
    overview = validated_fields.get("overview", "")
    content = validated_fields.get("content", "")
    confidence = validated_fields.get("confidence", 0.5)

    routing_key = explicit_routing_key or validated_fields.get("routing_key", "")

    # Fallback: generate slug from abstract if no routing_key
    if not routing_key:
        slug = "_".join(abstract.lower().split()[:3]).replace(",", "").replace(".", "")
        routing_key = slug or "unnamed"
        logger.info(f"Auto-generated routing_key='{routing_key}' for {tool_name} from abstract")

    # Handle temporal/actor/location fields
    when = validated_fields.get("when")
    who = validated_fields.get("who")
    where = validated_fields.get("where")

    # Handle evidence/attribution fields.
    evidence_quote = validated_fields.get("evidence_quote") if schema.memory_type in ("profile", "preference") else None
    attributed_speaker = validated_fields.get("attributed_speaker") if schema.memory_type == "profile" else None
    attribution_basis = validated_fields.get("attribution_basis") if schema.memory_type == "profile" else None

    # Handle tool-specific fields (for 'tool' category)
    tool_stats = None
    if schema.memory_type == "tool":
        tool_stats = {}
        tool_stat_fields = ["best_for", "optimal_params", "common_failures",
                           "recommendation"]
        for stat_field in tool_stat_fields:
            if stat_field in validated_fields and validated_fields[stat_field]:
                tool_stats[stat_field] = validated_fields[stat_field]
        if tool_stats:
            logger.debug(f"Included tool_stats for {tool_name}: {list(tool_stats.keys())}")

    return (
        schema.memory_type,
        schema.owner_scope,
        CandidateMemory(
            category=schema.memory_type,
            owner_scope=schema.owner_scope,
            routing_key=routing_key,
            abstract=abstract,
            overview=overview,
            content=content,
            confidence=confidence,
            when=when,
            who=who,
            where=where,
            tool_stats=tool_stats,
            evidence_quote=evidence_quote,
            attributed_speaker=attributed_speaker,
            attribution_basis=attribution_basis,
            provenance_ids=validated_fields.get("provenance_ids", []),
            metadata={"update_activation": activation,
                      "update_activation_version": "c1-integrated-activation-v1"}
            if activation is not None else {},
        ),
    )
