# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Dynamic tool definition builder from SchemaRegistry.

Generates OpenAI function calling tool definitions from schema metadata.
SchemaRegistry is the single source of truth for extraction tools.
"""

from logging import getLogger

from core.errors import ToolCallProtocolError
from core.models import CandidateMemory, ContextNode, RequestContext
from core.validation import (
    MAX_ABSTRACT_LEN,
    MAX_CONTENT_LEN,
    MAX_OVERVIEW_LEN,
    validate_attribution,
    validate_candidate,
)
from extraction.incremental_edit import (
    INCREMENTAL_TOOLS,
    build_incremental_tools,
    materialize_incremental_input,
)
from extraction.schemas.models import FieldType, MemoryTypeSchema
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
_UPDATE_MEMORY_TYPES = {"profile", "entity", "preference", "pattern"}
_TEXT_LIMITS = {"abstract": MAX_ABSTRACT_LEN, "overview": MAX_OVERVIEW_LEN,
                "content": MAX_CONTENT_LEN}
_UPDATE_INSTRUCTION = (
    "Bound upserts: copy read target_uri, expected_version, routing_key, exact scope_ref "
    "(even empty) and preference holder. Return complete content preserving unchanged "
    "facts; abstract/overview summarize it all. Never replace scope with a conversation "
    "date. Different scopes need separate unbound facts, without old-scope facts."
)
_PROFILE_SNAPSHOT_INSTRUCTION = (
    "Runtime storage contract: profile has ONE fixed file. When the input contains "
    "eligible, source-supported profile attributes, emit exactly ONE "
    "extract_profile operation containing ALL supported profile attributes in this input, "
    "both at creation and update. This single-file contract takes precedence over the "
    "attribute-per-call and separate-node examples; preserve their fact eligibility "
    "and attribution rules. "
)


def _profile_snapshot_instruction(include_updates: bool) -> str:
    return _PROFILE_SNAPSHOT_INSTRUCTION + (
        "For updates, read the existing profile and retain its unchanged facts. "
        "Use its routing_key; for creation use 'identity'." if include_updates
        else "For creation use routing_key='identity'."
    )


def _supports_read_target(schema: MemoryTypeSchema) -> bool:
    return (schema.operation_mode == "upsert" and schema.memory_type in _UPDATE_MEMORY_TYPES
            and (schema.is_single_file if schema.memory_type == "profile" else schema.is_multi_file))


def _source_identity_properties(schema: MemoryTypeSchema) -> dict:
    """Expose identity metadata only for the memory categories that use it."""
    if not schema.is_multi_file or schema.memory_type not in {"entity", "preference", "pattern", "event", "case"}:
        return {}
    properties = {
        "scope_ref": {"type": "string", "description":
            "Identity namespace, NOT evidence provenance. Use ONE explicitly established "
            "business/project/world boundary only when it distinguishes different same-named "
            "subjects or preference contexts. Otherwise use an empty string, including ordinary "
            "conversations about the same people across sessions. A source document title, "
            "chat label, message ID or observation date does not establish an identity namespace. "
            "Keep source references in provenance_ids and dates with the facts. A document title "
            "may be a scope only if the document itself defines a distinct world or namespace. "
            "Reuse the read identity scope for updates; never join independent scopes."},
    }
    if schema.memory_type == "entity":
        properties["object_name"] = {"type": "string", "description":
            "Stable source name of the one person, organization, project or object "
            "described. Copy its exact original name across updates: for Atlas in "
            "Project A, use object_name='Atlas' and scope_ref='Project A'. Never "
            "append project labels, descriptions or qualifiers to the name. "
            "Always include this field; use an empty string only for an unnamed subject."}
    if schema.memory_type in {"event", "case"}:
        properties["title"] = {"type": "string", "description":
            "Concise source-grounded title of this occurrence or case."}
    return properties


def build_update_instruction(registry: SchemaRegistry, categories=None, *, include_updates: bool = True) -> str:
    """Describe runtime write units independently of persisted schema fields."""
    schemas = [schema for schema in registry.list_enabled()
               if is_schema_version_compatible(schema) and _supports_read_target(schema)
               and (categories is None or schema.memory_type in categories)]
    parts = [_UPDATE_INSTRUCTION] if schemas and include_updates else []
    if any(schema.memory_type == "profile" and schema.is_single_file for schema in schemas):
        parts.append(_profile_snapshot_instruction(include_updates))
    return "\n\n".join(parts)


def build_extraction_tools(registry: SchemaRegistry, *, include_updates: bool = True,
                           topic_drafts: bool = False, activation_enabled: bool = False) -> list[dict]:
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

            if field.name in _TEXT_LIMITS:
                prop_def["maxLength"] = _TEXT_LIMITS[field.name]
                prop_def["description"] = (
                    prop_def.get("description", "")
                    + f" Maximum {_TEXT_LIMITS[field.name]} characters. "
                    "Keep category-relevant facts in content, not source narration; summarize in abstract/overview."
                )

            if field.enum:
                prop_def["enum"] = field.enum

            properties[field.name] = prop_def

            if field.required:
                required.append(field.name)

        for name, definition in _source_identity_properties(schema).items():
            if name not in properties:
                properties[name] = definition
                required.append(name)
        if schema.memory_type == "preference" and "who" in properties:
            properties["who"]["description"] = (
                properties["who"].get("description", "") + " "
                "The person, group or organization holding this preference, identified "
                "by the source. Use its name, preserving the existing holder on updates. "
                "Keep reporting-speaker evidence separate from the preference statement."
            )
        description = schema.description
        if schema.memory_type == "entity" and schema.is_multi_file:
            description += (
                "\nRuntime identity contract: each call describes exactly ONE subject in ONE identity namespace. "
                "The same subject mentioned in another conversation remains the same entity. "
                "Same-named objects belonging to different projects are distinct nodes: emit separate "
                "calls, each with its exact object_name and its own scope_ref and facts. "
                "For Atlas in Project A and another Atlas in Project B, emit two calls; never "
                "create a third aggregate Atlas node or a combined 'Project A / Project B' scope. "
                "This applies to both creation and updates. When only one scoped object changes, "
                "update that object without combining it with the unchanged object."
            )
        if _supports_read_target(schema):
            if include_updates:
                properties.update({
                    "target_uri": {"type": "string", "description": "Existing memory URI read for this update."},
                    "expected_version": {"type": "integer", "minimum": 1,
                                         "description": "Version from the memory read result."},
                })
                description += "\n" + _UPDATE_INSTRUCTION
            if schema.memory_type == "profile" and schema.is_single_file:
                description = _profile_snapshot_instruction(include_updates) + "\n\n" + description
        if activation_enabled and schema.memory_type != "relation":
            properties["update_activation"] = {
                "type": "string", "minLength": 1,
                "description": "One compact source-grounded declarative sentence describing this memory's explicit subject, concrete fact, and stated distinguishing context. Preserve uncertainty and negation. Generate alongside this candidate; do not invent scope, future queries, or links.",
            }
            required.append("update_activation")
        if topic_drafts and schema.memory_type in {"entity", "preference"}:
            properties.pop("overview", None)
            required = [name for name in required if name != "overview"]
            properties["abstract"]["description"] = (
                "Brief identity and topic descriptor for retrieving this subject, at most 200 characters. "
                "Preserve explicit plan/intent wording for intended or upcoming actions."
            )
            properties["content"]["description"] = (
                "Source fact draft: one atomic fact per line. Retain every supported exact name, "
                "relationship, date, number, condition, change and uncertainty once. "
                "Keep each action's stated status: intended, upcoming, ongoing, or completed. "
                "Preserve the holder and project identity of each fact. The node processor "
                "will organize these facts into the final memory card."
            )
            description += (
                "\nSource-draft output contract: abstract identifies the subject/topic for recall; "
                "content carries its complete source facts once. Keep exact identity fields and "
                "evidence_quote. Final card formatting and whole-node summaries belong to the later node processor."
            )

        tools.append({
            "name": schema.tool_name,
            "description": description,
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

    if include_updates and any(is_schema_version_compatible(schema) and _supports_read_target(schema)
                               for schema in registry.list_enabled()):
        incremental_tools = build_incremental_tools()
        if activation_enabled:
            descriptor = next(tool["input_schema"]["properties"]["update_activation"]
                              for tool in tools if "update_activation" in tool["input_schema"]["properties"])
            for tool in incremental_tools:
                tool["input_schema"]["properties"]["update_activation"] = dict(descriptor)
                tool["input_schema"]["required"].append("update_activation")
        tools.extend(incremental_tools)
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
    read_nodes: dict[str, ContextNode] | None = None,
    ctx: RequestContext | None = None,
    current_content: str | None = None,
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
    if tool_name in INCREMENTAL_TOOLS:
        # Keep dev's strict edit validation unchanged: the graph-only descriptor
        # is validated by the complete snapshot parser, not as an edit command.
        incremental_input = dict(tool_input)
        activation = incremental_input.pop("update_activation", None) if activation_enabled else None
        name, full_input = materialize_incremental_input(
            tool_name, incremental_input, registry, read_nodes, ctx, current_content,
        )
        if activation_enabled:
            full_input["update_activation"] = activation
        parsed = (parse_tool_call(name, full_input, registry, activation_enabled=True)
                  if activation_enabled else parse_tool_call(name, full_input, registry))
        if parsed is None:
            raise ToolCallProtocolError("incremental update did not produce a schema-valid complete snapshot")
        category, owner, candidate = parsed
        try:
            candidate = validate_candidate(candidate)
        except Exception as exc:
            raise ToolCallProtocolError("incremental update failed complete snapshot validation") from exc
        attributed = validate_attribution(candidate, user_id=ctx.user_id if ctx else None)
        if attributed.category != category or attributed.owner_scope != owner:
            raise ToolCallProtocolError("incremental update cannot change the read target's attribution")
        return category, owner, candidate

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

    identity_fields = {}
    for name in _source_identity_properties(schema):
        value = tool_input.get(name, validated_fields.get(name))
        if value is None:
            value = ""
        if not isinstance(value, str):
            errors.append(f"{name}: expected source identity text")
        else:
            identity_fields[name] = value

    if errors:
        logger.warning(f"Validation failed for tool {tool_name}: {errors}")
        return None

    target_uri = tool_input.get("target_uri", "")
    expected_version = tool_input.get("expected_version")
    if target_uri or expected_version is not None:
        if (not _supports_read_target(schema)
                or not isinstance(target_uri, str) or not target_uri.strip()
                or type(expected_version) is not int or expected_version < 1
                or not all(isinstance(tool_input.get(name), str) and tool_input[name].strip()
                           for name in ("abstract", "overview", "content"))):
            logger.warning("Invalid read target for tool %s", tool_name)
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
            target_uri=target_uri,
            expected_version=expected_version,
            **identity_fields,
            metadata={"update_activation": activation,
                      "update_activation_version": "c1-integrated-activation-v1"}
            if activation is not None else {},
        ),
    )


def validate_read_target(candidate: CandidateMemory, read_versions: dict[str, int],
                         *, read_nodes=None, ctx=None) -> None:
    """Bind a complete update to the snapshot actually supplied to extraction."""
    if candidate.target_uri and (candidate.target_uri not in read_versions
                                 or read_versions[candidate.target_uri] != candidate.expected_version):
        raise ToolCallProtocolError("memory update requires the URI and version of a successful read")

    if candidate.target_uri and read_nodes is not None and candidate.category in {"entity", "preference", "pattern"}:
        from commit.identity_resolver import MemoryIdentityResolver
        node = read_nodes.get(candidate.target_uri)
        if node is None or ctx is None or not MemoryIdentityResolver._in_scope(node, candidate, ctx):
            raise ToolCallProtocolError(
                "memory update must preserve the read target category, owner, source scope_ref "
                "(including an empty value) and preference holder")
