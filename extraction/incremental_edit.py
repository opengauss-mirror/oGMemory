"""Exact local edits of complete snapshots supplied to extraction."""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields

from core.errors import ToolCallProtocolError
from core.models import CandidateMemory, ContextNode, RequestContext
from core.validation import MAX_ABSTRACT_LEN, MAX_OVERVIEW_LEN
from extraction.schemas.registry import SchemaRegistry, schema_version_major

INCREMENTAL_TOOLS = frozenset({"edit_memory", "append_memory"})
UPDATE_CATEGORIES = frozenset({"profile", "entity", "preference", "pattern"})
_COMMON_FIELDS = {"target_uri", "expected_version", "abstract", "overview", "confidence",
                  "evidence_quote", "attributed_speaker", "attribution_basis"}
_FRESH_EVIDENCE = {"evidence_quote", "attributed_speaker", "attribution_basis", "provenance_ids"}


def build_incremental_tools() -> list[dict]:
    common = {
        "target_uri": {"type": "string", "minLength": 1},
        "expected_version": {"type": "integer", "minimum": 1},
        "abstract": {"type": "string", "minLength": 1, "maxLength": MAX_ABSTRACT_LEN},
        "overview": {"type": "string", "minLength": 1, "maxLength": MAX_OVERVIEW_LEN},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_quote": {"type": "string", "minLength": 1},
        "attributed_speaker": {"type": "string", "minLength": 1},
        "attribution_basis": {"type": "string", "enum": ["self_first_person", "self_named", "other_named"]},
    }
    description = (
        "Update ONLY a profile, entity, preference or pattern whose full body and version "
        "were successfully read in this extraction. Identity and unchanged facts are kept "
        "locally. Never use a catalog, list or summary as read authorization. Return complete "
        "updated abstract and overview summarizing the whole resulting memory. For profile "
        "supply NEW source evidence_quote, attributed_speaker and attribution_basis; never "
        "copy old evidence as support for a new change. For preference supply current evidence "
        "when available. Multiple edits to one read version are applied in call order. "
        "event/case, skill/tool and new memories require their extract_* tools. "
    )
    tools = []
    for name, fields, instruction in (
        ("edit_memory", {"search": {"type": "string", "minLength": 1},
                         "replacement": {"type": "string"}},
         "Replace one exact substring in content. search must occur exactly once; no fuzzy matching."),
        ("append_memory", {"text": {"type": "string", "minLength": 1}},
         "Append nonempty text to content. An exact text already present is not appended again."),
    ):
        tools.append({"name": name, "description": description + instruction,
                      "input_schema": {"type": "object", "properties": {**common, **fields},
                                       "required": ["target_uri", "expected_version", "abstract", "overview", "confidence", *fields],
                                       "additionalProperties": False}})
    return tools


def incremental_read_key(inputs: dict) -> tuple[str, int]:
    if not isinstance(inputs, dict):
        raise ToolCallProtocolError("incremental memory update requires object arguments")
    uri, version = inputs.get("target_uri"), inputs.get("expected_version")
    if not isinstance(uri, str) or not uri.strip() or type(version) is not int or version < 1:
        raise ToolCallProtocolError("incremental memory update requires a read URI and positive integer version")
    return uri, version


def materialize_incremental_input(
    name: str, inputs: dict, registry: SchemaRegistry,
    read_nodes: dict[str, ContextNode] | None, ctx: RequestContext | None,
    current_content: str | None = None,
) -> tuple[str, dict]:
    """Produce schema input without reading storage or accepting identity from the model."""
    def reject(reason):
        raise ToolCallProtocolError("incremental memory update " + reason)

    fields = {"search", "replacement"} if name == "edit_memory" else {"text"}
    if not isinstance(inputs, dict) or inputs.keys() - (_COMMON_FIELDS | fields):
        reject("contains unsupported fields")
    uri, version = incremental_read_key(inputs)
    if ctx is None:
        reject("requires a read URI and positive integer version")
    node = (read_nodes or {}).get(uri)
    if (node is None or node.uri != uri or type(node.metadata.get("version")) is not int
            or node.metadata["version"] != version):
        reject("requires the complete snapshot of that successful read")
    schema = registry.get_compatible(node.category)
    if (node.category not in UPDATE_CATEGORIES or schema is None or not schema.enabled
            or schema.operation_mode != "upsert"
            or not (schema.is_single_file if node.category == "profile" else schema.is_multi_file)):
        reject("does not support this category or schema")
    stored_schema = node.metadata.get("schema_version")
    if stored_schema is not None and schema_version_major(stored_schema) != schema_version_major(schema.version):
        reject("has an incompatible stored schema")
    owner = ctx.user_space_name() if schema.owner_scope == "user" else ctx.agent_space_name()
    from core.uri_resolver import URIResolver

    resolver = URIResolver(registry)
    directory = (resolver.resolve(node.category, {}, ctx) if schema.is_single_file
                 else resolver.get_directory_uri(node.category, ctx).rstrip("/"))
    valid_path = uri == directory if schema.is_single_file else uri.rsplit("/", 1)[0] == directory
    expected_context_type = "SKILL" if node.category == "pattern" else "MEMORY"
    if node.owner_space != owner or node.context_type != expected_context_type or not valid_path or not resolver.validate_uri(uri, ctx):
        reject("must retain the current category and owner namespace")
    if not all(isinstance(value, str) and value.strip() for value in (node.abstract, node.overview, node.content)):
        reject("requires a complete nonempty read body")
    for field in ("abstract", "overview"):
        if not isinstance(inputs.get(field), str) or not inputs[field].strip():
            reject("requires complete abstract and overview")
    confidence = inputs.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        reject("requires confidence between zero and one")
    for field in _FRESH_EVIDENCE & inputs.keys():
        if not isinstance(inputs[field], str) or not inputs[field].strip():
            reject("requires nonempty current evidence and attribution")
    content = node.content if current_content is None else current_content
    if name == "edit_memory":
        search, replacement = inputs.get("search"), inputs.get("replacement")
        if not isinstance(search, str) or not search.strip() or not isinstance(replacement, str):
            reject("requires nonempty search and string replacement")
        start = content.find(search)
        if start < 0 or content.find(search, start + 1) >= 0:
            reject("search must match exactly once")
        content = content[:start] + replacement + content[start + len(search):]
    else:
        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            reject("requires nonempty append text")
        if text not in content:
            content += ("" if content.endswith("\n") else "\n") + text
    if not content.strip():
        reject("cannot leave empty content")
    routing_key = node.metadata.get("routing_key")
    if not isinstance(routing_key, str) or not routing_key.strip():
        reject("requires the stored routing identity")
    # Retain schema fields and identity, but obtain evidence only from this call.
    full = {field.name: node.metadata[field.name] for field in schema.fields
            if field.name in node.metadata and field.name not in _FRESH_EVIDENCE}
    full.update({key: value for key, value in inputs.items() if key not in fields})
    full.update(content=content, routing_key=routing_key,
                object_name=node.metadata.get("object_name", ""), scope_ref=node.metadata.get("scope_ref", ""))
    return schema.tool_name, full


class _ComposedIncrementalCandidate(CandidateMemory):
    # Not a dataclass field: source checks must not change serialized memory or digests.
    source_steps: tuple[CandidateMemory, ...]


def incremental_source_steps(candidate: CandidateMemory) -> tuple[CandidateMemory, ...]:
    return candidate.source_steps if isinstance(candidate, _ComposedIncrementalCandidate) else (candidate,)


def coalesce_incremental_edits(entries: list[tuple[bool, CandidateMemory]]) -> list[CandidateMemory]:
    """Keep the final composed edit per read version; reject mixed snapshot contracts."""
    modes: dict[str, bool] = {}
    positions: dict[tuple[str, int | None], int] = {}
    result = []
    for incremental, candidate in entries:
        uri = candidate.target_uri
        if uri:
            if uri in modes and modes[uri] != incremental:
                raise ToolCallProtocolError("cannot mix incremental and complete snapshot operations for one target")
            modes[uri] = incremental
        key = (uri, candidate.expected_version)
        if incremental and key in positions:
            position = positions[key]
            previous = result[position]
            composed = _ComposedIncrementalCandidate(**{
                field.name: getattr(candidate, field.name) for field in dataclass_fields(CandidateMemory)
            })
            composed.confidence = min(previous.confidence, candidate.confidence)
            composed.source_steps = incremental_source_steps(previous) + incremental_source_steps(candidate)
            result[position] = composed
        else:
            if incremental:
                positions[key] = len(result)
            result.append(candidate)
    return result
