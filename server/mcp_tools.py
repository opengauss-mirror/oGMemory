"""Tool handlers for the oGMemory MCP adapter."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import quote, unquote

from core.models import RetrievedBlock
from core.resource_identity import ResourceIdentityError, normalize_posix_relative_path
from providers.token_tracker import request_usage_scope
from retrieval.context_reader import project_memory_unit
from retrieval.identity import memory_unit_key
from retrieval.result_ranker import is_temporal_query
from server.mcp_resources import (
    MEMORY_TYPES,
    external_id_from_uri,
    external_to_internal_uri_candidates,
    internal_to_external_uri,
    iter_internal_uri_candidates,
    parse_ogmemory_uri,
    public_type_from_internal,
    render_resource,
)
from server.mcp_security import (
    CODE_EVIDENCE_UNTRUSTED_NOTICE,
    UNTRUSTED_DATA_NOTICE,
    build_mcp_context,
    is_not_found_or_not_visible_error,
    raise_not_found_or_not_visible,
    with_untrusted_data_notice,
)
from server.mcp_trace import capture_write_trace
from skills.artifact import canonical_skill_uri

logger = logging.getLogger(__name__)

_FORBIDDEN_WRITE_FIELDS = {
    "type",
    "memory_type",
    "memory_id",
    "category",
    "context_type",
    "owner_space",
    "level",
    "metadata",
    "uri",
    "abstract",
    "overview",
    "owner_scope",
    "routing_key",
    "confidence",
    "when",
    "who",
    "where",
    "tool_stats",
    "evidence_quote",
    "attributed_speaker",
    "attribution_basis",
    "provenance_ids",
    "object_name",
    "scope_ref",
    "target_uri",
    "expected_version",
    "legacy_routing_key",
    "title",
    "complete_occurrence_context",
    "write_operation_id",
    "source_occurrence_id",
    "verified_routing_aliases",
    "verified_object_aliases",
    "source_contexts",
    "node_draft_indices",
}
_WRITE_ARGUMENT_FIELDS = {
    "session_id", "text", "messages", "agent_id", "client_request_id", "force_extract", "wait",
}
_REVIEWED_SKILL_NOTICE = (
    "This is a published Agent Skill from the oGMemory skill catalog. "
    "Activate it only when its description matches the current task."
)
_CODE_QUERY_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_CODE_QUERY_NORMALIZERS = (
    (re.compile(rf"^function\s+({_CODE_QUERY_IDENTIFIER})$", re.IGNORECASE), "{symbol} function"),
    (re.compile(rf"^def\s+({_CODE_QUERY_IDENTIFIER})$", re.IGNORECASE), "{symbol} function"),
    (
        re.compile(rf"^async\s+function\s+({_CODE_QUERY_IDENTIFIER})$", re.IGNORECASE),
        "{symbol} async function",
    ),
    (
        re.compile(rf"^async\s+def\s+({_CODE_QUERY_IDENTIFIER})$", re.IGNORECASE),
        "{symbol} async function",
    ),
    (re.compile(rf"^type\s+({_CODE_QUERY_IDENTIFIER})$", re.IGNORECASE), "{symbol} type"),
)
_CODE_SEARCH_STATUSES = {
    "disabled",
    "ok",
    "zero_result",
    "partial_failure",
    "error",
}
_CODE_LOCATOR_WITH_LINES = re.compile(r"^(?P<path>.+):(?P<start>[1-9][0-9]*)-(?P<end>[1-9][0-9]*)$")
_CODE_RENDERED_LOCATOR_WITH_LINES = re.compile(
    r"^(?P<path>[^\r\n]+?) \(lines [1-9][0-9]*-[1-9][0-9]*\)"
    r"(?:(?: symbol| signature)=[^\r\n]*)*$"
)
_CODE_RENDERED_LOCATOR_QUALIFIERS = re.compile(
    r"^(?:(?:symbol|signature)=[^\r\n]*)(?: (?:symbol|signature)=[^\r\n]*)*$"
)
_SAFE_ROOT_SLUG = re.compile(r"[A-Za-z0-9._-]+")
_URI_OR_DRIVE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_CODE_EXCERPT_CLIPPED_MARKER = "\n... [excerpt clipped] ..."


class SkillCatalogService(Protocol):
    @property
    def default_agent_id(self) -> str: ...

    @property
    def skill_search_scan_limit(self) -> int: ...

    def get_auth_service(self): ...

    def build_context(self, params, identity=None): ...

    def list_published_skills(self, ctx, *, limit: int, cursor: str | None): ...

    def read_published_skill(self, ctx, name: str, source: str | None = None): ...

    def list_skill_candidates(self, ctx, state: str | None = None): ...

    def add_managed_skill(self, ctx, data=None, **kwargs: Any): ...

    def update_managed_skill(self, ctx, name: str, data=None, **kwargs: Any): ...

    def delete_published_skill(self, ctx, name: str, source: str | None): ...


def list_memories(
    service: Any,
    *,
    limit: int = 100,
    cursor: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List a page of memory nodes visible to the current MCP identity."""

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    if cursor:
        parsed_cursor = parse_ogmemory_uri(cursor.partition("#")[0])
        if parsed_cursor.kind != "item" or parsed_cursor.memory_type == "skill":
            raise ValueError("cursor must reference an ogmemory:// memory item")

    _validate_request_size(
        service,
        {
            "limit": limit,
            "cursor": cursor,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    try:
        entries, has_more = _list_memory_node_entries_page(
            service,
            ctx,
            limit=limit,
            cursor=cursor,
        )
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise
    memories = [summary for _, summary in entries]
    payload = {
        "memories": memories,
        "count": len(memories),
        "truncated": has_more,
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }
    if has_more and memories:
        payload["next_cursor"] = _memory_list_cursor(entries[-1])
    return _truncate_memory_list_payload(
        payload, _configured_max_response_chars(service), entries
    )


def _list_memory_node_entries_page(
    service: Any,
    ctx: Any,
    *,
    limit: int,
    cursor: str | None,
) -> tuple[list[tuple[str, dict[str, Any]]], bool]:
    context_fs = service._get_context_fs()
    order_key = getattr(context_fs, "list_node_order_key", lambda uri: uri)
    candidates: list[tuple[str, dict[str, Any]]] = []
    fetch_limit = limit + 1
    cursor_key = _memory_list_cursor_key(context_fs, cursor, ctx)

    def append_nodes(base_uri: str, external_prefix: str) -> None:
        after_uri = None
        if cursor_key:
            if cursor_key.startswith(external_prefix):
                after_uri = f"{base_uri.rstrip('/')}/{cursor_key.removeprefix(external_prefix)}"
            elif cursor_key > external_prefix:
                return
        if hasattr(context_fs, "list_node_summaries_page"):
            nodes = context_fs.list_node_summaries_page(
                base_uri,
                ctx,
                limit=fetch_limit,
                after_uri=after_uri,
            )
        else:
            child_uris = sorted(context_fs.list_children(base_uri, ctx))
            if after_uri:
                child_uris = [uri for uri in child_uris if uri > after_uri]
            nodes = [
                context_fs.read_node(child_uri, ctx)
                for child_uri in child_uris[:fetch_limit]
            ]
        for node in nodes:
            sort_key = external_prefix + order_key(node.uri).rsplit("/", 1)[-1]
            if cursor_key is None or sort_key > cursor_key:
                candidates.append((sort_key, _memory_node_summary(node)))

    for memory_type, spec in MEMORY_TYPES.items():
        if memory_type == "skill":
            continue
        if memory_type == "profile":
            internal_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            external_uri = "ogmemory://memories/profile/self"
            if (cursor_key is None or external_uri > cursor_key) and context_fs.exists(
                internal_uri, ctx
            ):
                node = context_fs.read_node(internal_uri, ctx)
                candidates.append((external_uri, _memory_node_summary(node)))
            continue
        if memory_type == "pattern":
            append_nodes(
                f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/patterns",
                "ogmemory://memories/pattern/",
            )
            if ctx.agent_id:
                append_nodes(
                    f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/patterns",
                    "ogmemory://memories/pattern/agent/",
                )
            continue

        owner_id = ctx.agent_id if spec.owner == "agent" else ctx.user_id
        if not owner_id:
            continue
        owner_path = "agents" if spec.owner == "agent" else "users"
        append_nodes(
            f"ctx://{ctx.account_id}/{owner_path}/{owner_id}/memories/{spec.directory}",
            f"ogmemory://memories/{spec.public_type}/",
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[:limit], len(candidates) > limit


def _memory_list_cursor_key(context_fs: Any, cursor: str | None, ctx: Any) -> str | None:
    if not cursor:
        return None
    public_cursor, separator, storage_hint = cursor.partition("#")
    cursor_uris = external_to_internal_uri_candidates(public_cursor, ctx)
    stored_cursor = cursor_uris[0]
    if separator:
        if not storage_hint.startswith("stored="):
            raise ValueError("Invalid memory list cursor")
        stored_name = unquote(storage_hint.removeprefix("stored="))
        if unquote(stored_name) != parse_ogmemory_uri(public_cursor).memory_id:
            raise ValueError("Invalid memory list cursor")
        stored_cursor = f"{stored_cursor.rsplit('/', 1)[0]}/{stored_name}"
    elif "%" in public_cursor.rsplit("/", 1)[-1]:
        stored_cursor = next(
            (uri for uri in iter_internal_uri_candidates(public_cursor, ctx, context_fs)
             if context_fs.exists(uri, ctx)),
            stored_cursor,
        )
    order_key = getattr(context_fs, "list_node_order_key", lambda uri: uri)
    cursor_key = public_cursor.rsplit("/", 1)[0] + "/" + order_key(stored_cursor).rsplit("/", 1)[-1]
    if public_cursor == "ogmemory://memories/profile/self":
        return public_cursor
    return cursor_key


def _memory_list_cursor(entry: tuple[str, dict[str, Any]]) -> str:
    sort_key, summary = entry
    public_uri = summary["uri"]
    stored_name = sort_key.rsplit("/", 1)[-1]
    if stored_name == unquote(public_uri.rsplit("/", 1)[-1]):
        return public_uri
    return f"{public_uri}#stored={quote(stored_name, safe='')}"


def _memory_node_summary(node: Any) -> dict[str, Any]:
    external_uri = internal_to_external_uri(node.uri, category=node.category)
    return {
        "uri": external_uri,
        "type": public_type_from_internal(node.category, node.uri),
        "id": external_id_from_uri(external_uri),
        "abstract": node.abstract,
        "updated_at": node.metadata.get("updated_at", ""),
        "has_content": bool(node.content) or bool(node.metadata.get("has_content")),
    }


def search_skills(
    service: SkillCatalogService,
    *,
    query: str,
    top_k: int = 5,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Search only the published Agent Skill catalog."""

    if not (query or "").strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    rows, catalog_truncated = _all_published_skill_summaries(
        service,
        agent_id=agent_id,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    query_tokens = set(_search_tokens(query))
    for row in rows:
        haystack = set(_search_tokens(f"{row['name']} {row.get('description', '')}"))
        row["score"] = len(query_tokens & haystack) / max(1, len(query_tokens | haystack))
    skills = sorted(rows, key=lambda row: (-float(row["score"]), row["name"]))[:top_k]
    return {
        "query": query.strip(),
        "skills": skills,
        "truncated": catalog_truncated,
        "artifact_notice": _REVIEWED_SKILL_NOTICE,
    }


def list_skills(
    service: SkillCatalogService,
    *,
    agent_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List published Agent Skills without loading full SKILL.md bodies."""

    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to list skills")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    payload = service.list_published_skills(ctx, limit=limit, cursor=cursor)
    for row in payload["skills"]:
        row["uri"] = canonical_skill_uri(row["name"], row.get("source"))
    return payload


def read_skill(
    service: SkillCatalogService,
    *,
    name: str,
    source: str | None = None,
    include_files: bool = False,
    agent_id: str | None = None,
    max_chars: int = 50000,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Read one reviewed Agent Skill as a SKILL.md artifact."""

    if max_chars < 1 or max_chars > 50000:
        raise ValueError("max_chars must be between 1 and 50000")
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to read skills")
    node = (
        service.read_published_skill(ctx, name)
        if source is None
        else service.read_published_skill(ctx, name, source=source)
    )
    resolved_source = "imported" if node.metadata.get("ogmemory_managed") else "generated"
    uri = canonical_skill_uri(name, resolved_source)
    content = node.content[:max_chars]
    result = {
        "name": name,
        "description": node.abstract,
        "uri": uri,
        "source": resolved_source,
        "content": content,
        "truncated": len(node.content) > len(content),
        "artifact_type": "agent_skill",
        "review_status": "published",
        "artifact_notice": _REVIEWED_SKILL_NOTICE,
    }
    if include_files and resolved_source == "imported":
        from skills.management import (
            ManagedSkill,
            deserialize_package_file,
            package_file_manifest,
        )

        package = ManagedSkill(
            name=name,
            description=node.abstract,
            content=node.content,
            files=tuple(
                deserialize_package_file(record)
                for record in node.metadata.get("package_files") or []
            ),
        )
        result["files"] = package_file_manifest(package, source="imported")
    return result


def list_skill_candidates(
    service: SkillCatalogService,
    *,
    state: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List candidate summaries; candidate trace bodies are never returned."""

    if state and state not in {"accumulating", "held"}:
        raise ValueError("state must be one of: accumulating, held")
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to list skill candidates")
    candidates = service.list_skill_candidates(ctx, state=state)
    return {"candidates": candidates}


def install_skill(
    service: SkillCatalogService,
    *,
    data: str | dict[str, Any] | None = None,
    archive_base64: str | None = None,
    path: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Install an imported Agent Skill package."""

    _validate_request_size(
        service,
        {
            "data": data,
            "archive_base64": archive_base64,
            "path": path,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if path is not None and not allow_stdio_env_auth:
        raise PermissionError("Skill path imports require trusted stdio; use data or archive_base64")
    return service.add_managed_skill(
        ctx,
        data,
        archive_base64=archive_base64,
        path=path,
    )


def update_skill(
    service: SkillCatalogService,
    *,
    name: str,
    data: str | dict[str, Any] | None = None,
    archive_base64: str | None = None,
    path: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Replace an imported Agent Skill package."""

    _validate_request_size(
        service,
        {
            "name": name,
            "data": data,
            "archive_base64": archive_base64,
            "path": path,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if path is not None and not allow_stdio_env_auth:
        raise PermissionError("Skill path imports require trusted stdio; use data or archive_base64")
    return service.update_managed_skill(
        ctx,
        name,
        data,
        archive_base64=archive_base64,
        path=path,
    )


def delete_skill(
    service: SkillCatalogService,
    *,
    name: str,
    source: str,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Delete one published Agent Skill from the selected source."""

    _validate_request_size(
        service,
        {"name": name, "source": source, "agent_id": agent_id},
    )
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    return service.delete_published_skill(ctx, name, source=source)


def _all_published_skill_summaries(
    service: SkillCatalogService, **kwargs: Any
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    cursor = None
    scan_limit = service.skill_search_scan_limit
    while True:
        page = list_skills(
            service, limit=min(100, scan_limit - len(rows)), cursor=cursor, **kwargs
        )
        rows.extend(page["skills"])
        if len(rows) >= scan_limit:
            return rows[:scan_limit], bool(page.get("truncated"))
        if not page.get("truncated"):
            return rows, False
        next_cursor = page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            raise RuntimeError("skill catalog returned an invalid pagination cursor")
        cursor = next_cursor


def _search_tokens(value: str) -> list[str]:
    import re

    lowered = value.casefold()
    words = re.findall(r"[^\W_]{2,}", lowered, flags=re.UNICODE)
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered)
    cjk_bigrams = [
        run[index:index + 2]
        for run in cjk_runs
        for index in range(max(1, len(run) - 1))
    ]
    return words + cjk_bigrams


def search_code(
    service: Any,
    *,
    query: str,
    scope: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any] | str:
    if not (query or "").strip():
        raise ValueError("query must not be empty")
    if scope is not None and not isinstance(scope, Mapping):
        raise ValueError("scope must be a mapping")
    if budget is not None and not isinstance(budget, Mapping):
        raise ValueError("budget must be a mapping")
    _validate_request_size(
        service,
        {
            "query": query,
            "scope": scope,
            "budget": budget,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    normalized_query = _canonical_code_query(query)
    result = _public_result_mapping(
        service.search_code_resources(
            {
                "query": normalized_query,
                "scope": dict(scope or {}),
                "budget": _code_search_budget_for_response(service, budget),
                "_ctx": ctx,
            }
        ),
        operation="code search",
    )
    return _render_code_search_result(
        result,
        query=normalized_query,
        max_chars=_configured_max_response_chars(service),
    )


def _serialized_chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))


def _resize_original_output_recovery(
    payload: dict[str, Any],
    keep_chars: int,
) -> dict[str, Any]:
    result = dict(payload)
    original = result.get("original")
    if not isinstance(original, str):
        raise RuntimeError("original-output recovery returned an invalid result")
    bounded_original = original[:max(0, keep_chars)]
    truncated = bool(result.get("truncated")) or len(bounded_original) < len(original)
    result["original"] = bounded_original
    result["returned_chars"] = len(bounded_original)
    result["truncated"] = truncated

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        bounded_diagnostics = dict(diagnostics)
        bounded_diagnostics["output_chars"] = len(bounded_original)
        bounded_diagnostics["truncated"] = truncated
        result["diagnostics"] = bounded_diagnostics
    return result


def _bound_original_output_recovery_response(
    payload: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    result = with_untrusted_data_notice(payload)
    if _serialized_chars(result) <= max_chars:
        return result

    original = result.get("original")
    if not isinstance(original, str):
        raise ValueError(
            "original-output recovery response limit is too small for response envelope"
        )
    empty = _resize_original_output_recovery(result, 0)
    if _serialized_chars(empty) > max_chars:
        raise ValueError(
            "original-output recovery response limit is too small for response envelope"
        )

    low, high = 0, len(original)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _resize_original_output_recovery(result, middle)
        if _serialized_chars(candidate) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return _resize_original_output_recovery(result, low)


def _truncate_content_record(row: dict[str, Any], keep_chars: int) -> dict[str, Any]:
    import hashlib

    result = dict(row)
    original = str(result.get("content", ""))
    keep_chars = max(0, keep_chars)
    original_offset = int(
        result.get("actual_offset_bytes", result.get("offset_bytes", 0)) or 0
    )
    start_char = 0
    if keep_chars < len(original) and "match_offset_bytes" in result:
        encoded_original = original.encode("utf-8")
        relative_match = int(result.get("match_offset_bytes", original_offset)) - original_offset
        relative_match = max(0, min(len(encoded_original), relative_match))
        while (
            relative_match > 0
            and relative_match < len(encoded_original)
            and encoded_original[relative_match] & 0xC0 == 0x80
        ):
            relative_match -= 1
        match_start = len(encoded_original[:relative_match].decode("utf-8"))
        match_length_bytes = max(1, int(result.get("match_length_bytes", 1) or 1))
        match_end_bytes = min(len(encoded_original), relative_match + match_length_bytes)
        while (
            match_end_bytes > 0
            and match_end_bytes < len(encoded_original)
            and encoded_original[match_end_bytes] & 0xC0 == 0x80
        ):
            match_end_bytes -= 1
        match_end = max(
            match_start + 1,
            len(encoded_original[:match_end_bytes].decode("utf-8")),
        )
        match_chars = max(1, match_end - match_start)
        if keep_chars >= match_chars:
            available = keep_chars - match_chars
            start_char = max(0, match_start - available // 2)
            start_char = min(start_char, max(0, len(original) - keep_chars))
        else:
            start_char = match_start
    content = original[start_char:start_char + keep_chars]
    encoded = content.encode("utf-8")
    offset = original_offset + len(original[:start_char].encode("utf-8"))
    result["content"] = content
    if "offset_bytes" in result:
        result["offset_bytes"] = offset
    if "actual_offset_bytes" in result:
        result["actual_offset_bytes"] = offset
    if "actual_length_bytes" in result:
        result["actual_length_bytes"] = len(encoded)
    if "next_offset_bytes" in result:
        result["next_offset_bytes"] = offset + len(encoded)
    if "decoded_chars" in result:
        result["decoded_chars"] = len(content)
    if "content_sha256" in result:
        result["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    if "chunk_sha256" in result:
        result["chunk_sha256"] = hashlib.sha256(encoded).hexdigest()
    if "eof" in result and keep_chars < len(original):
        result["eof"] = False
    if "fragment_complete" in result and keep_chars < len(original):
        result["fragment_complete"] = False
        empty_sha = hashlib.sha256(b"").hexdigest()
        if "context_before" in result:
            result["context_before"] = ""
            result["context_before_offset_bytes"] = offset
            result["context_before_sha256"] = empty_sha
        if "context_after" in result:
            result["context_after"] = ""
            result["context_after_offset_bytes"] = offset + len(encoded)
            result["context_after_sha256"] = empty_sha
    if "returned_chars" in result:
        result["returned_chars"] = (
            len(content)
            + len(str(result.get("context_before", "")))
            + len(str(result.get("context_after", "")))
        )
    total_bytes = result.get("total_bytes")
    next_offset = result.get("next_offset_bytes")
    if isinstance(total_bytes, int) and isinstance(next_offset, int):
        remaining_bytes = max(0, total_bytes - next_offset)
        continuation_required = remaining_bytes > 0 and not bool(result.get("eof"))
        result["continuation_required"] = continuation_required
        result["remaining_bytes"] = remaining_bytes
        result["next_request"] = (
            {
                "offset_bytes": next_offset,
                "length_bytes": min(
                    remaining_bytes,
                    max(1, int(result.get("actual_length_bytes", 0) or 1)),
                ),
            }
            if continuation_required else None
        )
    return result


def bound_tool_result_read_response(
    payload: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    if _serialized_chars(payload) <= max_chars:
        return payload
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("tool-result response limit is too small for artifact metadata")
    empty = _truncate_content_record(payload, 0)
    if _serialized_chars(empty) > max_chars:
        raise ValueError("tool-result response limit is too small for artifact envelope")
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if _serialized_chars(_truncate_content_record(payload, middle)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return _truncate_content_record(payload, low)


def bound_tool_result_collection_response(
    payload: dict[str, Any],
    key: str,
    max_chars: int,
) -> dict[str, Any]:
    if _serialized_chars(payload) <= max_chars:
        return payload
    items = list(payload.get(key, []) or [])
    result = dict(payload)
    result[key] = []
    result["truncated"] = True
    if _serialized_chars(result) > max_chars:
        raise ValueError("tool-result response limit is too small for response metadata")

    for item in items:
        row = dict(item)
        candidate = dict(result)
        candidate[key] = [*result[key], row]
        if _serialized_chars(candidate) <= max_chars:
            result = candidate
            continue
        content = row.get("content")
        if isinstance(content, str):
            low, high = 0, len(content)
            fitted = None
            while low <= high:
                middle = (low + high) // 2
                bounded_row = _truncate_content_record(row, middle)
                candidate = dict(result)
                candidate[key] = [*result[key], bounded_row]
                if _serialized_chars(candidate) <= max_chars:
                    fitted = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            if fitted is not None and fitted[key][-1].get("content"):
                result = fitted
        break

    if key == "artifacts" and items:
        if not result[key]:
            raise ValueError("tool-result response limit is too small for one artifact summary")
        result["next_cursor"] = result[key][-1].get("artifact_id")
    return result


def list_tool_result_artifacts(
    service: Any,
    *,
    session_id: str,
    agent_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List metadata for verified artifacts owned by one complete session tuple."""
    if not session_id:
        raise ValueError("session_id is required")
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.list_tool_result_artifacts({
        "sessionId": session_id,
        "limit": limit,
        "cursor": cursor,
        "_ctx": ctx,
    })
    payload = {
        "_ogmemory_artifact_list_source": True,
        "schema_version": 1,
        "source_tool": "list_tool_result_artifacts",
        "truncated": False,
        "next_cursor": None,
        **payload,
    }
    max_chars = _configured_max_response_chars(service)
    effective_limit = max_chars
    while effective_limit > 0:
        bounded = bound_tool_result_collection_response(
            payload,
            "artifacts",
            effective_limit,
        )
        attested = service.attest_tool_result_source(bounded, ctx)
        overflow = _serialized_chars(attested) - max_chars
        if overflow <= 0:
            return attested
        effective_limit -= overflow
    raise ValueError("tool-result response limit is too small for list attestation")


def read_tool_result_artifact(
    service: Any,
    *,
    session_id: str,
    ref: str,
    offset_bytes: int = 0,
    length_bytes: int | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Read one bounded UTF-8 byte range.

    A transport response can be shorter than ``length_bytes``.  Callers must
    inspect ``continuation_required``/``eof`` and, while continuation is
    required, repeat the read using ``next_request.offset_bytes`` and
    ``next_request.length_bytes`` before answering from the complete Artifact.
    """
    if not session_id:
        raise ValueError("session_id is required")
    max_chars = _configured_max_response_chars(service)
    resolved_length = (
        int(service._cfg.tool_result_read_default_bytes)
        if length_bytes is None
        else length_bytes
    )
    if resolved_length < 1:
        raise ValueError("length_bytes must be positive")
    # A byte limit also bounds decoded Unicode characters and response size.
    resolved_length = min(resolved_length, max_chars)
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.read_tool_result_artifact({
        "sessionId": session_id,
        "ref": ref,
        "offset_bytes": offset_bytes,
        "length_bytes": resolved_length,
        "_ctx": ctx,
    })
    bounded = bound_tool_result_read_response(payload, max_chars)
    return service.attest_tool_result_source(bounded, ctx)


def search_tool_result_artifacts(
    service: Any,
    *,
    session_id: str,
    query: str,
    top_k: int = 5,
    max_chars: int = 20_000,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Return bounded exact lines, objects, rows, functions, or short paragraphs.

    Call this only when the stub and existing assistant analysis do not contain
    the required exact fact.  Use ``read_tool_result_artifact`` only if these
    precise fragments are still insufficient; never guess missing identifiers,
    numbers, status codes, errors, counts, or ordering.
    """
    if not session_id:
        raise ValueError("session_id is required")
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    response_limit = _configured_max_response_chars(service)
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.search_tool_result_artifacts({
        "sessionId": session_id,
        "query": query,
        "top_k": top_k,
        "max_chars": min(max(1, max_chars), response_limit),
        "_ctx": ctx,
    })
    bounded = bound_tool_result_collection_response(payload, "results", response_limit)
    bounded["results"] = [
        service.attest_tool_result_source(row, ctx)
        for row in bounded.get("results", [])
    ]
    return bounded


def query_memory(
    service: Any,
    *,
    query: str,
    types: list[str] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
    use_scene: bool | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not (query or "").strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    if use_scene is not None and not isinstance(use_scene, bool):
        raise ValueError("use_scene must be boolean")
    requested_types = types or []
    unknown = sorted(t for t in requested_types if t not in MEMORY_TYPES)
    if unknown:
        raise ValueError(f"Unknown memory types: {', '.join(unknown)}")
    _validate_request_size(
        service,
        {
            "query": query,
            "types": requested_types,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    categories = [MEMORY_TYPES[t].internal_category for t in requested_types] or None
    search_top_k = _internal_search_top_k(service, query, top_k)
    type_filters = {}
    if categories and any(
        spec.internal_category in categories and name not in requested_types
        for name, spec in MEMORY_TYPES.items()
    ):
        type_filters["memory_directories"] = sorted({MEMORY_TYPES[t].directory for t in requested_types})

    try:
        with request_usage_scope(ctx):
            result = service.get_read_api().search_memory(
                query.strip(),
                ctx,
                top_k=search_top_k,
                categories=categories,
                score_threshold=score_threshold,
                include_debug=False,
                fill_content_for_top_k=min(top_k, 3),
                **type_filters,
                **({"use_scene": use_scene} if use_scene is not None else {}),
            )
    except Exception as exc:
        recorder = getattr(service, "record_retrieval_metric", None)
        if callable(recorder):
            try:
                recorder(
                    ctx,
                    request_id="",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    scores=[],
                    categories=requested_types,
                    rerank_used=bool(getattr(getattr(service, "_cfg", None), "retrieval_global_result_rerank_enabled", False)),
                    failed=True,
                )
            except Exception:
                logger.debug("retrieval monitoring record failed", exc_info=True)
        raise_not_found_or_not_visible(exc)
        raise

    hits = []
    for hit in result.hits:
        try:
            row = _hit_to_public(hit)
        except ValueError:
            continue
        if requested_types and row["type"] not in requested_types:
            continue
        hits.append(row)
        if len(hits) == top_k:
            break

    payload = {
        "request_id": result.request_id,
        "query": result.query or query,
        "hits": hits,
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }
    response = _truncate_query_payload(payload, _configured_max_response_chars(service))
    recorder = getattr(service, "record_retrieval_metric", None)
    if callable(recorder):
        try:
            recorder(
                ctx,
                request_id=result.request_id,
                duration_ms=(time.perf_counter() - started) * 1000,
                scores=[float(hit.get("score") or 0) for hit in hits],
                categories=[str(hit.get("type") or "unknown") for hit in hits],
                rerank_used=bool(getattr(getattr(service, "_cfg", None), "retrieval_global_result_rerank_enabled", False)),
                trace=result.trace.to_dict() if hasattr(result.trace, "to_dict") else None,
            )
        except Exception:
            logger.debug("retrieval monitoring record failed", exc_info=True)
    return response


def edit_file(
    service: Any,
    *,
    locator: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Delegate one exact source edit through the authenticated facade."""

    if not isinstance(replace_all, bool):
        raise ValueError("replace_all must be a boolean")
    root_slug, file_path = _parse_code_edit_locator(locator)
    _validate_request_size(
        service,
        {
            "locator": locator,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    return _public_result_mapping(
        service.edit_code_file(
            {
                "file_path": file_path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
                "root_slug": root_slug,
                "_ctx": ctx,
            }
        ),
        operation="code file edit",
    )


def read_memory(
    service: Any,
    *,
    uri: str,
    unit_key: str | None = None,
    level: str = "all",
    max_chars: int = 12000,
    recover_original: bool = False,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    if level not in {"abstract", "overview", "content", "all"}:
        raise ValueError("level must be one of: abstract, overview, content, all")
    if max_chars < 1 or max_chars > 50000:
        raise ValueError("max_chars must be between 1 and 50000")
    if not isinstance(recover_original, bool):
        raise ValueError("recover_original must be a boolean")
    if unit_key is not None and (not isinstance(unit_key, str) or not unit_key.strip()):
        raise ValueError("unit_key must be a non-empty string")

    if recover_original:
        if unit_key is not None:
            raise ValueError("unit_key is not supported with recover_original")
        if level != "all":
            raise ValueError("level is not supported with recover_original")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("uri must be a filtered-output memory URI")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise ValueError("max_chars must be an integer with recover_original")
        _validate_request_size(
            service,
            {
                "uri": uri,
                "max_chars": max_chars,
                "recover_original": recover_original,
                "session_id": session_id,
                "agent_id": agent_id,
            },
        )
        max_chars = min(max_chars, _configured_max_response_chars(service))
        # Preserve a context supplied by the authenticated HTTP adapter.  When
        # the wrapper builds the context itself, use that same trusted context
        # for the recovery facade; do not reintroduce the optional raw
        # ``agent_id`` argument after transport/header resolution.  This keeps
        # the documented URI-only recovery call working for authenticated MCP
        # clients that bind the agent through the transport.
        ctx = request_context or build_mcp_context(
            service,
            {"session_id": session_id, "agent_id": agent_id},
            transport_headers=transport_headers,
            allow_stdio_env_auth=allow_stdio_env_auth,
            resolved_identity=resolved_identity,
            use_stdio_agent_default=True,
        )
        trusted_agent_id = getattr(ctx, "agent_id", "")
        try:
            result = _public_result_mapping(
                service.read_tool_output_original(
                    {
                        "memory_uri": uri,
                        "max_chars": max_chars,
                        "sessionId": session_id,
                        "agentId": trusted_agent_id,
                        "_ctx": ctx,
                    }
                ),
                operation="original-output recovery",
            )
            return _bound_original_output_recovery_response(result, max_chars)
        except Exception as exc:
            raise_not_found_or_not_visible(exc)
            raise

    _validate_request_size(
        service,
        {
            "uri": uri,
            "level": level,
            "unit_key": unit_key,
            "max_chars": max_chars,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    max_chars = min(max_chars, _configured_max_response_chars(service))

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    parsed = parse_ogmemory_uri(uri)

    if unit_key is not None and (parsed.kind != "item" or parsed.memory_type not in {
        "tool_memory", "tool_evidence", "experience"
    }):
        raise ValueError("unit_key requires a Tool or Experience memory URI")

    if parsed.kind != "item":
        payload = render_resource(service, uri, ctx)
        return _truncate_payload(payload, max_chars)

    try:
        if parsed.memory_type == "skill" and parsed.memory_id:
            source = parsed.skill_source or "generated"
            node = service.read_published_skill(ctx, parsed.memory_id, source=source)
            canonical_uri = internal_to_external_uri(node.uri, category=node.category)
            payload = {
                "uri": canonical_uri,
                "type": "skill",
                "id": parsed.memory_id,
                "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
            }
            if level in {"abstract", "all"}:
                payload["abstract"] = node.abstract
            if level in {"overview", "all"}:
                payload["overview"] = node.overview
            if level in {"content", "all"}:
                payload["content"] = node.content
            return _truncate_payload(payload, max_chars)

        context_fs = service._get_context_fs()
        last_exc: Exception | None = None
        node = None
        for internal_uri in iter_internal_uri_candidates(uri, ctx, context_fs):
            try:
                needs_full_node = unit_key is not None or (
                    level == "overview" and parsed.memory_type in {"tool_memory", "tool_evidence", "experience"})
                if hasattr(context_fs, "read_node_fields") and not needs_full_node:
                    node = context_fs.read_node_fields(internal_uri, ctx, level)
                else:
                    node = context_fs.read_node(internal_uri, ctx)
                break
            except Exception as exc:
                if not is_not_found_or_not_visible_error(exc):
                    raise
                last_exc = exc
        if node is None:
            if last_exc is not None:
                raise last_exc
            raise ValueError(f"URI does not reference a single memory: {uri}")
        if unit_key is not None:
            node = project_memory_unit(node, unit_key)
        else:
            from retrieval.context_reader import canonical_agent_node
            node = canonical_agent_node(node)
        canonical_uri = internal_to_external_uri(node.uri, category=node.category)
        canonical_parsed = parse_ogmemory_uri(canonical_uri)
        payload = {
            "uri": canonical_uri,
            "type": canonical_parsed.memory_type,
            "id": canonical_parsed.memory_id,
            "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
        }
        if unit_key is not None:
            payload.update(unit_key=unit_key, unit_revision=node.metadata["unit_revision"])
        if level in {"abstract", "all"}:
            payload["abstract"] = node.abstract
        if level in {"overview", "all"}:
            payload["overview"] = node.overview
        if level == "overview" and not node.overview and node.metadata.get("agent_memory_version") == 1:
            payload["content"] = node.content
            payload["level"] = "content"
        if level in {"content", "all"}:
            payload["content"] = node.content
        if unit_key is not None or node.metadata.get("grounding"):
            payload["grounding"] = node.metadata["grounding"]
        return _truncate_payload(payload, max_chars)
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise


def delete_memory(
    service: Any,
    *,
    uri: str,
    reason: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    client_request_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    parsed = parse_ogmemory_uri(uri)
    if parsed.kind != "item":
        raise ValueError("delete_memory requires a single ogmemory://memories/{type}/{id} URI")
    if parsed.memory_type == "skill":
        raise ValueError("generated skills cannot be deleted through ogm_delete_memory")
    _validate_request_size(
        service,
        {
            "uri": uri,
            "reason": reason,
            "session_id": session_id,
            "agent_id": agent_id,
            "client_request_id": client_request_id,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )

    try:
        last_exc: Exception | None = None
        result = None
        context_fs = getattr(service, "_get_context_fs", lambda: None)()
        for internal_uri in iter_internal_uri_candidates(uri, ctx, context_fs):
            try:
                result = service.forget_memory(
                    internal_uri,
                    ctx,
                    external_uri=uri,
                    reason=reason,
                    client_request_id=client_request_id,
                )
                break
            except Exception as exc:
                if not is_not_found_or_not_visible_error(exc):
                    raise
                last_exc = exc
        if result is None:
            if last_exc is not None:
                raise last_exc
            raise ValueError(f"URI does not reference a single memory: {uri}")
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise

    return {
        "ok": bool(result.get("ok", True)),
        "deleted": bool(result.get("deleted", True)),
        "mode": result.get("mode", "archive"),
        "uri": uri,
        "type": parsed.memory_type,
        "id": parsed.memory_id,
        "client_request_id": client_request_id,
    }


def _hit_to_public(hit: RetrievedBlock) -> dict[str, Any]:
    external_uri = internal_to_external_uri(hit.uri, category=hit.category)
    parsed = parse_ogmemory_uri(external_uri)
    row = {
        "uri": external_uri,
        "type": parsed.memory_type or public_type_from_internal(hit.category, hit.uri),
        "id": external_id_from_uri(external_uri),
        "score": hit.score,
        "abstract": hit.abstract,
        "has_content": hit.has_content or bool(hit.content_excerpt),
    }
    if hit.overview:
        row["overview"] = hit.overview
    if hit.content_excerpt:
        row["content_excerpt"] = hit.content_excerpt
    if unit_key := memory_unit_key(hit.metadata):
        row["unit_key"] = unit_key
        for field in ("unit_revision", "grounding"):
            if field in hit.metadata:
                row[field] = hit.metadata[field]
    for field in ("when", "who", "where"):
        value = (hit.metadata or {}).get(field)
        if value:
            row[field] = value
    return row


def _internal_search_top_k(service: Any, query: str, requested_top_k: int) -> int:
    if not is_temporal_query(query):
        return requested_top_k
    max_top_k = getattr(getattr(service.get_read_api(), "_cfg", None), "max_top_k", 50)
    return min(max_top_k, 50, max(requested_top_k * 3, 15))


def _bounded_grounding(sources: list, max_chars: int) -> tuple[list, int]:
    selected, used = [], 0
    for source in sources:
        cost = len(json.dumps(source, ensure_ascii=False))
        if used + cost > max_chars:
            break
        selected.append(source)
        used += cost
    return selected, used


def _truncate_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    remaining = max_chars
    truncated = bool(payload.get("truncated", False))
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str) and key in {"abstract", "overview", "content"}:
            if len(value) > remaining:
                result[key] = value[: max(0, remaining)]
                truncated = True
                remaining = 0
            else:
                result[key] = value
                remaining -= len(value)
        elif key == "grounding" and isinstance(value, list):
            result[key], used = _bounded_grounding(value, remaining)
            remaining -= used
            truncated = truncated or len(result[key]) < len(value)
        else:
            result[key] = value
    result["truncated"] = truncated
    return result


def _truncate_query_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    remaining = max_chars
    truncated = False
    result = dict(payload)
    truncated_hits: list[dict[str, Any]] = []
    for hit in payload.get("hits", []):
        if remaining <= 0:
            truncated = True
            break
        row = dict(hit)
        for field in ("abstract", "overview", "content_excerpt"):
            value = row.get(field)
            if not isinstance(value, str):
                continue
            if len(value) > remaining:
                row[field] = value[: max(0, remaining)]
                truncated = True
                remaining = 0
            else:
                remaining -= len(value)
        if isinstance(row.get("grounding"), list):
            sources = row["grounding"]
            row["grounding"], used = _bounded_grounding(sources, remaining)
            remaining -= used
            truncated = truncated or len(row["grounding"]) < len(sources)
        truncated_hits.append(row)
    result["hits"] = truncated_hits
    result["truncated"] = truncated
    return result


def _truncate_memory_list_payload(
    payload: dict[str, Any], max_chars: int, entries: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    result = dict(payload)
    result.setdefault("truncated", False)
    if len(json.dumps(result, ensure_ascii=False, default=str)) <= max_chars:
        return result

    memories = list(payload.get("memories", []))
    result["memories"] = []
    result["count"] = 0
    result["truncated"] = True
    result.pop("next_cursor", None)
    if len(json.dumps(result, ensure_ascii=False, default=str)) > max_chars:
        raise ValueError(
            "mcp_max_response_chars is too small for ogm_list_memories metadata"
        )
    for memory, entry in zip(memories, entries):
        candidate = dict(result)
        candidate["memories"] = [*result["memories"], memory]
        candidate["count"] = len(candidate["memories"])
        candidate["next_cursor"] = _memory_list_cursor(entry)
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) > max_chars:
            break
        result = candidate
    if not result["memories"]:
        raise ValueError(
            "mcp_max_response_chars is too small for one ogm_list_memories item"
        )
    return result


def _configured_max_request_bytes(service: Any) -> int:
    value = int(getattr(service._cfg, "mcp_max_request_bytes", 1048576) or 1048576)
    return max(1, value)


def _configured_max_response_chars(service: Any) -> int:
    value = int(getattr(service._cfg, "mcp_max_response_chars", 50000) or 50000)
    return max(1, min(value, 50000))


def _canonical_code_query(query: str) -> str:
    """Normalize supported symbol-query forms for canonical code search."""

    normalized = str(query or "").strip()
    for pattern, template in _CODE_QUERY_NORMALIZERS:
        match = pattern.fullmatch(normalized)
        if match is not None:
            return template.format(symbol=match.group(1))
    return normalized


def _safe_code_path(value: object) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or raw.startswith("/")
        or "\\" in raw
        or "\x00" in raw
        or _URI_OR_DRIVE_PREFIX.match(raw)
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        return ""
    try:
        return str(normalize_posix_relative_path(raw))
    except ResourceIdentityError:
        return ""


def _parse_code_edit_locator(locator: object) -> tuple[str, str]:
    """Decode one rendered, root-qualified code locator for the edit facade."""

    raw = str(locator or "").strip()
    match = _CODE_LOCATOR_WITH_LINES.fullmatch(raw)
    if match is not None:
        raw = match.group("path")
    else:
        rendered_match = _CODE_RENDERED_LOCATOR_WITH_LINES.fullmatch(raw)
        if rendered_match is not None:
            raw = rendered_match.group("path")
        else:
            qualifier_indexes = [
                index
                for marker in (" symbol=", " signature=")
                if (index := raw.find(marker)) >= 0
            ]
            if qualifier_indexes:
                qualifier_index = min(qualifier_indexes)
                qualifiers = raw[qualifier_index + 1 :]
                if _CODE_RENDERED_LOCATOR_QUALIFIERS.fullmatch(qualifiers) is None:
                    raise ValueError("locator must be a safe root-qualified code locator")
                raw = raw[:qualifier_index]
    safe = _safe_code_path(raw)
    if not safe or "/" not in safe:
        raise ValueError("locator must be a safe root-qualified code locator")
    root_slug, file_path = safe.split("/", 1)
    if (
        _SAFE_ROOT_SLUG.fullmatch(root_slug) is None
        or root_slug in {".", ".."}
        or not _safe_code_path(file_path)
    ):
        raise ValueError("locator must be a safe root-qualified code locator")
    return root_slug, file_path


def _safe_code_hit_locator(hit: Mapping[str, Any]) -> tuple[str, int | None, int | None]:
    relative_path = _safe_code_path(hit.get("relative_path"))
    root_slug = str(hit.get("root_slug") or "").strip()
    if root_slug and (
        _SAFE_ROOT_SLUG.fullmatch(root_slug) is None or root_slug in {".", ".."}
    ):
        root_slug = ""
    path = f"{root_slug}/{relative_path}" if root_slug and relative_path else relative_path

    start_line = hit.get("start_line")
    end_line = hit.get("end_line")
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or isinstance(end_line, bool)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < start_line
    ):
        start_line = None
        end_line = None

    if not path:
        locator = str(hit.get("locator") or "").strip()
        locator_match = _CODE_LOCATOR_WITH_LINES.fullmatch(locator)
        if locator_match is not None:
            path = _safe_code_path(locator_match.group("path"))
            locator_start = int(locator_match.group("start"))
            locator_end = int(locator_match.group("end"))
            if locator_end >= locator_start:
                start_line = locator_start
                end_line = locator_end
        else:
            path = _safe_code_path(locator)
    return path, start_line, end_line


def _safe_inline_code_metadata(value: object, *, max_chars: int) -> str:
    rendered = " ".join(str(value or "").split())
    return rendered[:max_chars]


def _code_search_hit_blocks(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_hits = result.get("evidence")
    if not isinstance(raw_hits, list):
        raw_hits = result.get("results")
    if not isinstance(raw_hits, list):
        return []

    blocks: list[tuple[str, str]] = []
    seen: set[tuple[str, int | None, int | None, str, str, str]] = set()
    for raw_hit in raw_hits:
        if not isinstance(raw_hit, Mapping):
            continue
        path, start_line, end_line = _safe_code_hit_locator(raw_hit)
        if not path:
            continue
        excerpt = str(raw_hit.get("excerpt") or raw_hit.get("content") or "").strip()
        if not excerpt:
            continue
        symbol = _safe_inline_code_metadata(raw_hit.get("symbol"), max_chars=200)
        signature = _safe_inline_code_metadata(raw_hit.get("signature"), max_chars=500)
        dedupe_key = (path, start_line, end_line, symbol, signature, excerpt)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        locator = path
        if start_line is not None and end_line is not None:
            locator = f"{path} (lines {start_line}-{end_line})"
        qualifiers = []
        if symbol:
            qualifiers.append(f"symbol={symbol}")
        if signature and signature != symbol:
            qualifiers.append(f"signature={signature}")
        if qualifiers:
            locator = f"{locator} {' '.join(qualifiers)}"
        blocks.append((locator, excerpt))
    return blocks


def _clip_code_excerpt(excerpt: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(excerpt) <= max_chars:
        return excerpt
    if max_chars <= len(_CODE_EXCERPT_CLIPPED_MARKER):
        return excerpt[:max_chars]
    return (
        excerpt[: max_chars - len(_CODE_EXCERPT_CLIPPED_MARKER)].rstrip()
        + _CODE_EXCERPT_CLIPPED_MARKER
    )


def _render_code_search_result(
    result: Mapping[str, Any],
    *,
    query: str,
    max_chars: int,
) -> str:
    """Project one bounded, deduplicated Agent-facing code-search result."""

    status = str(result.get("status") or "error").strip().lower()
    if status not in _CODE_SEARCH_STATUSES:
        status = "error"
    safe_query = " ".join(str(query or "").split())
    blocks = _code_search_hit_blocks(result)

    # Reserve the fixed header with the largest possible shown-count width so
    # adding blocks can never force a previously complete block over the cap.
    reserved_header = (
        f"# code_search status: {status}\n"
        f"# query: {safe_query}\n"
        f"# hit_count: {len(blocks)}\n"
        f"# {CODE_EVIDENCE_UNTRUSTED_NOTICE}\n"
    )
    if len(reserved_header) >= max_chars:
        return reserved_header[:max_chars]

    selected: list[str] = []
    used = len(reserved_header)
    for index, (locator, excerpt) in enumerate(blocks, start=1):
        prefix = f"\n# hit {index}\n{locator}\n"
        remaining = max_chars - used
        if remaining <= len(prefix):
            break
        clipped_excerpt = _clip_code_excerpt(excerpt, remaining - len(prefix))
        if not clipped_excerpt:
            break
        block = prefix + clipped_excerpt
        selected.append(block)
        used += len(block)
        if len(clipped_excerpt) < len(excerpt):
            break

    header = (
        f"# code_search status: {status}\n"
        f"# query: {safe_query}\n"
        f"# hit_count: {len(selected)}\n"
        f"# {CODE_EVIDENCE_UNTRUSTED_NOTICE}\n"
    )
    rendered = header + "".join(selected)
    return rendered[:max_chars]


def _public_result_mapping(value: object, *, operation: str) -> dict[str, Any]:
    """Require the facade result schema before returning it to an MCP caller."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{operation} returned an invalid result")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeError(f"{operation} returned an invalid result")
        result[key] = item
    return result


def _code_search_budget_for_response(
    service: Any,
    budget: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the MCP response cap through the canonical code-search budget."""

    response_cap = _configured_max_response_chars(service)
    effective = dict(budget or {})
    requested_batch = effective.get("batch_chars")
    if requested_batch is None:
        effective_batch = response_cap
    elif isinstance(requested_batch, int) and not isinstance(requested_batch, bool):
        effective_batch = min(requested_batch, response_cap)
    else:
        # Leave malformed values for the canonical DTO validator rather than
        # normalizing an adapter-only interpretation of them.
        effective_batch = requested_batch
    effective["batch_chars"] = effective_batch

    requested_snippet = effective.get("snippet_chars")
    if requested_snippet is None:
        effective_snippet = response_cap
        if isinstance(effective_batch, int) and not isinstance(effective_batch, bool):
            effective_snippet = min(effective_snippet, effective_batch)
    elif isinstance(requested_snippet, int) and not isinstance(requested_snippet, bool):
        effective_snippet = min(requested_snippet, response_cap)
    else:
        # The active-search request owns validation of invalid caller values.
        effective_snippet = requested_snippet
    effective["snippet_chars"] = effective_snippet
    return effective


def _validate_request_size(service: Any, payload: dict[str, Any]) -> None:
    max_bytes = _configured_max_request_bytes(service)
    encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"MCP request exceeds mcp_max_request_bytes ({max_bytes})")


def write_memory(
    service: Any,
    *,
    text: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    client_request_id: str | None = None,
    force_extract: bool = False,
    wait: bool = False,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    if session_id is None or not session_id.strip():
        raise ValueError("session_id is required for write_memory")
    normalized_messages = _normalize_write_messages(text=text, messages=messages)
    _validate_request_size(service, {"messages": messages or normalized_messages})
    _validate_request_size(
        service,
        {
            "messages": normalized_messages,
            "session_id": session_id,
            "agent_id": agent_id,
            "client_request_id": client_request_id,
            "force_extract": force_extract,
            "wait": wait,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    params: dict[str, Any] = {
        "sessionId": session_id,
        "messages": normalized_messages,
        "_ctx": ctx,
    }
    if agent_id:
        params["agentId"] = agent_id
    if client_request_id:
        params["clientRequestId"] = client_request_id
        params["client_request_id"] = client_request_id
    if force_extract:
        params["forceExtract"] = True
    if wait:
        params["wait"] = True

    try:
        result = service.after_turn(params)
    except Exception as exc:
        _record_mcp_write_audit(
            service,
            ctx,
            session_id=session_id,
            agent_id=agent_id,
            client_request_id=client_request_id,
            result="failed",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    raw_status = result.get("status")
    accepted = not (
        result.get("ok") is False
        or result.get("retryable") is True
        or (isinstance(raw_status, int) and 400 <= raw_status <= 599)
        or raw_status in {"failed", "error", "rejected"}
    )
    _record_mcp_write_audit(
        service,
        ctx,
        session_id=session_id,
        agent_id=agent_id,
        client_request_id=client_request_id,
        result="success" if accepted else "failed",
        details={
            "status": result.get("status", "accepted"),
            "accepted_messages": len(normalized_messages) if accepted else 0,
            "force_extract_requested": force_extract,
            "wait_requested": wait,
        },
    )

    accepted_messages = len(normalized_messages) if accepted else 0
    if accepted:
        capture_write_trace(
            service, ctx, messages or normalized_messages,
            session_id=session_id, client_request_id=client_request_id,
        )
    return {
        "ok": accepted,
        "status": result.get("status", "accepted"),
        "retryable": bool(result.get("retryable", False)),
        "session_id": session_id,
        "agent_id": agent_id or ctx.agent_id,
        "client_request_id": client_request_id,
        "submitted_messages": len(normalized_messages),
        "accepted_messages": accepted_messages,
        "stats": _public_write_stats(result),
    }


def validate_write_arguments(arguments: Mapping[str, Any]) -> None:
    unknown = sorted(set(arguments) - _WRITE_ARGUMENT_FIELDS)
    if unknown:
        raise ValueError(f"write_memory cannot set unsupported fields: {', '.join(unknown)}")


def _normalize_write_messages(
    *,
    text: str | None,
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if text is not None and messages is not None:
        raise ValueError("Provide either text or messages, not both")
    if text is not None:
        if not text.strip():
            raise ValueError("text must not be empty")
        return [{"role": "user", "content": text}]
    if not messages:
        raise ValueError("messages or text is required")

    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        forbidden = sorted(k for k in message if k in _FORBIDDEN_WRITE_FIELDS)
        if forbidden:
            raise ValueError(
                f"messages[{index}] cannot set atomic memory fields: {', '.join(forbidden)}"
            )
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{index}].role must be system, user, assistant, or tool")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        if not content and role == "assistant":
            content = message.get("reasoning_content") or message.get("reasoning")
            if not content and message.get("tool_calls"):
                content = json.dumps(message["tool_calls"], ensure_ascii=False)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        normalized_message: dict[str, Any] = {"role": role, "content": content}
        if message.get("id"):
            normalized_message["id"] = str(message["id"])
        if message.get("created_at"):
            normalized_message["created_at"] = str(message["created_at"])
        normalized.append(normalized_message)
    return normalized


def _record_mcp_write_audit(
    service: Any,
    ctx: Any,
    *,
    session_id: str,
    agent_id: str | None,
    client_request_id: str | None,
    result: str,
    details: dict[str, Any],
) -> None:
    get_audit_service = getattr(service, "get_audit_service", None)
    if not callable(get_audit_service):
        return
    try:
        get_audit_service().record(
            ctx.account_id,
            actor=ctx.user_id,
            target=session_id,
            action="mcp.write_memory",
            result=result,
            trace_id=ctx.trace_id,
            details={
                "role": str(ctx.role),
                "agent_id": agent_id or ctx.agent_id,
                "session_id": session_id,
                "client_request_id": client_request_id,
                **details,
            },
        )
    except Exception:
        return


def _public_write_stats(result: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "pending_tokens",
        "buffered_tokens",
        "candidates_extracted",
        "writes_completed",
        "writes_failed",
        "task_id",
        "reason",
    }
    stats: dict[str, Any] = {}
    for field in allowed_fields:
        value = result.get(field)
        if isinstance(value, (str, int, float, bool)) or value is None:
            stats[field] = value
    return stats
