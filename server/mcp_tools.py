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
    UNTRUSTED_DATA_NOTICE,
    build_mcp_context,
    is_not_found_or_not_visible_error,
    raise_not_found_or_not_visible,
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
    matching = [row for row in rows if row["score"] > 0]
    skills = sorted(matching, key=lambda row: (-float(row["score"]), row["name"]))[:top_k]
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

    lowered = value.casefold()
    words = re.findall(r"[^\W_]{2,}", lowered, flags=re.UNICODE)
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered)
    cjk_bigrams = [
        run[index:index + 2]
        for run in cjk_runs
        for index in range(max(1, len(run) - 1))
    ]
    return words + cjk_bigrams


def query_memory(
    service: Any,
    *,
    query: str,
    types: list[str] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
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


def read_memory(
    service: Any,
    *,
    uri: str,
    unit_key: str | None = None,
    level: str = "all",
    max_chars: int = 12000,
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
    if unit_key is not None and (not isinstance(unit_key, str) or not unit_key.strip()):
        raise ValueError("unit_key must be a non-empty string")

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
