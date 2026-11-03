"""Owner-space filtering helpers for retrieval stages."""

from __future__ import annotations

from core.enums import ContextType
from core.models import RequestContext, TypedQuery


def canonical_owner_space(space: str) -> str:
    """Normalize legacy user and agent prefixes while preserving the identifier."""
    for prefix in ("user_", "agent_"):
        if space.startswith(prefix):
            return space.replace(prefix, prefix[:-1] + ":", 1)
    return space


def allowed_owner_space_filter(
    typed_query: TypedQuery,
    ctx: RequestContext,
    *,
    enforce_visible_scope: bool = False,
) -> str | list[str] | None:
    if (
        typed_query.context_type == ContextType.RESOURCE.value
        and typed_query.resource_type == "code"
    ):
        return None
    if not enforce_visible_scope:
        return typed_query.owner_space

    visible_spaces = expand_owner_space_aliases(list(getattr(ctx, "visible_owner_spaces", ()) or []))
    authorized_spaces = visible_spaces or _default_owner_spaces(ctx)

    if typed_query.owner_space:
        requested = (
            list(typed_query.owner_space)
            if isinstance(typed_query.owner_space, list)
            else [typed_query.owner_space]
        )
        requested = expand_owner_space_aliases(requested)
        allowed = [space for space in requested if space in authorized_spaces]
        if not allowed:
            return "__inaccessible__"
        return _compact_owner_spaces(allowed)

    return authorized_spaces or "__inaccessible__"


def _default_owner_spaces(ctx: RequestContext) -> list[str]:
    spaces: list[str] = []
    if ctx.user_id:
        spaces.extend([ctx.user_space_name(), f"user_{ctx.user_id}"])
    if ctx.agent_id:
        spaces.extend([ctx.agent_space_name(), f"agent_{ctx.agent_id}"])
    return spaces


def expand_owner_space_aliases(spaces: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for space in spaces:
        for alias in _owner_space_aliases(space):
            if alias not in seen:
                expanded.append(alias)
                seen.add(alias)
    return expanded


def visible_agent_ids(ctx: RequestContext) -> list[str]:
    ids: set[str] = set()
    for space in expand_owner_space_aliases(list(getattr(ctx, "visible_owner_spaces", ()) or [])):
        if space.startswith("agent:"):
            agent_id = space.split(":", 1)[1]
            if agent_id:
                ids.add(agent_id)
    return sorted(ids)


def _owner_space_aliases(space: str) -> list[str]:
    if not isinstance(space, str) or not space:
        return []
    aliases = [space]
    if space.startswith("user:"):
        aliases.append(f"user_{space.split(':', 1)[1]}")
    elif space.startswith("user_"):
        aliases.append(f"user:{space.split('_', 1)[1]}")
    elif space.startswith("agent:"):
        aliases.append(f"agent_{space.split(':', 1)[1]}")
    elif space.startswith("agent_"):
        aliases.append(f"agent:{space.split('_', 1)[1]}")
    return aliases


def _compact_owner_spaces(spaces: list[str]) -> str | list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for space in spaces:
        if space in seen:
            continue
        compacted.append(space)
        seen.add(space)
    return compacted if len(compacted) > 1 else compacted[0]
