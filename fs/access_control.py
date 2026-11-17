"""Unified access control for ContextFS implementations.

Provides single permission check supporting:
- Account isolation (mandatory, cross-account denied)
- Shared agent access via visible_owner_spaces (SQL mode)
- Fallback to strict owner_id check (AGFS mode / empty visible_owner_spaces)

This module enables consistent permission logic across SQLContextFS and
AGFSContextFS, fixing the read/write behavior split when users have access
to shared agent spaces.
"""

from __future__ import annotations

from core.models import RequestContext

# owner_type (plural) → owner_space prefix (singular)
_OWNER_TYPE_TO_KIND = {"users": "user", "agents": "agent"}


def check_uri_access(
    ctx: RequestContext,
    components: dict[str, str],
    *,
    strict_mode: bool = False,
) -> bool:
    """Check if URI is accessible given RequestContext.

    Three-step check:
    1. Account match (always enforced, cross-account denied)
    2. Owner space with visible_owner_spaces (SQL mode, shared agents)
    3. Fallback to strict owner_id (AGFS mode or empty visible_owner_spaces)

    Args:
        ctx: RequestContext with account_id, user_id, agent_id, visible_owner_spaces
        components: Pre-parsed URI dict from parse_uri() with keys:
                   account, owner_type, owner_id, category, slug
        strict_mode: True for AGFS (ignore visible_owner_spaces), False for SQL

    Returns:
        True if accessible, False otherwise
    """
    # Step 1: Account isolation (mandatory)
    account = components.get("account", "")
    if account != ctx.account_id:
        return False

    # Step 2: Extract owner info
    owner_id = components.get("owner_id", "")
    owner_type = components.get("owner_type", "")
    if owner_type == "spaces":
        return False

    # No owner restriction (e.g., session-level URIs, resources)
    if not owner_id:
        return True

    # Session URIs are account-scoped, but state URIs require matching session_id
    if owner_type == "sessions":
        category = components.get("category", "")
        if category == "state":
            return owner_id == ctx.session_id
        return True

    # Step 3a: AGFS strict mode (no shared agents)
    if strict_mode:
        if owner_type == "users":
            return owner_id == ctx.user_id
        if owner_type == "agents":
            return owner_id == ctx.agent_id
        # Unknown owner_type: deny by default
        return False

    # Step 3b: SQL relaxed mode with visible_owner_spaces
    vos = ctx.visible_owner_spaces

    if vos and len(vos) > 0:
        kind = _OWNER_TYPE_TO_KIND.get(owner_type, owner_type)
        expected_space = f"{kind}:{owner_id}"
        return expected_space in vos

    # Step 3c: Fallback to strict check when visible_owner_spaces is empty
    if owner_type == "users":
        return owner_id == ctx.user_id
    if owner_type == "agents":
        return owner_id == ctx.agent_id

    # Unknown owner_type: deny by default
    return False
