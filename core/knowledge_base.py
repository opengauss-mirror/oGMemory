"""Tenant-owned knowledge bases and access grants."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from core.models import RequestContext, Role


def user_generation(created_at: object) -> str | None:
    if not isinstance(created_at, str):
        return None
    try:
        instant = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    return instant.astimezone(UTC).isoformat() if instant.tzinfo is not None else None


@dataclass(frozen=True)
class KnowledgeBaseContext(RequestContext):
    user_generation: str | None = None


@dataclass(frozen=True)
class KnowledgeBaseGrant:
    permission: Literal["read", "write"]
    user_generation: str


@dataclass(frozen=True)
class KnowledgeBase:
    account_id: str
    id: UUID
    name: str
    visibility: Literal["private", "shared"]
    owner_user_id: str | None
    backend: str
    namespace: str
    is_default: bool = False
    owner_user_generation: str | None = None

    def permissions(
        self, ctx: KnowledgeBaseContext, members: dict[str, KnowledgeBaseGrant]
    ) -> dict[str, bool]:
        if self.account_id != ctx.account_id:
            return {"read": False, "write": False, "manage": False}
        if self.visibility == "private":
            can_manage = bool(
                ctx.user_id == self.owner_user_id
                and ctx.user_generation
                and ctx.user_generation == self.owner_user_generation
            )
            permission = None
        else:
            can_manage = ctx.role in {Role.ADMIN, Role.ROOT}
            grant = members.get(ctx.user_id)
            permission = (
                grant.permission if grant and grant.user_generation == ctx.user_generation else None
            )
        return {
            "read": can_manage or permission in {"read", "write"},
            "write": can_manage or permission == "write",
            "manage": can_manage,
        }
