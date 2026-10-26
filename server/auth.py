"""Authentication and RequestContext construction for V1."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from core.models import RequestContext, Role
from server.api_keys import APIKeyManager


@dataclass(frozen=True)
class ResolvedIdentity:
    role: Role
    account_id: str | None = None
    user_id: str | None = None


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class ControlPlaneDisabledError(Exception):
    pass


def build_admin_target_context(service, actor, *, account_id, user_id, agent_id=""):
    """Resolve a registered user/agent within the actor's administrative scope."""
    if not user_id:
        raise ValueError("userId is required")
    if Role(actor.role) != Role.ROOT and account_id != actor.account_id:
        raise AuthorizationError("account access denied")
    admin = service.get_tenant_admin_service()
    users = admin.list_users(actor, account_id).get("users", [])
    if user_id not in {str(user["user_id"]) for user in users}:
        raise FileNotFoundError("user not found")
    if agent_id:
        agents = admin.list_agents(actor, account_id).get("agents", [])
        owned_agents = {
            str(agent["agent_id"])
            for agent in agents
            if str(agent.get("owner_user_id") or "") == user_id
        }
        if agent_id not in owned_agents:
            raise FileNotFoundError("agent not found under user")
    visible = [f"user:{user_id}"]
    if agent_id:
        visible.append(f"agent:{agent_id}")
    return RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=actor.session_id,
        trace_id=actor.trace_id,
        role=actor.role,
        visible_owner_spaces=tuple(visible),
    )


class AuthService:
    def __init__(self, cfg, key_manager: APIKeyManager):
        self._cfg = cfg
        self._key_manager = key_manager
        self._ensure_admin_accounts()

    def _ensure_admin_accounts(self):
        if not self._cfg.admin_api_keys:
            return
        existing = {row["account_id"] for row in self._key_manager.get_accounts()}
        for entry in self._cfg.admin_api_keys:
            if ":" in entry:
                account_id, _ = entry.split(":", 1)
            else:
                account_id = self._cfg.account_id
            if account_id and account_id not in existing:
                try:
                    self._key_manager.create_account(account_id, "admin")
                except ValueError:
                    pass
                existing.add(account_id)

    def role_control_active(self) -> bool:
        has_privileged = bool(self._cfg.root_api_key or self._cfg.admin_api_keys)
        return bool(self._cfg.role_control_enabled and has_privileged)

    def resolve_identity(self, headers: dict | None) -> ResolvedIdentity | None:
        if not self.role_control_active():
            return None
        headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        api_key = self._extract_api_key(headers)
        if not api_key:
            raise AuthenticationError("Missing API Key")

        if self._cfg.root_api_key and api_key == self._cfg.root_api_key:
            return ResolvedIdentity(
                role=Role.ROOT,
                account_id=headers.get("x-account-id") or self._cfg.account_id,
                user_id=headers.get("x-user-id") or "root",
            )

        admin_account = self._resolve_admin_account(api_key)
        if admin_account is not None:
            requested_account = headers.get("x-account-id") or admin_account
            if requested_account != admin_account:
                raise AuthenticationError("Admin API Key account mismatch")
            return ResolvedIdentity(
                role=Role.ADMIN,
                account_id=admin_account,
                user_id=headers.get("x-user-id") or "admin",
            )

        resolved = self._key_manager.resolve_user_key(
            api_key=api_key,
            account_hint=headers.get("x-account-id"),
        )
        if not resolved:
            raise AuthenticationError("Invalid API Key")

        role = Role.ADMIN if resolved["role"] == "admin" else Role.MEMBER
        return ResolvedIdentity(
            role=role,
            account_id=resolved["account_id"],
            user_id=resolved["user_id"],
        )

    def build_request_context(
        self,
        identity: ResolvedIdentity | None,
        *,
        account_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> RequestContext:
        if identity is None:
            return RequestContext(
                account_id=account_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                trace_id=str(uuid4()),
                role=Role.ROOT,
            )
        return RequestContext(
            account_id=identity.account_id or account_id,
            user_id=identity.user_id or user_id,
            agent_id=agent_id,
            session_id=session_id,
            trace_id=str(uuid4()),
            role=identity.role,
        )

    def require_role(self, ctx: RequestContext, *allowed_roles: Role) -> None:
        if not self.role_control_active():
            return
        if self._normalize_role(ctx.role) not in allowed_roles:
            raise AuthorizationError(
                f"Operation requires role: {', '.join(r.value for r in allowed_roles)}"
            )

    @staticmethod
    def _extract_api_key(headers: dict) -> str | None:
        api_key = headers.get("x-api-key")
        if api_key:
            return api_key.strip()
        authz = headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            return authz.split(" ", 1)[1].strip()
        return None

    @staticmethod
    def _normalize_role(role: Role | str) -> Role:
        if isinstance(role, Role):
            return role
        return Role(role)

    def _resolve_admin_account(self, api_key: str) -> str | None:
        for configured in self._cfg.admin_api_keys:
            if ":" in configured:
                account_id, key = configured.split(":", 1)
                if api_key == key:
                    return account_id
            elif api_key == configured:
                return self._cfg.account_id
        return None
