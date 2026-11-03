from __future__ import annotations

import pytest

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.auth import AuthenticationError, AuthService, ResolvedIdentity
from server.control_plane_store import ControlPlaneStore
from server.mcp_security import (
    build_mcp_context,
    has_mcp_http_request,
    headers_from_mcp_context,
    identity_from_mcp_context,
)


class _FakeAuth:
    def __init__(self, *, active: bool = False, role: Role = Role.MEMBER):
        self.active = active
        self.role = role
        self.headers = None

    def role_control_active(self):
        return self.active

    def resolve_identity(self, headers):
        self.headers = headers
        return ResolvedIdentity(role=self.role, account_id="acct-auth", user_id="user-auth")


class _FakeService:
    def __init__(self, auth):
        self.auth = auth
        self.params = None
        self.identity = None

    def get_auth_service(self):
        return self.auth

    def build_context(self, params, identity=None):
        self.params = params
        self.identity = identity
        account_id = identity.account_id if identity else "acct-default"
        user_id = identity.user_id if identity else "user-default"
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=params.get("agentId") or ("" if identity else "agent-default"),
            session_id=params.get("sessionId") or "unknown",
            trace_id="trace",
            role=identity.role if identity else Role.ROOT,
        )


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class _FakeRequestContext:
    def __init__(self, request):
        self.request = request


class _FakeMCPContext:
    def __init__(self, request):
        self.request_context = _FakeRequestContext(request)


def test_build_mcp_context_allows_only_scope_fields():
    service = _FakeService(_FakeAuth(active=False))

    ctx = build_mcp_context(
        service,
        {"session_id": "s1", "agent_id": "bot"},
    )

    assert ctx.account_id == "acct-default"
    assert ctx.user_id == "user-default"
    assert service.params == {"sessionId": "s1", "agentId": "bot"}


@pytest.mark.parametrize("field", ["accountId", "account_id", "userId", "user_id", "role"])
def test_build_mcp_context_rejects_identity_fields(field):
    service = _FakeService(_FakeAuth(active=False))

    with pytest.raises(ValueError, match="cannot set identity fields"):
        build_mcp_context(service, {field: "evil"})


def test_build_mcp_context_resolves_stdio_api_key_from_env(monkeypatch):
    auth = _FakeAuth(active=True)
    service = _FakeService(auth)
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "member-key")

    ctx = build_mcp_context(service, {"session_id": "s1"})

    assert auth.headers["X-API-Key"] == "member-key"
    assert ctx.account_id == "acct-auth"
    assert ctx.user_id == "user-auth"
    assert ctx.agent_id == ""
    assert service.params == {"sessionId": "s1"}


@pytest.mark.parametrize("role", ["user", "admin"])
@pytest.mark.parametrize("bearer", [False, True])
def test_mcp_registered_identity_is_bound_to_its_account_and_user(tmp_path, monkeypatch, role, bearer):
    manager = APIKeyManager(ControlPlaneStore(mount_prefix="", local_root=str(tmp_path)))
    manager.create_account("account-a", "admin")
    manager.create_account("account-b", "admin")
    key = manager.register_user("account-a", "same-user", role)
    manager.register_user("account-b", "same-user", role)
    auth = AuthService(OgMemConfig(role_control_enabled=True, root_api_key="root-key"), manager)
    service = _FakeService(auth)
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "root-key")
    monkeypatch.setenv("OGMEM_MCP_ACCOUNT_ID", "account-b")
    credential = {"Authorization": "Bearer " + key} if bearer else {"X-API-Key": key}
    headers = {**credential, "X-Account-ID": "account-a", "X-User-ID": "admin", "X-Role": "root"}

    ctx = build_mcp_context(service, {"session_id": "same-session"}, transport_headers=headers,
                            allow_stdio_env_auth=False)

    assert (ctx.account_id, ctx.user_id) == ("account-a", "same-user")
    assert ctx.role == (Role.ADMIN if role == "admin" else Role.MEMBER)
    with pytest.raises(AuthenticationError):
        build_mcp_context(service, {}, transport_headers={**headers, "X-Account-ID": "account-b"},
                          allow_stdio_env_auth=False)


def test_stdio_explicit_invalid_credential_does_not_use_environment_root(tmp_path, monkeypatch):
    auth = AuthService(OgMemConfig(role_control_enabled=True, root_api_key="root-key"),
                       APIKeyManager(ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))))
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "root-key")

    with pytest.raises(AuthenticationError):
        build_mcp_context(_FakeService(auth), {}, transport_headers={"Authorization": "Bearer unknown"})


def test_build_mcp_context_prefers_transport_headers_over_env(monkeypatch):
    auth = _FakeAuth(active=True)
    service = _FakeService(auth)
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "env-key")

    build_mcp_context(
        service,
        {},
        transport_headers={"X-API-Key": "header-key"},
    )

    assert auth.headers["X-API-Key"] == "header-key"


def test_build_mcp_context_uses_transport_agent_header_as_default():
    service = _FakeService(_FakeAuth(active=True))

    ctx = build_mcp_context(
        service,
        {"session_id": "s1"},
        transport_headers={"X-API-Key": "member-key", "X-Agent-ID": "shared-agent"},
        allow_stdio_env_auth=False,
    )

    assert ctx.agent_id == "shared-agent"
    assert service.params == {"sessionId": "s1", "agentId": "shared-agent"}


def test_build_mcp_context_tool_agent_argument_overrides_transport_header():
    service = _FakeService(_FakeAuth(active=True))

    ctx = build_mcp_context(
        service,
        {"agent_id": "tool-agent"},
        transport_headers={"X-API-Key": "member-key", "X-Agent-ID": "header-agent"},
        allow_stdio_env_auth=False,
    )

    assert ctx.agent_id == "tool-agent"
    assert service.params == {"agentId": "tool-agent"}


def test_build_mcp_context_preserves_authenticated_user_scope_without_agent():
    service = _FakeService(_FakeAuth(active=True))

    ctx = build_mcp_context(
        service,
        {"session_id": "s1"},
        transport_headers={"X-API-Key": "member-key"},
        allow_stdio_env_auth=False,
    )

    assert ctx.agent_id == ""
    assert service.params == {"sessionId": "s1"}


def test_build_mcp_context_allows_service_to_apply_its_own_default():
    service = _FakeService(_FakeAuth(active=False))

    ctx = build_mcp_context(service, {"session_id": "s1"})

    assert ctx.agent_id == "agent-default"
    assert service.params == {"sessionId": "s1"}


def test_build_mcp_context_constructs_root_context_without_service_visibility_lookup():
    auth = _FakeAuth(active=True, role=Role.ROOT)
    service = _FakeService(auth)

    def fail_build_context(params, identity=None):
        raise AssertionError("MCP ROOT context should not call MemoryService.build_context")

    service.build_context = fail_build_context

    ctx = build_mcp_context(
        service,
        {"session_id": "s1", "agent_id": "bot"},
        transport_headers={"X-API-Key": "root-key"},
        allow_stdio_env_auth=False,
    )

    assert ctx.account_id == "acct-auth"
    assert ctx.user_id == "user-auth"
    assert ctx.agent_id == "bot"
    assert ctx.session_id == "s1"
    assert ctx.role == Role.ROOT


def test_build_mcp_context_scopes_root_to_user_without_explicit_agent():
    auth = _FakeAuth(active=True, role=Role.ROOT)
    service = _FakeService(auth)

    ctx = build_mcp_context(
        service,
        {"session_id": "s1"},
        transport_headers={
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-auth",
            "X-User-ID": "user-auth",
        },
        allow_stdio_env_auth=False,
    )

    assert ctx.role == Role.ROOT
    assert ctx.user_id == "user-auth"
    assert ctx.visible_owner_spaces == ("user:user-auth",)


def test_build_mcp_context_can_disable_stdio_env_fallback(monkeypatch):
    auth = _FakeAuth(active=True)
    service = _FakeService(auth)
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "env-key")

    build_mcp_context(
        service,
        {},
        allow_stdio_env_auth=False,
    )

    assert auth.headers == {}


def test_build_mcp_context_reuses_transport_resolved_identity():
    auth = _FakeAuth(active=True)
    service = _FakeService(auth)
    identity = ResolvedIdentity(
        role=Role.MEMBER,
        account_id="cached-account",
        user_id="cached-user",
    )

    ctx = build_mcp_context(
        service,
        {},
        allow_stdio_env_auth=False,
        resolved_identity=identity,
    )

    assert auth.headers is None
    assert ctx.account_id == "cached-account"
    assert ctx.user_id == "cached-user"


def test_build_mcp_context_rejects_http_when_role_control_inactive():
    service = _FakeService(_FakeAuth(active=False))

    with pytest.raises(AuthenticationError, match="MCP HTTP requires"):
        build_mcp_context(
            service,
            {},
            transport_headers={},
            allow_stdio_env_auth=False,
        )


def test_headers_from_mcp_context_extracts_http_headers():
    ctx = _FakeMCPContext(
        _FakeRequest(
            {
                "authorization": "Bearer token",
                "x-api-key": "key",
            }
        )
    )

    headers = headers_from_mcp_context(ctx)

    assert headers["authorization"] == "Bearer token"
    assert headers["x-api-key"] == "key"
    assert has_mcp_http_request(ctx) is True


def test_headers_from_lowlevel_mcp_context_extracts_http_headers():
    class _LowLevelContext:
        request = _FakeRequest({"x-api-key": "key"})

    assert headers_from_mcp_context(_LowLevelContext()) == {"x-api-key": "key"}
    assert has_mcp_http_request(_LowLevelContext()) is True


def test_identity_from_mcp_context_extracts_transport_identity():
    identity = ResolvedIdentity(role=Role.MEMBER, account_id="acct", user_id="user")
    request = _FakeRequest({"x-api-key": "key"})
    request.scope = {"state": {"ogmem_mcp_identity": identity}}
    ctx = _FakeMCPContext(request)

    assert identity_from_mcp_context(ctx) is identity


def test_headers_from_mcp_context_handles_missing_context():
    assert headers_from_mcp_context(None) == {}
    assert has_mcp_http_request(None) is False


def test_context_without_headers_is_not_http_request():
    ctx = _FakeMCPContext(object())

    assert headers_from_mcp_context(ctx) == {}
    assert has_mcp_http_request(ctx) is False
