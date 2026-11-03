from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import quote

import pytest

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.auth import AuthorizationError
from server.mcp_security import MCPVisibilityError
from server.mcp_tools import delete_memory


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeService:
    def __init__(self, cfg=None):
        self._cfg = cfg or OgMemConfig()
        self.forget_call = None
        self.forget_calls = []
        self.raise_on_forget = None
        self.raise_by_uri = {}

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "s1",
            trace_id="t1",
            role=Role.MEMBER,
        )

    def forget_memory(self, uri, ctx, **kwargs):
        self.forget_call = {"uri": uri, "ctx": ctx, **kwargs}
        self.forget_calls.append(self.forget_call)
        if uri in self.raise_by_uri:
            raise self.raise_by_uri[uri]
        if self.raise_on_forget:
            raise self.raise_on_forget
        return {"ok": True, "deleted": True, "mode": "archive", "uri": kwargs.get("external_uri")}


def test_delete_memory_maps_external_uri_and_forgets():
    service = _FakeService()

    payload = delete_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        reason="stale",
        session_id="s1",
        client_request_id="req-1",
    )

    assert payload == {
        "ok": True,
        "deleted": True,
        "mode": "archive",
        "uri": "ogmemory://memories/preference/coding_style",
        "type": "preference",
        "id": "coding_style",
        "client_request_id": "req-1",
    }
    assert service.forget_call["uri"] == "ctx://acct-1/users/alice/memories/preferences/coding_style"
    assert service.forget_call["external_uri"] == "ogmemory://memories/preference/coding_style"
    assert service.forget_call["reason"] == "stale"


def test_delete_memory_rejects_request_larger_than_configured_limit():
    service = _FakeService(OgMemConfig(mcp_max_request_bytes=64))

    with pytest.raises(ValueError, match="mcp_max_request_bytes"):
        delete_memory(
            service,
            uri="ogmemory://memories/preference/coding_style",
            reason="x" * 256,
        )


def test_delete_memory_falls_back_to_legacy_agent_pattern_path():
    service = _FakeService()
    service.raise_by_uri = {
        "ctx://acct-1/users/alice/memories/patterns/review": FileNotFoundError("missing")
    }

    payload = delete_memory(
        service,
        uri="ogmemory://memories/pattern/review",
        agent_id="bot",
    )

    assert payload["deleted"] is True
    assert [call["uri"] for call in service.forget_calls] == [
        "ctx://acct-1/users/alice/memories/patterns/review",
        "ctx://acct-1/agents/bot/memories/patterns/review",
    ]


def test_delete_memory_does_not_fallback_after_storage_error():
    service = _FakeService()
    service.raise_by_uri = {
        "ctx://acct-1/users/alice/memories/patterns/review": RuntimeError("storage unavailable")
    }

    with pytest.raises(RuntimeError, match="storage unavailable"):
        delete_memory(
            service,
            uri="ogmemory://memories/pattern/review",
            agent_id="bot",
        )

    assert [call["uri"] for call in service.forget_calls] == [
        "ctx://acct-1/users/alice/memories/patterns/review",
    ]


def test_delete_memory_scoped_agent_pattern_does_not_delete_user_pattern_first():
    service = _FakeService()

    payload = delete_memory(
        service,
        uri="ogmemory://memories/pattern/agent/review",
        agent_id="bot",
    )

    assert payload["deleted"] is True
    assert payload["uri"] == "ogmemory://memories/pattern/agent/review"
    assert [call["uri"] for call in service.forget_calls] == [
        "ctx://acct-1/agents/bot/memories/patterns/review",
    ]


def test_delete_memory_requires_item_uri():
    with pytest.raises(ValueError, match="single ogmemory"):
        delete_memory(_FakeService(), uri="ogmemory://memories/preference")


def test_delete_memory_maps_permission_errors_to_not_found_or_not_visible():
    service = _FakeService()
    service.raise_on_forget = AuthorizationError("denied")

    with pytest.raises(MCPVisibilityError, match="not_found_or_not_visible"):
        delete_memory(service, uri="ogmemory://memories/preference/coding_style")


@pytest.mark.parametrize("suffix", ["?view=content", "#content", "%00", "%2Fchild"])
def test_delete_rejects_uri_suffixes_before_storage_mutation(suffix):
    service = _FakeService()
    with pytest.raises(ValueError):
        delete_memory(service, uri="ogmemory://memories/preference/target" + suffix)
    assert service.forget_calls == []


@pytest.mark.parametrize("stored_name", [
    "Mixed" + quote("历史", safe="").lower(),
    "Mixed%E5%8e%86%e5%8F%B2",
])
def test_delete_resolves_historical_percent_escape_case(stored_name):
    service = _FakeService()
    base = "ctx://acct-1/users/alice/memories/preferences/"
    stored_uri = base + stored_name
    service.raise_by_uri = {
        base + name: FileNotFoundError("missing")
        for name in ("Mixed历史", quote("Mixed历史", safe=""), "Mixed" + quote("历史", safe="").lower())
        if name != stored_name
    }
    service._get_context_fs = lambda: SimpleNamespace(list_children=lambda uri, ctx: [stored_uri])

    payload = delete_memory(service, uri="ogmemory://memories/preference/" + quote("Mixed历史", safe=""))

    assert payload["deleted"] is True
    assert service.forget_calls[-1]["uri"] == stored_uri
