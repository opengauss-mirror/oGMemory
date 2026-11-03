from types import SimpleNamespace

import pytest

import server.app as app_module
import server.mcp_tools as mcp_tools


def test_query_and_read_routes_reuse_native_memory_tool_contracts(monkeypatch):
    service = SimpleNamespace(_cfg=SimpleNamespace(http_ip_allowlist=[]))
    ctx = SimpleNamespace(agent_id="configured-agent")
    calls = []
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    monkeypatch.setattr(
        mcp_tools,
        "query_memory",
        lambda passed_service, **kwargs: (
            calls.append(("query", passed_service, kwargs)) or {"hits": []}
        ),
    )
    monkeypatch.setattr(
        mcp_tools,
        "read_memory",
        lambda passed_service, **kwargs: (
            calls.append(("read", passed_service, kwargs)) or {"uri": kwargs["uri"]}
        ),
    )
    client = app_module.app.test_client()

    query = client.post(
        "/api/v1/memories/query",
        json={
            "query": "editor preference",
            "types": ["preference"],
            "top_k": 3,
            "sessionId": "s1",
            "agentId": "model-controlled-agent",
        },
    )
    read = client.post(
        "/api/v1/memories/read",
        json={
            "uri": "ogmemory://memories/preference/1",
            "level": "overview",
            "sessionId": "s1",
        },
    )

    assert query.status_code == 200
    assert read.status_code == 200
    assert calls[0][0:2] == ("query", service)
    assert calls[0][2]["request_context"] is ctx
    assert calls[0][2]["agent_id"] == "configured-agent"
    assert calls[0][2]["types"] == ["preference"]
    assert calls[1][2]["uri"] == "ogmemory://memories/preference/1"


def test_write_and_delete_routes_follow_native_toolset_gates(monkeypatch):
    ctx = SimpleNamespace(agent_id="configured-agent")
    service = SimpleNamespace(
        _cfg=SimpleNamespace(
            http_ip_allowlist=[],
            mcp_default_toolset="all",
            mcp_enable_write_tools=False,
            mcp_enable_delete_tools=False,
        )
    )
    calls = []
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    monkeypatch.setattr(
        mcp_tools,
        "write_memory",
        lambda passed_service, **kwargs: calls.append(("write", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        mcp_tools,
        "delete_memory",
        lambda passed_service, **kwargs: calls.append(("delete", kwargs)) or {"deleted": True},
    )
    client = app_module.app.test_client()

    write = client.post(
        "/api/v1/memories/write",
        json={
            "text": "Remember blue.",
            "sessionId": "s1",
            "force_extract": "false",
        },
    )
    delete = client.post(
        "/api/v1/memories/delete",
        json={
            "uri": "ogmemory://memories/preference/1",
            "sessionId": "s1",
        },
    )

    assert write.status_code == 200
    assert delete.status_code == 200
    assert calls[0][1]["force_extract"] is False
    assert calls[0][1]["request_context"] is ctx
    assert calls[1][1]["uri"] == "ogmemory://memories/preference/1"

    service._cfg.mcp_default_toolset = "read-only"
    assert (
        client.post(
            "/api/v1/memories/write",
            json={
                "text": "blocked",
                "sessionId": "s1",
            },
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/memories/delete",
            json={
                "uri": "ogmemory://memories/preference/1",
                "sessionId": "s1",
            },
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    ("route", "field", "value"),
    [
        ("query", "top_k", None),
        ("query", "top_k", "not-a-number"),
        ("query", "top_k", {}),
        ("read", "max_chars", None),
        ("read", "max_chars", "not-a-number"),
        ("read", "max_chars", {}),
    ],
)
def test_memory_read_routes_reject_invalid_integer_parameters(
    monkeypatch, route, field, value
):
    service = SimpleNamespace(_cfg=SimpleNamespace(http_ip_allowlist=[]))
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(
        app_module,
        "_build_authenticated_context",
        lambda params: SimpleNamespace(agent_id="configured-agent"),
    )

    response = app_module.app.test_client().post(
        f"/api/v1/memories/{route}", json={field: value}
    )

    assert response.status_code == 400


@pytest.mark.parametrize("session_key", ["sessionId", "session_id"])
def test_write_route_persists_trace_for_the_requested_session(monkeypatch, session_key):
    from tests.unit.server.test_mcp_tools_write import _StoredService

    service = _StoredService()
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", service.build_context)
    client = app_module.app.test_client()
    for session_id in ["session-a", "session-b", "session-a"]:
        response = client.post("/api/v1/memories/write", json={
            session_key: session_id, "text": "Remember blue.", "client_request_id": "same-request",
        })
        assert response.status_code == 200
        assert response.json["ok"] is True
        turns = service.store._read_state(
            "acct-1", collections=("turns",), session_id=session_id,
        )["turns"]
        assert len(turns) == 1
        turn = next(iter(turns.values()))
        assert turn["session_id"] == session_id
        assert [event["event_type"] for event in turn["content"]] == ["user_message", "turn_end"]
