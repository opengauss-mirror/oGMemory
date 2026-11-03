from __future__ import annotations

import importlib.util

import pytest

_HAS_FLASK = importlib.util.find_spec("flask") is not None
pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")

from core.models import RequestContext
from features.tool_result_externalization.store_base import ArtifactStoreUnavailableError
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService

if _HAS_FLASK:
    import server.app as app_module
else:
    app_module = None


def _ctx() -> RequestContext:
    return RequestContext("acct", "user", "agent", "session", "trace")


@pytest.fixture(autouse=True)
def local_session_backend(monkeypatch):
    monkeypatch.setattr(MemoryService, "_get_context_fs", lambda self: None)


def _artifact_service(tmp_path, *, response_chars=800) -> MemoryService:
    service = MemoryService(
        OgMemConfig(
            account_id="acct",
            user_id="user",
            agent_id="agent",
            tool_result_externalization_enabled=True,
            tool_result_semantic_turn_mode="user_message",
            tool_result_single_chars=1,
            tool_result_http_max_response_chars=response_chars,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service.govern_first_view_tool_result({
        "sessionId": "session",
        "adapterId": "test",
        "turnId": "turn-artifact-service",
        "toolCallId": "call-artifact-service",
        "toolName": "fixture",
        "content": "needle " * 1000,
        "_ctx": _ctx(),
    })
    return service


def test_delete_session_http_cleans_artifacts(
    monkeypatch,
    tmp_path,
):
    service = _artifact_service(tmp_path)
    manager = service.get_session_manager()
    monkeypatch.setattr(
        manager,
        "durable_session_owner",
        lambda *args: ("user", "agent"),
    )
    monkeypatch.setattr(manager, "delete_session_archives", lambda *args: 0)
    monkeypatch.setattr(manager, "delete_persisted_session_state", lambda *args: 0)
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().delete(
        "/api/v1/sessions/session",
        json={"accountId": "acct", "userId": "user", "agentId": "agent"},
    )

    assert response.status_code == 200
    assert response.get_json()["cleaned"] is True
    assert service.list_tool_result_artifacts({
        "sessionId": "session", "_ctx": _ctx(),
    })["artifacts"] == []
def test_disabled_read_routes_stay_unavailable_but_session_delete_remains_available(
    monkeypatch,
):
    service = MemoryService(OgMemConfig(tool_result_externalization_enabled=False))
    service.delete_session = lambda _params: {
        "ok": True,
        "cleaned": True,
        "status": 200,
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    added = client.post("/api/v1/sessions/session/messages", json={
        "role": "assistant",
        "content": "top-level",
        "message": {"role": "toolResult", "content": "nested"},
    })
    deleted = client.delete("/api/v1/sessions/session")
    unavailable = [
        client.get("/api/v1/sessions/session/tool-results"),
        client.get("/api/v1/sessions/session/tool-results/tr_missing"),
        client.post("/api/v1/sessions/session/tool-results/search", json={"query": "x"}),
        client.post("/api/v1/call/list_tool_result_artifacts", json={
            "sessionId": "session",
        }),
    ]

    assert added.status_code == 200
    assert deleted.status_code == 200
    assert all(response.status_code == 404 for response in unavailable)
    buffer = service.get_session_manager().get_or_create("session")
    assert [(row.role, row.content) for row in buffer.messages] == [
        ("assistant", "top-level")
    ]
def test_artifact_store_failure_returns_retryable_503(monkeypatch):
    service = MemoryService(OgMemConfig(tool_result_externalization_enabled=True))
    service.search_tool_result_artifacts = lambda _params: (_ for _ in ()).throw(
        ArtifactStoreUnavailableError("backend details must not escape")
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/sessions/session/tool-results/search",
        json={"query": "needle"},
    )

    assert response.status_code == 503
    assert response.get_json() == {
        "ok": False,
        "error": "tool_result_artifact_store_unavailable",
        "retryable": True,
    }
    assert "backend details" not in response.get_data(as_text=True)
def test_http_search_caps_response_and_validates_top_k(monkeypatch, tmp_path):
    service = _artifact_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    path = "/api/v1/sessions/session/tool-results/search"

    response = client.post(path, json={
        "query": "needle", "top_k": 20, "max_chars": 10_000_000,
    })
    invalid = client.post(path, json={"query": "needle", "top_k": 21})

    assert response.status_code == 200
    assert len(response.get_data(as_text=True)) <= 801
    assert invalid.status_code == 400
