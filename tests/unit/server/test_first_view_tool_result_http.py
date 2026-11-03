from __future__ import annotations

import importlib.util

import pytest

from features.tool_result_externalization.store_local import (
    LocalToolResultArtifactStore,
)
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService

_HAS_FLASK = importlib.util.find_spec("flask") is not None
pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")

if _HAS_FLASK:
    import server.app as app_module
else:
    app_module = None


def test_first_view_http_rewrites_before_model_and_validates_identity(
    monkeypatch,
    tmp_path,
):
    service = MemoryService(
        OgMemConfig(
            account_id="acct",
            user_id="user",
            agent_id="agent",
            tool_result_externalization_enabled=True,
            tool_result_first_view_enabled=True,
            tool_result_single_chars=20,
            after_turn_threshold=999_999,
        ),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    monkeypatch.setattr(service, "_get_context_fs", lambda: None)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    body = {
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "turn-1",
        "toolCallId": "call-1",
        "toolName": "fixture",
        "content": [{"type": "text", "text": "request_id=req-123\n" * 4}],
        "rankingPrompt": "find request id",
        "accountId": "acct",
        "userId": "user",
        "agentId": "agent",
    }

    response = client.post("/api/v1/govern_first_view_tool_result", json=body)
    invalid = client.post(
        "/api/v1/govern_first_view_tool_result",
        json={**body, "toolCallId": ""},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["externalized"] == 1
    assert payload["identityStrength"] == "strong"
    assert "[ogmemory-tool-result v=2]" in payload["content"][0]["text"]
    assert invalid.status_code == 400
