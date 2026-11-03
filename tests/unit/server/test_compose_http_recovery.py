from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import server.app as app_module
from core.models import RequestContext
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from server.memory_service import _guard_session_mutation


class GuardedComposeService:
    _cfg = SimpleNamespace(http_ip_allowlist=[])
    _use_sql = True
    _write_api = None

    def __init__(self):
        self.busy = True
        self.composed = []

    @contextmanager
    def _session_mutation_scope(self, ctx, *, refresh=True):
        if self.busy:
            raise SessionMutationBusyError("session ownership is busy")
        yield None

    @_guard_session_mutation
    def compose(self, params):
        self.composed.append(params["prompt"])
        return {"retrievedEvidence": "verified memory", "messages": params["messages"]}


@pytest.mark.parametrize("perf_enabled", [False, True])
def test_compose_http_recovers_after_session_ownership_is_available(monkeypatch, perf_enabled):
    service = GuardedComposeService()
    ctx = RequestContext(account_id="acct", user_id="user", agent_id="agent",
                         session_id="session", trace_id="trace")
    recorder = SimpleNamespace(enabled=perf_enabled, snapshot_tokens=lambda: {},
                               finalize_tokens=lambda after, before: ({}, None, "none"))
    events = []
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    monkeypatch.setattr(app_module, "get_recorder", lambda: recorder)
    monkeypatch.setattr(app_module, "_emit_perf_stage", lambda *args: events.append(args))
    client = app_module.app.test_client()
    params = {"sessionId": "session", "prompt": "recall the event", "messages": []}

    busy = client.post("/api/v1/compose", json=params)
    service.busy = False
    ready = client.post("/api/v1/compose", json=params)

    assert busy.status_code == 503
    assert busy.get_json() == {"ok": False, "status": "processing", "retryable": True}
    assert ready.status_code == 200
    assert ready.get_json() == {"retrievedEvidence": "verified memory", "messages": []}
    assert service.composed == ["recall the event"]
    assert [event[6] for event in events] == ([False, True] if perf_enabled else [])
