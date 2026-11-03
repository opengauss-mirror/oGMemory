"""Synchronous after_turn may wait for a concurrent topic commit."""

from contextlib import contextmanager
from unittest.mock import Mock

from core.models import RequestContext
from server.memory_service import _guard_session_mutation
from service import extraction_retry


def test_synchronous_after_turn_waits_for_identity_lock_but_retry_worker_does_not(monkeypatch):
    ctx = RequestContext("acct", "user", "agent", "session", "trace")
    guard = Mock(wait_for_identity_lock=False, post_commit_drain=False)
    manager = Mock()
    manager.save_session_state.return_value = True

    class Service:
        _use_sql = True
        _write_api = None

        def build_context(self, params):
            return ctx

        def get_session_manager(self):
            return manager

        def _param_bool(self, params, *args, default=False):
            return params.get("wait", default) is True

        @contextmanager
        def _session_mutation_scope(self, request_ctx, *, refresh=True):
            yield guard

        @_guard_session_mutation
        def after_turn(self, params):
            return {"ok": True, "wait_for_identity_lock": guard.wait_for_identity_lock}

    service = Service()
    monkeypatch.setattr(extraction_retry, "background_retry_running", lambda: False)
    assert service.after_turn({"messages": [{}], "clientRequestId": "request"})["wait_for_identity_lock"] is True
    guard.wait_for_identity_lock = False
    assert service.after_turn({"messages": [{}], "wait": True})["wait_for_identity_lock"] is True
    guard.wait_for_identity_lock = False
    assert service.after_turn({"messages": [{}]})["wait_for_identity_lock"] is False
    monkeypatch.setattr(extraction_retry, "background_retry_running", lambda: True)
    assert service.after_turn({"messages": [{}], "clientRequestId": "request"})["wait_for_identity_lock"] is False
