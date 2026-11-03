from __future__ import annotations

import pytest

import server.app as app_module


class _Service:
    def __init__(self):
        self.shutdown_calls = 0
        self.recovery_starts = 0

    def start_extraction_recovery(self):
        self.recovery_starts += 1

    def shutdown(self):
        self.shutdown_calls += 1


def test_gunicorn_worker_starts_recovery_once(monkeypatch):
    from server.gunicorn_config import post_worker_init
    service = _Service()
    monkeypatch.setattr(app_module, "_service", None)
    monkeypatch.setattr(app_module, "MemoryService", lambda config: service)
    post_worker_init(None)
    assert app_module._get_service() is service
    assert service.recovery_starts == 1


def test_shutdown_service_closes_initialized_service_once(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)

    app_module._shutdown_service()
    app_module._shutdown_service()

    assert service.shutdown_calls == 1
    assert app_module._service is None


def test_run_http_server_shuts_down_service_when_server_exits(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "get_config", lambda: type("Config", (), {"http_port": 8090})())

    def stop_server(**kwargs):
        raise RuntimeError("server stopped")

    monkeypatch.setattr(app_module.app, "run", stop_server)

    with pytest.raises(RuntimeError, match="server stopped"):
        app_module._run_http_server()

    assert service.shutdown_calls == 1
    assert app_module._service is None
