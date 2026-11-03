"""Scene selection through public query transports over one shared pipeline."""

from types import SimpleNamespace

import pytest

from providers.unified_config import OgMemConfig
from server.mcp_server import create_mcp_server
from server.mcp_tools import query_memory
from server.memory_service import MemoryService
from service.api import ReadAPI
from tests.unit.retrieval.test_pipeline_scene import setup


@pytest.mark.parametrize("transport", ["python", "native", "http", "mcp"])
def test_public_query_scene_selection(setup, monkeypatch, transport):
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="memory"))
    api = ReadAPI(setup.pipeline)
    monkeypatch.setattr(service, "get_read_api", lambda: api)
    monkeypatch.setattr(service, "build_context", lambda *args, **kwargs: setup.ctx)
    monkeypatch.setattr(service, "get_auth_service", lambda: SimpleNamespace(role_control_active=lambda: False))
    monkeypatch.setattr(service, "record_retrieval_metric", lambda *args, **kwargs: None)
    if transport == "http":
        import server.app as app_module
        monkeypatch.setattr(app_module, "_service", service)
        monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: setup.ctx)
        client = app_module.app.test_client()
    if transport == "mcp":
        tool = create_mcp_server(service=service, config=service._cfg)._tool_manager.get_tool("ogm_query_memory")
    for selection, count in ((True, 2), (False, 1), (True, 2)):
        kwargs = {"query": "remember release", "use_scene": selection}
        if transport == "python":
            hits = api.search_memory(ctx=setup.ctx, **kwargs).hits
        elif transport == "native":
            hits = query_memory(service, request_context=setup.ctx, **kwargs)["hits"]
        elif transport == "mcp":
            hits = tool.fn(**kwargs)["hits"]
        else:
            response = client.post("/api/v1/memories/query", json=kwargs)
            assert response.status_code == 200
            hits = response.json["hits"]
        assert len(hits) == count
