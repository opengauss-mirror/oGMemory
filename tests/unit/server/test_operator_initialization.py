"""Exercise the real operator service's process-local initialization."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import server.app as app_module
from core.graph_enums import RawDataType
from core.models import RequestContext


def test_concurrent_initialization_keeps_submitted_tasks_queryable(monkeypatch):
    ctx = RequestContext(
        account_id="init-test", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )
    ready = Barrier(4)

    def config():
        time.sleep(0.1)
        return SimpleNamespace(storage_backend="agfs")

    monkeypatch.setattr(app_module, "_operator_service", None)
    monkeypatch.setattr(app_module, "get_config", config)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)

    def submit():
        ready.wait(timeout=5)
        service = app_module._get_operator_service()
        raw = service._graph.create_raw_node(
            raw_type=RawDataType.IMAGE, content="sample-image", ctx=ctx,
        )
        task_id = service.run_pipeline_async(
            raw.node_id, {"steps": [{"operator": "image_caption"}]}, ctx,
        )
        return service, task_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        submissions = list(executor.map(lambda _: submit(), range(4)))

    with app_module.app.test_client() as client:
        for service, task_id in submissions:
            for _ in range(100):
                response = client.get(f"/api/v1/operators/tasks/{task_id}")
                assert response.status_code == 200, response.get_json()
                record = response.get_json()
                if record["status"] == "completed":
                    break
                time.sleep(0.01)
            assert record["status"] == "completed", record
            assert record["result"]["ok"] is True
    assert all(service is app_module._operator_service for service, _ in submissions)
