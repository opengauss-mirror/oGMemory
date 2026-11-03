"""Real SQL initialization and HTTP extraction against a pre-existing raw node."""

import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import server.app as app_module
from core.graph_enums import RawDataType
from core.models import RequestContext
from providers.graph_store import SQLGraphStore
from service.graph_service import GraphService

pytestmark = pytest.mark.integration


def test_operator_initialization_reads_and_persists_sql_graph(monkeypatch):
    dsn = os.environ.get("OGMEM_TEST_DSN")
    if not dsn:
        pytest.skip("Set OGMEM_TEST_DSN to an isolated PostgreSQL database")
    ctx = RequestContext(
        account_id=f"operator-init-{uuid4()}", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )
    store = SQLGraphStore(dsn)
    graph = GraphService(store)
    raw = graph.create_raw_node(
        raw_type=RawDataType.IMAGE, content="sample-image", ctx=ctx,
    )
    monkeypatch.setattr(app_module, "_operator_service", None)
    monkeypatch.setattr(app_module, "get_config", lambda: SimpleNamespace(
        storage_backend="sql", sql_connection_string=dsn, sql_pool_size=2,
    ))
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
    try:
        with app_module.app.test_client() as client:
            response = client.post(
                "/api/v1/operators/image_caption/run",
                json={"raw_node_id": raw.node_id},
            )
        assert response.status_code == 200, response.get_json()
        result = response.get_json()
        assert result["ok"] is True
        persisted = store.get_node(result["extracted_node_id"], ctx)
        assert UUID(persisted.source_raw_id) == UUID(raw.node_id)
        assert persisted.extracted_content == result["extracted_content"]
    finally:
        connection = store._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM graph_nodes WHERE account_id = %s", (ctx.account_id,))
            connection.commit()
        finally:
            connection.close()
            for connection in store._pool:
                connection.close()
            service = app_module._operator_service
            if service is not None:
                for connection in getattr(service._graph._store, "_pool", []):
                    connection.close()
