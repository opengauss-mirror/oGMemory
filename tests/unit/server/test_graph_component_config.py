"""Graph representation settings reach durable writers and shared components."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.relation_store.projection_identity import FACTORIZED_REPRESENTATION_VERSION
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


@pytest.mark.parametrize("source", ["yaml", "environment"])
def test_factorized_representation_loads_from_supported_settings(monkeypatch, source):
    from providers import unified_config

    raw = {"graph": {}}
    monkeypatch.delenv("GRAPH_PROJECTION_REPRESENTATION", raising=False)
    if source == "yaml":
        raw["graph"]["projection_representation"] = FACTORIZED_REPRESENTATION_VERSION
    else:
        monkeypatch.setenv("GRAPH_PROJECTION_REPRESENTATION", FACTORIZED_REPRESENTATION_VERSION)
    monkeypatch.setattr(unified_config, "_load_yaml", lambda path: raw)
    config = OgMemConfig.load("graph-component-test.yaml")
    assert config.graph_projection_representation == FACTORIZED_REPRESENTATION_VERSION


def test_request_and_background_stores_share_service_component_cache(monkeypatch):
    config = OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql", graph_backend="sql",
        graph_projection_representation=FACTORIZED_REPRESENTATION_VERSION,
        graph_edge_embedding_profile="component-test-profile",
    )
    service = MemoryService(config=config)
    pool = object()
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: SimpleNamespace(model="test", dimension=3))
    stores = []

    def construct(**kwargs):
        store = SimpleNamespace(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr("providers.relation_store.sql_projection.SQLEdgeProjectionStore", construct)
    first, profile = service._graph_projection_components()
    second, second_profile = service._graph_projection_components()
    assert len(stores) == 2
    assert first.component_cache is second.component_cache is service._get_graph_component_cache()
    assert first.pool is second.pool is pool
    assert profile == second_profile == {
        "profile_id": "component-test-profile", "model": "test", "dimension": 3,
        "representation_version": FACTORIZED_REPRESENTATION_VERSION,
        "max_input_chars": 30000, "multimodal": False,
    }


@pytest.mark.parametrize("representation", ["relation-target-v2", FACTORIZED_REPRESENTATION_VERSION])
def test_canonical_relation_writer_receives_configured_representation(monkeypatch, representation):
    config = OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql", graph_backend="sql",
        graph_projection_representation=representation,
        graph_edge_embedding_profile="writer-test-profile",
        sql_connection_string="dbname=component-test",
    )
    service = MemoryService(config=config)
    monkeypatch.setattr("server.memory_service._HAS_SQL", True)
    pool = object()
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    factory = Mock()
    monkeypatch.setattr("server.memory_service.SQLRelationStore", factory)
    result = service._get_relation_store()
    assert result is factory.return_value
    factory.assert_called_once_with(
        pool=pool, edge_projection_profile_id="writer-test-profile",
        edge_projection_representation=representation,
    )
