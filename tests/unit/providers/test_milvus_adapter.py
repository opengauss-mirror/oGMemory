"""Provider contract regressions without a running Milvus server."""

import json
from unittest.mock import MagicMock

import pytest

from core.models import IndexRecord
from providers.vector_index.milvus_index import MilvusVectorIndex


@pytest.fixture
def index():
    obj = MilvusVectorIndex.__new__(MilvusVectorIndex)
    obj._timeout = 7.0
    obj._dimension = 3
    obj._collection_name = "test"
    obj._client = MagicMock()
    return obj


def record(**kwargs):
    return IndexRecord(
        id="one",
        uri="ctx://a/users/u/memories/entities/test",
        level=0,
        text="source",
        filters={"account_id": "a", "owner_space": "user:u"},
        metadata={"_embedding": [1.0, 0.0, 0.0], "grounding": [{"uri": "source"}], **kwargs},
    )


def test_upsert_scope_and_grounding_roundtrip(index):
    index.upsert([record(account_id="untrusted", uri="untrusted")])
    row = index._client.upsert.call_args.kwargs["data"][0]
    assert row["account_id"] == "a"
    assert row["uri"] == "ctx://a/users/u/memories/entities/test"
    assert row["memory_directory"] == "entities"
    assert "_embedding" not in row["metadata"]
    assert row["metadata"]["grounding"] == [{"uri": "source"}]


@pytest.mark.parametrize(
    "vector", [None, [], [1.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]
)
def test_reject_invalid_embeddings_before_write(index, vector):
    with pytest.raises(ValueError):
        index.upsert([record(_embedding=vector)])
    index._client.upsert.assert_not_called()


def test_reject_missing_tenant_before_write(index):
    item = record()
    item.filters = {}
    with pytest.raises(ValueError, match="account_id"):
        index.upsert([item])
    index._client.upsert.assert_not_called()


def test_filter_escaping_empty_lists_and_unknown_scope():
    hostile = 'a" or account_id != "a'
    expression = MilvusVectorIndex._build_filter({"account_id": hostile, "owner_space": []})
    assert f"account_id == {json.dumps(hostile)}" in expression
    assert "id in []" in expression
    with pytest.raises(ValueError, match="Unsupported"):
        MilvusVectorIndex._build_filter({"unknown_scope": "x"})


def test_search_filters_before_top_k_and_preserves_hit(index):
    index._client.search.return_value = [
        [
            {
                "distance": 0.9,
                "entity": {
                    "uri": "ctx://a/test",
                    "level": 0,
                    "metadata": {"grounding": [1], "abstract": "summary"},
                    "owner_space": "user:u",
                    "account_id": "a",
                    "parent_uri": "ctx://a/parent",
                },
            }
        ]
    ]
    hits = index.search_children(
        "ctx://a/parent", [1.0, 0.0, 0.0], {"account_id": "a", "memory_directory": ["entities"]}, 1
    )
    call = index._client.search.call_args.kwargs
    assert 'parent_uri == "ctx://a/parent"' in call["filter"]
    assert 'memory_directory in ["entities"]' in call["filter"]
    assert call["limit"] == 1 and call["consistency_level"] == "Strong"
    assert hits[0].metadata["grounding"] == [1]
    assert hits[0].abstract == "summary" and hits[0].score == 0.9


def test_get_embeddings_text_guard_and_iterator_close(index):
    iterator = index._client.query_iterator.return_value
    iterator.next.side_effect = [
        [
            {"uri": "x", "text": "old", "embedding": [1.0, 0.0, 0.0]},
            {"uri": "y", "text": "same", "embedding": [0.0, 1.0, 0.0]},
        ],
        [],
    ]
    assert index.get_embeddings(
        ["x", "y"], {"account_id": "a"}, expected_texts={"x": "new", "y": "same"}
    ) == {"y": [0.0, 1.0, 0.0]}
    iterator.close.assert_called_once()


def test_empty_operations_do_not_call_provider(index):
    index.upsert([])
    index.delete([])
    assert index.get_embeddings([], {}) == {}
    assert index.search_by_vector([], {}, 0) == []
    assert not index._client.mock_calls


def test_config_yaml_env_projection_and_roundtrip(tmp_path, monkeypatch):
    from providers.unified_config import OgMemConfig

    path = tmp_path / "milvus.yaml"
    path.write_text(
        "vector_db:\n  type: milvus\n  dimension: 3\n  milvus_collection: test_vectors\n"
    )
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "test-secret")
    config = OgMemConfig.load(str(path)).to_provider_config()
    assert config.vector_db_type == "milvus"
    assert config.milvus_collection_name == "test_vectors"
    assert config.milvus_uri == "http://milvus:19530"
    assert config.embedding_dimension == 3
    assert "test-secret" not in repr(config)
    assert config.to_ogmem_config().milvus_token == "test-secret"


def test_factory_routes_milvus(monkeypatch):
    from providers.config import ProviderConfig
    from providers.vector_index import factory

    constructor = MagicMock()
    monkeypatch.setattr("providers.vector_index.milvus_index.MilvusVectorIndex", constructor)
    factory.create_vector_index(ProviderConfig(vector_db_type=" MILVUS ", embedding_dimension=3))
    assert constructor.call_args.kwargs == dict(
        uri="http://localhost:19530",
        token="",
        collection_name="contextengine",
        dimension=3,
        db_name="default",
        timeout=30.0,
    )


def _schema():
    from pymilvus import DataType

    from providers.vector_index.milvus_index import _SCOPE_FIELDS

    return {
        "auto_id": False,
        "fields": [
            {"name": "id", "type": DataType.VARCHAR, "is_primary": True},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 3}},
            {"name": "text", "type": DataType.VARCHAR},
            {"name": "metadata", "type": DataType.JSON},
            {"name": "level", "type": DataType.INT64},
            *[{"name": key, "type": DataType.VARCHAR} for key in _SCOPE_FIELDS],
        ],
    }


def _client(monkeypatch):
    pytest.importorskip("pymilvus")
    client = MagicMock()
    client.describe_collection.return_value = _schema()
    client.describe_index.return_value = {"field_name": "embedding", "metric_type": "COSINE"}
    monkeypatch.setattr("pymilvus.MilvusClient", lambda **kwargs: client)
    return client


def test_startup_repairs_schema_left_without_index(monkeypatch):
    client = _client(monkeypatch)
    client.has_collection.return_value = True
    client.list_indexes.side_effect = [[], ["embedding"]]
    MilvusVectorIndex(dimension=3, timeout=7)
    client.create_collection.assert_not_called()
    client.create_index.assert_called_once()
    assert client.create_index.call_args.kwargs["timeout"] == 7
    client.load_collection.assert_called_once_with(collection_name="contextengine", timeout=7)


def test_startup_tolerates_only_duplicate_creation(monkeypatch):
    milvus_error = pytest.importorskip("pymilvus").MilvusException

    client = _client(monkeypatch)
    client.has_collection.return_value = False
    client.create_collection.side_effect = milvus_error(65535, "collection already exists")
    client.list_indexes.side_effect = [[], ["embedding"]]
    client.create_index.side_effect = milvus_error(65535, "index already exists")
    MilvusVectorIndex(dimension=3)
    client.load_collection.assert_called_once()


@pytest.mark.parametrize("method", ["create_collection", "create_index"])
def test_startup_preserves_permissions_and_transport_errors(monkeypatch, method):
    milvus_error = pytest.importorskip("pymilvus").MilvusException

    client = _client(monkeypatch)
    client.has_collection.return_value = False
    client.list_indexes.return_value = []
    getattr(client, method).side_effect = milvus_error(1, "permission denied")
    with pytest.raises(milvus_error, match="permission denied"):
        MilvusVectorIndex(dimension=3)
    client.close.assert_called_once()


def test_schema_dimension_mismatch_fails_before_index_creation(monkeypatch):
    client = _client(monkeypatch)
    client.has_collection.return_value = True
    schema = _schema()
    schema["fields"][1]["params"]["dim"] = 4
    client.describe_collection.return_value = schema
    with pytest.raises(ValueError, match="schema/dimension"):
        MilvusVectorIndex(dimension=3)
    client.create_index.assert_not_called()
    client.close.assert_called_once()


def test_timeout_propagates_through_runtime_operations(index):
    index._client.search.return_value = [[]]
    index._client.get.return_value = []
    index._client.query.return_value = [{"count(*)": 0}]
    index._client.query_iterator.return_value.next.return_value = []
    index.upsert([record()])
    index.delete(["one"])
    index.search_by_vector([1.0, 0.0, 0.0], {"account_id": "a"}, 1)
    index.get_record("one")
    index.count()
    index.get_embeddings(["uri"], {"account_id": "a"})
    for method in ["upsert", "delete", "search", "get", "query", "query_iterator"]:
        assert getattr(index._client, method).call_args.kwargs["timeout"] == 7.0
    assert "text" not in index._client.search.call_args.kwargs["output_fields"]


@pytest.mark.parametrize(
    "field,value", [("text", "中" * 21846), ("id", "x" * 513), ("metadata", {"blob": "x" * 65536})]
)
def test_reject_oversized_utf8_before_any_batch(index, field, value):
    items = [record() for _ in range(129)]
    if field == "metadata":
        items[-1].metadata.update(value)
    else:
        setattr(items[-1], field, value)
    with pytest.raises(ValueError, match="UTF-8"):
        index.upsert(items)
    index._client.upsert.assert_not_called()


def test_delete_scope_uses_server_filter_without_materializing_ids(index):
    index._client.query.return_value = [{"count(*)": 100_000}]
    assert index.delete_by_owner_space("a", "user:u") == 100_000
    assert index._client.delete.call_args.kwargs == {
        "collection_name": "test",
        "filter": '(account_id == "a") and (owner_space == "user:u")',
        "timeout": 7.0,
    }
    index._client.query_iterator.assert_not_called()
