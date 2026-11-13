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
    obj._client._get_connection.return_value.alloc_timestamp.return_value = 1001
    obj._client.query.return_value = []
    obj._client.reset_mock()
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
    index._client.query.side_effect = [[{"id": "one", "account_id": "a", "owner_space": "user:u", "level": 0}], []]
    index._client.search.return_value = [
        [
            {
                "id": "one",
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


def test_get_embeddings_text_guard_and_bounded_query(index):
    index._client.query.side_effect = [sorted([
        {"id": IndexRecord.generate_id("x", 0), "level": 0, "account_id": "a", "owner_space": "user:u", "uri": "x", "text": "old", "embedding": [1.0, 0.0, 0.0]},
        {"id": IndexRecord.generate_id("y", 0), "level": 0, "account_id": "a", "owner_space": "user:u", "uri": "y", "text": "same", "embedding": [0.0, 1.0, 0.0]},
    ], key=lambda row: row["id"])]
    assert index.get_embeddings(
        ["x", "y"], {"account_id": "a"}, expected_texts={"x": "new", "y": "same"}
    ) == {"y": [0.0, 1.0, 0.0]}
    calls = index._client.query.call_args_list
    assert all(call.kwargs["timeout"] == 7 for call in calls)
    assert len(calls) == 1
    index._client.query_iterator.assert_not_called()


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


@pytest.mark.parametrize("uri", [
    "https://user:synthetic-password@milvus.example:19530",
    "http://user:synthetic%40password@milvus.example:19530",
    "http://synthetic-user@milvus.example:19530",
    "http://milvus.example:19530",
])
def test_milvus_uri_is_private_but_preserved_for_connections(uri, monkeypatch, caplog):
    import logging

    from providers import unified_config

    config = unified_config.OgMemConfig(milvus_uri=uri)
    monkeypatch.setattr(unified_config, "_global_config", None)
    monkeypatch.setattr(unified_config.OgMemConfig, "load", lambda _: config)
    with caplog.at_level(logging.INFO):
        assert unified_config.get_config() is config
    assert config.dump_summary()["milvus_uri"] == "<configured>"
    projected = config.to_provider_config()
    assert config.milvus_uri == projected.milvus_uri == uri
    assert projected.to_ogmem_config().milvus_uri == uri
    assert uri not in caplog.text
    assert uri not in repr(config)
    assert uri not in repr(projected)


def _schema():
    from pymilvus import DataType

    from providers.vector_index.milvus_index import _SCOPE_FIELDS

    return {
        "auto_id": False,
        "fields": [
            {
                "name": "id",
                "type": DataType.VARCHAR,
                "is_primary": True,
                "params": {"max_length": 512},
            },
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 3}},
            {"name": "text", "type": DataType.VARCHAR, "params": {"max_length": 65535}},
            {"name": "metadata", "type": DataType.JSON},
            {"name": "level", "type": DataType.INT64},
            *[
                {"name": key, "type": DataType.VARCHAR, "params": {"max_length": 8192}}
                for key in _SCOPE_FIELDS
            ],
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
    client.list_indexes.side_effect = [[], ["embedding"], ["embedding"]]
    MilvusVectorIndex(dimension=3, timeout=7)
    client.create_collection.assert_not_called()
    assert client.create_index.call_count == 7
    assert client.create_index.call_args.kwargs["timeout"] == 7
    client.load_collection.assert_called_once_with(collection_name="contextengine", timeout=7)


def test_startup_tolerates_only_duplicate_creation(monkeypatch):
    milvus_error = pytest.importorskip("pymilvus").MilvusException

    client = _client(monkeypatch)
    client.has_collection.return_value = False
    client.create_collection.side_effect = milvus_error(65535, "collection already exists")
    client.list_indexes.side_effect = [[], ["embedding"], ["embedding"]]
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
    index._client.query.side_effect = [[], [], [{"id": "one", "level": 0, "account_id": "a", "owner_space": "user:u"}], [], [], [], [], []]
    index.upsert([record()])
    index.delete(["one"])
    index.search_by_vector([1.0, 0.0, 0.0], {"account_id": "a"}, 1)
    index.get_record("one")
    index.count()
    index.get_embeddings(["uri"], {"account_id": "a"})
    index._client.query.side_effect = [[], [{"id": "one", "logical_id": None}]]
    index.delete_account_data("a")
    for method in ["upsert", "delete", "search", "get", "query"]:
        assert getattr(index._client, method).call_args.kwargs["timeout"] == 7.0
    assert "text" not in index._client.search.call_args.kwargs["output_fields"]


@pytest.mark.parametrize(
    "field,value", [("text", "中" * 110000), ("id", "x" * 513), ("metadata", {"blob": "x" * 65536})]
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


def test_delete_scope_uses_bounded_pages_and_server_filter(index):
    index.count = MagicMock(return_value=1)
    index._client.query.return_value = [{"id": "legacy", "logical_id": None}]
    assert index.delete_by_owner_space("a", "user:u") == 1
    query = index._client.query.call_args.kwargs
    assert '(account_id == "a") and (owner_space == "user:u")' in query["filter"]
    assert query["limit"] == 1024
    assert index._client.delete.call_args.kwargs == {
        "collection_name": "test", "ids": ["legacy"], "timeout": 7.0,
    }
    index._client.query_iterator.assert_not_called()


def test_get_record_preserves_scope_when_reindexed(index):
    item = record(parent_uri="stale", extractor_type="stale")
    item.filters.update(
        parent_uri="ctx://a/users/u/memories",
        extractor_type="memory",
        space_id="space-1",
        resource_type="document",
        category="entity",
        context_type="MEMORY",
    )
    index.upsert([item])
    original = dict(index._client.upsert.call_args.kwargs["data"][0])
    index._client.query.side_effect = [[original], [], []]
    index._client.get.return_value = [original]
    saved = index.get_record(item.id)
    assert saved.filters["parent_uri"] == "ctx://a/users/u/memories"
    assert saved.metadata["parent_uri"] == "stale"  # Preserve the original JSON payload.
    assert saved.filters["extractor_type"] == "memory"
    index.upsert([saved])
    rewritten = index._client.upsert.call_args.kwargs["data"][0]
    for key in (
        "account_id",
        "owner_space",
        "space_id",
        "parent_uri",
        "extractor_type",
        "resource_type",
        "category",
        "context_type",
    ):
        assert rewritten[key] == original[key]


@pytest.mark.parametrize("name,limit", [("id", 511), ("text", 65534), ("account_id", 8191)])
def test_existing_short_varchar_schema_is_rejected(monkeypatch, name, limit):
    client = _client(monkeypatch)
    client.has_collection.return_value = True
    schema = _schema()
    next(field for field in schema["fields"] if field["name"] == name)["params"]["max_length"] = (
        limit
    )
    client.describe_collection.return_value = schema
    client.list_indexes.return_value = ["embedding"]
    with pytest.raises(ValueError, match="schema"):
        MilvusVectorIndex(dimension=3)
    client.load_collection.assert_not_called()
    client.close.assert_called_once()


@pytest.mark.parametrize("vector", [[1e100, 0.0, 0.0], [1e-100, 0.0, 0.0]])
def test_float32_overflow_or_zero_after_conversion_is_rejected(index, vector):
    with pytest.raises(ValueError, match="finite and non-zero"):
        index.upsert([record(_embedding=vector)])
    index._client.upsert.assert_not_called()


@pytest.mark.parametrize("capacity", [65387, 65536])
def test_metadata_at_capacity_does_not_grow_after_read_and_rewrite(index, capacity):
    item = record()
    item.filters.update(
        parent_uri="ctx://a/users/u/memories", space_id="space-1", extractor_type="memory"
    )
    item.metadata["payload"] = ""
    index.upsert([item])
    empty = index._client.upsert.call_args.kwargs["data"][0]["metadata"]
    overhead = len(json.dumps(empty, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    item.metadata["payload"] = "x" * (capacity - overhead)
    index.upsert([item])
    original = dict(index._client.upsert.call_args.kwargs["data"][0])
    assert len(json.dumps(original["metadata"], ensure_ascii=False).encode("utf-8")) == capacity
    for _ in range(3):
        index._client.query.side_effect = [[original], [], []]
        index._client.get.return_value = [original]
        saved = index.get_record(item.id)
        assert {k: v for k, v in saved.metadata.items() if k != "_embedding"} == original[
            "metadata"
        ]
        index.upsert([saved])
        rewritten = index._client.upsert.call_args.kwargs["data"][0]
        assert rewritten["metadata"] == original["metadata"]
        assert rewritten["space_id"] == "space-1"
        assert rewritten["parent_uri"] == "ctx://a/users/u/memories"


@pytest.mark.parametrize("text", ["x" * 73662, "中🙂" * 37449, "x" * 262144])
def test_large_text_survives_read_rewrite_and_shrinking(index, text):
    item = record()
    item.text = text
    index.upsert([item])
    row = dict(index._client.upsert.call_args.kwargs["data"][0])
    assert len(row["text"].encode()) <= 65535
    index._client.query.side_effect = [[row], [], []]
    index._client.get.return_value = [row]
    saved = index.get_record(item.id)
    assert saved.text == text
    index.upsert([saved])
    assert index._client.upsert.call_args.kwargs["data"][0] == row
    index._client.query.side_effect = [[row], [row], [], []]
    assert index.get_embeddings([item.uri], {}, expected_texts={item.uri: text})
    item.text = "short"
    index.upsert([item])
    rewritten = index._client.upsert.call_args.kwargs["data"][0]
    assert index._text(rewritten) == "short" and rewritten["text_1"] == ""


def test_keyset_pagination_escapes_primary_keys_and_fails_on_unordered_pages(index):
    index._client.query.side_effect = [[{"id": 'a"', "uri": "one"}], []]
    assert list(index._query_rows({"account_id": "a"}, ["uri"], page_size=1)) == [{"id": 'a"', "uri": "one"}]
    assert 'id > "a\\\""' in index._client.query.call_args.kwargs["filter"]
    index._client.query.side_effect = [[{"id": "b"}, {"id": "a"}]]
    with pytest.raises(RuntimeError, match="unordered"):
        list(index._query_rows({}, ["uri"]))


def test_query_timeout_propagates_without_using_sdk_iterator(index):
    index._client.query.side_effect = TimeoutError("deadline exceeded")
    with pytest.raises(TimeoutError):
        index.get_embeddings(["uri"], {})
    assert index._client.query.call_args.kwargs["timeout"] == 7
    index._client.query_iterator.assert_not_called()


def test_scalar_indexes_preserve_existing_and_fill_missing(index):
    pytest.importorskip("pymilvus")
    index._client.list_indexes.return_value = ["custom_account"]
    index._client.describe_index.return_value = {"field_name": "account_id"}
    index._ensure_scalar_indexes()
    fields = [c.kwargs["field_name"] for c in index._client.prepare_index_params.return_value.add_index.call_args_list]
    assert set(fields) == {"owner_space", "level", "parent_uri", "uri", "logical_id"}
    assert index._client.create_index.call_count == len(fields)
    assert all(c.kwargs["timeout"] == 7 for c in index._client.create_index.call_args_list)


def test_bulk_embeddings_use_one_rpc_and_do_not_fetch_documents(index):
    rows = [{"id": IndexRecord.generate_id(f"uri-{number}", 0), "level": 0, "account_id": "a", "owner_space": "user:u", "uri": f"uri-{number}",
             "embedding": [1., 0., 0.]} for number in range(128)]
    index._client.query.return_value = sorted(rows, key=lambda row: row["id"])
    assert len(index.get_embeddings([r["uri"] for r in rows], {"account_id": "a"})) == 128
    index._client.query.assert_called_once()
    call = index._client.query.call_args.kwargs
    assert call["output_fields"] == ["id", "logical_id", "account_id", "owner_space", "uri", "level", "embedding"]
    assert call["limit"] == 256 and call["timeout"] == 7
    assert call["consistency_level"] == "Strong"


def test_keyset_pagination_stops_on_short_page_and_keeps_same_uri_records(index):
    rows = [{"id": f"id-{number:03}", "level": 0, "account_id": "a", "owner_space": "user:u", "uri": "same", "embedding": [1., 0., 0.]}
            for number in range(529)]
    page = iter([rows[:256], rows[256:512], rows[512:]])
    def query(**kwargs):
        if kwargs["filter"].startswith("logical_id in"):
            keys = json.loads(kwargs["filter"].split(" in ", 1)[1].split(" or ", 1)[0])
            return [row for row in rows if row["id"] in keys]
        if kwargs["filter"].startswith("level == -2"):
            return []
        return next(page)
    index._client.query.side_effect = query
    assert index.get_embeddings(["same"], {}) == {"same": [1., 0., 0.]}
    calls = index._client.query.call_args_list
    calls = [c for c in calls if "embedding" in c.kwargs["output_fields"]]
    assert len(calls) == 3
    assert 'id > "id-255"' in calls[1].kwargs["filter"]
    assert 'id > "id-511"' in calls[2].kwargs["filter"]


def test_expected_texts_use_bounded_long_document_pages(index):
    index._client.query.return_value = []
    index.get_embeddings(["same"], {}, expected_texts={"same": "body"})
    call = index._client.query.call_args.kwargs
    worst_row_bytes = 5 * 65535 + 3 * 4 + 512 + 8192 + 1024
    assert 128 < call["limit"] <= 256
    assert call["limit"] * worst_row_bytes <= 48 * 1024 * 1024
    assert all(name in call["output_fields"] for name in ("text", "text_1", "text_4"))


@pytest.mark.parametrize("magnitude", [1e38, 1e20, 1e-20, 1e-30])
def test_float32_squared_norm_must_be_safe_before_any_batch_is_written(index, magnitude):
    with pytest.raises(ValueError, match="norm.*FLOAT32"):
        index.upsert([record(), record(_embedding=[magnitude, 0., 0.])])
    index._client.upsert.assert_not_called()
    with pytest.raises(ValueError, match="norm.*FLOAT32"):
        index.search_by_vector([magnitude, 0., 0.], {}, 1)
    index._client.search.assert_not_called()


@pytest.mark.parametrize("dimension", [1024, 32768])
def test_expected_text_query_budget_includes_dimension_and_all_text_fields(index, dimension):
    index._dimension = dimension
    index._client.query.return_value = []
    index.get_embeddings(["uri"], {}, expected_texts={"uri": "short"})
    call = index._client.query.call_args.kwargs
    worst_row = dimension * 4 + 512 + 8192 + 1024 + 5 * 65535
    assert call["limit"] * worst_row <= 48 * 1024 * 1024
    assert call["limit"] > 32
