"""Real Milvus acceptance. Set OGMEM_TEST_MILVUS_URI to opt in.

Only a UUID-named collection created by this test is removed during cleanup.
"""

import os
from uuid import uuid4

import pytest

from core.models import IndexRecord
from providers.vector_index.milvus_index import MilvusVectorIndex

pytestmark = pytest.mark.integration


@pytest.fixture
def index():
    uri = os.environ.get("OGMEM_TEST_MILVUS_URI")
    if not uri:
        pytest.skip("OGMEM_TEST_MILVUS_URI required for real Milvus acceptance")
    pytest.importorskip("pymilvus")
    obj = MilvusVectorIndex(
        uri=uri,
        token=os.environ.get("MILVUS_TOKEN", ""),
        collection_name="ogmemory_acceptance_" + uuid4().hex,
        dimension=3,
    )
    try:
        yield obj
    finally:
        obj._client.drop_collection(collection_name=obj._collection_name)
        obj.close()


def make_record(name, account="a", owner="user:u", directory="entities", vector=None, level=0):
    uri = f"ctx://{account}/users/u/memories/{directory}/{name}"
    return IndexRecord(
        id=IndexRecord.generate_id(uri, level),
        uri=uri,
        level=level,
        text="text " + name,
        filters={"account_id": account, "owner_space": owner},
        metadata={
            "_embedding": vector or [1.0, 0.0, 0.0],
            "abstract": "summary " + name,
            "parent_uri": f"ctx://{account}/parent",
            "category": "entity",
            "context_type": "MEMORY",
            "grounding": [{"source": name}],
        },
    )


def test_real_milvus_lifecycle_isolation_and_reconnect(index):
    target = make_record("target", vector=[0.8, 0.2, 0.0])
    other = make_record("other", account="b")
    outsider = make_record("outsider", owner="user:other")
    wrong_directory = make_record("preference", directory="preferences")
    content = make_record("target", vector=[0.8, 0.2, 0.0], level=2)
    index.upsert([target, other, outsider, wrong_directory, content])
    index.upsert([target])
    assert index.count() == 5
    scope = {
        "account_id": "a",
        "owner_space": "user:u",
        "level": 0,
        "memory_directory": ["entities"],
    }
    # Closer vectors belong to other scopes; filtering must happen before top_k.
    hits = index.search_by_vector([1.0, 0.0, 0.0], scope, 1)
    assert [hit.uri for hit in hits] == [target.uri]
    assert hits[0].score > 0.9
    assert hits[0].metadata["grounding"] == [{"source": "target"}]
    assert index.search_children("ctx://a/parent", [1.0, 0.0, 0.0], scope, 1)[0].uri == target.uri
    assert index.search_children("ctx://missing", [1.0, 0.0, 0.0], scope, 1) == []
    assert index.search_by_vector([1.0, 0.0, 0.0], {"account_id": []}, 1) == []
    assert index.count({"account_id": "b"}) == 1
    assert index.get_record(target.id).text == target.text
    assert index.get_embeddings([target.uri], scope, expected_texts={target.uri: "stale"}) == {}
    assert index.get_embeddings([target.uri], scope, expected_texts={target.uri: target.text})[
        target.uri
    ][0] == pytest.approx(0.8)
    target.text = "updated text"
    index.upsert([target])
    assert index.count() == 5
    assert index.get_record(target.id).text == "updated text"
    connection = dict(
        uri=os.environ["OGMEM_TEST_MILVUS_URI"],
        token=os.environ.get("MILVUS_TOKEN", ""),
        collection_name=index._collection_name,
        dimension=3,
    )
    reopened = MilvusVectorIndex(**connection)
    try:
        assert reopened.count() == 5
        assert reopened.search_by_vector([1.0, 0.0, 0.0], scope, 1)[0].uri == target.uri
    finally:
        reopened.close()
    with pytest.raises(ValueError, match="schema/dimension"):
        MilvusVectorIndex(**{**connection, "dimension": 4})
    assert index.delete_by_owner_space("a", "user:other") == 1
    assert index.count({"account_id": "b"}) == 1
    assert index.delete_account_data("a") == 3
    assert index.count() == 1
    index.delete([other.id, "nonexistent"])
    assert index.count() == 0


def test_real_filter_escaping_and_code_metadata(index):
    item = make_record("code")
    item.filters["account_id"] = 'quoted"account'
    item.metadata.update(resource_type="code", chunk_key="chunk-one", language="python")
    index.upsert([item])
    hits = index.search_by_vector(
        [1.0, 0.0, 0.0], {"account_id": 'quoted"account', "resource_type": "code"}, 1
    )
    assert hits[0].metadata["chunk_key"] == "chunk-one"
    assert index.search_by_vector([1.0, 0.0, 0.0], {"account_id": 'x" or id != "x'}, 1) == []
    assert index.count({"resource_type": "code"}) == 1


def test_real_paged_embedding_reads_and_owner_cleanup(index):
    records = [make_record(f"paged-{number}") for number in range(1005)]
    index.upsert(records)
    assert index.count({"account_id": "a"}) == 1005
    vectors = index.get_embeddings([record.uri for record in records], {"account_id": "a"})
    assert len(vectors) == 1005
    assert index.delete_by_owner_space("a", "user:u") == 1005
    assert index.count() == 0


def test_real_restart_repairs_missing_index_without_losing_data(index):
    item = make_record("recover-index")
    index.upsert([item])
    index._client.release_collection(collection_name=index._collection_name, timeout=30)
    index._client.drop_index(
        collection_name=index._collection_name, index_name="embedding", timeout=30
    )
    reopened = MilvusVectorIndex(
        uri=os.environ["OGMEM_TEST_MILVUS_URI"],
        token=os.environ.get("MILVUS_TOKEN", ""),
        collection_name=index._collection_name,
        dimension=3,
    )
    try:
        assert reopened.count() == 1
        assert reopened.search_by_vector([1.0, 0.0, 0.0], {"account_id": "a"}, 1)[0].uri == item.uri
    finally:
        reopened.close()


def test_real_concurrent_first_startup(index):
    from concurrent.futures import ThreadPoolExecutor

    name = "ogmemory_concurrent_" + uuid4().hex
    clients = []

    def create(_):
        return MilvusVectorIndex(
            uri=os.environ["OGMEM_TEST_MILVUS_URI"],
            token=os.environ.get("MILVUS_TOKEN", ""),
            collection_name=name,
            dimension=3,
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create, number) for number in range(4)]
            failures = []
            for future in futures:
                try:
                    clients.append(future.result())
                except Exception as exc:
                    failures.append(exc)
            assert not failures, [str(exc) for exc in failures]
        clients[0].upsert([make_record("concurrent")])
        assert all(client.count() == 1 for client in clients)
    finally:
        if index._client.has_collection(collection_name=name):
            index._client.drop_collection(collection_name=name)
        for client in clients:
            client.close()


def test_real_get_record_reindex_preserves_filter_only_scope(index):
    item = make_record("filter-roundtrip")
    item.filters.update(
        video_id="video-1", extractor_type="episodic_memory_generator", space_id="space-1"
    )
    # Reindexing must use persisted scalar filters without expanding the JSON payload.
    item.metadata["video_id"] = "stale"
    index.upsert([item])
    saved = index.get_record(item.id)
    assert saved.filters["video_id"] == "video-1"
    assert saved.metadata["video_id"] == "stale"  # Preserve the original JSON payload.
    assert saved.filters["extractor_type"] == "episodic_memory_generator"
    index.upsert([saved])
    scope = {
        "account_id": "a",
        "video_id": "video-1",
        "space_id": "space-1",
        "extractor_type": "episodic_memory_generator",
    }
    assert index.count(scope) == 1
    assert index.search_by_vector([1.0, 0.0, 0.0], scope, 1)[0].uri == item.uri


def test_real_metadata_at_capacity_survives_read_and_reindex(index):
    import json

    item = make_record("metadata-capacity")
    item.filters.update(video_id="video-1", space_id="space-1")
    item.metadata["payload"] = ""
    metadata = {
        "abstract": item.text[:200],
        **{k: v for k, v in item.metadata.items() if k != "_embedding"},
    }
    overhead = len(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    item.metadata["payload"] = "x" * (65536 - overhead)
    index.upsert([item])
    for _ in range(2):
        saved = index.get_record(item.id)
        metadata = {k: v for k, v in saved.metadata.items() if k != "_embedding"}
        assert len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) == 65536
        assert saved.filters["space_id"] == "space-1"
        index.upsert([saved])
    assert index.count({"account_id": "a", "space_id": "space-1", "video_id": "video-1"}) == 1


def test_real_large_unicode_body_and_same_uri_multiple_records(index):
    from dataclasses import replace

    item = make_record("long-skill", level=2)
    item.text = "中🙂" * 37449
    siblings = [replace(item, id=f"unit-{n:03}", text=f"unit-{n:03}") for n in range(69)]
    index.upsert([item, *siblings])
    assert index.count() == 70
    assert index.get_record(item.id).text == item.text
    assert index.get_embeddings([item.uri], {"account_id": "a"}, level=2,
                                expected_texts={item.uri: item.text})
    # The last sibling is beyond the first two keyset pages for this URI.
    assert index.get_embeddings([item.uri], {"account_id": "a"}, level=2,
                                expected_texts={item.uri: siblings[-1].text})
    saved = index.get_record(item.id)
    index.upsert([saved])
    assert index.get_record(item.id).text == item.text
    index.delete([item.id])
    assert index.get_record(item.id) is None and index.count() == 69


def test_real_scalar_index_repair_and_legacy_text_migration(index):
    item = make_record("legacy")
    index.upsert([item])
    # Simulate the original schema without suffix fields in a separate collection.
    from pymilvus import DataType, MilvusClient

    from providers.vector_index.milvus_index import _SCOPE_FIELDS, _VARCHAR_LIMITS

    name = "ogmemory_legacy_" + uuid4().hex
    client = index._client
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=3)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("metadata", DataType.JSON)
    schema.add_field("level", DataType.INT64)
    for field in _SCOPE_FIELDS:
        schema.add_field(field, DataType.VARCHAR, max_length=_VARCHAR_LIMITS[field])
    client.create_collection(collection_name=name, schema=schema)
    reopened = None
    try:
        row = client.get(collection_name=index._collection_name, ids=[item.id])[0]
        row = {k: v for k, v in row.items() if not k.startswith("text_")}
        client.insert(collection_name=name, data=[row])
        reopened = MilvusVectorIndex(uri=os.environ["OGMEM_TEST_MILVUS_URI"],
                                     token=os.environ.get("MILVUS_TOKEN", ""),
                                     collection_name=name, dimension=3)
        assert reopened.get_record(item.id).text == item.text
        names = client.list_indexes(collection_name=name)
        fields = {client.describe_index(collection_name=name, index_name=n)["field_name"] for n in names}
        assert {"embedding", "account_id", "owner_space", "level", "parent_uri", "uri"} <= fields
    finally:
        if reopened:
            reopened.close()
        client.drop_collection(collection_name=name)


def test_real_bulk_embedding_reads_need_one_bounded_rpc(index, monkeypatch):
    records = [make_record(f"bulk-{number}") for number in range(128)]
    index.upsert(records)
    query = index._client.query
    calls = []

    def tracked_query(**kwargs):
        calls.append(kwargs)
        return query(**kwargs)

    monkeypatch.setattr(index._client, "query", tracked_query)
    vectors = index.get_embeddings([record.uri for record in records], {"account_id": "a"})
    assert len(vectors) == 128
    assert len(calls) == 1
    assert calls[0]["timeout"] == index._timeout
    assert calls[0]["output_fields"] == ["id", "uri", "embedding"]
