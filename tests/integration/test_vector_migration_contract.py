"""Shared migration admission contract, exercised against actual vector stores."""

import os
import uuid
from contextlib import closing

import pytest

from core.models import IndexRecord
from providers.unified_config import OgMemConfig


@pytest.fixture(
    params=os.environ.get("MIGRATION_CONTRACT_BACKENDS", "opengauss,chroma,milvus").split(",")
)
def vector_target(request, tmp_path):
    backend = request.param
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type=backend,
        opengauss_dimension=3,
        chroma_persist_directory=str(tmp_path / "chroma"),
    )
    database = None
    if backend == "opengauss":
        admin = os.environ.get("MIGRATION_TEST_ADMIN_DSN")
        if not admin:
            pytest.skip("MIGRATION_TEST_ADMIN_DSN required")
        import psycopg2
        from psycopg2.extensions import make_dsn

        database = "migration_contract_" + uuid.uuid4().hex
        with closing(psycopg2.connect(admin)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                suffix = (
                    " DBCOMPATIBILITY 'PG'"
                    if os.environ.get("MIGRATION_TEST_DIALECT") == "opengauss"
                    else ""
                )
                cur.execute(f"CREATE DATABASE {database}{suffix}")
        cfg.opengauss_connection_string = make_dsn(admin, dbname=database)
        cfg.vector_db_dialect = os.environ.get("MIGRATION_TEST_DIALECT", "postgresql")
    elif backend == "chroma":
        pytest.importorskip("chromadb")
    elif backend == "milvus":
        pytest.importorskip("pymilvus")
        if not os.environ.get("OGMEM_TEST_MILVUS_URI"):
            pytest.skip("OGMEM_TEST_MILVUS_URI required")
        cfg.milvus_uri = os.environ["OGMEM_TEST_MILVUS_URI"]
        cfg.milvus_token = os.environ.get("MILVUS_TOKEN", "")
        cfg.milvus_collection_name = "migration_contract_" + uuid.uuid4().hex
    else:
        pytest.fail("unknown contract backend")
    opened = []

    def create(dimension=3, *, shadow=False):
        cfg.opengauss_dimension = dimension
        provider = cfg.to_provider_config()
        if shadow:
            name = "shadow_" + uuid.uuid4().hex
            provider.opengauss_table_name = name
            provider.chroma_collection_name = name
            provider.milvus_collection_name = name
        index = provider.create_vector_index()
        opened.append(index)
        return index

    try:
        yield create
    finally:
        if backend == "milvus" and opened:
            for name in {index._collection_name for index in opened}:
                opened[0]._client.drop_collection(collection_name=name)
        for index in opened:
            close = getattr(index, "close", None)
            if close:
                close()
        if database:
            with closing(psycopg2.connect(admin)) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE {database}")


def record():
    return IndexRecord(
        "contract",
        "ctx://acme/resources/document",
        0,
        "full index text " * 40,
        {
            "account_id": "acme",
            "owner_space": "resource:acme",
            "category": "document",
            "context_type": "RESOURCE",
            "resource_type": "code_chunk",
        },
        {
            "_embedding": [1.0, 0.0, 0.0],
            "abstract": "summary",
            "parent_uri": "ctx://acme/resources/",
            "canonical_revision": 2,
            "grounding": [{"uri": "ctx://acme/resources/evidence"}],
        },
    )


def test_shared_record_and_empty_target_contract(vector_target):
    index = vector_target()
    index.check_migration_target()
    original = record()
    index.upsert([original])
    restored = index.get_record(original.id)
    assert (restored.id, restored.uri, restored.level, restored.text) == (
        original.id,
        original.uri,
        original.level,
        original.text,
    )
    assert all(restored.filters.get(k) == v for k, v in original.filters.items())
    for key in ("abstract", "parent_uri", "canonical_revision", "grounding"):
        assert restored.metadata[key] == original.metadata[key]
    assert index.search_by_vector([1.0, 0.0, 0.0], original.filters, 10)
    assert not index.search_by_vector(
        [1.0, 0.0, 0.0], {**original.filters, "account_id": "other"}, 10
    )
    with pytest.raises(ValueError, match="not empty"):
        index.check_migration_target()


def test_space_projection_filters_round_trip_and_query(vector_target):
    index = vector_target()
    original = record()
    original.filters.update(space_key='team', authorization_resource_id=original.uri,
                            content_rev=7, state='ACTIVE')
    original.metadata['content_rev'] = 1  # The filter is authoritative.
    index.upsert([original])
    actual = index.get_record(original.id)
    assert all(actual.filters.get(key) == value for key, value in original.filters.items())
    assert index.search_by_vector([1.0, 0.0, 0.0], original.filters, 10)
    for key, value in {'space_key': 'other', 'authorization_resource_id': 'other',
                       'content_rev': 8, 'state': 'ARCHIVED'}.items():
        assert not index.search_by_vector([1.0, 0.0, 0.0], {**original.filters, key: value}, 10)


def test_shared_retained_schema_dimension_contract(vector_target):
    index = vector_target()
    index.upsert([record()])
    index.delete(["contract"])
    assert index.count() == 0
    with pytest.raises(ValueError, match="dimension|schema"):
        incompatible = vector_target(4)
        incompatible.check_migration_target()


def test_shadow_scan_preserves_live_records_and_embeddings(vector_target):
    from dataclasses import replace

    index = vector_target()
    original = record()
    items = [replace(original, id=f"copy-{i}", text=f"text {i}") for i in range(5)]
    index.upsert(items)
    index.upsert([replace(items[1], text="updated", metadata={**items[1].metadata, "_embedding": [0.0, 1.0, 0.0]})])
    index.delete([items[3].id])
    scanned = list(index.iter_records(batch_size=2))
    assert len(scanned) == 4
    assert {r.id for r in scanned} == {items[i].id for i in (0, 1, 2, 4)}
    for item in scanned:
        expected = index.get_record(item.id)
        assert item == expected
        assert item.metadata["_embedding"] == ([0.0, 1.0, 0.0] if item.id == "copy-1" else [1.0, 0.0, 0.0])
    shadow = vector_target(shadow=True)
    shadow.check_migration_target()
    shadow.upsert(scanned)
    copied = {item.id: item for item in shadow.iter_records(batch_size=1)}
    assert copied == {item.id: item for item in scanned}
    shadow.delete(["copy-1"])
    assert index.get_record("copy-1") is not None
    assert shadow.get_record("copy-1") is None
    assert shadow.search_by_vector([1.0, 0.0, 0.0], original.filters, 10)
    with pytest.raises(ValueError, match="batch_size"):
        list(index.iter_records(batch_size=0))
