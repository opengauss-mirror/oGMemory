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

    def create(dimension=3):
        cfg.opengauss_dimension = dimension
        index = cfg.to_provider_config().create_vector_index()
        opened.append(index)
        return index

    try:
        yield create
    finally:
        if backend == "milvus" and opened:
            opened[0]._client.drop_collection(collection_name=cfg.milvus_collection_name)
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


def test_shared_retained_schema_dimension_contract(vector_target):
    index = vector_target()
    index.upsert([record()])
    index.delete(["contract"])
    assert index.count() == 0
    with pytest.raises(ValueError, match="dimension|schema"):
        incompatible = vector_target(4)
        incompatible.check_migration_target()
