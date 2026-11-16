"""Opt-in real AGFS + PostgreSQL/pgvector migration matrix; isolated databases only."""

import json
import os
import subprocess
import sys
import uuid
from dataclasses import fields

import pytest

from providers.unified_config import OgMemConfig
from service.migration.engine import MigrationService
from service.migration.storage import storage

pytestmark = pytest.mark.skipif(
    not os.environ.get("MIGRATION_TEST_ADMIN_DSN"), reason="isolated migration services required"
)


_created_databases = []


@pytest.fixture(autouse=True)
def clean_isolated_databases():
    yield
    if not _created_databases:
        return
    import psycopg2

    conn = psycopg2.connect(os.environ["MIGRATION_TEST_ADMIN_DSN"])
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for name in _created_databases:
                cur.execute(f"DROP DATABASE IF EXISTS {name}")
    finally:
        _created_databases.clear()
        conn.close()


def new_config(backend, suffix):
    import psycopg2
    from psycopg2.extensions import make_dsn

    admin = os.environ["MIGRATION_TEST_ADMIN_DSN"]
    db = "migration_" + uuid.uuid4().hex
    conn = psycopg2.connect(admin)
    conn.autocommit = True
    with conn.cursor() as cur:
        compatibility = (
            " DBCOMPATIBILITY 'PG'"
            if os.environ.get("MIGRATION_TEST_DIALECT") == "opengauss"
            else ""
        )
        cur.execute(f"CREATE DATABASE {db}{compatibility}")
    _created_databases.append(db)
    conn.close()
    dsn = make_dsn(admin, dbname=db)
    dialect = os.environ.get("MIGRATION_TEST_DIALECT", "postgresql")
    cfg = OgMemConfig(
        storage_backend=backend,
        sql_connection_string=dsn,
        sql_dialect=dialect,
        agfs_base_url=os.environ["MIGRATION_TEST_AGFS_URL"],
        agfs_mount_prefix="/local/" + db + "/" + suffix,
        provider="mock",
        embedding_provider="mock",
        opengauss_dimension=8,
        vector_db_type="opengauss",
        opengauss_connection_string=dsn,
        vector_db_dialect=dialect,
        opengauss_table_name="migration_vectors",
    )
    return cfg


def fixtures():
    from extraction.agent_memory import render_agent_memory

    base = dict(
        account_id="acme",
        owner_space="user:alice",
        context_type="MEMORY",
        category="preference",
        level=0,
        status="ACTIVE",
        version=7,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-02-01T00:00:00+00:00",
        abstract="编辑器偏好",
        overview="使用编辑器",
        content="Prefer an editor with a dark theme.",
        metadata={},
    )
    node = dict(
        base,
        uri="ctx://acme/users/alice/memories/preferences/editor",
        metadata={"provenance_ids": ["prov:1:archive:archive1:"]},
    )
    archived = dict(base, uri="ctx://acme/users/alice/memories/preferences/old", status="ARCHIVED")
    directory = dict(base, uri="ctx://acme/users/alice/memories/preferences/", content="")
    other = dict(
        base, uri="ctx://other/users/alice/memories/preferences/editor", account_id="other"
    )
    fields = {
        "tool_name": "search",
        "units": [
            {"key": "batch", "title": "Batch", "content": "Batch requests.", "grounding": []}
        ],
    }
    agent = dict(
        base,
        uri="ctx://acme/agents/main/memories/tool_units/search",
        owner_space="agent:main",
        category="tool",
        content=render_agent_memory("tool", fields),
        metadata={
            "agent_memory_version": 1,
            "agent_memory_content_version": 3,
            "routing_key": "search",
            "structured_fields": fields,
            "agent_memory": {"units": {"batch": {"revision": 2, "grounding": []}}},
        },
    )
    skill = dict(
        base,
        uri="ctx://acme/agents/main/skills/imported/guide",
        owner_space="agent:main",
        context_type="SKILL",
        category="skill",
        content="# Guide\nUse the documented API.",
        metadata={"source": "imported"},
    )
    resource = dict(
        base,
        uri="ctx://acme/resources/document1",
        owner_space="resource:acme",
        context_type="RESOURCE",
        category="document",
        content="Reference documentation.",
        metadata={"resource_type": "document"},
    )
    edge = dict(
        from_uri=node["uri"],
        to_uri=archived["uri"],
        relation_type="related_to",
        weight=0.7,
        reason="历史偏好",
    )
    archive = dict(
        account_id="acme",
        session_id="session1",
        archive_id="archive1",
        owner_user_id="alice",
        owner_agent_id="main",
        abstract="历史",
        overview="历史对话",
        messages=[{"content": "保存偏好"}],
        metadata={"archive_status": "MERGED"},
        created_at="2026-01-01T00:00:00+00:00",
    )
    return {
        "nodes": [node, archived, directory, other, agent, skill, resource],
        "archives": [archive],
        "relations": [edge],
    }


@pytest.mark.parametrize(
    "source_backend,target_backend",
    [("agfs", "agfs"), ("agfs", "sql"), ("sql", "agfs"), ("sql", "sql")],
)
@pytest.mark.parametrize(
    "vector_backend", os.environ.get("MIGRATION_TEST_VECTOR_BACKENDS", "opengauss").split(",")
)
def test_four_way_migration_and_new_process_retrieval(
    tmp_path, source_backend, target_backend, vector_backend, request
):
    source = new_config(source_backend, "source")
    target = new_config(target_backend, "target")
    target.opengauss_dimension = 16
    assert vector_backend in {"opengauss", "chroma", "milvus"}
    target.vector_db_type = vector_backend
    if vector_backend == "chroma":
        pytest.importorskip("chromadb")
        target.chroma_persist_directory = str(tmp_path / "chroma")
    elif vector_backend == "milvus":
        pymilvus = pytest.importorskip("pymilvus")
        uri = os.environ.get("OGMEM_TEST_MILVUS_URI")
        if not uri:
            pytest.skip("OGMEM_TEST_MILVUS_URI required")
        target.milvus_uri = uri
        target.milvus_token = os.environ.get("MILVUS_TOKEN", "")
        target.milvus_collection_name = "migration_vectors_" + uuid.uuid4().hex

        def clean_collection():
            client = pymilvus.MilvusClient(uri=uri, token=target.milvus_token)
            try:
                client.drop_collection(collection_name=target.milvus_collection_name)
            finally:
                client.close()

        request.addfinalizer(clean_collection)
    source_assets, target_assets = tmp_path / "source-assets", tmp_path / "target-assets"
    source_assets.mkdir()
    (source_assets / "evidence.bin").write_bytes(b"attachment-evidence")
    data = fixtures()
    data["nodes"][0]["metadata"]["source_path"] = str(source_assets / "evidence.bin")
    with storage(source) as adapter:
        adapter.check(empty=True)
        for kind, rows in data.items():
            adapter.restore(kind, rows)
    package = tmp_path / "migration.tar.gz"
    exported = MigrationService(source, asset_root=source_assets).export(package)
    preview = MigrationService(target, asset_root=target_assets).preflight(package)
    assert preview["data"] == exported["data"]
    comparison = preview["compatibility"]
    assert comparison["source"]["embedding"]["dimension"] == 8
    assert comparison["target"]["embedding"]["dimension"] == 16
    assert "embedding.dimension" in comparison["changes"]
    for label, cfg in (("source", source), ("target", target)):
        description = comparison[label]["content_storage"]
        assert description["backend"] == cfg.storage_backend
        if cfg.storage_backend == "sql":
            assert description["database_type"] == cfg.sql_dialect
            assert description["database_version"]
    restored = MigrationService(target, asset_root=target_assets).restore(package)
    assert restored["data"] == exported["data"]
    assert restored["index_records"] == 13
    # A fresh interpreter uses the application's normal storage/provider readers.
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(
        json.dumps({f.name: getattr(target, f.name) for f in fields(target) if f.init})
    )
    check = r"""
import json,sys
from providers.unified_config import OgMemConfig
from core.models import RequestContext
from service.migration.storage import storage
from fs.sql_adapter import SQLContextFS
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from pyagfs import AGFSClient
cfg=OgMemConfig(**json.load(open(sys.argv[1])))
p=cfg.to_provider_config(); index=p.create_vector_index(); embedder=p.create_embedder()
assert index.count()==13
children=index.search_children('ctx://acme/users/alice/memories/',embedder.embed_texts(['preferences'])[0],{'account_id':'acme'},20)
assert len(children)==2 and all(h.uri=='ctx://acme/users/alice/memories/preferences/' for h in children)
hits=index.search_by_vector(embedder.embed_texts(['dark editor'])[0],top_k=20,filters={'account_id':'acme'})
assert hits and all(h.uri.startswith('ctx://acme/') for h in hits)
assert all('/old' not in h.uri for h in hits)
assert not index.search_by_vector(embedder.embed_texts(['dark editor'])[0],top_k=20,filters={'account_id':'absent'})
fs=SQLContextFS(connection_string=cfg.sql_connection_string,dialect=cfg.sql_dialect) if cfg.storage_backend=='sql' else AGFSContextFS(AGFSClient(api_base_url=cfg.agfs_base_url),cfg.agfs_mount_prefix)
ctx=RequestContext(account_id='acme',user_id='alice',agent_id='main',session_id='new',trace_id='verify')
n=fs.read_node('ctx://acme/users/alice/memories/preferences/editor',ctx)
assert n.content=='Prefer an editor with a dark theme.' and n.metadata['version']==7
from pathlib import Path
assert Path(n.metadata['source_path']).read_bytes()==b'attachment-evidence'
print('NEW_PROCESS_RETRIEVAL_OK',cfg.storage_backend)
"""
    result = subprocess.run(
        [sys.executable, "-c", check, str(cfg_file)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NEW_PROCESS_RETRIEVAL_OK" in result.stdout
    with pytest.raises(ValueError, match="not empty"):
        MigrationService(target, asset_root=target_assets).preflight(package)


def test_relation_projection_rebuild(tmp_path):
    import psycopg2

    source, target = new_config("sql", "source"), new_config("sql", "target")
    target.graph_backend = "sql"
    data = fixtures()
    # The endpoints are in different owner scopes within the same account.
    data["relations"][0]["to_uri"] = data["nodes"][4]["uri"]
    with storage(source) as adapter:
        for kind, rows in data.items():
            adapter.restore(kind, rows)
    package = tmp_path / "graph.tar.gz"
    MigrationService(source).export(package)
    result = MigrationService(target).restore(package)
    assert result["index_records"] == 13
    with psycopg2.connect(target.sql_connection_string) as conn, conn.cursor() as cur:
        cur.execute("SELECT edge_projection FROM relation_edges")
        projection = cur.fetchone()[0]
        assert projection is not None
        cur.execute("SELECT count(*) FROM edge_projection_jobs WHERE status<>'COMPLETE'")
        assert cur.fetchone()[0] == 0


def test_real_admin_api_migration_flow(tmp_path):
    import time

    from server.migration_server import create_app

    source, target = new_config("agfs", "source"), new_config("sql", "target")
    source.root_api_key = target.root_api_key = "isolated-test-key"
    with storage(source) as adapter:
        for kind, rows in fixtures().items():
            adapter.restore(kind, rows)

    def client(cfg, directory):
        app = create_app(cfg, tmp_path / directory, maintenance_confirmed=True)
        cli = app.test_client()
        assert cli.get("/api/migration/status").status_code == 401
        assert (
            cli.post("/api/migration/login", headers={"X-API-Key": "isolated-test-key"}).status_code
            == 200
        )
        return cli

    def job(cli, body):
        response = cli.post("/api/migration/jobs", json={**body, "maintenance_confirmed": True})
        assert response.status_code == 202, response.get_json()
        job_id = response.get_json()["job_id"]
        for _ in range(200):
            state = cli.get("/api/migration/jobs/" + job_id).get_json()
            if state["status"] != "running":
                assert state["status"] == "succeeded", state
                return state
            time.sleep(0.05)
        pytest.fail("job timeout")

    src = client(source, "source-jobs")
    exported = job(src, {"kind": "export"})
    package = src.get(f"/api/migration/jobs/{exported['id']}/download").data
    dst = client(target, "target-jobs")
    uploaded = dst.post(
        "/api/migration/uploads", data=package, content_type="application/octet-stream"
    )
    assert uploaded.status_code == 201
    artifact = uploaded.get_json()["artifact_id"]
    preview = job(dst, {"kind": "preflight", "artifact_id": artifact})
    comparison = preview["result"]["compatibility"]
    assert comparison["source"]["content_storage"]["backend"] == "agfs"
    assert comparison["target"]["content_storage"]["backend"] == "sql"
    assert "content_storage.backend" in comparison["changes"]
    result = job(dst, {"kind": "import", "artifact_id": artifact, "preflight_id": preview["id"]})
    assert result["result"]["index_records"] == 13


@pytest.mark.skipif(
    os.environ.get("MIGRATION_TEST_DIALECT") != "opengauss",
    reason="native lexical backend requires openGauss",
)
def test_native_lexical_rebuild(tmp_path):
    from providers.lexical_index import create_lexical_index

    source, target = new_config("sql", "lex-source"), new_config("sql", "lex-target")
    target.retrieval_hybrid_enabled = True
    with storage(source) as adapter:
        for kind, rows in fixtures().items():
            adapter.restore(kind, rows)
    package = tmp_path / "hybrid.tar.gz"
    MigrationService(source).export(package)
    MigrationService(target).restore(package)
    lexical = create_lexical_index(target)
    try:
        assert lexical.ready_scopes("acme", ["user:alice", "agent:main"]) == frozenset(
            {"user:alice", "agent:main"}
        )
        hits = lexical.search(
            "editor", top_k=10, filters={"account_id": "acme", "owner_space": "user:alice"}
        )
        assert hits and all("/old" not in hit.metadata["uri"] for hit in hits)
    finally:
        lexical.close()


def test_sql_relation_authority_and_embedded_fallback(tmp_path):
    from core.models import RelationEdge, RequestContext
    from providers.relation_store.sql_relation_store import SQLRelationStore

    source, target = new_config("sql", "relations"), new_config("agfs", "relations-target")
    data = fixtures()
    with storage(source) as adapter:
        for kind, rows in data.items():
            adapter.restore(kind, rows)
    edge = {**data["relations"][0], "weight": 0.9, "reason": "updated"}
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="main", session_id="", trace_id="test"
    )
    from fs.sql_adapter.pool import SharedConnectionPool

    pool = SharedConnectionPool(
        source.sql_connection_string, dialect=source.sql_dialect, pool_size=2
    )
    store = SQLRelationStore(pool=pool)
    try:
        store.upsert_edges([RelationEdge(**edge)], ctx)
        assert store.get_edges(edge["from_uri"], ctx)[0].weight == 0.9
    finally:
        pool.close_all()
    # Keep a second source with an embedded-only edge: fallback must still work.
    fallback = {**edge, "from_uri": edge["to_uri"], "to_uri": edge["from_uri"], "weight": 0.4}
    from contextlib import closing

    import psycopg2
    from psycopg2.extras import Json

    with closing(psycopg2.connect(source.sql_connection_string)) as conn, conn.cursor() as cur:
        cur.execute("SELECT set_config('app.account_id', %s, true)", ("acme",))
        cur.execute(
            "UPDATE context_nodes SET relations=%s WHERE uri=%s",
            (Json([fallback]), fallback["from_uri"]),
        )
        conn.commit()
    package = tmp_path / "relations.tar.gz"
    MigrationService(source).export(package)
    MigrationService(target).restore(package)
    with storage(target) as adapter:
        actual = list(adapter.records("relations"))
    assert sorted(actual, key=lambda r: r["from_uri"]) == sorted(
        [edge, fallback], key=lambda r: r["from_uri"]
    )


def test_agfs_skill_candidates_block_export_and_target_preflight(tmp_path):
    from pyagfs import AGFSClient

    from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
    from service.migration.package import PackageError

    source, target = new_config("agfs", "candidate-source"), new_config("agfs", "candidate-target")
    with storage(source) as adapter:
        for kind, rows in fixtures().items():
            adapter.restore(kind, rows)
    package = tmp_path / "clean.tar.gz"
    MigrationService(source).export(package)
    for cfg in (source, target):
        store = AGFSTraceEventStore(
            client=AGFSClient(api_base_url=cfg.agfs_base_url), mount_prefix=cfg.agfs_mount_prefix
        )
        common = dict(account_id="acme", owner_space="user:alice", agent_id="main", session_id="s")
        stop = store.insert(**common, event_id="stop", hook_type="Stop", content={})
        turn = store.finalize_session(**common, stop_event_index=stop["event_index"])
        store.enqueue_skill_candidate_from_trace(
            account_id="acme",
            agent_id="main",
            session_id="s",
            turn_id=turn["turn_id"],
            trace_uri=turn["uri"],
        )
        assert store.list_pending_skill_accounts() == ["acme"]
    with pytest.raises(PackageError, match="unsupported AGFS"):
        MigrationService(source).export(tmp_path / "unsupported.tar.gz")
    assert not (tmp_path / "unsupported.tar.gz").exists()
    with pytest.raises(PackageError, match="unsupported AGFS"):
        MigrationService(target).preflight(package)


@pytest.mark.parametrize("weight", [0, 1])
@pytest.mark.parametrize("legacy_package", [False, True])
def test_integer_relation_weight_restores_to_sql(tmp_path, weight, legacy_package):
    from service.migration.package import write_package

    source, target = new_config("agfs", "integer-source"), new_config("sql", "integer-target")
    data = fixtures()
    data["relations"][0]["weight"] = weight
    package = tmp_path / "integer.tar.gz"
    if legacy_package:
        # Older packages may retain integer JSON numbers and omit descriptions.
        write_package(package, data, source_backend="agfs", source_version="legacy")
    else:
        with storage(source) as adapter:
            for kind, rows in data.items():
                adapter.restore(kind, rows)
        MigrationService(source).export(package)
    preview = MigrationService(target).preflight(package)
    result = MigrationService(target).restore(package)
    assert result["data"] == preview["data"]
    assert result["index_records"] == 13
    with storage(target) as adapter:
        edges = list(adapter.records("relations"))
    assert len(edges) == 1 and edges[0]["weight"] == float(weight)


@pytest.mark.parametrize(
    "source_backend,target_backend",
    [("agfs", "agfs"), ("agfs", "sql"), ("sql", "agfs"), ("sql", "sql")],
)
def test_completed_write_receipts_are_excluded(tmp_path, source_backend, target_backend):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from commit.write_receipts import LOCATED_OPERATION_PREFIX, _locator_uri
    from core.models import RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from fs.sql_adapter.sql_context_fs import SQLContextFS
    from service.api import MemoryWriteAPI
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = (
        new_config(source_backend, "receipt-source"),
        new_config(target_backend, "receipt-target"),
    )
    ctx = RequestContext(
        account_id="acme",
        user_id="alice",
        agent_id="main",
        session_id="receipt-session",
        trace_id="receipt-test",
    )
    from fs.sql_adapter.pool import SharedConnectionPool

    pool = (
        SharedConnectionPool(source.sql_connection_string, dialect=source.sql_dialect, pool_size=2)
        if source_backend == "sql"
        else None
    )
    fs = (
        SQLContextFS(pool=pool)
        if source_backend == "sql"
        else AGFSContextFS(AGFSClient(api_base_url=source.agfs_base_url), source.agfs_mount_prefix)
    )
    try:
        api = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
        candidate = replace(
            memory("stage_results", category="preference"),
            write_operation_id=LOCATED_OPERATION_PREFIX + "migration-test",
            provenance_ids=[],
            metadata={
                "update_activation": "preference | next | 经验索引文件",
                "update_activation_version": "c1-integrated-activation-v1",
            },
        )
        result = api.write_memory(candidate, ctx)
        assert result["target_uri"]
        locator = _locator_uri(api._writer._identity_resolver.uris, candidate, ctx)
        assert fs.read_node(locator, ctx).category == "write_operation"
        assert "memory_write_receipts" in fs.read_node(result["target_uri"], ctx).metadata
        package = tmp_path / "receipts.tar.gz"
        exported = MigrationService(source).export(package)
        imported = MigrationService(target).restore(package)
        assert exported["data"]["nodes"]["count"] == 1
        assert imported["data"] == exported["data"]
        with storage(target) as adapter:
            nodes = list(adapter.records("nodes"))
        assert len(nodes) == 1 and nodes[0]["uri"] == result["target_uri"]
        assert all("/.write_operations/" not in n["uri"] for n in nodes)
        assert "memory_write_receipts" not in nodes[0]["metadata"]
        # New process opens normal providers, then uses public business readers.
        cfg_file = tmp_path / "consumer.json"
        cfg_file.write_text(
            json.dumps({f.name: getattr(target, f.name) for f in fields(target) if f.init})
        )
        checked = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration.migration_business_consumer",
                str(cfg_file),
                result["target_uri"],
                candidate.content,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "BUSINESS_CONSUMER_OK" in checked.stdout
    finally:
        close = getattr(fs, "close", None)
        if close:
            close()
        if pool is not None:
            pool.close_all()


def test_rebuilt_vectors_are_accepted_by_graph_and_activation(tmp_path):
    from core.models import IndexRecord, RequestContext, RetrievedBlock
    from index.activation_projection import ACTIVATION_VERSION, activation_hit_is_compatible
    from retrieval.graph_supplement import GraphSupplement

    source, target = new_config("sql", "model-source"), new_config("sql", "model-target")
    data = fixtures()
    node = data["nodes"][0]
    node["metadata"].update(
        update_activation="editor preference changed", update_activation_version=ACTIVATION_VERSION
    )
    with storage(source) as adapter:
        for kind, rows in data.items():
            adapter.restore(kind, rows)
    package = tmp_path / "model.tar.gz"
    MigrationService(source).export(package)
    MigrationService(target).restore(package)
    provider = target.to_provider_config()
    index, embedder = provider.create_vector_index(), provider.create_embedder()
    try:
        model = embedder.model
        profile = dict(
            profile_id="default",
            model=model,
            dimension=target.opengauss_dimension,
            representation_version="relation-target-v2",
        )
        ctx = RequestContext(
            account_id="acme", user_id="alice", agent_id="main", session_id="new", trace_id="test"
        )
        block = RetrievedBlock(
            uri=node["uri"],
            category=node["category"],
            owner_space=node["owner_space"],
            abstract=node["abstract"],
            overview=node["overview"],
            content_excerpt=node["content"],
            metadata={
                **node["metadata"],
                "version": node["version"],
                "context_type": node["context_type"],
            },
        )
        graph = GraphSupplement(None, None, index, embedder, profile)
        graph_vectors = len(graph._vectors(node["uri"], block, ctx))
        activation = index.get_record(IndexRecord.generate_id(node["uri"], 3))
        hits = index.search_by_vector(
            embedder.embed_texts([activation.text])[0],
            filters={"account_id": "acme", "owner_space": "user:alice", "level": 3},
            top_k=5,
        )
        hit = next(h for h in hits if h.uri == activation.uri)
        activation_compatible = activation_hit_is_compatible(
            hit, block, embedding_model=model, embedding_dimension=target.opengauss_dimension
        )
        assert {"graph_vectors": graph_vectors, "activation": activation_compatible} == {
            "graph_vectors": 2,
            "activation": True,
        }
    finally:
        index.close()


@pytest.mark.parametrize("mutation", ["comment", "trailer", "crc", "concatenated", "padding"])
def test_invalid_gzip_rejected_before_target_restore(tmp_path, mutation):
    import gzip

    from service.migration.package import DATASETS, PackageError, PackageLimits, write_package

    target = new_config("sql", "invalid-envelope")
    package = tmp_path / "invalid.tar.gz"
    write_package(package, {k: [] for k in DATASETS}, source_backend="agfs", source_version="test")
    raw = package.read_bytes()
    if mutation == "comment":
        end = raw.index(b"\0", 10) + 1 if raw[3] & 8 else 10
        raw = raw[:3] + bytes([raw[3] | 16]) + raw[4:end] + b"A" * 65536 + b"\0" + raw[end:]
    elif mutation == "trailer":
        raw = raw[:-8]
    elif mutation == "crc":
        raw = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
    elif mutation == "concatenated":
        raw += gzip.compress(b"another member")
    else:
        raw = gzip.compress(gzip.decompress(raw) + b"\0" * 65536)
    package.write_bytes(raw)
    service = MigrationService(target)
    # Exercise the same reader used by restore with a small deterministic budget.
    from functools import partial
    from unittest.mock import patch

    from service.migration.package import open_package

    with patch(
        "service.migration.engine.open_package",
        partial(open_package, limits=PackageLimits(max_bytes=1024, max_files=4)),
    ):
        with pytest.raises(PackageError):
            service.restore(package)
    with storage(target) as adapter:
        adapter.check(empty=True)
        assert all(not list(adapter.records(kind)) for kind in DATASETS)
    index = target.to_provider_config().create_vector_index()
    try:
        assert index.count() == 0
    finally:
        index.close()


def test_business_writes_migrate_to_graph_candidate_selection(tmp_path):
    from dataclasses import replace

    from core.models import RelationEdge, RequestContext
    from fs.sql_adapter.pool import SharedConnectionPool
    from fs.sql_adapter.sql_context_fs import SQLContextFS
    from providers.relation_store.sql_relation_store import SQLRelationStore
    from retrieval.context_reader import ContextReader
    from retrieval.graph_supplement import GraphSupplement
    from service.api import MemoryWriteAPI
    from service.graph_service import GraphService
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = new_config("sql", "graph-source"), new_config("sql", "graph-target")
    target.graph_backend = "sql"
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="main", session_id="graph", trace_id="graph"
    )
    pool = SharedConnectionPool(
        source.sql_connection_string, dialect=source.sql_dialect, pool_size=2
    )
    try:
        writer = MemoryWriteAPI(SQLContextFS(pool=pool), NoModel(), directory_summary_enabled=False)
        uris = [
            writer.write_memory(
                replace(memory(key, category="preference", content=text), provenance_ids=[]), ctx
            )["target_uri"]
            for key, text in (
                ("editor", "Use the editor for development."),
                ("terminal", "Use a terminal to run commands."),
            )
        ]
        SQLRelationStore(pool=pool).upsert_edges(
            [
                RelationEdge(
                    from_uri=uris[0],
                    to_uri=uris[1],
                    relation_type="related_to",
                    weight=1.0,
                    reason="Development tools",
                )
            ],
            ctx,
        )
    finally:
        pool.close_all()
    package = tmp_path / "graph-business.tar.gz"
    MigrationService(source).export(package)
    MigrationService(target).restore(package)
    pool = SharedConnectionPool(
        target.sql_connection_string, dialect=target.sql_dialect, pool_size=2
    )
    provider = target.to_provider_config()
    index, embedder = provider.create_vector_index(), provider.create_embedder()
    try:
        reader = ContextReader(SQLContextFS(pool=pool), storage_backend="sql")
        profile = dict(
            profile_id=target.graph_edge_embedding_profile,
            model=embedder.model,
            dimension=target.opengauss_dimension,
            representation_version=target.graph_projection_representation,
        )
        graph = GraphService(None, relation_reader=SQLRelationStore(pool=pool))
        consumer = GraphSupplement(graph, reader, index, embedder, profile)
        anchor = reader.read_authorized(uris[1], ctx)
        selected = consumer.select("Use the editor for development.", [anchor], [anchor], ctx)
        candidate = next(item for item in selected["selected"] if item["uri"] == uris[0])
        assert candidate["incoming"]
        assert candidate["visited_endpoint_similarity"] == pytest.approx(1.0)
        assert candidate["accepted"]
        from server.memory_service import MemoryService

        normal_service = MemoryService(target)
        try:
            baseline = [
                dict(
                    uri=anchor.uri,
                    abstract=anchor.abstract,
                    overview=anchor.overview,
                    content=anchor.content_excerpt,
                    category=anchor.category,
                )
            ]
            stats = {"graph_expansion": {"two_hop_supplement": selected}}
            query = "Use the editor for development."
            budget = {"working_set": 20000}

            def formatter(items, allocation):
                return normal_service._format_working_set(items, allocation, query=query)

            delivered = consumer.deliver(baseline, query, stats, ctx, budget, formatter)
            assert [item["uri"] for item in delivered] == [uris[1], uris[0]]
            assert formatter(delivered, budget).startswith(formatter(baseline, budget))
            assert "Use the editor for development." in formatter(delivered, budget)
        finally:
            normal_service.shutdown()
    finally:
        index.close()
        pool.close_all()


def test_trace_produced_agent_memory_rejects_unsupported_evidence(tmp_path, request):
    from pyagfs import AGFSClient

    from commit.sql_outbox_store import SQLOutboxStore
    from core.models import RequestContext
    from fs.agfs_adapter.agfs_trace_store import AGFSTraceEventStore
    from fs.sql_adapter.pool import SharedConnectionPool
    from fs.sql_adapter.sql_context_fs import SQLContextFS
    from server.agent_memory import process_trace_memory, trace_memory_event
    from server.mcp_trace import message_events
    from service.migration.package import PackageError
    from tests.unit.server.test_agent_memory import MESSAGES, ExtractionLLM

    source = new_config("sql", "trace-source")
    client = AGFSClient(api_base_url=source.agfs_base_url)
    pool = SharedConnectionPool(
        source.sql_connection_string, dialect=source.sql_dialect, pool_size=2
    )
    request.addfinalizer(pool.close_all)
    fs = SQLContextFS(pool=pool)
    traces = AGFSTraceEventStore(client=client, mount_prefix=source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "trace-session", "trace-request")
    for i, event in enumerate(
        message_events(MESSAGES) + [{"event_type": "turn_end", "status": "ok"}]
    ):
        inserted = traces.insert(
            event_id=str(i),
            account_id=ctx.account_id,
            owner_space=ctx.user_space_name(),
            agent_id=ctx.agent_id,
            session_id=ctx.session_id,
            hook_type=event["event_type"],
            content=event,
        )
    finalized = traces.finalize_session(
        account_id=ctx.account_id,
        owner_space=ctx.user_space_name(),
        agent_id=ctx.agent_id,
        session_id=ctx.session_id,
        stop_event_index=inserted["event_index"],
    )
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    event = trace_memory_event(finalized["uri"], ctx)
    outbox.register_once(event, ctx)
    process_trace_memory(event, trace_store=traces, outbox=outbox, fs=fs, llm=ExtractionLLM())
    outbox.mark_done(event, event.uri)
    with storage(source) as adapter:
        nodes = list(adapter.records("nodes"))
    assert nodes and all(finalized["uri"] in n["metadata"]["provenance_ids"] for n in nodes)
    with pytest.raises(PackageError, match="trace/tool evidence"):
        MigrationService(source).export(tmp_path / "trace.tar.gz")
    assert not (tmp_path / "trace.tar.gz").exists()


@pytest.mark.parametrize(
    "source_backend,target_backend",
    [
        ("agfs", "agfs"),
        ("agfs", "sql"),
        ("sql", "agfs"),
        ("sql", "sql"),
    ],
)
def test_archive_producer_to_provenance_consumer(tmp_path, source_backend, target_backend):
    from contextlib import ExitStack
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import RequestContext
    from core.provenance_resolver import ProvenanceResolver
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from fs.sql_adapter.pool import SharedConnectionPool
    from fs.sql_adapter.sql_context_fs import SQLContextFS
    from service.api import MemoryWriteAPI
    from session.archive_store import SessionArchiveStore
    from session.sql_archive_store import SQLSessionArchiveStore
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = (
        new_config(source_backend, "archive-source"),
        new_config(target_backend, "archive-target"),
    )
    ctx = RequestContext("acme", "alice", "main", "archive-session", "archive-test")
    messages = [{"id": "message-1", "role": "user", "content": "请记住：我偏好深色主题 🌙"}]
    with ExitStack() as cleanup:

        def open_readers(cfg):
            if cfg.storage_backend == "sql":
                pool = SharedConnectionPool(
                    cfg.sql_connection_string, dialect=cfg.sql_dialect, pool_size=2
                )
                cleanup.callback(pool.close_all)
                fs = SQLContextFS(pool=pool)
                return fs, SQLSessionArchiveStore(pool=pool)
            fs = AGFSContextFS(AGFSClient(api_base_url=cfg.agfs_base_url), cfg.agfs_mount_prefix)
            return fs, SessionArchiveStore(fs)

        source_fs, source_archives = open_readers(source)
        written = source_archives.write_archive(
            ctx.session_id, "主题偏好", "深色主题", messages, ctx
        )
        assert written.success, written.error
        provenance = ProvenanceResolver.build_id("archive", written.archive_id, ["message-1"])
        candidate = replace(
            memory("theme", category="preference", content="我偏好深色主题 🌙"),
            provenance_ids=[provenance],
        )
        uri = MemoryWriteAPI(source_fs, NoModel(), directory_summary_enabled=False).write_memory(
            candidate, ctx
        )["target_uri"]
        package = tmp_path / "archive.tar.gz"
        MigrationService(source).export(package)
        MigrationService(target).restore(package)
        target_fs, archives = open_readers(target)
        restored = target_fs.read_node(uri, ctx)
        parsed = ProvenanceResolver.parse_id(restored.metadata["provenance_ids"][0])
        entry = archives.read_archive_by_id(parsed["source_id"], ctx)
        assert entry is not None
        assert entry.messages == messages and entry.abstract == "深色主题"
        assert [m for m in entry.messages if m["id"] in parsed["detail"]] == messages
        assert archives.read_archive_by_id(written.archive_id, replace(ctx, user_id="bob")) is None
        assert (
            archives.read_archive_by_id(written.archive_id, replace(ctx, account_id="other"))
            is None
        )


@pytest.mark.parametrize("archive_endpoint", ["from_uri", "to_uri"])
def test_archive_relation_requires_node_endpoints_only_for_sql_graph(tmp_path, archive_endpoint):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI
    from service.migration.package import PackageError
    from session.archive_store import SessionArchiveStore
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = (
        new_config("agfs", "archive-edge-source"),
        new_config("sql", "archive-edge-target"),
    )
    target.graph_backend = "sql"
    client = AGFSClient(api_base_url=source.agfs_base_url)
    fs = AGFSContextFS(client, source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "edge-session", "edge-test")
    archived = SessionArchiveStore(fs).write_archive(
        ctx.session_id, "history", "history", [{"role": "user", "content": "theme"}], ctx
    )
    assert archived.success, archived.error
    uri = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(
        replace(memory("theme", category="preference"), provenance_ids=[]), ctx
    )["target_uri"]
    edge = dict(
        from_uri=uri,
        to_uri=uri,
        relation_type="derived_from",
        weight=1.0,
        reason="history evidence",
    )
    edge[archive_endpoint] = archived.uri
    AGFSRelationStore(client, source.agfs_mount_prefix).upsert_edges([RelationEdge(**edge)], ctx)
    package = tmp_path / "archive-edge.tar.gz"
    MigrationService(source).export(package)
    importer = MigrationService(target)
    for operation in (importer.preflight, importer.restore):
        with pytest.raises(PackageError, match="SQL graph projection requires node endpoints"):
            operation(package)
        with storage(target) as adapter:
            adapter.check(empty=True)
            assert all(
                not list(adapter.records(kind)) for kind in ("nodes", "archives", "relations")
            )
        index = target.to_provider_config().create_vector_index()
        try:
            assert index.count() == 0
        finally:
            index.close()
    target.graph_backend = "none"
    importer.preflight(package)
    importer.restore(package)
    with storage(target) as adapter:
        assert len(list(adapter.records("nodes"))) == 1
        assert len(list(adapter.records("archives"))) == 1
        assert list(adapter.records("relations")) == [edge]


@pytest.mark.parametrize("source_backend", ["agfs", "sql"])
@pytest.mark.parametrize("identity_case", ["account", "user", "agent"])
def test_sessions_identifier_survives_sql_migration(tmp_path, source_backend, identity_case):
    from contextlib import ExitStack

    from pyagfs import AGFSClient

    from core.models import ContextNode, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from fs.sql_adapter.pool import SharedConnectionPool
    from fs.sql_adapter.sql_context_fs import SQLContextFS

    source, target = new_config(source_backend, "id-source"), new_config("sql", "id-target")
    ctx = RequestContext(
        "sessions" if identity_case == "account" else "acme",
        "sessions" if identity_case == "user" else "alice",
        "sessions" if identity_case == "agent" else "main",
        "s",
        "test",
    )
    owner_type, owner = (
        ("agents", ctx.agent_id) if identity_case == "agent" else ("users", ctx.user_id)
    )
    uri = f"ctx://{ctx.account_id}/{owner_type}/{owner}/memories/preferences/editor"
    with ExitStack() as cleanup:
        if source_backend == "sql":
            pool = SharedConnectionPool(
                source.sql_connection_string, dialect=source.sql_dialect, pool_size=2
            )
            cleanup.callback(pool.close_all)
            fs = SQLContextFS(pool=pool)
        else:
            fs = AGFSContextFS(
                AGFSClient(api_base_url=source.agfs_base_url), source.agfs_mount_prefix
            )
        fs.write_node(
            ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="preference",
                level=0,
                owner_space=f"{owner_type[:-1]}:{owner}",
                abstract="editor",
                overview="editor",
                content="Use a dark editor.",
                metadata={},
            ),
            ctx,
        )
        package = tmp_path / "sessions-id.tar.gz"
        exported = MigrationService(source).export(package)
        assert exported["data"]["nodes"]["count"] == 1
        imported = MigrationService(target).restore(package)
        assert imported["data"] == exported["data"]
        pool = SharedConnectionPool(
            target.sql_connection_string, dialect=target.sql_dialect, pool_size=2
        )
        cleanup.callback(pool.close_all)
        assert SQLContextFS(pool=pool).read_node(uri, ctx).content == "Use a dark editor."


@pytest.mark.parametrize("graph", ["none", "sql"])
@pytest.mark.parametrize("alias", [True, False])
def test_directory_relation_uri_matches_sql_identity(tmp_path, graph, alias):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import ContextNode, RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI
    from service.migration.package import PackageError
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = new_config("agfs", "directory-source"), new_config("sql", "directory-target")
    target.graph_backend = graph
    client = AGFSClient(api_base_url=source.agfs_base_url)
    fs = AGFSContextFS(client, source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "s", "test")
    uri = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(
        replace(memory("editor", category="preference"), provenance_ids=[]), ctx
    )["target_uri"]
    directory = uri.rsplit("/", 1)[0] + "/"
    fs.write_node(
        ContextNode(
            uri=directory,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="preferences",
            overview="preferences",
            content="",
            metadata={},
        ),
        ctx,
    )
    edge = RelationEdge(
        from_uri=uri,
        to_uri=directory.rstrip("/") if alias else directory,
        relation_type="related_to",
        weight=1,
        reason="directory",
    )
    AGFSRelationStore(client, source.agfs_mount_prefix).upsert_edges([edge], ctx)
    package = tmp_path / "directory.tar.gz"
    MigrationService(source).export(package)
    importer = MigrationService(target)
    if alias:
        for operation in (importer.preflight, importer.restore):
            with pytest.raises(PackageError, match="exactly match stored URI"):
                operation(package)
        with storage(target) as adapter:
            adapter.check(empty=True)
        index = target.to_provider_config().create_vector_index()
        try:
            assert index.count() == 0
        finally:
            index.close()
    else:
        importer.preflight(package)
        importer.restore(package)
        with storage(target) as adapter:
            rows = list(adapter.records("relations"))
            nodes = {r["uri"] for r in adapter.records("nodes")}
        assert rows and all(r["from_uri"] in nodes and r["to_uri"] in nodes for r in rows)


def test_agfs_relation_group_uses_one_remote_read_and_write():
    from unittest.mock import patch

    from pyagfs import AGFSClient

    from core.models import ContextNode, RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.migration.storage import AGFSStorage

    cfg = new_config("agfs", "edge-batch")
    client = AGFSClient(api_base_url=cfg.agfs_base_url)
    ctx = RequestContext("acme", "alice", "main", "s", "test")
    uri = "ctx://acme/users/alice/memories/preferences/editor"
    AGFSContextFS(client, cfg.agfs_mount_prefix).write_node(
        ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="editor",
            overview="editor",
            content="editor",
            metadata={},
        ),
        ctx,
    )
    edges = [
        dict(from_uri=uri, to_uri=uri, relation_type=f"related_{i}", weight=1.0, reason="edge")
        for i in range(16)
    ]
    with (
        patch.object(client, "read", wraps=client.read) as read,
        patch.object(client, "write", wraps=client.write) as write,
    ):
        AGFSStorage(cfg, client).restore("relations", iter(edges))
        assert read.call_count == write.call_count == 1
    assert AGFSRelationStore(client, cfg.agfs_mount_prefix).get_edges(uri, ctx) == [
        RelationEdge(**e) for e in edges
    ]


@pytest.mark.parametrize("failure", ["refused", "authentication"])
def test_migration_job_logs_real_database_failure(tmp_path, caplog, failure):
    import logging
    import socket
    import time

    from psycopg2.extensions import make_dsn, parse_dsn

    from server.migration_server import create_app

    cfg = new_config("sql", "diagnostic-source")
    cfg.root_api_key = "diagnostic-admin-key"
    original_dsn = cfg.sql_connection_string
    app = create_app(cfg, tmp_path, maintenance_confirmed=True)
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": cfg.root_api_key})

    def export_job():
        response = client.post(
            "/api/migration/jobs", json={"kind": "export", "maintenance_confirmed": True}
        )
        assert response.status_code == 202
        job_id = response.get_json()["job_id"]
        for _ in range(200):
            job = client.get("/api/migration/jobs/" + job_id).get_json()
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        pytest.fail("migration job did not finish")

    assert export_job()["status"] == "succeeded"
    caplog.clear()
    caplog.set_level(logging.ERROR, logger="server.migration_server")
    reserved = socket.socket()
    try:
        if failure == "refused":
            # Reserve a non-listening port: real libpq connection refusal,
            # without stopping or changing the database used by other tests.
            reserved.bind(("127.0.0.1", 0))
            cfg.sql_connection_string = make_dsn(
                original_dsn, port=reserved.getsockname()[1], connect_timeout=1
            )
        else:
            cfg.sql_connection_string = make_dsn(
                original_dsn, password="deliberately-wrong-password", connect_timeout=2
            )
        job = export_job()
        assert job["status"] == "failed"
        assert (
            job["error"]
            == "OperationalError: operation failed; inspect isolated target before retrying"
        )
        records = [r for r in caplog.records if r.name == "server.migration_server"]
        assert len(records) == 1 and records[0].exc_info is None
        text = records[0].getMessage()
        assert f"job_id={job['id']} kind=export phase=validating" in text
        assert "service.migration.storage" in text and "OperationalError" in text
        if failure == "refused":
            assert "connection refused" in text.lower()
        else:
            assert any(word in text.lower() for word in ("password", "authentication", "login"))
        for secret in (
            parse_dsn(original_dsn)["password"],
            "deliberately-wrong-password",
            cfg.root_api_key,
            cfg.sql_connection_string,
        ):
            assert secret not in text and secret not in str(job)
    finally:
        reserved.close()
        cfg.sql_connection_string = original_dsn
    assert export_job()["status"] == "succeeded"
