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
