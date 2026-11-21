"""Opt-in real AGFS + PostgreSQL/pgvector migration matrix; isolated databases only."""

import json
import os
import subprocess
import sys
import uuid
from contextlib import ExitStack, closing
from dataclasses import fields

import pytest

from providers.unified_config import OgMemConfig
from service.migration.storage import storage
from tests.migration_support import MigrationHarness as MigrationService

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
    if backend == "agfs":
        with storage(cfg) as adapter:
            adapter._mkdir(cfg.agfs_mount_prefix)
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
    directory = dict(base, uri="ctx://acme/users/alice/memories/preferences/", content="",
                     metadata={"custom_directory": "preserved"})
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
    [("agfs", "agfs"), ("sql", "sql")],
)
@pytest.mark.parametrize(
    "vector_backend", os.environ.get("MIGRATION_TEST_VECTOR_BACKENDS", "opengauss").split(",")
)
def test_same_storage_migration_and_new_process_retrieval(
    tmp_path, source_backend, target_backend, vector_backend, request
):
    source = new_config(source_backend, "source")
    target = new_config(target_backend, "target")
    if target_backend == "sql":
        target.sql_dialect = "auto"
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
    target_assets.mkdir()
    (source_assets / "evidence.bin").write_bytes(b"attachment-evidence")
    data = fixtures()
    data["nodes"][0]["metadata"]["source_path"] = str(source_assets / "evidence.bin")
    with storage(source) as adapter:
        adapter.check(empty=True)
        for kind, rows in data.items():
            adapter.restore(kind, rows)
    package = tmp_path / "migration.tar.gz"
    exported = MigrationService(source, asset_root=source_assets).export(package)
    preview = MigrationService(target, asset_root=target_assets).preview(package)
    assert preview["data"] == exported["data"]
    comparison = preview["compatibility"]
    assert comparison["source"]["embedding"]["dimension"] == 8
    assert comparison["target"]["embedding"]["dimension"] == 16
    assert "embedding.dimension" in comparison["changes"]
    for label, cfg in (("source", source), ("target", target)):
        description = comparison[label]["content_storage"]
        assert description["backend"] == cfg.storage_backend
        if cfg.storage_backend == "sql":
            assert description["database_type"] == os.environ.get("MIGRATION_TEST_DIALECT", "postgresql")
            assert description["database_version"]
    restored = MigrationService(target, asset_root=target_assets).import_package(package)
    assert_merged_package(package, target, restored, exported["data"], target_assets)
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
    repeated = MigrationService(target, asset_root=target_assets).preview(package)
    assert repeated["counts"]["new"] == 0


def assert_merged_package(path, target, result, expected, asset_root=None):
    from service.migration.assets import Assets
    from service.migration.package import open_package
    assert result['data']['archives'] == expected['archives']
    assert result['data']['relations'] == expected['relations']
    assert result['data']['nodes']['count'] == expected['nodes']['count']
    with open_package(path) as package, storage(target) as adapter:
        assets = Assets(asset_root, package.root)
        actual = {r['uri']: assets.capture(r, copy_bytes=False) for r in adapter.records('nodes')}
        for original in package.records('nodes'):
            row = actual[original['uri']]
            if original['uri'].endswith('/'):
                # The merge rebuilds summaries from children instead of retaining
                # the source's stale directory summaries and timestamp/version.
                children = [child for child in actual.values()
                            if child['uri'].rstrip('/').rsplit('/', 1)[0] + '/' == row['uri']
                            and child['status'] == 'ACTIVE' and not child['metadata'].get('indexing_disabled')]
                assert row['metadata']['child_count'] == len(children)
                assert row['metadata']['categories'] == sorted({child['category'] for child in children})
                assert row['metadata']['node_type'] == 'directory'
                for key, value in original['metadata'].items():
                    if key not in {'node_type', 'child_count', 'categories'}:
                        assert row['metadata'][key] == value
                assert row['version'] == original['version'] + 1
                assert row['updated_at'] >= original['updated_at']
                assert row['abstract']
                for key in ('uri', 'account_id', 'owner_space', 'category', 'context_type', 'status', 'level', 'created_at', 'content'):
                    assert row[key] == original[key]
            else:
                assert row == original


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
    result = MigrationService(target).import_package(package)
    assert result["index_records"] == 13
    with closing(psycopg2.connect(target.sql_connection_string)) as conn, conn, conn.cursor() as cur:
        cur.execute("SELECT edge_projection FROM relation_edges")
        projection = cur.fetchone()[0]
        assert projection is not None
        cur.execute("SELECT count(*) FROM edge_projection_jobs WHERE status<>'COMPLETE'")
        assert cur.fetchone()[0] == 0


def test_real_admin_api_migration_flow(tmp_path, monkeypatch, request):
    import time

    from flask import Flask

    from server.memory_service import MemoryService
    from server.migration_api import create_migration_blueprint
    from server.service_factory import close_service, create_service

    source, target = new_config("agfs", "source"), new_config("agfs", "target")
    source.root_api_key = target.root_api_key = "isolated-test-key"
    with storage(source) as adapter:
        for kind, rows in fixtures().items():
            adapter.restore(kind, rows)

    lifecycle = ExitStack()
    request.addfinalizer(lifecycle.close)
    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    headers = {'X-API-Key': 'isolated-test-key'}
    def client(cfg, directory):
        cfg.role_control_enabled = True
        cfg.migration_workspace = str(tmp_path / directory)
        service = create_service(cfg)
        lifecycle.callback(close_service, service)
        app = Flask(__name__)
        app.register_blueprint(create_migration_blueprint(lambda: service))
        cli = app.test_client()
        assert cli.get('/api/v1/migration/status').status_code == 401
        return cli

    def job(cli, body):
        response = cli.post("/api/v1/migration/jobs", headers=headers, json={**body, "request_id": str(uuid.uuid4()), "maintenance_confirmed": True})
        assert response.status_code == 202, response.get_json()
        job_id = response.get_json()["job_id"]
        for _ in range(200):
            state = cli.get("/api/v1/migration/jobs/" + job_id, headers=headers).get_json()
            if state["status"] != "running":
                assert state["status"] == "succeeded", state
                return state
            time.sleep(0.05)
        pytest.fail("job timeout")

    src = client(source, "source-jobs")
    exported = job(src, {"kind": "export"})
    package = src.get(f"/api/v1/migration/jobs/{exported['id']}/download", headers=headers).data
    dst = client(target, "target-jobs")
    uploaded = dst.post(
        "/api/v1/migration/uploads", data=package, content_type="application/octet-stream", headers=headers
    )
    assert uploaded.status_code == 201
    artifact = uploaded.get_json()["artifact_id"]
    preview = job(dst, {"kind": "preflight", "artifact_id": artifact})
    comparison = preview["result"]["compatibility"]
    assert comparison["source"]["content_storage"]["backend"] == "agfs"
    assert comparison["target"]["content_storage"]["backend"] == target.storage_backend
    assert ("content_storage.backend" in comparison["changes"]) == (source.storage_backend != target.storage_backend)
    result = job(dst, {"kind": "import", "artifact_id": artifact, "preflight_id": preview["id"]})
    # Normal source service may refresh directory summaries before snapshot.
    # Validate the exported business records and actual target projection.
    assert result['result']['data']['nodes']['count'] == preview['result']['data']['nodes']['count']
    index = target.to_provider_config().create_vector_index()
    try:
        assert index.count() == result['result']['index_records'] > 0
    finally:
        index.close()


@pytest.mark.skipif(
    os.environ.get("MIGRATION_TEST_DIALECT") != "opengauss",
    reason="native lexical backend requires openGauss",
)
def test_native_lexical_rebuild(tmp_path):
    from providers.lexical_index import create_lexical_index

    source, target = new_config("sql", "lex-source"), new_config("sql", "lex-target")
    target.retrieval_hybrid_enabled = True
    target.retrieval_lexical_provider = "opengauss_native"
    with storage(source) as adapter:
        for kind, rows in fixtures().items():
            adapter.restore(kind, rows)
    package = tmp_path / "hybrid.tar.gz"
    MigrationService(source).export(package)
    MigrationService(target).import_package(package)
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

    source, target = new_config("sql", "relations"), new_config("sql", "relations-target")
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
    MigrationService(target).import_package(package)
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
        MigrationService(target).preview(package)




@pytest.mark.parametrize(
    "source_backend,target_backend",
    [("agfs", "agfs"), ("sql", "sql")],
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
        imported = MigrationService(target).import_package(package)
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
    MigrationService(target).import_package(package)
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
            service.import_package(package)
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
    MigrationService(target).import_package(package)
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
        MigrationService(target).import_package(package)
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




@pytest.mark.parametrize("source_backend", ["sql"])
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
        imported = MigrationService(target).import_package(package)
        assert imported["data"] == exported["data"]
        pool = SharedConnectionPool(
            target.sql_connection_string, dialect=target.sql_dialect, pool_size=2
        )
        cleanup.callback(pool.close_all)
        assert SQLContextFS(pool=pool).read_node(uri, ctx).content == "Use a dark editor."




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
    import socket
    import time
    from types import SimpleNamespace

    from psycopg2.extensions import make_dsn, parse_dsn

    from service.migration.jobs import InstanceJobs
    from service.migration.maintenance import MaintenanceStore

    cfg = new_config("sql", "diagnostic-source")
    cfg.root_api_key = "diagnostic-admin-key"
    original_dsn = cfg.sql_connection_string
    engine = MigrationService(cfg)
    migration = SimpleNamespace(cfg=cfg, workspace=tmp_path,
        gate=MaintenanceStore(tmp_path / 'gate.db', 'diagnostic'),
        export=lambda destination, task, progress: engine.export(destination, progress))
    queue = InstanceJobs(migration)

    def export_job():
        job_id = queue.submit('export', identity=str(uuid.uuid4()))['id']
        for _ in range(200):
            job = queue.get(job_id)
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        pytest.fail("migration job did not finish")

    assert export_job()["status"] == "succeeded"
    caplog.clear()
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
        text = job["error"]
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


@pytest.mark.parametrize("target_backend", ["agfs"])
def test_agfs_directory_source_alias_preserves_backend_semantics(tmp_path, target_backend):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import ContextNode, RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI
    from service.migration.package import PackageError
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = new_config("agfs", "source-alias"), new_config(target_backend, "alias-target")
    target.graph_backend = "none"
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
        from_uri=directory.rstrip("/"),
        to_uri=uri,
        relation_type="related_to",
        weight=1,
        reason="directory",
    )
    relations = AGFSRelationStore(client, source.agfs_mount_prefix)
    relations.upsert_edges([edge], ctx)
    assert relations.get_edges(directory, ctx) == [edge]
    package = tmp_path / "source-alias.tar.gz"
    MigrationService(source).export(package)
    importer = MigrationService(target)
    if target_backend == "agfs":
        importer.preview(package)
        importer.import_package(package)
        target_client = AGFSClient(api_base_url=target.agfs_base_url)
        assert AGFSRelationStore(target_client, target.agfs_mount_prefix).get_edges(
            directory, ctx
        ) == [RelationEdge(from_uri=directory, to_uri=edge.to_uri, relation_type=edge.relation_type,
                           weight=edge.weight, reason=edge.reason)]
    else:
        for operation in (importer.preview, importer.import_package):
            with pytest.raises(PackageError, match="exactly match stored URI"):
                operation(package)
        with storage(target) as adapter:
            adapter.check(empty=True)
        index = target.to_provider_config().create_vector_index()
        try:
            assert index.count() == 0
        finally:
            index.close()


@pytest.mark.parametrize("tail", ["buffered", "appended", "second_tar"])
def test_tar_tail_rejected_before_business_restore(tmp_path, tail):
    import gzip
    import io
    import tarfile

    from service.migration.package import DATASETS, PackageError, write_package

    target = new_config("sql", "tar-tail-target")
    package = tmp_path / "tail.tar.gz"
    record = {**fixtures()["nodes"][0], "metadata": {}}
    write_package(
        package,
        {"nodes": [record], "archives": [], "relations": []},
        source_backend="agfs",
        source_version="test",
    )
    raw = gzip.decompress(package.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        end = max(m.offset_data + ((m.size + 511) // 512) * 512 for m in archive)
    if tail == "buffered":
        assert end + 1024 < tarfile.RECORDSIZE
        raw = raw[: end + 1024] + b"hidden" + raw[end + 1030 :]
    elif tail == "appended":
        raw += b"hidden"
    else:
        raw += raw
    package.write_bytes(gzip.compress(raw))
    importer = MigrationService(target)
    for operation in (importer.preview, importer.import_package):
        with pytest.raises(PackageError, match="non-zero data after tar end marker"):
            operation(package)
    with storage(target) as adapter:
        adapter.check(empty=True)
        assert all(not list(adapter.records(kind)) for kind in DATASETS)
    index = target.to_provider_config().create_vector_index()
    try:
        assert index.count() == 0
    finally:
        index.close()






@pytest.mark.parametrize("payload", ["text", "metadata", "filters", "archived", "disabled", "unprojected"])
def test_agfs_sql_projection_nul_before_writes(tmp_path, payload):
    from service.migration.package import PackageError
    from tests.unit.service.migration.test_engine import node

    source, target = new_config("agfs", "projection-source"), new_config("agfs", "projection-target")
    row = node()
    if payload in {"text", "archived", "disabled"}:
        row["content"] = "a\x00b"
    elif payload == "filters":
        row["category"] = "a\x00b"
    else:
        row["metadata"]["who" if payload == "metadata" else "custom"] = "a\x00b"
    if payload == "archived":
        row["status"] = "ARCHIVED"
    if payload == "disabled":
        row["metadata"]["indexing_disabled"] = True
    with storage(source) as adapter:
        adapter.restore("nodes", [row])
    path = tmp_path / "projection.tar.gz"
    expected = MigrationService(source).export(path)["data"]
    importer = MigrationService(target)
    if payload in {"text", "metadata", "filters"}:
        for operation in (importer.preview, importer.import_package):
            with pytest.raises(PackageError, match="SQL target cannot store NUL"):
                operation(path)
            with storage(target) as adapter:
                adapter.check(empty=True)
            index = target.to_provider_config().create_vector_index()
            try:
                assert index.count() == 0
            finally:
                index.close()
    else:
        importer.preview(path)
        assert importer.import_package(path)["data"] == expected


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
def test_cli_real_http_producer_migration_and_consumer(tmp_path, backend, monkeypatch):
    import threading
    from dataclasses import replace

    from flask import Flask
    from werkzeug.serving import make_server

    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.migration_api import create_migration_blueprint
    from server.service_factory import close_service, create_service
    from tests.unit.commit.test_memory_identity import memory

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    key = uuid.uuid4().hex
    env = {**os.environ, 'OGMEM_ROOT_API_KEY': key}
    with ExitStack() as lifecycle:
        services, urls = [], []
        for side in ('source', 'target'):
            cfg = new_config(backend, side)
            cfg.root_api_key, cfg.role_control_enabled = key, True
            cfg.migration_workspace = str(tmp_path / side)
            service = create_service(cfg)
            lifecycle.callback(close_service, service)
            service.get_key_manager().create_account('acme', 'alice')
            services.append(service)
            app = Flask(side)
            app.register_blueprint(create_migration_blueprint(lambda service=service: service))
            server = make_server('127.0.0.1', 0, app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            lifecycle.callback(server.server_close)
            lifecycle.callback(thread.join, 5)
            lifecycle.callback(server.shutdown)
            urls.append(f'http://127.0.0.1:{server.server_port}')
        ctx = RequestContext('acme', 'alice', '', 'cli-test', 'cli-test', Role.MEMBER)
        content = 'Prefer compact CLI reports.'
        written = services[0].get_write_api().write_memory(
            replace(memory('cli-style', category='preference', content=content), provenance_ids=[]), ctx)
        def cli(url, *args):
            result = subprocess.run([sys.executable, '-m', 'cli.main', 'migration', '--base-url', url,
                                     '--wait-timeout', '60', *args], env=env, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
            return json.loads(result.stdout)
        package = tmp_path / 'cli-export.tar.gz'
        exported = cli(urls[0], 'export', '--output', str(package))
        assert exported['status'] == 'succeeded'
        preview = cli(urls[1], 'preflight', str(package))
        imported = cli(urls[1], 'import', '--preflight-id', preview['id'], '--yes')
        assert imported['status'] == 'succeeded'
        assert cli(urls[1], 'status', imported['id'])['id'] == imported['id']
        assert services[1].get_read_api().read_memory(written['target_uri'], ctx).content_excerpt == content
        repeated = cli(urls[1], 'preflight', str(package))
        assert repeated['result']['counts']['new'] == 0
        copy = tmp_path / 'cli-download.tar.gz'
        cli(urls[0], 'status', exported['id'], '--output', str(copy))
        assert copy.read_bytes() == package.read_bytes()


@pytest.mark.parametrize('method', ['write_node', 'write_node_with_outbox'])
def test_review_sql_conflict_and_snapshot_read(tmp_path, monkeypatch, method):
    from dataclasses import replace

    from core.errors import ConcurrentModificationError
    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.instance import InstanceMigration
    from service.migration.snapshot import SourceSnapshot
    from tests.unit.commit.test_memory_identity import memory

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config('sql', 'review-conflict')
    cfg.migration_workspace = str(tmp_path / 'control')
    service = create_service(cfg)
    try:
        ctx = RequestContext('acme', 'alice', '', 'review', 'review', Role.ROOT)
        written = service.get_write_api().write_memory(
            replace(memory('editor', category='preference', content='Use editor.'), provenance_ids=[]), ctx)
        fs = service._get_context_fs()
        current = fs.read_node(written['target_uri'], ctx)
        stale = replace(current, content='must not persist', metadata={**current.metadata, '_expected_version': current.metadata['version'] + 1})
        with service._admission.scope(write=True):
            token = service._admission._operation.get().token
            with pytest.raises(ConcurrentModificationError):
                getattr(fs, method)(stale, ctx)
        assert fs.read_node(current.uri, ctx).content == 'Use editor.'
        gate = service._admission.store
        with gate._transaction() as db:
            assert db.execute('SELECT 1 FROM operations WHERE token=?', (token,)).fetchone() is None
        with SourceSnapshot(gate, 'snapshot-read', timeout=2).scope():
            assert service._get_context_fs().read_node(current.uri, ctx).content == 'Use editor.'
            assert service.get_read_api().read_memory(current.uri, ctx).content_excerpt == 'Use editor.'
        result = InstanceMigration(service).export(tmp_path / 'export.tar.gz', task='export-after-conflict')
        assert result['data']['nodes']['count'] >= 1
    finally:
        close_service(service)


def test_review_archive_timestamp_preview_and_merge(tmp_path, monkeypatch):
    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.instance import InstanceMigration
    from service.migration.package import open_package, write_package

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config('sql', 'review-archive')
    cfg.migration_workspace = str(tmp_path / 'control')
    service = create_service(cfg)
    try:
        ctx = RequestContext('acme', 'alice', '', 'session1', 'review', Role.ROOT)
        store = service._get_archive_store()
        written = store.write_archive('session1', 'overview', 'abstract', [{'content': 'hello'}], ctx, archive_id='archive1')
        assert written.success
        migration = InstanceMigration(service)
        exported = tmp_path / 'archive.tar.gz'
        migration.export(exported, task='export-archive')
        with open_package(exported) as package:
            records = {kind: list(package.records(kind)) for kind in ('nodes', 'archives', 'relations')}
        records['archives'][0]['created_at'] = '2026-01-01T00:00:00+00:00'
        incoming = tmp_path / 'archive-different-time.tar.gz'
        write_package(incoming, records, source_backend='sql', source_version='test',
                      source_environment=package.manifest['source_environment'])
        assert migration.preview(incoming)['counts'] == {'new': 0, 'unchanged': 1, 'conflicts': 0}
        assert migration.merge(incoming, task='merge-archive')['phase'] == 'succeeded'
        assert service._get_archive_store().read_archive('session1', 'archive1', ctx).messages == [{'content': 'hello'}]
    finally:
        close_service(service)


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
@pytest.mark.parametrize('fail_attachment', [False, True])
def test_review_target_attachment_merge(tmp_path, monkeypatch, backend, fail_attachment):
    from dataclasses import replace

    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.assets import Assets
    from service.migration.instance import InstanceMigration
    from service.migration.package import open_package, write_package
    from tests.unit.commit.test_memory_identity import memory

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config(backend, 'target-attachment')
    cfg.migration_workspace = str(tmp_path / 'control')
    cfg.role_control_enabled, cfg.root_api_key = True, uuid.uuid4().hex
    assets = tmp_path / 'assets'
    assets.mkdir()
    manual = assets / 'manual.txt'
    manual.write_text('Original manual')
    cfg.migration_asset_root = str(assets)
    service = create_service(cfg)
    try:
        service.get_key_manager().create_account('acme', 'alice')
        ctx = RequestContext('acme', 'alice', '', 'session1', 'review', Role.ROOT)
        written = service.get_write_api().write_memory(replace(
            memory('editor', category='preference', content='Old preference',
                   metadata={'source_path': str(manual)}), provenance_ids=[]), ctx)
        # Candidate extraction does not copy arbitrary attachment metadata.
        # Persist the supported attachment reference through the real ContextFS.
        fs = service._get_context_fs()
        saved = fs.read_node(written['target_uri'], ctx)
        fs.write_node(replace(saved, metadata={**saved.metadata, '_expected_version': saved.metadata['version'], 'source_path': str(manual)}), ctx)
        assert fs.read_node(written['target_uri'], ctx).metadata['source_path'] == str(manual)
        migration = InstanceMigration(service, asset_root=assets)
        exported = tmp_path / 'old.tar.gz'
        migration.export(exported, task='attachment-export')
        with open_package(exported) as package:
            records = {kind: list(package.records(kind)) for kind in ('nodes', 'archives', 'relations')}
        for row in records['nodes']:
            if row['uri'] == written['target_uri']:
                row['metadata'].pop('source_path', None)
                row['content'] = 'New preference'
        incoming = tmp_path / 'incoming.tar.gz'
        write_package(incoming, records, source_backend=backend, source_version='test',
                      source_environment=package.manifest['source_environment'])
        # Public writes project asynchronously; import requires the target index
        # to have caught up before maintenance begins.
        import time

        from service.migration.package import PackageError
        deadline = time.monotonic() + 15
        while True:
            try:
                migration.preview(incoming)
                break
            except PackageError as exc:
                if (not any(text in str(exc) for text in ('index record', 'unfinished AGFS mutation', 'unrecognized AGFS state directory'))
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(.1)
        if fail_attachment:
            def fail(*args):
                raise OSError('attachment preparation failed')
            monkeypatch.setattr(Assets, 'restore', fail)
            with pytest.raises(OSError, match='attachment preparation'):
                migration.merge(incoming, task='attachment-merge')
        else:
            assert migration.merge(incoming, task='attachment-merge')['phase'] == 'succeeded'
        assert migration.gate.status() is None
        with migration.gate._transaction() as db:
            assert db.execute('SELECT count(*) FROM operations').fetchone()[0] == 0
        restored = service._get_context_fs().read_node(written['target_uri'], ctx)
        assert restored.metadata['source_path'] == str(manual)
        assert manual.read_text() == 'Original manual'
        assert service.get_read_api().read_memory(written['target_uri'], ctx).content_excerpt
    finally:
        close_service(service)


def test_review_real_http_agfs_symlink_is_rejected(tmp_path):
    import tempfile
    from pathlib import Path

    from pyagfs import AGFSClient

    from service.migration.package import PackageError
    from service.migration.tree_backup import AGFSTreeBackup
    local = os.environ.get('MIGRATION_TEST_AGFS_LOCAL_ROOT')
    if not local:
        pytest.skip('AGFS local mount is required for a real symlink fixture')
    with tempfile.TemporaryDirectory(prefix='migration-link-', dir=local) as folder:
        root = Path(folder)
        (root / 'outside.txt').write_text('not part of backup')
        (root / 'tree').mkdir()
        (root / 'tree/content.md').symlink_to(root / 'outside.txt')
        client = AGFSClient(api_base_url=os.environ['MIGRATION_TEST_AGFS_URL'])
        remote = '/local/' + root.name + '/tree'
        entry = client.ls(remote)[0]
        assert 'isLink' not in entry and entry['mode'] & (1 << 27)
        backup = AGFSTreeBackup(client, [remote], lambda: {})
        with pytest.raises(PackageError, match='file type'):
            list(backup._walk(remote))
        assert (root / 'tree/content.md').is_symlink()
        assert (root / 'outside.txt').read_text() == 'not part of backup'


@pytest.mark.parametrize('method', ['delete_node', 'archive_node', 'move_node'])
def test_missing_sql_node_does_not_block_snapshot(tmp_path, monkeypatch, method):
    from core.errors import AccessDeniedError, NodeNotFoundError
    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.instance import InstanceMigration

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config('sql', 'missing-node')
    cfg.migration_workspace = str(tmp_path / 'control')
    service = create_service(cfg)
    try:
        ctx = RequestContext('acme', 'alice', '', 's', 'review', Role.ROOT)
        fs = service._get_context_fs()
        uri = 'ctx://acme/sessions/s/stage_results/' + 'a' * 64
        args = (uri, uri[:-64] + 'b' * 64, ctx) if method == 'move_node' else (uri, ctx)
        with service._admission.scope(write=True):
            token = service._admission._operation.get().token
            with pytest.raises(NodeNotFoundError):
                getattr(fs, method)(*args)
        with service._admission.store._transaction() as db:
            assert db.execute('SELECT 1 FROM operations WHERE token=?', (token,)).fetchone() is None
        invalid_args = (uri, 'invalid', ctx) if method == 'move_node' else ('invalid', ctx)
        with pytest.raises(AccessDeniedError):
            getattr(fs, method)(*invalid_args)
        InstanceMigration(service, timeout=2).export(tmp_path / 'snapshot.tar.gz', task='after-missing')
    finally:
        close_service(service)


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
def test_archive_notes_survive_public_store_migration(tmp_path, monkeypatch, backend):
    from contextlib import ExitStack

    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.instance import InstanceMigration

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    with ExitStack() as stack:
        services = []
        for side in ('source', 'target'):
            cfg = new_config(backend, 'notes-' + side)
            cfg.migration_workspace = str(tmp_path / side)
            service = create_service(cfg)
            stack.callback(close_service, service)
            services.append(service)
        ctx = RequestContext('acme', 'alice', '', 'session1', 'review', Role.ROOT)
        metadata = {'operator_note': {'values': ['prov:manual', 'ctx://acme/custom']}}
        result = services[0]._get_archive_store().write_archive('session1', 'overview', 'abstract',
            [{'content': 'original message'}], ctx, archive_id='notes', metadata=metadata)
        assert result.success
        path = tmp_path / 'notes.tar.gz'
        InstanceMigration(services[0]).export(path, task='export-notes')
        migration = InstanceMigration(services[1])
        assert migration.preview(path)['counts']['new'] == 1
        assert migration.merge(path, task='merge-notes')['phase'] == 'succeeded'
        restored = services[1]._get_archive_store().read_archive('session1', 'notes', ctx)
        assert restored.messages == [{'content': 'original message'}]
        assert restored.metadata['operator_note'] == metadata['operator_note']


def _standalone_index_process(cfg, pipe):
    from scripts.run_index_service import CoordinatedIndexRunner, _build_service
    from service.migration.maintenance import MaintenanceError

    generation = 0
    def build(config, admission):
        nonlocal generation
        result = _build_service(config, admission)
        generation += 1
        return result
    runner = CoordinatedIndexRunner(cfg, build_service=build)
    last = runner.service
    try:
        pipe.send('ready')
        while True:
            command = pipe.recv()
            if command == 'close':
                break
            try:
                current = runner.service
                if current is not None:
                    last = current
                if command == 'project':
                    result = last._worker.run_once(last._outbox_store, [cfg.account_id], worker_id='review')
                    pipe.send(('projected', result))
                else:
                    pipe.send(('state', generation, current is not None,
                               current._vector_index.count() if current and current._vector_index else 0))
            except MaintenanceError:
                pipe.send(('blocked',))
    finally:
        runner.close()
        pipe.close()


@pytest.mark.parametrize('backend,mode', [('sql', 'default'), ('agfs', 'default'), ('agfs', 'agfs')])
def test_standalone_index_process_migration_and_rollback(tmp_path, monkeypatch, backend, mode):
    import multiprocessing
    import time
    from contextlib import ExitStack
    from dataclasses import replace

    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service
    from service.migration.instance import InstanceMigration
    from service.migration.rebuild import Rebuilder
    from service.migration.snapshot import SourceSnapshot
    from tests.unit.commit.test_memory_identity import memory

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    monkeypatch.setattr(MemoryService, '_start_outbox_worker', lambda self: None)
    ctx = RequestContext('acme', 'alice', '', 'review', 'test', Role.ROOT)
    with ExitStack() as stack:
        services = []
        for side in ('source', 'target'):
            cfg = new_config(backend, 'runner-' + side)
            cfg.migration_workspace = str(tmp_path / side)
            cfg.account_id, cfg.user_id = 'acme', 'alice'
            cfg.retrieval_mode = mode
            cfg.skill_sedimentation_enabled = False
            cfg.index_interval, cfg.index_workers = 3600, 2
            cfg.directory_summary_enabled = False
            cfg.vector_db_type = os.environ.get('MIGRATION_TEST_VECTOR_BACKEND', 'opengauss')
            cfg.chroma_persist_directory = str(tmp_path / (side + '-chroma'))
            cfg.milvus_uri = os.environ.get('OGMEM_TEST_MILVUS_URI', 'http://127.0.0.1:19530')
            cfg.milvus_collection_name = 'runner_' + uuid.uuid4().hex
            service = create_service(cfg)
            stack.callback(close_service, service)
            services.append(service)
        source, target = services
        def write(service, key, text):
            return service.get_write_api().write_memory(replace(
                memory(key, category='preference', content=text), provenance_ids=[]), ctx)['target_uri']
        incoming = write(source, 'incoming', 'incoming salamander')
        package = tmp_path / 'runner.tar.gz'
        InstanceMigration(source).export(package, task='runner-source')
        original = write(target, 'original', 'original tortoise')
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=_standalone_index_process, args=(cfg, child))
        process.start()
        child.close()
        def rpc(command):
            parent.send(command)
            assert parent.poll(20), 'index subprocess did not respond'
            return parent.recv()
        def running_after(generation):
            for _ in range(100):
                state = rpc('state')
                if state[0] == 'state' and state[1] > generation and state[2]:
                    return state[1]
                time.sleep(.05)
            pytest.fail('index subprocess did not rebuild clients')
        try:
            assert parent.poll(20) and parent.recv() == 'ready'
            assert rpc('project')[0] == 'projected'
            initial_generation = rpc('state')[1]
            migration = InstanceMigration(target)
            with SourceSnapshot(migration.gate, 'runner-snapshot', timeout=5).scope():
                assert rpc('project') == ('blocked',)
            migration.preview(package)
            with monkeypatch.context() as failure:
                failure.setattr(Rebuilder, 'verify', lambda *a: (_ for _ in ()).throw(RuntimeError('injected')))
                with pytest.raises(RuntimeError, match='injected'):
                    migration.merge(package, task='runner-rollback')
            restored_generation = running_after(initial_generation)
            assert target.get_read_api().read_memory(original, ctx).content_excerpt == 'original tortoise'
            assert rpc('project')[0] == 'projected'
            def progress(**fields):
                if fields.get('phase') == 'merging':
                    assert rpc('project') == ('blocked',)
            assert migration.merge(package, task='runner-merge', progress=progress)['phase'] == 'succeeded'
            running_after(restored_generation)
            assert rpc('project')[0] == 'projected'
            assert target.get_read_api().read_memory(incoming, ctx).content_excerpt == 'incoming salamander'
            if mode == 'default':
                assert rpc('state')[3] > 0
        finally:
            if process.is_alive():
                parent.send('close')
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert process.exitcode == 0


@pytest.mark.parametrize('operation', ['query', 'fetch', 'lock'])
def test_sql_snapshot_deadline_closes_connection_and_reopens_gate(tmp_path, monkeypatch, operation):
    import time

    import psycopg2

    from service.migration.maintenance import MaintenanceStore
    from service.migration.snapshot import SourceSnapshot
    from service.migration.storage import SQLStorage

    cfg = new_config('sql', 'deadline')
    with storage(cfg) as initialized:
        initialized.conn.commit()
    cfg.sql_dialect = 'auto'
    gate = MaintenanceStore(tmp_path / 'deadline.db', 'instance')
    connections = []
    original_connect = psycopg2.connect
    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn
    monkeypatch.setattr(psycopg2, 'connect', connect)
    blocker = psycopg2.connect(cfg.sql_connection_string)
    try:
        if operation == 'lock':
            with blocker.cursor() as cur:
                cur.execute('LOCK TABLE context_nodes IN ACCESS EXCLUSIVE MODE')
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            with SourceSnapshot(gate, 'snapshot', timeout=1).scope():
                adapter = SQLStorage(cfg)
                try:
                    if operation == 'query':
                        with adapter.conn.cursor() as cur:
                            cur.execute('SELECT pg_sleep(5)')
                    elif operation == 'fetch':
                        with adapter.conn.cursor(name='slow') as cur:
                            cur.itersize = 1
                            cur.execute('SELECT pg_sleep(CASE WHEN i=1 THEN 0 ELSE 5 END) FROM generate_series(1,2) i')
                            rows = iter(cur)
                            next(rows)
                            next(rows)
                    else:
                        list(adapter._rows('context_nodes', ['uri']))
                finally:
                    adapter.close()
        assert time.monotonic() - started < 3
        assert gate.status() is None
        assert all(conn.closed for conn in connections if conn is not blocker)
    finally:
        blocker.rollback()
        blocker.close()


def test_agfs_conflict_and_control_reads_with_full_admission(tmp_path, monkeypatch):
    from dataclasses import replace

    import server.app as app_module
    from core.errors import ConcurrentModificationError, NodeNotFoundError
    from core.models import ContextNode, RequestContext, Role
    from server.memory_service import MemoryService
    from server.service_factory import close_service, create_service

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config('agfs', 'review-maintenance')
    cfg.migration_workspace = str(tmp_path / 'control')
    cfg.role_control_enabled = True
    cfg.root_api_key = 'migration-test-root'
    service = create_service(cfg)
    gate = service._admission.store
    ctx = RequestContext(account_id='acme', user_id='alice', agent_id='agent',
                         session_id='s', trace_id='t', role=Role.ROOT)
    node = ContextNode(uri='ctx://acme/users/alice/memories/preferences/editor',
                       context_type='MEMORY', category='preferences', level=2, owner_space='user:alice',
                       abstract='original', overview='original', content='original', metadata={'version': 2})
    try:
        fs = service._get_context_fs()
        fs.write_node(node, ctx)
        with pytest.raises(ConcurrentModificationError):
            fs.write_node(replace(node, metadata={'_expected_version': 1}), ctx)
        fs.write_node(replace(node, metadata={'_expected_version': 2}), ctx)
        with pytest.raises(ConcurrentModificationError):
            fs.write_node(replace(node, metadata={'_expected_version': 0}), ctx)
        with pytest.raises(NodeNotFoundError):
            fs.move_node(node.uri + '-missing', node.uri + '-other', ctx)
        with pytest.raises(NodeNotFoundError):
            fs.delete_node(node.uri + '-missing', ctx)
        archives = service._get_archive_store()
        gate.enter('warm-archive', mode='export', resources={})
        try:
            gate.wait_drained('warm-archive', 2)
            assert service._get_archive_store() is archives
        finally:
            gate.reopen('warm-archive')
        service.get_key_manager().create_account('acme', 'alice')
        monkeypatch.setattr(app_module, '_service', service)
        gate.enter('control-test', mode='import', resources={})
        gate.wait_drained('control-test', 2)
        from service.migration.participants import wait_participants
        wait_participants(gate, 'control-test', 'drain', timeout=5)
        client = app_module.app.test_client()
        headers = {'X-API-Key': cfg.root_api_key, 'X-Account-ID': 'acme', 'X-User-ID': 'alice'}
        response = client.get('/api/v1/admin/accounts', headers=headers)
        assert response.status_code == 200, response.json
        assert client.get('/api/v1/admin/accounts/acme/roles', headers=headers).status_code == 200
        assert client.get('/api/v1/admin/accounts', headers={'X-API-Key': 'invalid'}).status_code == 401
    finally:
        if gate.status() is not None:
            from service.migration.participants import wait_participants
            gate.request_refresh('control-test')
            wait_participants(gate, 'control-test', 'refresh', timeout=5)
            gate.reopen('control-test')
        close_service(service)


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
def test_maintenance_preserves_newer_session_from_other_service(tmp_path, monkeypatch, backend):
    from core.models import RequestContext, Role
    from server.memory_service import MemoryService
    from server.migration_lifecycle import ServiceLifecycle
    from server.service_factory import close_service, create_service
    from session.session_manager import SessionManager
    from session.session_state import SessionState

    monkeypatch.setattr(MemoryService, 'start_extraction_recovery', lambda self: None)
    cfg = new_config(backend, 'session-checkpoint')
    cfg.migration_workspace = str(tmp_path / 'gate')
    cfg.skill_sedimentation_enabled = False
    first, second = create_service(cfg), create_service(cfg)
    ctx = RequestContext('acme', 'alice', 'agent', 's', 'review', Role.ROOT)
    try:
        for service, content in ((first, 'old message'), (second, 'latest message')):
            fs = service._get_context_fs()
            manager = SessionManager(get_context_fs=lambda fs=fs: fs)
            service._session_mgr = manager
            manager.add_message('s', 'user', content, ctx)
            with service._session_mutation_scope(ctx, refresh=False):
                assert manager.save_session_state('s', ctx)
        fs = second._get_context_fs()
        uri = SessionState.state_uri('acme', 's')
        before = fs.read_node(uri, ctx).content
        assert 'latest message' in before
        ServiceLifecycle(first)._save_sessions()
        assert fs.read_node(uri, ctx).content == before
    finally:
        close_service(first)
        close_service(second)
