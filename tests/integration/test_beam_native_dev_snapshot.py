"""Opt-in baseline/graph SQL/pgvector smoke, using owned disposable containers.

BEAM_SQL_SMOKE=1 python -m pytest -q tests/integration/test_beam_native_dev_snapshot.py
Requires a cached pgvector/pgvector:pg16 image; never pulls or calls a model.
"""
import os
import secrets
import subprocess
import time
import uuid

import pytest


@pytest.mark.skipif(os.environ.get("BEAM_SQL_SMOKE") != "1", reason="Explicit disposable Docker SQL smoke only")
@pytest.mark.parametrize("graph_backend", ["none", "sql"])
def test_native_schema_and_full_snapshot_roundtrip(tmp_path, monkeypatch, graph_backend):
    import core  # noqa: F401 -- Match the API's cold import order.
    from providers.unified_config import OgMemConfig
    from providers.embedder.openai_embedder import OpenAIEmbedder
    from server.memory_service import MemoryService
    from scripts.beam_eval_safety import SnapshotDB
    from scripts.beam_snapshot import freeze_qualified_snapshot, resolve_snapshot

    image = "pgvector/pgvector:pg16"
    subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, check=True)
    suffix = uuid.uuid4().hex[:12]
    container, database = "beam-dev-smoke-" + suffix, "beam_eval_dev_smoke_" + suffix
    monkeypatch.setattr(OpenAIEmbedder, "embed_texts", lambda *_: pytest.fail("Smoke must not call embedding API"))
    password = secrets.token_hex(20)
    env = {**os.environ, "POSTGRES_PASSWORD": password}
    subprocess.run(["docker", "run", "--pull=never", "--rm", "-d", "--name", container,
        "--label", "anttrail.eval.run=" + container, "--label", "anttrail.eval.role=postgres",
        "--tmpfs", "/var/lib/postgresql/data:rw,size=192m", "-p", "127.0.0.1::5432",
        "-e", "POSTGRES_PASSWORD", "-e", "POSTGRES_DB=" + database, image],
        env=env, stdout=subprocess.DEVNULL, check=True)
    service = None
    try:
        deadline = time.monotonic() + 60
        # The image's temporary initdb server listens only on a Unix socket.
        # Wait for the final TCP daemon, not that transient initialization step.
        while subprocess.run(["docker", "exec", container, "pg_isready", "-h", "127.0.0.1", "-U", "postgres", "-d", database],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            assert time.monotonic() < deadline, "Disposable PostgreSQL did not become ready"
            time.sleep(.2)
        port = int(subprocess.check_output(["docker", "port", container, "5432/tcp"], text=True).strip().rsplit(":", 1)[1])
        dsn = f"host=127.0.0.1 port={port} dbname={database} user=postgres password={password}"
        cfg = OgMemConfig(provider="mock", embedding_provider="openai", openai_api_key="unused",
            openai_base_url="http://127.0.0.1:9/v1", openai_embedding_api_key="unused",
            openai_embedding_base_url="http://127.0.0.1:9/v1", openai_embedding_model="text-embedding-3-small",
            storage_backend="sql", sql_connection_string=dsn, sql_dialect="postgresql",
            vector_db_type="opengauss", opengauss_connection_string=dsn, vector_db_dialect="postgresql",
            opengauss_dimension=1536, enable_cache=False, index_workers=0,
            graph_backend=graph_backend, enable_graph_expansion=graph_backend == "sql",
            graph_supplement_enabled=graph_backend == "sql", scene_enabled=False,
            scene_generation_enabled=False, scene_retrieval_enabled=False)
        service = MemoryService(cfg)
        service.get_read_api(start_outbox_worker=False)
        # Close the native shared SQL pool before any destructive test restore.
        service.shutdown()
        service._sql_pool.close_all()
        service = None
        db = SnapshotDB(container, database)
        # PR314's common SQL schema contains this queue even with graph off.
        # Its presence does not mean graph construction is enabled, and it must
        # still be included in the full fingerprint and settled before freeze.
        assert db.sql("SELECT to_regclass('public.edge_projection_jobs') IS NOT NULL") == "t"
        assert db.sql("SELECT count(*) FROM edge_projection_jobs") == "0"
        db.sql("INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,abstract,content) "
               "VALUES ('ctx://smoke/users/u/memories/entities/source','smoke','user:u','entity','MEMORY','Synthetic','Original detail')")
        db.sql("CREATE TABLE beam_smoke_extra(id BIGSERIAL PRIMARY KEY, body TEXT); "
               "INSERT INTO beam_smoke_extra(body) VALUES ('Synthetic row outside the required-table list')")
        before = db.fingerprint()
        assert before["tables"]["public.vector_index"]["count"] == 0
        root = tmp_path / "freeze"
        frozen = freeze_qualified_snapshot(db, root, {"run_id": container, "runtime": {
            "config": {"graph_backend": graph_backend, "storage_backend": "sql", "enable_cache": False}}})
        assert frozen["inventory"]["tables"] == before["tables"]
        assert frozen["inventory"]["sequences"] == before["sequences"]
        qualified = resolve_snapshot(root)
        db.sql("UPDATE context_nodes SET content='Mutated'; INSERT INTO beam_smoke_extra(body) VALUES ('Mutated')")
        assert db.fingerprint() != frozen["inventory"]
        db.restore(qualified)
        assert db.fingerprint() == frozen["inventory"]
        assert db.sql("SELECT content FROM context_nodes") == "Original detail"
        assert (root / "snapshot.dump").exists()
    finally:
        if service is not None and service._sql_pool is not None:
            service.shutdown()
            service._sql_pool.close_all()
        subprocess.run(["docker", "stop", "--time", "5", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
