import time
from types import SimpleNamespace

import pytest

from server.migration_server import create_app


class Service:
    def export(self, destination, progress):
        destination.write_bytes(b"package")
        return {"nodes": 1}

    def preflight(self, source):
        return {"size": source.stat().st_size}

    def restore(self, source, progress):
        return {"nodes": 1}


def cfg():
    return SimpleNamespace(
        root_api_key="test-secret", storage_backend="agfs", vector_db_type="chroma"
    )


def done(client, job_id):
    for _ in range(100):
        job = client.get("/api/migration/jobs/" + job_id).get_json()
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail("job did not finish")


def test_migration_gui_loads_shared_theme_before_sign_in(tmp_path):
    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"/memory-viewer/migration.css" in page.data
    for filename in ("theme.css", "migration.css"):
        response = client.get(f"/memory-viewer/{filename}")
        assert response.status_code == 200
        assert response.mimetype == "text/css"


def test_migration_mode_requires_operator_confirmation_and_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("OGMEM_MIGRATION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="maintenance"):
        create_app(cfg(), tmp_path)
    c = cfg()
    c.root_api_key = None
    with pytest.raises(ValueError, match="anonymous"):
        create_app(c, tmp_path, maintenance_confirmed=True)
    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    client = app.test_client()
    assert client.get("/api/migration/status").status_code == 401
    assert client.get("/").status_code == 200
    assert client.post("/api/v1/memories").status_code == 401
    assert client.post("/api/migration/login", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.post("/api/migration/login", headers={"X-API-Key": "test-secret"}).status_code == 200
    )
    assert client.get("/api/migration/status").get_json()["mode"] == "migration-only"
    assert client.post("/api/v1/memories").status_code == 404
    assert (
        client.post(
            "/api/migration/jobs",
            json={"kind": "export", "maintenance_confirmed": True},
            headers={"Origin": "https://attacker.invalid"},
        ).status_code
        == 403
    )


def test_gui_api_export_download_upload_preflight_import(tmp_path):
    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "test-secret"})
    result = client.post(
        "/api/migration/jobs", json={"kind": "export", "maintenance_confirmed": True}
    )
    assert result.status_code == 202
    job_id = result.get_json()["job_id"]
    assert done(client, job_id)["status"] == "succeeded"
    assert client.get(f"/api/migration/jobs/{job_id}/download").data == b"package"
    uploaded = client.post(
        "/api/migration/uploads", data=b"package", content_type="application/octet-stream"
    ).get_json()
    body = {
        "kind": "preflight",
        "artifact_id": uploaded["artifact_id"],
        "maintenance_confirmed": True,
    }
    plan = client.post("/api/migration/jobs", json=body).get_json()["job_id"]
    assert done(client, plan)["status"] == "succeeded"
    body.update(kind="import", preflight_id=plan)
    imported = client.post("/api/migration/jobs", json=body).get_json()["job_id"]
    assert done(client, imported)["status"] == "succeeded"
    assert client.get(f"/api/migration/jobs/{imported}/download").status_code == 400


def test_upload_bound_and_single_process_lock(tmp_path):
    app = create_app(
        cfg(), tmp_path, maintenance_confirmed=True, service=Service(), max_upload_bytes=8
    )
    with pytest.raises(RuntimeError, match="already in use"):
        create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "test-secret"})
    assert client.post("/api/migration/uploads", data=b"x" * 9).status_code == 413
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("known_length", [True, False])
def test_control_body_is_bounded_before_json_parse(tmp_path, monkeypatch, known_length):
    import io

    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "test-secret"})
    original = app.json.loads

    def bounded_loads(raw, *args, **kwargs):
        assert len(raw) <= 16 * 1024, "unbounded control body reached JSON parser"
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(app.json, "loads", bounded_loads)
    raw = b'{"kind":"export","maintenance_confirmed":true,"extra":"' + b"x" * 65536 + b'"}'
    if known_length:
        response = client.post("/api/migration/jobs", data=raw, content_type="application/json")
    else:
        response = client.post(
            "/api/migration/jobs",
            content_type="application/json",
            environ_overrides={
                "wsgi.input": io.BytesIO(raw),
                "wsgi.input_terminated": True,
                "CONTENT_LENGTH": "",
            },
        )
    assert response.status_code == 413
    assert not list(tmp_path.glob("job-*.json"))


@pytest.mark.parametrize("raw", [b"{", b"[" * 2000 + b"]" * 2000])
def test_malformed_control_json_is_rejected(tmp_path, raw):
    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    response = app.test_client().post(
        "/api/migration/jobs",
        data=raw,
        content_type="application/json",
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 400
    assert not list(tmp_path.glob("job-*.json"))


@pytest.mark.parametrize("terminate", [False, True], ids=["normal-exit", "terminated"])
@pytest.mark.parametrize("existing", [False, True], ids=["new-lockfile", "existing-lockfile"])
def test_workspace_lock_across_processes_and_after_exit(tmp_path, terminate, existing):
    import subprocess
    import sys

    from server.migration_server import MigrationJobs

    workspace = tmp_path / "shared"
    workspace.mkdir()
    if existing:
        (workspace / "process.lock").write_bytes(b"existing lock file")
    code = """
import sys
from server.migration_server import MigrationJobs
jobs = MigrationJobs(None, sys.argv[1], service=object())
print('LOCKED', flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(workspace)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "LOCKED", child.stderr.read()
        with pytest.raises(RuntimeError, match="already in use"):
            MigrationJobs(None, workspace, service=object())
        independent = MigrationJobs(None, tmp_path / "independent", service=object())
        independent.lockfile.close()
        if terminate:
            child.terminate()
        child.communicate(input="", timeout=15)
        if not terminate:
            assert child.returncode == 0
        recovered = MigrationJobs(None, workspace, service=object())
        recovered.lockfile.close()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)


def test_job_progress_can_be_polled_while_state_is_replaced(tmp_path):
    import threading
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=Service())
    jobs = app.extensions["migration_jobs"]
    job = {"id": str(uuid.uuid4()), "kind": "export", "status": "running", "count": 0}
    jobs.save(job)
    ready = threading.Barrier(4)

    def write_progress():
        ready.wait(timeout=15)
        for count in range(100):
            jobs.save({**job, "count": count})

    def poll_progress():
        client = app.test_client()
        ready.wait(timeout=15)
        for _ in range(100):
            assert 0 <= jobs.get(job["id"])["count"] < 100
            response = client.get("/api/migration/status", headers={"X-API-Key": "test-secret"})
            assert response.status_code == 200
            assert response.get_json()["jobs"][0]["id"] == job["id"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_progress)]
        futures.extend(executor.submit(poll_progress) for _ in range(3))
        for future in futures:
            future.result(timeout=30)
    assert jobs.get(job["id"])["count"] == 99


def test_unicode_preflight_report_survives_polling_and_restart(tmp_path):
    from server.migration_server import MigrationJobs

    class UnicodeService(Service):
        def preflight(self, source):
            return {"model": "本地模型🌍", "size": source.stat().st_size}

    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=UnicodeService())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "test-secret"})
    uploaded = client.post("/api/migration/uploads", data=b"package").get_json()
    job_id = client.post(
        "/api/migration/jobs",
        json={
            "kind": "preflight",
            "artifact_id": uploaded["artifact_id"],
            "maintenance_confirmed": True,
        },
    ).get_json()["job_id"]
    job = done(client, job_id)
    assert job["status"] == "succeeded", job
    assert job["result"]["model"] == "本地模型🌍"
    jobs = app.extensions["migration_jobs"]
    jobs.lockfile.close()
    restarted = MigrationJobs(cfg(), tmp_path, service=UnicodeService())
    try:
        assert restarted.get(job_id) == job
    finally:
        restarted.lockfile.close()


@pytest.mark.parametrize("kind", ["export", "preflight", "import"])
def test_failure_logs_redacted_cause_and_call_sites(tmp_path, caplog, monkeypatch, kind):
    import logging

    c = cfg()
    c.openai_embedding_api_key = "configured-embedding-credential"
    c.sql_connection_string = "host=localhost dbname=test password='db pass\\'word'"
    monkeypatch.setenv("OGMEM_MIGRATION_API_KEY", "migration-env-credential")

    class BrokenService(Service):
        def fail(self):
            try:
                raise ConnectionError(
                    "Connection refused; configured-embedding-credential db pass'word "
                    "migration-env-credential password='labelled-secret' "
                    'response={"api_key":"json-secret"}'
                )
            except ConnectionError as cause:
                raise RuntimeError("embedding connection failed") from cause

        def export(self, destination, progress):
            progress(phase="exporting")
            self.fail()

        def preflight(self, source):
            self.fail()

        def restore(self, source, progress):
            progress(phase="reindexing")
            self.fail()

    caplog.set_level(logging.ERROR, logger="server.migration_server")
    app = create_app(c, tmp_path, maintenance_confirmed=True, service=BrokenService())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "migration-env-credential"})
    artifact = client.post("/api/migration/uploads", data=b"package").get_json()["artifact_id"]
    jobs = app.extensions["migration_jobs"]
    job_id = jobs.submit(kind, artifact if kind != "export" else None)
    job = done(client, job_id)
    assert job["status"] == "failed"
    assert job["error"] == "RuntimeError: operation failed; inspect isolated target before retrying"
    records = [r for r in caplog.records if r.name == "server.migration_server"]
    assert len(records) == 1 and records[0].exc_info is None
    text = records[0].getMessage()
    phase = {"export": "exporting", "preflight": "validating", "import": "reindexing"}[kind]
    assert f"job_id={job_id} kind={kind} phase={phase}" in text
    assert "Connection refused" in text and "ConnectionError" in text
    assert "embedding connection failed" in text
    assert "Exception chain:" in text and ".fail:" in text and ".run:" in text
    assert "raise " not in text
    for secret in (
        "configured-embedding-credential",
        "db pass'word",
        "migration-env-credential",
        "labelled-secret",
        "json-secret",
        c.sql_connection_string,
    ):
        assert secret not in text and secret not in str(job)


def test_diagnostic_failure_does_not_leave_job_running(tmp_path, caplog, monkeypatch):
    from server import migration_server

    class BrokenService(Service):
        def export(self, destination, progress):
            raise RuntimeError("private failure")

    def broken_diagnostic(error, cfg):
        raise ValueError("private diagnostic failure")

    monkeypatch.setattr(migration_server, "_failure_diagnostic", broken_diagnostic)
    app = create_app(cfg(), tmp_path, maintenance_confirmed=True, service=BrokenService())
    client = app.test_client()
    client.post("/api/migration/login", headers={"X-API-Key": "test-secret"})
    job_id = app.extensions["migration_jobs"].submit("export")
    assert done(client, job_id)["status"] == "failed"
    assert "diagnostic unavailable" in caplog.text
    assert "private failure" not in caplog.text
    assert "private diagnostic failure" not in caplog.text
