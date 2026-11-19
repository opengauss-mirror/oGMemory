"""Offline migration-only HTTP service. Intentionally never imports MemoryService."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, session
from werkzeug.exceptions import (
    BadRequest,
    HTTPException,
    RequestEntityTooLarge,
    UnsupportedMediaType,
)

from service.migration.engine import MigrationService
from service.migration.package import PackageError

logger = logging.getLogger(__name__)


def _failure_diagnostic(error, cfg):
    from skills.sanitizer import sanitize_trace_text

    secrets = {os.environ.get("OGMEM_MIGRATION_API_KEY", "")}
    for name, value in vars(cfg).items():
        if isinstance(value, str) and re.search(
            r"key|password|secret|token|connection_string", name, re.I
        ):
            secrets.add(value)
            if "connection_string" in name and value:
                try:
                    from psycopg2.extensions import parse_dsn

                    secrets.add(parse_dsn(value).get("password", ""))
                except Exception:
                    pass  # A malformed DSN must not prevent failure reporting.
    secrets.discard("")
    parts, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # Keep call locations, but never source lines or local variable values.
        frames = [
            f"  {frame.f_globals.get('__name__', '<unknown>')}.{frame.f_code.co_name}:{line}"
            for frame, line in traceback.walk_tb(error.__traceback__)
        ]
        reason = sanitize_trace_text(str(error))
        for secret in sorted(secrets, key=len, reverse=True):
            reason = reason.replace(secret, "[REDACTED_SECRET]")
        parts.append("\n".join([*frames, f"{type(error).__name__}: {reason}"]))
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return "\nException chain:\n".join(parts)


class MigrationJobs:
    def __init__(self, cfg, workspace, service=None, asset_root=None):
        self.cfg = cfg
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = (self.root / "process.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                # Lock the same byte even when the existing file is nonempty.
                # Windows permits byte-range locks beyond the current EOF.
                self.lockfile.seek(0)
                msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise RuntimeError("migration workspace already in use") from None
        self.service = service or MigrationService(cfg, asset_root=asset_root)
        self.lock = threading.Lock()
        # Windows does not allow replacement while a reader holds the file.
        self.state_lock = threading.Lock()
        for path in self.root.glob("job-*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] == "running":
                job.update(
                    status="failed",
                    phase="interrupted",
                    error="Process stopped; clean the isolated target before retrying.",
                )
                self.save(job)

    def save(self, job):
        path = self.root / f"job-{job['id']}.json"
        temporary = path.with_suffix(".tmp")
        with self.state_lock:
            temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)

    def get(self, job_id):
        job_id = str(uuid.UUID(job_id))
        with self.state_lock:
            return json.loads((self.root / f"job-{job_id}.json").read_text(encoding="utf-8"))

    def artifact(self, artifact_id):
        artifact_id = str(uuid.UUID(artifact_id))
        path = self.root / f"upload-{artifact_id}.tar.gz"
        if not path.is_file():
            raise FileNotFoundError("upload not found")
        return path

    def submit(self, kind, artifact_id=None):
        if not self.lock.acquire(blocking=False):
            raise PackageError("another migration task is running")
        job = {"id": str(uuid.uuid4()), "kind": kind, "status": "running", "phase": "queued"}
        try:
            source = self.artifact(artifact_id) if kind != "export" else None
            self.save(job)
        except Exception:
            self.lock.release()
            raise

        def progress(**fields):
            job.update(fields)
            self.save(job)

        def run():
            try:
                if kind == "export":
                    result = self.service.export(self.root / f"export-{job['id']}.tar.gz", progress)
                elif kind == "preflight":
                    progress(phase="validating")
                    result = self.service.preflight(source)
                    result["artifact_id"] = artifact_id
                else:
                    result = self.service.restore(source, progress)
                progress(status="succeeded", phase="complete", result=result)
            except Exception as exc:
                # Raw driver errors may contain connection strings or model secrets.
                detail = (
                    str(exc)
                    if isinstance(exc, PackageError)
                    else f"{type(exc).__name__}: operation failed; inspect isolated target before retrying"
                )
                try:
                    diagnostic = _failure_diagnostic(exc, self.cfg)
                except Exception:
                    diagnostic = f"{type(exc).__name__}: diagnostic unavailable"
                # Only sanitized strings reach logging handlers; no raw exc_info.
                logger.error(
                    "Migration task failed job_id=%s kind=%s phase=%s\n%s",
                    job["id"],
                    kind,
                    job["phase"],
                    diagnostic,
                    extra={"account_id": "-", "trace_id": job["id"]},
                )
                progress(status="failed", error=detail)
            finally:
                self.lock.release()

        threading.Thread(target=run, name="memory-migration", daemon=True).start()
        return job["id"]


def create_app(
    cfg,
    workspace,
    *,
    maintenance_confirmed=False,
    service=None,
    max_upload_bytes=10 * 1024**3,
    asset_root=None,
):
    if not maintenance_confirmed:
        raise ValueError("stop all business writers and explicitly confirm maintenance")
    api_key = os.environ.get("OGMEM_MIGRATION_API_KEY") or cfg.root_api_key
    if not api_key:
        raise ValueError(
            "set OGMEM_MIGRATION_API_KEY or root_api_key; anonymous migration is forbidden"
        )
    app = Flask(__name__)
    app.secret_key = os.urandom(32)
    app.config.update(
        MAX_CONTENT_LENGTH=max_upload_bytes,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_NAME="ogmem_migration_"
        + hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:12],
    )
    jobs = MigrationJobs(cfg, workspace, service, asset_root)
    app.extensions["migration_jobs"] = jobs
    static = Path(__file__).parent / "memory_viewer_static"

    @app.before_request
    def authorize():
        if request.path in {
            "/",
            "/memory-viewer/migration",
            "/memory-viewer/migration.js",
            "/memory-viewer/migration.css",
            "/memory-viewer/styles.css",
            "/memory-viewer/theme.css",
        }:
            return None
        supplied = request.headers.get("X-API-Key", "")
        by_key = bool(supplied) and hmac.compare_digest(supplied, api_key)
        if not by_key and not session.get("migration_admin"):
            return jsonify(error="Explicit migration administrator authentication required"), 401
        if request.method != "GET" and request.headers.get("Origin"):
            if request.headers["Origin"].rstrip("/") != request.host_url.rstrip("/"):
                return jsonify(error="Cross-origin mutation forbidden"), 403
        return None

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, FileNotFoundError):
            return jsonify(error="Not found"), 404
        if isinstance(exc, (PackageError, ValueError)):
            return jsonify(error=str(exc)), 400
        return jsonify(error=f"{type(exc).__name__}: migration request failed"), 500

    @app.get("/")
    @app.get("/memory-viewer/migration")
    def page():
        return send_file(static / "migration.html")

    @app.get("/memory-viewer/migration.js")
    def javascript():
        return send_file(static / "migration.js")

    @app.get("/memory-viewer/styles.css")
    @app.get("/memory-viewer/theme.css")
    @app.get("/memory-viewer/migration.css")
    def stylesheet():
        return send_file(static / request.path.rsplit("/", 1)[-1])

    @app.post("/api/migration/login")
    def login():
        # This endpoint was already authenticated by the request hook.
        session["migration_admin"] = True
        session.permanent = False
        return jsonify(ok=True)

    @app.get("/api/migration/status")
    def status():
        with jobs.state_lock:
            return jsonify(
                mode="migration-only",
                storage=cfg.storage_backend,
                vector=cfg.vector_db_type,
                jobs=[
                    json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(
                        jobs.root.glob("job-*.json"), key=lambda p: p.stat().st_mtime, reverse=True
                    )
                ][:20],
            )

    @app.post("/api/migration/uploads")
    def upload():
        if not jobs.lock.acquire(blocking=False):
            return jsonify(error="Wait for the running task before uploading"), 409
        artifact_id = str(uuid.uuid4())
        path = jobs.root / f"upload-{artifact_id}.tar.gz"
        partial = path.with_suffix(".partial")
        try:
            digest, size = hashlib.sha256(), 0
            with partial.open("xb") as stream:
                while chunk := request.stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise PackageError("upload exceeds limit")
                    digest.update(chunk)
                    stream.write(chunk)
            partial.rename(path)
            return jsonify(artifact_id=artifact_id, size=size, sha256=digest.hexdigest()), 201
        finally:
            partial.unlink(missing_ok=True)
            jobs.lock.release()

    @app.post("/api/migration/jobs")
    def submit():
        # Control requests contain identifiers only; never parse an upload-sized
        # JSON body into memory, including requests without Content-Length.
        if not request.is_json:
            raise UnsupportedMediaType()
        raw = request.stream.read(16 * 1024 + 1)
        if len(raw) > 16 * 1024:
            raise RequestEntityTooLarge()
        try:
            body = app.json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise BadRequest("invalid migration control JSON") from exc
        if not isinstance(body, dict) or body.get("kind") not in {"export", "preflight", "import"}:
            raise PackageError("kind must be export, preflight or import")
        if body.get("maintenance_confirmed") is not True:
            raise PackageError("confirm maintenance before starting")
        if body["kind"] == "import":
            plan = jobs.get(body.get("preflight_id", ""))
            if (
                plan["kind"] != "preflight"
                or plan["status"] != "succeeded"
                or plan["result"]["artifact_id"] != body.get("artifact_id")
            ):
                raise PackageError("successful preflight required for this upload")
        return jsonify(job_id=jobs.submit(body["kind"], body.get("artifact_id"))), 202

    @app.get("/api/migration/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(jobs.get(job_id))

    @app.get("/api/migration/jobs/<job_id>/download")
    def download(job_id):
        job = jobs.get(job_id)
        if job["status"] != "succeeded" or job["kind"] != "export":
            raise PackageError("only completed exports can be downloaded")
        return send_file(
            jobs.root / f"export-{job['id']}.tar.gz",
            as_attachment=True,
            download_name=f"memory-{job['id']}.ogmem.tar.gz",
            conditional=True,
        )

    return app


def add_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--workspace", required=True, help="Private job/upload directory, outside business data"
    )
    parser.add_argument(
        "--maintenance-confirmed",
        action="store_true",
        help="All business writers have been stopped",
    )
    parser.add_argument(
        "--asset-root", help="Allowed source attachment root / durable target attachment directory"
    )
    parser.add_argument("--max-upload-mib", type=int, default=10240)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)


def run(args):
    from providers.unified_config import get_config

    if not 1 <= args.max_upload_mib <= 10240:
        raise ValueError("max-upload-mib must be between 1 and 10240")
    cfg = get_config(args.config)
    app = create_app(
        cfg,
        args.workspace,
        maintenance_confirmed=args.maintenance_confirmed,
        asset_root=args.asset_root,
        max_upload_bytes=args.max_upload_mib * 1024**2,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
