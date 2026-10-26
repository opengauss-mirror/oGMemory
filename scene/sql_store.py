"""SQL persistence for scene snapshots, leased jobs, and configuration.

The adapters share the existing connection pool and SQL dialect. Scope changes
use short database transactions; model calls execute after those transactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from threading import Lock

from psycopg2.extras import Json

from fs.sql_adapter.pool import PoolAdapterMixin
from fs.sql_adapter.sql_context_fs import SQLContextFS
from retrieval.owner_scope import expand_owner_space_aliases
from scene.config import ConfigRevisionConflict, SceneConfig, SceneConfigSnapshot
from scene.jobs import (
    InMemorySceneJobStore,
    SceneJob,
    SceneJobStatus,
    SceneLeaseLostError,
    current_scene_job,
    job_from_dict,
    job_to_dict,
)
from scene.models import PersistedScene, SceneCluster, SceneMember, SceneRecord, SceneSnapshot

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_work_accounts (
    account_id VARCHAR(128) PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS scene_scopes (
    account_id VARCHAR(128) NOT NULL,
    owner_space VARCHAR(128) NOT NULL,
    revision BIGINT NOT NULL DEFAULT 0,
    snapshot JSONB NOT NULL DEFAULT '{}',
    queue JSONB NOT NULL DEFAULT '{}',
    due_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, owner_space)
);
CREATE INDEX IF NOT EXISTS idx_scene_scope_due ON scene_scopes (account_id, due_at);
ALTER TABLE scene_scopes ADD COLUMN IF NOT EXISTS lexical_revision BIGINT NOT NULL DEFAULT -1;
CREATE TABLE IF NOT EXISTS scene_snapshot_history (
    account_id VARCHAR(128) NOT NULL,
    owner_space VARCHAR(128) NOT NULL,
    revision BIGINT NOT NULL,
    snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, owner_space, revision)
);
CREATE TABLE IF NOT EXISTS scene_jobs (
    account_id VARCHAR(128) NOT NULL,
    owner_space VARCHAR(128) NOT NULL,
    job_id VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_scene_jobs_scope ON scene_jobs (account_id, owner_space, updated_at);
CREATE TABLE IF NOT EXISTS scene_write_notices (
    account_id VARCHAR(128) NOT NULL,
    owner_space VARCHAR(128) NOT NULL,
    notice_id VARCHAR(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, owner_space, notice_id)
);
CREATE TABLE IF NOT EXISTS scene_config_revisions (
    account_id VARCHAR(128) NOT NULL,
    owner_space VARCHAR(128) NOT NULL,
    revision BIGINT NOT NULL,
    payload JSONB NOT NULL,
    source VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, owner_space, revision)
);
"""
for _table in ("scene_scopes", "scene_snapshot_history", "scene_jobs", "scene_write_notices", "scene_config_revisions"):
    _SCHEMA += f"""
ALTER TABLE {_table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {_table} FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON {_table}
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
"""


def _json(raw):
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def snapshot_to_dict(snapshot: SceneSnapshot) -> dict:
    return {"account_id": snapshot.account_id, "owner_space": snapshot.owner_space,
            "revision": snapshot.revision, "scenes": [asdict(scene) for scene in snapshot.scenes]}


def snapshot_from_dict(account_id: str, owner_space: str, revision: int, payload: dict) -> SceneSnapshot:
    records = []
    for raw in payload.get("scenes", []):
        fields = dict(raw)
        cluster = dict(fields.pop("cluster"))
        cluster["members"] = tuple(SceneMember(
            **{**member, "source_refs": tuple(member.get("source_refs", ()))})
            for member in cluster.get("members", ()))
        for name in ("centroid", "source_archive_ids"):
            cluster[name] = tuple(cluster.get(name, ()))
        cluster["boundary"] = tuple(tuple(item) for item in cluster.get("boundary", ()))
        records.append(SceneRecord(cluster=SceneCluster(**cluster), **fields))
    return SceneSnapshot(account_id, owner_space, revision, tuple(records))


class _SQLStore:
    _schema_lock = Lock()

    def __init__(self, pool) -> None:
        if pool is None:
            raise ValueError("shared SQL pool is required")
        self.pool = pool
        with self._schema_lock:
            connection = pool.get_connection()
            try:
                with connection.cursor() as cursor:
                    self._lock(cursor, "ogmemory:scene:schema")
                    pool.sql_dialect.validate_connection(cursor)
                    cursor.execute(pool.sql_dialect.prepare_schema(_SCHEMA))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                pool.return_connection(connection)

    @contextmanager
    def _transaction(self, account_id: str):
        if not account_id:
            raise ValueError("account_id is required")
        connection = self.pool.get_connection()
        try:
            PoolAdapterMixin.bind_tenant(connection, account_id)
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('statement_timeout', '30000', true)")
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self.pool.return_connection(connection)

    @staticmethod
    def _lock(cursor, namespace: str, *, try_only=False) -> bool:
        digest = hashlib.sha256(namespace.encode()).digest()
        key = int.from_bytes(digest[:8], "big", signed=True)
        function = "pg_try_advisory_xact_lock" if try_only else "pg_advisory_xact_lock"
        cursor.execute(f"SELECT {function}(%s)", (key,))
        row = cursor.fetchone()
        return bool(row and row[0]) if try_only else True

    @contextmanager
    def _scope_transaction(self, account_id: str, owner_space: str, *, try_only=False):
        if not owner_space:
            raise ValueError("owner_space is required")
        with self._transaction(account_id) as cursor:
            if not self._lock(cursor, f"ogmemory:scene:{account_id}:{owner_space}", try_only=try_only):
                yield None
                return
            self._lock(cursor, f"ogmemory:scene:account:{account_id}")
            cursor.execute(
                "INSERT INTO scene_work_accounts (account_id) "
                "SELECT %s WHERE NOT EXISTS (SELECT 1 FROM scene_work_accounts WHERE account_id=%s)",
                (account_id, account_id))
            cursor.execute(
                "INSERT INTO scene_scopes (account_id, owner_space) "
                "SELECT %s, %s WHERE NOT EXISTS (SELECT 1 FROM scene_scopes WHERE account_id=%s AND owner_space=%s)",
                (account_id, owner_space, account_id, owner_space))
            yield cursor

    def _accounts(self) -> list[str]:
        connection = self.pool.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT account_id FROM scene_work_accounts ORDER BY account_id")
                return [str(row[0]) for row in cursor.fetchall()]
        finally:
            connection.rollback()
            self.pool.return_connection(connection)


class SQLSceneRepository(_SQLStore):
    """Versioned scene snapshots with tenant isolation and optimistic publication."""

    def pending_projections(self, limit: int = 20):
        emitted = 0
        for account in self._accounts():
            with self._transaction(account) as cursor:
                cursor.execute(
                    "SELECT owner_space FROM scene_scopes "
                    "WHERE account_id=%s AND lexical_revision < revision "
                    "ORDER BY updated_at LIMIT %s", (account, limit - emitted))
                owners = [row[0] for row in cursor.fetchall()]
            for owner in owners:
                with self._transaction(account) as cursor:
                    cursor.execute(
                        "SELECT revision,snapshot FROM scene_scopes WHERE account_id=%s "
                        "AND owner_space=%s AND lexical_revision < revision", (account, owner))
                    row = cursor.fetchone()
                if row:
                    yield snapshot_from_dict(account, owner, int(row[0]), _json(row[1]))
                    emitted += 1
            if emitted >= limit:
                return

    def publish_projection(self, snapshot, publish) -> bool:
        """Serialize every publisher and acknowledge only the current scope revision."""
        with self._transaction(snapshot.account_id) as cursor:
            self._lock(cursor, f"ogmemory:scene:projection:{snapshot.account_id}:{snapshot.owner_space}")
            cursor.execute(
                "SELECT revision, lexical_revision FROM scene_scopes WHERE account_id=%s AND owner_space=%s",
                (snapshot.account_id, snapshot.owner_space))
            row = cursor.fetchone()
            if row is None or row[0] != snapshot.revision:
                return False
            if row[1] >= snapshot.revision:
                return True
            publish(snapshot)
            cursor.execute(
                "UPDATE scene_scopes SET lexical_revision=%s WHERE account_id=%s "
                "AND owner_space=%s AND revision=%s",
                (snapshot.revision, snapshot.account_id, snapshot.owner_space, snapshot.revision))
            return cursor.rowcount == 1

    def load_snapshot(self, account_id: str, owner_space: str) -> SceneSnapshot:
        with self._transaction(account_id) as cursor:
            cursor.execute("SELECT revision, snapshot FROM scene_scopes WHERE account_id=%s AND owner_space=%s",
                           (account_id, owner_space))
            row = cursor.fetchone()
        return snapshot_from_dict(account_id, owner_space, int(row[0]), _json(row[1])) if row else (
            SceneSnapshot(account_id, owner_space, 0))

    def compare_and_swap(self, expected: SceneSnapshot, replacement: SceneSnapshot) -> bool:
        return self.compare_and_swap_checked(expected, replacement, lambda reader: True)

    def compare_and_swap_checked(self, expected: SceneSnapshot, replacement: SceneSnapshot, check) -> bool:
        if (expected.account_id, expected.owner_space) != (replacement.account_id, replacement.owner_space):
            raise ValueError("scene snapshot scope changed")
        if replacement.revision != expected.revision + 1:
            raise ValueError("scene snapshot revision must advance once")
        if any(scene.cluster.owner_space != expected.owner_space for scene in replacement.scenes):
            raise ValueError("scene owner scope changed")
        payload = snapshot_to_dict(replacement)
        with self._scope_transaction(expected.account_id, expected.owner_space) as cursor:
            cursor.execute("LOCK TABLE context_nodes IN SHARE MODE")
            self._check_lease(cursor, expected)
            def reader(uri, ctx):
                return SQLContextFS.read_node_with_cursor(cursor, uri, ctx)

            if not self._check_sources(cursor, expected, replacement) or not check(reader):
                return False
            self._check_lease(cursor, expected)
            cursor.execute(
                "UPDATE scene_scopes SET revision=%s, snapshot=%s, updated_at=NOW() "
                "WHERE account_id=%s AND owner_space=%s AND revision=%s",
                (replacement.revision, Json(payload), expected.account_id, expected.owner_space, expected.revision))
            if cursor.rowcount != 1:
                return False
            cursor.execute(
                "INSERT INTO scene_snapshot_history (account_id, owner_space, revision, snapshot) VALUES (%s,%s,%s,%s)",
                (replacement.account_id, replacement.owner_space, replacement.revision, Json(payload)))
        return True

    @staticmethod
    def _check_lease(cursor, snapshot):
        job = current_scene_job.get()
        if job is None:
            return
        if (job.account_id, job.owner_space) != (snapshot.account_id, snapshot.owner_space):
            raise SceneLeaseLostError("scene job scope changed")
        cursor.execute("SELECT payload, clock_timestamp() FROM scene_jobs WHERE account_id=%s AND job_id=%s FOR UPDATE",
                       (job.account_id, job.job_id))
        row = cursor.fetchone()
        current = job_from_dict(_json(row[0])) if row else None
        if (current is None or current.status != SceneJobStatus.RUNNING
                or current.lease_token != job.lease_token or current.lease_until is None
                or current.lease_until <= row[1]):
            raise SceneLeaseLostError("scene publication lost its job lease")

    @staticmethod
    def _check_sources(cursor, expected, replacement):
        previous = {scene.scene_id: scene for scene in expected.scenes}
        versions = {}
        for scene in replacement.scenes:
            if scene.status != "ACTIVE" or scene == previous.get(scene.scene_id):
                continue
            for member in scene.cluster.members:
                existing = versions.setdefault(member.context_uri, member.node_version)
                if existing != member.node_version:
                    return False
        if not versions:
            return True
        cursor.execute(
            "SELECT uri, version, status, owner_space FROM context_nodes "
            "WHERE account_id=%s AND uri=ANY(%s) ORDER BY uri FOR SHARE",
            (expected.account_id, sorted(versions)))
        rows = cursor.fetchall()
        spaces = set(expand_owner_space_aliases([expected.owner_space]))
        return len(rows) == len(versions) and all(
            row[1] == versions[row[0]] and row[2] == "ACTIVE" and row[3] in spaces for row in rows)

    def load_active(self, account_id: str, owner_space: str) -> list[PersistedScene]:
        return [self._persisted(record) for record in self.load_snapshot(account_id, owner_space).scenes
                if record.status == "ACTIVE"]

    @staticmethod
    def _persisted(record: SceneRecord) -> PersistedScene:
        c = record.cluster
        return PersistedScene(
            record.scene_id, record.version, c.owner_space, c.centroid, c.quality, len(c.members),
            members=frozenset(member.context_uri for member in c.members), basis=c.basis,
            label=c.label, summary=c.summary, source_fingerprint=c.source_fingerprint,
            source_archive_ids=c.source_archive_ids, boundary=c.boundary,
            time_start=c.time_start, time_end=c.time_end)

    def contributions(self, account_id: str, owner_spaces, candidate_uris):
        candidates = set(candidate_uris)
        contributions = []
        for owner_space in owner_spaces:
            for record in self.load_snapshot(account_id, owner_space).scenes:
                if record.status == "ACTIVE":
                    scene = self._persisted(record)
                    contributions.extend((scene, member.context_uri, member.weight)
                                         for member in record.cluster.members if member.context_uri in candidates)
        return contributions

    def list_scopes(self, account_id: str) -> list[str]:
        with self._transaction(account_id) as cursor:
            cursor.execute("SELECT owner_space FROM scene_scopes WHERE account_id=%s ORDER BY owner_space",
                           (account_id,))
            return [str(row[0]) for row in cursor.fetchall()]


class SQLSceneJobStore(_SQLStore):
    """Persist the shared queue state machine under one short scope transaction."""

    @staticmethod
    def _load(cursor, account_id, owner_space) -> InMemorySceneJobStore:
        cursor.execute("SELECT queue, clock_timestamp() FROM scene_scopes WHERE account_id=%s AND owner_space=%s",
                       (account_id, owner_space))
        row = cursor.fetchone()
        queue = InMemorySceneJobStore(clock=lambda: row[1]) if row else InMemorySceneJobStore()
        payload = _json(row[0]) if row else {}
        if payload:
            queue._scopes[(account_id, owner_space)] = payload
        state = queue._scope(account_id, owner_space)
        if state["active"]:
            cursor.execute("SELECT payload FROM scene_jobs WHERE account_id=%s AND job_id=%s",
                           (account_id, state["active"]))
            job_row = cursor.fetchone()
            if job_row:
                job = job_from_dict(_json(job_row[0]))
                queue._jobs[job.job_id] = job
        return queue

    @staticmethod
    def _save(cursor, account_id, owner_space, queue):
        state = queue._scope(account_id, owner_space)
        for job in queue._jobs.values():
            payload = Json(job_to_dict(job))
            cursor.execute(
                "UPDATE scene_jobs SET payload=%s, updated_at=NOW() WHERE account_id=%s AND job_id=%s",
                (payload, account_id, job.job_id))
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO scene_jobs (account_id,owner_space,job_id,payload) VALUES (%s,%s,%s,%s)",
                    (account_id, owner_space, job.job_id, payload))
        cursor.execute(
            "UPDATE scene_scopes SET queue=%s, due_at=%s, updated_at=NOW() WHERE account_id=%s AND owner_space=%s",
            (Json(state), queue.due_at(account_id, owner_space), account_id, owner_space))
    def enqueue(self, account_id: str, owner_space: str, node_uris: Iterable[str], *, notice_id=None, **kwargs):
        if notice_id is not None and (not isinstance(notice_id, str) or not 1 <= len(notice_id) <= 512):
            raise ValueError("notice_id must contain 1..512 characters")
        with self._scope_transaction(account_id, owner_space) as cursor:
            queue = self._load(cursor, account_id, owner_space)
            if notice_id is not None:
                cursor.execute(
                    "SELECT 1 FROM scene_write_notices WHERE account_id=%s AND owner_space=%s AND notice_id=%s",
                    (account_id, owner_space, notice_id))
                if cursor.fetchone():
                    return queue._jobs.get(queue._scope(account_id, owner_space)["active"])
            result = queue.enqueue(account_id, owner_space, node_uris, **kwargs)
            self._save(cursor, account_id, owner_space, queue)
            if notice_id is not None:
                cursor.execute("INSERT INTO scene_write_notices (account_id,owner_space,notice_id) VALUES (%s,%s,%s)",
                               (account_id, owner_space, notice_id))
            return result

    def claim(self, worker_id: str, *, lease_seconds: float = 300) -> SceneJob | None:
        due = []
        for account_id in self._accounts():
            with self._transaction(account_id) as cursor:
                cursor.execute(
                    "SELECT owner_space, due_at FROM scene_scopes WHERE account_id=%s AND due_at<=NOW() "
                    "ORDER BY due_at, owner_space LIMIT 32", (account_id,))
                due.extend((row[1], account_id, str(row[0])) for row in cursor.fetchall())
        for _, account_id, owner_space in sorted(due):
            with self._scope_transaction(account_id, owner_space, try_only=True) as cursor:
                if cursor is None:
                    continue
                queue = self._load(cursor, account_id, owner_space)
                job = queue.claim(worker_id, lease_seconds=lease_seconds)
                self._save(cursor, account_id, owner_space, queue)
                if job:
                    return job
        return None

    def _change(self, job: SceneJob, operation: str, **kwargs) -> SceneJob:
        with self._scope_transaction(job.account_id, job.owner_space) as cursor:
            queue = self._load(cursor, job.account_id, job.owner_space)
            if job.job_id not in queue._jobs:
                raise SceneLeaseLostError("scene job was superseded")
            result = getattr(queue, operation)(job, **kwargs)
            self._save(cursor, job.account_id, job.owner_space, queue)
            return result

    def renew(self, job: SceneJob, *, lease_seconds: float = 300):
        return self._change(job, "renew", lease_seconds=lease_seconds)

    def mark_succeeded(self, job: SceneJob, *, reason="", remaining_uris=(), min_evidence_nodes=3,
                       deferred_uris=()):
        return self._change(job, "mark_succeeded", reason=reason, remaining_uris=remaining_uris,
                            min_evidence_nodes=min_evidence_nodes, deferred_uris=deferred_uris)

    def mark_waiting(self, job: SceneJob, *, reason="", remaining_uris=None, deferred_uris=()):
        return self._change(job, "mark_waiting", reason=reason, remaining_uris=remaining_uris,
                            deferred_uris=deferred_uris)

    def mark_retry(self, job: SceneJob, *, reason, delay_seconds):
        return self._change(job, "mark_retry", reason=reason, delay_seconds=delay_seconds)

    def mark_failed(self, job: SceneJob, *, reason):
        return self._change(job, "mark_failed", reason=reason)

    def flush(self, account_id: str, owner_space: str) -> SceneJob | None:
        with self._scope_transaction(account_id, owner_space) as cursor:
            queue = self._load(cursor, account_id, owner_space)
            result = queue.flush(account_id, owner_space)
            self._save(cursor, account_id, owner_space, queue)
            return result

    def get(self, job_id: str, *, account_id: str | None = None) -> SceneJob:
        for account in [account_id] if account_id else self._accounts():
            with self._transaction(account) as cursor:
                cursor.execute("SELECT payload FROM scene_jobs WHERE account_id=%s AND job_id=%s",
                               (account, job_id))
                row = cursor.fetchone()
                if row:
                    return job_from_dict(_json(row[0]))
        raise KeyError(job_id)

    def summary(self, account_id: str, owner_space: str) -> dict:
        with self._transaction(account_id) as cursor:
            queue = self._load(cursor, account_id, owner_space)
            result = queue.summary(account_id, owner_space)
            last_job = queue._scope(account_id, owner_space).get("last_job")
            if last_job:
                cursor.execute("SELECT payload FROM scene_jobs WHERE account_id=%s AND job_id=%s",
                               (account_id, last_job))
                row = cursor.fetchone()
                result["last_job"] = job_to_dict(job_from_dict(_json(row[0]))) if row else None
            cursor.execute(
                "SELECT payload FROM scene_jobs WHERE account_id=%s AND owner_space=%s "
                "ORDER BY updated_at DESC, job_id LIMIT 10", (account_id, owner_space))
            result["recent_jobs"] = [job_to_dict(job_from_dict(_json(row[0]))) for row in cursor.fetchall()]
            return result


class SQLSceneConfigProvider(_SQLStore):
    """Append-only configuration revisions, merged and published with scope CAS."""

    def __init__(self, pool, default: SceneConfig | None = None):
        self.default = default or SceneConfig()
        self.default.validate()
        self._listeners = []
        super().__init__(pool)

    def _read(self, cursor, account_id, owner_space) -> SceneConfigSnapshot:
        cursor.execute(
            "SELECT revision,payload,source FROM scene_config_revisions WHERE account_id=%s AND owner_space=%s "
            "ORDER BY revision DESC LIMIT 1", (account_id, owner_space))
        row = cursor.fetchone()
        return (SceneConfigSnapshot(self.default.merged(_json(row[1])), int(row[0]), str(row[2]))
                if row else SceneConfigSnapshot(self.default, 0, "system"))

    def get_effective(self, account_id: str, owner_space: str) -> SceneConfigSnapshot:
        with self._transaction(account_id) as cursor:
            return self._read(cursor, account_id, owner_space)

    def _insert(self, cursor, account_id, owner_space, snapshot):
        cursor.execute(
            "INSERT INTO scene_config_revisions (account_id,owner_space,revision,payload,source) VALUES (%s,%s,%s,%s,%s)",
            (account_id, owner_space, snapshot.revision, Json(asdict(snapshot.config)), snapshot.source))

    def publish(self, account_id: str, owner_space: str, expected_revision: int,
                overrides: Mapping[str, object], *, source="owner_space") -> SceneConfigSnapshot:
        with self._scope_transaction(account_id, owner_space) as cursor:
            current = self._read(cursor, account_id, owner_space)
            if current.revision != expected_revision:
                raise ConfigRevisionConflict("scene configuration revision changed")
            result = SceneConfigSnapshot(current.config.merged(overrides), current.revision + 1, source)
            self._insert(cursor, account_id, owner_space, result)
        self._notify(account_id, owner_space, result)
        return result

    def rollback(self, account_id: str, owner_space: str, target_revision: int) -> SceneConfigSnapshot:
        with self._scope_transaction(account_id, owner_space) as cursor:
            current = self._read(cursor, account_id, owner_space)
            target = self.default
            if target_revision != 0:
                cursor.execute(
                    "SELECT payload FROM scene_config_revisions WHERE account_id=%s AND owner_space=%s AND revision=%s",
                    (account_id, owner_space, target_revision))
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("unknown scene configuration revision")
                target = self.default.merged(_json(row[0]))
            result = SceneConfigSnapshot(target, current.revision + 1, "rollback")
            self._insert(cursor, account_id, owner_space, result)
        self._notify(account_id, owner_space, result)
        return result

    def subscribe(self, listener) -> None:
        self._listeners.append(listener)

    def _notify(self, account_id, owner_space, result):
        for listener in tuple(self._listeners):
            try:
                listener(account_id, owner_space, result)
            except Exception:
                logger.exception("Scene configuration notification failed at revision %s", result.revision)

    def export(self, account_id, owner_space) -> dict:
        snapshot = self.get_effective(account_id, owner_space)
        return {"revision": snapshot.revision, "source": snapshot.source, "values": asdict(snapshot.config)}
