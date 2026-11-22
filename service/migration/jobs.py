"""Durable migration jobs shared by the normal server's local worker processes."""
import hashlib
import json
import logging
import os
import threading
import time
import uuid

from service.migration.durability import sync_directory
from service.migration.file_lock import exclusive_file
from service.migration.maintenance import MaintenanceError
from service.migration.package import PackageError

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024**3


class InstanceJobs:
    def __init__(self, migration):
        self.migration = migration
        self.gate = migration.gate
        self.root = migration.workspace / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.gate._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS migration_jobs(id TEXT PRIMARY KEY, body TEXT NOT NULL)')

    def _save(self, job):
        job['updated_at'] = time.time()
        with self.gate._transaction() as db:
            db.execute('INSERT OR REPLACE INTO migration_jobs VALUES(?,?)', (job['id'], json.dumps(job)))

    def get(self, identity):
        identity = str(uuid.UUID(identity))
        with self.gate._transaction() as db:
            row = db.execute('SELECT body FROM migration_jobs WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise FileNotFoundError('migration task does not exist')
        return json.loads(row[0])

    def recent(self):
        # A free OS lock proves the task executor has exited, not that remote
        # mutations have completed. Only an unchanged export can reopen automatically.
        try:
            with exclusive_file(self.root / 'execution.lock'), self.gate.execution():
                with self.gate._transaction() as db:
                    for row in db.execute('SELECT id,body FROM migration_jobs').fetchall():
                        job = json.loads(row['body'])
                        if job['status'] == 'running':
                            job.update(status='failed', phase='interrupted', error=(
                                '导出进程已退出，请重新导出。' if job['kind'] == 'export' else
                                '任务执行进程已退出，请检查维护状态并恢复。'))
                            db.execute('UPDATE migration_jobs SET body=? WHERE id=?', (json.dumps(job), row['id']))
                    active = self.gate._active(db)
                    if active and active['mode'] == 'export' and not active['modified']:
                        # Both executor leases are free. Export never restores data;
                        # abandon its snapshot without deleting any writer permits.
                        db.execute("UPDATE tasks SET phase='unchanged' WHERE id=?", (active['id'],))
                        db.execute('UPDATE gate SET task=NULL')
        except (OSError, MaintenanceError):
            pass
        with self.gate._transaction() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT body FROM migration_jobs ORDER BY json_extract(body,'$.created_at') DESC LIMIT 20")]

    def artifact(self, identity):
        path = self.root / f'upload-{uuid.UUID(identity)}.tar.gz'
        if not path.is_file():
            raise FileNotFoundError('uploaded package does not exist')
        return path

    def upload(self, stream):
        identity = str(uuid.uuid4())
        path = self.root / f'upload-{identity}.tar.gz'
        size = 0
        try:
            with path.open('xb') as output:
                os.chmod(path, 0o600)
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise PackageError('migration package exceeds upload limit')
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            sync_directory(path.parent)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {'artifact_id': identity, 'size': size}

    def _existing(self, identity, kind, artifact, recover_task):
        try:
            previous = self.get(identity)
        except FileNotFoundError:
            return None
        if (previous['kind'], previous.get('artifact_id'), previous.get('recover_task')) != (kind, artifact, recover_task):
            raise PackageError('request identity is already bound to another operation')
        return previous

    def submit(self, kind, *, identity, artifact=None, recover_task=None, authorize=lambda: None):
        identity = str(uuid.UUID(identity))
        if kind not in {'export', 'preflight', 'import', 'recover'}:
            raise PackageError('unknown migration operation')
        source = self.artifact(artifact) if kind in {'preflight', 'import'} else None
        if kind == 'recover':
            original = self.get(recover_task)
            if original['kind'] != 'import':
                raise PackageError('only an import task can be recovered')
            artifact = original['artifact_id']
            source = self.artifact(artifact)
        previous = self._existing(identity, kind, artifact, recover_task)
        if previous:
            return previous
        lease = exclusive_file(self.root / 'execution.lock')
        try:
            lease.__enter__()
        except OSError as exc:
            raise MaintenanceError('another migration task is running') from exc
        job = dict(id=identity, kind=kind, artifact_id=artifact, recover_task=recover_task,
                   status='running', phase='queued', created_at=time.time())
        try:
            previous = self._existing(identity, kind, artifact, recover_task)
            if previous:
                lease.__exit__(None, None, None)
                return previous
            self._save(job)
            thread = threading.Thread(target=self._run, args=(job, source, authorize, lease), daemon=True)
            thread.start()
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        return dict(job)

    def _run(self, job, source, authorize, lease):
        authorized_write = False
        def progress(**fields):
            nonlocal authorized_write
            if fields.get('phase') == 'merging' and not authorized_write:
                authorize()
                authorized_write = True
            if fields.get('phase') == 'draining':
                job['maintenance_started_at'] = time.time()
            job.update(fields)
            self._save(job)
        try:
            authorize()
            kind, task = job['kind'], job['id']
            if kind == 'export':
                result = self.migration.export(self.root / f'export-{task}.tar.gz', task=task, progress=progress)
            elif kind == 'preflight':
                progress(phase='validating')
                result = {**self.migration.preview(source), 'artifact_id': job['artifact_id']}
                with source.open('rb') as stream:
                    result['archive_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            elif kind == 'import':
                result = self.migration.merge(source, task=task, progress=progress)
            else:
                result = self.migration.recover(source, task=job['recover_task'], progress=progress)
            if kind in {'import', 'recover'}:
                result = {'outcome': result['phase'], **(json.loads(result['result'] or 'null') or {})}
            job.update(status='succeeded', phase='complete', result=result)
            self._save(job)
        except Exception as exc:
            from service.migration.diagnostics import _failure_diagnostic
            try:
                diagnostic = _failure_diagnostic(exc, self.migration.cfg)
            except Exception:
                diagnostic = 'Migration failed; diagnostic unavailable.'
            logger.error('migration job_id=%s kind=%s phase=%s failed: %s',
                         job['id'], job['kind'], job['phase'], diagnostic)
            job.update(status='failed', error=diagnostic)
            self._save(job)
        finally:
            lease.__exit__(None, None, None)
