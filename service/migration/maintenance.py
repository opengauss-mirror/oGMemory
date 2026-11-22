"""Durable admission control for processes sharing one local instance workspace.

No timeout removes uncertain operations. Storage callers must acknowledge completion
only after their backend has finished; a crashed writer requires reconciliation.
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


from core.errors import MaintenanceError


class MaintenanceStore:
    def __init__(self, path, instance_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS instance (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO instance VALUES(1,?)", (instance_id,))
            if db.execute("SELECT identity FROM instance").fetchone()[0] != instance_id:
                raise MaintenanceError("workspace belongs to another instance")
            db.execute("CREATE TABLE IF NOT EXISTS operations (token TEXT PRIMARY KEY, writing INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, mode TEXT NOT NULL, phase TEXT NOT NULL, resources TEXT NOT NULL, backup TEXT, modified INTEGER NOT NULL DEFAULT 0, result TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS receipts (digest TEXT PRIMARY KEY, target TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS gate (id INTEGER PRIMARY KEY CHECK(id=1), task TEXT)")
            db.execute("INSERT OR IGNORE INTO gate(id,task) VALUES(1,NULL)")
            if 'content' not in {row[1] for row in db.execute('PRAGMA table_info(operations)')}:
                db.execute('ALTER TABLE operations ADD COLUMN content INTEGER NOT NULL DEFAULT 1')
            if 'participant' not in {row[1] for row in db.execute('PRAGMA table_info(operations)')}:
                db.execute('ALTER TABLE operations ADD COLUMN participant TEXT')
            if 'lifecycle' not in {row[1] for row in db.execute('PRAGMA table_info(gate)')}:
                db.execute("ALTER TABLE gate ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'drain'")
            if 'attempt' not in {row[1] for row in db.execute('PRAGMA table_info(gate)')}:
                db.execute('ALTER TABLE gate ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0')
            db.execute('CREATE TABLE IF NOT EXISTS participants '
                       '(id TEXT PRIMARY KEY, task TEXT, stage TEXT, error TEXT)')

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _active(db):
        return db.execute("SELECT tasks.* FROM gate JOIN tasks ON tasks.id=gate.task").fetchone()

    def admit(self, *, write=False, participant=None, content=True):
        with self._transaction() as db:
            task = self._active(db)
            if task and ((write and content) or task["mode"] == "import"):
                raise MaintenanceError("instance is in migration maintenance")
            if participant is not None and not db.execute('SELECT 1 FROM participants WHERE id=?', (participant,)).fetchone():
                raise MaintenanceError('business process is not registered')
            token = uuid.uuid4().hex
            db.execute("INSERT INTO operations(token,writing,participant,content) VALUES(?,?,?,?)",
                       (token, int(write), participant, int(write and content)))
            return token

    def upgrade(self, token, *, content=True):
        with self._transaction() as db:
            task = self._active(db)
            if task and (content or task["mode"] == "import"):
                raise MaintenanceError("writes are paused for migration")
            if db.execute("UPDATE operations SET writing=1,content=MAX(content,?) WHERE token=?", (int(content), token)).rowcount != 1:
                raise MaintenanceError("operation is no longer active")

    def finish_operation(self, token):
        with self._transaction() as db:
            if db.execute("DELETE FROM operations WHERE token=?", (token,)).rowcount != 1:
                raise MaintenanceError("operation is no longer active")

    def enter(self, task_id, *, mode, resources):
        if mode not in {"import", "export"}:
            raise ValueError("invalid migration mode")
        encoded = json.dumps(resources, sort_keys=True, allow_nan=False)
        with self._transaction() as db:
            prior = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if prior:
                if prior["mode"] != mode or prior["resources"] != encoded:
                    raise MaintenanceError("task identity does not match its original resources")
                return dict(prior)
            if self._active(db):
                raise MaintenanceError("another migration owns maintenance")
            db.execute("INSERT INTO tasks(id,mode,phase,resources) VALUES(?,?,?,?)", (task_id, mode, "draining", encoded))
            db.execute("UPDATE gate SET task=?,lifecycle='drain'", (task_id,))
            db.execute("UPDATE participants SET task=?,stage=NULL,error=NULL", (task_id,))
        return self.status(task_id)

    def status(self, task_id=None):
        with self._transaction() as db:
            row = (db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
                   if task_id else self._active(db))
            return dict(row) if row else None

    def _owned(self, db, task_id):
        task = self._active(db)
        if not task or task["id"] != task_id:
            raise MaintenanceError("task does not own maintenance")
        return task

    def wait_drained(self, task_id, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.retire_stopped_participants()
            with self._transaction() as db:
                task = self._owned(db, task_id)
                predicate = " WHERE writing=1 AND content=1" if task["mode"] == "export" else ""
                if not db.execute("SELECT 1 FROM operations" + predicate + " LIMIT 1").fetchone():
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError("operations have not drained; maintenance remains active")
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    def backed_up(self, task_id, manifest):
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task["mode"] != "import" or task["phase"] != "draining" or task["modified"]:
                raise MaintenanceError("backup cannot replace an existing recovery point")
            if db.execute("SELECT 1 FROM operations LIMIT 1").fetchone():
                raise MaintenanceError("operations have not drained")
            if not self._participants_ready(db, task_id, "drain"):
                raise MaintenanceError('business processes have not quiesced')
            db.execute("UPDATE tasks SET backup=?,phase='backed_up' WHERE id=?", (encoded, task_id))

    def begin_changes(self, task_id, resources):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            self._match_resources(task, resources)
            if task["phase"] != "backed_up" or not task["backup"]:
                raise MaintenanceError("verified backup is required before changing data")
            db.execute("UPDATE tasks SET modified=1,phase='modifying' WHERE id=?", (task_id,))

    def bind_resources(self, task_id, initial, resolved):
        """Resolve native resources after drain, before creating the backup."""
        with self._transaction() as db:
            task = self._owned(db, task_id)
            self._match_resources(task, initial)
            if task['phase'] != 'draining' or task['backup'] or task['modified']:
                raise MaintenanceError('recovery resources are already fixed')
            if db.execute('SELECT 1 FROM operations LIMIT 1').fetchone():
                raise MaintenanceError('cannot bind resources before drain')
            db.execute('UPDATE tasks SET resources=? WHERE id=?',
                       (json.dumps(resolved, sort_keys=True, allow_nan=False), task_id))

    @contextmanager
    def mutation(self, task_id, *, settled=lambda: False):
        """Keep an interrupted migration write visible to recovery's drain."""
        token = uuid.uuid4().hex
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task['phase'] not in {'modifying', 'recovering'}:
                raise MaintenanceError('migration write requires a verified backup')
            db.execute('INSERT INTO operations(token,writing) VALUES(?,1)', (token,))
        try:
            yield
        except BaseException:
            if settled():
                self.finish_operation(token)
            raise
        else:
            self.finish_operation(token)

    @staticmethod
    def _match_resources(task, resources):
        if json.loads(task["resources"]) != resources:
            raise MaintenanceError("target resources differ from the recovery manifest")

    def begin_recovery(self, task_id, resources):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            self._match_resources(task, resources)
            if task["phase"] not in {"modifying", "recovering"} or not task["backup"]:
                raise MaintenanceError("task cannot be rolled back")
            db.execute("UPDATE tasks SET phase='recovering' WHERE id=?", (task_id,))
            return json.loads(task["backup"])

    def applied(self, digest):
        with self._transaction() as db:
            row = db.execute("SELECT target FROM receipts WHERE digest=?", (digest,)).fetchone()
            return row[0] if row else None

    def verified(self, task_id, *, restored=False, result=None, receipts=()):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            expected = "recovering" if restored else "modifying"
            if task["phase"] != expected:
                raise MaintenanceError("invalid verification phase")
            for digest, target in receipts:
                if restored:
                    raise MaintenanceError("rollback cannot publish import receipts")
                prior = db.execute("SELECT target FROM receipts WHERE digest=?", (digest,)).fetchone()
                if prior and prior[0] != target:
                    raise MaintenanceError("source record already maps to another identity")
                db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?)", (digest, target))
            db.execute("UPDATE tasks SET phase=?,result=? WHERE id=?",
                       ("restored" if restored else "succeeded", json.dumps(result), task_id))

    def reopen(self, task_id):
        """Call only after all clients have discarded stale data and index caches."""
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task["modified"] and task["phase"] not in {"succeeded", "restored"}:
                raise MaintenanceError("modified data must be verified before reopening")
            if task['mode'] == 'import' and db.execute('SELECT 1 FROM operations LIMIT 1').fetchone():
                raise MaintenanceError('cannot reopen with unfinished operations')
            if task['mode'] == 'import' and db.execute('SELECT 1 FROM participants LIMIT 1').fetchone():
                if (db.execute('SELECT lifecycle FROM gate').fetchone()[0] != 'refresh'
                        or not self._participants_ready(db, task_id, "refresh")):
                    raise MaintenanceError('all participants must refresh before reopening')
            if not task["modified"]:
                db.execute("UPDATE tasks SET phase='unchanged' WHERE id=?", (task_id,))
            db.execute("UPDATE gate SET task=NULL")

    def owned_operation(self, task_id, participant):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            lifecycle = db.execute('SELECT lifecycle FROM gate').fetchone()[0]
            if task['mode'] != 'import' or (lifecycle == 'drain' and task['phase'] != 'draining'):
                raise MaintenanceError('lifecycle writes are not allowed during data restoration')
            if not db.execute('SELECT 1 FROM participants WHERE id=? AND task=?', (participant, task_id)).fetchone():
                raise MaintenanceError('maintenance participant is not registered')
            token = uuid.uuid4().hex
            db.execute('INSERT INTO operations(token,writing,participant) VALUES(?,1,?)', (token, participant))
            return token

    def participant_path(self, participant):
        if not isinstance(participant, str) or len(participant) != 32 or any(c not in '0123456789abcdef' for c in participant):
            raise MaintenanceError('invalid participant identity')
        return self.path.parent / (self.path.name + '.participants') / (participant + '.lock')

    def register_participant(self, participant):
        self.participant_path(participant)
        with self._transaction() as db:
            task = self._active(db)
            # A newly constructed process has no prior business state to flush.
            # Once backup starts it must stay dormant until the refresh barrier.
            stage = 'drain' if task and task['mode'] == 'import' and task['phase'] != 'draining' else None
            db.execute('INSERT INTO participants VALUES(?,?,?,NULL)', (participant, task['id'] if task else None, stage))

    def remove_participant(self, participant):
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM operations WHERE participant=? LIMIT 1', (participant,)).fetchone():
                raise MaintenanceError('participant has unfinished backend operations')
            db.execute('DELETE FROM participants WHERE id=?', (participant,))

    def lifecycle_request(self, participant):
        with self._transaction() as db:
            row = db.execute('SELECT p.task,p.stage,g.lifecycle,g.attempt,t.mode FROM participants p '
                             'JOIN gate g ON g.task=p.task JOIN tasks t ON t.id=p.task '
                             'WHERE p.id=?', (participant,)).fetchone()
            return dict(row) if row else None

    def acknowledge(self, participant, task_id, stage, *, error=None):
        with self._transaction() as db:
            self._owned(db, task_id)
            if db.execute('SELECT lifecycle FROM gate').fetchone()[0] != stage:
                raise MaintenanceError('lifecycle request changed')
            if db.execute('SELECT 1 FROM operations WHERE participant=?', (participant,)).fetchone():
                raise MaintenanceError('participant operations have not finished')
            if db.execute('UPDATE participants SET stage=?,error=? WHERE id=? AND task=?',
                          (stage if error is None else None, error, participant, task_id)).rowcount != 1:
                raise MaintenanceError('participant is not registered for this task')

    def request_refresh(self, task_id):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task['modified'] and task['phase'] not in {'succeeded', 'restored'}:
                raise MaintenanceError('unverified data cannot be exposed to refreshed clients')
            if db.execute('SELECT lifecycle FROM gate').fetchone()[0] != 'refresh':
                if (db.execute('SELECT 1 FROM operations LIMIT 1').fetchone()
                        or not self._participants_ready(db, task_id, "drain")):
                    raise MaintenanceError('participants have not drained')
                db.execute("UPDATE gate SET lifecycle='refresh'")
                db.execute('UPDATE participants SET stage=NULL,error=NULL WHERE task=?', (task_id,))
            self._retry_participants(db, task_id)

    @staticmethod
    def _retry_participants(db, task_id):
        if db.execute('SELECT 1 FROM operations LIMIT 1').fetchone():
            raise MaintenanceError('cannot retry lifecycle with unfinished operations')
        db.execute('UPDATE gate SET attempt=attempt+1')
        db.execute('UPDATE participants SET error=NULL WHERE task=?', (task_id,))

    def retry_participants(self, task_id):
        """Explicit recovery may retry settled callbacks, never uncertain writes."""
        self.retire_stopped_participants()
        with self._transaction() as db:
            self._owned(db, task_id)
            self._retry_participants(db, task_id)

    def participants_ready(self, task_id, stage):
        with self._transaction() as db:
            self._owned(db, task_id)
            if db.execute('SELECT lifecycle FROM gate').fetchone()[0] != stage:
                return False
            return self._participants_ready(db, task_id, stage)

    @staticmethod
    def _participants_ready(db, task_id, stage):
        return not db.execute('SELECT 1 FROM participants WHERE task=? '
                              'AND (stage IS NULL OR stage<>? OR error IS NOT NULL) LIMIT 1',
                              (task_id, stage)).fetchone()

    def retire_stopped_participants(self):
        from service.migration.file_lock import exclusive_file

        with self._transaction() as db:
            participants = [row[0] for row in db.execute('SELECT id FROM participants')]
        for participant in participants:
            try:
                with exclusive_file(self.participant_path(participant)):
                    # The process lease proves no reader remains. Writes may
                    # still complete remotely and must retain their permits.
                    with self._transaction() as db:
                        db.execute('DELETE FROM operations WHERE participant=? AND writing=0', (participant,))
                    self.remove_participant(participant)
            except OSError:
                pass  # A live process still holds its lease; never use heartbeat expiry.

    @contextmanager
    def execution(self):
        """Serialize task execution and recovery, including duplicate task IDs."""
        from contextlib import ExitStack

        from service.migration.file_lock import exclusive_file

        with ExitStack() as stack:
            try:
                stack.enter_context(exclusive_file(self.path.with_suffix('.executor.lock')))
            except OSError as exc:
                raise MaintenanceError('another migration executor is running') from exc
            yield
