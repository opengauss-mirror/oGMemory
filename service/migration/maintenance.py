"""Small durable state machine for migration maintenance coordination."""
import sqlite3
import time
import uuid
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

from service.migration.file_lock import exclusive_file


class OperationKind(Enum):
    READ = auto()
    RUNTIME_WRITE = auto()
    CONTENT_WRITE = auto()


_PENDING_ACK = "(participants.stage IS NOT tasks.phase OR participants.error IS NOT NULL)"


class MaintenanceError(RuntimeError):
    """Business operation is blocked by migration maintenance."""


class MaintenanceStore:
    """Persist the maintenance gate shared by service processes."""
    def __init__(self, path, instance_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS instance (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)")
            row = db.execute("SELECT identity FROM instance").fetchone()
            if row is None:
                db.execute("INSERT INTO instance VALUES(1,?)", (instance_id,))
            elif row[0] != instance_id:
                raise MaintenanceError("workspace belongs to another instance")
            db.execute("CREATE TABLE IF NOT EXISTS operations (token TEXT PRIMARY KEY, writing INTEGER NOT NULL, content INTEGER NOT NULL, participant TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, mode TEXT NOT NULL, phase TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS gate (id INTEGER PRIMARY KEY CHECK(id=1), task TEXT)")
            db.execute("INSERT OR IGNORE INTO gate(id,task) VALUES(1,NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS participants (id TEXT PRIMARY KEY, task TEXT, stage TEXT, error TEXT)")

    @classmethod
    def for_workspace(cls, directory):
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return cls(root / "maintenance.sqlite", str(root))

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

    def _active(self, db):
        return db.execute("SELECT tasks.* FROM gate JOIN tasks ON tasks.id=gate.task").fetchone()

    def _blocks(self, db, kind):
        task = self._active(db)
        return task is not None and (task["mode"] == "import" or kind is OperationKind.CONTENT_WRITE)

    def admit(self, *, participant_id: str, kind: OperationKind = OperationKind.READ):
        with self._transaction() as db:
            if db.execute(
                "SELECT 1 FROM participants WHERE id=?", (participant_id,),
            ).fetchone() is None:
                raise MaintenanceError("participant is not registered")
            if self._blocks(db, kind):
                raise MaintenanceError("instance is in migration maintenance")
            token = uuid.uuid4().hex
            db.execute(
                "INSERT INTO operations(token,writing,content,participant) VALUES(?,?,?,?)",
                (token, int(kind is not OperationKind.READ), int(kind is OperationKind.CONTENT_WRITE), participant_id),
            )
            return token

    def upgrade_to_runtime_write(self, token):
        with self._transaction() as db:
            if self._blocks(db, OperationKind.RUNTIME_WRITE):
                raise MaintenanceError("instance is in migration maintenance")
            if not db.execute("UPDATE operations SET writing=1 WHERE token=?", (token,)).rowcount:
                raise MaintenanceError("operation has ended")

    def finish_operation(self, token):
        with self._transaction() as db:
            db.execute("DELETE FROM operations WHERE token=?", (token,))

    def enter(self, task_id, *, mode):
        if mode not in {"export", "import"}:
            raise ValueError("invalid maintenance mode")
        with self._transaction() as db:
            if self._active(db) is not None:
                raise MaintenanceError("another migration owns maintenance")
            db.execute("INSERT INTO tasks(id,mode,phase) VALUES(?,?,?)", (task_id, mode, "drain"))
            db.execute("UPDATE gate SET task=? WHERE id=1", (task_id,))
            if mode == "import":
                db.execute("UPDATE participants SET task=?,stage=NULL,error=NULL", (task_id,))

    def status(self, task_id=None):
        with self._transaction() as db:
            task = self._active(db)
            if task is not None and (task_id is None or task["id"] == task_id):
                return dict(task)
            return None

    def _owned(self, db, task_id):
        task = self._active(db)
        if task is None or task["id"] != task_id:
            raise MaintenanceError("task does not own maintenance")
        return task

    @staticmethod
    def _drained(db, mode):
        query = "SELECT 1 FROM operations"
        if mode == "export":
            query += " WHERE writing=1 AND content=1"
        return db.execute(query + " LIMIT 1").fetchone() is None

    def wait_drained(self, task_id, timeout):
        """Wait for in-flight operations and the target's current stage acknowledgements."""
        deadline = time.monotonic() + timeout
        while True:
            with self._transaction() as db:
                task = self._owned(db, task_id)
                if task["mode"] == "export":
                    ready = self._drained(db, "export")
                else:
                    ready = self._reap_and_check_import_ready(db, task_id)
                if ready:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("maintenance has not drained")
            time.sleep(min(.05, remaining))

    def reopen(self, task_id):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task["mode"] == "import" and (
                task["phase"] != "refresh"
                or not self._reap_and_check_import_ready(db, task_id)
            ):
                raise MaintenanceError("target has not finished refreshing")
            db.execute("UPDATE participants SET task=NULL,stage=NULL,error=NULL WHERE task=?", (task_id,))
            db.execute("UPDATE gate SET task=NULL WHERE id=1")
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    def participant_path(self, participant):
        return self.path.with_name(f"{self.path.stem}.participant-{participant}.lock")

    def register_participant(self, participant):
        with self._transaction() as db:
            task = self._active(db)
            task_id = task["id"] if task is not None and task["mode"] == "import" else None
            db.execute("INSERT INTO participants(id,task) VALUES(?,?)", (participant, task_id))

    def remove_participant(self, participant):
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM operations WHERE participant=? LIMIT 1", (participant,)).fetchone():
                raise MaintenanceError("participant still has active operations")
            db.execute("DELETE FROM participants WHERE id=?", (participant,))

    def lifecycle_request(self, participant):
        with self._transaction() as db:
            row = db.execute(
                "SELECT tasks.id,tasks.phase,participants.stage, "
                f"{_PENDING_ACK} AS needs_ack, "
                "EXISTS(SELECT 1 FROM operations WHERE participant=participants.id) AS busy "
                "FROM gate JOIN tasks ON tasks.id=gate.task "
                "JOIN participants ON participants.task=tasks.id "
                "WHERE tasks.mode='import' AND participants.id=?",
                (participant,),
            ).fetchone()
            return dict(row) if row else None

    def acknowledge(self, participant, task_id, stage, *, error=None):
        with self._transaction() as db:
            db.execute(
                "UPDATE participants SET stage=CASE WHEN ? IS NULL THEN ? ELSE stage END,error=? "
                "WHERE id=? AND task=?",
                (error, stage, error, participant, task_id),
            )

    def _reap_and_check_import_ready(self, db, task_id):
        """Remove stopped participants' read permits before checking stage readiness."""
        pending = db.execute(
            "SELECT participants.id FROM participants JOIN tasks ON participants.task=tasks.id "
            f"WHERE tasks.id=? AND {_PENDING_ACK}",
            (task_id,),
        ).fetchall()
        ready = True
        for row in pending:
            participant = row["id"]
            try:
                with exclusive_file(self.participant_path(participant)):
                    # A dead process cannot have an in-flight read. Its writes
                    # may have reached the backend and still need reconciliation.
                    db.execute("DELETE FROM operations WHERE participant=? AND writing=0", (participant,))
                    db.execute(
                        "DELETE FROM participants WHERE id=? "
                        "AND NOT EXISTS (SELECT 1 FROM operations WHERE participant=?)",
                        (participant, participant),
                    )
            except BlockingIOError:
                ready = False
        return ready and self._drained(db, "import")

    def request_refresh(self, task_id):
        with self._transaction() as db:
            task = self._owned(db, task_id)
            if task["mode"] != "import" or task["phase"] != "drain":
                raise MaintenanceError("task is not in target maintenance drain")
            if not self._reap_and_check_import_ready(db, task_id):
                raise MaintenanceError("target has not finished draining")
            db.execute("UPDATE tasks SET phase='refresh' WHERE id=?", (task_id,))

    def execution_lock(self):
        """Acquire the executor lock or immediately raise BlockingIOError if held."""
        return exclusive_file(self.path.with_suffix(".executor.lock"))
