"""Cross-process participant lifecycle for maintenance coordination."""
import threading
import uuid
from contextlib import ExitStack

from service.migration.admission import Admission
from service.migration.file_lock import exclusive_file
from service.migration.maintenance import MaintenanceError, OperationKind


class Participant:
    def __init__(self, store, *, pause, refresh, resume, interval=.1):
        self.store = store
        self.id = uuid.uuid4().hex
        self.admission = Admission(store, self.id)
        self.pause, self.refresh, self.resume = pause, refresh, resume
        self.interval = interval
        self.stop = threading.Event()
        self._lease = exclusive_file(store.participant_path(self.id))
        self._lease.__enter__()
        try:
            store.register_participant(self.id)
        except BaseException:
            self._lease.__exit__(None, None, None)
            raise
        self._thread = None
        self._paused = False

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._watch, daemon=True, name="migration-participant")
            self._thread.start()

    def start_service(self):
        with ExitStack() as stack:
            try:
                stack.enter_context(self.admission.scope(OperationKind.RUNTIME_WRITE))
            except MaintenanceError:
                active = self.store.status()
                if active is None or active["mode"] != "import":
                    raise
                self._paused = True
            else:
                self._resume()
        self.start()

    def _resume(self):
        self._paused = self.resume() is False

    def _watch(self):
        while not self.stop.wait(self.interval):
            try:
                self.step()
            except Exception:
                # Stage failures remain in the gate until the callback succeeds.
                continue

    def step(self):
        request = self.store.lifecycle_request(self.id)
        if request is None:
            if self._paused:
                with self.admission.scope(OperationKind.RUNTIME_WRITE):
                    self._resume()
            return
        task_id = request["id"]
        stage = request["phase"]
        if request["busy"] or not request["needs_ack"]:
            return
        self._paused = True
        try:
            if request["stage"] is None and self.pause() is False:
                raise MaintenanceError("participant could not pause")
            if stage == "refresh" and self.refresh() is False:
                raise MaintenanceError("participant could not refresh")
            self.store.acknowledge(self.id, task_id, stage)
        except Exception as exc:
            self.store.acknowledge(self.id, task_id, stage, error=type(exc).__name__)
            raise

    def close(self):
        self.stop.set()
        if self._thread is not None:
            self._thread.join(1)
            if self._thread.is_alive():
                raise MaintenanceError("participant watcher has not stopped")
        self.store.remove_participant(self.id)
        self._lease.__exit__(None, None, None)
