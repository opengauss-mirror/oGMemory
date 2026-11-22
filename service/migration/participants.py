"""Coordinate process quiescence and cache refresh through the shared local gate."""
import logging
import threading
import time
import uuid
from contextlib import contextmanager

from service.migration.admission import Admission
from service.migration.file_lock import exclusive_file
from service.migration.maintenance import MaintenanceError

logger = logging.getLogger(__name__)


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
        active = store.status()
        self._paused = active is not None
        self._failed = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._watch, name='migration-participant', daemon=True)
            self._thread.start()

    def start_service(self):
        try:
            with self.admission.scope(write=True):
                self.resume()
            self._paused = False
        except MaintenanceError:
            if self.store.status() is None:
                raise
            self._paused = True
        self.start()

    @contextmanager
    def shutdown_scope(self, timeout):
        with self.admission.scope(write=True, _shutdown=True):
            try:
                self._stop_watcher(timeout)
                self.admission.stop_accepting()
                yield
            except BaseException:
                self.stop.clear()
                if self._thread is None or not self._thread.is_alive():
                    self._thread = None
                    self.start()
                raise

    def wait_for_operations(self, deadline):
        """Drain this process, excluding the current shutdown operation."""
        token = self.admission.current_token()
        while self.store.has_other_operations(self.id, token):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MaintenanceError('business operations have not stopped')
            time.sleep(min(.05, remaining))

    def step(self):
        request = self.store.lifecycle_request(self.id)
        if request is None:
            if self._paused:
                with self.admission.scope(write=True):
                    self.resume()
                self._paused = False
                self._failed = None
            return
        if request['mode'] == 'export':
            return  # Export only pauses writes; existing readers and caches stay live.
        task, stage = request['task'], request['lifecycle']
        attempt = (task, stage, request['attempt'])
        if request['stage'] == stage or self._failed == attempt:
            return
        if stage == 'drain':
            try:
                self.store.wait_drained(task, 0)
            except TimeoutError:
                return
        try:
            if stage == 'drain':
                # Poll without reserving a business permit. A long-running reader
                # keeps the target closed; no callback may flush over that reader.
                with self.admission.maintenance_scope(task):
                    if self.pause() is not True:
                        raise MaintenanceError('business workers have not stopped')
                self._paused = True
            elif stage == 'refresh':
                with self.admission.maintenance_scope(task):
                    self.refresh()
            else:
                raise MaintenanceError('unknown lifecycle request')
            self.store.acknowledge(self.id, task, stage)
        except Exception as exc:
            self._failed = attempt
            try:
                self.store.acknowledge(self.id, task, stage, error=type(exc).__name__)
            except MaintenanceError:
                pass  # Unknown completion retains its operation and prevents opening.
            raise

    def _watch(self):
        while not self.stop.is_set():
            try:
                self.step()
            except Exception as exc:
                logger.error('Migration participant requires attention: %s', type(exc).__name__)
            self.stop.wait(self.interval)

    def _stop_watcher(self, timeout):
        self.stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise MaintenanceError('maintenance callback is still running')
    def close(self, *, timeout=5):
        self._stop_watcher(timeout)
        # The caller must already have stopped its business workers. Never retire
        # a process with uncertain operations merely because its watcher stopped.
        self.store.remove_participant(self.id)
        self._lease.__exit__(None, None, None)


def wait_participants(store, task, stage, *, timeout):
    deadline = time.monotonic() + timeout
    while True:
        store.retire_stopped_participants()
        if store.participants_ready(task, stage):
            return
        if time.monotonic() >= deadline:
            raise MaintenanceError('business processes have not acknowledged maintenance')
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
