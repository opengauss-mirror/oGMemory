"""Fix persisted content while readers, model work and index projections continue."""
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from service.migration.maintenance import MaintenanceError


class SourceSnapshot:
    def __init__(self, gate, task, *, timeout=30):
        self.gate = gate
        self.task = task
        self.timeout = timeout
        self.timestamp = None

    @contextmanager
    def scope(self):
        with self.gate.execution():
            if self.gate.status(self.task) is not None:
                raise MaintenanceError('snapshot task already exists; use its saved result or a new task')
            self.gate.enter(self.task, mode='export', resources={'timeout': self.timeout})
            deadline = time.monotonic() + self.timeout

            def check():
                if time.monotonic() >= deadline:
                    raise TimeoutError('source snapshot exceeded its write-pause budget')

            try:
                self.gate.wait_drained(self.task, max(0, deadline - time.monotonic()))
                self.timestamp = datetime.now(UTC).isoformat()
                check()
                from core.operation_deadline import operation_deadline
                with operation_deadline(deadline - time.monotonic()):
                    yield check
                    check()
            finally:
                # Export performs no business writes. Even an unsuccessful drain
                # can release this write barrier without clearing source data.
                self.gate.reopen(self.task)
