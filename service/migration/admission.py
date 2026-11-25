"""Backend-independent admission scopes for the migration maintenance gate."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import asyncio
import threading

from service.migration.maintenance import MaintenanceError, OperationKind


@dataclass
class Operation:
    token: str
    owner: tuple
    kind: OperationKind
    active: bool = True
    uncertain: bool = False


class Admission:
    def __init__(self, store, participant_id: str):
        self.store = store
        self.participant_id = participant_id
        self._operation = ContextVar(f"migration-operation-{id(self)}", default=None)

    @staticmethod
    def _owner():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), task

    @contextmanager
    def scope(self, kind: OperationKind = OperationKind.READ):
        current = self._operation.get()
        owner = self._owner()
        # A content permit ends with persistence, even inside a longer request.
        if (current is not None and current.active and current.owner == owner
                and (kind is not OperationKind.CONTENT_WRITE or current.kind is kind)):
            if kind is OperationKind.RUNTIME_WRITE and current.kind is OperationKind.READ:
                self.store.upgrade_to_runtime_write(current.token)
                current.kind = kind
            yield
            return
        token = self.store.admit(participant_id=self.participant_id, kind=kind)
        operation = Operation(token, owner, kind)
        marker = self._operation.set(operation)
        try:
            yield
        finally:
            operation.active = False
            try:
                if not operation.uncertain:
                    self.store.finish_operation(token)
            finally:
                self._operation.reset(marker)

    def mark_write_uncertain(self):
        """Retain admission when the caller cannot confirm a write has ended."""
        operation = self._operation.get()
        if operation is None or not operation.active or operation.owner != self._owner() or operation.kind is OperationKind.READ:
            raise MaintenanceError("active write required")
        operation.uncertain = True

    @contextmanager
    def content_scope(self):
        with self.scope(OperationKind.CONTENT_WRITE):
            yield
