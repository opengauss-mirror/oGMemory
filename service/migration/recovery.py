"""Backup-before-write and fail-closed rollback for a maintenance import.

The caller supplies real backup adapters and an application operation. Admission,
worker quiescence, and cache refresh are mandatory instance lifecycle callbacks.
"""
import hashlib
import json
from pathlib import Path

from service.migration.maintenance import MaintenanceError


class RecoveryImport:
    def __init__(self, gate, workspace, components, *, quiesce, refresh, request_identity=None,
                 resource_binding=None):
        self.gate = gate
        self.workspace = Path(workspace)
        self.component_factory = components if callable(components) else None
        self.components = {} if self.component_factory else dict(components)
        self.resource_binding = resource_binding
        if self.component_factory and not resource_binding:
            raise ValueError('deferred resources require a configuration binding')
        if ((not self.components and not self.component_factory)
                or any(not name.isidentifier() for name in self.components)):
            raise ValueError("named backup components are required")
        self.quiesce = quiesce
        self.refresh = refresh
        self.request_identity = request_identity

    def _root(self, task):
        return self.workspace / hashlib.sha256(task.encode()).hexdigest()

    def _resources(self):
        resources = {name: component.identity() for name, component in self.components.items()}
        if self.resource_binding is not None:
            resources['_configuration'] = self.resource_binding
        if self.request_identity is not None:
            resources["_request"] = self.request_identity
        return resources

    def _open_components(self):
        if self.component_factory and not self.components:
            components = dict(self.component_factory())
            if not components or any(not name.isidentifier() or name.startswith('_') for name in components):
                raise ValueError('named backup components are required')
            self.components = components

    def _check_request(self, state):
        resources = json.loads(state['resources'])
        if (resources.get('_request') != self.request_identity
                or resources.get('_configuration') != self.resource_binding):
            raise MaintenanceError('target configuration or request differs from original task')

    def _assert_stopped(self):
        if self.quiesce() is not True:
            raise MaintenanceError("backend completion is unknown; maintenance remains active")

    def run(self, task, apply, verify, *, timeout=30, progress=lambda **kw: None, prepare=lambda: None, receipts=lambda: ()):
        with self.gate.execution():
            prior = self.gate.status(task)
            if prior:
                self._check_request(prior)
                if not self.component_factory and prior["resources"] != json.dumps(self._resources(), sort_keys=True, allow_nan=False):
                    raise MaintenanceError("target resources differ from original task")
                if prior["phase"] in {"succeeded", "restored", "unchanged"}:
                    self._reopen_if_owned(task)
                    return self.gate.status(task)
                raise MaintenanceError("interrupted task requires recovery before retrying")
            self.gate.enter(task, mode="import", resources=self._resources())
            try:
                progress(phase="draining")
                self.gate.wait_drained(task, timeout)
                self._assert_stopped()
                prepare()
                if self.component_factory:
                    initial = self._resources()
                    self._open_components()
                    self.gate.bind_resources(task, initial, self._resources())
                root = self._root(task)
                root.mkdir(parents=True, mode=0o700)
                progress(phase="backing_up")
                backup = {name: component.capture(root / name) for name, component in self.components.items()}
                # Revalidate the complete recovery set before the first mutation.
                self._validate_all(task, backup)
                self.gate.backed_up(task, backup)
                self.gate.begin_changes(task, self._resources())
            except Exception:
                self._assert_stopped()
                if not self.gate.status(task)["modified"]:
                    self._reopen_if_owned(task)
                raise
            try:
                progress(phase="merging")
                result = apply()
                progress(phase="verifying")
                verify()
                self._assert_stopped()
            except Exception:
                self._recover(task, progress)
                raise
            # Outside the rollback handler: a lost success acknowledgment must
            # never cause already-verified data to be cleared and reapplied.
            self.gate.verified(task, result=result, receipts=receipts())
            self._reopen_if_owned(task)
            return self.gate.status(task)

    def _validate_all(self, task, backup):
        if set(backup) != set(self.components):
            raise MaintenanceError("recovery component scope changed")
        for name, component in self.components.items():
            component.validate(self._root(task) / name, backup[name])

    def recover(self, task, *, progress=lambda **kw: None):
        with self.gate.execution():
            state = self.gate.status(task)
            if state is None:
                raise MaintenanceError("unknown recovery task")
            self._check_request(state)
            if not self.component_factory and json.loads(state["resources"]) != self._resources():
                raise MaintenanceError("target resources differ from original task")
            if state["phase"] in {"succeeded", "restored", "unchanged"}:
                self._reopen_if_owned(task)
                return self.gate.status(task)
            self.gate.retry_participants(task)
            if not state["modified"]:
                self._assert_stopped()
                self._reopen_if_owned(task)
                return self.gate.status(task)
            self._assert_stopped()
            self._open_components()
            if json.loads(state['resources']) != self._resources():
                raise MaintenanceError('target resources differ from original task')
            return self._recover(task, progress)

    def _recover(self, task, progress):
        self._assert_stopped()
        state = self.gate.status(task)
        backup = json.loads(state["backup"])
        self._validate_all(task, backup)  # No component may clear before all validate.
        self.gate.begin_recovery(task, self._resources())
        progress(phase="recovering")
        for name, component in self.components.items():
            component.restore(self._root(task) / name, backup[name])
        for name, component in self.components.items():
            component.verify_live(self._root(task) / name, backup[name])
        self._assert_stopped()
        self.gate.verified(task, restored=True)
        self._reopen_if_owned(task)
        return self.gate.status(task)

    def _reopen_if_owned(self, task):
        active = self.gate.status()
        if active and active["id"] == task:
            self.refresh()  # Must complete for every process before opening admission.
            self.gate.reopen(task)
