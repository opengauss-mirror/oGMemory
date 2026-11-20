"""Migration execution for services registered in the local instance gate."""
import hashlib
import json
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path

from service.migration.bundle import backup_components, configuration_binding
from service.migration.engine import MigrationService
from service.migration.maintenance import MaintenanceError
from service.migration.package import open_package
from service.migration.participants import wait_participants
from service.migration.rebuild import Rebuilder
from service.migration.recovery import RecoveryImport
from service.migration.snapshot import SourceSnapshot
from service.migration.storage import storage


class InstanceMigration:
    def __init__(self, service, *, asset_root=None, timeout=30):
        participant = getattr(service, '_migration_participant', None)
        if participant is None:
            raise MaintenanceError('server is not registered for coordinated migration')
        self.service = service
        self.cfg = service._cfg
        self.gate = participant.store
        self.workspace = Path(service._migration_workspace)
        self.asset_root = asset_root
        self.timeout = timeout

    def export(self, destination, *, task, progress=lambda **kw: None):
        snapshot = SourceSnapshot(self.gate, task, timeout=self.timeout)
        engine = MigrationService(self.cfg, asset_root=self.asset_root, snapshot_scope=snapshot.scope)
        result = engine.export(destination, progress)
        return {**result, 'snapshot_at': snapshot.timestamp}

    def preview(self, source):
        # Reuse the registered native clients until the import drain closes them.
        @contextmanager
        def indexes():
            from types import SimpleNamespace
            if getattr(self.cfg, "agfs_retrieval_enabled", False):
                yield SimpleNamespace(vector=None, embedder=None)
                return
            yield SimpleNamespace(vector=self.service._get_shared_vector_index(),
                                  embedder=self.service._get_shared_embedder())
        with self.service._admission.scope(write=True):
            return MigrationService(self.cfg, asset_root=self.asset_root).preview(source, indexes=indexes)

    def _track(self, obj, names, task, *, settled=lambda: False):
        for name in names:
            original = getattr(obj, name)
            @wraps(original)
            def call(*args, _original=original, **kwargs):
                with self.gate.mutation(task, settled=settled):
                    return _original(*args, **kwargs)
            setattr(obj, name, call)
        return obj

    @contextmanager
    def _execution(self, task):
        binding = configuration_binding(self.cfg, self.asset_root)
        with ExitStack() as resources:
            adapter = indexes = None

            @contextmanager
            def adapter_factory(cfg):
                nonlocal adapter
                adapter = resources.enter_context(storage(cfg))
                def settled():
                    if adapter.backend != 'sql':
                        return False
                    try:
                        adapter.conn.rollback()
                        return True
                    except Exception:
                        return False
                self._track(adapter, ('merge_nodes', 'restore', 'replace_relations'), task, settled=settled)
                if getattr(cfg, 'shared_spaces_enabled', False):
                    self._track(adapter, ('restore_space_catalog', 'publish_space_projections'), task, settled=settled)
                yield adapter

            def index_factory(cfg):
                nonlocal indexes
                indexes = Rebuilder(cfg)
                resources.callback(indexes.close)
                if indexes.vector is not None:
                    self._track(indexes.vector, ('upsert', 'delete'), task)
                if indexes.lexical is not None:
                    self._track(indexes.lexical, ('begin_rebuild', 'replace_scopes', 'abort_rebuild'), task)
                self._track(indexes, ('_graph',), task)
                return indexes

            def quiesce():
                self.gate.wait_drained(task, self.timeout)
                wait_participants(self.gate, task, 'drain', timeout=self.timeout)
                return True

            def refresh():
                # Includes backup SQL connections; no migration-owned native
                # client may remain open when business caches reopen.
                resources.close()
                self.gate.request_refresh(task)
                wait_participants(self.gate, task, 'refresh', timeout=self.timeout)

            def components(incoming):
                nonlocal adapter, indexes
                if adapter is None:
                    resources.enter_context(adapter_factory(self.cfg))
                if indexes is None:
                    index_factory(self.cfg)
                result = backup_components(self.cfg, incoming, adapter, indexes, resources,
                                           asset_root=self.asset_root)
                for component in result.values():
                    self._track(component, ('restore',), task)
                return result

            @contextmanager
            def recovery_factory(request_identity, incoming):
                yield RecoveryImport(
                    self.gate, self.workspace / 'backups',
                    lambda: components(incoming),
                    quiesce=quiesce, refresh=refresh, request_identity=request_identity,
                    resource_binding=binding,
                )

            engine = MigrationService(self.cfg, adapter_factory=adapter_factory, index_factory=index_factory,
                                      asset_root=self.asset_root)
            yield engine, recovery_factory

    def merge(self, source, *, task, progress=lambda **kw: None, shared_selection=None):
        with self._execution(task) as (engine, factory):
            return engine.merge(source, task=task, recovery_factory=factory,
                                llm=self.service.get_llm(), progress=progress, shared_selection=shared_selection)

    def recover(self, source, *, task, progress=lambda **kw: None):
        from service.migration.space_scope import incoming_scope

        with open_package(source) as package, self._execution(task) as (_, factory):
            request = {'manifest_sha256': hashlib.sha256(json.dumps(
                package.manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
            with incoming_scope(package) as (incoming, _), factory(request, incoming) as recovery:
                return recovery.recover(task, progress=progress)
