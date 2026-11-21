"""Exercise the merge engine with isolated lifecycle and backup fixtures.

Production entry-point coverage uses InstanceMigration and the normal HTTP API.
These helpers keep protocol/backend tests independent of service scheduling.
"""
import json
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

from service.migration.bundle import backup_components
from service.migration.engine import MigrationService
from service.migration.maintenance import MaintenanceStore
from service.migration.recovery import RecoveryImport


class MemoryBackup:
    """In-memory byte transport fixture backup, never used for real backends."""
    def __init__(self, client):
        self.client = client

    def identity(self):
        return {'fixture': 'memory'}

    def capture(self, path):
        self.saved = deepcopy((self.client.files, self.client.directories))
        return {'fixture': True}

    def validate(self, path, manifest):
        assert manifest == {'fixture': True}

    def verify_live(self, path, manifest):
        self.validate(path, manifest)
        assert (self.client.files, self.client.directories) == self.saved

    def restore(self, path, manifest):
        self.client.files, self.client.directories = deepcopy(self.saved)


class MigrationHarness(MigrationService):
    def import_package(self, source, progress=lambda **kw: None):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as lifetime:
            gate = MaintenanceStore(Path(tmp) / 'gate.db', 'fixture')
            adapter = indexes = None
            original_adapter, original_indexes = self.adapter_factory, self.index_factory

            @contextmanager
            def adapter_factory(cfg):
                nonlocal adapter
                adapter = lifetime.enter_context(original_adapter(cfg))
                yield adapter

            def index_factory(cfg):
                nonlocal indexes
                from service.migration.rebuild import Rebuilder
                indexes = (original_indexes or Rebuilder)(cfg)
                return indexes

            def components(incoming):
                if hasattr(getattr(adapter, 'client', None), 'files'):
                    return {'content': MemoryBackup(adapter.client)}
                return backup_components(self.cfg, incoming, adapter, indexes, lifetime,
                                         asset_root=self.asset_root)

            @contextmanager
            def recovery(request, incoming):
                yield RecoveryImport(gate, Path(tmp) / 'backups', lambda: components(incoming),
                    quiesce=lambda: True, refresh=lambda: None, request_identity=request,
                    resource_binding={'fixture': str(tmp)})

            self.adapter_factory, self.index_factory = adapter_factory, index_factory
            try:
                result = self.merge(source, task=str(uuid.uuid4()), recovery_factory=recovery,
                                    llm=None, progress=progress)
                return json.loads(result['result'])
            finally:
                self.adapter_factory, self.index_factory = original_adapter, original_indexes


def write_described_package(*args, source_backend, source_version, vector_backend="chroma", database_type="postgresql", **kwargs):
    """Build synthetic backend fixtures with an explicit source environment."""
    from service.migration.package import write_package
    kwargs.setdefault("source_environment", {
        "embedding": {"provider": "mock", "model": "MockEmbedder", "dimension": 3, "model_version": None},
        "content_storage": {"backend": source_backend,
                            "database_type": database_type if source_backend == "sql" else None,
                            "database_version": None},
        "vector_storage": {"backend": vector_backend},
    })
    return write_package(*args, source_backend=source_backend, source_version=source_version, **kwargs)
