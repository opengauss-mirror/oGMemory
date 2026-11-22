"""Real native AGFS producer, fixed content package, and archive consumer."""
from contextlib import contextmanager

from commit.outbox_store import OutboxStore
from core.models import RequestContext, Role
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from providers.unified_config import OgMemConfig
from service.migration.admission import Admission
from service.migration.engine import MigrationService
from service.migration.maintenance import MaintenanceStore
from service.migration.package import open_package
from service.migration.snapshot import SourceSnapshot
from service.migration.storage import AGFSStorage
from session.archive_store import SessionArchiveStore
from tests.unit.fs.test_native_agfs_client import native


def test_native_content_snapshot_excludes_later_archive_and_runtime(native, tmp_path):
    client, mount, _ = native
    gate = MaintenanceStore(tmp_path / 'gate.db', 'content')
    admission = Admission(gate)
    admission.guard(client)
    fs = admission.guard(AGFSContextFS(client=client, mount_prefix=mount))
    archive = SessionArchiveStore(fs)
    outbox = admission.guard(OutboxStore(client=client, fs=fs, mount_prefix=mount))
    ctx = RequestContext('acme', 'alice', 'agent', 's', 't', Role.MEMBER)
    first = archive.write_archive('s', 'before overview', 'before abstract',
                                  [{'role': 'user', 'content': 'before'}], ctx=ctx)
    assert first.success
    cfg = OgMemConfig(provider='mock', agfs_mount_prefix=mount,
                      storage_backend='agfs', skill_sedimentation_enabled=False)
    @contextmanager
    def adapter_factory(_):
        yield AGFSStorage(cfg, client=client)
    snapshot = SourceSnapshot(gate, 'export', timeout=5)
    events = []
    def progress(**event):
        phase = event.get('phase')
        if phase == 'snapshotting':
            assert gate.status() is not None
            # Runtime Outbox publication need not be frozen or even parseable.
            outbox.register_write(fs.read_node(first.uri, ctx), ctx)
            assert archive.read_archive('s', first.archive_id, ctx=ctx) is not None
        if phase == 'exporting':
            assert gate.status() is None
            later = archive.write_archive('s', 'after overview', 'after abstract',
                                          [{'role': 'user', 'content': 'after'}], ctx=ctx)
            assert later.success
            events.append(later)
    package = tmp_path / 'snapshot.tar.gz'
    # A live request remains admitted while copying and packing proceed.
    with admission.scope(write=True):
        MigrationService(cfg, adapter_factory=adapter_factory, snapshot_scope=snapshot.scope).export(package, progress)
    with open_package(package) as saved:
        rows = list(saved.records('archives'))
        assert len(rows) == 1
        assert rows[0]['messages'] == [{'role': 'user', 'content': 'before'}]
    assert archive.read_archive('s', events[0].archive_id, ctx=ctx) is not None
