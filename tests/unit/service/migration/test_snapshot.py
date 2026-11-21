import pytest

from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.snapshot import SourceSnapshot


def test_snapshot_readers_continue_and_writes_resume(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    other = Admission(MaintenanceStore(tmp_path / 'gate.db', 'instance'))
    snapshot = SourceSnapshot(gate, 'export')
    with snapshot.scope() as check:
        check()
        with other.scope():
            pass
        with pytest.raises(MaintenanceError):
            with other.scope(write=True):
                pass
        assert snapshot.timestamp
    with other.scope(write=True):
        pass


def test_snapshot_timeout_reopens_source_without_changing_writer(tmp_path):
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    admission = Admission(gate)
    with admission.scope(write=True):
        with pytest.raises(TimeoutError):
            with SourceSnapshot(gate, 'export', timeout=0).scope():
                pytest.fail('unfinished writer cannot be snapshotted')
        assert gate.status() is None
    with admission.scope(write=True):
        pass
