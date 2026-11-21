from dataclasses import replace

import pytest

from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.recovery import RecoveryImport
from service.migration.tree_backup import AGFSTreeBackup
from service.migration.vector_backup import VectorBackup
from tests.unit.service.migration.test_tree_backup import LocalClient
from tests.unit.service.migration.test_vector_backup import record


@pytest.fixture
def stored_vector(tmp_path):
    pytest.importorskip("chromadb")
    from providers.vector_index.chroma_index import ChromaVectorIndex
    return ChromaVectorIndex(collection_name="recovery", persist_directory=str(tmp_path / "chroma"), dimension=3)


@pytest.fixture
def recovery(tmp_path, stored_vector):
    vector = stored_vector
    client = LocalClient(tmp_path / "remote")
    root = client.path("/local/accounts")
    root.mkdir(parents=True)
    (root / "state").write_bytes(b"original runtime")
    vector.upsert([record()])
    gate = MaintenanceStore(tmp_path / "control.db", "instance")
    importer = RecoveryImport(gate, tmp_path / "backup", {
        "content": AGFSTreeBackup(client, ["/local/accounts"], lambda: {"root": "accounts"}),
        "vector": VectorBackup(vector, lambda: {"collection": "original"}),
    }, quiesce=lambda: True, refresh=lambda: None)
    return gate, importer, root, vector


def changed(root, vector):
    (root / "state").write_bytes(b"changed")
    (root / "imported").write_bytes(b"new")
    vector.upsert([replace(record(), text="changed"), replace(record(), id="new")])


def assert_restored(gate, root, vector):
    assert gate.status() is None
    assert (root / "state").read_bytes() == b"original runtime"
    assert not (root / "imported").exists()
    assert vector.count() == 1
    assert vector.get_record("original").text == "original"


def test_verification_failure_restores_content_and_real_vectors(recovery):
    gate, importer, root, vector = recovery

    def apply():
        with pytest.raises(MaintenanceError):
            gate.admit()
        changed(root, vector)

    def verify():
        raise ValueError("validation failed")

    with pytest.raises(ValueError, match="validation failed"):
        importer.run("task", apply, verify)
    assert gate.status("task")["phase"] == "restored"
    assert_restored(gate, root, vector)
    # A repeated recovery after reopening is read-only.
    (root / "after-recovery").write_bytes(b"new business write")
    importer.recover("task")
    assert (root / "after-recovery").read_bytes() == b"new business write"


def test_process_interruption_requires_explicit_recovery(recovery):
    gate, importer, root, vector = recovery

    def interrupted():
        changed(root, vector)
        raise SystemExit("process stopped")

    with pytest.raises(SystemExit):
        importer.run("task", interrupted, lambda: None)
    with pytest.raises(MaintenanceError):
        gate.admit()
    importer.recover("task")
    assert_restored(gate, root, vector)


def test_success_ack_loss_does_not_roll_back_committed_data(recovery):
    gate, importer, root, vector = recovery

    def refresh_failure():
        raise OSError("client refresh unavailable")

    importer.refresh = refresh_failure
    with pytest.raises(OSError):
        importer.run("task", lambda: changed(root, vector), lambda: None)
    assert gate.status()["phase"] == "succeeded"
    importer.refresh = lambda: None
    importer.recover("task")
    assert gate.status() is None
    assert (root / "state").read_bytes() == b"changed"
    assert vector.count() == 2


def test_unknown_completion_and_wrong_target_keep_maintenance(recovery):
    gate, importer, root, vector = recovery

    def failed():
        changed(root, vector)
        importer.quiesce = lambda: False
        raise OSError("timeout; write may still be active")

    with pytest.raises(MaintenanceError, match="unknown"):
        importer.run("task", failed, lambda: None)
    assert (root / "state").read_bytes() == b"changed"
    importer.quiesce = lambda: True
    importer.components["vector"].identity = lambda: {"collection": "other"}
    with pytest.raises(MaintenanceError, match="resources differ"):
        importer.recover("task")
    assert (root / "state").read_bytes() == b"changed"
    assert gate.status() is not None


def test_only_one_executor_may_run(tmp_path):
    gate = MaintenanceStore(tmp_path / "gate.db", "instance")
    with gate.execution():
        with pytest.raises(MaintenanceError, match="executor"):
            with gate.execution():
                pass


def test_deferred_resources_open_after_drain_and_bind_before_changes(recovery, tmp_path):
    gate, original, root, vector = recovery
    calls = []

    def stopped():
        assert gate.status()['id'] == 'deferred'
        calls.append('stopped')
        return True

    def components():
        assert calls[-1] == 'stopped'
        assert gate.status()['phase'] == 'draining'
        calls.append('opened')
        return original.components

    importer = RecoveryImport(gate, tmp_path / 'deferred', components,
                              resource_binding={'instance': 'configured'},
                              quiesce=stopped, refresh=lambda: None)
    importer.run('deferred', lambda: changed(root, vector), lambda: None)
    assert calls[:2] == ['stopped', 'opened']
    import json
    resources = json.loads(gate.status('deferred')['resources'])
    assert resources['vector'] == {'collection': 'original'}
    assert resources['_configuration'] == {'instance': 'configured'}
    # Reading a completed result cannot reopen a native client over live users.
    importer = RecoveryImport(gate, tmp_path / 'deferred', lambda: pytest.fail('must not open'),
                              resource_binding={'instance': 'configured'},
                              quiesce=lambda: pytest.fail('must not stop'), refresh=lambda: None)
    assert importer.run('deferred', lambda: pytest.fail('must not apply'), lambda: None)['phase'] == 'succeeded'


def test_deferred_recovery_rejects_changed_native_identity(recovery, tmp_path):
    gate, original, root, vector = recovery
    importer = RecoveryImport(gate, tmp_path / 'deferred', lambda: original.components,
                              resource_binding={'instance': 'configured'},
                              quiesce=lambda: True, refresh=lambda: None)
    def interrupted():
        changed(root, vector)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        importer.run('task', interrupted, lambda: None)
    original.components['vector'].identity = lambda: {'collection': 'replaced'}
    resumed = RecoveryImport(gate, tmp_path / 'deferred', lambda: original.components,
                             resource_binding={'instance': 'configured'},
                             quiesce=lambda: True, refresh=lambda: None)
    with pytest.raises(MaintenanceError, match='resources differ'):
        resumed.recover('task')
    assert (root / 'state').read_bytes() == b'changed'
    assert gate.status()['modified']
