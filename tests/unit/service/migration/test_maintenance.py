import multiprocessing

import pytest

from service.migration.maintenance import MaintenanceError, MaintenanceStore


def rejected_in_process(path, result):
    gate = MaintenanceStore(path, "instance")
    try:
        gate.admit()
    except MaintenanceError:
        result.put(True)
    else:
        result.put(False)


def test_import_blocks_new_readers_and_writers_across_processes(tmp_path):
    gate = MaintenanceStore(tmp_path / "state.db", "instance")
    reader = gate.admit()
    gate.enter("task", mode="import", resources={"root": "a"})
    for write in (False, True):
        with pytest.raises(MaintenanceError):
            gate.admit(write=write)
    with pytest.raises(TimeoutError):
        gate.wait_drained("task", 0)
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=rejected_in_process, args=(gate.path, result))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=1) is True
    gate.finish_operation(reader)
    gate.wait_drained("task", 0)
    gate.reopen("task")
    gate.finish_operation(gate.admit())


def test_export_allows_reads_but_not_write_upgrade(tmp_path):
    gate = MaintenanceStore(tmp_path / "state.db", "instance")
    writer = gate.admit(write=True)
    gate.enter("snapshot", mode="export", resources={})
    reader = gate.admit()
    with pytest.raises(MaintenanceError):
        gate.upgrade(reader)
    with pytest.raises(TimeoutError):
        gate.wait_drained("snapshot", 0)
    gate.finish_operation(writer)
    gate.wait_drained("snapshot", 0)
    gate.finish_operation(reader)
    gate.reopen("snapshot")


def test_crash_recovery_cannot_reopen_or_restore_wrong_target(tmp_path):
    path = tmp_path / "state.db"
    gate = MaintenanceStore(path, "instance")
    target = {"root": "original", "collection": "vectors"}
    gate.enter("task", mode="import", resources=target)
    with pytest.raises(MaintenanceError):
        gate.begin_changes("task", target)
    gate.backed_up("task", {"checksum": "verified"})
    gate.begin_changes("task", target)
    restarted = MaintenanceStore(path, "instance")
    with pytest.raises(MaintenanceError):
        restarted.reopen("task")
    with pytest.raises(MaintenanceError):
        restarted.begin_recovery("task", {"root": "other"})
    assert restarted.begin_recovery("task", target) == {"checksum": "verified"}
    restarted.verified("task", restored=True)
    restarted.reopen("task")
    with pytest.raises(MaintenanceError):
        restarted.begin_recovery("task", target)


def test_success_survives_response_loss_without_rollback(tmp_path):
    gate = MaintenanceStore(tmp_path / "state.db", "instance")
    gate.enter("task", mode="import", resources={})
    gate.backed_up("task", {})
    gate.begin_changes("task", {})
    gate.verified("task", result={"inserted": 2})
    restarted = MaintenanceStore(gate.path, "instance")
    assert restarted.status()["phase"] == "succeeded"
    with pytest.raises(MaintenanceError):
        restarted.begin_recovery("task", {})
    restarted.reopen("task")
    assert restarted.status() is None
    assert restarted.enter("task", mode="import", resources={})["phase"] == "succeeded"
