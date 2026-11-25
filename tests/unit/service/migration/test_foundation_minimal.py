import multiprocessing
from contextlib import closing

import pytest

from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceError, MaintenanceStore, OperationKind
from service.migration.participants import Participant


def _participant(store):
    return Participant(store, pause=lambda: None, refresh=lambda: None, resume=lambda: None)


@pytest.fixture
def participant(tmp_path):
    with closing(_participant(MaintenanceStore(tmp_path / "gate.sqlite", "instance"))) as participant:
        yield participant


def test_admit_requires_registered_participant(participant):
    store = participant.store
    with pytest.raises(TypeError, match="participant"):
        Admission(store)
    with pytest.raises(TypeError, match="participant"):
        store.admit()
    for invalid in (None, "unregistered"):
        with pytest.raises(MaintenanceError, match="not registered"):
            store.admit(participant_id=invalid)
    store.enter("import", mode="import")
    participant.step()
    store.wait_drained("import", 0)


def test_export_blocks_content_writes_but_keeps_reads(participant):
    store, admission = participant.store, participant.admission
    with admission.scope():
        store.enter("export", mode="export")
        with pytest.raises(MaintenanceError):
            with admission.content_scope():
                pass
        with admission.scope():
            pass
        store.wait_drained("export", 0)
    store.reopen("export")


def test_import_blocks_all_business_operations(participant):
    store, admission = participant.store, participant.admission
    store.enter("import", mode="import")
    with pytest.raises(MaintenanceError):
        with admission.scope():
            pass
    participant.step()
    store.request_refresh("import")
    participant.step()
    store.reopen("import")


def test_unknown_write_keeps_operation_for_reconciliation(tmp_path):
    store = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    participant = _participant(store)
    admission = participant.admission
    with pytest.raises(RuntimeError):
        with admission.content_scope():
            admission.mark_write_uncertain()
            raise RuntimeError("response lost")
    store.enter("export", mode="export")
    with pytest.raises(TimeoutError):
        store.wait_drained("export", 0)
    with pytest.raises(MaintenanceError, match="active operations"):
        participant.close()


def test_participant_invokes_service_callbacks_in_order(tmp_path):
    store = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    events = []
    participant = Participant(store, pause=lambda: events.append("pause"),
                              refresh=lambda: events.append("refresh"),
                              resume=lambda: events.append("resume"))
    try:
        store.enter("import", mode="import")
        participant.pause = lambda: False
        with pytest.raises(MaintenanceError, match="could not pause"):
            participant.step()
        with pytest.raises(TimeoutError):
            store.wait_drained("import", 0)
        with pytest.raises(MaintenanceError):
            store.request_refresh("import")
        with pytest.raises(MaintenanceError):
            store.reopen("import")
        participant.pause = lambda: events.append("pause")
        participant.step()
        assert events == ["pause"]
        store.wait_drained("import", 0)
        store.request_refresh("import")
        participant.step()
        assert events == ["pause", "refresh"]
        store.reopen("import")
        participant.resume = lambda: events.append("retry") or False
        participant.step()
        participant.resume = lambda: events.append("resume")
        participant.step()
        assert events == ["pause", "refresh", "retry", "resume"]
    finally:
        participant.close()


def test_runtime_write_cannot_enter_content_scope_during_export(participant):
    store, admission = participant.store, participant.admission
    with admission.scope(OperationKind.RUNTIME_WRITE):
        store.enter("export", mode="export")
        with pytest.raises(MaintenanceError):
            with admission.content_scope():
                pass


def _hold_operation(path, write, ready, release):
    gate = MaintenanceStore(path, "instance")
    kind = OperationKind.CONTENT_WRITE if write else OperationKind.READ
    with closing(_participant(gate)) as participant, participant.admission.scope(kind):
        ready.set()
        if not release.wait(10):
            raise TimeoutError("controller did not release operation")


def _join_worker(worker):
    worker.join(10)
    if worker.is_alive():
        worker.terminate()
        worker.join(5)


@pytest.mark.parametrize(("mode", "write", "must_drain"), [
    ("export", False, False),
    ("export", True, True),
    ("import", False, True),
    ("import", True, True),
])
def test_gate_drains_operations_in_another_process(tmp_path, mode, write, must_drain):
    gate = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    worker = ctx.Process(target=_hold_operation, args=(gate.path, write, ready, release))
    worker.start()
    try:
        assert ready.wait(10)
        gate.enter("job", mode=mode)
        if must_drain:
            with pytest.raises(TimeoutError):
                gate.wait_drained("job", 0)
        else:
            gate.wait_drained("job", 0)
    finally:
        release.set()
        _join_worker(worker)
    assert worker.exitcode == 0
    gate.wait_drained("job", 0)


@pytest.mark.parametrize("phase", ["drain", "refresh"])
def test_late_start_and_failed_callback_keep_target_closed(tmp_path, phase):
    gate = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    gate.enter("import", mode="import")
    if phase == "refresh":
        gate.request_refresh("import")
    events = []
    participant = Participant(gate, pause=lambda: events.append("pause"),
                              refresh=lambda: False, resume=lambda: events.append("resume"))
    try:
        with pytest.raises(MaintenanceError):
            with participant.admission.scope():
                pytest.fail("late participant must not open storage")
        if phase == "drain":
            with pytest.raises(MaintenanceError):
                gate.request_refresh("import")
            participant.step()
            gate.request_refresh("import")
        with pytest.raises(MaintenanceError, match="could not refresh"):
            participant.step()
        with pytest.raises(MaintenanceError):
            gate.reopen("import")
        participant.refresh = lambda: events.append("refresh")
        participant.step()
        gate.reopen("import")
        participant.step()
        expected_pause_count = 1 if phase == "drain" else 2
        assert events == ["pause"] * expected_pause_count + ["refresh", "resume"]
    finally:
        participant.close()


def test_export_leaves_participants_running_and_content_permit_ends_early(tmp_path):
    gate = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    unexpected = lambda: pytest.fail("export must not pause or refresh a participant")
    participant = Participant(gate, pause=unexpected, refresh=unexpected, resume=lambda: None)
    try:
        with participant.admission.scope(OperationKind.RUNTIME_WRITE):
            with participant.admission.content_scope():
                pass
            gate.enter("export", mode="export")
            participant.step()
            gate.wait_drained("export", 0)
        gate.reopen("export")
    finally:
        participant.close()


def _exit_in_maintenance(path, crashed):
    import os

    gate = MaintenanceStore(path, "instance")
    with gate.execution_lock():
        if crashed:
            participant = _participant(gate)
            if crashed in {"reader", "writer"}:
                kind = OperationKind.CONTENT_WRITE if crashed == "writer" else OperationKind.READ
                gate.admit(participant_id=participant.id, kind=kind)
        gate.enter("import", mode="import")
        os._exit(0)


@pytest.mark.parametrize("crashed", [None, "idle", "reader", "writer"])
def test_process_exit_releases_executor_lock_but_preserves_target_gate(tmp_path, crashed):
    gate = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    worker = multiprocessing.get_context("spawn").Process(target=_exit_in_maintenance, args=(gate.path, crashed))
    worker.start()
    _join_worker(worker)
    assert worker.exitcode == 0
    with closing(_participant(gate)) as participant:
        with gate.execution_lock(), pytest.raises(MaintenanceError, match="in migration maintenance"):
            with participant.admission.scope():
                pytest.fail("import must deny new readers")
    assert gate.status()["id"] == "import"
    if crashed == "writer":
        with pytest.raises(TimeoutError):
            gate.wait_drained("import", 0)
        with pytest.raises(MaintenanceError, match="not finished draining"):
            gate.request_refresh("import")
    else:
        gate.wait_drained("import", 0)
        gate.request_refresh("import")
        gate.reopen("import")
        with closing(_participant(gate)) as participant, participant.admission.scope():
            assert gate.status() is None


def test_startup_resume_retries_on_false(tmp_path, monkeypatch):
    gate = MaintenanceStore(tmp_path / "gate.sqlite", "instance")
    results = iter((False, None))
    participant = Participant(gate, pause=lambda: None, refresh=lambda: None, resume=lambda: next(results))
    monkeypatch.setattr(participant, "start", lambda: None)
    try:
        participant.start_service()
        participant.step()
        participant.step()
        assert next(results, "exhausted") == "exhausted"
    finally:
        participant.close()
