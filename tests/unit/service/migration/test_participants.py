import multiprocessing

import pytest

from service.migration.maintenance import MaintenanceError, MaintenanceStore
from service.migration.participants import Participant, wait_participants


def participant_process(path, events, finish):
    store = MaintenanceStore(path, 'instance')
    def pause():
        events.put('paused')
        return True
    participant = Participant(store, pause=pause, refresh=lambda: events.put('refreshed'),
                              resume=lambda: events.put('resumed'))
    participant.start()
    events.put('ready')
    finish.wait(10)
    participant.close()


def test_cross_process_drain_refresh_and_reopen(tmp_path):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    ctx = multiprocessing.get_context('spawn')
    events, finish = ctx.Queue(), ctx.Event()
    process = ctx.Process(target=participant_process, args=(store.path, events, finish))
    process.start()
    try:
        assert events.get(timeout=5) == 'ready'
        token = store.admit()
        store.enter('task', mode='import', resources={})
        with pytest.raises(MaintenanceError):
            store.backed_up('task', {})
        store.finish_operation(token)
        wait_participants(store, 'task', 'drain', timeout=5)
        assert events.get(timeout=2) == 'paused'
        store.backed_up('task', {})
        store.begin_changes('task', {})
        store.verified('task')
        with pytest.raises(MaintenanceError):
            store.reopen('task')
        store.request_refresh('task')
        wait_participants(store, 'task', 'refresh', timeout=5)
        assert events.get(timeout=2) == 'refreshed'
        store.reopen('task')
        assert events.get(timeout=2) == 'resumed'
    finally:
        finish.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0


def test_unknown_callback_write_keeps_participant_and_gate(tmp_path):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    participant = None
    def pause():
        with participant.admission.backend(write=True):
            raise TimeoutError('lost storage response')
    participant = Participant(store, pause=pause, refresh=lambda: None, resume=lambda: None)
    store.enter('task', mode='import', resources={})
    with pytest.raises(TimeoutError):
        participant.step()
    with pytest.raises(MaintenanceError):
        store.request_refresh('task')
    with pytest.raises(MaintenanceError):
        participant.close()
    # Explicitly release only this test's lease; the uncertain operation stays.
    participant._lease.__exit__(None, None, None)
    with pytest.raises(MaintenanceError):
        store.retire_stopped_participants()
    assert store.status()['id'] == 'task'


def test_new_process_during_restore_is_dormant_until_refresh(tmp_path):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    store.enter('task', mode='import', resources={})
    store.backed_up('task', {})
    store.begin_changes('task', {})
    events = []
    participant = Participant(store, pause=lambda: pytest.fail('new process must not flush'),
                              refresh=lambda: events.append('refresh'), resume=lambda: events.append('resume'))
    try:
        participant.step()
        with pytest.raises(MaintenanceError):
            with participant.admission.scope():
                pytest.fail('business must stay blocked')
        store.verified('task')
        store.request_refresh('task')
        participant.step()
        store.reopen('task')
        participant.step()
        assert events == ['refresh', 'resume']
    finally:
        participant.close()


@pytest.mark.parametrize('writing', [False, True])
def test_dead_process_permits_preserve_unknown_writes(tmp_path, writing):
    import subprocess
    import sys

    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    script = '''
import os, sys
from service.migration.maintenance import MaintenanceStore
from service.migration.participants import Participant
store = MaintenanceStore(sys.argv[1], 'instance')
p = Participant(store, pause=lambda: True, refresh=lambda: None, resume=lambda: None)
with p.admission.scope(write=sys.argv[2] == 'True'):
    os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(store.path), str(writing)], check=True)
    store.enter('task', mode='import', resources={})
    if writing:
        with pytest.raises(MaintenanceError, match='unfinished'):
            store.wait_drained('task', .05)
    else:
        store.wait_drained('task', .05)
    with store._transaction() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0] == int(writing)
        assert db.execute('SELECT count(*) FROM participants').fetchone()[0] == int(writing)


@pytest.mark.parametrize('stage', ['drain', 'refresh'])
def test_settled_lifecycle_failure_retries_only_on_explicit_request(tmp_path, stage):
    store = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    calls = []
    def callback():
        calls.append(stage)
        if len(calls) == 1:
            raise OSError('transient local failure')
        return True
    p = Participant(store, pause=callback if stage == 'drain' else lambda: True,
                    refresh=callback if stage == 'refresh' else lambda: None, resume=lambda: None)
    try:
        store.enter('task', mode='import', resources={})
        if stage == 'refresh':
            p.step()
            store.request_refresh('task')
        with pytest.raises(OSError):
            p.step()
        p.step()
        assert len(calls) == 1
        if stage == 'refresh':
            store.request_refresh('task')
        else:
            store.retry_participants('task')
        p.step()
        assert calls == [stage, stage]
        assert store.participants_ready('task', stage)
        store.request_refresh('task')
        p.step()
        store.reopen('task')
    finally:
        p.close()
