"""Shared-space SQL transactions participate in the instance maintenance gate."""
from unittest.mock import MagicMock

import pytest

from server.space_repository import SpaceRepository
from service.migration.admission import Admission
from service.migration.maintenance import MaintenanceError, MaintenanceStore


@pytest.fixture
def repository(tmp_path):
    repo = object.__new__(SpaceRepository)
    repo.pool = MagicMock()
    repo._migration_admission = Admission(MaintenanceStore(tmp_path / 'gate.db', 'spaces'))
    return repo


def test_export_blocks_space_writes_and_allows_reads(repository):
    gate = repository._migration_admission.store
    gate.enter('snapshot', mode='export', resources={})
    with pytest.raises(MaintenanceError):
        with repository.transaction('acme'):
            pytest.fail('space write must not start')
    repository.pool.get_connection.assert_not_called()
    with repository.transaction('acme', write=False):
        pass
    gate.wait_drained('snapshot', 0)


def test_import_blocks_space_reads_before_opening_connection(repository):
    repository._migration_admission.store.enter('restore', mode='import', resources={})
    with pytest.raises(MaintenanceError):
        with repository.transaction('acme', write=False):
            pytest.fail('space read must not start')
    repository.pool.get_connection.assert_not_called()


@pytest.mark.parametrize('mode', ['export', 'import'])
@pytest.mark.parametrize('failure_stage', ['statement', 'commit'])
@pytest.mark.parametrize('rollback_fails', [False, True])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_sql_completion_requires_confirmed_rollback(repository, rollback_fails, error_type, failure_stage, mode):
    conn = repository.pool.get_connection.return_value
    if failure_stage == 'commit':
        conn.commit.side_effect = error_type('commit response lost')
    if rollback_fails:
        conn.rollback.side_effect = RuntimeError('rollback unconfirmed')
    with pytest.raises(error_type):
        with repository.transaction('acme'):
            if failure_stage == 'statement':
                raise error_type('statement rejected')
    conn.rollback.assert_called_once()
    if rollback_fails:
        repository.pool.discard_connection.assert_called_once_with(conn)
        repository.pool.return_connection.assert_not_called()
    else:
        repository.pool.discard_connection.assert_not_called()
    gate = repository._migration_admission.store
    gate.enter('restore', mode=mode, resources={})
    if rollback_fails or failure_stage == 'commit':
        with pytest.raises(TimeoutError):
            gate.wait_drained('restore', 0)
    else:
        gate.wait_drained('restore', 0)


def test_discarded_connection_is_never_reused_even_if_close_fails():
    import threading

    from fs.sql_adapter.pool import SharedConnectionPool

    pool = object.__new__(SharedConnectionPool)
    pool._pool = []
    pool._semaphore = threading.BoundedSemaphore(1)
    assert pool._semaphore.acquire(blocking=False)
    conn = MagicMock()
    conn.close.side_effect = RuntimeError('close failed')
    with pytest.raises(RuntimeError, match='close failed'):
        pool.discard_connection(conn)
    assert pool._pool == []
    assert pool._semaphore.acquire(blocking=False)
