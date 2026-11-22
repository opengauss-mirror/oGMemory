"""SQL I/O obeys the source snapshot deadline."""
import pytest
from service.migration.maintenance import MaintenanceStore


def test_sql_snapshot_stalled_handshake_is_bounded(tmp_path):
    import socket
    import threading
    import time

    from providers.unified_config import OgMemConfig
    from service.migration.snapshot import SourceSnapshot
    from service.migration.sql_snapshot import SnapshotConnection

    stop = threading.Event()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    listener.settimeout(2)
    def stalled_server():
        with listener:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                return
            with conn:
                stop.wait(2)
    thread = threading.Thread(target=stalled_server)
    thread.start()
    gate = MaintenanceStore(tmp_path / 'gate.db', 'instance')
    cfg = OgMemConfig(storage_backend='sql', sql_connection_string=(
        f'host=127.0.0.1 port={listener.getsockname()[1]} dbname=ignored user=ignored'))
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError), SourceSnapshot(gate, 'snapshot', timeout=.15).scope():
            SnapshotConnection(cfg.sql_connection_string, started + .15)
        assert time.monotonic() - started < 1
        assert gate.status() is None
    finally:
        stop.set()
        thread.join(3)
