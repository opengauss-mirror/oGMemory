from fs.sql_adapter import pool as module
from fs.sql_adapter.pool import SharedConnectionPool


def test_dns_resolution_preserves_host_identity_and_fallback_ports(monkeypatch):
    def resolve(host, port, **kwargs):
        if host == "missing.invalid":
            raise module.socket.gaierror("unavailable")
        return [(0, 0, 0, '', (address, port)) for address in ('::1', '127.0.0.1')]

    monkeypatch.delenv("PGHOSTADDR", raising=False)
    monkeypatch.setattr(module.socket, "getaddrinfo", resolve)
    options = module._resolved_hosts(
        "host=missing.invalid,db.example port=5432,15438 sslmode=verify-full",
        module.time.monotonic() + 1,
    )
    assert options == {
        "host": "db.example,db.example", "hostaddr": "::1,127.0.0.1", "port": "15438,15438",
    }


def test_dns_worker_capacity_recovers_after_timed_out_request(monkeypatch):
    import pytest

    release = module.threading.Event()
    finished = module.threading.Event()
    slots = module.threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, "_dns_slots", slots)

    def resolve(*args, **kwargs):
        release.wait(2)
        finished.set()
        return [(0, 0, 0, '', ('127.0.0.1', 5432))]

    monkeypatch.delenv("PGHOSTADDR", raising=False)
    monkeypatch.setattr(module.socket, "getaddrinfo", resolve)
    try:
        with pytest.raises(TimeoutError):
            module._resolved_hosts("host=db.example", module.time.monotonic() + 0.02)
        with pytest.raises(TimeoutError):
            module._resolved_hosts("host=db.example", module.time.monotonic() + 0.02)
    finally:
        release.set()
        assert finished.wait(2)
    assert slots.acquire(timeout=1)
    slots.release()


class _Connection:
    closed = 0

    def close(self):
        self.closed = 1


class _StaleConnection(_Connection):
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _sql):
            raise RuntimeError("stale socket")

    def cursor(self):
        return self._Cursor()


def test_new_connection_has_bounded_connect_timeout(monkeypatch):
    calls = []
    connection = _Connection()

    def connect(dsn, **kwargs):
        calls.append((dsn, kwargs))
        return connection

    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    monkeypatch.setattr(module.psycopg2, "connect", connect)
    pool = SharedConnectionPool("dbname=test", connect_timeout_seconds=7)

    borrowed = pool.get_connection()

    assert borrowed is connection
    assert calls == [("dbname=test", {"connect_timeout": 7})]
    pool.return_connection(borrowed)
    pool.close_all()


def test_connect_timeout_must_be_positive(monkeypatch):
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)

    try:
        SharedConnectionPool("dbname=test", connect_timeout_seconds=0)
    except ValueError as exc:
        assert str(exc) == "connect_timeout_seconds must be positive"
    else:
        raise AssertionError("zero timeout should be rejected")


def test_default_borrow_discards_stale_idle_connection(monkeypatch):
    stale = _StaleConnection()
    fresh = _Connection()
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    monkeypatch.setattr(module.psycopg2, "connect", lambda *_args, **_kwargs: fresh)
    pool = SharedConnectionPool("dbname=test")
    pool.return_connection(stale)

    assert pool.get_connection() is fresh
    assert stale.closed == 1


def test_bounded_retrying_caller_can_skip_idle_preflight(monkeypatch):
    stale = _StaleConnection()
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    pool = SharedConnectionPool("dbname=test")
    pool.return_connection(stale)

    assert pool.get_connection(validate=False) is stale


def test_auto_probe_completes_with_delayed_wire_responses():
    import struct
    import threading
    import time

    def message(kind, body):
        return kind + struct.pack("!I", len(body) + 4) + body

    def read_exact(conn, size):
        data = b""
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            assert chunk
            data += chunk
        return data

    def serve(listener):
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(3)
            size = struct.unpack("!I", read_exact(conn, 4))[0]
            read_exact(conn, size - 4)
            time.sleep(.02)
            conn.sendall(message(b"R", struct.pack("!I", 0))
                         + message(b"S", b"server_version\x0016.0\x00")
                         + message(b"S", b"client_encoding\x00UTF8\x00")
                         + message(b"S", b"DateStyle\x00ISO, MDY\x00") + message(b"Z", b"I"))
            assert read_exact(conn, 1) == b"Q"
            size = struct.unpack("!I", read_exact(conn, 4))[0]
            assert read_exact(conn, size - 4) == b"SELECT version()\x00"
            time.sleep(.02)
            row = b"PostgreSQL 16.0"
            description = struct.pack("!H", 1) + b"version\x00" + struct.pack("!IhIhih", 0, 0, 25, -1, -1, 0)
            conn.sendall(message(b"T", description)
                         + message(b"D", struct.pack("!Hi", 1, len(row)) + row)
                         + message(b"C", b"SELECT 1\x00") + message(b"Z", b"I"))

    with module.socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        worker = threading.Thread(target=serve, args=(listener,), daemon=True)
        worker.start()
        dsn = f"host=127.0.0.1 port={listener.getsockname()[1]} dbname=test user=test sslmode=disable"
        started = time.monotonic()
        dialect = module.resolve_connection_dialect("auto", dsn, 2)
        worker.join(3)
    assert dialect.name == "postgresql"
    assert .04 <= time.monotonic() - started < 2
