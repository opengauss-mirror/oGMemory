"""Unit tests for protected workspace-sync control-plane persistence.

The local-store test uses the real JSON-backed ``ControlPlaneStore``.  SQL
tests use a mocked connection/cursor in place of PostgreSQL, so they verify
SQL construction and serialization only, not a database integration path. A
small fake AGFS client exercises only the atomic mutex primitive contract, not
an AGFS backend integration path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from server.control_plane_store import ControlPlaneStore
from server.sql_control_plane_store import SQLControlPlaneStore


def test_registration_lock_excludes_other_stores_and_releases_on_failure(tmp_path, monkeypatch):
    first, second = (ControlPlaneStore("/mount", local_root=str(tmp_path)) for _ in range(2))
    with pytest.raises(ValueError, match="failed registration"):
        with first.code_workspace_registration_lock():
            # Even a long-running owner must not lose its lock to a timeout.
            monkeypatch.setattr("server.control_plane_store.time.time", lambda: 10**12)
            with pytest.raises(RuntimeError, match="registration is busy"):
                with second.code_workspace_registration_lock():
                    pytest.fail("concurrent registration entered")
            raise ValueError("failed registration")
    with second.code_workspace_registration_lock():
        pass


def test_registration_preserves_older_writers_lock(tmp_path):
    store = ControlPlaneStore("/mount", local_root=str(tmp_path))
    legacy = tmp_path / "_system" / "code_workspace_registration.lock"
    legacy.parent.mkdir()
    legacy.touch()
    with pytest.raises(RuntimeError, match="busy"), store.code_workspace_registration_lock():
        pytest.fail("older writer still owns the lock")
    legacy.unlink()
    with store.code_workspace_registration_lock(), pytest.raises(FileExistsError), legacy.open("xb"):
        pytest.fail("older writer bypassed the session lock")


@pytest.mark.parametrize("outcome", ["success", "exception", "disconnect"])
def test_sql_registration_uses_one_transaction_and_cleans_up(monkeypatch, outcome):
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    import psycopg2

    connection = MagicMock()
    connect = Mock(return_value=connection)
    monkeypatch.setattr("server.sql_control_plane_store.psycopg2.connect", connect)
    store = SQLControlPlaneStore.__new__(SQLControlPlaneStore)
    store._connection_string = "test-dsn"
    pooled = MagicMock()
    store._get_connection = Mock(return_value=pooled)
    store._return_connection = Mock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ({"root_id": "root"}, None, None)
    cursor.fetchall.return_value = []
    error = ValueError if outcome == "exception" else psycopg2.InterfaceError
    with nullcontext() if outcome == "success" else pytest.raises(error):
        with store.code_workspace_registration_lock():
            connection.set_session.assert_called_once_with(isolation_level="READ COMMITTED")
            assert store.read_code_workspace_sync_state("acct", "root")["root_id"] == "root"
            assert store.list_code_workspace_sync_registrations() == []
            store.mutate_code_workspace_sync_state("acct", "root", lambda state: state)
            connection.commit.assert_not_called()
            connection.rollback.assert_not_called()
            if outcome == "exception":
                raise ValueError("failed registration")
            if outcome == "disconnect":
                cursor.execute.side_effect = psycopg2.InterfaceError("connection closed")
                store.mutate_code_workspace_sync_state("acct", "root", lambda state: state)
    connect.assert_called_once_with("test-dsn", connect_timeout=5)
    assert connection.commit.call_count == (outcome == "success")
    store._get_connection.assert_not_called()
    connection.close.assert_called_once()
    with store._workspace_sync_connection() as conn:
        assert conn is pooled
    store._return_connection.assert_called_once_with(pooled)


@pytest.mark.parametrize("backend", ["local", "agfs"])
@pytest.mark.parametrize("raw", [b'{"private":', b'null', b'[]', b'\xff'])
def test_workspace_registry_rejects_unreadable_records(tmp_path, backend, raw):
    client = FakeAtomicAGFSClient()
    client.ls = lambda path: ["acct"] if path == "/mount/accounts" else ["root.json"]
    kwargs = {"local_root": str(tmp_path)} if backend == "local" else {"client": client}
    store = ControlPlaneStore("/mount", **kwargs)
    path = store.code_workspace_sync_path("acct", "root")
    store._ensure_parent(path)
    store._write_bytes(path, raw)
    for read in (lambda: store.read_code_workspace_sync_state("acct", "root"),
                 store.list_code_workspace_sync_registrations):
        with pytest.raises(RuntimeError, match="^workspace registry record is unreadable$"):
            read()
    assert store.read_code_workspace_sync_state("acct", "absent") is None


def test_local_workspace_sync_state_is_copied_and_enumerated_by_registration(tmp_path):
    store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    first = {
        "root_id": "root-a",
        "workspace_path": "/private/a",
        "dirty": True,
        "snapshot": {"a.py": {"mtime_ns": 1, "size": 1}},
    }
    second = {
        "root_id": "root-b",
        "workspace_path": "/private/b",
        "dirty": False,
        "snapshot": {},
    }

    store.write_code_workspace_sync_state("acct-a", "root-a", first)
    store.write_code_workspace_sync_state("acct-b", "root-b", second)
    returned = store.read_code_workspace_sync_state("acct-a", "root-a")
    assert returned is not None
    returned["snapshot"]["a.py"]["size"] = 99

    stored = store.read_code_workspace_sync_state("acct-a", "root-a")
    assert stored is not None
    assert stored["snapshot"]["a.py"]["size"] == 1
    assert store.list_code_workspace_sync_states("acct-a") == [("root-a", first)]
    assert store.list_code_workspace_sync_registrations() == [
        ("acct-a", "root-a", first),
        ("acct-b", "root-b", second),
    ]


@pytest.mark.parametrize("other_account", [False, True], ids=["fresh-store", "unregistered-account"])
@pytest.mark.parametrize("reason", ["no such directory", "permission denied", "request timeout"])
def test_agfs_registration_handles_directory_scan_errors(tmp_path, monkeypatch, other_account, reason):
    from contextlib import nullcontext
    from pyagfs import AGFSClientError

    from tests.unit.service.test_code_workspace_sync import FakeLifecycle, _config, _ctx, _service

    client = FakeAtomicAGFSClient()
    missing = "/mount/accounts"
    if other_account:
        missing += "/other/_system/code_workspace_sync"

    def ls(path):
        if path == missing:
            raise AGFSClientError(f"{reason}: {path}")
        assert other_account and path == "/mount/accounts"
        return ["other"]

    client.ls = Mock(side_effect=ls)
    store = ControlPlaneStore("/mount", client=client)
    # This double injects listing errors. Session exclusion is tested on real AGFS.
    monkeypatch.setattr(store, "code_workspace_registration_lock", nullcontext)
    service = _service(store, FakeLifecycle(), config=_config(code_workspace_allowed_roots=[str(tmp_path)]))
    if reason == "no such directory":
        result = service.register(path=str(tmp_path), ctx=_ctx(), name="repo")
        assert result["accepted"]
        assert store.read_code_workspace_sync_state("acct", result["root_id"]) is not None
    else:
        with pytest.raises(RuntimeError, match="^workspace registry directory is unreadable$"):
            service.register(path=str(tmp_path), ctx=_ctx(), name="repo")
        assert not client.files
    client.ls.assert_any_call(missing)


@pytest.mark.parametrize("denied", ["accounts", "accounts/acct/_system/code_workspace_sync"])
def test_local_registration_refuses_unreadable_directory(tmp_path, monkeypatch, denied):
    import os
    from pathlib import Path

    from tests.unit.service.test_code_workspace_sync import FakeLifecycle, _config, _ctx, _service

    store = ControlPlaneStore("", local_root=str(tmp_path / "registry"))
    service = _service(store, FakeLifecycle(), config=_config(code_workspace_allowed_roots=[str(tmp_path)]))
    service.register(path=str(tmp_path), ctx=_ctx(), name="repo")
    scandir = os.scandir

    def guarded_scandir(path):
        if Path(path) == tmp_path / "registry" / denied:
            raise PermissionError(path)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)
    with pytest.raises(RuntimeError, match="^workspace registry directory is unreadable$"):
        service.register(path=str(tmp_path), ctx=_ctx("other"), name="repo")
    assert not (tmp_path / "registry/accounts/other").exists()


@pytest.mark.parametrize("backend", ["local", "agfs"])
@pytest.mark.parametrize("fail_publish", [False, True])
def test_registry_scan_sees_only_published_state(tmp_path, monkeypatch, backend, fail_publish):
    import os

    client = FakeAtomicAGFSClient()
    client.ls = lambda path: (
        ["acct"] if path == "/mount/accounts" else
        [name.rsplit("/", 1)[1] for name in client.files if name.startswith(path + "/")]
    )
    kwargs = {"local_root": str(tmp_path)} if backend == "local" else {"client": client}
    store = ControlPlaneStore("/mount", **kwargs)
    reader = ControlPlaneStore("/mount", **kwargs)
    old, new = {"version": 1}, {"version": 2}
    store.write_code_workspace_sync_state("acct", "root", old)
    owner, method = (os, "replace") if backend == "local" else (client, "mv")
    publish = getattr(owner, method)

    def scan_before_publish(source, target):
        assert reader.list_code_workspace_sync_registrations() == [("acct", "root", old)]
        if fail_publish:
            raise OSError("publication failed")
        publish(source, target)

    intercept = Mock(side_effect=scan_before_publish)
    monkeypatch.setattr(owner, method, intercept)
    if fail_publish:
        with pytest.raises(OSError, match="publication failed"):
            store.write_code_workspace_sync_state("acct", "root", new)
    else:
        store.write_code_workspace_sync_state("acct", "root", new)
    intercept.assert_called_once()
    assert reader.list_code_workspace_sync_registrations() == [("acct", "root", old if fail_publish else new)]
    files = list(tmp_path.rglob("*.tmp")) if backend == "local" else list(client.files)
    assert not any(str(path).endswith(".tmp") for path in files)


class FakeAtomicAGFSClient:
    """Minimal AGFS double for the short control-plane mutex protocol."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.created: list[str] = []
        self.writes: list[tuple[str, bytes]] = []
        self.removed: list[str] = []
        self.directories: list[str] = []
        self.read_overrides: dict[str, bytes] = {}
        self.read_sequences: dict[str, list[bytes]] = {}
        self.stats: dict[str, dict[str, object]] = {}
        self.on_write = None

    def create(self, path: str) -> None:
        if path in self.files:
            raise RuntimeError("already exists")
        self.files[path] = b""
        self.stats[path] = {"modTime": datetime.now(UTC).isoformat()}
        self.created.append(path)

    def read(self, path: str) -> bytes:
        sequence = self.read_sequences.get(path)
        if sequence:
            return sequence.pop(0)
        if path in self.read_overrides:
            return self.read_overrides[path]
        try:
            return self.files[path]
        except KeyError as exc:
            raise RuntimeError("not found") from exc

    def write(self, path: str, data: bytes) -> None:
        self.files[path] = data
        self.stats.setdefault(path, {"modTime": datetime.now(UTC).isoformat()})
        self.writes.append((path, data))
        if self.on_write is not None:
            self.on_write(path, data)

    def rm(self, path: str) -> None:
        self.files.pop(path, None)
        self.stats.pop(path, None)
        self.removed.append(path)

    def mv(self, source: str, target: str) -> None:
        self.files[target] = self.files.pop(source)
        self.stats[target] = self.stats.pop(source)

    def mkdir(self, path: str) -> None:
        self.directories.append(path)

    def stat(self, path: str) -> dict[str, object]:
        try:
            return self.stats[path]
        except KeyError as exc:
            raise RuntimeError("not found") from exc


def test_agfs_workspace_sync_mutex_uses_atomic_create_and_matching_token():
    """Unit-only: fake AGFS verifies mutex ownership, not backend parity."""

    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: {"root_id": "root-a", "version": 1, "dirty": True},
    )

    state_path = store.code_workspace_sync_path("acct", "root-a")
    lock_path = state_path + ".lock"
    lock_writes = [(path, payload) for path, payload in client.writes if path == lock_path]
    assert result["dirty"] is True
    assert client.created == [lock_path]
    assert len(lock_writes) == 1
    lock_marker = json.loads(lock_writes[0][1].decode("utf-8"))
    assert lock_marker["kind"] == "code_workspace_sync_mutex"
    assert lock_marker["token"]
    assert lock_path in client.removed
    assert lock_path not in client.files
    assert json.loads(client.files[state_path].decode("utf-8"))["root_id"] == "root-a"


def test_agfs_workspace_sync_mutation_fails_closed_without_atomic_create():
    store = ControlPlaneStore(mount_prefix="/control", client=object())

    with pytest.raises(RuntimeError, match="atomic workspace sync state locking"):
        store.mutate_code_workspace_sync_state(
            "acct",
            "root-a",
            lambda current: {"root_id": "root-a"},
        )


def test_agfs_workspace_sync_mutex_never_releases_a_replaced_token():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    lock_path = store.code_workspace_sync_path("acct", "root-a") + ".lock"
    state_path = store.code_workspace_sync_path("acct", "root-a")

    def replace_after_state_write(path: str, _data: bytes) -> None:
        if path.startswith(state_path + ".") and path.endswith(".tmp"):
            client.read_overrides[lock_path] = json.dumps(
                {
                    "kind": "code_workspace_sync_mutex",
                    "token": "other-worker",
                    "expires_at": 9999999999,
                }
            ).encode("utf-8")

    client.on_write = replace_after_state_write

    store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: {"root_id": "root-a"},
    )

    assert lock_path not in client.removed


def test_agfs_workspace_sync_mutex_reclaims_stale_uninitialized_lock():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    state_path = store.code_workspace_sync_path("acct", "root-a")
    lock_path = state_path + ".lock"
    client.files[lock_path] = b""
    client.stats[lock_path] = {
        "modTime": (datetime.now(UTC) - timedelta(seconds=6)).isoformat()
    }

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda _current: {"root_id": "root-a", "dirty": True},
    )

    assert result["dirty"] is True
    assert lock_path in client.removed
    assert json.loads(client.files[state_path].decode("utf-8"))["root_id"] == "root-a"


def test_agfs_workspace_sync_mutex_recognizes_its_token_after_transient_read():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    lock_path = store.code_workspace_sync_path("acct", "root-a") + ".lock"
    client.read_sequences[lock_path] = [b""]

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda _current: {"root_id": "root-a", "dirty": True},
    )

    assert result["dirty"] is True
    assert lock_path in client.removed


def _sql_store_with_cursor(cursor: Mock, connection: Mock) -> SQLControlPlaneStore:
    store = SQLControlPlaneStore.__new__(SQLControlPlaneStore)
    store._get_connection = lambda: connection
    store._return_connection = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=cursor)
    context.__exit__ = Mock(return_value=False)
    connection.cursor.return_value = context
    return store


def test_sql_workspace_sync_transition_locks_row_and_serializes_lease():
    """Unit-only: a mocked cursor checks the SQL transition contract."""

    cursor = Mock()
    connection = Mock()
    store = _sql_store_with_cursor(cursor, connection)
    expires_at = datetime.fromtimestamp(150.0, UTC)
    cursor.fetchone.return_value = ({"root_id": "root-a", "dirty": True}, "old", expires_at)
    observed_current = []

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: (
            observed_current.append(current)
            or {
                **current,
                "dirty": False,
                "lease_owner": "worker-new",
                "lease_expires_at": 300.0,
            }
        ),
    )

    queries = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("ON CONFLICT (account_id, root_id) DO NOTHING" in query for query in queries)
    assert any("FOR UPDATE" in query for query in queries)
    update_call = cursor.execute.call_args_list[-1]
    assert "to_timestamp(%s)" in update_call.args[0]
    assert update_call.args[1][1:3] == ("worker-new", 300.0)
    assert observed_current == [
        {
            "root_id": "root-a",
            "dirty": True,
            "lease_owner": "old",
            "lease_expires_at": 150.0,
        }
    ]
    assert result["lease_owner"] == "worker-new"
    assert result["lease_expires_at"] == 300.0
    connection.commit.assert_called_once_with()
    store._return_connection.assert_called_once_with(connection)


def test_sql_workspace_sync_registration_enumeration_restores_lease_fields():
    """Unit-only: mocked SQL rows prove restart-recovery serialization."""

    cursor = Mock()
    connection = Mock()
    store = _sql_store_with_cursor(cursor, connection)
    cursor.fetchall.return_value = [
        (
            "acct-a",
            "root-a",
            {"workspace_path": "/private/a"},
            "worker-a",
            datetime.fromtimestamp(250.0, UTC),
        ),
        (
            "acct-b",
            "root-b",
            {"workspace_path": "/private/b"},
            None,
            None,
        ),
    ]

    registrations = store.list_code_workspace_sync_registrations()

    assert registrations == [
        (
            "acct-a",
            "root-a",
            {
                "workspace_path": "/private/a",
                "lease_owner": "worker-a",
                "lease_expires_at": 250.0,
            },
        ),
        (
            "acct-b",
            "root-b",
            {
                "workspace_path": "/private/b",
                "lease_owner": None,
                "lease_expires_at": None,
            },
        ),
    ]
    query = cursor.execute.call_args.args[0]
    assert "ORDER BY account_id, root_id" in query
    connection.rollback.assert_called_once_with()
