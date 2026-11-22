
import pytest

from service.migration.package import PackageError
from service.migration.tree_backup import AGFSTreeBackup


from tests.migration_fixtures import LocalClient


def test_raw_tree_restore_preserves_runtime_and_binary_files(tmp_path):
    client = LocalClient(tmp_path / "remote")
    root = client.path("/local/accounts")
    (root / "a/.outbox").mkdir(parents=True)
    (root / "a/.outbox/pending.json").write_bytes(b'{"status":"PENDING"}')
    (root / "a/binary").write_bytes(b"\x00\xff" * 600000)
    (root / "empty").mkdir()
    neighbor = client.path("/local/control")
    neighbor.write_bytes(b"credential")
    backup = AGFSTreeBackup(client, ["/local/accounts"], lambda: {"instance": "a"})
    destination = tmp_path / "backup"
    manifest = backup.capture(destination)
    (root / "a/binary").write_bytes(b"changed")
    (root / "added").write_bytes(b"new")
    backup.restore(destination, manifest)
    assert (root / "a/binary").read_bytes() == b"\x00\xff" * 600000
    assert not (root / "added").exists()
    assert (root / "empty").is_dir()
    assert neighbor.read_bytes() == b"credential"
    backup.restore(destination, manifest)


def test_corrupt_file_rejected_before_clearing_root(tmp_path):
    client = LocalClient(tmp_path / "remote")
    root = client.path("/local/accounts")
    root.mkdir(parents=True)
    (root / "data").write_bytes(b"original")
    backup = AGFSTreeBackup(client, ["/local/accounts"], lambda: {"instance": "a"})
    destination = tmp_path / "backup"
    manifest = backup.capture(destination)
    blob = next(p for p in destination.iterdir() if p.name != "inventory.sqlite")
    blob.write_bytes(b"broken")
    with pytest.raises(PackageError, match="checksum"):
        backup.restore(destination, manifest)
    assert (root / "data").read_bytes() == b"original"


@pytest.mark.parametrize("roots", [["/"], ["/a/../b"], ["/a", "/a/b"]])
def test_unsafe_roots_rejected(roots):
    with pytest.raises(ValueError):
        AGFSTreeBackup(None, roots, lambda: {})


def test_write_capability_failure_never_deletes_original(tmp_path):
    client = LocalClient(tmp_path / "remote")
    root = client.path("/local/accounts")
    root.mkdir(parents=True)
    (root / "original").write_bytes(b"original")
    backup = AGFSTreeBackup(client, ["/local/accounts"], lambda: {})
    destination = tmp_path / "backup"
    manifest = backup.capture(destination)
    def unavailable(*args, **kwargs):
        raise OSError("write unavailable")
    client.put = unavailable
    with pytest.raises(OSError):
        backup.restore(destination, manifest)
    assert (root / "original").read_bytes() == b"original"


@pytest.mark.parametrize('mode', [1 << 27, 1 << 26, 1 << 25, 1 << 24, 1 << 19])
def test_real_agfs_mode_rejects_special_files_before_read(tmp_path, mode):
    from types import SimpleNamespace
    client = SimpleNamespace(ls=lambda path: [{'name': 'content.md', 'mode': mode | 0o777, 'isDir': False}])
    backup = AGFSTreeBackup(client, ['/local/accounts'], lambda: {})
    with pytest.raises(PackageError, match='file type'):
        list(backup._walk('/local/accounts'))
