import pytest

from service.migration.local_backup import LocalTreeBackup
from service.migration.package import PackageError


def test_attachment_restore_preserves_bytes_and_neighbor(tmp_path):
    root = tmp_path / 'assets'
    (root / 'empty').mkdir(parents=True)
    (root / 'binary').write_bytes(b'\x00\xff' * 600000)
    neighbor = tmp_path / 'credentials'
    neighbor.write_bytes(b'untouched')
    backup = LocalTreeBackup(root)
    manifest = backup.capture(tmp_path / 'backup')
    (root / 'binary').write_bytes(b'changed')
    (root / 'new').write_bytes(b'imported')
    backup.restore(tmp_path / 'backup', manifest)
    assert (root / 'binary').read_bytes() == b'\x00\xff' * 600000
    assert (root / 'empty').is_dir()
    assert not (root / 'new').exists()
    assert neighbor.read_bytes() == b'untouched'
    backup.restore(tmp_path / 'backup', manifest)


def test_corruption_rejected_before_clearing_assets(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    (root / 'data').write_bytes(b'original')
    backup = LocalTreeBackup(root)
    manifest = backup.capture(tmp_path / 'backup')
    next(p for p in (tmp_path / 'backup').iterdir() if p.name != 'inventory.sqlite').write_bytes(b'corrupt')
    with pytest.raises(PackageError, match='checksum'):
        backup.restore(tmp_path / 'backup', manifest)
    assert (root / 'data').read_bytes() == b'original'


def test_replaced_asset_directory_rejected_before_clear(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    backup = LocalTreeBackup(root)
    manifest = backup.capture(tmp_path / 'backup')
    root.rename(tmp_path / 'old')
    root.mkdir()
    (root / 'other-instance').write_bytes(b'keep')
    with pytest.raises(PackageError, match='target differs'):
        backup.restore(tmp_path / 'backup', manifest)
    assert (root / 'other-instance').read_bytes() == b'keep'


def test_link_and_overlapping_backup_are_rejected(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    backup = LocalTreeBackup(root)
    with pytest.raises(PackageError, match='outside'):
        backup.capture(root / 'backup')
    (root / 'outside').symlink_to(tmp_path)
    with pytest.raises(PackageError, match='links'):
        backup.capture(tmp_path / 'backup')
