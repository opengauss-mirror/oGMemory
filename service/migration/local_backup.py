"""Recovery of the instance-owned attachment directory, preserving actual bytes."""
import hashlib
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

from service.migration.durability import sync_directory
from service.migration.package import PackageError
from service.migration.vector_backup import _digest


class LocalTreeBackup:
    def __init__(self, root, *, max_bytes=50 * 1024**3, max_entries=1_000_000):
        self.root = Path(root).resolve()
        if self.root == Path(self.root.anchor) or not self.root.is_dir():
            raise PackageError('attachment recovery requires an existing owned directory')
        self.max_bytes, self.max_entries = max_bytes, max_entries

    def identity(self):
        stat = self.root.stat()
        return {'path': str(self.root.resolve()), 'device': stat.st_dev, 'inode': stat.st_ino}

    def _walk(self):
        pending = [self.root]
        count = 0
        while pending:
            for path in sorted(pending.pop().iterdir()):
                if path.is_symlink() or not (path.is_dir() or path.is_file()):
                    raise PackageError('attachment recovery does not support links or special files')
                relative = path.relative_to(self.root).as_posix()
                count += 1
                if count > self.max_entries or relative.count('/') > 64:
                    raise PackageError('attachment recovery entry/depth limit exceeded')
                directory = path.is_dir()
                yield relative, directory
                if directory:
                    pending.append(path)

    def capture(self, destination):
        destination = Path(destination).resolve()
        if destination.is_relative_to(self.root):
            raise PackageError('backup must be outside the attachment directory')
        destination.mkdir(mode=0o700)
        resource = self.identity()
        size, count = 0, 0
        with closing(sqlite3.connect(destination / 'inventory.sqlite')) as db:
            db.execute('CREATE TABLE files(path TEXT PRIMARY KEY,directory INTEGER,blob TEXT,digest TEXT,mode INTEGER)')
            for relative, directory in self._walk():
                blob = hashlib.sha256(relative.encode()).hexdigest()
                digest = None
                if not directory:
                    source = self.root / relative
                    if size + source.stat().st_size > self.max_bytes:
                        raise PackageError('attachment backup exceeds byte limit')
                    digest_builder = hashlib.sha256()
                    with source.open('rb') as inp, (destination / blob).open('xb') as out:
                        while chunk := inp.read(1024 * 1024):
                            size += len(chunk)
                            if size > self.max_bytes:
                                raise PackageError('attachment backup exceeds byte limit')
                            digest_builder.update(chunk)
                            out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                    digest = digest_builder.hexdigest()
                    if _digest(source) != digest:
                        raise PackageError('attachment changed during backup')
                db.execute('INSERT INTO files VALUES(?,?,?,?,?)',
                           (relative, directory, blob, digest, stat.S_IMODE((self.root / relative).stat().st_mode) & 0o777))
                count += 1
            db.commit()
        sync_directory(destination)
        manifest = {'resource': resource, 'inventory_sha256': _digest(destination / 'inventory.sqlite'),
                    'entries': count, 'bytes': size}
        self.verify_live(destination, manifest)
        return manifest

    def _inventory(self, destination):
        path = (Path(destination) / 'inventory.sqlite').resolve()
        return closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True))

    def validate(self, destination, manifest):
        destination = Path(destination)
        if self.identity() != manifest['resource']:
            raise PackageError('attachment recovery target differs from backup')
        if destination.resolve().is_relative_to(self.root):
            raise PackageError('backup overlaps the attachment directory')
        if _digest(destination / 'inventory.sqlite') != manifest['inventory_sha256']:
            raise PackageError('attachment backup inventory checksum mismatch')
        with self._inventory(destination) as db:
            for relative, directory, blob, digest, mode in db.execute('SELECT * FROM files'):
                path = Path(relative)
                if (path.is_absolute() or '..' in path.parts or '\\' in relative
                        or hashlib.sha256(relative.encode()).hexdigest() != blob
                        or not isinstance(mode, int) or not 0 <= mode <= 0o777):
                    raise PackageError('unsafe attachment backup path')
                if not directory and _digest(destination / blob) != digest:
                    raise PackageError('attachment backup checksum mismatch')
            if db.execute('SELECT count(*) FROM files').fetchone()[0] != manifest['entries']:
                raise PackageError('attachment backup entry count differs')

    def restore(self, destination, manifest):
        self.validate(destination, manifest)
        # Preserve the bound root inode; never delete an ancestor or follow links.
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
        with self._inventory(destination) as db:
            for relative, directory, blob, _, mode in db.execute('SELECT * FROM files ORDER BY length(path),path'):
                path = self.root / relative
                if directory:
                    path.mkdir(mode=0o700)
                else:
                    with (Path(destination) / blob).open('rb') as inp, os.fdopen(
                            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as out:
                        shutil.copyfileobj(inp, out, length=1024 * 1024)
                        out.flush()
                        os.fchmod(out.fileno(), mode) if hasattr(os, 'fchmod') else os.chmod(path, mode)
                        os.fsync(out.fileno())
            for relative, mode in db.execute('SELECT path,mode FROM files WHERE directory=1 ORDER BY length(path) DESC'):
                path = self.root / relative
                os.chmod(path, mode)
                sync_directory(path)
        sync_directory(self.root)
        self.verify_live(destination, manifest)

    def verify_live(self, destination, manifest):
        self.validate(destination, manifest)
        count = 0
        with self._inventory(destination) as db:
            for relative, directory in self._walk():
                row = db.execute('SELECT directory,digest FROM files WHERE path=?', (relative,)).fetchone()
                if row is None or bool(row[0]) != directory:
                    raise PackageError('restored attachment entries differ')
                if not directory and _digest(self.root / relative) != row[1]:
                    raise PackageError('restored attachment bytes differ')
                count += 1
        if count != manifest['entries']:
            raise PackageError('restored attachment count differs')
