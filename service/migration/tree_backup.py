"""Byte-preserving recovery of explicitly owned AGFS trees under maintenance."""
import hashlib
import math
import os
import posixpath
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from service.migration.durability import sync_directory
from service.migration.package import PackageError
from service.migration.vector_backup import _digest


def entry_directory(entry):
    """Validate Go os.FileMode as returned by HTTP AGFS and AGFS_Ls."""
    directory = bool(entry.get('isDir'))
    mode = entry.get('mode')
    if entry.get('isLink'):
        raise PackageError('unsafe AGFS backup entry: symbolic link')
    if mode is not None:
        if type(mode) is not int or not 0 <= mode <= 0xffffffff:
            raise PackageError('invalid AGFS file mode')
        kind = mode & 0x8f280000  # ModeType: dir, link, device, pipe, socket, irregular.
        if kind != (0x80000000 if directory else 0):
            raise PackageError('unsafe AGFS backup entry: unsupported file type')
    return directory


class AGFSTreeBackup:
    def __init__(self, client, roots, identity, *, max_bytes=50 * 1024**3, max_entries=1_000_000):
        self.client = client
        self.roots = tuple(roots)
        self.identity = identity
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        if not self.roots or len(set(self.roots)) != len(self.roots):
            raise ValueError("distinct owned roots are required")
        for root in self.roots:
            if (not root.startswith("/") or root == "/" or posixpath.normpath(root) != root
                    or "\\" in root or any(other != root and root.startswith(other + "/") for other in self.roots)):
                raise ValueError("unsafe or overlapping recovery roots")

    def _walk(self, root):
        from fs.agfs_adapter.agfs_context_fs import _is_not_found_error
        try:
            entries = self.client.ls(root)
        except Exception as exc:
            if _is_not_found_error(exc):
                return
            raise
        yield "", True
        pending = [("", entries)]
        count = 1
        while pending:
            parent, entries = pending.pop()
            for entry in entries:
                name = entry["name"]
                if (not isinstance(name, str) or name in {"", ".", ".."}
                        or "/" in name or "\\" in name or entry.get("isLink")):
                    raise PackageError("unsafe AGFS backup entry")
                relative = f"{parent}/{name}".lstrip("/")
                if relative.count("/") > 64:
                    raise PackageError("AGFS backup directory depth exceeded")
                count += 1
                if count > self.max_entries:
                    raise PackageError("AGFS backup entry limit exceeded")
                directory = entry_directory(entry)
                yield relative, directory
                if directory:
                    pending.append((relative, self.client.ls(root + "/" + relative)))

    def _chunks(self, path):
        offset = 0
        while True:
            data = self.client.read(path, offset=offset, size=1024 * 1024)
            if not isinstance(data, bytes):
                raise PackageError("AGFS backup requires binary reads")
            if not data:
                return
            yield data
            offset += len(data)

    def _put(self, path, data):
        write_stream = getattr(self.client, 'write_stream', None)
        if callable(write_stream):
            from io import BytesIO
            write_stream(path, BytesIO(data) if isinstance(data, bytes) else data)
            return
        # pyagfs.write disables timeouts for streams and retries consumed streams.
        # Use its ordinary /files transport with an explicit timeout and no retry.
        timeout = float(self.client.timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise PackageError("AGFS recovery requires a finite RPC timeout")
        response = self.client.session.put(
            f"{self.client.api_base}/files", params={"path": path},
            data=data, timeout=timeout,
        )
        try:
            response.raise_for_status()
        finally:
            response.close()

    def probe_restore(self):
        """Prove the actual byte-write path before any original root is removed."""
        from fs.agfs_adapter.agfs_context_fs import _is_not_found_error
        parents = set()
        for root in self.roots:
            parent = posixpath.dirname(root)
            while parent != "/":
                try:
                    self.client.ls(parent)
                    break
                except Exception as exc:
                    if not _is_not_found_error(exc):
                        raise
                    parent = posixpath.dirname(parent)
            parents.add(parent)
        for parent in parents:
            probe = parent.rstrip("/") + "/.migration-probe-" + uuid.uuid4().hex
            self.client.mkdir(probe)
            # On an uncertain failed request leave only this task-owned probe;
            # do not race a late write with cleanup, and never touch original data.
            self._put(probe + "/bytes", b"\x00\xffrecovery")
            if b"".join(self._chunks(probe + "/bytes")) != b"\x00\xffrecovery":
                raise PackageError("AGFS recovery write probe differs")
            self._put(probe + "/bytes", b"x")
            if b"".join(self._chunks(probe + "/bytes")) != b"x":
                raise PackageError("AGFS recovery overwrite probe differs")
            self.client.rm(probe, recursive=True)

    def capture(self, destination):
        destination = Path(destination)
        destination.mkdir(mode=0o700)
        resource = self.identity()
        self.probe_restore()
        total = 0
        with closing(sqlite3.connect(destination / "inventory.sqlite")) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE files (root TEXT, path TEXT, directory INTEGER, blob TEXT, digest TEXT, PRIMARY KEY(root,path))")
            for root in self.roots:
                for relative, directory in self._walk(root):
                    blob = hashlib.sha256((root + "/" + relative).encode()).hexdigest()
                    digest = hashlib.sha256()
                    if not directory:
                        with (destination / blob).open("xb") as stream:
                            for chunk in self._chunks(root + "/" + relative):
                                total += len(chunk)
                                if total > self.max_bytes:
                                    raise PackageError("AGFS backup byte limit exceeded")
                                stream.write(chunk)
                                digest.update(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                    db.execute("INSERT INTO files VALUES(?,?,?,?,?)", (root, relative, int(directory), blob, digest.hexdigest()))
            db.commit()
        sync_directory(destination)
        manifest = {"resource": resource, "roots": list(self.roots), "sha256": _digest(destination / "inventory.sqlite"), "bytes": total}
        self.verify_live(destination, manifest)
        return manifest

    def validate(self, destination, manifest):
        destination = Path(destination)
        if self.identity() != manifest["resource"] or list(self.roots) != manifest["roots"]:
            raise PackageError("AGFS recovery target differs from backup")
        if _digest(destination / "inventory.sqlite") != manifest["sha256"]:
            raise PackageError("AGFS backup inventory checksum mismatch")
        with closing(sqlite3.connect(f"{(destination / 'inventory.sqlite').resolve().as_uri()}?mode=ro", uri=True)) as db:
            for root, relative, directory, blob, digest in db.execute("SELECT * FROM files"):
                expected = hashlib.sha256((root + "/" + relative).encode()).hexdigest()
                if root not in self.roots or blob != expected or relative.startswith("/") or ".." in relative.split("/"):
                    raise PackageError("AGFS backup path differs from owned roots")
                if not directory and _digest(destination / blob) != digest:
                    raise PackageError("AGFS backup file checksum mismatch")

    def restore(self, destination, manifest):
        from fs.agfs_adapter.agfs_context_fs import _is_not_found_error
        destination = Path(destination)
        self.validate(destination, manifest)  # Validate every file before removing any root.
        self.probe_restore()
        for root in self.roots:
            try:
                self.client.rm(root, recursive=True)
            except Exception as exc:
                if not _is_not_found_error(exc):
                    raise
        with closing(sqlite3.connect(f"{(destination / 'inventory.sqlite').resolve().as_uri()}?mode=ro", uri=True)) as db:
            for root, relative, directory, blob, _ in db.execute("SELECT * FROM files ORDER BY root,length(path),path"):
                path = root + ("/" + relative if relative else "")
                if directory:
                    self.client.mkdir(path)
                else:
                    with (destination / blob).open("rb") as stream:
                        self._put(path, stream)
        self.verify_live(destination, manifest)

    def verify_live(self, destination, manifest):
        self.validate(destination, manifest)
        with closing(sqlite3.connect(f"{(Path(destination) / 'inventory.sqlite').resolve().as_uri()}?mode=ro", uri=True)) as db:
            count = 0
            for root in self.roots:
                for relative, directory in self._walk(root):
                    row = db.execute("SELECT directory,digest FROM files WHERE root=? AND path=?", (root, relative)).fetchone()
                    if row is None or bool(row[0]) != directory:
                        raise PackageError("restored AGFS entries differ from backup")
                    if not directory:
                        digest = hashlib.sha256()
                        for chunk in self._chunks(root + "/" + relative):
                            digest.update(chunk)
                        if digest.hexdigest() != row[1]:
                            raise PackageError("restored AGFS file differs from backup")
                    count += 1
            if count != db.execute("SELECT count(*) FROM files").fetchone()[0]:
                raise PackageError("restored AGFS entry count differs from backup")
