"""Internal vector recovery points. Never call an embedding provider on restore."""
import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from core.models import IndexRecord
from service.migration.durability import sync_directory
from service.migration.package import PackageError


def _encode(record):
    raw = asdict(record)
    vector = raw["metadata"].get("_embedding")
    if not isinstance(vector, list) or not vector or any(
        isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
        for n in vector
    ):
        raise PackageError("vector backup requires the stored finite embedding")
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _equivalent(stored, record):
    expected, actual = json.loads(stored), json.loads(_encode(record))
    before = expected['metadata'].pop('_embedding')
    after = actual['metadata'].pop('_embedding')
    # Cosine backends may renormalize float32 vectors on reinsertion. Only
    # vector round-off is tolerated; every business field remains exact.
    return (expected == actual and len(before) == len(after)
            and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                    for a, b in zip(before, after, strict=True)))


class VectorBackup:
    """The identity callback must resolve the actual backend resource on every call.

    Callers hold maintenance throughout capture/restore. A manifest is stored in
    the durable task, outside the data and collection that this component clears.
    """
    def __init__(self, index, identity):
        self.index = index
        self.identity = identity

    def capture(self, path):
        path = Path(path)
        # Do not replace a previously verified recovery point.
        with path.open("xb"):
            pass
        os.chmod(path, 0o600)
        resource = self.identity()
        count = 0
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE records (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            for record in self.index.iter_records():
                db.execute("INSERT INTO records VALUES(?,?)", (record.id, _encode(record)))
                count += 1
            db.commit()
        sync_directory(path.parent)
        manifest = {"resource": resource, "sha256": _digest(path), "records": count}
        self.verify_live(path, manifest)
        return manifest

    def validate(self, path, manifest):
        if self.identity() != manifest["resource"]:
            raise PackageError("vector recovery target differs from backup")
        if _digest(path) != manifest["sha256"]:
            raise PackageError("vector backup checksum mismatch")
        with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise PackageError("vector backup database is corrupt")
            count = 0
            for identity, body in db.execute("SELECT id,body FROM records"):
                record = IndexRecord(**json.loads(body))
                if record.id != identity or _encode(record) != body:
                    raise PackageError("vector backup record differs from its identity")
                count += 1
            if count != manifest["records"]:
                raise PackageError("vector backup record count mismatch")

    def restore(self, path, manifest):
        self.validate(path, manifest)  # Before the first deletion.
        # Snapshot IDs before deleting: offset-based scans would otherwise skip rows.
        with closing(sqlite3.connect("")) as ids:
            ids.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            for record in self.index.iter_records():
                ids.execute("INSERT INTO ids VALUES(?)", (record.id,))
            cursor = ids.execute("SELECT id FROM ids")
            while batch := cursor.fetchmany(256):
                self.index.delete([row[0] for row in batch])
        with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as db:
            cursor = db.execute("SELECT body FROM records ORDER BY id")
            while batch := cursor.fetchmany(64):
                self.index.upsert([IndexRecord(**json.loads(row[0])) for row in batch])
        self.verify_live(path, manifest)

    def verify_live(self, path, manifest):
        self.validate(path, manifest)
        with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as db:
            count = 0
            for record in self.index.iter_records():
                row = db.execute("SELECT body FROM records WHERE id=?", (record.id,)).fetchone()
                if row is None or not _equivalent(row[0], record):
                    raise PackageError("restored vector differs from backup")
                count += 1
            if count != manifest["records"]:
                raise PackageError("restored vector count differs from backup")
