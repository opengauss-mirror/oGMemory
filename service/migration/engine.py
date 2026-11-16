"""Single-process, offline export/restore with preflight and full data verification."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from cli import __version__
from service.migration.assets import Assets
from service.migration.description import compatibility, describe
from service.migration.package import DATASETS, PackageError, open_package, write_package
from service.migration.records import (
    archive_uri,
    identity,
    normalize_archive,
    normalize_node,
    normalize_relation,
    validate_record,
)
from service.migration.storage import storage


def record_key(kind, record):
    if kind == "nodes":
        return record["uri"].rstrip("/")
    if kind == "archives":
        return archive_uri(record)
    return json.dumps([record[k] for k in ("from_uri", "to_uri", "relation_type")])


class Inventory:
    """Disk-backed validation avoids retaining every node or ID in process memory."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE records (kind TEXT, key TEXT, body TEXT, PRIMARY KEY(kind,key))"
        )
        self.db.execute(
            "CREATE TABLE refs (source TEXT, target TEXT, account TEXT, archive INTEGER)"
        )
        self.db.execute("CREATE TABLE archives (account TEXT, id TEXT, uri TEXT)")
        self.db.execute("CREATE TABLE locations (path TEXT PRIMARY KEY)")
        self.db.execute("CREATE INDEX archive_identity ON archives(account,id)")
        self.db.execute(
            "CREATE INDEX record_scope ON records "
            "(kind,json_extract(body,'$.account_id'),json_extract(body,'$.owner_space'),key)"
        )

    def close(self):
        self.db.close()

    def add(self, kind, r):
        validate_record(kind, r)
        r = {
            "nodes": normalize_node,
            "archives": normalize_archive,
            "relations": normalize_relation,
        }[kind](r)
        key = record_key(kind, r)
        body = json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False)
        try:
            self.db.execute("INSERT INTO records VALUES(?,?,?)", (kind, key, body))
        except sqlite3.IntegrityError as exc:
            old = self.db.execute(
                "SELECT body FROM records WHERE kind=? AND key=?", (kind, key)
            ).fetchone()[0]
            if kind == "relations" and old == body:
                return
            raise PackageError(f"duplicate {kind} identity") from exc
        if kind in {"nodes", "archives"}:
            from fs.agfs_adapter.agfs_context_fs import uri_to_path

            uri = r["uri"] if kind == "nodes" else archive_uri(r)
            try:
                self.db.execute("INSERT INTO locations VALUES(?)", (uri_to_path(uri),))
            except sqlite3.IntegrityError as exc:
                raise PackageError("two records resolve to the same storage path") from exc
        if kind == "archives":
            self.db.execute(
                "INSERT INTO archives VALUES(?,?,?)", (r["account_id"], r["archive_id"], key)
            )
        elif kind == "relations":
            for target in (r["from_uri"], r["to_uri"]):
                self.db.execute(
                    "INSERT INTO refs VALUES(?,?,?,0)",
                    (
                        key,
                        target.rstrip("/"),
                        "",
                    ),
                )
        elif kind == "nodes":
            self._references(r["metadata"], key, r["account_id"])

    def _references(self, value, source, account):
        if isinstance(value, dict):
            for child in value.values():
                self._references(child, source, account)
        elif isinstance(value, list):
            for child in value:
                self._references(child, source, account)
        elif isinstance(value, str) and (
            value.startswith(("trace://", "tool-result://"))
            or (
                value.startswith("ctx://")
                and unquote(urlsplit(value).path).split("/")[1:2] == ["traces"]
            )
        ):
            raise PackageError("external trace/tool evidence is not supported by migration v1")
        elif isinstance(value, str) and value.startswith("prov:"):
            from core.provenance_resolver import ProvenanceResolver

            parsed = ProvenanceResolver.parse_id(value)
            if parsed["version"] != 1:
                raise PackageError("unsupported provenance version")
            if parsed["source_type"] == "memory" and identity(parsed["source_id"])[0] != account:
                raise PackageError("cross-account provenance reference")
            if parsed["source_type"] not in {"memory", "archive"}:
                raise PackageError("unsupported provenance source; cannot preserve evidence")
            self.db.execute(
                "INSERT INTO refs VALUES(?,?,?,?)",
                (
                    source,
                    parsed["source_id"].rstrip("/"),
                    account,
                    int(parsed["source_type"] == "archive"),
                ),
            )

    def finish(self):
        self.db.commit()
        missing = self.db.execute("""SELECT source,target FROM refs f WHERE
            (archive=0 AND NOT EXISTS(SELECT 1 FROM records r WHERE r.key=f.target AND r.kind IN ('nodes','archives')))
            OR (archive=1 AND (SELECT count(*) FROM archives a WHERE a.account=f.account AND a.id=f.target)<>1)
            LIMIT 1""").fetchone()
        if missing:
            raise PackageError(f"missing or ambiguous reference: {missing[1]}")
        # A history archive must not also be imported as an ordinary node.
        if self.db.execute(
            "SELECT 1 FROM records a JOIN records b ON a.key=b.key WHERE a.kind='nodes' AND b.kind='archives' LIMIT 1"
        ).fetchone():
            raise PackageError("duplicate archive/node identity")

    def records(self, kind, *, scope=None):
        sql = "SELECT body FROM records WHERE kind=?"
        params = [kind]
        if scope is not None:
            sql += (
                " AND json_extract(body,'$.account_id')=? AND json_extract(body,'$.owner_space')=?"
            )
            params.extend(scope)
        for (body,) in self.db.execute(sql + " ORDER BY key", params):
            yield json.loads(body)

    def summary(self):
        result = {}
        for kind in DATASETS:
            count, digest = 0, hashlib.sha256()
            for r in self.records(kind):
                digest.update(
                    json.dumps(
                        r, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                )
                digest.update(b"\n")
                count += 1
            result[kind] = {"count": count, "sha256": digest.hexdigest()}
        return result


@contextmanager
def inventory(path, reader):
    inv = Inventory(path)
    try:
        for kind in DATASETS:
            for r in reader(kind):
                inv.add(kind, r)
        inv.finish()
        yield inv
    finally:
        inv.close()


class MigrationService:
    def __init__(self, cfg, *, adapter_factory=storage, index_factory=None, asset_root=None):
        self.cfg = cfg
        self.adapter_factory = adapter_factory
        self.index_factory = index_factory
        self.asset_root = asset_root

    def export(self, destination, progress=lambda **kw: None):
        destination = Path(destination)
        progress(phase="validating")
        with (
            tempfile.TemporaryDirectory(dir=destination.parent) as tmp,
            self.adapter_factory(self.cfg) as adapter,
        ):
            adapter.check()
            assets = Assets(self.asset_root, tmp)

            def read(kind):
                for record in adapter.records(kind):
                    yield assets.capture(record) if kind == "nodes" else record

            with inventory(Path(tmp) / "source.sqlite", read) as inv:
                progress(phase="exporting", counts=inv.summary())
                staged = Path(tmp) / "package.tar.gz"
                write_package(
                    staged,
                    {kind: inv.records(kind) for kind in DATASETS},
                    source_backend=adapter.backend,
                    source_version="ogmem/" + __version__,
                    source_environment=describe(self.cfg, adapter),
                    blobs=assets.files.values(),
                )
                with open_package(staged):
                    pass
                os.link(staged, destination)
                with destination.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                return {
                    "data": inv.summary(),
                    "source_backend": adapter.backend,
                    "archive_bytes": destination.stat().st_size,
                    "archive_sha256": digest,
                }

    def _validate_source(self, package, inv):
        assets = Assets(self.asset_root, package.root)
        assets.validate(package, inv)
        # Reject malformed typed Agent records before any target writes.
        from service.migration.rebuild import records_for_node, validate_relation_endpoints

        for node in inv.records("nodes"):
            records_for_node(node)
        validate_relation_endpoints(self.cfg, inv)
        return assets

    def _check_target(self, adapter):
        from service.migration.rebuild import validate_target_model

        adapter.check(empty=True)
        target = describe(self.cfg, adapter)
        validate_target_model(self.cfg, target["embedding"]["dimension"])
        return target

    def preflight(self, source):
        with (
            open_package(source) as package,
            inventory(package.root / "inventory.sqlite", package.records) as inv,
        ):
            self._validate_source(package, inv)
            with self.adapter_factory(self.cfg) as adapter:
                target = self._check_target(adapter)
            with self._indexes() as indexes:
                indexes.check_empty()
            return {
                "data": inv.summary(),
                "source_backend": package.manifest["source_backend"],
                "target_backend": self.cfg.storage_backend,
                "vector_backend": self.cfg.vector_db_type,
                "compatibility": compatibility(package.manifest.get("source_environment"), target),
            }

    @contextmanager
    def _indexes(self):
        from service.migration.rebuild import Rebuilder

        try:
            obj = (self.index_factory or Rebuilder)(self.cfg)
        except ValueError as exc:
            raise PackageError(
                "target vector configuration or stored index schema is invalid"
            ) from exc
        try:
            yield obj
        finally:
            obj.close()

    def restore(self, source, progress=lambda **kw: None):
        progress(phase="validating")
        # A GUI preview is advisory. Revalidate the current package and target,
        # then retain this same validated staging area through restore/verify.
        with (
            open_package(source) as package,
            inventory(package.root / "inventory.sqlite", package.records) as inv,
        ):
            assets = self._validate_source(package, inv)
            expected = inv.summary()
            with self.adapter_factory(self.cfg) as adapter:
                self._check_target(adapter)
                with self._indexes() as indexes:
                    indexes.check_empty()
                    # Probe the model before writing any canonical data.
                    indexes.probe()
                    progress(phase="restoring", counts={k: dict(v) for k, v in expected.items()})
                    for kind in ("nodes", "archives", "relations"):
                        records = inv.records(kind)
                        if kind == "nodes":
                            records = (assets.restore(package, r) for r in records)
                        adapter.restore(kind, records)
                    progress(phase="reindexing")
                    indexed = indexes.rebuild(inv, progress)
                    progress(phase="verifying")

                    def read_restored(kind):
                        for record in adapter.records(kind):
                            yield (
                                assets.capture(record, copy_bytes=False)
                                if kind == "nodes"
                                else record
                            )

                    with inventory(package.root / "restored.sqlite", read_restored) as restored:
                        if expected != restored.summary():
                            raise PackageError("restored data differs from migration package")
                    indexes.verify(inv)
                    return {
                        "data": expected,
                        "index_records": indexed,
                        "target_backend": adapter.backend,
                        "next_step": "Stop migration mode, configure identities and start the normal service.",
                    }
