"""Single-process, offline export/restore with preflight and full data verification."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from urllib.parse import unquote, urlsplit

from cli import __version__
from service.migration.assets import Assets
from service.migration.description import compatibility, describe
from service.migration.package import (
    DATASETS,
    PackageError,
    open_package,
    package_identity,
    write_package,
)
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
        shared = kind == 'space_nodes'
        if shared:
            kind = 'nodes'
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
        if kind in {"nodes", "archives"} and not shared:
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
def inventory(path, reader, *, shared_records=(), validate_references=True):
    inv = Inventory(path)
    try:
        for kind in DATASETS:
            for r in reader(kind):
                inv.add(kind, r)
        for record in shared_records:
            inv.add('space_nodes', record)
        if validate_references:
            inv.finish()
        yield inv
    finally:
        inv.close()


class MigrationService:
    def __init__(self, cfg, *, adapter_factory=storage, index_factory=None, asset_root=None, snapshot_scope=None):
        self.cfg = cfg
        self.adapter_factory = adapter_factory
        self.index_factory = index_factory
        self.asset_root = asset_root
        self.snapshot_scope = snapshot_scope or nullcontext

    def export(self, destination, progress=lambda **kw: None):
        destination = Path(destination)
        progress(phase="validating")
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp, ExitStack() as lifetime:
            assets = Assets(self.asset_root, tmp)
            shared_description = None
            spaces = None
            with self.snapshot_scope() as check_snapshot, self.adapter_factory(self.cfg) as adapter:
                adapter.check()
                progress(phase="snapshotting")
                source_backend = adapter.backend
                environment = describe(self.cfg, adapter)

                def read(kind):
                    for record in adapter.records(kind):
                        if callable(check_snapshot):
                            check_snapshot()
                        yield assets.capture(record) if kind == "nodes" else record

                inv = lifetime.enter_context(inventory(Path(tmp) / "source.sqlite", read,
                    validate_references=not getattr(self.cfg, 'shared_spaces_enabled', False)))
                if getattr(self.cfg, 'shared_spaces_enabled', False):
                    from service.migration.space_catalog import SpaceCatalog
                    from service.migration.space_inventory import SpaceInventory

                    catalog = SpaceCatalog(adapter)
                    shared_description = catalog.description()
                    def shared_read():
                        for record in adapter.records('space_nodes'):
                            if callable(check_snapshot):
                                check_snapshot()
                            yield assets.capture(record)
                    spaces = SpaceInventory(Path(tmp) / 'spaces.sqlite', catalog.records(), shared_read())
                    lifetime.callback(spaces.close)
                    spaces.validate_dependencies(inv)
                    for record in spaces.records('space_nodes'):
                        inv.add('space_nodes', record)
                    inv.finish()
            # All records and attachment bytes are fixed. Release the source write
            # barrier before compression; later source changes are not included.
            progress(phase="exporting", counts=inv.summary())
            staged = Path(tmp) / "package.tar.gz"
            datasets = {kind: inv.records(kind) for kind in DATASETS}
            if spaces is not None:
                datasets['nodes'] = (node for node in inv.records('nodes')
                                     if not node['owner_space'].startswith('space:'))
                datasets.update({kind: spaces.records(kind) for kind in ('spaces', 'space_nodes')})
            write_package(
                staged,
                datasets,
                source_backend=source_backend,
                source_version="ogmem/" + __version__,
                source_environment=environment,
                blobs=assets.files.values(),
                shared_spaces=shared_description,
            )
            with open_package(staged):
                pass
            os.link(staged, destination)
            with destination.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return {
                "data": inv.summary(),
                "source_backend": source_backend,
                "archive_bytes": destination.stat().st_size,
                "archive_sha256": digest,
                **({'shared_spaces': {'spaces': spaces.count(), 'action': 'exported'}} if spaces is not None else {}),
            }

    def _validate_source(self, package, inv):
        assets = Assets(self.asset_root, package.root)
        assets.validate(package, inv, check_target=False)
        # Reject malformed typed Agent records before any target writes.
        from service.migration.rebuild import records_for_node, validate_relation_endpoints

        for node in inv.records("nodes"):
            records_for_node(node)
        validate_relation_endpoints(self.cfg, inv)
        return assets

    def preview(self, source, *, indexes=None):
        from service.migration.merge import fingerprint, lookup
        from service.migration.rebuild import (
            validate_target_model,
            validate_vector_records,
            validate_vector_target,
            verify_vector_inventory,
        )
        from service.migration.space_scope import discard_shared, incoming_scope, shared_records
        from service.migration.space_selection import select_spaces

        with open_package(source) as package, incoming_scope(package) as (incoming, spaces):
            assets = self._validate_source(package, incoming)
            # Reuse this process's registered native client. Opening a second
            # Chroma client before the drain would retain the old native cache.
            with self.adapter_factory(self.cfg) as adapter, ExitStack() as lifetime:
                adapter.check()
                target = describe(self.cfg, adapter)
                validate_target_model(self.cfg, target['embedding']['dimension'])
                projection = lifetime.enter_context((indexes or self._indexes)())
                vector_free = getattr(self.cfg, "agfs_retrieval_enabled", False)
                if not vector_free and projection.embedder is None:
                    projection.embedder = self.cfg.to_provider_config().create_embedder()
                vector, embedder = projection.vector, projection.embedder
                if not vector_free:
                    validate_vector_target(vector, include_embedding=False)
                def read(kind):
                    for record in adapter.records(kind):
                        yield assets.capture(record, copy_bytes=False) if kind == 'nodes' else record
                counts = {'new': 0, 'unchanged': 0, 'conflicts': 0}
                with inventory(package.root / 'target.sqlite', read,
                        shared_records=shared_records(self.cfg, adapter, assets)) as existing:
                    selection = select_spaces(package.manifest, spaces, self.cfg, adapter, existing, vector)
                    if selection['action'] != 'import':
                        discard_shared(incoming)
                    assets.validate(package, incoming, check_declared=False)
                    adapter.validate_restore(incoming)
                    if not vector_free:
                        validate_vector_records(vector, embedder, self.cfg.to_provider_config().embedding_dimension, incoming)
                        verify_vector_inventory(vector, existing, include_embedding=False)
                    for kind in ('nodes', 'archives', 'relations'):
                        for record in incoming.records(kind):
                            previous = lookup(existing, kind, record_key(kind, record))
                            name = ('new' if previous is None else
                                    'unchanged' if fingerprint(previous) == fingerprint(record) else 'conflicts')
                            counts[name] += 1
            return {'data': incoming.summary(), 'counts': counts,
                    'shared_spaces': selection,
                    'compatibility': compatibility(package.manifest.get('source_environment'), target)}

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

    def merge(self, source, *, task, recovery_factory, llm, progress=lambda **kw: None, shared_selection=None):
        """Merge into a backed-up, exclusively admitted target instance.

        The instance controller supplies bound backup adapters and lifecycle
        callbacks.
        """
        from service.migration.merge import MergePlan, PolicyConflictResolver, lookup
        from service.migration.rebuild import (
            records_for_node,
            validate_relation_endpoints,
            validate_target_model,
            validate_vector_target,
        )
        from service.migration.space_scope import discard_shared, incoming_scope, shared_records
        from service.migration.space_selection import confirm_selection, select_spaces

        with open_package(source) as package, ExitStack() as lifetime:
            request_identity = package_identity(package.manifest)
            incoming, spaces = lifetime.enter_context(incoming_scope(package))
            assets = self._validate_source(package, incoming)
            recovery = lifetime.enter_context(recovery_factory(request_identity, incoming))
            if recovery.request_identity != request_identity:
                raise PackageError('recovery task must be bound to the validated package')
            plan = adapter = indexes = None
            target = None
            selection = None
            refresh = recovery.refresh

            def release_and_refresh():
                if indexes is not None:
                    indexes.close()
                refresh()

            recovery.refresh = release_and_refresh

            def prepare():
                nonlocal adapter, indexes, target, plan, selection
                progress(phase='planning')
                adapter = lifetime.enter_context(self.adapter_factory(self.cfg))
                adapter.check()
                target_description = describe(self.cfg, adapter)
                validate_target_model(self.cfg, target_description['embedding']['dimension'])
                indexes = lifetime.enter_context(self._indexes())
                if indexes.vector is not None:
                    validate_vector_target(indexes.vector)

                def read(kind):
                    for record in adapter.records(kind):
                        yield assets.capture(record, copy_bytes=False) if kind == 'nodes' else record

                target = lifetime.enter_context(inventory(package.root / 'target.sqlite', read,
                    shared_records=shared_records(self.cfg, adapter, assets)))
                selection = select_spaces(package.manifest, spaces, self.cfg, adapter, target, indexes.vector)
                confirm_selection(selection, shared_selection)
                if selection['action'] != 'import':
                    discard_shared(incoming)
                assets.validate(package, incoming, check_declared=False)
                adapter.validate_restore(incoming)
                indexes.validate_existing()
                from service.migration.rebuild import verify_vector_inventory
                if indexes.vector is not None:
                    verify_vector_inventory(indexes.vector, target)
                indexes.validate_records(incoming)

            def apply():
                nonlocal plan
                # Conflict policies may invoke a model. Execute them only once
                # the complete recovery point is durable, then validate outputs
                # before their first business write.
                resolver = PolicyConflictResolver(target, llm,
                    card_format_enabled=bool(getattr(self.cfg, 'memory_card_format_enabled', False)))
                plan = MergePlan(package.root / 'merged.sqlite', target, incoming, resolver,
                                 applied=recovery.gate.applied)
                lifetime.callback(plan.close)
                adapter.validate_restore(plan.inventory)
                validate_relation_endpoints(self.cfg, plan.inventory)
                indexes.validate_records(plan.inventory)
                assets.validate_merged(package, plan.inventory.records('nodes'))
                # Resolve attachments before entering the backend mutation scope;
                # preparation failures must not become unknown remote writes.
                plan.db.execute('CREATE TABLE restored_nodes(body TEXT NOT NULL)')
                for record in plan.changes('nodes'):
                    restored = assets.restore(package, record)
                    plan.db.execute('INSERT INTO restored_nodes VALUES(?)', (json.dumps(restored),))
                progress(phase='merging', counts=plan.counts)
                adapter.merge_nodes(json.loads(body) for (body,) in plan.db.execute('SELECT body FROM restored_nodes'))
                if selection['action'] == 'import':
                    adapter.restore_space_catalog(spaces.rows)
                adapter.restore('archives', plan.changes('archives'))
                plan.db.execute('CREATE TABLE affected_sources(source TEXT PRIMARY KEY)')
                plan.db.executemany('INSERT OR IGNORE INTO affected_sources VALUES(?)',
                                    ((record['from_uri'],) for record in plan.changes('relations')))
                edges = (json.loads(body) for (body,) in plan.db.execute(
                    "SELECT body FROM records WHERE kind='relations' AND "
                    "json_extract(body,'$.from_uri') IN (SELECT source FROM affected_sources) ORDER BY key"))
                adapter.replace_relations(edges)
                # Delete only obsolete levels of changed nodes. Unchanged vectors
                # retain their actual values; newly imported vectors use target models.
                for record in plan.changes('nodes'):
                    old = lookup(target, 'nodes', record_key('nodes', record))
                    if old and indexes.vector is not None:
                        new_ids = {item.id for item in records_for_node(record)}
                        obsolete = [item.id for item in records_for_node(old) if item.id not in new_ids]
                        if obsolete:
                            indexes.vector.delete(obsolete)
                count = indexes.rebuild(plan.inventory, progress, changed=lambda: plan.changes('nodes'))
                if selection['action'] == 'import':
                    adapter.publish_space_projections(spaces.records('space_nodes'), indexes.vector)
                return {'data': plan.inventory.summary(), 'changes': plan.counts, 'index_records': count,
                        'shared_spaces': selection}

            def verify():
                def read(kind):
                    for record in adapter.records(kind):
                        yield assets.capture(record, copy_bytes=False) if kind == 'nodes' else record
                with inventory(package.root / 'verified.sqlite', read,
                        shared_records=shared_records(self.cfg, adapter, assets)) as actual:
                    if actual.summary() != plan.inventory.summary():
                        raise PackageError('merged data differs from the validated plan')
                indexes.verify(plan.inventory)
                if selection['action'] == 'import':
                    adapter.verify_space_catalog(spaces)

            return recovery.run(task, apply, verify, prepare=prepare,
                                receipts=lambda: plan.receipts(), progress=progress)
