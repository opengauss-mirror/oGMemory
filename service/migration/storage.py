"""Strict offline canonical storage adapters. Never run extraction or workers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from itertools import groupby
from typing import Protocol

from service.migration.package import PackageError
from service.migration.records import (
    ARCHIVE_FIELDS,
    EDGE_FIELDS,
    MIRRORS,
    NODE_FIELDS,
    RUNTIME_META,
    archive_to_node,
    archive_uri,
    identity,
    node_to_archive,
    normalize_archive,
    normalize_node,
    runtime_node,
    write_operation_path,
)


class MigrationStorage(Protocol):
    """Canonical content contract; adapters own layout, permissions and types."""

    backend: str

    def check(self, *, empty: bool = False) -> None: ...
    def description(self) -> dict: ...
    def validate_restore(self, inv) -> None: ...
    def records(self, kind: str) -> Iterator[dict]: ...
    def restore(self, kind: str, records: Iterable[dict]) -> None: ...
    def close(self) -> None: ...


class SQLStorage:
    backend = "sql"

    def __init__(self, cfg):
        import psycopg2

        if not cfg.sql_connection_string:
            raise PackageError("storage.connection_string is required")
        self.cfg = cfg
        self.conn = psycopg2.connect(cfg.sql_connection_string)
        from fs.sql_adapter.pool import resolve_connection_dialect
        from fs.sql_adapter.schema import ensure_schema

        try:
            self.dialect = resolve_connection_dialect(
                cfg.sql_dialect or "auto", cfg.sql_connection_string, 10
            )
            # Both roles bypass forced tenant RLS on their respective engines.
            privilege = (
                "rolsuper OR rolsystemadmin" if self.dialect.name == "opengauss" else "rolsuper"
            )
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT {privilege} FROM pg_roles WHERE rolname = current_user")
                if not cur.fetchone()[0]:
                    raise PackageError(
                        "all-account SQL migration requires a database administrator with RLS bypass"
                    )
            ensure_schema(self.conn, self.dialect)
        except Exception:
            self.conn.close()
            raise

    def validate_restore(self, inv):
        # SQL identity constraints are checked by the canonical record validator.
        pass

    def description(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = re.search(
                r"(?:openGauss(?:-lite)?|PostgreSQL)\s+([0-9][0-9A-Za-z.+_-]*)",
                cur.fetchone()[0],
                re.I,
            )
        return {
            "backend": self.backend,
            "database_type": self.dialect.name,
            "database_version": version[1] if version else None,
        }

    def close(self):
        self.conn.close()

    def _rows(self, table, columns, where=""):
        # Identifiers and predicates are internal constants, never request input.
        with self.conn.cursor(name="migration_" + table) as cur:
            cur.itersize = 256
            cur.execute(f"SELECT {','.join(columns)} FROM {table} {where}")
            for row in cur:
                yield dict(zip(columns, row, strict=True))

    def check(self, *, empty=False):
        if empty and self.cfg.scene_enabled:
            raise PackageError("scene extensions are not supported by migration v1")
        with self.conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname=current_schema()")
            tables = {r[0] for r in cur}
            unsupported = {
                "graph_nodes",
                "graph_edges",
                "scene_scopes",
                "scene_snapshot_history",
                "skill_candidates",
                "skill_candidate_evidence",
                "skill_example_evidence",
                "tool_result_artifacts",
            }
            for table in sorted(tables & unsupported):
                cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
                if cur.fetchone():
                    raise PackageError(f"unsupported business data: {table}")
            cur.execute(
                "SELECT 1 FROM outbox_events WHERE status IN ('PENDING','PROCESSING','FAILED','DLQ') "
                "AND event_type NOT IN ('UPSERT_CONTEXT','PROJECT_GRAPH') LIMIT 1"
            )
            if cur.fetchone():
                raise PackageError("unfinished business operations; finish them before migration")
            if empty:
                for table in (
                    "context_nodes",
                    "relation_edges",
                    "session_archives",
                    "outbox_events",
                    "edge_projection_jobs",
                ):
                    cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    if cur.fetchone():
                        raise PackageError("target storage is not empty")
        self.conn.commit()

    def records(self, kind):
        if kind == "nodes":
            for r in self._rows("context_nodes", NODE_FIELDS):
                if runtime_node(r):
                    continue
                if identity(r["uri"])[1][0] == "sessions":
                    raise PackageError(
                        "SQL archive stored in context_nodes; reconcile duplicate layouts first"
                    )
                yield normalize_node(r)
        elif kind == "archives":
            for r in self._rows("session_archives", ARCHIVE_FIELDS):
                yield normalize_archive(r)
        else:
            yield from self._rows("relation_edges", EDGE_FIELDS)
            # Match SQLRelationStore: use embedded edges only when this source
            # has no authoritative rows, rather than merging stale copies.
            for row in self._rows(
                "context_nodes",
                ("uri", "relations"),
                "WHERE NOT EXISTS (SELECT 1 FROM relation_edges re "
                "WHERE re.from_uri=context_nodes.uri AND re.account_id=context_nodes.account_id)",
            ):
                for edge in row["relations"]:
                    if edge["from_uri"] != row["uri"]:
                        raise PackageError("embedded relation source differs from node")
                    yield {k: edge[k] for k in EDGE_FIELDS}
        self.conn.commit()

    def restore(self, kind, records):
        from psycopg2.extras import Json

        columns = {"nodes": NODE_FIELDS, "archives": ARCHIVE_FIELDS, "relations": EDGE_FIELDS}[kind]
        table = {
            "nodes": "context_nodes",
            "archives": "session_archives",
            "relations": "relation_edges",
        }[kind]
        if kind == "relations":
            columns = (*columns, "account_id")
        with self.conn.cursor() as cur:
            for n, record in enumerate(records, 1):
                r = dict(record)
                if kind == "relations":
                    r["account_id"] = identity(r["from_uri"])[0]
                cur.execute("SELECT set_config('app.account_id', %s, true)", (r["account_id"],))
                values = [Json(r[k]) if k in {"metadata", "messages"} else r[k] for k in columns]
                cur.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))})",
                    values,
                )
                if kind == "relations":
                    cur.execute(
                        "UPDATE context_nodes SET relations=relations || %s::jsonb WHERE uri=%s",
                        (Json([record]), record["from_uri"]),
                    )
                if n % 256 == 0:
                    self.conn.commit()
        self.conn.commit()


class AGFSStorage:
    backend = "agfs"
    FILES = {"content.md", ".abstract.md", ".overview.md", ".meta.json", ".relations.json"}
    RUNTIME_DIRS = {".outbox-done", "_system", ".trace"}

    def __init__(self, cfg, client=None):
        from pyagfs import AGFSClient

        self.cfg = cfg
        self.client = client or AGFSClient(api_base_url=cfg.agfs_base_url)
        self.prefix = cfg.agfs_mount_prefix.rstrip("/")

    def description(self):
        return {"backend": self.backend, "database_type": None, "database_version": None}

    def close(self):
        pass

    def _ls(self, path):
        try:
            return self.client.ls(path)
        except Exception as exc:
            from fs.agfs_adapter.agfs_context_fs import _is_not_found_error

            if _is_not_found_error(exc):
                return []
            raise

    def _walk(self, path, *, include_runtime=False):
        entries = self._ls(path)
        names = {e["name"] for e in entries}
        if any(not isinstance(n, str) or n in {".", ".."} or "/" in n or "\\" in n for n in names):
            raise PackageError("invalid AGFS directory entry")
        if ".delete_tombstone.json" in names or any(
            n.startswith(".") and (n.endswith("-lock") or n.endswith(".tmp")) for n in names
        ):
            raise PackageError("unfinished AGFS mutation; recover before export")
        if ".meta.json" in names:
            raw = json.loads(self.client.read(path + "/.meta.json"))
            if raw.get("category") == "state" and identity(raw.get("uri", ""))[1][0] == "sessions":
                if include_runtime:
                    yield path, raw
                return  # Runtime session windows are intentionally excluded from exports.
            unknown = {e["name"] for e in entries if not e.get("isDir")} - self.FILES
            if unknown:
                raise PackageError(f"unsupported AGFS payload files: {sorted(unknown)}")
            yield path, raw
        elif any(not e.get("isDir") for e in entries):
            raise PackageError("unrecognized AGFS business files")
        for e in sorted(entries, key=lambda x: x["name"]):
            if e.get("isDir"):
                name = e["name"]
                relative = (path + "/" + name).removeprefix(self.prefix + "/accounts/")
                if write_operation_path(relative.split("/")[1:]):
                    if include_runtime:
                        yield from self._walk(path + "/" + name, include_runtime=True)
                    continue
                parts = relative.split("/")
                if len(parts) == 2 and name in self.RUNTIME_DIRS:
                    continue
                if len(parts) == 4 and parts[1] == "sessions" and name == "stage_results":
                    if include_runtime:
                        yield from self._walk(path + "/" + name, include_runtime=True)
                    continue
                if name == ".outbox" and (len(parts) == 2 or ".meta.json" in names):
                    self._check_outbox(path + "/" + name)
                    continue
                if name.startswith("."):
                    raise PackageError("unrecognized AGFS state directory")
                yield from self._walk(path + "/" + name, include_runtime=include_runtime)

    def _check_outbox(self, path):
        for e in self._ls(path):
            if e.get("isDir"):
                raise PackageError("outbox contains recovery/dead-letter state")
            event = json.loads(self.client.read(path + "/" + e["name"]))
            if event.get("event_type") not in {
                "UPSERT_CONTEXT",
                "PROJECT_GRAPH",
            }:
                raise PackageError("unfinished business operation in AGFS outbox")

    def _check_skill_data(self):
        root = self.prefix + "/.ogmemory/trace-store/accounts"
        for entry in self._ls(root):
            name = entry["name"]
            if not entry.get("isDir") or name in {".", ".."} or "/" in name or "\\" in name:
                raise PackageError("unrecognized AGFS trace state")
            account = root + "/" + name
            for collection in ("candidates", "example_evidence"):
                if self._ls(account + "/" + collection):
                    raise PackageError(f"unsupported AGFS business data: {collection}")
            # Legacy stores embed collections in state.json; a backup can be
            # the recoverable state when the main file is absent.
            for item in self._ls(account):
                if item["name"] in {"state.json", "state.json.bak"}:
                    state = json.loads(self.client.read(account + "/" + item["name"]))
                    if state.get("candidates") or state.get("example_evidence"):
                        raise PackageError("unsupported AGFS skill candidate/evidence data")

    def check(self, *, empty=False):
        if empty and self.cfg.scene_enabled:
            raise PackageError("scene extensions are not supported by migration v1")
        # A nonexistent source prefix is a configuration error, not an empty export.
        if not empty:
            self.client.ls(self.prefix or "/")
        self._check_skill_data()
        # Consume the scan before publishing a package or writing a target.
        for _ in self._walk(self.prefix + "/accounts", include_runtime=empty):
            if empty:
                raise PackageError("target storage is not empty")

    def validate_restore(self, inv):
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        # AGFS localfs uses filesystem names, whose capacity is measured in
        # bytes rather than the character limits of SQL identity columns.
        for kind in ("nodes", "archives"):
            for record in inv.records(kind):
                uri = archive_uri(record) if kind == "archives" else record["uri"]
                path = self.prefix + uri_to_path(uri)
                if any(len(part.encode("utf-8")) > 255 for part in path.split("/")):
                    raise PackageError("AGFS target path component exceeds 255 UTF-8 bytes")

    def _node(self, path, meta):
        account, _ = identity(meta["uri"])
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        if path.rstrip("/") != (self.prefix + uri_to_path(meta["uri"])).rstrip("/"):
            raise PackageError("AGFS node URI does not match its physical location")
        r = {
            k: meta.get(k)
            for k in NODE_FIELDS
            if k not in {"metadata", "content", "abstract", "overview"}
        }
        r["account_id"] = account
        r["metadata"] = {k: v for k, v in meta.items() if k not in MIRRORS | RUNTIME_META}
        for field, file in [
            ("content", "content.md"),
            ("abstract", ".abstract.md"),
            ("overview", ".overview.md"),
        ]:
            r[field] = self.client.read(path + "/" + file).decode("utf-8")
        return r

    def records(self, kind):
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        for path, meta in self._walk(self.prefix + "/accounts"):
            r = self._node(path, meta)
            is_archive = identity(r["uri"])[1][0] == "sessions" and r["category"] == "history"
            if kind == "nodes" and not is_archive:
                yield normalize_node(r)
            elif kind == "archives" and is_archive:
                yield node_to_archive(r)
            elif kind == "relations":
                names = {e["name"] for e in self._ls(path)}
                if ".relations.json" in names:
                    edges = json.loads(self.client.read(path + "/.relations.json"))
                    if not isinstance(edges, list):
                        raise PackageError("invalid relations file")
                    for edge in edges:
                        identity(edge["from_uri"])
                        source_path = self.prefix + uri_to_path(edge["from_uri"])
                        if source_path.rstrip("/") != path.rstrip("/"):
                            raise PackageError("relation source differs from owning node")
                        yield {k: edge[k] for k in EDGE_FIELDS}

    def _mkdir(self, path):
        current = ""
        for part in path.strip("/").split("/"):
            current += "/" + part
            try:
                self.client.ls(current)  # Mount roots already exist and cannot be mkdir'ed.
                continue
            except Exception as exc:
                from fs.agfs_adapter.agfs_context_fs import _is_not_found_error

                if not _is_not_found_error(exc):
                    raise
            self.client.mkdir(current)

    def restore(self, kind, records):
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        if kind == "relations":
            # Inventory orders relations by source URI; read/write once per group.
            for source_uri, group in groupby(records, key=lambda r: r["from_uri"]):
                path = self.prefix + uri_to_path(source_uri) + ".relations.json"
                edges = json.loads(self.client.read(path))
                edges.extend(group)
                self.client.write(path, json.dumps(edges, ensure_ascii=False).encode())
            return
        for r in records:
            if kind == "archives":
                r = archive_to_node(r)
            path = self.prefix + uri_to_path(r["uri"])
            self._mkdir(path)
            meta = {**r["metadata"], **{k: r[k] for k in MIRRORS if k in r and k != "account_id"}}
            for name, value in [
                ("content.md", r["content"]),
                (".abstract.md", r["abstract"]),
                (".overview.md", r["overview"]),
                (".relations.json", "[]"),
                (".meta.json", json.dumps(meta, ensure_ascii=False)),
            ]:
                self.client.write(path + name, value.encode("utf-8"))


@contextmanager
def storage(cfg):
    if cfg.storage_backend == "sql":
        adapter = SQLStorage(cfg)
    elif cfg.storage_backend == "agfs":
        adapter = AGFSStorage(cfg)
    else:
        raise PackageError("unsupported storage.backend")
    try:
        yield adapter
    finally:
        adapter.close()
