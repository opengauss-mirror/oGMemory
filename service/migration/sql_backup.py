"""Internal same-database recovery, preserving SQL types with COPY CSV.

Only explicitly owned tables are touched. The caller supplies an administrator
connection and holds instance maintenance. Known projection triggers are suspended
transactionally during restore; constraints and unrelated tables stay intact.
"""
import csv
import os
import tempfile
from itertools import groupby
from pathlib import Path

from psycopg2 import extensions, sql

from service.migration.durability import sync_directory
from service.migration.package import PackageError
from service.migration.vector_backup import _digest


class SQLBackup:
    def __init__(self, connection, schema, tables, identity):
        self.conn = connection
        self.schema = schema
        self.tables = tuple(tables)
        self.identity = identity
        if not tables or len(set(tables)) != len(tables):
            raise ValueError("distinct owned tables are required")

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    def _layout(self):
        result = []
        if "context_nodes" in self.tables and not {"edge_projection_jobs", "relation_edges"}.issubset(self.tables):
            raise PackageError("context recovery must include its trigger-maintained relation tables")
        with self.conn.cursor() as cur:
            for name in self.tables:
                cur.execute("SELECT c.oid FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relname=%s AND c.relkind='r'", (self.schema, name))
                row = cur.fetchone()
                if not row:
                    raise PackageError("SQL recovery table is missing")
                oid = row[0]
                cur.execute("SELECT tgname,tgenabled FROM pg_trigger WHERE tgrelid=%s AND NOT tgisinternal ORDER BY tgname", (oid,))
                triggers = [list(row) for row in cur]
                allowed = {"trg_invalidate_edge_projections", "trg_remove_edge_projections",
                           "ogmemory_space_source_revision", "ogmemory_space_source_change",
                           "ogmemory_queue_agfs_cleanup"} if name == "context_nodes" else set()
                if any(trigger not in allowed or enabled not in {'O', 'D'} for trigger, enabled in triggers):
                    raise PackageError("SQL recovery table has unsupported side-effect triggers")
                if any(trigger in {'ogmemory_queue_agfs_cleanup', 'ogmemory_space_source_change'}
                       for trigger, _ in triggers) and 'outbox_events' not in self.tables:
                    raise PackageError("context recovery must include its AGFS cleanup queue")
                cur.execute("SELECT attname,format_type(atttypid,atttypmod),attnotnull FROM pg_attribute WHERE attrelid=%s AND attnum>0 AND NOT attisdropped ORDER BY attnum", (oid,))
                result.append({"table": name, "oid": oid, "columns": [list(r) for r in cur], "triggers": triggers})
        return result

    def _order(self, layout):
        names = {table["oid"]: table["table"] for table in layout}
        dependencies = {name: set() for name in names.values()}
        with self.conn.cursor() as cur:
            cur.execute("SELECT conrelid,confrelid FROM pg_constraint WHERE contype='f'")
            for child, parent in cur:
                if parent in names and child not in names:
                    raise PackageError("SQL backup scope has an external dependent table")
                if child in names and parent in names and child != parent:
                    dependencies[names[child]].add(names[parent])
        ordered = []
        while dependencies:
            ready = sorted(name for name, parents in dependencies.items() if not parents)
            if not ready:
                raise PackageError("cyclic SQL table dependencies require a dedicated recovery adapter")
            for name in ready:
                ordered.append(name)
                dependencies.pop(name)
            for parents in dependencies.values():
                parents.difference_update(ready)
        return ordered

    def _copy_out(self, name, file):
        # Stable order permits byte-for-byte verification independent of heap order.
        query = sql.SQL("COPY (SELECT t.* FROM {} t ORDER BY row_to_json(t)::text) TO STDOUT WITH CSV").format(self._table(name))
        with self.conn.cursor() as cur:
            cur.copy_expert(query.as_string(self.conn), file)

    def capture(self, destination):
        destination = Path(destination)
        destination.mkdir(mode=0o700)
        resource = self.identity()
        layout = self._layout()
        ordered = self._order(layout)
        files = {}
        sequences = []
        try:
            for number, name in enumerate(ordered):
                filename = f"table-{number}.csv"
                with (destination / filename).open("xb") as stream:
                    self._copy_out(name, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                files[name] = {"file": filename, "sha256": _digest(destination / filename)}
                with self.conn.cursor() as cur:
                    columns = next(t["columns"] for t in layout if t["table"] == name)
                    for column, _, _ in columns:
                        cur.execute("SELECT pg_get_serial_sequence(%s,%s)", (self._table(name).as_string(self.conn), column))
                        sequence = cur.fetchone()[0]
                        if sequence:
                            cur.execute("SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=%s::regclass", (sequence,))
                            namespace, seq_name = cur.fetchone()
                            cur.execute(sql.SQL("SELECT last_value,is_called FROM {}").format(sql.Identifier(namespace, seq_name)))
                            value, called = cur.fetchone()
                            sequences.append({"table": name, "column": column, "sequence": sequence, "value": value, "called": called})
            self.conn.rollback()  # Capture is read-only; do not retain a transaction.
        except BaseException:
            self.conn.rollback()
            raise
        sync_directory(destination)
        manifest = {"resource": resource, "schema": self.schema, "layout": layout,
                    "order": ordered, "files": files, "sequences": sequences}
        self.verify_live(destination, manifest)
        return manifest

    def validate(self, destination, manifest):
        if self.identity() != manifest["resource"] or manifest["schema"] != self.schema or self._layout() != manifest["layout"]:
            raise PackageError("SQL recovery target or table layout differs from backup")
        if self._order(manifest["layout"]) != manifest["order"] or set(manifest["files"]) != set(self.tables):
            raise PackageError("SQL recovery table scope differs from backup")
        for number, name in enumerate(manifest["order"]):
            entry = manifest["files"][name]
            if entry["file"] != f"table-{number}.csv" or _digest(Path(destination) / entry["file"]) != entry["sha256"]:
                raise PackageError("SQL backup checksum mismatch")
        with self.conn.cursor() as cur:
            for entry in manifest["sequences"]:
                if entry["table"] not in self.tables:
                    raise PackageError("SQL sequence outside recovery scope")
                cur.execute("SELECT pg_get_serial_sequence(%s,%s)", (self._table(entry["table"]).as_string(self.conn), entry["column"]))
                if cur.fetchone()[0] != entry["sequence"]:
                    raise PackageError("SQL sequence identity differs from backup")

    def _copy_in(self, cur, name, path, columns):
        command = sql.SQL("COPY {} FROM STDIN WITH CSV").format(self._table(name))
        names = [column[0] for column in columns]
        if 'account_id' not in names:
            with path.open('rb') as stream:
                cur.copy_expert(command.as_string(self.conn), stream)
            return
        # openGauss enforces account CHECK constraints even for SYSADMIN.
        # COPY must use the stored account, including during rollback. Preserve
        # CSV values exactly; a quoted empty string differs from SQL NULL.
        account_index = names.index('account_id')
        csv.field_size_limit(2**31 - 1)
        encoding = extensions.encodings[self.conn.encoding]
        with path.open('r', encoding=encoding, newline='') as source:
            def rows():
                lines = []
                def capture_lines():
                    for line in source:
                        lines.append(line)
                        yield line
                for row in csv.reader(capture_lines()):
                    yield row[account_index], ''.join(lines)
                    lines.clear()
            for account, group in groupby(rows(), key=lambda row: row[0]):
                cur.execute("SELECT set_config('app.account_id',%s,true)", (account,))
                with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode='w+', encoding=encoding, newline='') as stream:
                    for _, raw in group:
                        stream.write(raw)
                    stream.seek(0)
                    cur.copy_expert(command.as_string(self.conn), stream)

    def restore(self, destination, manifest):
        try:
            self.validate(destination, manifest)
            with self.conn.cursor() as cur:
                # Replay stored rows exactly: source epochs must not regenerate,
                # and deleting imported nodes must not enqueue new GC work.
                for table in manifest['layout']:
                    for trigger, enabled in table['triggers']:
                        if enabled == 'O':
                            cur.execute(sql.SQL('ALTER TABLE {} DISABLE TRIGGER {}').format(
                                self._table(table['table']), sql.Identifier(trigger)))
                for name in reversed(manifest["order"]):
                    cur.execute(sql.SQL("DELETE FROM {}").format(self._table(name)))
                for name in manifest["order"]:
                    columns = next(table['columns'] for table in manifest['layout'] if table['table'] == name)
                    self._copy_in(cur, name, Path(destination) / manifest['files'][name]['file'], columns)
                for entry in manifest["sequences"]:
                    cur.execute("SELECT setval(%s::regclass,%s,%s)", (entry["sequence"], entry["value"], entry["called"]))
                for table in manifest['layout']:
                    for trigger, enabled in table['triggers']:
                        if enabled == 'O':
                            cur.execute(sql.SQL('ALTER TABLE {} ENABLE TRIGGER {}').format(
                                self._table(table['table']), sql.Identifier(trigger)))
            self.conn.commit()
            self.verify_live(destination, manifest)
        except BaseException:
            self.conn.rollback()
            raise

    def verify_live(self, destination, manifest):
        self.validate(destination, manifest)
        for name in manifest["order"]:
            with tempfile.TemporaryFile() as stream:
                self._copy_out(name, stream)
                stream.seek(0)
                import hashlib
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != manifest["files"][name]["sha256"]:
                    raise PackageError("restored SQL data differs from backup")
        self.conn.rollback()
