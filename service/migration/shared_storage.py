"""Migration content adapter for SQL catalogs used by shared-space deployments."""
from contextlib import contextmanager

from service.migration.records import NODE_FIELDS, normalize_node
from service.migration.storage import AGFSStorage, SQLStorage


def portable_space_node(record):
    # Projection readiness belongs to the target; source jobs are not migrated.
    runtime = {'shared_projection_ready', 'indexed_content_rev',
               'shared_projection_event_id', 'job_id'}
    return {**record, 'metadata': {key: value for key, value in record['metadata'].items()
                                 if key not in runtime}}


class SharedStorage(SQLStorage):
    """Use the same SQL catalog and optional immutable AGFS bodies as the service."""

    supported_extra_tables = frozenset({'spaces', 'space_items', 'space_jobs'})
    snapshot_events = (*SQLStorage.snapshot_events, 'DELETE_CONTEXT')

    def __init__(self, cfg):
        self._preserving = False
        self.bridge = self.pool = self.files = None
        super().__init__(cfg)
        self.backend = cfg.storage_backend
        try:
            if self.backend == 'agfs':
                from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
                from fs.sql_adapter.pool import SharedConnectionPool

                self.files = AGFSStorage(cfg)
                self.client, self.prefix = self.files.client, self.files.prefix
                self.pool = SharedConnectionPool(cfg.sql_connection_string, pool_size=1,
                                                 dialect=cfg.sql_dialect or 'auto')
                self.bridge = AGFSSpaceBridge(self.pool, self.client, self.prefix)
        except BaseException:
            self.close()
            raise

    def _commit(self):
        if not self._preserving:
            super()._commit()

    @contextmanager
    def _node_write(self):
        if self._preserving:
            yield
            return
        # INSERT would otherwise generate a fresh shared_source_epoch, breaking
        # the space_item_sources references captured in the same source snapshot.
        self.conn.commit()
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT tgenabled FROM pg_trigger WHERE "
                            "tgrelid='context_nodes'::regclass AND tgname='ogmemory_space_source_revision'")
                row = cur.fetchone()
                enabled = row is not None and row[0] == 'O'
                if enabled:
                    cur.execute('ALTER TABLE context_nodes DISABLE TRIGGER ogmemory_space_source_revision')
            self._preserving = True
            yield
            if enabled:
                with self.conn.cursor() as cur:
                    cur.execute('ALTER TABLE context_nodes ENABLE TRIGGER ogmemory_space_source_revision')
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        finally:
            self._preserving = False

    def _rows(self, table, columns, where=''):
        if table == 'context_nodes' and tuple(columns) == NODE_FIELDS and not where:
            where = "WHERE owner_space NOT LIKE 'space:%'"
        for row in super()._rows(table, columns, where):
            if table == 'context_nodes' and tuple(columns) == NODE_FIELDS:
                if self.bridge is not None and not row['metadata'].get('agfs_purged'):
                    self.bridge._load_row(row)
                row['metadata'] = {key: value for key, value in row['metadata'].items()
                                   if key != 'agfs_content'}
            yield row

    def records(self, kind):
        if kind == 'space_nodes':
            for row in self._rows('context_nodes', NODE_FIELDS, "WHERE owner_space LIKE 'space:%'"):
                yield portable_space_node(normalize_node(row))
            self._commit()
            return
        for row in super().records(kind):
            if kind != 'nodes' or not row['owner_space'].startswith('space:'):
                yield row

    def _stage(self, records):
        for record in records:
            if self.bridge is None or record['metadata'].get('agfs_purged'):
                yield record
            else:
                pointer = self.bridge.store_shared(record['uri'], record['content'],
                                                   record['abstract'], record['overview'], record['metadata'])
                yield {**record, 'metadata': {**record['metadata'], **pointer}}

    def restore(self, kind, records):
        if kind in {'nodes', 'space_nodes'}:
            if kind == 'space_nodes':
                records = (portable_space_node(record) for record in records)
            with self._node_write():
                super().restore('nodes', self._stage(records))
        else:
            super().restore(kind, records)

    def publish_space_projections(self, records, vector):
        """Mark target projections ready only after readback, under maintenance."""
        from psycopg2.extras import Json

        from service.migration.package import PackageError
        from service.migration.rebuild import verify_node_projection

        try:
            with self.conn.cursor() as cur:
                for record in records:
                    if not verify_node_projection(vector, record):
                        continue
                    cur.execute("SELECT set_config('app.account_id',%s,true)", (record['account_id'],))
                    cur.execute("UPDATE context_nodes SET metadata=metadata || %s::jsonb "
                                "WHERE uri=%s AND account_id=%s AND owner_space=%s AND version=%s",
                                (Json({'shared_projection_ready': True,
                                       'indexed_content_rev': record['version']}),
                                 record['uri'], record['account_id'], record['owner_space'], record['version']))
                    if cur.rowcount != 1:
                        raise PackageError('shared-space content changed before projection verification')
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def merge_nodes(self, records):
        with self._node_write():
            super().merge_nodes(self._stage(records))

    def restore_space_catalog(self, reader):
        from service.migration.space_catalog import SpaceCatalog
        SpaceCatalog(self).restore(reader)

    def verify_space_catalog(self, expected_inventory):
        import json
        import tempfile
        from pathlib import Path

        from service.migration.package import PackageError
        from service.migration.space_catalog import COLUMNS, SpaceCatalog, record_key
        from service.migration.space_inventory import SpaceInventory

        with tempfile.TemporaryDirectory() as root:
            actual = SpaceInventory(Path(root) / 'actual.sqlite', SpaceCatalog(self).records())
            try:
                for table in COLUMNS:
                    expected_count = 0
                    for record in expected_inventory.rows(table):
                        expected_count += 1
                        expected = dict(record)
                        if table == 'space_rules' and expected['state'] == 'ACTIVE':
                            expected['state'] = 'PAUSED'
                        if table == 'space_rule_revisions':
                            rule = expected_inventory._find('space_rules', record)
                            if expected['rule_rev'] == rule['rule_rev'] and rule['state'] == 'ACTIVE':
                                expected['state'] = 'PAUSED'
                        found = actual.db.execute('SELECT body FROM catalog WHERE kind=? AND key=?',
                                                  (table, record_key(table, record))).fetchone()
                        if found is None or json.loads(found[0]) != expected:
                            raise PackageError('restored shared-space catalog differs from package')
                    if actual.db.execute('SELECT count(*) FROM catalog WHERE kind=?', (table,)).fetchone()[0] != expected_count:
                        raise PackageError('restored shared-space catalog count differs from package')
            finally:
                actual.close()

    def _ls(self, path):
        return self.files._ls(path)

    def _mkdir(self, path):
        return self.files._mkdir(path)

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        if self.pool is not None:
            self.pool.close_all()
        if self.files is not None:
            self.files.close()
        super().close()
