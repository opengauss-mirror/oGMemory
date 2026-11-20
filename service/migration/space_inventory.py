"""Disk-backed shared-space DTO and reference validation before target writes."""
import json
import sqlite3

from service.migration.package import PackageError
from service.migration.records import identity, normalize_node, validate_record
from service.migration.space_catalog import COLUMNS, normalize, record_key


class SpaceInventory:
    def __init__(self, path, rows=(), nodes=()):
        self.db = sqlite3.connect(path)
        try:
            self.db.execute('CREATE TABLE catalog (kind TEXT,key TEXT,body TEXT,PRIMARY KEY(kind,key))')
            self.db.execute('CREATE TABLE nodes (uri TEXT PRIMARY KEY,body TEXT)')
            for record in rows:
                if not isinstance(record, dict) or set(record) != {'table', 'row'}:
                    raise PackageError('invalid shared-space record envelope')
                table = record['table']
                if not isinstance(table, str) or table not in COLUMNS:
                    raise PackageError('unsupported shared-space table')
                row = normalize(table, record['row'])
                self.db.execute('INSERT INTO catalog VALUES(?,?,?)',
                                (table, record_key(table, row), json.dumps(row, sort_keys=True)))
            for record in nodes:
                validate_record('space_nodes', record)
                row = normalize_node(record)
                self.db.execute('INSERT INTO nodes VALUES(?,?)', (row['uri'], json.dumps(row, sort_keys=True)))
            self.db.commit()
        except sqlite3.IntegrityError as exc:
            self.db.close()
            raise PackageError('duplicate shared-space identity') from exc
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def rows(self, table):
        if table not in COLUMNS:
            raise PackageError('unsupported shared-space table')
        for (body,) in self.db.execute('SELECT body FROM catalog WHERE kind=? ORDER BY key', (table,)):
            yield json.loads(body)

    def records(self, kind):
        if kind == 'space_nodes':
            for (body,) in self.db.execute('SELECT body FROM nodes ORDER BY uri'):
                yield json.loads(body)
        elif kind == 'spaces':
            for table in COLUMNS:
                for row in self.rows(table):
                    yield {'table': table, 'row': row}
        else:
            raise PackageError('unsupported shared-space dataset')

    def _find(self, table, row):
        value = self.db.execute('SELECT body FROM catalog WHERE kind=? AND key=?',
                                (table, record_key(table, row))).fetchone()
        if value is None:
            raise PackageError('missing shared-space dependency: ' + table)
        return json.loads(value[0])

    def count(self):
        return self.db.execute("SELECT count(*) FROM catalog WHERE kind='spaces'").fetchone()[0]

    def validate_dependencies(self, ordinary):
        """Check tenant-bound references; identities on the target are separate."""
        for table in COLUMNS:
            for row in self.rows(table):
                account, space = row['account_id'], row['space_id']
                if table != 'spaces':
                    self._find('spaces', row)
                if table == 'space_rules':
                    self._find('space_rule_revisions', row)
                if table == 'space_rule_revisions':
                    self._find('space_rules', row)
                if table == 'space_items':
                    self._find('space_rule_revisions', row)
                    value = self.db.execute('SELECT body FROM nodes WHERE uri=?', (row['uri'],)).fetchone()
                    if value is None:
                        raise PackageError('shared-space item has no content node')
                    node = json.loads(value[0])
                    if node['account_id'] != account or node['owner_space'] != 'space:' + space:
                        raise PackageError('shared-space item content owner differs')
                    if row['superseded_by']:
                        self._find('space_items', {**row, 'item_id': row['superseded_by']})
                if table in {'space_item_sources', 'space_read_acl'}:
                    self._find('space_items', row)
                if table == 'space_read_acl':
                    self._find('space_members', row)
                if table == 'space_item_sources':
                    binding = self._find('space_source_bindings', row)
                    uri = row['source_uri']
                    if identity(uri)[0] != account:
                        raise PackageError('cross-account shared-space evidence')
                    found = ordinary.db.execute("SELECT body FROM records WHERE kind='nodes' AND key=?",
                                                (uri.rstrip('/'),)).fetchone()
                    if found is None:
                        raise PackageError('shared-space evidence is missing from ordinary memories')
                    node = json.loads(found[0])
                    if node['uri'] != uri or node['owner_space'] != binding['principal']:
                        raise PackageError('shared-space evidence does not match its authorized source')
        for node in self.records('space_nodes'):
            if not self.db.execute("SELECT 1 FROM catalog WHERE kind='space_items' "
                                   "AND json_extract(body,'$.uri')=?", (node['uri'],)).fetchone():
                raise PackageError('shared-space content node has no item')

    def target_identity_error(self, conn):
        """Return a skip reason without creating or changing target identities."""
        with conn.cursor() as cur:
            for table in ('spaces', 'space_members', 'space_source_bindings'):
                for row in self.rows(table):
                    account = row['account_id']
                    if table == 'spaces':
                        principals = [('user', row['owner_user_id'])]
                    elif table == 'space_members':
                        principals = [('user', row['user_id'])] if row['user_id'] else [('agent', row['agent_id'])]
                    else:
                        kind, _, ident = row['principal'].partition(':')
                        principals = [('user', row['authorized_by']), (kind, ident)]
                    for kind, ident in principals:
                        if kind not in {'user', 'agent'} or not ident:
                            raise PackageError('invalid shared-space principal')
                        if kind == 'user':
                            cur.execute("SELECT 1 FROM cp_users u JOIN cp_accounts a USING(account_id) "
                                        "WHERE u.account_id=%s AND u.user_id=%s AND u.status='active' AND a.status='active'",
                                        (account, ident))
                        else:
                            cur.execute("SELECT 1 FROM cp_agents g JOIN cp_users u ON u.account_id=g.account_id "
                                        "AND u.user_id=g.owner_user_id JOIN cp_accounts a ON a.account_id=g.account_id "
                                        "WHERE g.account_id=%s AND g.agent_id=%s AND g.status='active' "
                                        "AND u.status='active' AND a.status='active'", (account, ident))
                        if cur.fetchone() is None:
                            return 'target_identity_missing'
                    if table == 'space_source_bindings' and row['deleted_at'] is None and row['state'] == 'ACTIVE':
                        cur.execute("SELECT 1 FROM cp_users grantor WHERE grantor.account_id=%s "
                                    "AND grantor.user_id=%s AND (grantor.role='admin' "
                                    "OR %s='user:'||grantor.user_id OR EXISTS(SELECT 1 FROM cp_agents a "
                                    "WHERE a.account_id=grantor.account_id AND 'agent:'||a.agent_id=%s "
                                    "AND a.owner_user_id=grantor.user_id))",
                                    (account, row['authorized_by'], row['principal'], row['principal']))
                        if cur.fetchone() is None:
                            return 'target_source_authority_changed'
        return None
