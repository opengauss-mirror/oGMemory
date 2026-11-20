"""Whitelisted shared-space business rows; no credentials or source job state."""
import hashlib
import json
from datetime import datetime

from psycopg2 import sql
from psycopg2.extras import Json

from core.sql_payload import validate_sql_strings
from service.migration.package import PackageError
from service.migration.records import iso

# Order follows the production foreign keys. Only these tables can be restored.
COLUMNS = {
    'spaces': 'account_id space_id name owner_user_id state authz_rev created_at updated_at deleted_at',
    'space_members': 'account_id space_id principal user_id agent_id role created_at updated_at',
    'space_source_bindings': 'account_id space_id binding_id principal memory_types authorized_by grant_rev state created_at updated_at deleted_at',
    'space_rules': 'account_id space_id rule_id name rule_rev state created_at updated_at deleted_at',
    'space_rule_revisions': 'account_id space_id name state rule_id rule_rev spec spec_hash created_at updated_at',
    'space_items': 'account_id space_id topic_key evidence_fingerprint item_id rule_id rule_rev uri kind state acl_mode acl_origin quality_score last_used_at expires_at purge_after policy_snapshot superseded_by created_at updated_at checked_rule_rev',
    'space_item_sources': 'account_id space_id item_id binding_id source_uri grant_rev source_content_rev source_hash created_at updated_at source_epoch',
    'space_read_acl': 'account_id space_id item_id principal created_at updated_at',
}
COLUMNS = {table: tuple(fields.split()) for table, fields in COLUMNS.items()}
KEYS = {
    'spaces': (), 'space_members': ('principal',), 'space_source_bindings': ('binding_id',),
    'space_rules': ('rule_id',), 'space_rule_revisions': ('rule_id', 'rule_rev'),
    'space_items': ('item_id',), 'space_item_sources': ('item_id', 'binding_id', 'source_uri'),
    'space_read_acl': ('item_id', 'principal'),
}
JSON_FIELDS = {'memory_types', 'spec', 'policy_snapshot'}
NULLABLE = {'deleted_at', 'last_used_at', 'purge_after', 'superseded_by', 'checked_rule_rev',
            'user_id', 'agent_id'}
INTEGER_FIELDS = {'authz_rev', 'grant_rev', 'rule_rev', 'source_content_rev', 'checked_rule_rev'}
STATES = {
    'spaces': {'ACTIVE', 'ARCHIVED'}, 'space_source_bindings': {'ACTIVE', 'PAUSED'},
    'space_rules': {'ACTIVE', 'PAUSED', 'ARCHIVED'},
    'space_rule_revisions': {'ACTIVE', 'PAUSED', 'ARCHIVED'},
    'space_items': {'ACTIVE', 'RECOMPUTING', 'SUPERSEDED', 'EXPIRED', 'ARCHIVED', 'PURGED'},
}


def normalize(table, row):
    if not isinstance(table, str) or table not in COLUMNS or not isinstance(row, dict) or set(row) != set(COLUMNS[table]):
        raise PackageError('unsupported shared-space record fields')
    row = {key: iso(value) if isinstance(value, datetime) else value for key, value in row.items()}
    try:
        validate_sql_strings(row)
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PackageError('invalid shared-space record value') from exc
    for key, value in row.items():
        if value is None and key in NULLABLE:
            continue
        if key in JSON_FIELDS:
            expected = list if key == 'memory_types' else dict
            if not isinstance(value, expected):
                raise PackageError('invalid shared-space JSON value')
        elif key in INTEGER_FIELDS:
            if type(value) is not int or value < 1:
                raise PackageError('invalid shared-space revision')
        elif key == 'quality_score':
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise PackageError('invalid shared-space quality score')
            row[key] = float(value)
        elif key.endswith('_at') or key == 'purge_after':
            row[key] = iso(value)
        elif not isinstance(value, str):
            raise PackageError('invalid shared-space text value')
        else:
            limit = 512 if key in {'uri', 'source_uri'} else 134 if key == 'principal' else 128
            if key in {'space_id', 'rule_id', 'item_id', 'binding_id', 'superseded_by',
                       'spec_hash', 'source_hash', 'evidence_fingerprint'}:
                limit = 64
            if len(value) > limit:
                raise PackageError('shared-space field exceeds storage limit')
    if not row['account_id'] or not row['space_id']:
        raise PackageError('invalid shared-space identity')
    if table in STATES and row['state'] not in STATES[table]:
        raise PackageError('invalid shared-space state')
    if table == 'space_members':
        kind = 'user' if row['user_id'] is not None else 'agent'
        member = row[kind + '_id']
        if (not member or row[('agent' if kind == 'user' else 'user') + '_id'] is not None
                or row['principal'] != kind + ':' + member or row['role'] not in {'manager', 'member'}
                or (row['role'] == 'manager' and kind != 'user')):
            raise PackageError('invalid shared-space member')
    if table == 'space_rule_revisions':
        from server.space_rules import digest
        if digest(row['spec']) != row['spec_hash']:
            raise PackageError('shared-space rule checksum mismatch')
    if table == 'space_items' and (row['kind'] not in {'experience', 'skill'}
            or row['acl_mode'] not in {'inherit', 'restricted'} or row['acl_origin'] not in {'rule', 'manual'}):
        raise PackageError('invalid shared-space item policy')
    return row


def record_key(table, row):
    return json.dumps([row[key] for key in ('account_id', 'space_id', *KEYS[table])], separators=(',', ':'))


class SpaceCatalog:
    def __init__(self, adapter):
        self.adapter = adapter
        self.conn = adapter.conn

    def description(self):
        layout = []
        with self.conn.cursor() as cur:
            for table, columns in COLUMNS.items():
                cur.execute("SELECT column_name,data_type,character_maximum_length,is_nullable "
                            "FROM information_schema.columns WHERE table_schema=current_schema() "
                            "AND table_name=%s ORDER BY column_name", (table,))
                fields = [list(row) for row in cur]
                if {row[0] for row in fields} != set(columns):
                    raise PackageError('shared-space SQL schema is missing or incompatible')
                layout.append([table, fields])
        return {'database_type': self.adapter.dialect.name,
                'schema_sha256': hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()}

    def records(self):
        self.description()
        for table, columns in COLUMNS.items():
            for row in self.adapter._rows(table, columns):
                yield {'table': table, 'row': normalize(table, row)}

    def restore(self, reader):
        """Restore a prevalidated catalog on an empty, exclusively maintained target.

        reader(table) must repeatably yield business row dictionaries; the caller
        validates references, target identities and the full target-empty condition.
        """
        self.description()
        try:
            with self.conn.cursor() as cur:
                for table in COLUMNS:
                    cur.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(table)))
                    if cur.fetchone():
                        raise PackageError('shared-space catalog target is not empty')
                for table, columns in COLUMNS.items():
                    command = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                        sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
                        sql.SQL(',').join(sql.Placeholder() for _ in columns))
                    for raw in reader(table):
                        row = normalize(table, raw)
                        if table == 'space_rules' and row['state'] == 'ACTIVE':
                            row = {**row, 'state': 'PAUSED'}
                        cur.execute("SELECT set_config('app.account_id',%s,true)", (row['account_id'],))
                        cur.execute(command, [Json(row[key]) if key in JSON_FIELDS else row[key] for key in columns])
                # Fresh paused rules have no inherited queue, lease or progress.
                for rule in reader('space_rules'):
                    cur.execute("SELECT set_config('app.account_id',%s,true)", (rule['account_id'],))
                    if rule['state'] == 'ACTIVE':
                        # list_rules reads the current revision's state. Keep
                        # its published snapshot consistent with the paused rule.
                        cur.execute("UPDATE space_rule_revisions SET state='PAUSED' "
                                    "WHERE account_id=%s AND space_id=%s AND rule_id=%s AND rule_rev=%s",
                                    (rule['account_id'], rule['space_id'], rule['rule_id'], rule['rule_rev']))
                    cur.execute('INSERT INTO space_rule_runtime(account_id,space_id,rule_id) VALUES(%s,%s,%s)',
                                (rule['account_id'], rule['space_id'], rule['rule_id']))
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
