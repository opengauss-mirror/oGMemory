"""Tenant-scoped shared-space configuration and authorized content reads."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from psycopg2.errors import UniqueViolation
from psycopg2.extras import Json, RealDictCursor, execute_values

from server.space_job_repository import SpaceJobRepository
from server.space_privacy import redact_item
from server.space_rules import (
    SpaceError,
    digest,
    identifier,
    principal,
    validate_acl,
    validate_rule,
)


def json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def lock_key(account_id, space_id=None):
    text = f'shared-space:{account_id}' if space_id is None else f'shared-space:{account_id}:{space_id}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big', signed=True)


class SpaceRepository(SpaceJobRepository):
    def __init__(self, pool, initialize=True, content_store=None):
        self.pool = pool
        self.dialect = pool.sql_dialect
        self.content_store = content_store
        self._cursor_key = hashlib.sha256(os.environ.get('OGMEM_SPACE_CURSOR_SECRET', str(pool.connection_string)).encode()).digest()
        if initialize:
            schema = Path(__file__).with_name('space_schema.sql').read_text(encoding='utf8')
            events = Path(__file__).parent.parent / 'fs/sql_adapter/space_events.sql'
            event_schema = events.read_text(encoding='utf8') if events.exists() else ''
            fingerprint = hashlib.sha256((schema + '\n' + event_schema).encode()).hexdigest()
            conn = pool.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute('SELECT pg_advisory_xact_lock(%s)', (lock_key('__schema__'),))
                    if not self.dialect.table_exists(cur, 'shared_space_schema_state'):
                        cur.execute('CREATE TABLE shared_space_schema_state (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint CHAR(64) NOT NULL)')
                    cur.execute('SELECT fingerprint FROM shared_space_schema_state WHERE id=1')
                    installed = cur.fetchone()
                    if not installed or installed[0] != fingerprint:
                        cur.execute(self.dialect.prepare_schema(schema))
                        if event_schema:
                            cur.execute(event_schema)
                        cur.execute('DELETE FROM shared_space_schema_state WHERE id=1')
                        cur.execute('INSERT INTO shared_space_schema_state(id,fingerprint) VALUES(1,%s)', (fingerprint,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                pool.return_connection(conn)

    @contextmanager
    def transaction(self, account_id, space_id=None, *, write=True):
        admission = getattr(self, '_migration_admission', None)
        with admission.scope(write=write) if admission is not None else nullcontext():
            conn = self.pool.get_connection()
            discard = False
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT set_config('app.account_id', %s, true)", (account_id,))
                    if write:
                        cur.execute('SELECT pg_advisory_xact_lock(%s)', (lock_key(account_id),))
                        if space_id:
                            cur.execute('SELECT pg_advisory_xact_lock(%s)', (lock_key(account_id, space_id),))
                    yield cur
                conn.commit()
            except BaseException as error:
                try:
                    conn.rollback()
                except BaseException as rollback_error:
                    discard = True
                    if admission is not None and write:
                        admission._operation.get().uncertain = True
                    raise error from rollback_error
                raise
            finally:
                if discard:
                    self.pool.discard_connection(conn)
                else:
                    self.pool.return_connection(conn)

    @staticmethod
    def _admin(ctx):
        return str(getattr(ctx.role, 'value', ctx.role)).upper() in ('ADMIN', 'ROOT')

    def _identity(self, cur, ctx):
        cur.execute('SELECT u.role,u.status FROM cp_users u JOIN cp_accounts a USING(account_id) WHERE u.account_id=%s AND u.user_id=%s AND a.status=%s', (ctx.account_id, ctx.user_id, 'active'))
        user = cur.fetchone()
        if not user or user['status'] != 'active':
            raise SpaceError(403, 'REGISTERED_USER_REQUIRED', 'An active registered user is required')
        return user

    def _space(self, cur, ctx, space_id, *, manage=False, include_deleted=False):
        user = self._identity(cur, ctx)
        cur.execute('SELECT s.*,m.role AS member_role FROM spaces s LEFT JOIN space_members m ON m.account_id=s.account_id AND m.space_id=s.space_id AND m.principal=%s WHERE s.account_id=%s AND s.space_id=%s', ('user:' + ctx.user_id, ctx.account_id, space_id))
        row = cur.fetchone()
        if not row or (row['deleted_at'] is not None and not include_deleted):
            raise SpaceError(404, 'SPACE_NOT_FOUND', 'Space not found')
        row['is_admin'] = self._admin(ctx) and user['role'] == 'admin'
        governance = row['owner_user_id'] == ctx.user_id or row['member_role'] == 'manager' or row['is_admin']
        if not governance and not row['member_role']:
            raise SpaceError(403, 'SPACE_ACCESS_DENIED', 'Space membership required')
        if manage and not governance:
            raise SpaceError(403, 'SPACE_MANAGEMENT_REQUIRED', 'Space management permission required')
        row['capabilities'] = {'manage_space': governance, 'delete_space': row['owner_user_id'] == ctx.user_id or row['is_admin'], 'manage_members': governance, 'manage_acl': governance, 'manage_rules': governance, 'create_rule': governance and row['state'] == 'ACTIVE', 'inspect': True, 'search': row['state'] == 'ACTIVE'}
        return row

    @staticmethod
    def _cas(actual, expected):
        if expected is None:
            raise SpaceError(428, 'PRECONDITION_REQUIRED', 'If-Match is required')
        if int(expected) != actual:
            raise SpaceError(412, 'REVISION_MISMATCH', 'Configuration changed; reload and save')

    @staticmethod
    def _limit(limit):
        return max(1, min(int(limit), 200))

    def _paging(self, cur, ctx, scope, cursor):
        cur.execute('SELECT authz_rev FROM cp_accounts WHERE account_id=%s', (ctx.account_id,))
        revision = cur.fetchone()['authz_rev']
        binding = digest([ctx.account_id, ctx.user_id, ctx.agent_id, scope, revision])
        after = ''
        if cursor:
            try:
                raw, signature = cursor.split('.')
                expected = hmac.new(self._cursor_key, raw.encode(), hashlib.sha256).hexdigest()
                value = json.loads(base64.urlsafe_b64decode(raw + '=' * (-len(raw) % 4)))
                if not hmac.compare_digest(signature, expected) or value['binding'] != binding or value['expires'] < time.time():
                    raise ValueError('Expired cursor')
                after = value['after']
            except (ValueError, KeyError, TypeError):
                raise SpaceError(400, 'INVALID_CURSOR', 'Reload the list to refresh its cursor') from None
        return after, binding

    def _page(self, rows, limit, key, binding):
        cursor = self.cursor_after(rows[limit - 1][key], binding) if len(rows) > limit else None
        return {'items': json_value(rows[:limit]), 'next_cursor': cursor}

    def cursor_after(self, after, binding):
        data = {'after': str(after), 'binding': binding, 'expires': int(time.time()) + 900}
        raw = base64.urlsafe_b64encode(json.dumps(data, separators=(',', ':')).encode()).decode().rstrip('=')
        return raw + '.' + hmac.new(self._cursor_key, raw.encode(), hashlib.sha256).hexdigest()

    def search_page_context(self, ctx, scope, cursor):
        with self.transaction(ctx.account_id, write=False) as cur:
            self._identity(cur, ctx)
            after, binding = self._paging(cur, ctx, ['shared-search', scope], cursor)
        if after and (not isinstance(after, str) or not after.isdecimal()):
            raise SpaceError(400, 'INVALID_CURSOR', 'Reload the search to refresh its cursor')
        return int(after or 0), binding

    def _mutate(self, ctx, space_id, operation, payload, expected_rev, request_id, action, *, manage=False, creating=False, deleting_space=False):
        if request_id and (not isinstance(request_id, str) or len(request_id) > 128):
            raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid idempotency key')
        with self.transaction(ctx.account_id, space_id) as cur:
            space = None
            if creating:
                self._identity(cur, ctx)
            else:
                space = self._space(cur, ctx, space_id, manage=manage, include_deleted=deleting_space)
                if deleting_space and not space['capabilities']['delete_space']:
                    raise SpaceError(403, 'SPACE_OWNER_REQUIRED', 'Space owner or tenant administrator required')
            fingerprint = digest({'payload': payload, 'expected_rev': expected_rev})
            if request_id:
                cur.execute('SELECT request_hash,response_body FROM space_requests WHERE account_id=%s AND actor_id=%s AND operation=%s AND request_key=%s AND expires_at>NOW()', (ctx.account_id, ctx.user_id, operation, request_id))
                old = cur.fetchone()
                if old:
                    if old['request_hash'] != fingerprint:
                        raise SpaceError(409, 'IDEMPOTENCY_CONFLICT', 'Idempotency key already used')
                    return old['response_body']
            try:
                result = json_value(action(cur, space))
            except UniqueViolation as exc:
                if exc.diag.constraint_name == 'uq_spaces_account_name':
                    raise SpaceError(409, 'SPACE_NAME_EXISTS', 'A shared space with this name already exists') from None
                raise
            cur.execute('UPDATE cp_accounts SET authz_rev=authz_rev+1 WHERE account_id=%s', (ctx.account_id,))
            details = {'space_id': space_id, 'request_id': request_id}
            if operation.startswith(('space.rule:', 'space.rule.delete:')):
                details['rule_id'] = operation.rsplit(':', 1)[-1]
            if operation.startswith(('space.source:', 'space.source.delete:')):
                details['binding_id'] = operation.rsplit(':', 1)[-1]
            if operation.startswith('space.acl:'):
                details['item_id'] = payload['item_id']
            cur.execute('INSERT INTO cp_audit_logs(log_id,account_id,actor,target,action,timestamp,result,trace_id,details) VALUES (%s,%s,%s,%s,%s,NOW(),%s,%s,%s)', (str(uuid4()), ctx.account_id, ctx.user_id, 'space:' + space_id, operation.split(':', 1)[0], 'success', ctx.trace_id, Json(details)))
            if request_id:
                cur.execute('DELETE FROM space_requests WHERE account_id=%s AND actor_id=%s AND operation=%s AND request_key=%s', (ctx.account_id, ctx.user_id, operation, request_id))
                cur.execute("INSERT INTO space_requests(account_id,actor_id,operation,request_key,request_hash,response_body,expires_at) VALUES (%s,%s,%s,%s,%s,%s,NOW()+INTERVAL '7 days')", (ctx.account_id, ctx.user_id, operation, request_id, fingerprint, Json(result)))
            return result

    @staticmethod
    def _dirty(cur, account_id, space_id):
        cur.execute('UPDATE space_rule_runtime SET dirty_generation=dirty_generation+1,next_run_at=NOW(),updated_at=NOW() WHERE account_id=%s AND space_id=%s', (account_id, space_id))

    @staticmethod
    def _cancel_jobs(cur, account_id, space_id, rule_id=None):
        cur.execute("""UPDATE space_jobs SET state='SUPERSEDED',claim_token=NULL,lease_until=NULL,
            result=result || %s::jsonb,snapshot=snapshot - 'sources',updated_at=NOW()
            WHERE account_id=%s AND space_id=%s AND (%s IS NULL OR rule_id=%s)
            AND state IN ('QUEUED','RUNNING')""",
            (Json({'reason': 'deleted'}), account_id, space_id, rule_id, rule_id))

    @staticmethod
    def _bump(cur, account_id, space_id):
        cur.execute('UPDATE spaces SET authz_rev=authz_rev+1,updated_at=NOW() WHERE account_id=%s AND space_id=%s RETURNING authz_rev', (account_id, space_id))
        return cur.fetchone()['authz_rev']

    @staticmethod
    def _registered(cur, account_id, subject):
        kind, identity = principal(subject)
        if kind == 'user':
            cur.execute("SELECT * FROM cp_users WHERE account_id=%s AND user_id=%s AND status='active'", (account_id, identity))
        else:
            cur.execute("""SELECT a.* FROM cp_agents a JOIN cp_users u
                ON u.account_id=a.account_id AND u.user_id=a.owner_user_id AND u.status='active'
                WHERE a.account_id=%s AND a.agent_id=%s AND a.status='active'""", (account_id, identity))
        row = cur.fetchone()
        if not row:
            raise SpaceError(400, 'REGISTERED_PRINCIPAL_REQUIRED', 'Select an active registered member')
        return kind, identity, row

    def create_space(self, ctx, payload, *, request_id=None, expected_rev=None):
        space_id = identifier(payload.get('space_id') or ('sp-' + uuid4().hex[:20]), 'space_id')
        name = self._name(payload.get('name'))
        def action(cur, _):
            cur.execute('SELECT 1 FROM spaces WHERE account_id=%s AND space_id=%s', (ctx.account_id, space_id))
            if cur.fetchone():
                raise SpaceError(409, 'SPACE_EXISTS', 'Space identifier already exists')
            self._available_name(cur, ctx.account_id, space_id, name)
            cur.execute('INSERT INTO spaces(account_id,space_id,name,owner_user_id) VALUES(%s,%s,%s,%s) RETURNING *', (ctx.account_id, space_id, name, ctx.user_id))
            row = dict(cur.fetchone())
            cur.execute('INSERT INTO space_members(account_id,space_id,principal,user_id,role) VALUES(%s,%s,%s,%s,%s)', (ctx.account_id, space_id, 'user:' + ctx.user_id, ctx.user_id, 'manager'))
            return {'space': row, 'space_id': space_id, 'authz_rev': 1}
        # The request scope remains stable when the server allocates an identifier.
        return self._mutate(ctx, space_id, 'space.create', payload, expected_rev, request_id, action, creating=True)

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise SpaceError(400, 'INVALID_ARGUMENT', 'name requires 1 to 128 characters')
        return value.strip()

    @staticmethod
    def _available_name(cur, account_id, space_id, name):
        cur.execute('SELECT 1 FROM spaces WHERE account_id=%s AND btrim(name)=%s AND space_id<>%s AND deleted_at IS NULL',
                    (account_id, name, space_id))
        if cur.fetchone():
            raise SpaceError(409, 'SPACE_NAME_EXISTS', 'A shared space with this name already exists')

    def list_principals(self, ctx, limit=50, cursor=None, kind=None, q=None, space_id=None):
        limit = self._limit(limit)
        kind, q = kind or None, (q or '').strip()
        if kind not in (None, 'user', 'agent') or len(q) > 128:
            raise SpaceError(400, 'INVALID_ARGUMENT', 'Select a member type and a search term of at most 128 characters')
        with self.transaction(ctx.account_id, write=False) as cur:
            if space_id:
                is_admin = self._space(cur, ctx, space_id, manage=True)['is_admin']
            else:
                user = self._identity(cur, ctx)
                is_admin = self._admin(ctx) and user['role'] == 'admin'
            cursor, binding = self._paging(cur, ctx, ['principals', kind, q, space_id], cursor)
            params = {'account': ctx.account_id, 'space': space_id, 'query': q, 'limit': limit + 1}
            branches = []
            for subject_type, table, column in [('agent', 'cp_agents', 'agent_id'), ('user', 'cp_users', 'user_id')]:
                prefix = subject_type + ':'
                if (kind and kind != subject_type) or (cursor and cursor.split(':', 1)[0] > subject_type):
                    continue
                params['after_' + subject_type] = cursor[len(prefix):] if cursor.startswith(prefix) else ''
                owner = "p.owner_user_id" if subject_type == 'agent' else 'NULL::text'
                join = "JOIN cp_users owner ON owner.account_id=p.account_id AND owner.user_id=p.owner_user_id AND owner.status='active'" if subject_type == 'agent' else ''
                branches.append(f"""(SELECT '{prefix}'||p.{column} AS principal,'{subject_type}' AS type,
                    p.{column} AS id,p.{column} AS display_name,p.status,{owner} AS owner_user_id
                    FROM {table} p {join}
                    WHERE p.account_id=%(account)s AND p.status='active' AND p.{column}>%(after_{subject_type})s
                    AND (%(query)s='' OR position(lower(%(query)s) in lower(p.{column}))>0)
                    AND (%(space)s IS NULL OR NOT EXISTS(SELECT 1 FROM space_members m
                        WHERE m.account_id=p.account_id AND m.space_id=%(space)s AND m.principal='{prefix}'||p.{column}))
                    ORDER BY p.{column} LIMIT %(limit)s)""")
            if not branches:
                return self._page([], limit, 'principal', binding)
            cur.execute('SELECT * FROM (' + ' UNION ALL '.join(branches) + ') candidates ORDER BY principal LIMIT %(limit)s', params)
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                row['can_contribute'] = is_admin or (row['id'] == ctx.user_id if row['type'] == 'user' else row['owner_user_id'] == ctx.user_id)
            return self._page(rows, limit, 'principal', binding)

    def list_spaces(self, ctx, space_id=None, limit=50, cursor=None):
        with self.transaction(ctx.account_id, write=False) as cur:
            if space_id:
                row = dict(self._space(cur, ctx, space_id))
                cur.execute('SELECT m.principal,m.role,a.owner_user_id FROM space_members m LEFT JOIN cp_agents a ON a.account_id=m.account_id AND a.agent_id=m.agent_id WHERE m.account_id=%s AND m.space_id=%s ORDER BY principal', (ctx.account_id, space_id))
                row['members'] = [dict(m) for m in cur.fetchall()]
                for member in row['members']:
                    kind, identity = principal(member['principal'])
                    if kind == 'agent':
                        owned = member['owner_user_id'] == ctx.user_id
                    else:
                        owned = identity == ctx.user_id
                    member['can_contribute'] = row['is_admin'] or owned
                cur.execute('SELECT binding_id,principal,memory_types,state,grant_rev,authorized_by FROM space_source_bindings WHERE account_id=%s AND space_id=%s AND deleted_at IS NULL ORDER BY binding_id', (ctx.account_id, space_id))
                row['sources'] = [dict(source) for source in cur.fetchall()]
                eligible = {m['principal'] for m in row['members'] if m['can_contribute']}
                for source in row['sources']:
                    source['can_manage'] = source['principal'] in eligible
                    source['can_delete'] = row['capabilities']['manage_space'] or source['can_manage']
                row['viewer_user_id'] = ctx.user_id
                return {'space': json_value(row)}
            user = self._identity(cur, ctx)
            limit = self._limit(limit)
            cursor, binding = self._paging(cur, ctx, ['spaces'], cursor)
            cur.execute("SELECT s.*,m.role AS member_role FROM spaces s LEFT JOIN space_members m ON m.account_id=s.account_id AND m.space_id=s.space_id AND m.principal=%s WHERE s.account_id=%s AND s.space_id>%s AND s.deleted_at IS NULL AND (%s OR s.owner_user_id=%s OR m.principal IS NOT NULL) ORDER BY s.space_id LIMIT %s", ('user:' + ctx.user_id, ctx.account_id, cursor or '', self._admin(ctx) and user['role'] == 'admin', ctx.user_id, limit + 1))
            rows = list(cur.fetchall())
            for row in rows:
                row['is_admin'] = self._admin(ctx) and user['role'] == 'admin'
                governance = row['owner_user_id'] == ctx.user_id or row['member_role'] == 'manager' or row['is_admin']
                row['capabilities'] = {'manage_space': governance, 'delete_space': row['owner_user_id'] == ctx.user_id or row['is_admin'], 'manage_members': governance, 'manage_acl': governance, 'manage_rules': governance, 'create_rule': governance and row['state'] == 'ACTIVE', 'inspect': True, 'search': row['state'] == 'ACTIVE'}
            return self._page(rows, limit, 'space_id', binding)

    def update_space(self, ctx, space_id, payload, *, expected_rev, request_id=None):
        def action(cur, space):
            self._cas(space['authz_rev'], expected_rev)
            name = self._name(payload.get('name', space['name']))
            self._available_name(cur, ctx.account_id, space_id, name)
            state = payload.get('state', space['state'])
            if state not in ('ACTIVE', 'ARCHIVED'):
                raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid space state')
            owner = payload.get('owner_user_id', space['owner_user_id'])
            if (owner != space['owner_user_id'] or state != space['state']) and not space['capabilities']['delete_space']:
                raise SpaceError(403, 'SPACE_OWNER_REQUIRED', 'Space owner or tenant administrator required')
            self._registered(cur, ctx.account_id, 'user:' + owner)
            cur.execute('SELECT 1 FROM space_members WHERE account_id=%s AND space_id=%s AND principal=%s', (ctx.account_id, space_id, 'user:' + owner))
            if not cur.fetchone():
                raise SpaceError(400, 'MEMBER_REQUIRED', 'Space owner must be a member')
            cur.execute('UPDATE spaces SET name=%s,state=%s,owner_user_id=%s,authz_rev=authz_rev+1,updated_at=NOW() WHERE account_id=%s AND space_id=%s RETURNING *', (name, state, owner, ctx.account_id, space_id))
            result = dict(cur.fetchone())
            self._dirty(cur, ctx.account_id, space_id)
            return {'space': result, 'authz_rev': result['authz_rev']}
        return self._mutate(ctx, space_id, f'space.update:{space_id}', payload, expected_rev, request_id, action, manage=True)

    def delete_space(self, ctx, space_id, *, expected_rev, request_id=None):
        def action(cur, space):
            if space['deleted_at'] is not None:
                raise SpaceError(404, 'SPACE_NOT_FOUND', 'Space not found')
            self._cas(space['authz_rev'], expected_rev)
            cur.execute("""UPDATE spaces SET state='ARCHIVED',deleted_at=NOW(),authz_rev=authz_rev+1,
                updated_at=NOW() WHERE account_id=%s AND space_id=%s RETURNING authz_rev""",
                (ctx.account_id, space_id))
            revision = cur.fetchone()['authz_rev']
            self._cancel_jobs(cur, ctx.account_id, space_id)
            self._dirty(cur, ctx.account_id, space_id)
            return {'deleted': True, 'space_id': space_id, 'authz_rev': revision}
        return self._mutate(ctx, space_id, f'space.delete:{space_id}', {}, expected_rev,
                            request_id, action, manage=True, deleting_space=True)

    def set_members(self, ctx, space_id, payload, *, expected_rev, request_id=None):
        members = payload.get('members')
        if not isinstance(members, list) or len(members) > 1000:
            raise SpaceError(400, 'INVALID_ARGUMENT', 'members must contain at most 1000 entries')
        def action(cur, space):
            self._cas(space['authz_rev'], expected_rev)
            wanted = {}
            for member in members:
                if not isinstance(member, dict):
                    raise SpaceError(400, 'INVALID_ARGUMENT', 'Each member must include a principal and role')
                subject = member.get('principal')
                kind, identity = principal(subject)
                role = member.get('role', 'member')
                if role not in ('manager', 'member') or (kind == 'agent' and role == 'manager'):
                    raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid member role')
                if subject in wanted:
                    raise SpaceError(400, 'INVALID_ARGUMENT', 'Each member must appear once')
                wanted[subject] = (identity if kind == 'user' else None, identity if kind == 'agent' else None, role)
            owner = 'user:' + space['owner_user_id']
            if owner not in wanted:
                raise SpaceError(400, 'OWNER_REQUIRED', 'Include the space owner')
            cur.execute('SELECT principal,role FROM space_members WHERE account_id=%s AND space_id=%s', (ctx.account_id, space_id))
            current = {row['principal']: row['role'] for row in cur.fetchall()}
            users = [row[0] for row in wanted.values() if row[0] is not None]
            agents = [row[1] for row in wanted.values() if row[1] is not None]
            cur.execute("""SELECT 'user:'||user_id AS principal FROM cp_users
                WHERE account_id=%s AND user_id=ANY(%s) AND status='active'
                UNION ALL SELECT 'agent:'||a.agent_id FROM cp_agents a
                JOIN cp_users u ON u.account_id=a.account_id AND u.user_id=a.owner_user_id AND u.status='active'
                WHERE a.account_id=%s AND a.agent_id=ANY(%s) AND a.status='active'""",
                (ctx.account_id, users, ctx.account_id, agents))
            registered = {row['principal'] for row in cur.fetchall()}
            if owner in current and wanted[owner][2] == current[owner]:
                registered.add(owner)
            if registered != set(wanted):
                raise SpaceError(400, 'REGISTERED_PRINCIPAL_REQUIRED', 'Select active registered members from this tenant')
            removed = list(current.keys() - wanted.keys())
            if removed:
                cur.execute("UPDATE space_source_bindings SET state='PAUSED',grant_rev=grant_rev+1,updated_at=NOW() WHERE account_id=%s AND space_id=%s AND state='ACTIVE' AND principal=ANY(%s)", (ctx.account_id, space_id, removed))
                cur.execute('DELETE FROM space_members WHERE account_id=%s AND space_id=%s AND principal=ANY(%s)', (ctx.account_id, space_id, removed))
            changed = [(ctx.account_id, space_id, subject, row[2]) for subject, row in wanted.items()
                       if subject in current and current[subject] != row[2]]
            if changed:
                execute_values(cur, """UPDATE space_members m SET role=v.role,updated_at=NOW()
                    FROM (VALUES %s) AS v(account_id,space_id,principal,role)
                    WHERE m.account_id=v.account_id AND m.space_id=v.space_id AND m.principal=v.principal""", changed, page_size=1000)
            added = [(ctx.account_id, space_id, subject, *row) for subject, row in wanted.items() if subject not in current]
            if added:
                execute_values(cur, 'INSERT INTO space_members(account_id,space_id,principal,user_id,agent_id,role) VALUES %s', added, page_size=1000)
            revision = self._bump(cur, ctx.account_id, space_id)
            self._dirty(cur, ctx.account_id, space_id)
            return {'space_id': space_id, 'authz_rev': revision, 'members': members}
        return self._mutate(ctx, space_id, f'space.members:{space_id}', payload, expected_rev, request_id, action, manage=True)

    def put_source(self, ctx, space_id, binding_id, payload, *, expected_rev, request_id=None):
        identifier(binding_id, 'binding_id')
        def action(cur, space):
            subject = payload.get('principal')
            kind, _, registered = self._registered(cur, ctx.account_id, subject)
            if not (space['is_admin'] or (subject == 'user:' + ctx.user_id) or (kind == 'agent' and registered['owner_user_id'] == ctx.user_id)):
                raise SpaceError(403, 'SOURCE_OWNER_REQUIRED', 'Source contribution requires its owner')
            cur.execute('SELECT 1 FROM space_members WHERE account_id=%s AND space_id=%s AND principal=%s', (ctx.account_id, space_id, subject))
            if not cur.fetchone():
                raise SpaceError(400, 'MEMBER_REQUIRED', 'Source must be a current member')
            cur.execute('SELECT * FROM space_source_bindings WHERE account_id=%s AND space_id=%s AND binding_id=%s', (ctx.account_id, space_id, binding_id))
            old = cur.fetchone()
            if old and old['deleted_at'] is not None:
                raise SpaceError(404, 'SOURCE_NOT_FOUND', 'Source binding not found')
            self._cas(old['grant_rev'] if old else 0, expected_rev)
            if old and old['principal'] != subject:
                raise SpaceError(400, 'IMMUTABLE_SOURCE', 'Binding principal is fixed')
            types = payload.get('memory_types', [])
            if not isinstance(types, list) or not 1 <= len(types) <= 32:
                raise SpaceError(400, 'INVALID_ARGUMENT', 'Select 1 to 32 memory types')
            for memory_type in types:
                identifier(memory_type, 'memory type')
                if kind == 'user' and memory_type in ('experience', 'case'):
                    raise SpaceError(400, 'SOURCE_SCOPE_MISMATCH', 'Experience and case sources belong to agents')
            state = payload.get('state', 'ACTIVE')
            if state not in ('ACTIVE', 'PAUSED'):
                raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid source state')
            if old:
                cur.execute('UPDATE space_source_bindings SET memory_types=%s,state=%s,authorized_by=%s,grant_rev=grant_rev+1,updated_at=NOW() WHERE account_id=%s AND space_id=%s AND binding_id=%s RETURNING *', (Json(sorted(set(types))), state, ctx.user_id, ctx.account_id, space_id, binding_id))
            else:
                cur.execute('SELECT 1 FROM space_source_bindings WHERE account_id=%s AND space_id=%s AND principal=%s AND deleted_at IS NULL',
                            (ctx.account_id, space_id, subject))
                if cur.fetchone():
                    raise SpaceError(409, 'SOURCE_EXISTS', 'A source binding for this member already exists')
                cur.execute('INSERT INTO space_source_bindings(account_id,space_id,binding_id,principal,memory_types,authorized_by,state) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *', (ctx.account_id, space_id, binding_id, subject, Json(sorted(set(types))), ctx.user_id, state))
            result = dict(cur.fetchone())
            result['authz_rev'] = self._bump(cur, ctx.account_id, space_id)
            self._dirty(cur, ctx.account_id, space_id)
            return result
        return self._mutate(ctx, space_id, f'space.source:{space_id}:{binding_id}', payload, expected_rev, request_id, action)

    def delete_source(self, ctx, space_id, binding_id, *, expected_rev, request_id=None):
        identifier(binding_id, 'binding_id')
        def action(cur, space):
            cur.execute("""SELECT b.*,a.owner_user_id FROM space_source_bindings b
                LEFT JOIN cp_agents a ON a.account_id=b.account_id AND b.principal='agent:'||a.agent_id
                WHERE b.account_id=%s AND b.space_id=%s AND b.binding_id=%s AND b.deleted_at IS NULL""",
                (ctx.account_id, space_id, binding_id))
            source = cur.fetchone()
            if not source:
                raise SpaceError(404, 'SOURCE_NOT_FOUND', 'Source binding not found')
            if not (space['capabilities']['manage_space'] or source['principal'] == 'user:' + ctx.user_id or source['owner_user_id'] == ctx.user_id):
                raise SpaceError(403, 'SOURCE_OWNER_REQUIRED', 'Source owner or space manager required')
            self._cas(source['grant_rev'], expected_rev)
            cur.execute("""UPDATE space_source_bindings SET state='PAUSED',deleted_at=NOW(),
                grant_rev=grant_rev+1,updated_at=NOW() WHERE account_id=%s AND space_id=%s AND binding_id=%s
                RETURNING grant_rev""", (ctx.account_id, space_id, binding_id))
            revision = cur.fetchone()['grant_rev']
            self._cancel_jobs(cur, ctx.account_id, space_id)
            self._dirty(cur, ctx.account_id, space_id)
            return {'deleted': True, 'binding_id': binding_id, 'grant_rev': revision,
                    'authz_rev': self._bump(cur, ctx.account_id, space_id)}
        return self._mutate(ctx, space_id, f'space.source.delete:{space_id}:{binding_id}', {},
                            expected_rev, request_id, action)

    def save_rule(self, ctx, space_id, rule_id, payload, *, expected_rev, request_id=None):
        identifier(rule_id, 'rule_id')
        name = self._name(payload.get('name'))
        state = payload.get('state', 'ACTIVE')
        if state not in ('ACTIVE', 'PAUSED', 'ARCHIVED'):
            raise SpaceError(400, 'INVALID_ARGUMENT', 'Invalid rule state')
        spec = validate_rule(payload.get('spec'))
        def action(cur, space):
            if space['state'] != 'ACTIVE':
                raise SpaceError(409, 'SPACE_ARCHIVED', 'Space is archived')
            cur.execute('SELECT r.*,v.spec FROM space_rules r JOIN space_rule_revisions v USING(account_id,space_id,rule_id,rule_rev) WHERE r.account_id=%s AND r.space_id=%s AND r.rule_id=%s', (ctx.account_id, space_id, rule_id))
            old = cur.fetchone()
            if old and old['deleted_at'] is not None:
                raise SpaceError(404, 'RULE_NOT_FOUND', 'Rule not found')
            self._cas(old['rule_rev'] if old else 0, expected_rev)
            old_spec = old['spec'] if old else {}
            if spec['output_acl'] != old_spec.get('output_acl', {'mode': 'inherit', 'readers': []}) and not space['capabilities']['manage_acl']:
                raise SpaceError(403, 'ACL_MANAGEMENT_REQUIRED', 'Read policy changes require space management permission')
            subjects = set(spec.get('source_subjects', [])) | set(spec['output_acl']['readers'])
            previous = set(old_spec.get('source_subjects', [])) | set(old_spec.get('output_acl', {}).get('readers', []))
            for subject in subjects - previous:
                self._registered(cur, ctx.account_id, subject)
                cur.execute('SELECT 1 FROM space_members WHERE account_id=%s AND space_id=%s AND principal=%s', (ctx.account_id, space_id, subject))
                if not cur.fetchone():
                    raise SpaceError(400, 'MEMBER_REQUIRED', 'New rule selections must be current members')
            revision = old['rule_rev'] + 1 if old else 1
            if old:
                cur.execute('UPDATE space_rules SET name=%s,rule_rev=%s,state=%s,updated_at=NOW() WHERE account_id=%s AND space_id=%s AND rule_id=%s', (name, revision, state, ctx.account_id, space_id, rule_id))
            else:
                cur.execute('INSERT INTO space_rules(account_id,space_id,rule_id,name,rule_rev,state) VALUES(%s,%s,%s,%s,%s,%s)', (ctx.account_id, space_id, rule_id, name, revision, state))
            spec_hash = digest(spec)
            cur.execute('INSERT INTO space_rule_revisions(account_id,space_id,rule_id,rule_rev,name,state,spec,spec_hash) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)', (ctx.account_id, space_id, rule_id, revision, name, state, Json(spec), spec_hash))
            if old:
                cur.execute('UPDATE space_rule_runtime SET dirty_generation=dirty_generation+1,next_run_at=NOW(),updated_at=NOW() WHERE account_id=%s AND space_id=%s AND rule_id=%s', (ctx.account_id, space_id, rule_id))
            else:
                cur.execute('INSERT INTO space_rule_runtime(account_id,space_id,rule_id,dirty_generation) VALUES(%s,%s,%s,1)', (ctx.account_id, space_id, rule_id))
            return {'rule_id': rule_id, 'rule_rev': revision, 'spec_hash': spec_hash, 'state': state}
        return self._mutate(ctx, space_id, f'space.rule:{space_id}:{rule_id}', payload, expected_rev, request_id, action, manage=True)

    def delete_rule(self, ctx, space_id, rule_id, *, expected_rev, request_id=None):
        identifier(rule_id, 'rule_id')
        def action(cur, space):
            cur.execute('SELECT rule_rev FROM space_rules WHERE account_id=%s AND space_id=%s AND rule_id=%s AND deleted_at IS NULL', (ctx.account_id, space_id, rule_id))
            rule = cur.fetchone()
            if not rule:
                raise SpaceError(404, 'RULE_NOT_FOUND', 'Rule not found')
            self._cas(rule['rule_rev'], expected_rev)
            revision = rule['rule_rev'] + 1
            cur.execute("""INSERT INTO space_rule_revisions(account_id,space_id,rule_id,rule_rev,name,state,spec,spec_hash)
                SELECT account_id,space_id,rule_id,%s,name,'ARCHIVED',spec,spec_hash FROM space_rule_revisions
                WHERE account_id=%s AND space_id=%s AND rule_id=%s AND rule_rev=%s""",
                (revision, ctx.account_id, space_id, rule_id, rule['rule_rev']))
            cur.execute("""UPDATE space_rules SET rule_rev=%s,state='ARCHIVED',deleted_at=NOW(),updated_at=NOW()
                WHERE account_id=%s AND space_id=%s AND rule_id=%s""", (revision, ctx.account_id, space_id, rule_id))
            self._cancel_jobs(cur, ctx.account_id, space_id, rule_id)
            return {'deleted': True, 'rule_id': rule_id, 'rule_rev': revision}
        return self._mutate(ctx, space_id, f'space.rule.delete:{space_id}:{rule_id}', {},
                            expected_rev, request_id, action, manage=True)

    def list_rules(self, ctx, space_id, rule_id=None, revision=None, limit=50, cursor=None):
        limit = self._limit(limit)
        with self.transaction(ctx.account_id, write=False) as cur:
            self._space(cur, ctx, space_id)
            cursor, binding = self._paging(cur, ctx, ['rules', space_id, rule_id, revision], cursor)
            cur.execute('SELECT r.rule_id,v.name,v.state,v.rule_rev,v.spec,v.spec_hash FROM space_rules r JOIN space_rule_revisions v ON v.account_id=r.account_id AND v.space_id=r.space_id AND v.rule_id=r.rule_id AND v.rule_rev=COALESCE(%s,r.rule_rev) WHERE r.account_id=%s AND r.space_id=%s AND r.deleted_at IS NULL AND (%s IS NULL OR r.rule_id=%s) AND r.rule_id>%s ORDER BY r.rule_id LIMIT %s', (revision, ctx.account_id, space_id, rule_id, rule_id, cursor or '', limit + 1))
            return self._page(cur.fetchall(), limit, 'rule_id', binding)

    def set_acl(self, ctx, space_id, payload, *, expected_rev, request_id=None):
        acl = validate_acl(payload.get('acl', payload))
        item_id = identifier(payload.get('item_id'), 'item_id')
        def action(cur, space):
            self._cas(space['authz_rev'], expected_rev)
            cur.execute('SELECT state,superseded_by FROM space_items WHERE account_id=%s AND space_id=%s AND item_id=%s', (ctx.account_id, space_id, item_id))
            item = cur.fetchone()
            if item is None:
                raise SpaceError(404, 'ITEM_NOT_FOUND', 'Shared item not found')
            if item['state'] not in ('ACTIVE', 'RECOMPUTING') or item['superseded_by']:
                raise SpaceError(412, 'ITEM_CHANGED', 'Team memory changed; reload the current version before editing permissions')
            if acl['readers']:
                cur.execute("""SELECT m.principal FROM space_members m
                    LEFT JOIN cp_users u ON u.account_id=m.account_id AND u.user_id=m.user_id AND u.status='active'
                    LEFT JOIN cp_agents a ON a.account_id=m.account_id AND a.agent_id=m.agent_id AND a.status='active'
                    LEFT JOIN cp_users owner ON owner.account_id=a.account_id AND owner.user_id=a.owner_user_id AND owner.status='active'
                    WHERE m.account_id=%s AND m.space_id=%s AND m.principal=ANY(%s)
                    AND (u.user_id IS NOT NULL OR (a.agent_id IS NOT NULL AND owner.user_id IS NOT NULL))""",
                    (ctx.account_id, space_id, acl['readers']))
                if {row['principal'] for row in cur.fetchall()} != set(acl['readers']):
                    raise SpaceError(400, 'MEMBER_REQUIRED', 'ACL readers must be active registered team members')
            cur.execute("UPDATE space_items SET acl_mode=%s,acl_origin='manual',updated_at=NOW() WHERE account_id=%s AND space_id=%s AND item_id=%s", (acl['mode'], ctx.account_id, space_id, item_id))
            cur.execute('DELETE FROM space_read_acl WHERE account_id=%s AND space_id=%s AND item_id=%s', (ctx.account_id, space_id, item_id))
            if acl['readers']:
                execute_values(cur, 'INSERT INTO space_read_acl(account_id,space_id,item_id,principal) VALUES %s',
                               [(ctx.account_id, space_id, item_id, subject) for subject in acl['readers']], page_size=1000)
            return {'item_id': item_id, 'authz_rev': self._bump(cur, ctx.account_id, space_id), 'acl': acl}
        return self._mutate(ctx, space_id, f'space.acl:{space_id}:{item_id}', payload, expected_rev, request_id, action, manage=True)

    def permissions(self, ctx, space_id, item_id=None):
        with self.transaction(ctx.account_id, write=False) as cur:
            space = self._space(cur, ctx, space_id)
            result = {'space_id': space_id, 'authz_rev': space['authz_rev'], 'capabilities': space['capabilities'], 'principal': 'user:' + ctx.user_id, 'user_id': ctx.user_id, 'viewer_user_id': ctx.user_id, 'is_admin': space['is_admin']}
        if item_id:
            result['can_read'] = any(i['item_id'] == item_id for i in self.visible_items(ctx, [space_id], item_ids=[item_id], hydrate=False))
        return result

    def list_jobs(self, ctx, space_id, rule_id=None, generation=None, limit=50, cursor=None, job_id=None, state=None):
        limit = self._limit(limit)
        with self.transaction(ctx.account_id, write=False) as cur:
            self._space(cur, ctx, space_id, manage=True)
            cursor, binding = self._paging(cur, ctx, ['jobs', space_id, rule_id, generation, job_id, state], cursor)
            cur.execute('SELECT job_id,rule_id,rule_rev,operation,generation,state,attempts,result,last_error,created_at,updated_at FROM space_jobs WHERE account_id=%s AND space_id=%s AND (%s IS NULL OR rule_id=%s) AND (%s IS NULL OR generation>=%s) AND (%s IS NULL OR job_id=%s) AND (%s IS NULL OR state=%s) AND job_id>%s ORDER BY job_id LIMIT %s', (ctx.account_id, space_id, rule_id, rule_id, generation, generation, job_id, job_id, state, state, cursor or '', limit + 1))
            return self._page(cur.fetchall(), limit, 'job_id', binding)

    def list_audit(self, ctx, space_id, limit=50, cursor=None, action=None, rule_id=None, item_id=None, from_time=None, until=None):
        limit = self._limit(limit)
        with self.transaction(ctx.account_id, write=False) as cur:
            self._space(cur, ctx, space_id, manage=True)
            cursor, binding = self._paging(cur, ctx, ['audit', space_id, action, rule_id, item_id, from_time, until], cursor)
            cur.execute("SELECT * FROM cp_audit_logs WHERE account_id=%s AND target=%s AND log_id>%s AND (%s IS NULL OR action=%s) AND (%s IS NULL OR details->>'rule_id'=%s) AND (%s IS NULL OR details->>'item_id'=%s) AND (%s IS NULL OR timestamp>=%s::timestamptz) AND (%s IS NULL OR timestamp<=%s::timestamptz) ORDER BY log_id LIMIT %s", (ctx.account_id, 'space:' + space_id, cursor or '', action, action, rule_id, rule_id, item_id, item_id, from_time, from_time, until, until, limit + 1))
            return self._page(cur.fetchall(), limit, 'log_id', binding)

    def visible_items(self, ctx, space_ids=None, uris=None, limit=100, *, item_ids=None, query=None, kind=None, after='', hydrate=True):
        """Return authorized matches in item-ID order for cursor pagination."""
        while True:
            with self.transaction(ctx.account_id, write=False) as cur:
                rows = self._visible_rows(cur, ctx, space_ids, uris, limit, item_ids=item_ids,
                                          query=query, kind=kind, after=after, hydrate=hydrate)
            visible = self._hydrate_items(ctx, rows) if hydrate else rows
            if visible or not rows:
                return visible
            after = rows[-1]['item_id']

    def _visible_rows(self, cur, ctx, space_ids=None, uris=None, limit=100, *, item_ids=None, query=None, kind=None, after='', inspect=False, hydrate=True):
        if uris == [] or space_ids == [] or item_ids == []:
            return []
        params = {'account': ctx.account_id, 'user': ctx.user_id,
                  'user_principal': 'user:' + ctx.user_id,
                  'agent': ctx.agent_id or '', 'agent_principal': 'agent:' + (ctx.agent_id or ''),
                  'agent_visible': ('agent:' + (ctx.agent_id or '')) in ctx.visible_owner_spaces,
                  'inspect': inspect, 'after': after, 'spaces': space_ids, 'uris': uris, 'items': item_ids, 'query': query or '', 'kind': kind,
                  'limit': max(1, min(int(limit), 2000))}
        projection = "i.item_id,i.space_id,i.uri,i.kind,n.version,v.spec->'privacy_policy' AS _privacy_policy,json_build_object('agfs_content',(n.metadata->'agfs_content')::json) AS metadata"
        if hydrate:
            projection = """i.*,n.content,n.abstract,n.overview,n.metadata,n.version,v.spec->'privacy_policy' AS _privacy_policy,
  (SELECT count(*) FROM space_item_sources counts WHERE counts.account_id=i.account_id AND counts.space_id=i.space_id AND counts.item_id=i.item_id) AS source_count"""
        item_source = 'space_items'
        uri_filter = 'AND (%(uris)s IS NULL OR i.uri=ANY(%(uris)s))'
        if self.dialect.name == 'opengauss' and uris is not None:
            # OFFSET 0 keeps the URI batch filter separate from per-item authorization joins.
            item_source = '(SELECT * FROM space_items WHERE account_id=%(account)s AND uri=ANY(%(uris)s) OFFSET 0)'
            uri_filter = ''
        sql = f"""
SELECT {projection}
FROM {item_source} i
JOIN spaces s USING(account_id,space_id)
JOIN space_rules rules ON rules.account_id=i.account_id AND rules.space_id=i.space_id AND rules.rule_id=i.rule_id
JOIN space_rule_revisions v ON v.account_id=rules.account_id AND v.space_id=rules.space_id AND v.rule_id=rules.rule_id AND v.rule_rev=rules.rule_rev
JOIN context_nodes n ON n.account_id=i.account_id AND n.uri=i.uri
JOIN cp_accounts a ON a.account_id=i.account_id AND a.status='active'
JOIN cp_users u ON u.account_id=i.account_id AND u.user_id=%(user)s AND u.status='active'
LEFT JOIN space_members m ON m.account_id=i.account_id AND m.space_id=i.space_id AND m.principal=%(user_principal)s
WHERE i.account_id=%(account)s AND i.item_id>%(after)s AND n.owner_space LIKE 'space:%%'
  AND s.deleted_at IS NULL AND rules.deleted_at IS NULL
  AND ((%(inspect)s AND (u.role='admin' OR s.owner_user_id=%(user)s OR m.role='manager')
        AND i.state<>'PURGED') OR
       (s.state='ACTIVE' AND i.state='ACTIVE' AND n.status='ACTIVE' AND i.expires_at>NOW()
        AND n.metadata->>'shared_projection_ready'='true'
        AND n.metadata->>'indexed_content_rev'=n.version::text))
  AND (%(query)s='' OR to_tsvector('simple',n.abstract||' '||n.content) @@ plainto_tsquery('simple',%(query)s)
       OR strpos(lower(n.abstract||' '||n.content),lower(%(query)s))>0)
  AND (%(kind)s IS NULL OR i.kind=%(kind)s)
  AND (%(spaces)s IS NULL OR i.space_id=ANY(%(spaces)s))
  {uri_filter}
  AND (%(items)s IS NULL OR i.item_id=ANY(%(items)s))
  AND (u.role='admin' OR s.owner_user_id=%(user)s OR m.role='manager'
       OR (m.principal IS NOT NULL AND (i.acl_mode='inherit' OR EXISTS(
           SELECT 1 FROM space_read_acl r WHERE r.account_id=i.account_id AND r.space_id=i.space_id
           AND r.item_id=i.item_id AND r.principal=%(user_principal)s))))
  AND (%(agent)s='' OR EXISTS(
      SELECT 1 FROM cp_agents ag JOIN cp_users agent_owner ON agent_owner.account_id=ag.account_id
      AND agent_owner.user_id=ag.owner_user_id AND agent_owner.status='active'
      JOIN space_members am ON am.account_id=ag.account_id
      AND am.agent_id=ag.agent_id AND am.space_id=i.space_id
      WHERE ag.account_id=i.account_id AND ag.agent_id=%(agent)s AND ag.status='active'
      AND (ag.owner_user_id=%(user)s OR u.role='admin' OR %(agent_visible)s)
      AND (i.acl_mode='inherit' OR EXISTS(SELECT 1 FROM space_read_acl ar
          WHERE ar.account_id=i.account_id AND ar.space_id=i.space_id
          AND ar.item_id=i.item_id AND ar.principal=%(agent_principal)s))))
  AND EXISTS(SELECT 1 FROM space_item_sources x WHERE x.account_id=i.account_id AND x.space_id=i.space_id AND x.item_id=i.item_id)
  AND NOT EXISTS (
      SELECT 1 FROM space_item_sources x
      LEFT JOIN space_source_bindings b ON b.account_id=x.account_id AND b.space_id=x.space_id AND b.binding_id=x.binding_id
      LEFT JOIN space_members sm ON sm.account_id=b.account_id AND sm.space_id=b.space_id AND sm.principal=b.principal
      LEFT JOIN cp_users su ON su.account_id=sm.account_id AND su.user_id=sm.user_id AND su.status='active'
      LEFT JOIN cp_agents sa ON sa.account_id=sm.account_id AND sa.agent_id=sm.agent_id AND sa.status='active'
      LEFT JOIN cp_users agent_owner ON agent_owner.account_id=sa.account_id AND agent_owner.user_id=sa.owner_user_id AND agent_owner.status='active'
      LEFT JOIN cp_users grantor ON grantor.account_id=b.account_id AND grantor.user_id=b.authorized_by AND grantor.status='active'
      LEFT JOIN context_nodes source ON source.account_id=x.account_id AND source.uri=x.source_uri
      WHERE x.account_id=i.account_id AND x.space_id=i.space_id AND x.item_id=i.item_id
      AND (b.binding_id IS NULL OR b.deleted_at IS NOT NULL OR b.state<>'ACTIVE' OR b.grant_rev<>x.grant_rev
           OR sm.principal IS NULL OR (su.user_id IS NULL AND sa.agent_id IS NULL)
           OR grantor.user_id IS NULL OR (sa.agent_id IS NOT NULL AND agent_owner.user_id IS NULL)
           OR (grantor.role='admin' OR b.principal='user:'||grantor.user_id OR sa.owner_user_id=grantor.user_id) IS NOT TRUE
           OR source.uri IS NULL OR source.status<>'ACTIVE' OR source.version<>x.source_content_rev
           OR COALESCE(source.metadata->>'shared_source_epoch','')<>x.source_epoch))
ORDER BY i.item_id LIMIT %(limit)s
"""
        if self.dialect.name == 'opengauss':
            cur.execute('SET LOCAL join_collapse_limit=1; SET LOCAL from_collapse_limit=1')
        cur.execute(sql, params)
        return json_value(cur.fetchall())

    def _hydrate_items(self, ctx, rows, *, cur=None, inspect=False):
        if not rows:
            return []
        current = rows
        if self.content_store:
            rows = self.content_store.load_content(rows)
            with nullcontext(cur) if cur is not None else self.transaction(ctx.account_id, write=False) as checked:
                current = self._visible_rows(checked, ctx, uris=[row['uri'] for row in rows],
                                             limit=len(rows), inspect=inspect, hydrate=False)
        by_uri = {row['uri']: row for row in current}
        result = []
        for row in rows:
            latest = by_uri.get(row['uri'])
            if latest is None or latest['version'] != row['version']:
                continue
            policy = latest['_privacy_policy']
            row.pop('_privacy_policy', None)
            item = redact_item(row, policy)
            item.pop('files', None)
            metadata = item.get('metadata', {})
            for key in ('package_files', 'package_skill_md', 'skill_name'):
                metadata.pop(key, None)
            result.append(item)
        return result

    def list_items(self, ctx, space_id, item_id=None, limit=50, cursor=None, state=None, file=None):
        if file is not None:
            raise SpaceError(400, 'TEXT_ONLY', 'Team memory supports text content')
        limit = self._limit(limit)
        ctx = replace(ctx, agent_id='')
        with self.transaction(ctx.account_id, write=False) as cur:
            space = self._space(cur, ctx, space_id)
            scope = ['items', space_id, item_id, state, file]
            cursor, _ = self._paging(cur, ctx, scope, cursor)
            inspect = space['capabilities']['manage_space'] and state != 'ACTIVE'
            if inspect:
                cur.execute("SELECT i.* FROM space_items i JOIN space_rules r USING(account_id,space_id,rule_id) WHERE i.account_id=%s AND i.space_id=%s AND i.kind='experience' AND r.deleted_at IS NULL AND (%s IS NULL OR i.item_id=%s) AND (%s IS NULL OR i.state=%s) AND i.item_id>%s ORDER BY i.item_id LIMIT %s", (ctx.account_id, space_id, item_id, item_id, state, state, cursor, limit + 1))
                rows = json_value(cur.fetchall())
                candidates = self._visible_rows(cur, ctx, [space_id], uris=[row['uri'] for row in rows], limit=limit + 1, inspect=True)
            else:
                rows = self._visible_rows(cur, ctx, [space_id], item_ids=[item_id] if item_id else None, limit=limit + 1, after=cursor, kind='experience') if state in (None, 'ACTIVE') else []
                candidates = rows
            scanned_after = rows[-1]['item_id'] if len(rows) > limit else None
            visible = {row['item_id']: row for row in self._hydrate_items(ctx, candidates, cur=cur, inspect=inspect)}
            if self.content_store and candidates:
                space = self._space(cur, ctx, space_id)
                inspect = space['capabilities']['manage_space'] and state != 'ACTIVE'
            rows = [row for row in rows if row['item_id'] in visible
                    or (inspect and 'content' not in row)]
            for row in rows:
                row.update(visible.get(row['item_id'], {}))
                row['sources'] = []
                row['readers'] = []
            by_id = {row['item_id']: row for row in rows}
            if rows:
                cur.execute('SELECT item_id,principal FROM space_read_acl WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)', (ctx.account_id, space_id, list(by_id)))
                for reader in cur.fetchall():
                    by_id[reader['item_id']]['readers'].append(reader['principal'])
            if space['capabilities']['manage_space'] and visible:
                cur.execute('SELECT item_id,binding_id,source_uri,grant_rev,source_content_rev FROM space_item_sources WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)', (ctx.account_id, space_id, list(visible)))
                for source in cur.fetchall():
                    by_id[source['item_id']]['sources'].append(dict(source))
            for row in rows:
                metadata = row.get('metadata', {})
                row['content_rev'] = row.get('version')
                row['indexed_content_rev'] = metadata.get('indexed_content_rev')
                row['index_state'] = ('READY' if metadata.get('shared_projection_ready')
                                      and row['content_rev'] == row['indexed_content_rev'] else 'PENDING')
            _, binding = self._paging(cur, ctx, scope, None)
            page = self._page(rows, limit, 'item_id', binding)
            if scanned_after and page['next_cursor'] is None:
                page['next_cursor'] = self.cursor_after(scanned_after, binding)
            return page
