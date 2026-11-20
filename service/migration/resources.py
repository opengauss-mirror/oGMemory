"""Bind recovery to declared business namespaces, excluding control-plane data."""
from __future__ import annotations

from service.migration.package import PackageError

SQL_BUSINESS_TABLES = frozenset({
    'context_nodes', 'relation_edges', 'session_archives', 'outbox_events',
    'edge_projection_jobs', 'dream_recalls', 'extraction_work_accounts',
    'code_workspace_sync_roots', 'trace_events', 'trace_sessions', 'trace_session_events',
    'skill_candidates', 'skill_candidate_evidence', 'skill_example_evidence',
    'skill_work_accounts', 'tool_result_artifacts',
    'spaces', 'space_members', 'space_source_bindings', 'space_rules',
    'space_rule_revisions', 'space_rule_runtime', 'space_items',
    'space_item_sources', 'space_read_acl', 'space_jobs', 'space_requests',
})
AGFS_ACCOUNT_ROOTS = frozenset({'users', 'agents', 'sessions', 'resources',
                                '.outbox', '.outbox-done', '.trace', '.space-content'})


def _component(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or any(char in value for char in '/\\\x00%')):
        raise PackageError('invalid recovery namespace')
    return value


def agfs_business_roots(adapter, source):
    """Include prospective new accounts before the backup, never account _system."""
    declared = getattr(source, 'recovery_accounts', None)
    accounts = ({_component(account) for account in declared} if declared is not None else
                {_component(record['account_id']) for kind in ('nodes', 'archives')
                 for record in source.records(kind)})
    for entry in adapter._ls(adapter.prefix + '/accounts'):
        if not entry.get('isDir'):
            raise PackageError('unexpected file in AGFS account namespace')
        accounts.add(_component(entry['name']))
    roots = [f'{adapter.prefix}/accounts/{account}/{name}'
             for account in sorted(accounts) for name in sorted(AGFS_ACCOUNT_ROOTS)]
    # Trace work queues include an instance-wide account index outside accounts/.
    roots.append(adapter.prefix + '/.ogmemory/trace-store')
    return roots


def sql_business_tables(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT tablename FROM pg_tables WHERE schemaname=current_schema()')
        tables = {row[0] for row in cur}
    required = {'context_nodes', 'relation_edges', 'outbox_events', 'session_archives', 'edge_projection_jobs'}
    if not required.issubset(tables):
        raise PackageError('SQL business schema is incomplete')
    return sorted(tables & SQL_BUSINESS_TABLES)


def vector_backup(vector):
    from service.migration.vector_backup import VectorBackup

    identity = getattr(vector, 'migration_identity', None)
    if not callable(identity):
        raise PackageError('vector backend does not expose a recovery resource identity')
    identity()  # Resolve the actual resource before accepting a recovery scope.
    return VectorBackup(vector, identity)
