"""Choose the whole shared-space collection, never merge individual spaces."""
from service.migration.package import PackageError
from service.migration.space_catalog import SpaceCatalog


def select_spaces(manifest, spaces, cfg, adapter, existing, vector):
    count = spaces.count()
    nodes = spaces.db.execute('SELECT count(*) FROM nodes').fetchone()[0]
    report = {'spaces': count, 'nodes': nodes, 'action': 'none', 'reason': None}
    if not count:
        return report
    reason = None
    if not getattr(cfg, 'shared_spaces_enabled', False):
        reason = 'target_shared_spaces_disabled'
    elif manifest['source_backend'] != adapter.backend:
        reason = 'content_backend_differs'
    else:
        catalog = SpaceCatalog(adapter)
        try:
            target = catalog.description()
        except PackageError:
            reason = 'shared_schema_differs'
        else:
            source = manifest['shared_spaces']
            if source['database_type'] != target['database_type']:
                reason = 'database_type_differs'
            elif source['schema_sha256'] != target['schema_sha256']:
                reason = 'shared_schema_differs'
        if reason is None:
            if (existing.db.execute('SELECT 1 FROM records LIMIT 1').fetchone()
                    or next(catalog.records(), None) is not None or vector.count()):
                reason = 'target_business_not_empty'
            else:
                reason = spaces.target_identity_error(adapter.conn)
    return {**report, 'action': 'skipped' if reason else 'import', 'reason': reason}


def confirm_selection(current, confirmed):
    if current['spaces'] and current != confirmed:
        raise PackageError('shared-space migration selection changed; run preflight and confirm again')
