"""Assemble recovery adapters from a drained instance's actual resources."""
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from service.migration.local_backup import LocalTreeBackup
from service.migration.package import PackageError
from service.migration.resources import agfs_business_roots, sql_business_tables, vector_backup
from service.migration.sql_backup import SQLBackup
from service.migration.tree_backup import AGFSTreeBackup


def _endpoint(value):
    url = urlsplit(value)
    return {'scheme': url.scheme, 'host': url.hostname, 'port': url.port, 'path': url.path}


def _database(value):
    from psycopg2.extensions import parse_dsn
    options = parse_dsn(value or '')
    return {key: options.get(key) for key in ('host', 'hostaddr', 'port', 'dbname', 'service')}


def configuration_binding(cfg, asset_root):
    fields = ('retrieval_mode', 'retrieval_agfs_fts_enabled', 'retrieval_agfs_fts_ranker', 'storage_backend', 'shared_spaces_enabled', 'sql_dialect', 'vector_db_type', 'vector_db_dialect',
              'opengauss_table_name', 'opengauss_dimension', 'chroma_collection_name',
              'milvus_collection_name', 'milvus_db_name', 'provider', 'embedding_provider',
              'st_model', 'openai_embedding_model', 'embedding_multimodal',
              'embedding_max_input_chars', 'graph_backend', 'graph_edge_embedding_profile',
              'graph_projection_representation', 'retrieval_hybrid_enabled', 'retrieval_lexical_provider')
    result = {key: getattr(cfg, key, None) for key in fields}
    result.update(sql=_database(cfg.sql_connection_string), vector_sql=_database(cfg.opengauss_connection_string),
                  agfs=_endpoint(cfg.agfs_base_url), agfs_prefix=cfg.agfs_mount_prefix,
                  milvus=_endpoint(cfg.milvus_uri),
                  chroma_directory=str(Path(cfg.chroma_persist_directory).resolve()),
                  attachments=str(Path(asset_root).resolve()) if asset_root else None)
    result['agfs_transport'] = cfg.agfs_transport
    if cfg.agfs_transport == 'native':
        import hashlib
        result['agfs_native'] = dict(
            library=str(Path(cfg.agfs_native_library).resolve()),
            config=str(Path(cfg.agfs_native_config).resolve()),
            config_sha256=hashlib.sha256(Path(cfg.agfs_native_config).read_bytes()).hexdigest())
    lexical = dict(getattr(cfg, 'retrieval_lexical_options', {}) or {})
    connection = lexical.pop('connection_string', None)
    # Provider options are schema-validated by the lexical factory; credentials
    # belong only to its connection field, which is reduced to a locator.
    result['lexical'] = {key: value for key, value in lexical.items() if key == 'table_name' or isinstance(value, (int, float, bool))}
    result['lexical_database'] = _database(connection)
    return result


def _sql_backup(connection_string, tables, lifetime):
    import psycopg2

    from fs.sql_adapter.resource_identity import sql_resource_identity

    conn = psycopg2.connect(connection_string, connect_timeout=10)
    lifetime.callback(conn.close)
    names = tables(conn) if callable(tables) else tables
    with conn.cursor() as cur:
        cur.execute('SELECT current_schema()')
        schema = cur.fetchone()[0]
    return SQLBackup(conn, schema, names, lambda: sql_resource_identity(conn, names))


def _agfs_identity(adapter, roots):
    from fs.agfs_adapter.agfs_context_fs import _is_not_found_error

    directory = adapter.prefix + '/.ogmemory'
    marker = directory + '/migration-instance'
    try:
        value = adapter.client.read(marker, size=128)
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
        adapter._mkdir(directory)
        value = str(uuid4()).encode('ascii')
        adapter.client.write(marker, value)
    try:
        UUID(value.decode('ascii'))
    except (ValueError, UnicodeError) as exc:
        raise PackageError('invalid AGFS instance identity') from exc

    def identity():
        actual = adapter.client.read(marker, size=128)
        if actual != value:
            raise PackageError('AGFS recovery instance has changed')
        return {'endpoint': _endpoint(adapter.cfg.agfs_base_url), 'instance': value.decode('ascii'),
                'roots': roots}
    return identity


def backup_components(cfg, incoming, adapter, indexes, lifetime, *, asset_root=None):
    """Call only after all processes have acknowledged the maintenance drain."""
    result = {}
    if cfg.storage_backend == 'sql':
        result['content'] = _sql_backup(cfg.sql_connection_string, sql_business_tables, lifetime)
    else:
        roots = agfs_business_roots(adapter, incoming)
        identity = _agfs_identity(adapter, roots)
        result['content'] = AGFSTreeBackup(adapter.client, roots, identity)
        if cfg.shared_spaces_enabled:
            result['metadata'] = _sql_backup(cfg.sql_connection_string, sql_business_tables, lifetime)
    if indexes.vector is not None:
        result['vectors'] = vector_backup(indexes.vector)
    if indexes.lexical is not None:
        from providers.lexical_index.sql_index import SQLNativeProjectionStore
        lexical = indexes.lexical
        if not isinstance(lexical, SQLNativeProjectionStore):
            raise PackageError('lexical backend does not expose a recoverable storage scope')
        tables = [getattr(lexical, f'_{name}_table') for name in (
            'metadata', 'docs', 'tombstones', 'rebuilds', 'rebuild_docs', 'scope_epochs', 'scope_tombstones')]
        result['lexical'] = _sql_backup(lexical._shared_pool.connection_string, tables, lifetime)
    if asset_root:
        result['attachments'] = LocalTreeBackup(asset_root)
    return result
