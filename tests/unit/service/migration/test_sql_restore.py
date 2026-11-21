"""SQL-specific migration constraints and relation mirror write cost."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from tests.migration_support import write_described_package as write_package

from service.migration.package import PackageError
from service.migration.storage import SQLStorage
from tests.unit.service.migration.test_engine import Files, archive, node, service


def sql_target():
    target = service(Files())
    target.cfg.storage_backend = 'sql'
    adapter = SQLStorage.__new__(SQLStorage)
    adapter.conn = MagicMock()
    adapter.check = lambda **kwargs: None
    adapter.records = lambda kind: iter(())
    adapter.description = lambda: {
        'backend': 'sql', 'database_type': 'postgresql', 'database_version': '16',
    }

    @contextmanager
    def factory(cfg):
        yield adapter

    target.adapter_factory = factory
    return target, adapter


def package(tmp_path, field, value, backend="sql"):
    r, a = node(), archive()
    edge = dict(from_uri=r['uri'], to_uri=r['uri'], relation_type='related_to',
                reason='reason', weight=1.0)
    if field == 'text':
        r['content'] = value
    elif field == 'metadata_value':
        r['metadata']['nested'] = [{'value': value}]
    elif field == 'metadata_key':
        r['metadata']['nested'] = [{value: 'value'}]
    elif field == 'message':
        a['messages'] = [{'content': value}]
    elif field == 'archive_metadata':
        a['metadata']['nested'] = [{'value': value}]
    else:
        edge['reason'] = value
    path = tmp_path / 'source.tar.gz'
    write_package(path, {'nodes': [r], 'archives': [a], 'relations': [edge]},
                  source_backend=backend, source_version='test')
    return path


@pytest.mark.parametrize('field', ['text', 'metadata_value', 'metadata_key', 'message',
                                   'archive_metadata', 'reason'])
def test_sql_nul_rejected_before_any_write(tmp_path, field):
    path = package(tmp_path, field, 'a\x00b')
    target, adapter = sql_target()
    for operation in (target.preview, target.import_package):
        with pytest.raises(PackageError, match='SQL target cannot store NUL'):
            operation(path)
        adapter.conn.cursor.assert_not_called()
        adapter.conn.commit.assert_not_called()


@pytest.mark.parametrize('field', ['text', 'metadata_value', 'metadata_key', 'message',
                                   'archive_metadata', 'reason'])
def test_sql_accepts_literal_escape(tmp_path, field):
    path = package(tmp_path, field, r'a\u0000b')
    target, _ = sql_target()
    target.preview(path)


def test_agfs_preserves_nul_archive(tmp_path):
    path = package(tmp_path, 'message', 'a\x00b', backend='agfs')
    target = service(Files())
    target.preview(path)
    assert target.import_package(path)['data']['archives']['count'] == 1


def test_relation_mirrors_once_per_source_across_commits():
    sql = SQLStorage.__new__(SQLStorage)
    sql.conn = MagicMock()
    cursor = sql.conn.cursor.return_value.__enter__.return_value
    tenant = None
    mirrors = {}
    inserts = []

    def commit():
        nonlocal tenant
        tenant = None

    def execute(query, params):
        nonlocal tenant
        if query.startswith('SELECT set_config'):
            tenant = params[0]
        elif query.startswith('INSERT'):
            assert params[-1] == tenant
            inserts.append(params)
        else:
            assert query == 'UPDATE context_nodes SET relations=%s::jsonb WHERE uri=%s AND account_id=%s'
            payload, uri, account = params
            assert tenant == account
            assert uri not in mirrors
            mirrors[uri] = payload.adapted

    sql.conn.commit.side_effect = commit
    cursor.execute.side_effect = execute
    records = []
    for account, name, count in [('a', 'first', 256), ('a', 'second', 257), ('b', 'third', 1000)]:
        uri = f'ctx://{account}/users/u/memories/preferences/{name}'
        records.extend(dict(from_uri=uri, to_uri=uri, relation_type=f'rel{i:04}',
                            reason='reason', weight=1.0) for i in range(count))
    sql.restore('relations', iter(records))
    assert len(inserts) == len(records)
    assert list(mirrors.values()) == [records[:256], records[256:513], records[513:]]
    assert sql.conn.commit.call_count == len(records) // 256 + 1


@pytest.mark.parametrize('payload', ['text', 'filters', 'metadata', 'model', 'archived',
                                   'disabled', 'unprojected', 'archive'])
def test_agfs_sql_vector_checks_only_projected_strings(tmp_path, payload):
    from types import SimpleNamespace

    from providers.vector_index.opengauss_index import OpenGaussVectorIndex
    from service.migration.rebuild import Rebuilder
    from tests.unit.service.migration.test_engine import Indexes

    row, archived = node(), archive()
    value = 'before\x00after'
    if payload in {'text', 'archived', 'disabled'}:
        row['content'] = value
    if payload == 'filters':
        row['category'] = value
    if payload == 'metadata':
        row['metadata']['who'] = value
    if payload == 'archived':
        row['status'] = 'ARCHIVED'
    if payload == 'disabled':
        row['metadata']['indexing_disabled'] = True
    if payload == 'unprojected':
        row['metadata']['custom'] = value
    if payload == 'archive':
        archived['messages'] = [{'content': value}]
    path = tmp_path / 'source.tar.gz'
    write_package(path, {'nodes': [row], 'archives': [archived], 'relations': []},
                  source_backend='agfs', source_version='test')

    class CheckedIndexes(Indexes):
        validate_records = Rebuilder.validate_records
        _records = Rebuilder._records

        def __init__(self, cfg):
            super().__init__(cfg)
            self.vector = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
            self.vector.count = lambda: 0
            self.vector.check_migration_target = lambda **kw: None
            self.embedder = SimpleNamespace(model=value if payload == 'model' else 'mock')
            self.provider = SimpleNamespace(embedding_dimension=8)

    files = Files()
    target = service(files)
    target.index_factory = CheckedIndexes
    for operation in (target.preview, target.import_package):
        if payload in {'text', 'filters', 'metadata', 'model'}:
            with pytest.raises(PackageError, match='SQL target cannot store NUL'):
                operation(path)
            assert not files.files and not files.directories
        else:
            operation(path)


@pytest.mark.parametrize('operation', ['restore', 'merge_nodes', 'replace_relations'])
def test_failed_sql_mutation_releases_transaction_before_recovery(operation):
    adapter = SQLStorage.__new__(SQLStorage)
    adapter.conn = MagicMock()
    cursor = adapter.conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = RuntimeError('database rejected statement')
    record = node()
    with pytest.raises(RuntimeError, match='database rejected'):
        if operation == 'restore':
            adapter.restore('nodes', [record])
        elif operation == 'merge_nodes':
            adapter.merge_nodes([record])
        else:
            adapter.replace_relations([dict(from_uri=record['uri'], to_uri=record['uri'],
                                           relation_type='related_to', reason='', weight=1.0)])
    adapter.conn.rollback.assert_called_once()
