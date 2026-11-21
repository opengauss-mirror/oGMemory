"""Database/vector transitions; content storage remains unchanged."""
import pytest

from tests.integration import test_memory_migration as common
from tests.integration.test_memory_migration import clean_isolated_databases
from tests.integration.test_migration_vector_free import exercise_vector_free_migration

pytestmark = common.pytestmark


@pytest.mark.parametrize('backend', ['sql', 'agfs'])
@pytest.mark.parametrize('vector_backend', ['chroma', 'milvus'])
def test_cross_vector_backend_migration(tmp_path, backend, vector_backend, request):
    common.exercise_migration(tmp_path, backend, backend, vector_backend, request,
                              source_vector_backend='opengauss')


@pytest.mark.parametrize('source_mode,target_mode', [('default', 'agfs'), ('agfs', 'default')])
def test_vector_mode_transition(tmp_path, monkeypatch, source_mode, target_mode):
    exercise_vector_free_migration(tmp_path, monkeypatch, source_mode, target_mode, 'literal')


@pytest.mark.parametrize('source_type,target_type', [
    ('postgresql', 'opengauss'), ('opengauss', 'postgresql'),
])
def test_cross_sql_database_type(tmp_path, source_type, target_type):
    import os
    from uuid import uuid4
    from contextlib import ExitStack
    import psycopg2
    from psycopg2.extensions import make_dsn
    from providers.unified_config import OgMemConfig
    from service.migration.storage import storage
    from tests.migration_support import MigrationHarness

    pytest.importorskip('chromadb')
    admins = {name: os.environ.get('MIGRATION_TEST_' + name.upper() + '_ADMIN_DSN')
              for name in ('postgresql', 'opengauss')}
    if not all(admins.values()):
        pytest.skip('isolated PostgreSQL and openGauss administrator DSNs required')
    def drop(admin, database):
        conn = psycopg2.connect(admin)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute('DROP DATABASE ' + database)
        finally:
            conn.close()
    with ExitStack() as cleanup:
        configs = []
        for label, dialect in [('source', source_type), ('target', target_type)]:
            database = 'migration_' + uuid4().hex
            conn = psycopg2.connect(admins[dialect])
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute('CREATE DATABASE ' + database + (
                        " DBCOMPATIBILITY 'PG'" if dialect == 'opengauss' else ''))
                cleanup.callback(drop, admins[dialect], database)
            finally:
                conn.close()
            configs.append(OgMemConfig(storage_backend='sql', sql_dialect=dialect,
                sql_connection_string=make_dsn(admins[dialect], dbname=database),
                provider='mock', embedding_provider='mock', vector_db_type='chroma',
                opengauss_dimension=8, graph_backend='none', retrieval_hybrid_enabled=False,
                chroma_persist_directory=str(tmp_path / label)))
        source, target = configs
        with storage(source) as adapter:
            for kind, rows in common.fixtures().items():
                adapter.restore(kind, rows)
        path = tmp_path / 'database.tar.gz'
        expected = MigrationHarness(source).export(path)['data']
        migration = MigrationHarness(target)
        preview = migration.preview(path)
        assert 'content_storage.database_type' in preview['compatibility']['changes']
        result = migration.import_package(path)
        common.assert_merged_package(path, target, result, expected)
