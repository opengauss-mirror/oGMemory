"""Source transitions are checked before target access, for preview and import."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from service.migration.description import validate_migration_scope
from service.migration.engine import MigrationService
from service.migration.package import DATASETS, PackageError, write_package


def source_environment():
    return {
        'embedding': {'provider': 'mock', 'model': 'old-model', 'dimension': 3, 'model_version': None},
        'content_storage': {'backend': 'sql', 'database_type': 'opengauss', 'database_version': '6'},
        'vector_storage': {'backend': 'opengauss'},
    }


def target_config():
    return SimpleNamespace(storage_backend='sql', sql_dialect='opengauss',
                           vector_db_type='opengauss', embedding_provider='st',
                           st_model='new-model', opengauss_dimension=384)


def test_same_storage_allows_model_and_dimension_changes():
    source = source_environment()
    before = deepcopy(source)
    cfg = target_config()
    validate_migration_scope({'source_environment': source, 'source_backend': 'sql'}, cfg)
    assert source == before and cfg.st_model == 'new-model'


@pytest.mark.parametrize('change', ['content', 'database', 'vector', 'missing', 'unknown_database', 'unknown_vector'])
@pytest.mark.parametrize('operation', ['preview', 'merge'])
def test_unsupported_source_rejected_before_target_access(tmp_path, change, operation):
    source = source_environment()
    if change == 'content':
        source['content_storage'].update(backend='agfs', database_type=None)
    elif change in {'database', 'unknown_database'}:
        source['content_storage']['database_type'] = 'postgresql' if change == 'database' else None
    elif change in {'vector', 'unknown_vector'}:
        source['vector_storage']['backend'] = 'chroma' if change == 'vector' else None
    path = tmp_path / 'source.tar.gz'
    write_package(path, {kind: [] for kind in DATASETS},
                  source_backend=source['content_storage']['backend'], source_version='test',
                  source_environment=None if change == 'missing' else source)
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported source reached target resources or backup')
    migration = MigrationService(target_config(), adapter_factory=forbidden, index_factory=forbidden)
    with pytest.raises(PackageError, match='同源迁移'):
        if operation == 'preview':
            migration.preview(path)
        else:
            migration.merge(path, task='rejected', recovery_factory=forbidden, llm=None)
