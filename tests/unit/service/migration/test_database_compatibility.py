"""Database changes are described without applying source configuration."""
from copy import deepcopy

import pytest

from service.migration.description import compatibility
from tests.unit.service.migration.test_description import environment


@pytest.mark.parametrize('database', ['opengauss', 'postgresql'])
@pytest.mark.parametrize('vector', ['opengauss', 'chroma', 'milvus', 'none'])
def test_database_transition_description(database, vector):
    _, source = environment()
    source['content_storage'].update(backend='sql', database_type=database)
    source['vector_storage']['backend'] = vector
    target = deepcopy(source)
    target['content_storage']['database_type'] = 'postgresql' if database == 'opengauss' else 'opengauss'
    target['vector_storage']['backend'] = 'chroma' if vector != 'chroma' else 'opengauss'
    before = deepcopy(target)
    result = compatibility(source, target)
    assert target == before
    assert 'content_storage.database_type' in result['changes']
    assert 'vector_storage.backend' in result['changes']
    assert any('SQL 数据库类型发生变化' in text for text in result['notices'])
    assert any('向量后端发生变化' in text for text in result['notices'])
