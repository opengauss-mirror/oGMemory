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
