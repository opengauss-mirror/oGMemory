from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('bad_dimension', [False, True])
def test_legacy_chroma_uses_query_dimension_validation(bad_dimension):
    from providers.vector_index.chroma_index import ChromaVectorIndex
    class LegacyCollection:
        name = 'legacy'
        def count(self):
            return 0
        def query(self, **kwargs):
            if bad_dimension:
                raise ValueError('stored dimension differs')
            return {}
    legacy = LegacyCollection()
    index = object.__new__(ChromaVectorIndex)
    index._locked = nullcontext
    index._collection = legacy
    index._client = SimpleNamespace(get_collection=lambda name: legacy)
    index._dimension = 3
    if bad_dimension:
        with pytest.raises(ValueError, match='dimension'):
            index.check_migration_target()
    else:
        index.check_migration_target()
    # The old SDK cannot meet cross-process metadata-only preflight semantics.
    with pytest.raises(ValueError, match='dimension'):
        index.check_migration_target(include_embedding=False)
