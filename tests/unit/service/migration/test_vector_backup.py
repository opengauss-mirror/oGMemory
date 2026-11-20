from dataclasses import replace

import pytest

from core.models import IndexRecord
from service.migration.package import PackageError
from service.migration.vector_backup import VectorBackup


@pytest.fixture
def vector(tmp_path):
    pytest.importorskip("chromadb")
    from providers.vector_index.chroma_index import ChromaVectorIndex
    return ChromaVectorIndex(collection_name="backup-test", persist_directory=str(tmp_path / "chroma"), dimension=3)


def record():
    return IndexRecord(id="original", uri="ctx://a/users/u/memories/preferences/x", level=0,
                       text="original", filters={"account_id": "a", "owner_space": "user:u"},
                       metadata={"_embedding": [1., 0., 0.], "embedding_model": "test", "embedding_dimension": 3})


def test_restore_actual_vectors_and_remove_imported_rows(vector, tmp_path):
    vector.upsert([record()])
    backup = VectorBackup(vector, lambda: {"collection": "backup-test"})
    path = tmp_path / "backup.sqlite"
    manifest = backup.capture(path)
    vector.upsert([replace(record(), text="changed"), replace(record(), id="imported")])
    backup.restore(path, manifest)
    assert vector.count() == 1
    assert vector.get_record("original").text == "original"
    assert vector.get_record("original").metadata["_embedding"] == [1., 0., 0.]
    backup.restore(path, manifest)  # Retry after a lost completion response.
    assert vector.count() == 1


def test_wrong_identity_and_corrupt_backup_reject_before_delete(vector, tmp_path):
    vector.upsert([record()])
    backup = VectorBackup(vector, lambda: {"collection": "original"})
    path = tmp_path / "backup.sqlite"
    manifest = backup.capture(path)
    backup.identity = lambda: {"collection": "other"}
    with pytest.raises(PackageError, match="target differs"):
        backup.restore(path, manifest)
    backup.identity = lambda: {"collection": "original"}
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(PackageError, match="checksum"):
        backup.restore(path, manifest)
    assert vector.count() == 1


def test_recovery_comparison_tolerates_only_vector_roundoff():
    from service.migration.vector_backup import _encode, _equivalent

    original = record()
    original.metadata['_embedding'] = [-0.5390625, -0.3359375298023224, -0.09375]
    encoded = _encode(original)
    restored = replace(original, metadata={**original.metadata,
        '_embedding': [-0.5390625, -0.3359375596046448, -0.09375]})
    assert _equivalent(encoded, restored)
    assert not _equivalent(encoded, replace(restored, text='changed'))
    assert not _equivalent(encoded, replace(restored, filters={'account_id': 'other'}))
    assert not _equivalent(encoded, replace(restored, metadata={**restored.metadata,
        'embedding_model': 'other'}))
    assert not _equivalent(encoded, replace(restored, metadata={**restored.metadata,
        '_embedding': [-0.5390625, -0.3359, -0.09375]}))
