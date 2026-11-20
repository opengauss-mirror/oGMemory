import multiprocessing

import pytest


def write_in_process(root):
    from core.models import IndexRecord
    from providers.vector_index.chroma_index import ChromaVectorIndex

    index = ChromaVectorIndex(persist_directory=root, dimension=3)
    index.upsert([IndexRecord(id='imported', uri='ctx://a/users/u/memories/preferences/imported', level=0,
                              text='imported', filters={'account_id': 'a', 'owner_space': 'user:u'},
                              metadata={'_embedding': [0., 1., 0.]})])
    index.release_for_migration()


def test_chroma_reopens_after_cross_process_import(tmp_path):
    pytest.importorskip('chromadb')
    from providers.vector_index.chroma_index import ChromaVectorIndex
    from tests.unit.service.migration.test_vector_backup import record

    root = str(tmp_path / 'chroma')
    original = ChromaVectorIndex(persist_directory=root, dimension=3)
    original.upsert([record()])
    filters = {'account_id': 'a', 'owner_space': 'user:u'}
    original.search_by_vector([1., 0., 0.], filters, 10)  # Populate the native query cache.
    identity = original.migration_identity()
    original.release_for_migration()
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(target=write_in_process, args=(root,))
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail('Chroma writer did not finish')
    assert process.exitcode == 0
    reopened = ChromaVectorIndex(persist_directory=root, dimension=3)
    try:
        assert reopened.migration_identity() == identity
        hits = reopened.search_by_vector([0., 1., 0.], filters, 10)
        assert len(hits) == 2
        assert hits[0].uri.endswith('/imported')
    finally:
        reopened.release_for_migration()
