"""Two service processes publishing to the same native openGauss scene index."""

from dataclasses import replace
from multiprocessing import get_context

import pytest

from fs.sql_adapter import SharedConnectionPool
from providers.lexical_index.native_sql_index import OpenGaussLexicalIndex
from scene.candidate_index import LazySceneIndex, SceneIndexRefresh
from scene.sql_store import SQLSceneRepository
from tests.integration.test_scene_sql_persistence import _scene, _write
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql


def _delayed_process(dsn, account, ready, resume, results):
    pool = SharedConnectionPool(dsn, pool_size=1, dialect="opengauss")
    repository = SQLSceneRepository(pool)
    projection = LazySceneIndex(
        lambda: OpenGaussLexicalIndex(connection_string=dsn, table_name="scene_native_probe"),
        projection_queue=repository,
    )
    try:
        old = repository.load_snapshot(account, "user:alice")
        ready.set()
        if not resume.wait(30):
            raise TimeoutError("projection writer did not resume")
        projection.search("Release", old, 5)
        results.put("completed")
    finally:
        projection.close()
        pool.close_all()


def test_native_projection_keeps_latest_revision_across_processes(scene_sql):
    pool, fs, repository, _, _, ctx = scene_sql
    if pool.dialect != "opengauss":
        pytest.skip("native BM25 acceptance runs on openGauss")
    memory = _write(fs, ctx, "projection")
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    old = replace(initial, revision=1, scenes=(_scene("release", memory),))
    assert repository.compare_and_swap(initial, old)
    runtime = get_context("spawn")
    ready, resume, results = runtime.Event(), runtime.Event(), runtime.Queue()
    delayed = runtime.Process(target=_delayed_process,
                              args=(pool.connection_string, ctx.account_id, ready, resume, results))
    native = OpenGaussLexicalIndex(connection_string=pool.connection_string, table_name="scene_native_probe")
    current = LazySceneIndex(lambda: native, projection_queue=repository)
    delayed.start()
    try:
        assert ready.wait(30)
        latest = replace(old, revision=2, scenes=(replace(old.scenes[0], version=2),))
        assert repository.compare_and_swap(old, latest)
        SceneIndexRefresh(repository, current)()
        resume.set()
        assert results.get(timeout=30) == "completed"
        delayed.join(30)
        assert delayed.exitcode == 0
        assert current.search("Release", latest, 5) == ["release"]
        assert list(repository.pending_projections()) == []
    finally:
        resume.set()
        delayed.join(30)
        if delayed.is_alive():
            delayed.terminate()
            delayed.join(10)
        current.close()
        results.close()
