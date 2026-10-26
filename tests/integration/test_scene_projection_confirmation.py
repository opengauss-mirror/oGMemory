"""Projection recovery after reads precede durable scope creation."""

from dataclasses import replace

import pytest

from providers.lexical_index.native_sql_index import OpenGaussLexicalIndex
from scene.candidate_index import LazySceneIndex, SceneIndexRefresh
from tests.integration.test_scene_sql_persistence import _scene, _write
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_candidate_index import MemoryIndex


@pytest.fixture
def projection(scene_sql):
    pool, _, repository, _, _, _ = scene_sql
    provider = (OpenGaussLexicalIndex(connection_string=pool.connection_string, table_name="scene_confirmation")
                if pool.dialect == "opengauss" else MemoryIndex())
    index = LazySceneIndex(lambda: provider, projection_queue=repository)
    yield index
    index.close()


@pytest.mark.parametrize("create_scope", ["configuration", "enqueue"])
def test_empty_search_then_scope_creation_completes_projection(scene_sql, projection, create_scope):
    _, _, repository, jobs, config, ctx = scene_sql
    empty = repository.load_snapshot(ctx.account_id, "user:alice")
    assert projection.search("Release", empty, 5) == []
    if create_scope == "configuration":
        config.publish(ctx.account_id, "user:alice", 0, {"lexical_enabled": True})
    else:
        jobs.enqueue(ctx.account_id, "user:alice", ["pending-source"], token_count=1)
    SceneIndexRefresh(repository, projection)()
    assert list(repository.pending_projections()) == []


def test_empty_scope_batch_yields_to_next_scene_projection(scene_sql, projection):
    _, fs, repository, _, config, ctx = scene_sql
    for number in range(20):
        owner = f"user:empty-{number}"
        projection.search("Release", repository.load_snapshot(ctx.account_id, owner), 5)
        config.publish(ctx.account_id, owner, 0, {"lexical_enabled": True})
    memory = _write(fs, ctx, "later-scene")
    original = repository.load_snapshot(ctx.account_id, "user:alice")
    current = replace(original, revision=1, scenes=(_scene("release", memory),))
    assert repository.compare_and_swap(original, current)
    refresh = SceneIndexRefresh(repository, projection)
    refresh()
    refresh()
    assert list(repository.pending_projections()) == []
    assert projection.search("Release", current, 5) == ["release"]
