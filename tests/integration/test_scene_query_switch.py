"""Read persisted scenes with independent query selection and a paused generator."""

from dataclasses import replace

import pytest

from index.index_record_builder import build_index_records
from providers.unified_config import OgMemConfig
from providers.vector_index.opengauss_index import OpenGaussVectorIndex
from retrieval.identity import canonical_uri
from scene.models import SceneMember
from server.memory_service import MemoryService
from tests.integration.test_scene_sql_persistence import scene_sql, _write, _scene
from tests.unit.scene.test_adapters import Embedder


@pytest.mark.parametrize("retrieval_default", [True, False])
def test_read_only_scene_runtime_loads_persisted_snapshot(scene_sql, monkeypatch, retrieval_default):
    pool, fs, repository, jobs, _, ctx = scene_sql
    direct, related = [_write(fs, ctx, name) for name in ("release", "follow-up")]
    index = OpenGaussVectorIndex(pool=pool, dimension=2, table_name="scene_query_vectors")
    for node, vector in ((direct, [1.0, 0.0]), (related, [0.0, 1.0])):
        records = build_index_records(node)
        for record in records:
            record.metadata["_embedding"] = vector
        index.upsert(records)
    initial = repository.load_snapshot(ctx.account_id, "user:alice")
    record = _scene("release-scene", direct)
    record = replace(record, cluster=replace(record.cluster, members=(
        SceneMember(direct.uri, 1.0, node_version=1), SceneMember(related.uri, 1.0, node_version=1),
    )))
    published = replace(initial, revision=1, scenes=(record,))
    assert repository.compare_and_swap(initial, published)
    job = jobs.enqueue(ctx.account_id, "user:alice", [direct.uri], force=True)
    service = MemoryService(config=OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        sql_connection_string=pool.connection_string, scene_enabled=True,
        scene_generation_enabled=False, scene_retrieval_enabled=retrieval_default,
        retrieval_expand_hierarchy=False,
    ))
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_shared_vector_index", lambda: index)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: Embedder())
    monkeypatch.setattr(service, "_get_archive_store", lambda: None)
    monkeypatch.setattr(service, "_get_relation_store", lambda: None)
    try:
        api = service.get_read_api(start_outbox_worker=False)
        for selection in (None, True, False, None):
            result = api.search_memory("release", ctx, use_scene=selection, score_threshold=0.75)
            expected = retrieval_default if selection is None else selection
            assert {canonical_uri(hit.uri) for hit in result.hits} == (
                {direct.uri, related.uri} if expected else {direct.uri}
            )
        assert repository.load_snapshot(ctx.account_id, "user:alice") == published
        state = jobs.summary(ctx.account_id, "user:alice")
        assert state["active_job"]["job_id"] == job.job_id
        assert state["active_job"]["status"] == "PENDING"
        assert service.get_scene_service().status(ctx)["running"] is False
    finally:
        service.shutdown(join_timeout=2)
