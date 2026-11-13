"""Worker merge wiring retains graph projection and Dev's independent modes."""

from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("service_threading")

from index.outbox_worker import OutboxWorker
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from tests.unit.retrieval.test_graph_supplement import PROFILE


@pytest.mark.parametrize("path", ["background", "synchronous"])
@pytest.mark.parametrize("scene,generation", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("graph", [False, True])
def test_both_worker_paths_wire_graph_independently_of_scene_generation(
    monkeypatch, path, scene, generation, graph,
):
    cfg = OgMemConfig(
        provider="mock", vector_db_type="memory", storage_backend="sql",
        graph_backend="sql" if graph else "none", scene_enabled=scene,
        scene_generation_enabled=generation, directory_summary_enabled=False,
        index_workers=1,
    )
    service = MemoryService(config=cfg)
    pool, fs, outbox, vector, embedder = (Mock() for _ in range(5))
    projection = Mock()
    projection_factory = Mock(return_value=projection)
    monkeypatch.setattr("providers.relation_store.sql_projection.SQLEdgeProjectionStore", projection_factory)
    monkeypatch.setattr("server.memory_service._HAS_SQL", True)
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    monkeypatch.setattr(service, "_graph_profile", lambda: PROFILE)
    monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda _: outbox)
    monkeypatch.setattr(service, "_get_shared_projection_index", lambda: vector)
    monkeypatch.setattr(service, "_get_shared_embedder", lambda: embedder)
    monkeypatch.setattr(service, "_supports_batch_claim", lambda _: False)
    scene_callback = Mock()
    monkeypatch.setattr(service, "_scene_projection_changed", scene_callback)
    # Run the production constructor but never start a polling/background loop.
    thread = Mock()
    monkeypatch.setattr("server.memory_service.threading.Thread", Mock(return_value=thread))
    workers = []
    stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    def construct(*args, **kwargs):
        worker = OutboxWorker(*args, **kwargs)
        worker.run_once = Mock(return_value=stats)
        workers.append(worker)
        return worker

    monkeypatch.setattr("index.outbox_worker.OutboxWorker", construct)
    if path == "background":
        service._start_outbox_worker()
        thread.start.assert_called_once()
    else:
        assert service.drain_outbox_sync(account_id="acme") == stats
        thread.start.assert_not_called()
    assert len(workers) == 1
    worker = workers[0]
    assert worker._edge_projection_store is (projection if graph else None)
    assert worker._edge_projection_profile is (PROFILE if graph else None)
    assert worker._memory_projection_handler is (scene_callback if scene and generation else None)
    assert worker._vector_index is vector and worker._embedder is embedder
    assert worker._agent_memory_handler == service._process_agent_memory
    assert worker._llm is None
    if graph:
        projection_factory.assert_called_once_with(pool=pool)
    else:
        projection_factory.assert_not_called()
    scene_callback.assert_not_called()
    if path == "synchronous":
        worker.run_once.assert_called_once_with(outbox_store=outbox, account_ids=["acme"])


def test_dev_buffer_and_score_gap_defaults_are_not_historical_benchmark_overrides():
    cfg = OgMemConfig(provider="mock", vector_db_type="memory")
    assert cfg.after_turn_threshold == 400
    assert cfg.source_occurrence_reconciliation_enabled is False
    items = [{"uri": str(i), "score": score} for i, score in enumerate([1., .98, .95, .84, .82])]
    # The .1158 relative gap is accepted by Dev's .10, but not historic .15.
    assert MemoryService._truncate_by_score_gap(items) == items[:3]
    assert MemoryService._truncate_by_score_gap(items, gap_ratio=.15) == items
