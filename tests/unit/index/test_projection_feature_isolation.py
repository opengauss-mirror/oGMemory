"""Preserve upstream Scene arguments and each separately enabled feature."""
from unittest.mock import Mock

from index.outbox_worker import OutboxWorker


def test_upstream_positional_scene_callback_keeps_its_parameter_slot():
    callback = Mock()
    worker = OutboxWorker(Mock(), Mock(), None, None, None, True, None, None, callback)
    assert worker._memory_projection_handler is callback
    assert worker._edge_projection_store is None


def test_graph_only_drain_does_not_require_scene_handler():
    store, embedder = Mock(), Mock()
    profile = {"profile_id": "test"}
    store.drain_sync.return_value = 2
    worker = OutboxWorker(Mock(), embedder, edge_projection_components=(store, profile))
    stats = {}
    worker._drain_graph(["account"], stats)
    store.drain_sync.assert_called_once_with("account", embedder, profile)
    assert worker._memory_projection_handler is None
    assert stats == {"edge_projections": 2}


def test_scene_only_has_no_graph_projection_work():
    worker = OutboxWorker(Mock(), Mock(), memory_projection_handler=Mock())
    stats = {}
    worker._drain_graph(["account"], stats)
    assert worker._edge_projection_store is None
    assert stats == {}
