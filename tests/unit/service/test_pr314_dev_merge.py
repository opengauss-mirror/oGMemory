"""Public read facade forwards independent Scene and graph request controls."""

from unittest.mock import Mock

import pytest

from core.models import SearchMemoryResult
from extraction import Extractor
from service.api import MemoryWriteAPI, ReadAPI
from tests.unit.retrieval.test_graph_supplement import CTX


@pytest.mark.parametrize("scene", [None, False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_search_controls_coexist_without_dropping_false_or_adding_default_keywords(scene, collect):
    pipe = Mock()
    pipe.run.return_value = SearchMemoryResult(hits=[], trace=None)
    ReadAPI(pipe).search_memory(
        "release facts", CTX, use_scene=scene, collect_graph_supplement=collect,
    )
    kwargs = pipe.run.call_args.kwargs
    assert kwargs.get("collect_graph_supplement", False) is collect
    assert ("collect_graph_supplement" in kwargs) is collect
    assert ("use_scene" in kwargs) is (scene is not None)
    if scene is not None:
        assert kwargs["use_scene"] is scene


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("reconcile", [False, True])
def test_graph_does_not_select_historical_extractor_or_override_native_occurrence_policy(graph, reconcile):
    api = MemoryWriteAPI(
        Mock(), Mock(), graph_enabled=graph,
        relation_store=Mock() if graph else None,
        source_occurrence_reconciliation_enabled=reconcile,
    )
    assert len(api._pipeline._extractors) == 1
    extractor = api._pipeline._extractors[0]
    assert type(extractor) is Extractor
    assert extractor._activation_enabled is graph
    assert extractor._source_occurrence_reconciliation_enabled is reconcile
    assert callable(extractor._occurrence_update_eligible)
