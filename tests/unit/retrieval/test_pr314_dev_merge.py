"""Graph and Dev Scene controls remain independent after the additive merge."""

from unittest.mock import Mock

import pytest

from core.models import RetrievalConfig
from retrieval.identity import canonical_uri
from retrieval.pipeline import RetrievalPipeline
from service.api import ReadAPI
from tests.unit.retrieval.test_graph_supplement import CTX, render
from tests.unit.retrieval.test_graph_third_review import components, pipeline
from tests.unit.retrieval.test_pipeline_scene import setup as scene_setup


def test_legacy_constructor_scene_enabled_positional_slot_is_not_graph_retriever():
    scene_ranker, scene_source, graph, supplement = Mock(), Mock(), Mock(), Mock()
    pipe = RetrievalPipeline(
        Mock(), Mock(), None, Mock(), RetrievalConfig(), None, None,
        scene_ranker, scene_source, False,
        graph_retriever=graph, graph_supplement=supplement,
    )
    assert pipe._scene_enabled is False
    assert pipe._scene_stage.source is scene_source
    assert pipe._graph_retriever is graph
    assert pipe._graph_supplement is supplement


@pytest.mark.parametrize("selection", [None, False])
def test_public_graph_search_and_append_work_with_scene_disabled(selection):
    _, reader, graph, lane, embedder, (a, b, c) = components()
    pipe = pipeline(reader, graph, lane, embedder, a)
    # Exercise both the service-level default-off and the explicit request-off.
    pipe._scene_enabled = selection is False
    scene = Mock()
    scene.bind.side_effect = AssertionError("Disabled Scene must not initialize")
    pipe._scene_stage = scene
    api = ReadAPI(pipe, read_service=reader, config=pipe.cfg)
    result = api.search_memory(
        "fact", CTX, context_type="MEMORY", fill_content_for_top_k=5,
        include_debug=True, collect_graph_supplement=True, use_scene=selection,
    )
    assert [hit.uri for hit in result.hits] == [a, b]
    assert c in [row["uri"] for row in result.trace.graph_expansion["two_hop_supplement"]["selected"]]
    original = [{"uri": hit.uri, "category": hit.category,
                 "content": hit.content_excerpt} for hit in result.hits]
    stats = {"graph_expansion": result.trace.graph_expansion}
    delivered = api.append_graph_supplement(
        original, "fact", stats, CTX, {"working_set": 10000}, render,
    )
    assert delivered[:len(original)] == original
    assert c in [row["uri"] for row in delivered[len(original):]]
    assert stats["graph_supplement_delivery"]["baseline_prefix_unchanged"] is True
    assert len(original) == 2
    scene.bind.assert_not_called()
    embedder.embed_texts.assert_not_called()  # Reuse the original query vector.


def test_public_both_off_search_preserves_native_evidence_even_when_components_exist():
    _, reader, graph, lane, embedder, (a, _, _) = components()
    pipe = pipeline(reader, graph, lane, embedder, a)
    pipe.cfg.enable_graph_expansion = False
    pipe._scene_enabled = False
    scene = Mock()
    scene.bind.side_effect = AssertionError("Both-off search must not initialize Scene")
    pipe._scene_stage = scene
    graph.search_with_report = Mock(side_effect=AssertionError("Graph is disabled"))
    lane.select = Mock(side_effect=AssertionError("Graph supplement is disabled"))
    result = ReadAPI(pipe, config=pipe.cfg).search_memory(
        "fact", CTX, context_type="MEMORY", include_debug=True,
        collect_graph_supplement=True,
    )
    assert [hit.uri for hit in result.hits] == [a]
    assert "two_hop_supplement" not in result.trace.graph_expansion
    scene.bind.assert_not_called()
    graph.search_with_report.assert_not_called()
    lane.select.assert_not_called()


@pytest.mark.parametrize("selection", [None, True, False])
def test_public_scene_only_search_ignores_graph_collection(scene_setup, selection):
    pipe = scene_setup.pipeline
    assert not pipe.cfg.enable_graph_expansion
    graph = Mock()
    graph.search_with_report.side_effect = AssertionError("Scene-only must not traverse graph")
    supplement = Mock()
    supplement.select.side_effect = AssertionError("Scene-only must not collect graph")
    pipe._graph_retriever, pipe._graph_supplement = graph, supplement
    result = ReadAPI(pipe, config=pipe.cfg).search_memory(
        "remember release", scene_setup.ctx, context_type="MEMORY",
        include_debug=True, collect_graph_supplement=True, use_scene=selection,
    )
    expected = {scene_setup.direct.uri}
    if selection is not False:
        expected.add(scene_setup.related.uri)
    assert {canonical_uri(hit.uri) for hit in result.hits} == expected
    assert "two_hop_supplement" not in result.trace.graph_expansion
    graph.search_with_report.assert_not_called()
    supplement.select.assert_not_called()
