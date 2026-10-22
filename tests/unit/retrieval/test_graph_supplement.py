"""Live graph lane contract: existing services, no LLM, append-only delivery."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import NodeNotFoundError
from core.models import IndexRecord, RequestContext, RetrievedBlock, RetrievalConfig
from core.models import SeedResult, TypedQuery
from retrieval.graph_supplement import GraphSupplement
from retrieval.pipeline import RetrievalPipeline
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import ReadAPI
from service.graph_service import GraphService

PROFILE = dict(profile_id="test", model="test-embedding", dimension=2,
               representation_version="relation-target-v2")
CTX = RequestContext("acct", "user", "agent", "session", "trace")
ROOT = "ctx://acct/users/user/memories/entities/"
FORMAT = MemoryService._format_working_set


def render(items, budget):
    return FORMAT(None, items, budget)


def fixture():
    blocks = {ROOT+n: RetrievedBlock(uri=ROOT+n, abstract=n, overview=n,
              content_excerpt="Memory " + n, owner_space="user:user", category="entity",
              metadata={"version": 1}) for n in "abcdef"}
    edges = []

    def edge(a, b, *, vector=(1., 0.), weight=1., relation="related", version=1):
        edges.append(dict(source_uri=ROOT+a, target_uri=ROOT+b, relation_type=relation,
            weight=weight, source_revision=version, target_revision=1,
            edge_projection=dict(vector=list(vector), embedding_profile="test",
                model="test-embedding", dimension=2, representation_version="relation-target-v2",
                source_revision=version, target_revision=1)))

    reader = Mock()
    def read(uri, ctx):
        if uri not in blocks:
            raise NodeNotFoundError(uri)
        return blocks[uri]
    reader.read_authorized.side_effect = read
    relation_reader = Mock()
    relation_reader.get_traversal_neighbors.side_effect = lambda uri, ctx, limit, embedding_profile=None: [
        e for e in edges if uri in (e["source_uri"], e["target_uri"])]
    graph = GraphService(None, relation_reader=relation_reader)
    index = Mock(spec=["get_record"])
    index.get_record.return_value = None
    embedder = Mock()
    embedder.embed_texts.return_value = [[1., 0.]]
    lane = GraphSupplement(graph, reader, index, embedder, PROFILE)
    return lane, blocks, edge


def select(lane, query="memory", anchor="a"):
    candidates = [SimpleNamespace(uri=ROOT+anchor)]
    return lane.select(query, candidates, candidates, CTX, query_vector=[1., 0.])


def test_service_delegation_two_hops_and_stored_query_reuse():
    lane, blocks, edge = fixture()
    edge("a", "b")
    edge("b", "c")
    edge("c", "d")
    result = select(lane)
    assert [r["uri"] for r in result["selected"]] == [ROOT+"b", ROOT+"c"]
    assert result["selected"][1]["path"] == [ROOT+n for n in "abc"]
    assert result["selected"][1]["score"] == pytest.approx(.8)
    lane.embedder.embed_texts.assert_not_called()
    assert result["generative_llm_calls"] == 0
    lane.graph._relation_reader.get_traversal_neighbors.assert_called()


def test_parallel_edges_use_selected_edge_path_quality():
    lane, _, edge = fixture()
    edge("a", "b", weight=.4)
    edge("a", "b", weight=.9, relation="other")
    edge("b", "c")
    result = select(lane)
    assert next(r for r in result["selected"] if r["uri"] == ROOT+"c")["score"] == pytest.approx(.72)


@pytest.mark.parametrize("change", ["supersedes", "conflict_alert", "stale", "denied", "inactive", "foreign"])
def test_rejected_edges_and_nodes(change):
    lane, blocks, edge = fixture()
    edge("a", "b", relation=change if change in ("supersedes", "conflict_alert") else "related",
         version=2 if change == "stale" else 1)
    if change == "denied":
        del blocks[ROOT+"b"]
    if change == "inactive":
        blocks[ROOT+"b"].metadata["status"] = "archived"
    if change == "foreign":
        lane.graph._relation_reader.get_traversal_neighbors.side_effect = None
        lane.graph._relation_reader.get_traversal_neighbors.return_value = [dict(
            source_uri=ROOT+"a", target_uri="ctx://acct/users/other/memories/x", relation_type="related")]
    assert select(lane)["selected"] == []


def test_incoming_node_vector_only_adds_eligibility_not_score():
    lane, blocks, edge = fixture()
    edge("b", "a", vector=(0., 1.))
    assert select(lane)["selected"] == []
    lane.index.get_record.return_value = IndexRecord(
        id="id", uri=ROOT+"b", level=0, text="b",
        filters={"account_id": "acct", "owner_space": "user:user"},
        metadata={"record_kind": "node_abstract", "canonical_revision": 1, "_embedding": [1., 0.],
                  "embedding_model": PROFILE["model"], "embedding_dimension": 2})
    result = select(lane)
    assert result["selected"][0]["incoming"] is True
    assert result["selected"][0]["score"] == 0
    # Revision and contradictory model reject node-vector eligibility.
    lane.index.get_record.return_value.metadata["canonical_revision"] = 2
    assert select(lane)["selected"] == []
    lane.index.get_record.return_value.metadata.update(canonical_revision=1, model="wrong")
    assert select(lane)["selected"] == []


def test_delivery_reuses_native_formatter_without_reordering_or_clipping():
    lane, blocks, edge = fixture()
    edge("a", "b")
    original = [dict(uri=ROOT+"a", content="Original evidence", category="entity")]
    stats = {"graph_expansion": {"two_hop_supplement": select(lane)}}
    budget = {"working_set": 1000}
    result = lane.deliver(original, "memory", stats, CTX, budget, render)
    assert result[:1] == original and len(original) == 1
    assert result[1]["uri"] == ROOT+"b"
    assert render(result, budget).startswith(render(original, budget))
    assert stats["graph_supplement_delivery"]["added_uris"] == [ROOT+"b"]


@pytest.mark.parametrize("budget", [{}, {"working_set": 0}, {"working_set": 100}, {"working_set": 101}])
def test_no_addition_when_native_budget_missing_small_or_exhausted(budget):
    lane, _, edge = fixture()
    edge("a", "b")
    items = [dict(uri=ROOT+"a", content="x"*1000)]
    stats = {"graph_expansion": {"two_hop_supplement": select(lane)}}
    assert lane.deliver(items, "memory", stats, CTX, budget, render) == items


@pytest.mark.parametrize("change", ["revision", "body", "authorization"])
def test_revalidate_at_injection(change):
    lane, blocks, edge = fixture()
    edge("a", "b")
    stats = {"graph_expansion": {"two_hop_supplement": select(lane)}}
    if change == "revision":
        blocks[ROOT+"b"].metadata["version"] = 2
    elif change == "body":
        blocks[ROOT+"b"].content_excerpt = "changed"
    else:
        del blocks[ROOT+"b"]
    assert lane.deliver([], "memory", stats, CTX, {"working_set": 1000}, render) == []


def test_query_and_context_binding():
    lane, _, edge = fixture()
    edge("a", "b")
    stats = {"graph_expansion": {"two_hop_supplement": select(lane)}}
    for query, ctx in (("different", CTX), ("memory", replace(CTX, session_id="other"))):
        with pytest.raises(ValueError, match="Cross-request"):
            lane.deliver([], query, stats, ctx, {"working_set": 1000}, render)


def test_shared_instance_has_no_cross_query_state():
    lane, _, edge = fixture()
    edge("a", "b")
    edge("e", "f")
    def run(anchor):
        return [r["uri"] for r in select(lane, anchor, anchor)["selected"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, ["a", "e"]*10)) == [[ROOT+"b"], [ROOT+"f"]]*10


def test_read_facade_preserves_original_evidence_on_failure_and_graph_off():
    lane, _, edge = fixture()
    edge("a", "b")
    pipeline = SimpleNamespace(_graph_supplement=lane)
    cfg = RetrievalConfig(enable_graph_expansion=True)
    api = ReadAPI(pipeline, config=cfg)
    items = [{"uri": ROOT+"a", "content": "unchanged"}]
    stats = {"graph_expansion": {"two_hop_supplement": select(lane)}}
    assert api.append_graph_supplement(items, "wrong query", stats, CTX, {}, render) is items
    assert stats["graph_supplement_delivery"]["error"] == "ValueError"
    cfg.enable_graph_expansion = False
    lane.deliver = Mock(side_effect=AssertionError("Graph off must not read"))
    assert api.append_graph_supplement(items, "memory", {}, CTX, {}, render) is items
    lane.deliver.assert_not_called()


@pytest.mark.parametrize("enabled,opt_in,supplement_enabled", [
    (False, True, True), (True, False, True), (True, True, False), (True, True, True)])
def test_pipeline_only_collects_for_opted_in_primary_graph_search(enabled, opt_in, supplement_enabled):
    lane, _, edge = fixture()
    edge("a", "b")
    lane.select = Mock(wraps=lane.select)
    pipe = RetrievalPipeline(Mock(), Mock(), None, Mock(), RetrievalConfig(
        enable_graph_expansion=enabled, graph_supplement_enabled=supplement_enabled),
        graph_retriever=Mock(), graph_supplement=lane)
    pipe._run_seed = Mock(return_value=SeedResult(query_vector=[1., 0.]))
    leaf = SimpleNamespace(uri=ROOT+"a")
    pipe._run_expand = Mock(return_value=[leaf])
    pipe._expand_by_graph = Mock(return_value=[leaf])
    pipe._run_assembly = Mock(return_value=[])
    pipe._run_planner = Mock(return_value=[TypedQuery("memory", "MEMORY", [])])
    result = pipe.run("memory", CTX, collect_graph_supplement=opt_in)
    if enabled and opt_in and supplement_enabled:
        lane.select.assert_called_once()
        assert result.trace.graph_expansion["two_hop_supplement"]["selected"]
        lane.embedder.embed_texts.assert_not_called()
    else:
        lane.select.assert_not_called()


def test_supplement_config_yaml_then_environment(monkeypatch):
    from providers import unified_config
    monkeypatch.setenv("GRAPH_SUPPLEMENT_ENABLED", "false")
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {"graph": {}})
    assert OgMemConfig.load("unused").graph_supplement_enabled is False
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {"graph": {"supplement_enabled": True}})
    assert OgMemConfig.load("unused").graph_supplement_enabled is True


def test_connected_anchor_limit_and_total_graph_delivery_cap():
    lane, blocks, edge = fixture()
    for i in range(20):
        for name in (f"anchor{i}", f"neighbor{i}"):
            blocks[ROOT+name] = replace(blocks[ROOT+"a"], uri=ROOT+name,
                                       content_excerpt=name, metadata={"version": 1})
        edge(f"anchor{i}", f"neighbor{i}")
    candidates = [SimpleNamespace(uri=ROOT+f"anchor{i}") for i in range(20)]
    audit = lane.select("memory", candidates, candidates, CTX, query_vector=[1., 0.])
    assert audit["anchors"] == [ROOT+f"anchor{i}" for i in range(5)]
    items = [dict(uri=ROOT+f"original{i}", content="Original", metadata={"retrieval_origin": "graph_neighbor"})
             for i in range(14)]
    result = lane.deliver(items, "memory", {"graph_expansion": {"two_hop_supplement": audit}},
                          CTX, {"working_set": 10000}, render)
    assert len(result) == 15 and result[:14] == items


def test_local_lane_union_is_at_most_six_unique_neighbors():
    lane, blocks, edge = fixture()
    for i in range(30):
        name = f"neighbor{i}"
        blocks[ROOT+name] = replace(blocks[ROOT+"a"], uri=ROOT+name,
                                   content_excerpt="rare "*i, metadata={"version": 1})
        edge("a", name, weight=1/(i+1))
    audit = select(lane, "rare")
    assert 3 <= len(audit["selected"]) <= 6
    assert len(audit["selected"]) == len({r["uri"] for r in audit["selected"]})


def test_nonfinite_query_rejected_and_embedding_count_visible():
    lane, _, edge = fixture()
    edge("a", "b")
    with pytest.raises(ValueError):
        lane.select("memory", [SimpleNamespace(uri=ROOT+"a")], [], CTX, query_vector=[float("nan"), 0.])
    audit = lane.select("memory", [SimpleNamespace(uri=ROOT+"a")], [], CTX)
    assert audit["additional_embedding_calls"] == 1
    lane.embedder.embed_texts.assert_called_once_with(["memory"])
