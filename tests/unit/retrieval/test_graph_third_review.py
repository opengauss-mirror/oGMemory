"""PR314: canonical snapshots, requested context types and lexical graph recall.

The fake filesystem applies writes at real stage boundaries. It does not fabricate
GraphHits, bypass canonical hydration or depend on the implementation's receipt
field names.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import NodeNotFoundError
from core.lexical import LexicalHit
from core.models import ContextNode, LeafHit, RelationNeighbor, RetrievalConfig, SeedResult, TypedQuery
from retrieval.context_reader import ContextReader
from retrieval.graph_retriever import OneHopRetriever
from retrieval.graph_supplement import GraphSupplement
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from service.api import ReadAPI
from service.graph_service import GraphService
from tests.unit.retrieval.test_graph_supplement import CTX, PROFILE, ROOT, render


class MemoryFS:
    def __init__(self, nodes):
        self.nodes = nodes
        self.before_read = None
        self.reads = {}

    def read_node(self, uri, ctx):
        self.reads[uri] = self.reads.get(uri, 0) + 1
        if self.before_read:
            self.before_read(uri, self.reads[uri])
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return deepcopy(self.nodes[uri])

    def exists(self, uri, ctx):
        node = self.nodes.get(uri)
        return node is not None and node.metadata.get("status", "active") == "active"


def node(uri, *, context_type="MEMORY", version=1):
    skill = context_type == "SKILL"
    return ContextNode(uri=uri, context_type=context_type, category="tool" if skill else "entity",
        level=3, owner_space="agent:agent" if skill else "user:user", abstract=uri.rsplit("/", 1)[-1],
        overview="Stable overview", content="Original canonical body", metadata={"version": version})


def projection():
    return dict(vector=[1., 0.], embedding_profile=PROFILE["profile_id"], model=PROFILE["model"],
                dimension=2, representation_version=PROFILE["representation_version"],
                source_revision=1, target_revision=1)


def components(*, skill_bridge=False, before_edge_return=None):
    a, b, c = ROOT+"a", ROOT+"b", ROOT+"c"
    if skill_bridge:
        b = "ctx://acct/agents/agent/skills/tool"
    fs = MemoryFS({a: node(a), b: node(b, context_type="SKILL" if skill_bridge else "MEMORY"), c: node(c)})
    reader = ContextReader(fs, storage_backend="sql")
    rows = [dict(source_uri=s, target_uri=t, relation_type="supports", weight=1.,
                 source_revision=1, target_revision=1, edge_projection=projection())
            for s, t in ((a, b), (b, c))]
    relations = Mock()

    def outgoing(uri, ctx, **kwargs):
        result = [RelationNeighbor(uri=e["target_uri"], category=fs.nodes[e["target_uri"]].category,
            reason="Old revision-one relationship", weight=1., relation_type="supports",
            source_revision=1, target_revision=1, edge_projection=projection())
            for e in rows if e["source_uri"] == uri]
        if before_edge_return:
            before_edge_return(fs)
        return result

    relations.get_relation_neighbors.side_effect = outgoing
    relations.get_traversal_neighbors.side_effect = lambda uri, ctx, **kwargs: [
        dict(e) for e in rows if uri in (e["source_uri"], e["target_uri"])]
    service = GraphService(None, relation_reader=relations)
    embedder = Mock()
    embedder.embed_texts.return_value = [[1., 0.]]
    graph = OneHopRetriever(service, embedder=embedder, embedding_profile=PROFILE)
    index = Mock(spec=["get_record"])
    index.get_record.return_value = None
    lane = GraphSupplement(service, reader, index, embedder, PROFILE)
    return fs, reader, graph, lane, embedder, (a, b, c)


def pipeline(reader, graph, lane, embedder, anchor, *, before_assembly=None):
    cfg = RetrievalConfig(enable_graph_expansion=True, expand_hierarchy=False)
    assembly = ResultRanker(cfg)
    if before_assembly:
        original = assembly.assemble

        def assemble(*args, **kwargs):
            before_assembly()
            return original(*args, **kwargs)

        assembly.assemble = assemble
    seed = Mock()
    seed.embedder = embedder
    seed.search.return_value = SeedResult(query_vector=[1., 0.])
    result = RetrievalPipeline(Mock(), seed, None, assembly, cfg, context_reader=reader,
                               graph_retriever=graph, graph_supplement=lane)
    result._run_planner = Mock(return_value=[TypedQuery("fact", "MEMORY", [], top_k=5)])
    result._run_expand = Mock(return_value=[LeafHit(anchor, .9, category="entity",
        content_excerpt="Original ordinary evidence", metadata={"version": 1})])
    return result


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_one_hop_discards_projection_if_endpoint_updated_after_edge_query(endpoint):
    def update(fs):
        uri = ROOT+("a" if endpoint == "source" else "b")
        fs.nodes[uri] = replace(fs.nodes[uri], content="New revision-two unrelated content", metadata={"version": 2})

    fs, reader, graph, lane, embedder, (a, b, _) = components(before_edge_return=update)
    pipe = pipeline(reader, graph, lane, embedder, a)
    api = ReadAPI(pipe, read_service=reader, config=pipe.cfg)
    result = api.search_memory("fact", CTX, context_type="MEMORY", include_debug=True)
    assert [hit.uri for hit in result.hits] == [a]
    assert not any("New revision-two" in (hit.content_excerpt or "") for hit in result.hits)


@pytest.mark.parametrize("change", ["source_revision", "source_archived", "target_revision", "target_archived"])
def test_one_hop_revalidates_projection_after_final_assembly_rereads(change):
    fs, reader, graph, lane, embedder, (a, b, _) = components()

    def mutate():
        uri = a if change.startswith("source") else b
        metadata = {"version": 2} if change.endswith("revision") else {"version": 1, "status": "archived"}
        fs.nodes[uri] = replace(fs.nodes[uri], content="Changed after graph hydration", metadata=metadata)

    pipe = pipeline(reader, graph, lane, embedder, a, before_assembly=mutate)
    result = ReadAPI(pipe).search_memory("fact", CTX, context_type="MEMORY", fill_content_for_top_k=5)
    assert b not in [hit.uri for hit in result.hits]


def test_one_hop_unchanged_projection_keeps_score_reason_and_content():
    fs, reader, graph, lane, embedder, (a, b, _) = components()
    result = ReadAPI(pipeline(reader, graph, lane, embedder, a)).search_memory(
        "fact", CTX, context_type="MEMORY", fill_content_for_top_k=5)
    added = next(hit for hit in result.hits if hit.uri == b)
    assert added.score == 1.
    assert added.content_excerpt == "Original canonical body"
    assert added.metadata["graph_explanation"] == "Old revision-one relationship"


def test_all_memories_keeps_skill_hits_and_expands_only_memory_seeds():
    fs, reader, graph, lane, embedder, (anchor, neighbor, second_hop) = components()
    skill = "ctx://acct/agents/agent/skills/deploy"
    fs.nodes[skill] = node(skill, context_type="SKILL")
    pipe = pipeline(reader, graph, lane, embedder, anchor)
    pipe._run_planner = RetrievalPipeline._run_planner.__get__(pipe)
    pipe.planner = QueryPlanner(pipe.cfg)
    pipe._run_expand.side_effect = lambda tq, *args, **kwargs: (
        [LeafHit(anchor, .8, category="entity")] if tq.context_type == "MEMORY" else
        [LeafHit(skill, .95, category="tool")] if not tq.categories else []
    )
    request = {"hints": {"include_all_memories": True}, "collect_graph_supplement": True,
               "include_debug": True, "fill_content_for_top_k": 5}
    result = ReadAPI(pipe).search_memory("fact", CTX, **request)
    hits = {hit.uri: hit for hit in result.hits}
    supplement = (result.trace.graph_expansion or {}).get("two_hop_supplement", {})
    selected = {row["uri"] for row in supplement.get("selected", [])}
    print(f"request={request}; hits={list(hits)}; two_hop={sorted(selected)}")
    assert {anchor, neighbor, skill} <= hits.keys()
    assert hits[neighbor].metadata["graph_anchor_uri"] == anchor
    assert second_hop in selected


def test_explicit_memory_search_rejects_skill_results_and_skill_bridge():
    fs, reader, graph, lane, embedder, (a, b, c) = components(skill_bridge=True)
    pipe = pipeline(reader, graph, lane, embedder, a)
    result = ReadAPI(pipe).search_memory("fact", CTX, context_type="MEMORY", include_debug=True,
                                        collect_graph_supplement=True)
    assert [hit.uri for hit in result.hits] == [a]
    supplement = result.trace.graph_expansion["two_hop_supplement"]
    assert supplement["selected"] == []
    assert b not in supplement["expanded_nodes"]
    assert c not in [row["uri"] for row in supplement["selected"]]


def test_unfiltered_supplement_still_allows_cross_type_edges():
    fs, reader, graph, lane, embedder, (a, b, c) = components(skill_bridge=True)
    initial = [SimpleNamespace(uri=a)]
    selected = lane.select("fact", initial, initial, CTX, query_vector=[1., 0.])
    assert {row["uri"] for row in selected["selected"]} == {b, c}


def test_memory_supplement_delivery_rechecks_context_type_after_selection():
    fs, reader, graph, lane, embedder, (a, b, _) = components()
    pipe = pipeline(reader, graph, lane, embedder, a)
    search = ReadAPI(pipe).search_memory("fact", CTX, context_type="MEMORY", include_debug=True,
                                         collect_graph_supplement=True)
    # The first hop is already ordinary output; the second hop remains supplemental.
    c = ROOT+"c"
    assert c in [row["uri"] for row in search.trace.graph_expansion["two_hop_supplement"]["selected"]]
    fs.nodes[c] = replace(fs.nodes[c], context_type="SKILL")
    items = [{"uri": a, "content": "Original ordinary evidence", "category": "entity"}]
    stats = {"graph_expansion": search.trace.graph_expansion}
    assert lane.deliver(items, "fact", stats, CTX, {"working_set": 10000}, render) == items


def test_lexical_only_search_gets_valid_query_embedding_for_graph_supplement():
    fs, reader, _, lane, embedder, (a, b, _) = components()
    cfg = RetrievalConfig(hybrid_enabled=True, semantic_weight=0., lexical_weight=1.,
                          enable_graph_expansion=True, expand_hierarchy=False)
    lexical = Mock()
    lexical.supported_filters = frozenset({"account_id", "owner_space", "level", "context_type"})
    lexical.search.return_value = [LexicalHit("lexical-a", 1., dict(uri=a, category="entity",
        context_type="MEMORY", level=2, account_id="acct", owner_space="user:user"))]
    vector = Mock()
    seed = SeedRetriever(vector, embedder, cfg, lexical, SimpleNamespace(ensure_ready=lambda *args: True))
    pipe = RetrievalPipeline(Mock(), seed, None, ResultRanker(cfg), cfg,
                             context_reader=reader, graph_supplement=lane)
    pipe._run_planner = Mock(return_value=[TypedQuery("fact", "MEMORY", [], top_k=5,
                                                     account_id="acct", owner_space="user:user")])
    result = ReadAPI(pipe).search_memory("fact", CTX, context_type="MEMORY", include_debug=True,
                                        collect_graph_supplement=True)
    assert [hit.uri for hit in result.hits] == [a]
    assert "two_hop_supplement_error" not in result.trace.graph_expansion
    assert b in [row["uri"] for row in result.trace.graph_expansion["two_hop_supplement"]["selected"]]
    embedder.embed_texts.assert_called_once_with(["fact"])
    vector.search_by_vector.assert_not_called()


@pytest.mark.parametrize("change", ["source_revision", "source_archived", "bridge_revision",
                                    "bridge_archived", "bridge_deleted", "bridge_context_type"])
def test_two_hop_final_delivery_rechecks_entire_path(change):
    fs, reader, graph, lane, embedder, (a, b, c) = components()
    search = ReadAPI(pipeline(reader, graph, lane, embedder, a)).search_memory(
        "fact", CTX, context_type="MEMORY", include_debug=True, collect_graph_supplement=True)
    audit = search.trace.graph_expansion["two_hop_supplement"]
    assert c in [row["uri"] for row in audit["selected"]]
    uri = a if change.startswith("source") else b
    if change.endswith("deleted"):
        del fs.nodes[uri]
    elif change.endswith("context_type"):
        fs.nodes[uri] = replace(fs.nodes[uri], context_type="SKILL")
    else:
        metadata = ({"version": 2} if change.endswith("revision")
                    else {"version": 1, "status": "archived"})
        fs.nodes[uri] = replace(fs.nodes[uri], metadata=metadata)
    items = [{"uri": a, "content": "Original ordinary evidence", "category": "entity"}]
    stats = {"graph_expansion": search.trace.graph_expansion}
    assert lane.deliver(items, "fact", stats, CTX, {"working_set": 10000}, render) == items


@pytest.mark.parametrize("change", ["source_revision", "source_archived", "target_revision", "target_type"])
def test_compose_rehydration_does_not_mix_old_edge_score_with_new_endpoint(change):
    from core.models import SearchMemoryResult
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from session.models import SessionWindowState

    fs, reader, graph, lane, embedder, (a, b, c) = components()
    pipe = pipeline(reader, graph, lane, embedder, a)
    api = ReadAPI(pipe, read_service=reader, config=pipe.cfg)
    service = MemoryService(config=OgMemConfig(topic_detection_enabled=False,
        rolling_compress_enabled=False, session_state_bridge_enabled=False))
    service._get_context_fs = Mock(return_value=None)
    service._graph_enabled = Mock(return_value=True)
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service.get_llm = Mock(side_effect=AssertionError("Compose must not call an LLM"))

    native_search = api.search_memory

    def search(**kwargs):
        return SearchMemoryResult() if kwargs.get("categories") == ["session_summary"] else native_search(**kwargs)

    # This is the same canonical reader used by the pipeline and read facade;
    # no fabricated graph metadata or receipt fields are supplied by the test.
    api.search_memory = search
    service.get_read_api = Mock(return_value=api)
    original_search = service._search_working_set

    def find_then_update(query, ctx):
        result = original_search(query, ctx)
        assert b in [item["uri"] for item in result[0]]
        uri = a if change.startswith("source") else b
        if change == "target_type":
            fs.nodes[uri] = replace(fs.nodes[uri], context_type="SKILL")
        else:
            metadata = ({"version": 1, "status": "archived"} if change.endswith("archived")
                        else {"version": 2})
            fs.nodes[uri] = replace(fs.nodes[uri], content="Post-search incompatible canonical body", metadata=metadata)
        return result

    service._search_working_set = find_then_update
    result = service.compose({"_ctx": CTX, "sessionId": CTX.session_id,
                              "query": "fact", "messages": [{"role": "user", "content": "fact"}]})
    assert b not in result["retrievedEvidence"]
    assert c not in result["retrievedEvidence"]
    if change != "source_archived":
        assert a in result["retrievedEvidence"]
    service.get_llm.assert_not_called()
