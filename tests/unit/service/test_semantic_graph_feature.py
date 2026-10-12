"""Feature boundary tests: no graph overhead when off, upstream prompt intact."""
import threading
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.models import CandidateMemory, ContextNode, RequestContext, WritePlan
from extraction import Extractor
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, parse_tool_call
from index.activation_projection import ACTIVATION_VERSION
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.api import MemoryWriteAPI


CTX = RequestContext("acct","alice","agent","session","trace")


def test_graph_toggle_selects_archived_prompt_and_relation_tool():
    registry = SchemaRegistry()
    off = build_extraction_tools(registry)
    on = deepcopy(build_extraction_tools(registry,activation_enabled=True))
    relations = [tool for tool in on if tool["name"] == "extract_relation"]
    assert len(relations) == 1
    assert "update_activation" not in relations[0]["input_schema"]["properties"]
    on = [tool for tool in on if tool["name"] != "extract_relation"]
    for tool in on:
        assert tool["input_schema"]["properties"].pop("update_activation")
        tool["input_schema"]["required"].remove("update_activation")
    assert on == off
    plain = Extractor(Mock(),schema_registry=registry)
    graph = Extractor(Mock(),schema_registry=registry,activation_enabled=True)
    original = plain._build_prompt("English","I prefer dark coffee",ctx=CTX)
    enhanced = graph._build_prompt("English","I prefer dark coffee",ctx=CTX)
    assert "UPDATE ACTIVATION — ONE SENTENCE PER MEMORY" in enhanced
    assert 'update_activation: "Caroline' in enhanced
    assert "Do NOT skip relationships" in enhanced
    assert "extract_relation" in enhanced
    assert "abstract/overview/content/update_activation" in enhanced
    assert "UPDATE ACTIVATION" not in original
    assert "extract_relation" not in original
    assert "CONNECTION DESCRIPTOR" not in enhanced
    for tool in on:
        assert "entity_key" not in tool["input_schema"]["properties"]
        assert "predicate" not in tool["input_schema"]["properties"]


def test_descriptor_keeps_original_content_and_provenance_fields():
    raw = dict(routing_key="coffee",abstract="Prefers coffee",content="I prefer dark coffee",
               update_activation="Alice prefers dark coffee.")
    parsed = parse_tool_call("extract_preference",raw,SchemaRegistry(),activation_enabled=True)
    assert parsed is not None
    candidate = parsed[2]
    assert candidate.content == raw["content"] and candidate.abstract == raw["abstract"]
    assert candidate.metadata["update_activation"] == raw["update_activation"]
    assert candidate.metadata["update_activation_version"] == ACTIVATION_VERSION
    assert parse_tool_call("extract_preference",{**raw,"update_activation":""},SchemaRegistry(),activation_enabled=True) is None


@pytest.mark.parametrize("enabled",[False,True])
@pytest.mark.parametrize("generation", ["edges", "empty", "failure"])
def test_ingestion_calls_graph_only_when_enabled(enabled, generation):
    fs,llm,relations,recommendation = Mock(),Mock(),Mock(),Mock()
    nodes = []
    candidates = []
    plans = []
    for name in ("a","b"):
        metadata = dict(update_activation="Concrete fact "+name,update_activation_version=ACTIVATION_VERSION,
                        provenance_ids=[name+"-source"])
        uri = f"ctx://acct/users/alice/memories/entity/{name}"
        candidates.append(CandidateMemory("entity","user",name,name,name,name,.9,
                                           provenance_ids=metadata["provenance_ids"],metadata=metadata))
        nodes.append(ContextNode(uri,"MEMORY","entity",2,"user:alice",name,name,name,metadata))
        plans.append(WritePlan("create",uri))
    fs.read_node.side_effect = lambda uri,ctx: next(n for n in nodes if n.uri == uri)
    recommendation.recommend.return_value = ([],None)
    llm.complete_json.return_value = {"relations":[dict(source_node_ref=nodes[0].uri,
        target_node_ref=nodes[1].uri,relation="explains",evidence_ids=["a-source","b-source"])]}
    if generation == "empty": llm.complete_json.return_value = {"relations": []}
    if generation == "failure": llm.complete_json.side_effect = RuntimeError("provider unavailable")
    api = MemoryWriteAPI(fs,llm,graph_enabled=enabled,relation_store=relations,
                         relation_candidate_service=recommendation)
    api._pipeline = Mock()
    api._pipeline.extract.return_value = candidates
    api._pipeline.filter_by_confidence.return_value = candidates
    api._writer = Mock()
    api._writer.write_candidates.return_value = plans
    if enabled:
        from extraction.relation_proposals import attach, parse_proposal
        attach(candidates, [parse_proposal(dict(from_routing_key="a", to_routing_key="b",
            from_category="entity", to_category="entity", relation_type="supports",
            reason="facts", weight=.9))])
    result = api.commit_session([dict(role="user",content="facts")],CTX,archive_id="archive")
    if enabled:
        assert result["graph"]["relation_persisted"] == (2 if generation == "edges" else 1)
        llm.complete_json.assert_called_once()
        assert "NODE-EXTRACTION RELATION PROPOSALS" not in llm.complete_json.call_args.args[0]
        stored = relations.upsert_edges.call_args.args[0]
        assert stored[0].relation_type == "supports"
        assert stored[0].from_uri == plans[0].target_uri
        assert all("_extraction_relation_proposals" not in c.metadata for c in candidates)
        recommendation.recommend.assert_called_once()
        relations.upsert_edges.assert_called_once()
    else:
        assert "graph" not in result
        llm.complete_json.assert_not_called()
        recommendation.recommend.assert_not_called()
        relations.upsert_edges.assert_not_called()


@pytest.mark.parametrize("enabled",[False,True])
def test_runtime_reuses_reader_index_embedder_and_graph_service(enabled):
    service = MemoryService.__new__(MemoryService)
    service._cfg = OgMemConfig(graph_backend="sql" if enabled else "none",storage_backend="sql")
    service._init_lock = threading.RLock()
    service._read_api = None
    service._lexical_index = None
    embedder = Mock(model="test-model",dimension=2)
    index = Mock()
    service._get_shared_embedder = Mock(return_value=embedder)
    service._get_shared_vector_index = Mock(return_value=index)
    service._get_relation_store = Mock(return_value=Mock())
    service._get_context_fs = Mock(return_value=Mock())
    service._start_outbox_worker = Mock()
    pipe = service.get_read_api()._pipeline
    if enabled:
        assert pipe._graph_supplement.index is index
        assert pipe._graph_supplement.reader is pipe._context_reader
        assert pipe._graph_supplement.embedder is embedder
        assert pipe._graph_supplement.graph is pipe._graph_retriever._graph_store
    else:
        assert pipe._graph_retriever is None and pipe._graph_supplement is None
    embedder.embed_texts.assert_not_called()


def test_feature_defaults_and_invalid_config():
    assert OgMemConfig().graph_backend == "none"
    with pytest.raises(ValueError):
        OgMemConfig(graph_backend="sql",storage_backend="agfs")
    with pytest.raises(ValueError):
        OgMemConfig(graph_expand_top_k=6)


def test_changed_descriptor_is_not_skipped_as_duplicate_content():
    from dataclasses import replace
    from commit.context_writer import ContextWriter
    fs,policy = Mock(),Mock()
    policy.plan.return_value = WritePlan("create","ctx://acct/users/alice/memories/entities/project")
    writer = ContextWriter(fs,llm=Mock(),policy_router=policy)
    original = CandidateMemory("entity","user","project","Project","Project overview","Project content",.9,
        metadata=dict(update_activation="Project uses a database.",update_activation_version=ACTIVATION_VERSION))
    changed = replace(original,metadata={**original.metadata,"update_activation":"Project uses a relational database."})
    writer.write_candidate(original,CTX)
    writer.write_candidate(changed,CTX)
    writer.write_candidate(changed,CTX)
    assert fs.write_node.call_count == 2
    assert fs.write_node.call_args.args[0].metadata["update_activation"] == changed.metadata["update_activation"]
