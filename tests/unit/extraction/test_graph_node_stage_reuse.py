"""Graph annotations accompany source drafts through staged node processing."""

from dataclasses import asdict
from datetime import datetime

from core.models import CandidateMemory
from extraction import Extractor
from extraction.relation_proposals import KEY, parse_proposal
from extraction.tools import ExtractionRuntime, _Span
from tests.unit.extraction.test_node_processing import node_payload, operation, setup
from tests.unit.extraction.test_parallel_node_stages import Stages


def test_graph_source_and_node_stages_replay_complete_annotations():
    ctx, fs, llm, api = setup(lambda _: [])
    activation = "Caroline's grandmother gave her a necklace with a heart charm."
    proposal = parse_proposal({"from_routing_key": "necklace", "to_routing_key": "grandmother",
                               "relation_type": "belongs_to", "reason": activation, "weight": .9})
    native_calls, node_calls = [], []

    def complete(**kwargs):
        if kwargs["tools"][0]["name"] == "process_memory_nodes":
            node_calls.append(kwargs)
            draft = node_payload(kwargs)["drafts"][0]
            assert draft["metadata"]["update_activation"] == activation
            return [{"tool": "process_memory_nodes", "input": {
                "operations": [operation([0], fact=activation)], "related_reconciliations": []}}]
        native_calls.append(kwargs)
        offered = {tool["name"]: tool for tool in kwargs["tools"]}
        assert "update_activation" in offered["extract_entity"]["input_schema"]["required"]
        return [{"tool": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "scope_ref": "",
            "abstract": "Caroline's necklace.", "content": activation, "confidence": .95,
            "update_activation": activation}}, {"tool": "extract_relation", "input": proposal}]

    llm.complete_with_tools = complete
    extractor = Extractor(llm, activation_enabled=True, node_processing_enabled=True,
                          fs=fs, schema_registry=api._schema_registry, uri_resolver=api._uri_resolver)
    extractor._identify_spans = lambda *args: [_Span(0, 0, categories=["entity"])]
    source = [{"id": "source", "role": "user", "content": activation}]
    stages = Stages()
    results = []
    for _ in range(2):
        runtime = ExtractionRuntime(stages)
        with runtime.activate():
            candidates = extractor.extract(source, ctx, session_time=datetime(2026, 10, 3),
                                           archive_id="graph-source")
        assert candidates[0].metadata[KEY] == [proposal]
        candidates = api._node_processor().process(candidates, source, ctx, checkpoint=stages,
                                                  reconcile=lambda: runtime.reconcile(source, ctx, "graph-source"))
        assert candidates[0].metadata[KEY] == [proposal]
        candidate = CandidateMemory(**asdict(candidates[0]))
        assert candidate.metadata["update_activation"] == activation
        assert candidate.node_reconciliation_role == "primary"
        assert candidate.node_reconciliation_group
        results.append(candidate)
    assert results[0] == results[1]
    assert len(native_calls) == len(node_calls) == 1
    assert stages.saved == ["native_extraction", "node_processing"]
