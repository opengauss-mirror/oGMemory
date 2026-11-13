"""Native relations survive complete dual-pass extraction and occurrence pairing."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction import Extractor
from extraction.relation_proposals import KEY, take
from extraction.schemas.registry import SchemaRegistry
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS

CTX = RequestContext("relation-runs", "reader", "agent", "session", "trace")
TIME = datetime(2026, 10, 1, tzinfo=UTC)
MESSAGES = [{"id": "repair-source", "role": "user", "content":
             "On 2026-10-01 I repaired the receiver. I replaced its cable. "
             "I rebooted the receiver. The receiver repair follows the manual."}]


class NativeRelationsLLM:
    """Controls provider outputs, not the real extraction/merge/write call chain."""

    def __init__(self, category="case", *, proposals=((), ("receiver",)),
                 bodies=(("I replaced its cable.",), ("I replaced its cable.",)),
                 generation="empty"):
        self.category, self.proposals, self.bodies = category, proposals, bodies
        self.generation = generation
        self.native_calls, self.json_calls = [], []
        self.assembly_calls = []

    def complete_with_tools(self, *, prompt, tools, tool_choice, temperature=None):
        self.native_calls.append(temperature)
        run = 0 if temperature is None else 1
        assert "I repaired the receiver" in prompt
        calls = [{"tool": "extract_" + self.category, "input": dict(
            routing_key="receiver_repair", abstract="Receiver repaired",
            overview=body, content=body, confidence=.95 if run == 0 else .8,
            title="Receiver repair", when="2026-10-01", who="reader",
            update_activation="The receiver was repaired by replacing its cable.",
        )} for body in self.bodies[run]]
        if any(tool["name"] == "extract_relation" for tool in tools):
            calls.extend({"tool": "extract_relation", "input": dict(
                from_routing_key="receiver_repair", from_category=self.category,
                to_routing_key=target, to_category="entity", relation_type="refers_to",
                reason="The repair concerns the " + target, weight=.9,
            )} for target in self.proposals[run])
        return calls

    def complete_json(self, prompt, schema):
        properties = set(schema["properties"])
        if properties == {"categories"}:
            # Dev now selects schemas before graph-disabled extraction. Track
            # assembly separately from the extraction/reconciliation calls
            # pinned by this fixture's existing relation-provenance assertions.
            self.assembly_calls.append(properties)
            return {"categories": [self.category]}
        self.json_calls.append(properties)
        if properties == {"spans"}:
            return {"spans": [{"start": 0, "end": 0, "categories": [self.category]}]}
        if "decision" in properties:
            return {"decision": "new"}
        if properties == {"relations"}:
            if self.generation == "failure":
                raise RuntimeError("post-extraction relation model unavailable")
            return {"relations": []}
        assert properties == {"groups"}, properties
        # Keep distinct source-backed drafts separate. No occurrence algorithm
        # is mocked: the real partition validator consumes this provider result.
        payload = json.loads(prompt.rsplit("\n", 1)[-1])
        groups = []
        group_item = schema["properties"]["groups"]["items"]
        if "oneOf" in group_item:
            return {"groups": [{"members": [index]} for index in range(len(payload["drafts"]))]}
        group_schema = group_item["properties"]
        for index, raw in enumerate(payload["drafts"]):
            draft = raw.get("draft", raw)
            group = {"members": [index], **{key: draft[key] for key in
                     ("abstract", "overview", "content", "title", "when", "who", "where", "scope_ref")}}
            if "identity_evidence" in group_schema:
                group["identity_evidence"] = []
            groups.append(group)
        return {"groups": groups}


def extract(llm, *, graph=True, reconcile=False):
    extractor = Extractor(llm, activation_enabled=graph,
                          source_occurrence_reconciliation_enabled=reconcile)
    return extractor.extract(deepcopy(MESSAGES), CTX, session_time=TIME, archive_id="repair-archive")


@pytest.mark.parametrize("category", ["case", "event"])
@pytest.mark.parametrize("reconcile", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_second_run_only_proposal_survives_public_dual_pass_extraction(category, reconcile, graph):
    llm = NativeRelationsLLM(category)
    candidates = extract(llm, graph=graph, reconcile=reconcile)
    assert len(candidates) == 1
    assert candidates[0].content == "I replaced its cable."
    # The high-confidence first run wins; only its losing second run proposed
    # the edge. The actual span identification, native parser and normalizers run.
    assert candidates[0].confidence == .95
    assert llm.native_calls == [None, 0]
    proposals = take(candidates)
    assert [p["to_routing_key"] for p in proposals] == (["receiver"] if graph else [])
    assert all(KEY not in candidate.metadata for candidate in candidates)
    assert llm.json_calls == [{"spans"}]


@pytest.mark.parametrize("category", ["case", "event"])
@pytest.mark.parametrize("reconcile", [False, True])
def test_distinct_proposals_from_both_runs_are_unioned_once(category, reconcile):
    llm = NativeRelationsLLM(category, proposals=(("receiver",), ("receiver", "manual")))
    candidates = extract(llm, reconcile=reconcile)
    assert len(candidates) == 1
    assert [p["to_routing_key"] for p in take(candidates)] == ["receiver", "manual"]
    assert llm.json_calls == [{"spans"}]


@pytest.mark.parametrize("reconcile", [False, True])
def test_independent_case_facts_survive_with_losing_pass_proposals(reconcile):
    llm = NativeRelationsLLM(bodies=(("I replaced its cable.",), ("I rebooted the receiver.",)))
    candidates = extract(llm, reconcile=reconcile)
    assert [candidate.content for candidate in candidates] == ["I replaced its cable.", "I rebooted the receiver."]
    assert [p["to_routing_key"] for p in take(candidates)] == ["receiver"]


def test_same_run_multiplicity_is_not_collapsed_to_preserve_proposals():
    llm = NativeRelationsLLM(bodies=(("I replaced its cable.",) * 2, ("I replaced its cable.",) * 2))
    candidates = extract(llm)
    assert len(candidates) == 2
    assert len(take(candidates)) == 1


@pytest.mark.parametrize("generation", ["empty", "failure"])
def test_second_run_native_edge_persists_when_later_generation_adds_nothing(generation):
    llm = NativeRelationsLLM(generation=generation)
    fs, store = MemoryFS(), Mock()
    registry = SchemaRegistry()
    uris = URIResolver(registry)
    receiver = uris.resolve("entity", {"routing_key": "receiver"}, CTX)
    fs.write_node(ContextNode(receiver, "MEMORY", "entity", 2, CTX.user_space_name(),
                              "Receiver", "Receiver", "The receiver was repaired.",
                              {"routing_key": "receiver"}), CTX)
    api = MemoryWriteAPI(fs, llm, schema_registry=registry, uri_resolver=uris,
                         graph_enabled=True, relation_store=store,
                         directory_summary_enabled=False)
    result = api.commit_session(deepcopy(MESSAGES), CTX, session_time=TIME, archive_id="repair-archive")
    assert result["writes_completed"] == 1
    assert result["graph"]["node_extraction_relation_proposals"] == 1
    assert result["graph"]["native_relation_accepted"] == 1
    assert result["graph"]["relation_persisted"] == 1
    stored = store.upsert_edges.call_args.args[0]
    assert len(stored) == 1 and stored[0].to_uri == receiver
    assert stored[0].from_uri == result["plans"][0]["target_uri"]
    assert stored[0].relation_type == "refers_to"
    assert all(KEY not in node.metadata for node in fs.nodes.values())


def test_proposal_survives_later_cross_span_occurrence_reconciliation():
    from tests.unit.extraction.test_source_occurrence_reconciliation import (
        LATE,
        SourceLLM,
        messages,
    )

    class CrossSpanRelationsLLM(SourceLLM):
        def complete_with_tools(self, **kwargs):
            calls = super().complete_with_tools(**kwargs)
            for call in calls:
                call["input"]["update_activation"] = "The receiver was repaired."
            if LATE in kwargs["prompt"] and kwargs.get("temperature") == 0:
                calls.append({"tool": "extract_relation", "input": dict(
                    from_routing_key="atlas_inspection", from_category="case",
                    to_routing_key="receiver", to_category="entity", relation_type="refers_to",
                    reason="The repair concerns the receiver", weight=.9)})
            return calls

    llm = CrossSpanRelationsLLM(category="case")
    source = messages("I, Nora, repaired the receiver once. I replaced its cable.",
                      LATE + " For that same receiver repair, I rebooted it and it worked.")
    result = Extractor(llm, activation_enabled=True, source_occurrence_reconciliation_enabled=True).extract(
        source, CTX, session_time=TIME, archive_id="cross-span-source")
    assert len(result) == 1 and "rebooted" in result[0].content
    assert [p["to_routing_key"] for p in take(result)] == ["receiver"]
    assert llm.native_calls.count((True, 0)) == 1


def test_rejected_span_does_not_donate_relations_to_another_span():
    from core.models import CandidateMemory
    from extraction.relation_proposals import attach
    from extraction.tools import _Span

    good = CandidateMemory("case", "agent", "good", "Good", "Good", "Good", .9)
    rejected = CandidateMemory("case", "agent", "rejected", "Rejected", "Rejected", "Rejected", .2)
    attach([rejected], [dict(from_routing_key="rejected", to_routing_key="receiver",
                           from_category="case", to_category="entity", relation_type="refers_to",
                           reason="Unaccepted draft relation", weight=.9)])
    result = Extractor(Mock())._merge_span_results([(_Span(0, 0), [good]), (_Span(1, 1), [rejected])], CTX)
    assert result == [good]
    assert take(result) == []
