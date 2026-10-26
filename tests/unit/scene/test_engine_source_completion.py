"""Canonical provenance reaches each engine stage and successful recovery."""

import json
from copy import deepcopy

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from scene.decision_sources import node_source_refs, supporting_sources
from tests.unit.scene.test_decision_application import application_case
from tests.unit.scene.test_decision_preflight import JSONModel


def complete_sources(nodes, source_kind):
    planning = [ProvenanceResolver.build_id("archive", "planning", [f"m{index}"]) for index in range(4)]
    review = ProvenanceResolver.build_id("archive", "review")
    for item in nodes:
        item.metadata["source_refs"] = list(planning)
    holder = nodes[0] if source_kind == "direct" else nodes[-1]
    holder.metadata["source_refs"].append(review)
    if source_kind == "parent":
        for item in nodes[:-1]:
            item.metadata["source_refs"] = [holder.uri]
    return holder, planning


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
@pytest.mark.parametrize("calls", [2, 3])
@pytest.mark.parametrize("source_kind", ["direct", "parent"])
def test_fifth_canonical_source_completes_every_engine_stage(action, calls, source_kind):
    snapshot, nodes, embeddings, proposal, _ = application_case(action)
    holder, planning = complete_sources(nodes, source_kind)
    bounded = SceneEvidenceBuilder().build(snapshot, nodes, embeddings)
    assert next(card for card in bounded["memories"] if card["uri"] == holder.uri)["source_refs"] == planning
    original = deepcopy((snapshot, nodes, embeddings))
    outputs = [{"status": "READY"}, proposal.model_dump()]
    if calls == 3:
        outputs.append({"decision": "ACCEPT"})
    model = JSONModel(*outputs)
    engine = SceneDecisionEngine(model, SceneConfig(max_llm_calls_per_batch=calls))
    result = engine.decide_with_result(snapshot, nodes, embeddings)

    assert result.action == action, result.reason
    assert result.calls == len(model.requests) == calls
    assert result.snapshot.revision == snapshot.revision + 1
    assert result.pending_member_uris == (nodes[-1].uri,)
    preflight = json.loads(model.requests[0][0].split("\nPACKET:\n", 1)[1])
    decision = json.loads(model.requests[1][0].removeprefix(DECISION_PROMPT))
    packets = [preflight, decision]
    if calls == 3:
        packets.append(json.loads(model.requests[2][0].removeprefix(QUALITY_PROMPT))["evidence"])
    for packet in packets:
        context = [*packet["memories"], *packet.get("provenance_context", [])]
        selected = [card for card in packet["memories"] if card["uri"] in proposal.evidence_uris]
        assert supporting_sources(selected, context) == {("archive", "planning"), ("archive", "review")}
        described = {tuple(packet["source_groups"][index]) for card in selected for index in card["source_group_ids"]}
        assert described == supporting_sources(selected, context)
        assert next(card for card in context if card["uri"] == holder.uri)["source_refs"] == list(node_source_refs(holder))
    assert all(len(prompt) <= engine.limits.max_prompt_chars for prompt, _ in model.requests)
    for member in result.snapshot.scenes[0].cluster.members:
        item = next(item for item in nodes if item.uri == member.context_uri)
        assert member.source_refs == node_source_refs(item)
    assert (snapshot, nodes, embeddings) == original


def test_selected_member_source_completion_resumes_a_waiting_batch():
    snapshot, nodes, embeddings, proposal, _ = application_case()
    selected, _ = complete_sources(nodes, "direct")
    review = selected.metadata["source_refs"].pop()
    nodes[-1].metadata["source_refs"].append(review)
    wait = {"action": "WAIT", "reason": "Await a source for the selected release memories"}
    model = JSONModel({"status": "READY"}, proposal.model_dump(), wait,
                      {"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)
    waiting = engine.decide_with_result(snapshot, nodes, embeddings)
    assert waiting.action == "WAIT"
    assert waiting.reason == wait["reason"]
    assert waiting.calls == 3
    assert waiting.pending_member_uris == tuple(item.uri for item in nodes)
    assert "independent_sources_required" in model.requests[2][0]

    selected.metadata["source_refs"].append(review)
    completed = engine.decide_with_result(snapshot, nodes, embeddings)
    assert completed.action == "CREATE"
    assert completed.calls == 3
    assert completed.snapshot.revision == snapshot.revision + 1
    assert completed.pending_member_uris == (nodes[-1].uri,)
