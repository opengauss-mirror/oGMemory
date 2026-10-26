"""Canonical evidence expansion, checked publication and checkpoint recovery."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.decision_contracts import DecisionLimits
from scene.decision_prompts import DECISION_PROMPT
from tests.unit.scene.test_adapters import adapter
from tests.unit.scene.test_decision_application import application_case
from tests.unit.scene.test_decision_preflight import JSONModel


class EvidenceModel(JSONModel):
    def __init__(self, proposal, tool="search_related_nodes", callback=lambda: None):
        super().__init__(proposal.model_dump(), {"decision": "ACCEPT"})
        self.tool = tool
        self.callback = callback
        self.tool_calls = 0

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        self.tool_calls += 1
        return [{"tool": self.tool, "input": {"query": "release review"}}], ""

    def complete_json(self, prompt, schema):
        result = super().complete_json(prompt, schema)
        if schema["title"] == "QualityGateOutput":
            self.callback()
        return result


def source_case(action="CREATE"):
    snapshot, nodes, embeddings, proposal, _ = application_case(action)
    source = adapter(nodes)
    source.repository.snapshot = replace(snapshot, revision=0)
    return source, nodes, embeddings, proposal


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_refresh_publishes_complete_canonical_provenance(action):
    source, nodes, _, proposal = source_case(action)
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    result = SceneDecisionEngine(model).refresh(source, "acct", "user:alice", node_uris=[n.uri for n in nodes])
    assert result.action == action
    assert source.repository.snapshot.scenes == result.snapshot.scenes
    assert source.repository.snapshot.revision == result.snapshot.revision == 1
    assert source.repository.snapshot.scenes[0].version == (1 if action == "CREATE" else 2)
    assert result.calls == 3
    assert result.pending_member_uris == (nodes[-1].uri,)


@pytest.mark.parametrize("tool", ["search_related_nodes", "search_scene_cards"])
def test_preflight_expands_canonical_members_with_caller_inputs_preserved(tool):
    source, nodes, embeddings, proposal = source_case("UPDATE" if tool == "search_scene_cards" else "CREATE")
    snapshot = source.repository.snapshot
    offered = nodes[:3]
    offered_vectors = {n.uri: embeddings[n.uri] for n in offered}
    before = deepcopy((offered, offered_vectors))
    if tool == "search_related_nodes":
        source.search_related_nodes = lambda *args: [source.node_card(n) for n in nodes[3:]]
    else:
        source.search_scene_cards = lambda *args: [{"scene_id": snapshot.scenes[0].scene_id}]
        # The selected existing scene contributes the remaining indexed members.
        record = snapshot.scenes[0]
        members = tuple(replace(record.cluster.members[0], context_uri=n.uri) for n in nodes)
        snapshot = replace(snapshot, scenes=(replace(record, cluster=replace(record.cluster, members=members)),))
        proposal = proposal.model_copy(update={"draft": proposal.draft.model_copy(update={
            "member_uris": [n.uri for n in nodes],
        }), "unassigned_member_uris": []})
    model = EvidenceModel(proposal, tool)
    engine = SceneDecisionEngine(model, evidence_reader=source)
    result = engine.decide_with_result(snapshot, offered, offered_vectors)
    assert result.action == proposal.action
    assert result.calls == model.tool_calls + len(model.requests) == 3
    assert [m.context_uri for m in result.snapshot.scenes[0].cluster.members] == proposal.draft.member_uris
    packet = json.loads(model.requests[0][0].removeprefix(DECISION_PROMPT))
    assert {card["uri"] for card in packet["memories"]} == {n.uri for n in nodes}
    assert (offered, offered_vectors) == before
    assert engine.for_config(SceneConfig()).evidence_reader is source


def test_expanded_source_revision_recovers_before_checked_publication():
    source, nodes, _, proposal = source_case()
    source.search_related_nodes = lambda *args: [source.node_card(n) for n in nodes[3:]]
    extra = nodes[3]

    def update_source():
        source.fs.nodes[extra.uri] = replace(extra, content="Revised release review", metadata={
            **extra.metadata, "version": extra.metadata.get("version", 0) + 1,
        })

    engine = SceneDecisionEngine(EvidenceModel(proposal, callback=update_source), evidence_reader=source)
    offered = [n.uri for n in nodes[:3]]
    with pytest.raises(RuntimeError, match="evidence changed"):
        engine.refresh(source, "acct", "user:alice", node_uris=offered)
    assert source.repository.snapshot.revision == 0
    engine = SceneDecisionEngine(EvidenceModel(proposal), evidence_reader=source)
    result = engine.refresh(source, "acct", "user:alice", node_uris=offered)
    assert result.action == "CREATE"
    member = next(m for m in result.snapshot.scenes[0].cluster.members if m.context_uri == extra.uri)
    assert member.node_version == source.fs.nodes[extra.uri].metadata["version"]
    assert source.repository.snapshot.revision == 1


def test_bounded_expansion_publishes_a_complete_scene_within_node_limit():
    source, nodes, _, proposal = source_case()
    source.search_related_nodes = lambda *args: [source.node_card(n) for n in nodes[3:]]
    proposal = proposal.model_copy(update={"unassigned_member_uris": []})
    engine = SceneDecisionEngine(EvidenceModel(proposal), evidence_reader=source,
                                 limits=DecisionLimits(max_input_nodes=4))
    result = engine.refresh(source, "acct", "user:alice", node_uris=[n.uri for n in nodes[:3]])
    assert result.action == "CREATE"
    assert len(result.snapshot.scenes[0].cluster.members) == 4
    assert result.calls == 3


def test_waiting_publication_resumes_with_fresh_model_budget():
    source, nodes, _, proposal = source_case()
    model = JSONModel({"status": "WAIT", "reason": "Await review"},
                      {"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)
    uris = [n.uri for n in nodes]
    first = engine.refresh(source, "acct", "user:alice", node_uris=uris)
    assert first.action == "WAIT"
    assert first.pending_member_uris == tuple(uris)
    assert first.calls == 1
    result = engine.refresh(source, "acct", "user:alice", node_uris=uris)
    assert result.action == "CREATE"
    assert result.calls == 3
    assert source.repository.snapshot.revision == 1
