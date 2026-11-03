"""Resolved source identities stay visible through tool completion and publication."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision import SceneDecisionEngine
from scene.decision_prompts import DECISION_PROMPT, QUALITY_PROMPT
from scene.decision_sources import supporting_sources, with_source_groups
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import source_case


def archive_ref(identifier, message):
    return ProvenanceResolver.build_id("archive", identifier, [message])


@pytest.mark.parametrize("selection", [(0,), (1,), (0, 1), (1, 2), (0, 1, 2), (1, 3), (0, 1, 3), (0, 1, 2, 3, 4)])
def test_source_groups_match_selected_archive_identities(selection):
    archive = "ctx://acct/sessions/conversation/history/shared-archive"
    other_scope = "ctx://acct/sessions/another-conversation/history/shared-archive"
    cards = [
        {"uri": "ctx://acct/users/alice/memories/events/short", "source_refs": [archive_ref("shared-archive", "one")]},
        {"uri": "ctx://acct/users/alice/memories/events/full", "source_refs": [archive_ref(archive, "two")]},
        {"uri": "ctx://acct/users/alice/memories/events/scoped", "source_refs": [archive_ref(other_scope, "three")]},
        {"uri": "ctx://acct/users/alice/memories/events/same", "source_refs": [archive_ref(archive, "four")]},
        {"uri": "ctx://acct/users/alice/memories/events/other", "source_refs": [archive_ref("other-archive", "one")]},
    ]
    packet = {"memories": cards, "tool_evidence": []}
    original = deepcopy(packet)
    grouped = with_source_groups(packet)
    selected_ids = {index for position in selection for index in grouped["memories"][position]["source_group_ids"]}
    for short_id, full_ids in grouped["source_group_aliases"]:
        if selected_ids.intersection(full_ids):
            selected_ids.discard(short_id)
    selected_roots = {tuple(grouped["source_groups"][index]) for index in selected_ids}
    assert selected_roots == supporting_sources([cards[position] for position in selection], cards)
    assert grouped["memories"][1]["source_group_ids"] == grouped["memories"][3]["source_group_ids"]
    assert [card["source_refs"] for card in grouped["memories"]] == [card["source_refs"] for card in cards]
    assert with_source_groups(grouped) == grouped
    assert packet == original


def test_tool_context_resolves_complete_multihop_source_groups():
    uris = [f"ctx://acct/users/alice/memories/events/{name}" for name in ("child", "middle", "root")]
    root_ref = archive_ref("session-original", "message-original")
    child = {"uri": uris[0], "source_refs": [ProvenanceResolver.build_id("memory", uris[1])]}
    middle = {"uri": uris[1], "source_refs": [ProvenanceResolver.build_id("memory", uris[2])]}
    parent = {"uri": uris[2], "source_refs": [root_ref], "node_fingerprint": "complete-root-fingerprint"}
    packet = {"memories": [child], "tool_evidence": [{"tool": "read_source_context", "results": [middle, parent]}]}
    original = deepcopy(packet)
    grouped = with_source_groups(packet)
    assert grouped["source_groups"] == [("archive", "session-original")]
    assert grouped["memories"][0]["source_group_ids"] == [0]
    assert grouped["tool_evidence"] == packet["tool_evidence"]
    assert packet == original


@pytest.mark.parametrize("action", ["CREATE", "UPDATE", "MERGE"])
def test_real_source_tool_completion_publishes_reviewed_canonical_scene(action):
    source, nodes, _, proposal = source_case(action)
    parent_uri = "ctx://acct/users/alice/memories/event/source-parent"
    parent_ref = ProvenanceResolver.build_id("memory", parent_uri)
    parent = replace(nodes[0], uri=parent_uri, metadata={"source_refs": [archive_ref("archive-a", "parent-message")]})
    source.fs.nodes[parent_uri] = parent
    source.vector_index.vectors[parent_uri] = [1.0, 0.0]
    source.vector_index.texts[parent_uri] = parent.abstract
    for index, item in enumerate(nodes):
        refs = [parent_ref] if index == 0 else [archive_ref("archive-b", f"message-{index}")]
        nodes[index] = replace(item, metadata={**item.metadata, "source_refs": refs})
        source.fs.nodes[item.uri] = nodes[index]
    before = deepcopy(source.fs.nodes)

    class SourceContextModel(JSONModel):
        def __init__(self):
            super().__init__(proposal.model_dump(), {"decision": "ACCEPT", "reason": "Two original archives support the complete scene"})
            self.tool_calls = 0

        def complete_with_tools_messages(self, messages, tools, tool_choice):
            self.tool_calls += 1
            self.preflight = json.loads(messages[1]["content"])
            return [{"tool": "read_source_context", "input": {"source_refs": [parent_ref], "limit": 8}}], ""

    model = SourceContextModel()
    engine = SceneDecisionEngine(model, SceneConfig(max_llm_calls_per_batch=4), evidence_reader=source)
    result = engine.refresh(source, "acct", "user:alice", node_uris=[item.uri for item in nodes])
    assert result.action == action
    assert result.calls == model.tool_calls + len(model.requests) == 3
    assert result.snapshot.revision == source.repository.snapshot.revision == 1
    assert result.snapshot.scenes[0].version == (1 if action == "CREATE" else 2)
    assert source.fs.nodes == before
    assert ("memory", parent_uri) in map(tuple, model.preflight["source_groups"])
    decision_packet = json.loads(model.requests[0][0].removeprefix(DECISION_PROMPT))
    quality = json.loads(model.requests[1][0].removeprefix(QUALITY_PROMPT))
    assert set(map(tuple, decision_packet["source_groups"])) == {("archive", "archive-a"), ("archive", "archive-b")}
    context = [*decision_packet["memories"], *decision_packet["tool_evidence"][0]["results"]]
    for card in decision_packet["memories"]:
        groups = {tuple(decision_packet["source_groups"][index]) for index in card["source_group_ids"]}
        assert groups == supporting_sources([card], context)
    assert decision_packet["tool_evidence"][0]["results"][0]["uri"] == parent_uri
    assert quality["evidence"]["source_groups"] == decision_packet["source_groups"]
    assert quality["decision"] == proposal.model_dump()
    assert {member.context_uri for scene in result.snapshot.scenes for member in scene.cluster.members} == set(proposal.draft.member_uris)
    assert result.pending_member_uris == (nodes[-1].uri,)
