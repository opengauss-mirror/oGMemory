"""Bounded decisions preserve complete candidates and checkpoint remaining input."""

from copy import deepcopy

from scene.decision import SceneDecisionEngine
from scene.decision_contracts import DecisionLimits
from scene.decision_window import decision_window
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import source_case


def test_window_keeps_complete_candidate_and_source_closure_in_queue_order():
    uris = [f"ctx://acct/users/alice/memories/events/{index}" for index in range(8)]
    cards = [{"uri": uri, "text": "Complete evidence", "source_refs": [f"archive:{index}"]}
             for index, uri in enumerate(uris)]
    cards[0]["source_refs"] = [uris[7]]
    scenes = [{"scene_id": "first", "member_uris": uris[1:3]},
              {"scene_id": "later", "member_uris": uris[4:6]}]
    packet = {"memories": cards, "offered_member_uris": uris, "existing_scenes": scenes,
              "candidate_groups": [], "tool_evidence": [{"tool": "search_scene_cards", "results": scenes}]}
    original = deepcopy(packet)
    window = decision_window(packet, [uris[3], uris[0]], lambda value: len(value["memories"]) <= 5)
    assert {card["uri"] for card in window["memories"]} == {uris[index] for index in (0, 1, 2, 3, 7)}
    assert window["existing_scenes"] == [scenes[0]]
    assert window["decision_window"]["deferred_scene_ids"] == ["later"]
    assert window["tool_evidence"] == original["tool_evidence"]
    assert packet == original


def test_small_configured_budget_publishes_complete_reviewed_scene():
    source, nodes, _, proposal = source_case()
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model, limits=DecisionLimits(max_prompt_chars=8_000))
    result = engine.refresh(source, "acct", "user:alice", node_uris=[node.uri for node in nodes])
    assert result.action == "CREATE"
    assert result.calls == 3
    assert result.snapshot.revision == 1
    assert result.pending_member_uris == (nodes[-1].uri,)
    assert result.reviewed_member_uris == tuple(sorted(node.uri for node in nodes))
    assert result.deferred_member_uris == ()
    assert all(len(prompt) <= 8_000 for prompt, _ in model.requests)


def test_publication_checkpoint_contains_available_unassigned_evidence():
    source, nodes, _, proposal = source_case()
    retired = "ctx://acct/users/alice/memories/events/retired-item"
    requested = [node.uri for node in nodes] + [retired]
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    result = SceneDecisionEngine(model).refresh(source, "acct", "user:alice", node_uris=requested)
    assert result.action == "CREATE"
    assert result.calls == 3
    assert result.snapshot.revision == 1
    assert result.pending_member_uris == (nodes[-1].uri,)
    assert result.reviewed_member_uris == tuple(sorted(node.uri for node in nodes))
    assert result.deferred_member_uris == ()


def test_published_window_keeps_deferred_requests_in_checkpoint():
    source, nodes, _, proposal = source_case()
    members = [node.uri for node in nodes[:3]]
    proposal = proposal.model_copy(update={
        "draft": proposal.draft.model_copy(update={"member_uris": members}),
        "evidence_uris": members, "unassigned_member_uris": [],
    })
    model = JSONModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
    engine = SceneDecisionEngine(model)
    engine._fits_window = lambda packet: len(packet["memories"]) <= 3
    result = engine.refresh(source, "acct", "user:alice", node_uris=[node.uri for node in nodes])
    assert result.action == "CREATE"
    assert result.calls == 3
    assert result.reviewed_member_uris == tuple(members)
    assert result.deferred_member_uris == tuple(node.uri for node in nodes[3:])
    assert result.pending_member_uris == result.deferred_member_uris
    assert {member.context_uri for scene in result.snapshot.scenes for member in scene.cluster.members} == set(members)


def test_native_preflight_receives_the_budgeted_evidence_packet():
    import json

    from scene.decision_calls import CallBudget

    class NativeModel:
        def complete_with_tools_messages(self, messages, tools, tool_choice):
            self.messages = messages
            return [], '{"status":"READY"}'

    model = NativeModel()
    packet = {"memories": [{"uri": f"ctx://acct/users/alice/memories/events/{index}",
              "text": "Canonical event. " * 1000, "source_refs": [f"source-{index}"]} for index in range(8)],
              "offered_member_uris": [], "existing_scenes": [], "candidate_groups": []}
    original = deepcopy(packet)
    engine = SceneDecisionEngine(model, evidence_reader=object())
    result = engine._preflight.run(packet, CallBudget(4))
    assert result.status == "READY"
    wire = json.loads(model.messages[1]["content"])
    assert all(len(card["text"]) == 800 and card["text_truncated"] for card in wire["memories"])
    assert [card["source_refs"] for card in wire["memories"]] == [card["source_refs"] for card in packet["memories"]]
    assert sum(len(message["content"]) for message in model.messages) <= 64_000
    assert packet == original


def test_window_preserves_canonical_sources_reached_through_tool_cards():
    from core.provenance_resolver import ProvenanceResolver
    from scene.decision_sources import supporting_sources, with_source_groups

    uris = [f"ctx://acct/users/alice/memories/events/{name}" for name in ("child", "tool", "root", "other")]
    root_ref = ProvenanceResolver.build_id("archive", "original", ["message"])
    cards = [{"uri": uris[0], "source_refs": [uris[1]]},
             {"uri": uris[2], "source_refs": [root_ref]},
             {"uri": uris[3], "source_refs": ["other-source"]}]
    tool = {"uri": uris[1], "source_refs": [uris[2]]}
    packet = with_source_groups({
        "memories": cards, "offered_member_uris": [card["uri"] for card in cards],
        "existing_scenes": [{"scene_id": "other", "member_uris": uris[2:]}],
        "tool_evidence": [{"tool": "read_source_context", "results": [tool]}],
    })
    original = deepcopy(packet)
    window = decision_window(packet, [uris[0]], lambda value: len(value["memories"]) <= 2)
    assert [card["uri"] for card in window["memories"]] == [uris[0], uris[2]]
    roots = supporting_sources([window["memories"][0]], [*window["memories"], tool])
    assert roots == {("archive", "original")}
    assert window["source_groups"] == sorted(roots)
    assert [card["source_group_ids"] for card in window["memories"]] == [[0], [0]]
    assert window["tool_evidence"] == original["tool_evidence"]
    assert packet == original


def test_window_rebuilds_source_directory_and_alias_indices_within_budget():
    import json
    from core.provenance_resolver import ProvenanceResolver
    from scene.decision_sources import with_source_groups

    uris = [f"ctx://acct/users/alice/memories/events/{index}" for index in range(20)]
    archive = "ctx://acct/sessions/conversation/history/shared"
    refs = [ProvenanceResolver.build_id("archive", name, ["message"])
            for name in ("shared", archive, *[f"unused-{i}" + "x" * 100 for i in range(18)])]
    packet = with_source_groups({
        "memories": [{"uri": uri, "source_refs": [ref]} for uri, ref in zip(uris, refs)],
        "offered_member_uris": uris, "existing_scenes": [],
    })
    original = deepcopy(packet)
    fits = lambda value: len(json.dumps(value["source_groups"])) < 160
    window = decision_window(packet, uris[:2], fits)
    assert fits(window)
    assert [card["uri"] for card in window["memories"]] == uris[:2]
    assert window["source_groups"] == [("archive", archive), ("archive", "shared")]
    assert window["source_group_aliases"] == [(1, [0])]
    assert [card["source_group_ids"] for card in window["memories"]] == [[1], [0]]
    assert packet == original
